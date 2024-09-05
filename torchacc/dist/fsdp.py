import functools
from types import MethodType
from typing import Any, Dict, Optional, Set
from enum import auto, Enum

import torch
import torch.fx as fx
from torch.fx.passes.split_module import split_module
import torch_xla.distributed.fsdp as xla_fsdp
import torch_xla.core.xla_model as xm

from torchacc.config import Config
from torchacc.dist import ParallelModule
import torchacc.utils.checkpoint as checkpoint
import torchacc.utils.trace as trace
import torchacc.utils.utils as utils
import torchacc.utils.optim_utils as optim_utils


def split_fsdp_wrap_modules(
        graph_model: fx.GraphModule,
        layer_cls: Set[str],
        model_name: Optional[str] = None,
        qualname_map: Optional[Dict[str, str]] = None) -> fx.GraphModule:
    curr_mod = None
    curr_idx = 0

    modules_types = {}

    def split_callback(n: torch.fx.node.Node):
        nonlocal curr_mod, curr_idx
        found = False
        if "nn_module_stack" in n.meta:
            for mod, t in n.meta["nn_module_stack"].items():
                type = t[1]
                if type.__name__ in layer_cls:
                    if mod != curr_mod:
                        curr_mod = mod
                        curr_idx += 1
                        modules_types[f"submod_{curr_idx}"] = type.__name__
                    found = True
                    break
        if not found and curr_mod is not None:
            curr_mod = None
            curr_idx += 1
        return curr_idx

    # Ask split_module to return mapping from new qualname to old qualname
    new_qualname_map: Dict[str, str] = {}
    split = split_module(graph_model, None, split_callback, new_qualname_map)
    # This is needed. FSDP will register a hook for the input tensor,
    # which will result in the recompilation of the computational graph.
    trace.move_single_param_to_callee(split, new_qualname_map)

    # Update qualname_map
    # TODO: the names of the submodules of the model need to be restored.
    if qualname_map is not None:
        for k, v in new_qualname_map.items():
            v = f"{model_name}.{v}"
            if v in qualname_map and k != v:
                assert k not in qualname_map
                qualname_map[k] = qualname_map[v]
                del qualname_map[v]
            elif v not in qualname_map:
                assert k not in qualname_map
                qualname_map[k] = v

    # Update the class name of the wrap modules
    for mod_name, mod_type in modules_types.items():
        assert hasattr(split, mod_name)
        mod = getattr(split, mod_name)
        mod.__class__.__name__ = mod_type
    return split


def fx_auto_wrap_policy(
    module: torch.nn.Module,
    recurse: bool,
    unwrapped_params: int,
    layer_cls: Set[str],
) -> bool:
    """A convenient auto wrap policy for fx models. If the submodule
    is an instance of layer_cls, the submodule will be wrapped
    as a FSDP unit. Otherwise, all the other remainder submodules are wrapped
    by the outermost FSDP unit.
    Return if a module should be wrapped during FSDP auto wrapping.
    The first three parameters are required by :func:`_recursive_wrap`.

    Args:
        module (nn.Module):
            The module to be considered in this decision.
        recurse (bool):
            Indicate if this is called to make a decision on whether we
            should recurse down a subgraph of the module structure.
            If False, it means this function is called to make a decision
            on whether we should wrap the said module.
        unwrapped_params (int):
            The number of parameters yet to be wrapped in this module.
        layer_cls (Set[str]):
            Submodules with one of the `layer_cls` names
            will be wrapped as separated FSDP units
    """
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or reminder
        return module.__class__.__name__ in layer_cls


class StateDictType(Enum):
    """
    This enum indicates that which type of ``state_dict`` the FSDP module is
    currently processing (returning or loading).
    The default value is FULL_STATE_DICT.
    ..note::
        FSDP currently supports one type of ``state_dict``:
            ``optim_state_dict/load_optim_state_dict`: this pair of APIs return and load
            the non-sharded, unflattened parameters.
    """

    FULL_STATE_DICT = auto()


class FullyShardedDataParallel(ParallelModule):
    """Implementation of fully sharded data parallel.

    Args:
        model (torch.nn.Module): The model to enable fully sharded data parallel.
        config (torchacc.Config): Configuration for TorchAcc.
    """

    def __init__(self, model: torch.nn.Module, config: Config, **kwargs):
        super().__init__(model, config)
        self.model = self.fsdp(model, config)

    def _get_underlay_model(self):
        return self.model

    def fsdp(self, model: torch.nn.Module, config: Config):
        if isinstance(model, fx.GraphModule):
            layer_cls = set()
            # Filter out some existing models, such as nn.Linear.
            for name in config.dist.fsdp.wrap_layer_cls:
                cls = utils.get_module_class_from_name(model, name)
                if cls is None:
                    layer_cls.add(name)
            model = split_fsdp_wrap_modules(model, layer_cls)
            auto_wrap_policy = functools.partial(
                fx_auto_wrap_policy,
                layer_cls=config.dist.fsdp.wrap_layer_cls,
            )
        else:
            layer_cls = set()
            for name in config.dist.fsdp.wrap_layer_cls:
                cls = utils.get_module_class_from_name(model, name)
                assert cls, f"class {name} in fsdp.wrap_layer_cls not found in model"
                layer_cls.add(cls)
            auto_wrap_policy = functools.partial(
                xla_fsdp.wrap.transformer_auto_wrap_policy,
                # Transformer layer class to wrap
                transformer_layer_cls=layer_cls,
            )

        dtype = torch.float32
        if config.compute.fp16:
            dtype = torch.float16
        if config.compute.bf16:
            dtype = torch.bfloat16

        # (wenting.swt): When using fsdp, disable autocast for precision conversion
        # Instead, use low precision for all intermediate calculations
        # Only the output is float32. This is to align with Stanford Alpaca's fsdp implementation
        if config.compute.fp16 or config.compute.bf16:
            model._original_forward = model.forward
            model_forward_func = model.forward.__func__ if hasattr(
                model.forward, "__func__") else model.forward
            new_forward = torch.cuda.amp.autocast(dtype=dtype)(
                model_forward_func)
            model.forward = MethodType(new_forward, model)
            model.forward = MethodType(
                utils.convert_outputs_to_fp32(model.forward.__func__), model)

        auto_wrapper_callable = None
        if config.memory.gc and (config.memory.gc_cls
                                 == config.dist.fsdp.wrap_layer_cls):
            gc_cnt = config.memory.gc_cnt

            def auto_wrapper_callable(m, *args, **kwargs):
                nonlocal gc_cnt
                if gc_cnt is None:
                    m = checkpoint.checkpoint_module(m)
                elif gc_cnt > 0:
                    m = checkpoint.checkpoint_module(m)
                    gc_cnt -= 1
                return xla_fsdp.XlaFullyShardedDataParallel(m, *args, **kwargs)

        model = xla_fsdp.XlaFullyShardedDataParallel(
            model,
            flatten_parameters=config.dist.fsdp.flatten_parameters,
            sync_module_states=config.dist.fsdp.sync_module_states,
            opt_flatten_overlap=True,
            pin_layout_in_collective_ops=False,
            auto_wrap_policy=auto_wrap_policy,
            auto_wrapper_callable=auto_wrapper_callable,
            compute_dtype=dtype,
            buffer_dtype=dtype,
            sharding_groups=self.mesh.get_fsdp_rank_groups(),
            sharding_rank=self.mesh.get_fsdp_rank(),
            sharding_world_size=self.mesh.get_fsdp_num())
        return model

    def clip_grad_norm_(self, max_grad_norm):
        if hasattr(self.model, "clip_grad_norm_"):
            self.model.clip_grad_norm_(max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_grad_norm)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def optim_state_dict(
            self,
            optim: torch.optim.Optimizer,
            state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
            rank0_only: bool = True,
            cpu_offload: bool = True) -> Dict[str, Any]:
        """
        Transform the state-dict of an optimizer corresponding to a sharded model.

        The given state-dict can be transformed to one type now: full optimizer state_dict.

        For full optimizer state_dict, all states are unflattened and not sharded.
        Rank0 only and CPU offload can be specified to avoid OOM.
        
        Args:
            optim (torch.optim.Optimizer): Optimizer for self.model's
                parameters.
            state_dict_type: (StateDictType):
                which type of ``state_dict`` the FSDP module is 
                currently processing (returning or loading)
                The default value is FULL_STATE_DICT.
            rank0_only: (bool): control whether only rank0 return the
                state-dict of optimizer.
                The default value is True.
            cpu_offload: (bool):  whether move the state-dict to cpu.
                The default value is True.

        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            self.model. The sharding of the optimizer state is based on
            ``state_dict_type``.
        """
        # we only support FULL_STATE_DICT and flatten parameters now
        if state_dict_type != StateDictType.FULL_STATE_DICT:
            raise NotImplementedError(
                "we only support 'FULL_SATE_DICT' StateDictType now")
        if not self.model.flatten_parameters:
            raise NotImplementedError(
                "we only support flatten_parameters=True now")

        shard_meta_data = self.model.get_shard_metadata()
        sharded_optim_state = optim.state_dict()['state']
        optim_state_param_groups = optim.state_dict()['param_groups']
        # unflattened and consolidated state_dict
        consolidate_optim_state_dict: Dict[str, Any] = {
            'state': {},
            'param_groups': {}
        }

        # (rank0_only and self.model.rank == 0) or (not rank0_only)
        if not rank0_only or self.model.rank == 0:
            consolidate_optim_state_dict[
                'param_groups'] = optim_state_param_groups
            consolidate_optim_state_dict['param_groups'][0]['params'].clear()

        for layer_state, (layer_name, layer_params) in zip(
                sharded_optim_state.values(),
                shard_meta_data['flatten_info'].items()):
            param_names, param_shapes, param_numels = layer_params
            # get full_param_names of each layer in optim_state_dict
            full_names = optim_utils._get_layer_full_names(
                layer_name, param_names)
            unflat_state_dict = {fn: {} for fn in full_names}

            if not rank0_only or self.model.rank == 0:
                consolidate_optim_state_dict['param_groups'][0][
                    'params'].append(full_names)

            for state_name, state_params in layer_state.items():
                tensor_buffer = optim_utils._all_gather_state(
                    state_params, self.model)

                if not rank0_only or self.model.rank == 0:
                    _, full_params = optim_utils._unflatten_optim_params(
                        tensor_buffer, layer_name, param_names, param_shapes,
                        param_numels)

                    for fn, fp in zip(full_names, full_params):
                        if cpu_offload:
                            xm.mark_step()
                            unflat_state_dict[fn][state_name] = fp.cpu()
                        else:
                            unflat_state_dict[fn][state_name] = fp
                xm.mark_step()
        consolidate_optim_state_dict['state'] = unflat_state_dict

        return consolidate_optim_state_dict

    def load_optim_state_dict(
            self,
            optim_state_dict: Dict[str, Any],
            optim: torch.optim.Optimizer,
            state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
            rank0_only: bool = True) -> Dict[str, Any]:
        """
        Convert an optimizer state-dict so that it can be loaded into the optimizer associated with the FSDP model.

        Given a ``optim_state_dict`` that is transformed through
        :meth:`optim_state_dict`, it gets converted to the optimizer
        state_dict that can be loaded to ``optim`` which is the optimizer for
        self.model.
        
        The given state-dict can be transformed from one type now: full optimizer state_dict.
        
        Args:
            optim (torch.optim.Optimizer): Optimizer for self.model's
                parameters.
            optim_state_dict (Dict[str, Any]): The optimizer states to be loaded.
            state_dict_type(StateDictType):  
                which type of ``state_dict`` the FSDP module is 
                currently processing (returning or loading)
                The default value is FULL_STATE_DICT.
            rank0_only: (bool): control whether load state_dict only from
                rank0 at the begining.
                The default value is True.
        
        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            self.model. The sharding of the optimizer state is based on
            ``state_dict_type``.
        """
        # we only support FULL_STATE_DICT and flatten parameters now
        if state_dict_type != StateDictType.FULL_STATE_DICT:
            raise NotImplementedError(
                "we only support 'FULL_SATE_DICT' StateDictType now")
        if not self.model.flatten_parameters:
            raise NotImplementedError(
                "we only support flatten_parameters=True now")
        shard_meta_data = self.model.get_shard_metadata()
        unflat_optim_state = optim_state_dict

        flat_optim_state: Dict[str, Any] = {'state': {}, 'param_groups': {}}

        # broadcast on global ranks instead of sharding_groups
        unflat_optim_state = optim_utils._broadcast_processed_state(
            unflat_optim_state, xm.get_ordinal(), xm.xrt_world_size())
        unflat_state = unflat_optim_state['state']

        # flatten and sharded state_dict
        for idx, (layer_name, (params)) in enumerate(
                shard_meta_data['flatten_info'].items()):
            param_names, _, _ = params
            # names of a flatten layer
            full_names = optim_utils._get_layer_full_names(
                layer_name, param_names)
            flat_value: Dict[str, Any] = {}
            # broadcast tensor to other ranks per layer per state
            for state_name in unflat_state[full_names[0]].keys():
                tensor_buffer_list = []
                # we need the params of a whole layer state to be flatten and shard
                for name in full_names:
                    state_params = unflat_state[name][state_name]
                    # all ranks have same scalar tensor(step) which has been broadcasted in
                    # broadcast_processed_state above
                    if isinstance(state_params,
                                  torch.Tensor) and state_params.dim() == 0:
                        flat_value[state_name] = state_params
                        break

                    tensor_buffer = optim_utils._broadcast_state(
                        state_params, self.model)
                    tensor_buffer_list.append(tensor_buffer)

                flat_tensor = optim_utils._flatten_optim_state(
                    tensor_buffer_list)

                if len(flat_tensor):
                    flat_value[state_name] = self.model._get_shard(flat_tensor)

            flat_optim_state['state'][idx] = flat_value
            xm.mark_step()

        # first params is [0, the number of fsdp wrapped layer - 1]
        # and other params are all none
        flat_optim_state['param_groups'] = unflat_optim_state['param_groups']
        flat_optim_state['param_groups'][0]['params'] = [
            i for i in range(0, len(flat_optim_state['state'].keys()))
        ]

        return flat_optim_state
