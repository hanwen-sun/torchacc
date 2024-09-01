import functools
from types import MethodType
from typing import Dict, Optional, Set, Any
import copy

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

    def optim_state_dict(self,
                         optim: torch.optim.Optimizer,
                         full_state_dict: str,
                         rank0_only: bool = True,
                         cpu_offload: bool = True) -> Dict[str, Any]:
        # we only support full_state_dict now
        assert full_state_dict == "FULL_STATE_DICT"
        shard_meta_data = self.model.get_shard_metadata()

        sharded_optim_state = optim.state_dict()['state']
        optim_state_param_groups = optim.state_dict()['param_groups']
        optim_state_param_groups[0]['params'].clear()

        consolidate_optim_state: Dict[str, Any] = {'state': {}, 'param_groups': {}}
        consolidate_optim_state['param_groups'] = optim_state_param_groups
        
        for (layer_idx, layer_state), (layer_name, (param_names, param_shapes, param_numels)) in zip(
        sharded_optim_state.items(), shard_meta_data['flatten_info'].items()):
        # 分层all_gather, unflatten, to_cpu
        # unflatten需要得到 shard_meta_data中的相关信息; 对于每一层layer, 使用的param相同;
            for state_name, state_params in layer_state.items():
                if (state_params.dim() == 0):
                    continue
                # prepare tensor buffer for all_gather
                # we consume the loaded params is flattened
                shape_list = list(state_params.size())
                shape_list[0] = shape_list[0] * self.model.world_size
                buffer_size = tuple(shape_list)
                tensor_buffer = state_params.new_zeros(*buffer_size)
                 
                tensor_buffer = self.model.all_gather_op(state_params, groups=self.model.sharding_groups)
                xm.rendezvous("optim_state_all_gather")
                
                # we now only support rank0_only
                if rank0_only and self.model.rank == 0:
                    full_names, full_params = optim_utils._unflatten_optim_params(
                    tensor_buffer, layer_name, param_names, param_shapes, param_numels)
                    # 删除tensor_buffer
                    del tensor_buffer
                    for fn, fp in zip(full_names, full_params):
                        if fn not in consolidate_optim_state['state'].keys():
                            consolidate_optim_state['state'].setdefault(fn, {})
                        if cpu_offload:
                            if 'step' not in consolidate_optim_state['state'][fn].keys():
                                consolidate_optim_state['state'][fn]['step'] = layer_state['step'].cpu()
                            consolidate_optim_state['state'][fn][state_name] = fp.cpu()
                        else:
                            if 'step' not in consolidate_optim_state['state'][fn].keys():
                                consolidate_optim_state['state'][fn]['step'] = layer_state['step']
                            consolidate_optim_state['state'][fn][state_name] = fp
                        consolidate_optim_state['param_groups'][0]['params'].append(fn)

        return consolidate_optim_state

    def load_optim_state_dict(self,
                              optim_state_dict: Dict[str, Any],
                              optim: torch.optim.Optimizer,
                              full_state_dict: str,
                              rank0_only: bool = True) -> Dict[str, Any]:
        # we now only support rank0 to load the state_dict;
        # we now only support FULL_STATE_DICT
        assert full_state_dict == 'FULL_STATE_DICT'  
        shard_meta_data = self.model.get_shard_metadata()
        #print(shard_meta_data)
        unflat_optim_state = copy.deepcopy(optim_state_dict)

        unflat_optim_state = optim_utils._broadcast_processed_state(unflat_optim_state, self.model.rank, self.model.world_size, self.model.sharding_groups)
        
        unflat_state = unflat_optim_state['state']
        xm.rendezvous("broadcast processed state")
        
        # flatten and sharded state_dict
        flat_optim_state: Dict[str, Any] = {'state': {}, 'param_groups': {}}
        
        for idx, (layer_name, (param_names, _, _)) in enumerate(shard_meta_data['flatten_info'].items()):
            full_names = optim_utils._shard_name_to_optim_name(layer_name, param_names)
            
            for name in full_names:
                for state_name, state in unflat_state[name].items():
                    if self.model.rank == 0:
                        tensor_buffer = state.to(self.model.xla_device)
                    else:
                        tensor_buffer = torch.zeros(state.shape, dtype=state.dtype, device=self.model.xla_device)
                    
                    # inplement broadcast
                    root_ordinal = xm.get_ordinal() if self.model.rank == 0 else -1

                    #  这里是否可以用list(tensor)优化
                    self.model.collective_broadcast_op(
                        [tensor_buffer],
                        root_ordinal=root_ordinal,
                        groups=self.model.sharding_groups)
                    
                    xm.rendezvous("broadcast unflat_state")
                    
                    # post processing
                    unflat_state[name][state_name] = tensor_buffer
                    del tensor_buffer
            # flatten optim_state
            state_names = unflat_state[full_names[0]].keys()
            
            flat_value: Dict[str, Any] = {}
            for state_name in state_names:
                flat_tensor = optim_utils._flatten_optim_state(unflat_state, state_name, full_names)
                if flat_tensor.dim() != 0:
                    shard_tensor = self.model._get_shard(flat_tensor)  
                else:
                    shard_tensor = flat_tensor
                flat_value[state_name] = shard_tensor            
            flat_optim_state['state'][idx] = flat_value
            # 这里要不要del
        
        # 这first item of flat_optim_state is 0 to the number of fsdp wrapped layer
        # and other param_groups are all none
        flat_optim_state['param_groups'] = unflat_optim_state['param_groups']
        flat_optim_state['param_groups'][0]['params'] = [i for i in range(0, len(flat_optim_state['state'].keys()))]
        
        return flat_optim_state  