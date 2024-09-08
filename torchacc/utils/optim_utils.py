import torch
import torch.distributed as dist
from typing import NamedTuple, Optional
from torch.utils._pytree import tree_map_only

import torch_xla.core.xla_model as xm


# transform name from shard_meta_data to original optim state_dict name of each layer
def get_layer_full_names(layer_name, param_names):
    full_names = []

    prefix = None
    layer_name_split = layer_name.split('.')
    if len(layer_name_split) >= 2 and layer_name_split[
            0] == "_fsdp_wrapped_module" and layer_name_split[1] == "model":
        prefix = ".".join(layer_name_split[1:4])
        prefix = prefix + '.'

    for name in param_names:
        name_splits = name.split(".")
        if name_splits[0] == '_fpw_module':
            new_name = ".".join(name_splits[1:])
            if prefix:
                new_name = prefix + new_name
            full_names.append(new_name)
        else:
            full_names.append(name)

    return full_names

def get_layer_full_info(shard_metadata, model_state_dict):
  full_name_list = []
  orig_size_list = []
  
  # consolidate the sharded parameters
  for name in model_state_dict.keys():
    is_sharded = False
    name_splits = name.split(".")
    # TODO: check whether is it necessary
    if name_splits[0] == 'model':
        name = ".".join(name_splits[1:])
        name_splits = name.split(".")
    for idx, sep in enumerate(name_splits):
      if sep.startswith("_fsdp_shard"):
        is_sharded = True
        prefix = ".".join(name_splits[:idx])
        suffix = ".".join(name_splits[idx:])
        break
    
    #p_info = shard_metadata["shard_info"][prefix][suffix]
    p_info = shard_metadata["shard_info"][prefix][suffix]
    orig_name = p_info["_orig_name"]
    orig_size = p_info["_orig_size"]
    if is_sharded:
      full_name = orig_name
      if prefix != "":
        full_name = prefix + "." + orig_name
    else:
      # unsharded buffers (we'll just use rank 0's state dict for buffers)
      full_name = name
    full_name_list.append(full_name)
    orig_size_list.append(orig_size)  
  
  # flatten_parameters = True
  flatten_info = shard_metadata["flatten_info"]
  if flatten_info != {}:
    full_name_list_ = []
    orig_size_list_ = []
    for name in full_name_list:
      if "_fsdp_wrapped_module.flat_param_" in name:
        metadata = flatten_info[name]
        prefix = ".".join(name.split(".")[:-1])
        param_names, param_shapes, param_numels = metadata
        full_names = param_names

        if prefix != "":
          full_names = [prefix + "." + n for n in full_names]
        
        full_names = [fn.replace("_fsdp_wrapped_module.", "").replace("_fpw_module.", "") for fn in full_names]

        full_name_list_.append(full_names)
        orig_size_list_.append(param_shapes)

    return full_name_list_, orig_size_list_ 
  
  full_name_list = [fn.replace("_fsdp_wrapped_module.", "").replace("_fpw_module.", "") for fn in full_name_list]
  
  return full_name_list, orig_size_list


def unflatten_optim_params(params, layer_name, param_names, param_shapes,
                           param_numels):
    full_names = get_layer_full_names(layer_name, param_names)

    if params.dim() == 0:
        full_params = [params for _ in range(len(full_names))]
    else:
        full_params = [
            t.view(s)
            for (t, s) in zip(params.split(param_numels), param_shapes)
        ]

    return full_names, full_params


class _PosDimTensorInfo(NamedTuple):
    """
    Attributes:
        shape (torch.Size): Sharded tensor shape (which is equal to the
            unsharded tensor shape if the tensor is optimizer state for a
            non-FSDP parameter and is hence not sharded).
        dtype (torch.dtype): Data type of the tensor.
    """

    shape: torch.Size
    dtype: torch.dtype


def _setup_gloo_distributed(group):
    pg = dist.new_group(ranks=group, backend="gloo")
    return pg


def _cleanup_gloo_distributed(pg):
    dist.destroy_process_group(pg)


def broadcast_processed_state(optim_state: dict[str, any], rank,
                              sharding_groups):
    objects: list[Any] = [None]
    if rank == 0:
        objects[0] = tree_map_only(
            torch.Tensor,
            lambda v: v.cpu() if v.dim() == 0 else _PosDimTensorInfo(
                v.shape, v.dtype),  # type: ignore[union-attr]
            optim_state,
        )
    ordinal = xm.get_ordinal()
    new_group = []

    for group in sharding_groups:
        if ordinal in group:
            new_group = group
            break

    pg_group = _setup_gloo_distributed(new_group)
    dist.broadcast_object_list(objects, src=new_group[0], group=pg_group)
    _cleanup_gloo_distributed(pg_group)
    if rank == 0:
        return optim_state
    else:
        return objects[0]


def broadcast_state(state_params, model):
    device = model.xla_device
    if model.rank == 0 and isinstance(state_params, torch.Tensor):
        tensor_buffer = state_params.to(device)
    else:
        tensor_buffer = torch.zeros(
            state_params.shape, dtype=state_params.dtype, device=device)

    # Since broadcast employs all-reduce, here we only need to ensure that root_ordinal
    # is different from xm.get_ordinal() on the non-root nodes
    root_ordinal = xm.get_ordinal() if model.rank == 0 else -1

    model.collective_broadcast_op([tensor_buffer],
                                  root_ordinal=root_ordinal,
                                  groups=model.sharding_groups)

    return tensor_buffer


def all_gather_state(state_params, model):
    if state_params.dim() == 0:
        return state_params

    tensor_buffer = model.all_gather_op(
        state_params, groups=model.sharding_groups)

    return tensor_buffer


def flatten_optim_state(param_list):
    if len(param_list) == 0:
        return param_list

    flat_tensors = [torch.flatten(param) for param in param_list]

    return torch.cat(flat_tensors, dim=0)
