import torch
import torch.distributed as dist
from typing import NamedTuple, Optional
from torch.utils._pytree import tree_map_only

import torch_xla.core.xla_model as xm

# get orig name from shard_meta_data, and transform it to original optim state_dict name
# of each layer
def _shard_name_to_optim_name(layer_name, param_names):
  full_names = []
  
  prefix = None
  layer_name_split = layer_name.split('.') 
  if len(layer_name_split) >= 2 and layer_name_split[0] == "_fsdp_wrapped_module" and layer_name_split[1] == "model":
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


def _unflatten_optim_params(params, layer_name, param_names, param_shapes, param_numels):
  full_params = [
      t.view(s) for (t, s) in zip(params.split(param_numels), param_shapes)
  ]
  
  full_names = _shard_name_to_optim_name(layer_name, param_names)

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

def setup_gloo_distributed(world_size):
    # create a new gloo process group
    new_group_ranks = [r for r in range(world_size)]
    pg = dist.new_group(ranks=new_group_ranks, backend="gloo")
    return pg

def cleanup_gloo_distributed(pg):
    dist.destroy_process_group(pg)

def _broadcast_processed_state(
  optim_state: dict[str, any],
  rank: int,
  world_size: int
):
    objects: list[Any] = [None]
    if rank == 0:
        objects[0] = tree_map_only(
            torch.Tensor,
            lambda v: v.cpu() if v.dim() == 0 else _PosDimTensorInfo(v.shape, v.dtype),  # type: ignore[union-attr]
            optim_state,
        )
    
    pg_group = setup_gloo_distributed(world_size)
    dist.broadcast_object_list(objects, src=0, group=pg_group)
    cleanup_gloo_distributed(pg_group)

    if rank == 0:
        return optim_state
    else:
        return objects[0]    

def _flatten_optim_state(unflat_optim_state, state_name, full_names):
    param_list = []
    for name in full_names:
      param_list.append(unflat_optim_state[name][state_name])
    
    # scalar tensor do not need flatten
    if param_list[0].dim() == 0:
      return param_list[0]
    
    flat_tensors = [
      torch.flatten(param) for param in param_list
    ]
    
    return torch.cat(flat_tensors, dim=0)
  
def _optim_state_info(consolidate_optim_state):
    print("save optim state info:")
    state = consolidate_optim_state['state']
    for layer_name, layer_state in state.items():
      print("layer_name: " + layer_name)
      for state_name, state_value in layer_state.items():
        print(f"state_name: {state_name}, shape: {state_value.shape}, sum: {state_value.sum()}")
    
    print("save optim param_group info:")
    print(consolidate_optim_state['param_groups'])