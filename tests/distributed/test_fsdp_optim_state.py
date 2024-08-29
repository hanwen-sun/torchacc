import torch
import torchacc as ta

from utils.distributed import MultiProcessTestBase, init_pg, skip_if_lt_x_gpu
from torchacc.dist.fsdp import FullyShardedDataParallel as FSDP

def _init_model(device: torch.device):
    model = FSDP(torch.nn.Linear(4, 4).to(device), auto_wrap_policy=auto_wrap_policy)
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)
    model((torch.rand(4, 4).to(device))).sum().backward()
    optim.step()

    return model, optim

def _step_model(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        device: torch.device = torch.device("cuda"),
        num_iters: int = 1,
    ):
    torch.manual_seed(0)  # set seed for determinism
    model(torch.rand(4, 4)).sum().backward()
    optim.step()

def _step_model_without_apply(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
):
    torch.manual_seed(0)  # set seed for determinism
    model(torch.rand(4, 4)).sum().backward()

def _check_state(fsdp_osd1, fsdp_osd2):
    state1 = fsdp_osd1['state']
    state2 = fsdp_osd2['state']
    
    for key in state1.keys():
        dict1 = state1[key]
        dict2 = state2[key]
        for state_name in dict1.keys():
            #print(state_name)
            tensor1 = dict1[state_name]
            tensor2 = dict2[state_name]
            #print(tensor1)
            if not torch.equal(tensor1, tensor2):
                print(f"Difference found at key: {key}-{state_name}")
                print(f"Tensor 1: {tensor1}")
                print(f"Tensor 2: {tensor2}")
            else:
                print(f"No difference at key: {key}-{state_name}")

def _check_param_groups(fsdp_osd1, fsdp_osd2):
    param1 = fsdp_osd1['param_groups']
    param2 = fsdp_osd2['param_groups']

    for (value1, value2) in zip(param1, param2):
        for key in value1.keys():
            if value1[key] != value2[key]:
                print(f"Difference found at key: {key}")
                print(f"key 1: {value1[key]}")
                print(f"key 2: {value2[key]}")
            else:
                print(f"No difference at key: {key}")
    

class FSDPOptimStateTest(MultiProcessTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(2)
    @init_pg("lazy")
    def test_fsdp_optim_state():
        device = ta.lazy_device()
        
        model_1, optim_1 = _init_model(device=device)
        _step_model(model_1, optim_1)
    
        fsdp_osd1 = FSDP.optim_state_dict(model1, optim1)
        
        new_group_ranks = [r for r in range(self.world_size) if r % 2 == 0]
        new_group = dist.new_group(ranks=new_group_ranks)
        if self.rank not in new_group_ranks:
           return
        
        model_2, optim_2 = _init_model(device=device)
        fsdp_osd_to_load = FSDP.load_state_dict(fsdp_osd1, optim_2)
        optim_2.load_state_dict(fsdp_osd_to_load)
        
        _step_model_without_apply(model_2, optim_2)
        
        fsdp_osd2 = FSDP.optim_state_dict(model2, optim2)
        
        # check the equality of fsdp_osd1 and fsdp_osd2s
        _check_params(fsdp_osd1, fsdp_osd2)
        _check_param_groups(fsdp_osd1, fsdp_osd2)