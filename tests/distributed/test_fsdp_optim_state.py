import torch
import torchacc as ta

from utils.distributed import MultiProcessTestBase, init_pg, skip_if_lt_x_gpu
from torchacc.dist.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torchacc.config import Config

def _init_model(config: Config):
    model = FSDP(torch.nn.Linear(128, 128, bias=False), config)
    
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)
    
    model((torch.rand(128, 128).to(ta.lazy_device()))).sum().backward()
    optim.step()
    
    return model, optim

def _step_model(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        device: torch.device = torch.device("cuda"),
        num_iters: int = 1,
    ):
    torch.manual_seed(0)  # set seed for determinism
    model(torch.rand(128, 128).to(ta.lazy_device())).sum().backward()
    optim.step()

def _step_model_without_apply(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
):
    torch.manual_seed(0)  # set seed for determinism
    model(torch.rand(128, 128).to(ta.lazy_device())).sum().backward()

def _check_state(fsdp_osd1, fsdp_osd2):
    state1 = fsdp_osd1['state']
    state2 = fsdp_osd2['state']
    #print("check state!")
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

    print("check param groups!")
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
    def test_fsdp_optim_state(self):
        config1 = Config()
        config1.dist.fsdp.size = self.world_size
        model_1, optim_1 = _init_model(config=config1)
        
        _step_model(model_1, optim_1)
        
        fsdp_osd1 = FSDP.optim_state_dict(model_1, optim_1, full_state_dict="FULL_STATE_DICT")
        #new_group_ranks = [0, 1]
        #new_group = dist.new_group(ranks=new_group_ranks)
        #if self.rank not in new_group_ranks:
        #   import os
        #   os.exit()

        config2 = Config()
        config2.dist.fsdp.size = 4
        model_2, optim_2 = _init_model(config=config2)
        
        fsdp_osd_to_load = FSDP.load_optim_state_dict(model_2, fsdp_osd1, optim_2, "FULL_STATE_DICT")
        
        optim_2.load_state_dict(fsdp_osd_to_load)
        
        _step_model_without_apply(model_2, optim_2)
        
        fsdp_osd2 = FSDP.optim_state_dict(model_2, optim_2, full_state_dict="FULL_STATE_DICT")
        
        _check_state(fsdp_osd1, fsdp_osd2)
        _check_param_groups(fsdp_osd1, fsdp_osd2)
        
        import pdb
        pdb.set_trace()
        '''
        if self.rank in new_group_ranks:     
            config2 = Config()
            config2.dist.fsdp.size = 2
            
            model_2, optim_2 = _init_model(config=config2)
         
            #print("model_2")
            #print(model_2)

            #fsdp_osd_to_load = FSDP.load_optim_state_dict(model_2, fsdp_osd1, optim_2, "FULL_STATE_DICT")
            
            #import pdb
            #pdb.set_trace()
            
            #optim_2.load_state_dict(fsdp_osd_to_load)
            
            #_step_model(model_2, optim_2)

            #fsdp_osd2 = FSDP.optim_state_dict(model_2, optim_2, full_state_dict="FULL_STATE_DICT")
            
            #_check_state(fsdp_osd1, fsdp_osd2)
            #_check_param_groups(fsdp_osd1, fsdp_osd2)
        else:
            return
            
        '''