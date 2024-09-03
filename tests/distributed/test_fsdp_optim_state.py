import copy

import torch
import torch.distributed as dist
import torchacc as ta
from torchacc.dist.fsdp import FullyShardedDataParallel as FSDP
from torchacc.config import Config

from utils.distributed import MultiProcessTestBase, init_pg, skip_if_lt_x_gpu


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 1024)
        self.fc4 = torch.nn.Linear(1024, 1024)
        self.fc5 = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


def _init_model(config: Config):
    model = Net()
    model = FSDP(model, config)
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)

    return model, optim


def _step_model(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    num_iters: int = 1,
):
    torch.manual_seed(0)  # set seed for determinism
    optim.zero_grad()
    batch_size = 1024
    device = ta.lazy_device()

    for i in range(num_iters):
        data1 = torch.rand(batch_size, 1024).to(device)
        data2 = torch.zeros(batch_size, dtype=torch.int64).to(device)
        loss = model(data1)
        loss = torch.nn.functional.nll_loss(loss, data2)
        loss.backward()
        optim.step()


def _step_model_without_update(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
):
    # do forward and backward and no optim.step()
    torch.manual_seed(0)  # set seed for determinism
    optim.zero_grad()
    batch_size = 1024
    device = ta.lazy_device()

    data1 = torch.rand(batch_size, 1024).to(device)
    data2 = torch.zeros(batch_size, dtype=torch.int64).to(device)
    loss = model(data1)
    loss = torch.nn.functional.nll_loss(loss, data2)
    loss.backward()


def _check_optim_state(fsdp_osd1, fsdp_osd2):
    state1 = fsdp_osd1['state']
    state2 = fsdp_osd2['state']
    for key in state1.keys():
        dict1 = state1[key]
        dict2 = state2[key]
        for state_name in dict1.keys():
            tensor1 = dict1[state_name]
            tensor2 = dict2[state_name]
            assert torch.equal(tensor1, tensor2)


def _check_optim_param_groups(fsdp_osd1, fsdp_osd2):
    param1 = fsdp_osd1['param_groups']
    param2 = fsdp_osd2['param_groups']

    for (value1, value2) in zip(param1, param2):
        for key in value1.keys():
            assert value1[key] == value2[key]


class FSDPOptimStateTest(MultiProcessTestBase):

    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(2)
    @init_pg("lazy")
    def test_fsdp_optim_state_flatten_gpu4_4(self):
        # we first init a model
        config1 = Config()
        config1.dist.fsdp.size = self.world_size
        model_1, optim_1 = _init_model(config=config1)
        # iter 10 steps
        _step_model(model_1, optim_1, 10)
        # get the optim_state_dict for model1
        fsdp_osd1 = FSDP.optim_state_dict(
            model_1, optim_1, full_state_dict="FULL_STATE_DICT")

        # init a new model with same world_size
        config2 = Config()
        config2.dist.fsdp.size = self.world_size
        model_2, optim_2 = _init_model(config=config2)
        # we may change fsdp_osd1 in load_optim_state_dict
        fsdp_osd1_ = copy.deepcopy(fsdp_osd1)
        # model_2 load the optim_state_dict from model_1
        fsdp_osd_to_load = FSDP.load_optim_state_dict(model_2, fsdp_osd1_,
                                                      optim_2,
                                                      "FULL_STATE_DICT")
        optim_2.load_state_dict(fsdp_osd_to_load)
        _step_model_without_update(model_2, optim_2)
        fsdp_osd2 = FSDP.optim_state_dict(
            model_2, optim_2, full_state_dict="FULL_STATE_DICT")

        _check_optim_state(fsdp_osd1, fsdp_osd2)
        _check_optim_param_groups(fsdp_osd1, fsdp_osd2)

    @skip_if_lt_x_gpu(2)
    @init_pg("lazy")
    def test_fsdp_optim_state_flatten_gpu4_2(self):
        config1 = Config()
        config1.dist.fsdp.size = self.world_size
        model_1, optim_1 = _init_model(config=config1)
        _step_model(model_1, optim_1, 10)
        fsdp_osd1 = FSDP.optim_state_dict(
            model_1, optim_1, full_state_dict="FULL_STATE_DICT")

        # we create a new group with world_size / 2
        new_world_size = self.world_size / 2
        new_group_ranks = list(range(int(new_world_size)))
        new_group = dist.new_group(ranks=new_group_ranks)

        # init model 2 with world_size / 2
        config2 = Config()
        config2.dist.fsdp.size = 2
        model_2, optim_2 = _init_model(config=config2)
        # we may change fsdp_osd1 in load_optim_state_dict
        fsdp_osd1_ = copy.deepcopy(fsdp_osd1)
        # model_2 load the optim_state_dict from model_1
        fsdp_osd_to_load = FSDP.load_optim_state_dict(model_2, fsdp_osd1_,
                                                      optim_2,
                                                      "FULL_STATE_DICT")
        optim_2.load_state_dict(fsdp_osd_to_load)
        _step_model_without_update(model_2, optim_2)
        fsdp_osd2 = FSDP.optim_state_dict(
            model_2, optim_2, full_state_dict="FULL_STATE_DICT")

        if self.rank in new_group_ranks:
            _check_optim_state(fsdp_osd1, fsdp_osd2)
            _check_optim_param_groups(fsdp_osd1, fsdp_osd2)
