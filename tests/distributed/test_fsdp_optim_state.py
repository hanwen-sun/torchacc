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


def _train_step(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    num_iters: int = 1,
):
    optim.zero_grad()
    batch_size = 1024
    device = ta.lazy_device()

    for i in range(num_iters):
        data = torch.rand(batch_size, 1024).to(device)
        labels = torch.zeros(batch_size, dtype=torch.int64).to(device)
        loss = model(data)
        loss = torch.nn.functional.nll_loss(loss, labels)
        loss.backward()
        optim.step()


def _train_step_without_update(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
):
    # do forward and backward and no optim.step()
    optim.zero_grad()
    batch_size = 1024
    device = ta.lazy_device()

    data = torch.rand(batch_size, 1024).to(device)
    labels = torch.zeros(batch_size, dtype=torch.int64).to(device)
    loss = model(data)
    loss = torch.nn.functional.nll_loss(loss, labels)
    loss.backward()


def _check_optim_state(fsdp_osd1, fsdp_osd2):
    state1 = fsdp_osd1['state']
    state2 = fsdp_osd2['state']

    assert state1.keys() == state2.keys()
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

    # the length of param is same to the wrapped layer
    assert len(param1) == len(param2)

    for (value1, value2) in zip(param1, param2):
        assert value1.keys() == value2.keys()
        for key in value1.keys():
            assert value1[key] == value2[key]


def _get_fsdp_osd_1(world_size, full_state_dict, rank0_only):
    config1 = Config()
    config1.dist.fsdp.size = world_size
    model_1, optim_1 = _init_model(config=config1)
    # iter 10 steps
    _train_step(model_1, optim_1, 10)
    # get the optim_state_dict for model1
    if full_state_dict:
        fsdp_osd1 = FSDP.full_optim_state_dict(
            model_1, optim_1, rank0_only=rank0_only)
    else:
        fsdp_osd1 = FSDP.sharded_optim_state_dict(model_1, optim_1)

    return fsdp_osd1


def _get_fsdp_osd_2(world_size, rank, new_group_rank, fsdp_osd1,
                    full_state_dict, rank0_only):
    # init a new model with new world_size
    config2 = Config()
    config2.dist.fsdp.size = world_size
    model_2, optim_2 = _init_model(config=config2)

    if rank not in new_group_rank:
        return
    # model_2 load the optim_state_dict from model_1
    fsdp_osd_to_load = FSDP.load_optim_state_dict(
        model_2, fsdp_osd1, rank0_only=rank0_only)
    optim_2.load_state_dict(fsdp_osd_to_load)
    _train_step_without_update(model_2, optim_2)
    if full_state_dict:
        fsdp_osd2 = FSDP.full_optim_state_dict(
            model_2, optim_2, rank0_only=rank0_only)
    else:
        fsdp_osd2 = FSDP.sharded_optim_state_dict(model_2, optim_2)

    return fsdp_osd2


class FSDPOptimStateTest(MultiProcessTestBase):

    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(2)
    @init_pg("lazy")
    def test_fsdp4_full_optim_state_rank0_only_flatten(self):
        rank0_only = True
        full_state_dict = True
        fsdp_osd1 = _get_fsdp_osd_1(self.world_size, full_state_dict,
                                    rank0_only)
        new_world_size = self.world_size
        new_group_ranks = list(range(int(new_world_size)))
        fsdp_osd2 = _get_fsdp_osd_2(self.world_size, self.rank, new_group_ranks,
                                    fsdp_osd1, full_state_dict, rank0_only)
        _check_optim_state(fsdp_osd1, fsdp_osd2)
        _check_optim_param_groups(fsdp_osd1, fsdp_osd2)

    @skip_if_lt_x_gpu(2)
    @init_pg("lazy")
    def test_fsdp4_to_2_full_optim_state_rank0_only_flatten(self):
        rank0_only = True
        full_state_dict = True
        fsdp_osd1 = _get_fsdp_osd_1(self.world_size, full_state_dict,
                                    rank0_only)
        # create new communication group
        assert self.world_size % 2 == 0
        new_world_size = self.world_size // 2
        new_group_ranks = list(range(int(new_world_size)))
        new_group = dist.new_group(ranks=new_group_ranks)
        fsdp_osd2 = _get_fsdp_osd_2(new_world_size, self.rank, new_group_ranks,
                                    fsdp_osd1, full_state_dict, rank0_only)

        if self.rank in new_group_ranks:
            _check_optim_state(fsdp_osd1, fsdp_osd2)
            _check_optim_param_groups(fsdp_osd1, fsdp_osd2)

    @skip_if_lt_x_gpu(2)
    @init_pg("lazy")
    def test_fsdp4_full_optim_state_not_rank0_only_flatten(self):
        rank0_only = False
        full_state_dict = True
        fsdp_osd1 = _get_fsdp_osd_1(self.world_size, full_state_dict,
                                    rank0_only)
        new_world_size = self.world_size
        new_group_ranks = list(range(int(new_world_size)))
        fsdp_osd2 = _get_fsdp_osd_2(self.world_size, self.rank, new_group_ranks,
                                    fsdp_osd1, full_state_dict, rank0_only)
        _check_optim_state(fsdp_osd1, fsdp_osd2)
        _check_optim_param_groups(fsdp_osd1, fsdp_osd2)

    @skip_if_lt_x_gpu(2)
    @init_pg("lazy")
    def test_fsdp4_to_2_full_optim_state_not_rank0_only_flatten(self):
        rank0_only = False
        full_state_dict = True
        fsdp_osd1 = _get_fsdp_osd_1(self.world_size, full_state_dict,
                                    rank0_only)
        assert self.world_size % 2 == 0
        new_world_size = self.world_size // 2
        new_group_ranks = list(range(int(new_world_size)))
        new_group = dist.new_group(ranks=new_group_ranks)
        fsdp_osd2 = _get_fsdp_osd_2(new_world_size, self.rank, new_group_ranks,
                                    fsdp_osd1, full_state_dict, rank0_only)
        if self.rank in new_group_ranks:
            _check_optim_state(fsdp_osd1, fsdp_osd2)
            _check_optim_param_groups(fsdp_osd1, fsdp_osd2)

    @skip_if_lt_x_gpu(2)
    @init_pg("lazy")
    def test_fsdp4_sharded_optim_state(self):
        full_state_dict = False
        rank0_only = False
        fsdp_osd1 = _get_fsdp_osd_1(self.world_size, full_state_dict,
                                    rank0_only)
        new_world_size = self.world_size
        new_group_ranks = list(range(int(new_world_size)))
        fsdp_osd2 = _get_fsdp_osd_2(self.world_size, self.rank, new_group_ranks,
                                    fsdp_osd1, full_state_dict, rank0_only)
        fsdp_osd1 = fsdp_osd1['optimizer']
        fsdp_osd2 = fsdp_osd2['optimizer']
        _check_optim_state(fsdp_osd1, fsdp_osd2)
        _check_optim_param_groups(fsdp_osd1, fsdp_osd2)
