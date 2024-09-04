#!/bin/bash
set -exo pipefail

export PYTHONPATH=$PYTHONPATH:.

export CUDA_VISIBLE_DEVICES=4,5,6,7

function test_unittests() {
    pytest tests/core/test_bucketing_loader.py
    pytest tests/distributed/test_dist_ops.py
    pytest tests/distributed/test_fsdp_optim_state.py
    pytest tests/ops/test_flash_attn.py
    pytest tests/ops/test_context_parallel.py
}

test_unittests
