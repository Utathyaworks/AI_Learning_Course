import numpy as np
from submission import *  # noqa: F401,F403
import itertools
import pytest

GRID = {"lr": [1e-3, 3e-4], "wd": [0.0, 0.01, 0.1], "seed": [0, 1]}

def test_matches_itertools_product():
    combos = list(itertools.product(*GRID.values()))
    for i, c in enumerate(combos):
        assert grid_point(GRID, i) == dict(zip(GRID, c))

def test_out_of_range():
    with pytest.raises(IndexError):
        grid_point(GRID, 12)

def test_script_directives():
    s = sbatch_script("sweep", GRID, "python train.py", gpus=2, hours=7, cpus=8, mem_gb=32)
    lines = s.strip().splitlines()
    assert lines[0] == "#!/bin/bash"
    for d in ["--job-name=sweep", "--array=0-11", "--gres=gpu:2", "--time=07:00:00",
              "--cpus-per-task=8", "--mem=32G", "--output=logs/%x_%A_%a.out"]:
        assert any(l.startswith("#SBATCH") and d in l for l in lines), d
    assert lines[-1] == "python train.py --index $SLURM_ARRAY_TASK_ID"
