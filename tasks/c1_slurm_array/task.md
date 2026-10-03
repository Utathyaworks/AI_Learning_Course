# Slurm job-array sweep

You will run hyperparameter sweeps as Slurm job arrays. Implement two helpers.

1. `grid_point(grid, index)`: `grid` is a dict of name → list of values. Return the configuration (dict) at
   position `index` of the Cartesian product, iterating keys in the dict's order with the **last key changing
   fastest** (like `itertools.product`). Raise `IndexError` for an out-of-range index.
2. `sbatch_script(job_name, grid, command, gpus=1, hours=4, cpus=4, mem_gb=16)`: return the text of an sbatch
   script that
   - starts with `#!/bin/bash`;
   - has `#SBATCH` lines for `--job-name`, `--array=0-(N-1)` where N is the grid size, `--gres=gpu:<gpus>`,
     `--time=HH:00:00` (two-digit hours), `--cpus-per-task`, `--mem=<mem_gb>G` and
     `--output=logs/%x_%A_%a.out`;
   - ends with a line running `command` followed by ` --index $SLURM_ARRAY_TASK_ID`.
