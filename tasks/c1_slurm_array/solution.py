import math

def grid_point(grid, index):
    keys = list(grid)
    sizes = [len(grid[k]) for k in keys]
    total = math.prod(sizes)
    if not 0 <= index < total:
        raise IndexError(index)
    out = {}
    for k, n in zip(reversed(keys), reversed(sizes)):
        index, r = divmod(index, n)
        out[k] = grid[k][r]
    return {k: out[k] for k in keys}

def sbatch_script(job_name, grid, command, gpus=1, hours=4, cpus=4, mem_gb=16):
    n = math.prod(len(v) for v in grid.values())
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{n - 1}",
        f"#SBATCH --gres=gpu:{gpus}",
        f"#SBATCH --time={hours:02d}:00:00",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem_gb}G",
        "#SBATCH --output=logs/%x_%A_%a.out",
        "",
        f"{command} --index $SLURM_ARRAY_TASK_ID",
    ]
    return "\n".join(lines) + "\n"
