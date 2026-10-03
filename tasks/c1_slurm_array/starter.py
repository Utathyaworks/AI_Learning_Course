def grid_point(grid: dict, index: int) -> dict:
    raise NotImplementedError

def sbatch_script(job_name: str, grid: dict, command: str, gpus: int = 1, hours: int = 4,
                  cpus: int = 4, mem_gb: int = 16) -> str:
    raise NotImplementedError
