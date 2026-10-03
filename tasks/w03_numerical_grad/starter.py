import numpy as np

def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    raise NotImplementedError

def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    raise NotImplementedError
