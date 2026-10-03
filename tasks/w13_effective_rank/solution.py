import numpy as np

def effective_rank(Z):
    Z = np.asarray(Z, float)
    Z = Z - Z.mean(0)
    s = np.linalg.svd(Z, compute_uv=False)
    if s.sum() == 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))
