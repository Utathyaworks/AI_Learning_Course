import numpy as np

def _ce_diag(S):
    S = S - S.max(1, keepdims=True)
    logp = S - np.log(np.exp(S).sum(1, keepdims=True))
    return -np.mean(np.diag(logp))

def info_nce(z1, z2, temperature=0.1):
    a = z1 / np.linalg.norm(z1, axis=1, keepdims=True)
    b = z2 / np.linalg.norm(z2, axis=1, keepdims=True)
    S = a @ b.T / temperature
    return float(0.5 * (_ce_diag(S) + _ce_diag(S.T)))
