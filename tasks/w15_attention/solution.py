import numpy as np

def causal_mask(L):
    return np.tril(np.ones((L, L), dtype=bool))

def attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    s = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d)
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    w = np.exp(s)
    w = w / w.sum(-1, keepdims=True)
    return w @ V, w
