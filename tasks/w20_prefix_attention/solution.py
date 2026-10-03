import numpy as np

def prefix_attention(Q, K, V, P_k, P_v):
    Kc = np.vstack([P_k, K]); Vc = np.vstack([P_v, V])
    s = Q @ Kc.T / np.sqrt(Q.shape[1])
    s = s - s.max(1, keepdims=True)
    w = np.exp(s); w /= w.sum(1, keepdims=True)
    return w @ Vc, w[:, :len(P_k)].sum(1)
