import numpy as np

def _phi(x):
    x = np.asarray(x, float)
    return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0)))

def linear_attention_quadratic(Q, K, V, causal=True):
    A = _phi(Q) @ _phi(K).T
    if causal:
        A = np.tril(A)
    return (A @ V) / A.sum(1, keepdims=True)

def linear_attention_recurrent(Q, K, V):
    fq, fk = _phi(Q), _phi(K)
    S = np.zeros((Q.shape[1], V.shape[1])); z = np.zeros(Q.shape[1]); out = []
    for t in range(len(Q)):
        S += np.outer(fk[t], V[t]); z += fk[t]
        out.append(fq[t] @ S / (fq[t] @ z))
    return np.array(out)

def rope(X, positions, base=10000.0):
    X = np.asarray(X, float); n, d = X.shape
    theta = base ** (-np.arange(0, d, 2) / d)
    ang = np.asarray(positions, float)[:, None] * theta[None]
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(X)
    out[:, 0::2] = X[:, 0::2] * c - X[:, 1::2] * s
    out[:, 1::2] = X[:, 0::2] * s + X[:, 1::2] * c
    return out

def alibi_bias(n, n_heads):
    m = 2.0 ** (-8.0 * (np.arange(n_heads) + 1) / n_heads)
    i, j = np.indices((n, n))
    B = -m[:, None, None] * (i - j)[None].astype(float)
    return np.where((j <= i)[None], B, -np.inf)

def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, bytes_per=2):
    return int(2 * n_layers * n_kv_heads * d_head * seq_len * bytes_per)
