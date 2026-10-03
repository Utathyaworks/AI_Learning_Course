import numpy as np

def _softmax(S):
    S = S - S.max(-1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(-1, keepdims=True)

def additive_attention(q, K, V, Wq, Wk, v):
    e = np.tanh(q @ Wq.T + K @ Wk.T) @ v
    w = _softmax(e)
    return w @ V, w

def make_mask(n, kind, window=None, prefix=None):
    i, j = np.indices((n, n))
    if kind == "full":
        return np.ones((n, n), bool)
    if kind == "causal":
        return j <= i
    if kind == "window":
        return (j <= i) & (i - j < window)
    if kind == "prefix":
        return (j <= i) | ((i < prefix) & (j < prefix))
    raise ValueError(kind)

def multihead_attention(X, Wq, Wk, Wv, Wo, h, mask=None, Y=None):
    Y = X if Y is None else Y
    n, d = X.shape; dh = d // h
    Q = (X @ Wq).reshape(n, h, dh).transpose(1, 0, 2)
    K = (Y @ Wk).reshape(len(Y), h, dh).transpose(1, 0, 2)
    V = (Y @ Wv).reshape(len(Y), h, dh).transpose(1, 0, 2)
    S = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
    if mask is not None:
        S = np.where(mask[None], S, -np.inf)
    O = (_softmax(S) @ V).transpose(1, 0, 2).reshape(n, d)
    return O @ Wo

def latent_cross_attention(L, X, Wq, Wk, Wv, Wo, h):
    return multihead_attention(L, Wq, Wk, Wv, Wo, h, Y=X)
