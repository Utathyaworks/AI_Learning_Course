import numpy as np

def additive_attention(q, K, V, Wq, Wk, v):
    raise NotImplementedError

def make_mask(n, kind, window=None, prefix=None):
    raise NotImplementedError

def multihead_attention(X, Wq, Wk, Wv, Wo, h, mask=None, Y=None):
    raise NotImplementedError

def latent_cross_attention(L, X, Wq, Wk, Wv, Wo, h):
    raise NotImplementedError
