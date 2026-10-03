import numpy as np

def attention(Q, K, V, causal=False):
    raise NotImplementedError

def tiled_attention(Q, K, V, block=16):
    raise NotImplementedError

def decode_with_cache(Q, K, V):
    raise NotImplementedError

def gqa_expand(K, n_q_heads):
    raise NotImplementedError
