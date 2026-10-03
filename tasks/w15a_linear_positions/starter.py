import numpy as np

def linear_attention_quadratic(Q, K, V, causal=True):
    raise NotImplementedError

def linear_attention_recurrent(Q, K, V):
    raise NotImplementedError

def rope(X, positions, base=10000.0):
    raise NotImplementedError

def alibi_bias(n, n_heads):
    raise NotImplementedError

def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, bytes_per=2):
    raise NotImplementedError
