import numpy as np

def lora_forward(x, W, A, B, alpha):
    r = A.shape[1]
    return x @ W + (alpha / r) * (x @ A) @ B

def lora_merge(W, A, B, alpha):
    return W + (alpha / A.shape[1]) * A @ B

def lora_init(d_in, d_out, r, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1 / np.sqrt(r), (d_in, r)), np.zeros((r, d_out))

def lora_param_count(d_in, d_out, r):
    return r * (d_in + d_out)
