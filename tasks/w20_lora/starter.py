import numpy as np

def lora_forward(x, W, A, B, alpha):
    raise NotImplementedError

def lora_merge(W, A, B, alpha):
    raise NotImplementedError

def lora_init(d_in, d_out, r, seed=0):
    raise NotImplementedError

def lora_param_count(d_in, d_out, r):
    raise NotImplementedError
