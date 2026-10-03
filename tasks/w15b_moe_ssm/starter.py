import numpy as np

def topk_route(logits, k):
    raise NotImplementedError

def load_balance_loss(logits, idx):
    raise NotImplementedError

def discretize_zoh(A, B, dt):
    raise NotImplementedError

def ssm_recurrent(Ad, Bd, C, u):
    raise NotImplementedError

def ssm_kernel(Ad, Bd, C, L):
    raise NotImplementedError

def ssm_convolutional(Ad, Bd, C, u):
    raise NotImplementedError
