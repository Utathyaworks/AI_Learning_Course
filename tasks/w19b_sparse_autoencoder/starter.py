import numpy as np

def normalize_decoder(W_dec):
    raise NotImplementedError

def sae_forward(X, W_enc, b_enc, W_dec, b_dec):
    raise NotImplementedError

def sae_loss(X, f, X_hat, l1):
    raise NotImplementedError

def topk_encode(pre, k):
    raise NotImplementedError

def sae_metrics(X, f, X_hat):
    raise NotImplementedError
