import numpy as np

def forward(tokens, W, patch=None):
    raise NotImplementedError

def qk_circuit(W):
    raise NotImplementedError

def ov_circuit(W):
    raise NotImplementedError

def direct_logit_attribution(cache, W, pos, token):
    raise NotImplementedError

def patching_effect(clean, corrupt, W, pos, token):
    raise NotImplementedError
