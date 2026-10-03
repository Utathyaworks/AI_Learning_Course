import numpy as np

def batchnorm_forward_backward(x, gamma, beta, dout, eps=1e-5):
    mu = x.mean(0); var = x.var(0)
    inv = 1 / np.sqrt(var + eps)
    xh = (x - mu) * inv
    out = gamma * xh + beta
    n = x.shape[0]
    dgamma = (dout * xh).sum(0); dbeta = dout.sum(0)
    dxh = dout * gamma
    dx = inv / n * (n * dxh - dxh.sum(0) - xh * (dxh * xh).sum(0))
    return out, dx, dgamma, dbeta
