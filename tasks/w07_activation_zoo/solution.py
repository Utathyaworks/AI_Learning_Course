import numpy as np
from scipy.special import erf

LAM, ALPHA = 1.0507009873554805, 1.6732632423543772

def sigmoid(x):
    x = np.asarray(x, float)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))

def sigmoid_grad(x):
    s = sigmoid(x); return s * (1 - s)

def relu(x): return np.maximum(np.asarray(x, float), 0.0)
def relu_grad(x): return (np.asarray(x, float) > 0).astype(float)

def leaky_relu(x, a=0.01):
    x = np.asarray(x, float); return np.where(x > 0, x, a * x)

def leaky_relu_grad(x, a=0.01):
    return np.where(np.asarray(x, float) > 0, 1.0, a)

def elu(x, a=1.0):
    x = np.asarray(x, float); return np.where(x > 0, x, a * np.expm1(np.minimum(x, 0)))

def elu_grad(x, a=1.0):
    x = np.asarray(x, float); return np.where(x > 0, 1.0, a * np.exp(np.minimum(x, 0)))

def selu(x): return LAM * elu(x, ALPHA)
def selu_grad(x): return LAM * elu_grad(x, ALPHA)

def _Phi(x): return 0.5 * (1 + erf(np.asarray(x, float) / np.sqrt(2)))
def _phi(x): x = np.asarray(x, float); return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

def gelu(x): x = np.asarray(x, float); return x * _Phi(x)
def gelu_grad(x): x = np.asarray(x, float); return _Phi(x) + x * _phi(x)

def gelu_tanh(x):
    x = np.asarray(x, float)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def silu(x): x = np.asarray(x, float); return x * sigmoid(x)
def silu_grad(x):
    s = sigmoid(x); return s * (1 + np.asarray(x, float) * (1 - s))

def softplus(x):
    x = np.asarray(x, float); return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))

def softplus_grad(x): return sigmoid(x)

def log_sigmoid(x): return -softplus(-np.asarray(x, float))

def swiglu(x, W, V, W2):
    return (silu(x @ W) * (x @ V)) @ W2
