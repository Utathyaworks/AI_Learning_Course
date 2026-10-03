import numpy as np
from scipy.linalg import expm

def topk_route(logits, k):
    idx = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(logits, idx, 1)
    g = np.exp(top - top.max(1, keepdims=True))
    return idx, g / g.sum(1, keepdims=True)

def load_balance_loss(logits, idx):
    T, E = logits.shape
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    f = np.bincount(idx[:, 0], minlength=E) / T
    return float(E * np.sum(f * p.mean(0)))

def discretize_zoh(A, B, dt):
    A = np.asarray(A, float); B = np.asarray(B, float)
    Ad = expm(A * dt)
    Bd = np.linalg.solve(A, (Ad - np.eye(len(A))) @ B)
    return Ad, Bd

def ssm_recurrent(Ad, Bd, C, u):
    x = np.zeros(Ad.shape[0]); y = []
    for ut in u:
        x = Ad @ x + Bd.ravel() * ut
        y.append(float(np.asarray(C).ravel() @ x))
    return np.array(y)

def ssm_kernel(Ad, Bd, C, L):
    K, v = [], Bd.ravel().copy()
    c = np.asarray(C).ravel()
    for _ in range(L):
        K.append(c @ v); v = Ad @ v
    return np.array(K)

def ssm_convolutional(Ad, Bd, C, u):
    K = ssm_kernel(Ad, Bd, C, len(u))
    return np.array([np.dot(K[:t + 1][::-1], u[:t + 1]) for t in range(len(u))])
