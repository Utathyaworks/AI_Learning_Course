import numpy as np

def soft_threshold(v, t):
    v = np.asarray(v, float)
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)

def _L(X):
    return float(np.linalg.eigvalsh(X.T @ X).max())

def lasso_ista(X, y, lam, iters=5000):
    L = _L(X); w = np.zeros(X.shape[1])
    for _ in range(iters):
        w = soft_threshold(w - X.T @ (X @ w - y) / L, lam / L)
    return w

def project_l2_ball(v, r):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v if n <= r else v * (r / n)

def project_simplex(v):
    v = np.asarray(v, float)
    u = np.sort(v)[::-1]; css = np.cumsum(u)
    k = np.nonzero(u - (css - 1) / np.arange(1, len(v) + 1) > 0)[0][-1]
    tau = (css[k] - 1) / (k + 1)
    return np.maximum(v - tau, 0.0)

def ridge_constrained(X, y, r, iters=5000):
    L = _L(X); w = np.zeros(X.shape[1])
    for _ in range(iters):
        w = project_l2_ball(w - X.T @ (X @ w - y) / L, r)
    return w

def ridge_penalized(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
