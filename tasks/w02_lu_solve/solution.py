import numpy as np

def lu_solve(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    vec = b.ndim == 1
    B = b.reshape(len(b), -1).copy()
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    perm = np.arange(n)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if p != k:
            U[[k, p], k:] = U[[p, k], k:]
            L[[k, p], :k] = L[[p, k], :k]
            perm[[k, p]] = perm[[p, k]]
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    B = B[perm]
    y = np.zeros_like(B)
    for i in range(n):
        y[i] = B[i] - L[i, :i] @ y[:i]
    x = np.zeros_like(y)
    for i in reversed(range(n)):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x[:, 0] if vec else x
