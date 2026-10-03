import numpy as np

def gram_schmidt(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    for j in range(n):
        v = A[:, j].copy()
        for _ in range(2):  # twice for numerical stability
            v -= Q[:, :j] @ (Q[:, :j].T @ v)
        Q[:, j] = v / np.linalg.norm(v)
    return Q
