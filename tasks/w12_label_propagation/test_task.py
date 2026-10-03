import numpy as np
from submission import *  # noqa: F401,F403
def test_chain_interpolates():
    n = 5
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1
    F = label_propagation(W, [0, 4], np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(F[:, 0], [1, 0.75, 0.5, 0.25, 0])
    assert np.allclose(F.sum(1), 1)

def test_two_clusters():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 0.3, (10, 2)); B = rng.normal(5, 0.3, (10, 2))
    X = np.vstack([A, B])
    d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    W = np.exp(-d2); np.fill_diagonal(W, 0)
    F = label_propagation(W, [0, 10], np.eye(2))
    assert np.all(F[:10].argmax(1) == 0) and np.all(F[10:].argmax(1) == 1)

def test_harmonic_property():
    rng = np.random.default_rng(1)
    W = rng.random((7, 7)); W = W + W.T; np.fill_diagonal(W, 0)
    F = label_propagation(W, [0, 1], np.eye(2))
    for u in range(2, 7):
        assert np.allclose(F[u], W[u] @ F / W[u].sum())
