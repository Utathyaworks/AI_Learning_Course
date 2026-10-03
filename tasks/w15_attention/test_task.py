import numpy as np
from submission import *  # noqa: F401,F403
def test_weights_sum_to_one_and_shape():
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(2, 3, 5, 8)), rng.normal(size=(2, 3, 7, 8)), rng.normal(size=(2, 3, 7, 4))
    out, w = attention(Q, K, V)
    assert out.shape == (2, 3, 5, 4) and np.allclose(w.sum(-1), 1)

def test_causal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 4))
    out, w = attention(X, X, X, causal_mask(6))
    assert np.allclose(np.triu(w, 1), 0)
    assert np.allclose(out[0], X[0])

def test_scaling():
    q = np.array([[1.0, 1.0, 1.0, 1.0]]); k = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    _, w = attention(q, k, np.eye(2))
    e = np.exp(4 / 2)
    assert np.allclose(w[0], [e / (e + 1), 1 / (e + 1)])
