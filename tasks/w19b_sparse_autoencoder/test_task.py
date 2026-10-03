import numpy as np
from submission import *  # noqa: F401,F403
def test_decoder_norm():
    W = normalize_decoder(np.random.default_rng(0).normal(size=(6, 4)))
    assert np.allclose(np.linalg.norm(W, axis=1), 1)

def test_perfect_dictionary_recovers_sparse_codes():
    rng = np.random.default_rng(1)
    d, m, n = 8, 8, 500
    D = np.eye(d)[rng.permutation(d)]                      # exact orthonormal features (a permutation)
    codes = (rng.random((n, m)) < 0.2) * rng.uniform(0.5, 2, (n, m))
    X = codes @ D
    f, Xh = sae_forward(X, D.T, np.zeros(m), D, np.zeros(d))
    assert np.allclose(f, codes) and np.allclose(Xh, X)
    m_ = sae_metrics(X, f, Xh)
    assert np.isclose(m_["fve"], 1) and np.isclose(m_["l0"], np.mean(np.sum(codes > 0, 1)))
    assert np.isclose(sae_loss(X, f, Xh, 0.1), 0.1 * np.mean(codes.sum(1)))

def test_topk():
    pre = np.array([[0.5, -1.0, 2.0, 1.0], [3.0, 2.0, 1.0, 0.0]])
    assert np.allclose(topk_encode(pre, 2), [[0, 0, 2, 1], [3, 2, 0, 0]])

def test_dead_latents():
    X = np.ones((4, 2)); f = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 1], [1, 0, 0.0]])
    assert np.isclose(sae_metrics(X, f, X)["dead"], 1 / 3)
