import numpy as np
from submission import *  # noqa: F401,F403
def test_shapes_and_orthonormal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    c, v, P = pca(X, 3)
    assert c.shape == (3, 5) and v.shape == (3,) and P.shape == (100, 3)
    assert np.allclose(c @ c.T, np.eye(3), atol=1e-8)
    assert np.all(np.diff(v) <= 1e-12)

def test_recovers_direction():
    rng = np.random.default_rng(1)
    t = rng.normal(size=500) * 5
    d = np.array([3.0, 4.0]) / 5
    X = np.outer(t, d) + rng.normal(size=(500, 2)) * 0.1 + 7
    c, v, P = pca(X, 1)
    assert abs(abs(c[0] @ d) - 1) < 1e-3

def test_variance_matches_projection():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4)) @ rng.normal(size=(4, 4))
    c, v, P = pca(X, 4)
    assert np.allclose(P.var(0, ddof=1), v, atol=1e-8)
    assert np.allclose(v.sum(), np.trace(np.cov(X.T)), atol=1e-8)
