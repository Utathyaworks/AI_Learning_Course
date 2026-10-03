import numpy as np
from submission import *  # noqa: F401,F403
def _obj(X, y, w, b, lam):
    r = y - X @ w - b
    return r @ r + lam * w @ w

def test_matches_ols_when_lam_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)); y = X @ np.array([1.0, -2.0, 0.5]) + 3.0
    w, b = ridge_fit(X, y, 0.0)
    assert np.allclose(w, [1.0, -2.0, 0.5], atol=1e-8) and abs(b - 3.0) < 1e-8

def test_optimality():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 4)); y = rng.normal(size=40)
    w, b = ridge_fit(X, y, 2.5)
    base = _obj(X, y, w, b, 2.5)
    for _ in range(20):
        dw = rng.normal(size=4) * 1e-3; db = rng.normal() * 1e-3
        assert _obj(X, y, w + dw, b + db, 2.5) >= base - 1e-9

def test_no_intercept():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 2)); y = rng.normal(size=30)
    w, b = ridge_fit(X, y, 1.0, fit_intercept=False)
    assert b == 0.0
    assert np.allclose(w, np.linalg.solve(X.T @ X + np.eye(2), X.T @ y))

def test_shrinks_with_lambda():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 3)); y = rng.normal(size=30)
    n1 = np.linalg.norm(ridge_fit(X, y, 0.1)[0]); n2 = np.linalg.norm(ridge_fit(X, y, 100.0)[0])
    assert n2 < n1
