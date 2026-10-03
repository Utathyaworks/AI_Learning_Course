import numpy as np
from submission import *  # noqa: F401,F403
from scipy.optimize import minimize

def _data(seed=0, n=60, d=8):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d)); w = np.zeros(d); w[:3] = [2.0, -1.5, 1.0]
    return X, X @ w + 0.1 * r.normal(size=n)

def test_soft_threshold():
    assert np.allclose(soft_threshold([-3, -0.5, 0, 0.5, 3], 1), [-2, 0, 0, 0, 2])

def test_lasso_is_sparse_and_optimal():
    X, y = _data()
    lam = 5.0
    w = lasso_ista(X, y, lam)
    assert np.sum(np.abs(w) > 1e-8) <= 4 and np.all(np.abs(w[:3]) > 0.5)
    g = X.T @ (X @ w - y)                      # KKT: |g_j| <= lam where w_j = 0, g_j = -lam sign(w_j) otherwise
    nz = np.abs(w) > 1e-8
    assert np.allclose(g[nz], -lam * np.sign(w[nz]), atol=1e-5) and np.all(np.abs(g[~nz]) <= lam + 1e-5)

def test_projections():
    v = np.array([3.0, 4.0])
    assert np.allclose(project_l2_ball(v, 1), [0.6, 0.8]) and np.allclose(project_l2_ball(v, 10), v)
    p = project_simplex([0.5, 2.0, -1.0])
    assert np.allclose(p, [0, 1, 0]) and np.isclose(project_simplex([0.2, 0.3, 0.1]).sum(), 1)
    r = np.random.default_rng(1)
    for _ in range(20):
        v = r.normal(size=6) * 3; p = project_simplex(v)
        res = minimize(lambda x: np.sum((x - v) ** 2), np.ones(6) / 6, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
                       bounds=[(0, None)] * 6, method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
        assert np.allclose(p, res.x, atol=1e-5)

def test_penalized_and_constrained_ridge_agree():
    X, y = _data(2)
    lam = 20.0
    wp = ridge_penalized(X, y, lam)
    wc = ridge_constrained(X, y, np.linalg.norm(wp), iters=20000)
    assert np.allclose(wc, wp, atol=1e-6)        # same point for the matching radius
