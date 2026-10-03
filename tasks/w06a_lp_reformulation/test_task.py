import numpy as np
from submission import *  # noqa: F401,F403
from scipy.optimize import minimize

def test_lad_resists_outliers():
    r = np.random.default_rng(0)
    x = np.linspace(0, 10, 40); X = np.column_stack([np.ones_like(x), x])
    y = 1.0 + 2.0 * x + 0.05 * r.normal(size=40); y[::8] += 30          # 5 big outliers
    w = lad_regression(X, y)
    ls = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(w, [1, 2], atol=0.1) and abs(ls[0] - 1) > 1
    obj = lambda v: np.abs(y - X @ v).sum()
    assert obj(w) <= obj(ls) + 1e-9 and obj(w) <= minimize(obj, ls, method="Nelder-Mead").fun + 1e-6

def test_chebyshev():
    X = np.column_stack([np.ones(3), [0.0, 1.0, 2.0]])
    w, s = chebyshev_regression(X, np.array([0.0, 2.0, 0.0]))
    assert np.isclose(s, 1.0) and np.allclose(w, [1.0, 0.0])
    assert np.isclose(np.max(np.abs(np.array([0, 2, 0]) - X @ w)), s)

def test_diet():
    cost = [2.0, 3.0]; nutrients = [[1, 2], [3, 1]]; minimum = [8, 9]
    x, total = diet_problem(cost, nutrients, minimum)
    assert np.isclose(total, 13.0) and np.allclose(x, [2, 3]) and np.all(np.array(nutrients) @ x >= np.array(minimum) - 1e-9)
