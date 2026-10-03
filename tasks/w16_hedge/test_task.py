import numpy as np
from submission import *  # noqa: F401,F403
def test_first_round_uniform():
    P, _ = hedge(np.array([[1.0, 0.0, 0.5]]), 0.5)
    assert np.allclose(P[0], 1 / 3)

def test_matches_formula():
    rng = np.random.default_rng(0)
    L = rng.random((20, 4)); eta = 0.3
    P, _ = hedge(L, eta)
    cum = np.vstack([np.zeros(4), np.cumsum(L, 0)[:-1]])
    ref = np.exp(-eta * cum); ref /= ref.sum(1, keepdims=True)
    assert np.allclose(P, ref)

def test_regret_bound():
    rng = np.random.default_rng(1)
    T, n = 5000, 10
    L = rng.random((T, n)); L[:, 3] *= 0.8
    eta = np.sqrt(8 * np.log(n) / T)
    _, R = hedge(L, eta)
    assert R <= np.sqrt(T * np.log(n) / 2) + 1e-9

def test_long_horizon_stable():
    L = np.ones((20000, 3)); L[:, 0] = 0
    P, R = hedge(L, 1.0)
    assert np.all(np.isfinite(P)) and P[-1, 0] > 0.999
