import numpy as np
from submission import *  # noqa: F401,F403
def test_gini_values():
    assert gini([0, 0, 0]) == 0.0
    assert abs(gini([0, 1]) - 0.5) < 1e-12
    assert abs(gini([0, 1, 2]) - 2 / 3) < 1e-12

def test_perfect_split():
    t, g = best_split([1, 2, 3, 10, 11, 12], [0, 0, 0, 1, 1, 1])
    assert abs(t - 6.5) < 1e-12 and g == 0.0

def test_constant_feature():
    t, g = best_split([5, 5, 5, 5], [0, 1, 0, 1])
    assert t is None and abs(g - 0.5) < 1e-12

def test_against_brute():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 10, 40).astype(float); y = (x + rng.integers(0, 4, 40) > 6).astype(int)
    t, g = best_split(x, y)
    vals = np.unique(x)
    cands = [(a + b) / 2 for a, b in zip(vals[:-1], vals[1:])]
    best = min((len(y[x <= c]) * gini(y[x <= c]) + len(y[x > c]) * gini(y[x > c])) / len(y) for c in cands)
    assert abs(g - best) < 1e-12
