import numpy as np
from submission import *  # noqa: F401,F403
def test_additive_game():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    phi = shapley_values(lambda S: sum(a[i] for i in S), 4)
    assert np.allclose(phi, a)

def test_glove_game():
    # players 0,1 own left gloves, player 2 a right glove; a pair is worth 1
    v = lambda S: float(min(len(S & {0, 1}), len(S & {2})))
    phi = shapley_values(v, 3)
    assert np.allclose(phi, [1 / 6, 1 / 6, 2 / 3])

def test_efficiency_and_null():
    rng = np.random.default_rng(0)
    w = rng.random(5); w[4] = 0
    v = lambda S: (sum(w[i] for i in S)) ** 2
    phi = shapley_values(v, 5)
    assert np.isclose(phi.sum(), v(frozenset(range(5))) - v(frozenset()))
    assert abs(phi[4]) < 1e-12
