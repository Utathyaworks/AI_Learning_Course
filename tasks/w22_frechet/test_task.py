import numpy as np
from submission import *  # noqa: F401,F403
def test_identical_is_zero():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5)); S = A @ A.T
    assert abs(frechet_distance(np.ones(5), S, np.ones(5), S)) < 1e-8

def test_diagonal_closed_form():
    s1, s2 = np.array([1.0, 4.0]), np.array([9.0, 1.0])
    d = frechet_distance(np.zeros(2), np.diag(s1), np.array([1.0, 2.0]), np.diag(s2))
    assert np.isclose(d, 5 + ((np.sqrt(s1) - np.sqrt(s2)) ** 2).sum())

def test_stats():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(200, 3))
    m, S = stats(f)
    assert np.allclose(m, f.mean(0)) and np.allclose(S, np.cov(f.T))

def test_symmetric():
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
    S1, S2 = A @ A.T, B @ B.T
    m1, m2 = rng.normal(size=4), rng.normal(size=4)
    assert np.isclose(frechet_distance(m1, S1, m2, S2), frechet_distance(m2, S2, m1, S1))
