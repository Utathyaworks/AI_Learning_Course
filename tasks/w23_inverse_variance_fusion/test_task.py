import numpy as np
from submission import *  # noqa: F401,F403
def test_two_sensors():
    m, v = fuse(np.array([0.0, 10.0]), np.array([1.0, 4.0]))
    assert np.isclose(m, 2.0) and np.isclose(v, 0.8)

def test_matches_kalman_update():
    # fusing a prior N(x, p) with a measurement N(z, r) equals one Kalman update
    x, p, z, r = 1.0, 2.0, 3.0, 0.5
    m, v = fuse(np.array([x, z]), np.array([p, r]))
    k = p / (p + r)
    assert np.isclose(m, x + k * (z - x)) and np.isclose(v, (1 - k) * p)

def test_missing_sensors():
    means = np.array([[1.0, 5.0], [3.0, 7.0]]); var = np.ones((2, 2))
    avail = np.array([[True, False], [True, False]])
    m, v = fuse(means, var, avail)
    assert np.isclose(m[0], 2.0) and np.isclose(v[0], 0.5)
    assert np.isnan(m[1]) and np.isinf(v[1])
