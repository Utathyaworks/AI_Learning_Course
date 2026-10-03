import numpy as np
from submission import *  # noqa: F401,F403
def test_lambda_zero_is_td():
    r = np.array([1.0, 0.0, 2.0]); v = np.array([0.5, 0.2, 0.1]); d = np.array([False, False, False])
    adv, _ = gae(r, v, d, 0.3, 0.9, 0.0)
    assert np.allclose(adv, r + 0.9 * np.array([0.2, 0.1, 0.3]) - v)

def test_lambda_one_is_mc():
    r = np.array([1.0, 0.0, 2.0]); v = np.zeros(3); d = np.array([False, False, True])
    adv, ret = gae(r, v, d, 99.0, 0.5, 1.0)
    assert np.allclose(ret, [1 + 0.5 * 0 + 0.25 * 2, 0 + 0.5 * 2, 2])

def test_done_cuts_bootstrap():
    r = np.array([1.0, 1.0]); v = np.array([0.0, 0.0]); d = np.array([True, False])
    adv, _ = gae(r, v, d, 10.0, 1.0, 1.0)
    assert np.isclose(adv[0], 1.0) and np.isclose(adv[1], 11.0)
