import numpy as np
from submission import *  # noqa: F401,F403
def test_random_is_about_log_n():
    rng = np.random.default_rng(0)
    z1 = rng.normal(size=(64, 128)); z2 = rng.normal(size=(64, 128))
    assert abs(info_nce(z1, z2, temperature=100.0) - np.log(64)) < 0.01

def test_aligned_pairs_low_loss():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(16, 32))
    assert info_nce(z, z + 0.01 * rng.normal(size=z.shape), 0.05) < 0.05

def test_scale_invariant_and_symmetric():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(8, 4)), rng.normal(size=(8, 4))
    assert abs(info_nce(a, b) - info_nce(3 * a, 0.5 * b)) < 1e-10
    assert abs(info_nce(a, b) - info_nce(b, a)) < 1e-10
