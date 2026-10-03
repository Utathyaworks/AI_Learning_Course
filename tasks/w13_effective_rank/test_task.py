import numpy as np
from submission import *  # noqa: F401,F403
def test_isotropic_full_rank():
    rng = np.random.default_rng(0)
    Z = np.linalg.qr(rng.normal(size=(200, 200)))[0][:, :10]
    Z = Z - Z.mean(0)
    assert abs(effective_rank(Z) - 10) < 0.2

def test_rank_one():
    rng = np.random.default_rng(1)
    Z = np.outer(rng.normal(size=100), rng.normal(size=8))
    assert abs(effective_rank(Z) - 1) < 1e-6

def test_between():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(500, 6)) * np.array([10, 1, 1, 0.1, 0.1, 0.1])
    r = effective_rank(Z)
    assert 1 < r < 6
