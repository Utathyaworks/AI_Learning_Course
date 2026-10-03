import numpy as np
from submission import *  # noqa: F401,F403
def test_uniform_when_scale_tiny():
    rng = np.random.default_rng(0)
    i, t = rng.normal(size=(8, 16)), rng.normal(size=(8, 16))
    loss, _ = clip_loss(i, t, np.log(1e-6))
    assert abs(loss - np.log(8)) < 1e-4

def test_aligned_low_loss():
    rng = np.random.default_rng(1)
    i = rng.normal(size=(8, 64))
    loss, _ = clip_loss(i, i + 0.01 * rng.normal(size=i.shape), np.log(100))
    assert loss < 0.01

def test_scale_gradient_numeric():
    rng = np.random.default_rng(2)
    i, t = rng.normal(size=(6, 5)), rng.normal(size=(6, 5))
    _, g = clip_loss(i, t, 1.3)
    num = (clip_loss(i, t, 1.3 + 1e-6)[0] - clip_loss(i, t, 1.3 - 1e-6)[0]) / 2e-6
    assert abs(g - num) < 1e-6
