import numpy as np
from submission import *  # noqa: F401,F403
def test_no_change_equals_minus_mean_adv():
    A = np.array([1.0, -2.0, 0.5])
    loss, g, cf = ppo_clip_loss(np.zeros(3), np.zeros(3), A)
    assert np.isclose(loss, -A.mean()) and np.allclose(g, -A / 3) and cf == 0.0

def test_clipping_positive_advantage():
    loss, g, cf = ppo_clip_loss(np.log([1.5]), np.zeros(1), np.array([2.0]), 0.2)
    assert np.isclose(loss, -1.2 * 2.0) and g[0] == 0.0 and cf == 1.0

def test_negative_advantage_not_clipped_when_ratio_high():
    loss, g, cf = ppo_clip_loss(np.log([1.5]), np.zeros(1), np.array([-2.0]), 0.2)
    assert np.isclose(loss, 1.5 * 2.0) and np.isclose(g[0], 1.5 * 2.0) and cf == 0.0

def test_gradient_numeric():
    rng = np.random.default_rng(0)
    ln, lo, A = rng.normal(0, 0.1, 6), rng.normal(0, 0.1, 6), rng.normal(size=6)
    _, g, _ = ppo_clip_loss(ln, lo, A)
    for i in range(6):
        e = np.zeros(6); e[i] = 1e-7
        num = (ppo_clip_loss(ln + e, lo, A)[0] - ppo_clip_loss(ln - e, lo, A)[0]) / 2e-7
        assert abs(num - g[i]) < 1e-5
