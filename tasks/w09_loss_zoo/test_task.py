import numpy as np
from submission import *  # noqa: F401,F403
def _bce(z, t):
    return np.mean(np.logaddexp(0, z) - t * z)

def test_focal_reduces_to_bce():
    rng = np.random.default_rng(0)
    z = rng.normal(size=50); t = (rng.random(50) > 0.5).astype(float)
    assert np.isclose(focal_loss(z, t, alpha=0.5, gamma=0.0), 0.5 * _bce(z, t))

def test_focal_downweights_easy_and_stable():
    easy = focal_loss(np.array([8.0]), np.array([1.0])); hard = focal_loss(np.array([-2.0]), np.array([1.0]))
    assert easy < 1e-6 < hard
    assert np.isfinite(focal_loss(np.array([1e4, -1e4]), np.array([0.0, 1.0])))

def test_dice():
    t = np.array([[0, 1], [1, 1]], float)
    assert np.isclose(dice_loss(t, t, eps=0.0), 0.0)
    assert np.isclose(dice_loss(1 - t, t, eps=0.0), 1.0)

def test_pinball_minimizer_is_quantile():
    rng = np.random.default_rng(1)
    y = rng.normal(size=2001)
    grid = np.linspace(-3, 3, 1201)
    best = grid[np.argmin([pinball_loss(y, c, 0.9) for c in grid])]
    assert abs(best - np.quantile(y, 0.9)) < 0.02

def test_label_smoothing():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(6, 4)); y = rng.integers(0, 4, 6)
    zs = z - z.max(1, keepdims=True); logp = zs - np.log(np.exp(zs).sum(1, keepdims=True))
    ce = -logp[np.arange(6), y].mean()
    assert np.isclose(label_smoothing_ce(z, y, 0.0), ce)
    assert np.isclose(label_smoothing_ce(z, y, 0.2), 0.8 * ce + 0.2 * (-logp.mean(1)).mean())
