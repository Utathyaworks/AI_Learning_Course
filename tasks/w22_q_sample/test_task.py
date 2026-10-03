import numpy as np
from submission import *  # noqa: F401,F403
def test_schedule():
    b = linear_betas(1000)
    assert len(b) == 1000 and np.isclose(b[0], 1e-4) and np.isclose(b[-1], 0.02)
    ab = alpha_bars(b)
    assert np.all(np.diff(ab) < 0) and ab[-1] < 1e-3

def test_closed_form_matches_iterative():
    rng = np.random.default_rng(0)
    b = linear_betas(50); x0 = rng.normal(size=10000)
    x = x0.copy()
    for t in range(30):
        x = np.sqrt(1 - b[t]) * x + np.sqrt(b[t]) * rng.normal(size=x.shape)
    ab = alpha_bars(b)[29]
    xs = q_sample(x0, 29, b, rng.normal(size=x0.shape))
    assert abs(np.corrcoef(x, x0)[0, 1] - np.corrcoef(xs, x0)[0, 1]) < 0.03
    assert abs(x.var() - xs.var()) < 0.05

def test_posterior_mean_consistency():
    b = linear_betas(100); t = 40
    x0 = np.array([1.0, -1.0]); eps = np.array([0.3, 0.7])
    xt = q_sample(x0, t, b, eps)
    ab = alpha_bars(b)
    mu = posterior_mean(x0, xt, t, b)
    alt = (xt - b[t] / np.sqrt(1 - ab[t]) * eps) / np.sqrt(1 - b[t])
    assert np.allclose(mu, alt)
