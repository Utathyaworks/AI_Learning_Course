import numpy as np
from submission import *  # noqa: F401,F403
from scipy import stats

def test_importance_gaussian_mean():
    rng = np.random.default_rng(0)
    logp = lambda x: -0.5 * (x - 2.0) ** 2            # N(2, 1), unnormalized
    qs = lambda n, r: r.normal(0, 3, size=n)
    ql = lambda x: stats.norm(0, 3).logpdf(x)
    est, ess = importance_estimate(lambda x: x, logp, qs, ql, 50000, rng)
    assert abs(est - 2.0) < 0.05 and 0 < ess <= 50000

def test_light_tailed_proposal_has_low_ess():
    rng = np.random.default_rng(1)
    logp = lambda x: -0.5 * (x / 3) ** 2
    _, good = importance_estimate(lambda x: x, logp, lambda n, r: r.normal(0, 4, n), stats.norm(0, 4).logpdf, 20000, rng)
    _, bad = importance_estimate(lambda x: x, logp, lambda n, r: r.normal(0, 0.7, n), stats.norm(0, 0.7).logpdf, 20000, rng)
    assert bad < 0.2 * good

def test_mh_targets_correlated_gaussian():
    rng = np.random.default_rng(2)
    C = np.array([[1.0, 0.8], [0.8, 1.0]]); P = np.linalg.inv(C)
    logp = lambda x: -0.5 * x @ P @ x
    s, acc = metropolis_hastings(logp, [3.0, -3.0], 40000, 0.8, rng)
    s = s[5000:]
    assert s.shape == (35000, 2) and 0.2 < acc < 0.8
    assert np.allclose(s.mean(0), 0, atol=0.12)
    assert np.allclose(np.cov(s.T), C, atol=0.15)
