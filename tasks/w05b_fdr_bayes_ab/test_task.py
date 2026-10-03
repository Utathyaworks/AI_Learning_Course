import numpy as np
from submission import *  # noqa: F401,F403
from statsmodels.stats.multitest import multipletests

def test_holm_and_bh_match_statsmodels():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = np.concatenate([rng.uniform(0, 0.01, 5), rng.uniform(0, 1, 25)])
        rng.shuffle(p)
        assert np.array_equal(holm(p), multipletests(p, 0.05, "holm")[0])
        assert np.array_equal(benjamini_hochberg(p), multipletests(p, 0.05, "fdr_bh")[0])

def test_bh_step_up():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    assert benjamini_hochberg(p).all() and holm(p).sum() == 1

def test_beta_posterior():
    ap, bp, mean, (lo, hi) = beta_posterior(30, 100)
    assert (ap, bp) == (31, 71) and np.isclose(mean, 31 / 102) and lo < mean < hi

def test_prob_b_beats_a():
    assert np.isclose(prob_b_beats_a(50, 100, 50, 100), 0.5, atol=1e-4)
    rng = np.random.default_rng(1)
    a = rng.beta(1 + 120, 1 + 880, 400000); b = rng.beta(1 + 150, 1 + 850, 400000)
    assert np.isclose(prob_b_beats_a(120, 1000, 150, 1000), np.mean(b > a), atol=3e-3)
    assert prob_b_beats_a(10, 1000, 100, 1000) > 0.9999
