import numpy as np
from submission import *  # noqa: F401,F403
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar

def test_wilson_matches_statsmodels():
    for k, n in [(0, 20), (3, 20), (91, 100), (1800, 2000)]:
        assert np.allclose(wilson_interval(k, n), proportion_confint(k, n, method="wilson"))

def test_wilson_stays_inside_unit_interval():
    lo, hi = wilson_interval(20, 20)
    assert 0 <= lo < 1 and hi <= 1 + 1e-12

def test_mcnemar():
    rng = np.random.default_rng(0)
    a = rng.random(500) < 0.85; b = a.copy(); flip = rng.random(500) < 0.08; b[flip] = ~b[flip]
    t = [[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]]
    assert np.isclose(mcnemar_exact(a, b), mcnemar(t, exact=True).pvalue)
    assert mcnemar_exact(a, a) == 1.0

def test_paired_t():
    rng = np.random.default_rng(1)
    x = rng.normal(size=12); y = x + rng.normal(0.3, 0.2, size=12)
    r = stats.ttest_rel(x, y)
    assert np.allclose(paired_t(x, y), (r.statistic, r.pvalue))

def test_crlb():
    assert np.isclose(cramer_rao_bernoulli(0.9, 1000), 9e-5)
