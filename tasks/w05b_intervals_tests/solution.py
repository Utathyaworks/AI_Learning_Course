import numpy as np
from scipy import stats

def wilson_interval(k, n, conf=0.95):
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))

def mcnemar_exact(a_correct, b_correct):
    a = np.asarray(a_correct, bool); b_ = np.asarray(b_correct, bool)
    b = int(np.sum(a & ~b_)); c = int(np.sum(~a & b_))
    if b + c == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))

def paired_t(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    return float(t), float(2 * stats.t.sf(abs(t), n - 1))

def cramer_rao_bernoulli(p, n):
    return float(p * (1 - p) / n)
