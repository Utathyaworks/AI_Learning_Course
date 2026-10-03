import numpy as np
from scipy import stats, integrate

def holm(pvals, alpha=0.05):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); rej = np.zeros(m, bool)
    for i, j in enumerate(order):
        if p[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    return rej

def benjamini_hochberg(pvals, q=0.05):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p)
    ok = np.nonzero(p[order] <= q * np.arange(1, m + 1) / m)[0]
    rej = np.zeros(m, bool)
    if len(ok):
        rej[order[:ok[-1] + 1]] = True
    return rej

def beta_posterior(successes, trials, a=1.0, b=1.0):
    ap, bp = a + successes, b + trials - successes
    d = stats.beta(ap, bp)
    return ap, bp, float(d.mean()), (float(d.ppf(0.025)), float(d.ppf(0.975)))

def prob_b_beats_a(sa, na, sb, nb):
    A = stats.beta(1 + sa, 1 + na - sa); B = stats.beta(1 + sb, 1 + nb - sb)
    lo, hi = min(A.ppf(1e-12), B.ppf(1e-12)), max(A.ppf(1 - 1e-12), B.ppf(1 - 1e-12))
    val, _ = integrate.quad(lambda t: B.pdf(t) * A.cdf(t), lo, hi, limit=500, points=[A.mean(), B.mean()])
    return float(val)
