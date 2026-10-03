import numpy as np

def fuse(means, variances, available=None):
    m = np.asarray(means, float); v = np.asarray(variances, float)
    a = np.ones_like(m, bool) if available is None else np.asarray(available, bool)
    prec = np.where(a, 1.0 / v, 0.0)
    tot = prec.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(tot > 0, (prec * np.where(a, m, 0.0)).sum(0) / np.where(tot > 0, tot, 1.0), np.nan)
        var = np.where(tot > 0, 1.0 / np.where(tot > 0, tot, 1.0), np.inf)
    return mean, var
