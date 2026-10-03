import numpy as np

def transform(f, a=1, b=1, c=0, d=0):
    return lambda x: a * f(b * (np.asarray(x, float) - c)) + d

def bisect_root(f, lo, hi, tol=1e-10):
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return float(lo)
    if fhi == 0:
        return float(hi)
    if np.sign(flo) == np.sign(fhi):
        raise ValueError("f(lo) and f(hi) must have opposite signs")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi); fm = f(mid)
        if fm == 0:
            return float(mid)
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
    return float(0.5 * (lo + hi))

def is_monotone(y):
    d = np.diff(np.asarray(y, float))
    if np.all(d >= 0):
        return "increasing"
    if np.all(d <= 0):
        return "decreasing"
    return "neither"

def is_convex_on_grid(y, tol=1e-12):
    y = np.asarray(y, float)
    return bool(np.all(y[:-2] - 2 * y[1:-1] + y[2:] >= -tol))

def fit_power_law(x, y):
    k, loga = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(loga)), float(k)

def fit_exponential(x, y):
    k, loga = np.polyfit(np.asarray(x, float), np.log(y), 1)
    return float(np.exp(loga)), float(k)
