import numpy as np

def gini(y):
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return float(1 - (p * p).sum())

def best_split(x, y):
    x = np.asarray(x, float); y = np.asarray(y)
    vals = np.unique(x)
    if len(vals) == 1:
        return None, gini(y)
    best = (None, np.inf)
    for a, b in zip(vals[:-1], vals[1:]):
        t = (a + b) / 2
        L, R = y[x <= t], y[x > t]
        g = (len(L) * gini(L) + len(R) * gini(R)) / len(y)
        if g < best[1] - 1e-15:
            best = (t, g)
    return best[0], float(best[1])
