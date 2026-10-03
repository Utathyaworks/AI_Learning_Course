import numpy as np

def kalman_1d(zs, q, r, x0, p0):
    x, p = float(x0), float(p0)
    xs, ps = [], []
    for z in zs:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        xs.append(x); ps.append(p)
    return np.array(xs), np.array(ps)
