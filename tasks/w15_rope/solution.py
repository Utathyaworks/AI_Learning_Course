import numpy as np

def apply_rope(x, positions, base=10000.0):
    x = np.asarray(x, float)
    L, d = x.shape
    k = np.arange(d // 2)
    theta = base ** (-2 * k / d)
    ang = np.asarray(positions, float)[:, None] * theta[None, :]
    c, s = np.cos(ang), np.sin(ang)
    x0, x1 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x0 * c - x1 * s
    out[:, 1::2] = x0 * s + x1 * c
    return out
