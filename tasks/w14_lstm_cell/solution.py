import numpy as np

def _sig(z):
    return 1 / (1 + np.exp(-z))

def lstm_cell(x, h, c, W, U, b):
    a = x @ W + h @ U + b
    H = h.shape[1]
    i, f, g, o = a[:, :H], a[:, H:2*H], a[:, 2*H:3*H], a[:, 3*H:]
    c2 = _sig(f) * c + _sig(i) * np.tanh(g)
    return _sig(o) * np.tanh(c2), c2
