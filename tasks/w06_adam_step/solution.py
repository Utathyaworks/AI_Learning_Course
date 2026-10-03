import numpy as np

def adam_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0, decoupled=True):
    w = np.asarray(w, float); g = np.asarray(g, float)
    if not decoupled and weight_decay:
        g = g + weight_decay * w
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
    step = lr * mh / (np.sqrt(vh) + eps)
    if decoupled and weight_decay:
        w = w - lr * weight_decay * w
    return w - step, m, v
