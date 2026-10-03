import numpy as np

def softmax_cross_entropy(logits, y):
    z = np.asarray(logits, float)
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    n = len(y)
    loss = -logp[np.arange(n), y].mean()
    p = np.exp(logp)
    p[np.arange(n), y] -= 1
    return float(loss), p / n
