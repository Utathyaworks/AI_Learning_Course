import numpy as np

def logistic_loss_grad(X, y, w, b, l2=0.0):
    X = np.asarray(X, float); y = np.asarray(y, float)
    z = X @ w + b
    # log(1 + e^z) - y z, computed stably
    loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * w @ w
    p = np.exp(-np.logaddexp(0.0, -z))  # stable sigmoid
    r = (p - y) / len(y)
    return float(loss), X.T @ r + l2 * w, float(r.sum())
