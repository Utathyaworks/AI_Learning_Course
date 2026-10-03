import numpy as np

def mlp_loss_grads(X, y, W1, b1, W2, b2):
    a = X @ W1 + b1
    h = np.maximum(a, 0)
    z = h @ W2 + b2
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    n = len(y)
    loss = -logp[np.arange(n), y].mean()
    dz = np.exp(logp); dz[np.arange(n), y] -= 1; dz /= n
    dW2 = h.T @ dz; db2 = dz.sum(0)
    dh = dz @ W2.T
    da = dh * (a > 0)
    dW1 = X.T @ da; db1 = da.sum(0)
    return float(loss), {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
