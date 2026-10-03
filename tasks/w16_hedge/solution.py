import numpy as np

def hedge(losses, eta):
    L = np.asarray(losses, float)
    T, n = L.shape
    logw = np.zeros(n)
    P = np.empty((T, n))
    for t in range(T):
        p = np.exp(logw - logw.max()); p /= p.sum()
        P[t] = p
        logw -= eta * L[t]
    regret = float((P * L).sum() - L.sum(0).min())
    return P, regret
