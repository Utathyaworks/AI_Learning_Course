import numpy as np

def kmeans_step(X, centers):
    X = np.asarray(X, float); C = np.asarray(centers, float)
    d2 = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
    d2 = np.maximum(d2, 0)
    labels = np.argmin(d2, axis=1)
    inertia = float(d2[np.arange(len(X)), labels].sum())
    new = C.copy()
    for j in range(len(C)):
        m = labels == j
        if m.any():
            new[j] = X[m].mean(0)
    return labels, new, inertia
