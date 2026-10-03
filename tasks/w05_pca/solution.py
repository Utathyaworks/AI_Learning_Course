import numpy as np

def pca(X, k):
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    C = Xc.T @ Xc / (len(X) - 1)
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1][:k]
    comps = vecs[:, idx].T
    return comps, vals[idx], Xc @ comps.T
