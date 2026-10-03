import numpy as np

def label_propagation(W, labeled_idx, Y_l):
    W = np.asarray(W, float); n = len(W)
    L = np.asarray(labeled_idx)
    U = np.setdiff1d(np.arange(n), L)
    D = np.diag(W.sum(1))
    Luu = (D - W)[np.ix_(U, U)]
    Fu = np.linalg.solve(Luu, W[np.ix_(U, L)] @ Y_l)
    F = np.zeros((n, Y_l.shape[1]))
    F[L] = Y_l; F[U] = Fu
    return F
