import numpy as np

def ridge_fit(X, y, lam, fit_intercept=True):
    X = np.asarray(X, float); y = np.asarray(y, float)
    if fit_intercept:
        xm, ym = X.mean(0), y.mean()
        Xc, yc = X - xm, y - ym
    else:
        Xc, yc = X, y
    d = X.shape[1]
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ yc)
    b = float(ym - xm @ w) if fit_intercept else 0.0
    return w, b
