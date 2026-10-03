import numpy as np
from scipy.optimize import linprog

def lad_regression(X, y):
    n, d = X.shape
    c = np.concatenate([np.zeros(d), np.ones(n)])
    A = np.block([[X, -np.eye(n)], [-X, -np.eye(n)]])
    b = np.concatenate([y, -y])
    bounds = [(None, None)] * d + [(0, None)] * n
    r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return r.x[:d]

def chebyshev_regression(X, y):
    n, d = X.shape
    c = np.concatenate([np.zeros(d), [1.0]])
    ones = np.ones((n, 1))
    A = np.block([[X, -ones], [-X, -ones]])
    b = np.concatenate([y, -y])
    r = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * d + [(0, None)], method="highs")
    return r.x[:d], float(r.x[d])

def diet_problem(cost, nutrients, minimum):
    r = linprog(np.asarray(cost, float), A_ub=-np.asarray(nutrients, float), b_ub=-np.asarray(minimum, float),
                bounds=[(0, None)] * len(cost), method="highs")
    return r.x, float(r.fun)
