import numpy as np

def transform(f, a=1, b=1, c=0, d=0):
    raise NotImplementedError

def bisect_root(f, lo, hi, tol=1e-10):
    raise NotImplementedError

def is_monotone(y):
    raise NotImplementedError

def is_convex_on_grid(y, tol=1e-12):
    raise NotImplementedError

def fit_power_law(x, y):
    raise NotImplementedError

def fit_exponential(x, y):
    raise NotImplementedError
