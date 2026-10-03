import numpy as np
from scipy import stats

def wilson_interval(k, n, conf=0.95):
    raise NotImplementedError

def mcnemar_exact(a_correct, b_correct) -> float:
    raise NotImplementedError

def paired_t(x, y):
    raise NotImplementedError

def cramer_rao_bernoulli(p, n) -> float:
    raise NotImplementedError
