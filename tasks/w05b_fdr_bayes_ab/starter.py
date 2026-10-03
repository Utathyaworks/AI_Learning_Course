import numpy as np
from scipy import stats

def holm(pvals, alpha=0.05):
    raise NotImplementedError

def benjamini_hochberg(pvals, q=0.05):
    raise NotImplementedError

def beta_posterior(successes, trials, a=1.0, b=1.0):
    raise NotImplementedError

def prob_b_beats_a(sa, na, sb, nb) -> float:
    raise NotImplementedError
