import numpy as np

def gaussian_condition(mu, S, idx_obs, x_obs):
    raise NotImplementedError

def kl_gaussians(mu0, S0, mu1, S1) -> float:
    raise NotImplementedError

def hoeffding_n(eps, delta) -> int:
    raise NotImplementedError

def hoeffding_halfwidth(n, delta) -> float:
    raise NotImplementedError
