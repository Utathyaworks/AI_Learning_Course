import numpy as np

def linear_betas(T, beta_1=1e-4, beta_T=0.02):
    raise NotImplementedError

def alpha_bars(betas):
    raise NotImplementedError

def q_sample(x0, t, betas, noise):
    raise NotImplementedError

def posterior_mean(x0, xt, t, betas):
    raise NotImplementedError
