import numpy as np

def linear_betas(T, beta_1=1e-4, beta_T=0.02):
    return np.linspace(beta_1, beta_T, T)

def alpha_bars(betas):
    return np.cumprod(1 - np.asarray(betas, float))

def q_sample(x0, t, betas, noise):
    ab = alpha_bars(betas)[t]
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * noise

def posterior_mean(x0, xt, t, betas):
    ab = alpha_bars(betas); b = np.asarray(betas, float)
    c0 = np.sqrt(ab[t - 1]) * b[t] / (1 - ab[t])
    ct = np.sqrt(1 - b[t]) * (1 - ab[t - 1]) / (1 - ab[t])
    return c0 * x0 + ct * xt
