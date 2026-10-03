# Diffusion forward process in closed form

Implement the DDPM forward process.

- `alpha_bars(betas)`: cumulative products `ᾱ_t = Π_{s≤t} (1 − β_s)` (index t from 0).
- `q_sample(x0, t, betas, noise)`: `x_t = √ᾱ_t · x0 + √(1 − ᾱ_t) · noise`.
- `linear_betas(T, beta_1=1e-4, beta_T=0.02)`: the linear schedule as an array of length T.
- `posterior_mean(x0, xt, t, betas)` (t ≥ 1): mean of q(x_{t−1} | x_t, x0)
  `= √ᾱ_{t−1} β_t/(1−ᾱ_t) · x0 + √(1−β_t)(1−ᾱ_{t−1})/(1−ᾱ_t) · x_t`.
