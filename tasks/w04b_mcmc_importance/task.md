# Importance sampling and Metropolis–Hastings

Implement:

- `importance_estimate(f, logp, q_sample, q_logpdf, n, rng)`: self-normalized importance-sampling estimate of
  E_p[f(X)] where `logp` is an **unnormalized** log-density. Draw `x = q_sample(n, rng)` (shape (n,) or (n, d)),
  weights `w ∝ exp(logp(x) − q_logpdf(x))` (stabilize with the max). Return `(estimate, ess)` where
  `ess = (Σw)² / Σw²`.
- `metropolis_hastings(logp, x0, n, step, rng)`: random-walk MH with Gaussian proposals of std `step`. Return
  `(samples, acceptance_rate)` with `samples` of shape (n, d) (the chain after each step, including repeats).
