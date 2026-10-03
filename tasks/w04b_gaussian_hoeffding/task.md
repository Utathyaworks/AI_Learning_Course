# Gaussian conditioning and concentration bounds

Implement:

- `gaussian_condition(mu, S, idx_obs, x_obs)`: for x ~ N(mu, S), return `(mean, cov)` of the unobserved coordinates
  given that the coordinates `idx_obs` equal `x_obs` (Schur complement; solve, don't invert).
- `kl_gaussians(mu0, S0, mu1, S1)`: KL( N(mu0, S0) ‖ N(mu1, S1) ).
- `hoeffding_n(eps, delta)`: the smallest integer n such that Hoeffding's two-sided bound guarantees
  P(|mean − p| ≥ eps) ≤ delta for i.i.d. variables in [0, 1], i.e. `2·exp(−2 n eps²) ≤ delta`.
- `hoeffding_halfwidth(n, delta)`: the matching half-width eps for a given n.
