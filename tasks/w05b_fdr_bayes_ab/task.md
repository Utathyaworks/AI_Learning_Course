# Multiple testing and a Bayesian A/B test

Implement:

- `holm(pvals, alpha=0.05)`: boolean array, True where Holm's step-down procedure rejects.
- `benjamini_hochberg(pvals, q=0.05)`: boolean array, True where BH rejects (largest k with p_(k) ≤ k·q/m; reject
  the k smallest).
- `beta_posterior(successes, trials, a=1.0, b=1.0)`: return `(alpha_post, beta_post, mean, (lo, hi))` for a
  Beta(a, b) prior on a Bernoulli rate, with an equal-tailed 95% credible interval.
- `prob_b_beats_a(sa, na, sb, nb)`: P(θ_B > θ_A) under independent Beta(1, 1) priors, computed by numerical
  integration over θ (not sampling), accurate to 1e-4.
