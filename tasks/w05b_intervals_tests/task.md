# Wilson interval, McNemar and the paired t-test

Implement statistics for comparing models (you may use `scipy.stats` distributions, not its tests).

- `wilson_interval(k, n, conf=0.95)`: Wilson score interval for a proportion k/n.
- `mcnemar_exact(a_correct, b_correct)`: two boolean arrays (same test examples). With b = #(A right, B wrong) and
  c = #(A wrong, B right), return the exact two-sided p-value `min(1, 2·P(Bin(b+c, 0.5) ≤ min(b, c)))`
  (1.0 when b + c = 0).
- `paired_t(x, y)`: return `(t, p)` for the two-sided paired t-test of mean(x − y) = 0.
- `cramer_rao_bernoulli(p, n)`: the Cramér–Rao lower bound on the variance of an unbiased estimator of p from n
  Bernoulli trials.
