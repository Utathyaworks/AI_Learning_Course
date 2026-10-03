# Uncertainty-weighted sensor fusion

Several sensors estimate the same quantity with Gaussian noise. Implement
`fuse(means, variances, available=None)`:

- `means`, `variances`: arrays of shape (M,) or (M, N) (M sensors, N independent estimates).
- `available`: optional boolean mask of the same shape; missing sensors are ignored.
- Return `(fused_mean, fused_variance)` using inverse-variance weighting: weights `1/σ²`, fused variance
  `1 / Σ (1/σ²)`. If no sensor is available for an estimate, return `nan` mean and `inf` variance there.
