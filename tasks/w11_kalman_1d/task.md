# 1-D Kalman filter (random-walk model)

Implement `kalman_1d(zs, q, r, x0, p0)` for the model `x_t = x_{t−1} + w`, `w ~ N(0, q)`, and
measurement `z_t = x_t + v`, `v ~ N(0, r)`.

For each measurement: predict (`p ← p + q`), then update with gain `k = p / (p + r)`.
Return `(xs, ps)`: arrays of the posterior mean and variance after each measurement.
