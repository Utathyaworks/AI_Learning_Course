# BatchNorm forward and backward (training mode)

Implement `batchnorm_forward_backward(x, gamma, beta, dout, eps=1e-5)` for `x` of shape (N, D).

Forward (training mode): normalize each feature with the **batch** mean and biased variance, then scale and shift.
Return `(out, dx, dgamma, dbeta)` where `dout` is the upstream gradient. Derive `dx` yourself.
