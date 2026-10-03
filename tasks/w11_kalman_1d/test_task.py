import numpy as np
from submission import *  # noqa: F401,F403
def test_static_signal_equals_running_mean():
    # q = 0: the filter is a precision-weighted running average of prior and measurements.
    rng = np.random.default_rng(0)
    z = 5 + rng.normal(size=50)
    p0, x0 = 1e3, 0.0
    n = np.arange(1, 51)
    xs, ps = kalman_1d(z, 0.0, 1.0, x0, p0)
    assert np.allclose(xs, (x0 / p0 + np.cumsum(z)) / (1 / p0 + n), atol=1e-9)
    assert np.allclose(ps, 1 / (1 / p0 + n), atol=1e-9)

def test_trusts_precise_sensor():
    xs, _ = kalman_1d([3.0, -2.0], 1.0, 1e-12, 0.0, 1.0)
    assert np.allclose(xs, [3.0, -2.0], atol=1e-6)

def test_variance_converges():
    _, ps = kalman_1d(np.zeros(200), 0.1, 1.0, 0.0, 10.0)
    q, r = 0.1, 1.0
    p_pred = (q + np.sqrt(q * q + 4 * q * r)) / 2   # steady-state predicted variance
    assert abs(ps[-1] - p_pred * r / (p_pred + r)) < 1e-6
