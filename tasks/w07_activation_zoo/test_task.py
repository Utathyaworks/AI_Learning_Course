import numpy as np
from submission import *  # noqa: F401,F403
import warnings

NAMES = ["sigmoid", "relu", "leaky_relu", "elu", "selu", "gelu", "silu", "softplus"]
XS = np.array([-4.0, -1.3, -0.2, 0.3, 0.9, 2.5, 5.0])

def test_gradients_by_finite_differences():
    g = globals(); h = 1e-6
    for n in NAMES:
        f, df = g[n], g[n + "_grad"]
        num = (f(XS + h) - f(XS - h)) / (2 * h)
        assert np.allclose(df(XS), num, atol=1e-5), n

def test_known_values():
    assert np.allclose(sigmoid(0.0), 0.5) and np.allclose(softplus(0.0), np.log(2))
    assert np.allclose(relu([-1.0, 2.0]), [0, 2]) and np.allclose(relu_grad([-1.0, 0.0, 2.0]), [0, 0, 1])
    assert np.allclose(leaky_relu([-2.0, 3.0], 0.1), [-0.2, 3.0])
    assert np.allclose(elu(-1e6), -1.0) and np.allclose(selu(-1e6), -1.0507009873554805 * 1.6732632423543772)
    assert np.allclose(gelu(1.0), 0.8413447460685429) and np.allclose(silu(1.0), 0.7310585786300049)

def test_gelu_tanh_is_close():
    x = np.linspace(-6, 6, 1001)
    assert np.max(np.abs(gelu_tanh(x) - gelu(x))) < 1e-3

def test_selu_self_normalizes():
    rng = np.random.default_rng(0)
    z = rng.normal(size=2_000_000)
    y = selu(z)
    assert abs(y.mean()) < 5e-3 and abs(y.var() - 1) < 1e-2

def test_stability():
    big = np.array([-1000.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s, sp, ls = sigmoid(big), softplus(big), log_sigmoid(big)
    assert np.allclose(s, [0, 1]) and np.allclose(sp, [0, 1000]) and np.allclose(ls, [-1000, 0])
    assert np.isclose(log_sigmoid(-30.0), -30.0 - np.log1p(np.exp(-30.0)))

def test_swiglu():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 6)); W, V = rng.normal(size=(6, 8)), rng.normal(size=(6, 8)); W2 = rng.normal(size=(8, 6))
    a = x @ W
    ref = ((a / (1 + np.exp(-a))) * (x @ V)) @ W2
    assert swiglu(x, W, V, W2).shape == (4, 6) and np.allclose(swiglu(x, W, V, W2), ref)
