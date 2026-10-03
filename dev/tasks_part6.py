"""Coding tasks, part 6 (activation functions, Week 7)."""

T = []

T.append(dict(
id="w07_activation_zoo", title="Activation zoo: forward, gradient and numerical stability",
prompt="""Implement these element-wise activations on NumPy arrays, each with its derivative `<name>_grad(x)`:

`sigmoid`, `relu` (use 0 as the gradient at 0), `leaky_relu(x, a=0.01)`, `elu(x, a=1.0)`,
`selu` (λ = 1.0507009873554805, α = 1.6732632423543772), `gelu` (exact, x·Φ(x)),
`silu` (x·σ(x)), `softplus` (log(1 + eˣ)).

Also implement:
- `gelu_tanh(x)`: the tanh approximation 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
- `log_sigmoid(x)`: log σ(x).
- `swiglu(x, W, V, W2)`: the SwiGLU MLP `(silu(x @ W) * (x @ V)) @ W2`.

`sigmoid`, `softplus` and `log_sigmoid` must be finite and accurate at |x| = 1000 (no overflow warnings turned
into NaN or inf).""",
starter='''import numpy as np

def sigmoid(x): raise NotImplementedError
def sigmoid_grad(x): raise NotImplementedError
def relu(x): raise NotImplementedError
def relu_grad(x): raise NotImplementedError
def leaky_relu(x, a=0.01): raise NotImplementedError
def leaky_relu_grad(x, a=0.01): raise NotImplementedError
def elu(x, a=1.0): raise NotImplementedError
def elu_grad(x, a=1.0): raise NotImplementedError
def selu(x): raise NotImplementedError
def selu_grad(x): raise NotImplementedError
def gelu(x): raise NotImplementedError
def gelu_grad(x): raise NotImplementedError
def gelu_tanh(x): raise NotImplementedError
def silu(x): raise NotImplementedError
def silu_grad(x): raise NotImplementedError
def softplus(x): raise NotImplementedError
def softplus_grad(x): raise NotImplementedError
def log_sigmoid(x): raise NotImplementedError
def swiglu(x, W, V, W2): raise NotImplementedError
''',
solution='''import numpy as np
from scipy.special import erf

LAM, ALPHA = 1.0507009873554805, 1.6732632423543772

def sigmoid(x):
    x = np.asarray(x, float)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))

def sigmoid_grad(x):
    s = sigmoid(x); return s * (1 - s)

def relu(x): return np.maximum(np.asarray(x, float), 0.0)
def relu_grad(x): return (np.asarray(x, float) > 0).astype(float)

def leaky_relu(x, a=0.01):
    x = np.asarray(x, float); return np.where(x > 0, x, a * x)

def leaky_relu_grad(x, a=0.01):
    return np.where(np.asarray(x, float) > 0, 1.0, a)

def elu(x, a=1.0):
    x = np.asarray(x, float); return np.where(x > 0, x, a * np.expm1(np.minimum(x, 0)))

def elu_grad(x, a=1.0):
    x = np.asarray(x, float); return np.where(x > 0, 1.0, a * np.exp(np.minimum(x, 0)))

def selu(x): return LAM * elu(x, ALPHA)
def selu_grad(x): return LAM * elu_grad(x, ALPHA)

def _Phi(x): return 0.5 * (1 + erf(np.asarray(x, float) / np.sqrt(2)))
def _phi(x): x = np.asarray(x, float); return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

def gelu(x): x = np.asarray(x, float); return x * _Phi(x)
def gelu_grad(x): x = np.asarray(x, float); return _Phi(x) + x * _phi(x)

def gelu_tanh(x):
    x = np.asarray(x, float)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def silu(x): x = np.asarray(x, float); return x * sigmoid(x)
def silu_grad(x):
    s = sigmoid(x); return s * (1 + np.asarray(x, float) * (1 - s))

def softplus(x):
    x = np.asarray(x, float); return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))

def softplus_grad(x): return sigmoid(x)

def log_sigmoid(x): return -softplus(-np.asarray(x, float))

def swiglu(x, W, V, W2):
    return (silu(x @ W) * (x @ V)) @ W2
''',
tests='''
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
'''))
