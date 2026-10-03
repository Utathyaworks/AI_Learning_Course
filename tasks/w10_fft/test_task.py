import numpy as np
from submission import *  # noqa: F401,F403
def test_matches_numpy():
    rng = np.random.default_rng(0)
    for n in [1, 2, 8, 64, 256]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        assert np.allclose(fft(x), np.fft.fft(x))

def test_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.normal(size=128)
    assert np.allclose(ifft(fft(x)), x)

def test_convolution_theorem():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=32), rng.normal(size=32)
    circ = np.array([sum(a[k] * b[(n - k) % 32] for k in range(32)) for n in range(32)])
    assert np.allclose(ifft(fft(a) * fft(b)).real, circ)
