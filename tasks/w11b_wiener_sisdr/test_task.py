import numpy as np
from submission import *  # noqa: F401,F403
def test_gain_properties():
    P = np.array([1.0, 2.0, 11.0, 0.5]); N = np.ones(4)
    g = wiener_gain(P, N)
    assert np.allclose(g, [0.0, 0.5, 10 / 11, 0.0])
    assert np.allclose(wiener_gain(P, N, floor=0.1), np.maximum(g, 0.1))

def test_gain_equals_power_subtraction():
    rng = np.random.default_rng(0)
    N = rng.random(50) + 0.1; P = N * (1 + 5 * rng.random(50))
    assert np.allclose(wiener_gain(P, N), 1 - N / P)

def test_snr_known():
    rng = np.random.default_rng(1)
    s = rng.normal(size=8000); n = rng.normal(size=8000)
    n = n / np.linalg.norm(n) * np.linalg.norm(s) / 10    # 20 dB
    assert np.isclose(snr_db(s, s + n), 20.0)

def test_si_sdr_scale_invariant():
    rng = np.random.default_rng(2)
    s = rng.normal(size=4000); n = rng.normal(size=4000) * 0.1
    a = si_sdr(s, s + n)
    assert np.isclose(a, si_sdr(s, 3.7 * (s + n)))
    assert si_sdr(s, s + 0.01 * n) > a
