import numpy as np
from submission import *  # noqa: F401,F403
def test_no_filter():
    l = np.array([1.0, 2.0, 3.0])
    p = sampling_probs(l)
    assert np.allclose(p, np.exp(l) / np.exp(l).sum())

def test_top_p_keeps_nucleus():
    l = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    p = sampling_probs(l, top_p=0.8)
    assert np.allclose(p, [0.5 / 0.8, 0.3 / 0.8, 0, 0])

def test_temperature_sharpens():
    l = np.array([1.0, 2.0])
    assert sampling_probs(l, temperature=0.1)[1] > sampling_probs(l, temperature=1.0)[1]

def test_tiny_top_p_is_greedy():
    p = sampling_probs(np.array([0.1, 3.0, 2.9]), top_p=1e-6)
    assert np.allclose(p, [0, 1, 0])
