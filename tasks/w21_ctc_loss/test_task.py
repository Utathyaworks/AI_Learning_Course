import numpy as np
from submission import *  # noqa: F401,F403
import itertools

def _collapse(path, blank=0):
    out, prev = [], None
    for p in path:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return out

def _brute(lp, target):
    T, C = lp.shape
    tot = -np.inf
    for path in itertools.product(range(C), repeat=T):
        if _collapse(path) == list(target):
            tot = np.logaddexp(tot, sum(lp[t, p] for t, p in enumerate(path)))
    return -tot

def _rand_lp(rng, T, C):
    z = rng.normal(size=(T, C))
    return z - np.log(np.exp(z).sum(1, keepdims=True))

def test_against_brute_force():
    rng = np.random.default_rng(0)
    for target in [[1], [1, 2], [1, 1], [2, 1, 2]]:
        lp = _rand_lp(rng, 5, 3)
        assert np.isclose(ctc_loss(lp, target), _brute(lp, target))

def test_impossible():
    rng = np.random.default_rng(1)
    assert ctc_loss(_rand_lp(rng, 2, 3), [1, 1, 1]) == float("inf")

def test_long_sequence_finite():
    rng = np.random.default_rng(2)
    assert np.isfinite(ctc_loss(_rand_lp(rng, 400, 20), list(rng.integers(1, 20, 50))))
