import numpy as np
from submission import *  # noqa: F401,F403
import pytest

def _brute(y, s):
    pos = [b for a, b in zip(y, s) if a]; neg = [b for a, b in zip(y, s) if not a]
    tot = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return tot / (len(pos) * len(neg))

def test_perfect_and_reversed():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0

def test_ties():
    assert abs(roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-12

def test_random_vs_brute():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 30); y[0], y[1] = 0, 1
        s = rng.integers(0, 6, 30).astype(float)
        assert abs(roc_auc(y, s) - _brute(y, s)) < 1e-12

def test_one_class():
    with pytest.raises(ValueError):
        roc_auc([1, 1], [0.2, 0.4])
