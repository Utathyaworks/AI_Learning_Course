import numpy as np
from submission import *  # noqa: F401,F403
import math

def test_warmup():
    assert abs(lr_at(0, 100, 10, 1.0) - 0.1) < 1e-12
    assert abs(lr_at(9, 100, 10, 1.0) - 1.0) < 1e-12

def test_peak_and_end():
    assert abs(lr_at(10, 100, 10, 1.0, 0.1) - 1.0) < 1e-12
    assert abs(lr_at(99, 100, 10, 1.0, 0.1) - 0.1) < 1e-12
    assert lr_at(150, 100, 10, 1.0, 0.1) == 0.1

def test_midpoint_and_monotone():
    mid = 10 + (99 - 10) / 2
    assert abs(lr_at(int(mid), 100, 10, 2.0) - (1 + math.cos(math.pi * (int(mid) - 10) / 89))) < 1e-12
    vals = [lr_at(s, 100, 10, 1.0) for s in range(10, 100)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))
