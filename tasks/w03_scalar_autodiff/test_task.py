import numpy as np
from submission import *  # noqa: F401,F403
import math

def test_simple_chain():
    a = Value(2.0); b = Value(-3.0)
    c = a * b + a ** 2
    c.backward()
    assert abs(c.data - (-6 + 4)) < 1e-12
    assert abs(a.grad - (b.data + 2 * a.data)) < 1e-12
    assert abs(b.grad - a.data) < 1e-12

def test_reuse_accumulates():
    x = Value(3.0)
    y = x * x + x
    y.backward()
    assert abs(x.grad - 7.0) < 1e-12

def test_numbers_on_both_sides():
    x = Value(2.0)
    y = 1 / x + 2 * x - 3 + (5 - x) / 4
    y.backward()
    assert abs(y.data - (0.5 + 4 - 3 + 0.75)) < 1e-12
    assert abs(x.grad - (-0.25 + 2 - 0.25)) < 1e-12

def test_tanh_exp_log_against_numeric():
    def f(v):
        return ((v * 0.7).tanh() + v.exp() * 0.1 - (v + 2).log()) * v
    x = Value(0.4); y = f(x); y.backward()
    def g(t):
        return (math.tanh(0.7 * t) + 0.1 * math.exp(t) - math.log(t + 2)) * t
    num = (g(0.4 + 1e-6) - g(0.4 - 1e-6)) / 2e-6
    assert abs(x.grad - num) < 1e-6
