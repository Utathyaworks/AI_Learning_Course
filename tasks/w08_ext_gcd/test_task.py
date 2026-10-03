import numpy as np
from submission import *  # noqa: F401,F403
import math, random
import pytest

def test_bezout_random():
    rnd = random.Random(0)
    for _ in range(500):
        a, b = rnd.randint(-10**6, 10**6), rnd.randint(-10**6, 10**6)
        g, x, y = ext_gcd(a, b)
        assert g == math.gcd(a, b) and a * x + b * y == g

def test_inverse():
    for m in [7, 26, 97, 1000]:
        for a in range(1, m):
            if math.gcd(a, m) == 1:
                assert (a * mod_inverse(a, m)) % m == 1
    with pytest.raises(ValueError):
        mod_inverse(6, 9)

def test_crt():
    assert crt([2, 3, 2], [3, 5, 7]) == 23
    rnd = random.Random(1)
    for _ in range(100):
        x = rnd.randint(0, 3 * 5 * 7 * 11 - 1)
        assert crt([x % 3, x % 5, x % 7, x % 11], [3, 5, 7, 11]) == x
