import numpy as np
from itertools import combinations
from math import factorial

def shapley_values(value, n):
    phi = np.zeros(n)
    cache = {}
    def v(S):
        if S not in cache:
            cache[S] = value(S)
        return cache[S]
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(others, k):
                S = frozenset(S)
                phi[i] += w * (v(S | {i}) - v(S))
    return phi
