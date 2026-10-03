import numpy as np

def fft(x):
    x = np.asarray(x, dtype=complex)
    n = len(x)
    if n == 1:
        return x.copy()
    even, odd = fft(x[0::2]), fft(x[1::2])
    tw = np.exp(-2j * np.pi * np.arange(n // 2) / n) * odd
    return np.concatenate([even + tw, even - tw])

def ifft(X):
    X = np.asarray(X, dtype=complex)
    return np.conj(fft(np.conj(X))) / len(X)
