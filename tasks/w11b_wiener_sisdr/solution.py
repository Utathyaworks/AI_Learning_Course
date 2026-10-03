import numpy as np

def wiener_gain(noisy_power, noise_power, floor=0.0):
    P = np.asarray(noisy_power, float); N = np.asarray(noise_power, float)
    xi = np.maximum(P / N - 1.0, 0.0)
    return np.maximum(xi / (1.0 + xi), floor)

def snr_db(clean, estimate):
    c = np.asarray(clean, float); e = c - np.asarray(estimate, float)
    return float(10 * np.log10(np.sum(c ** 2) / np.sum(e ** 2)))

def si_sdr(reference, estimate):
    s = np.asarray(reference, float); s = s - s.mean()
    h = np.asarray(estimate, float); h = h - h.mean()
    target = (h @ s) / (s @ s) * s
    e = h - target
    return float(10 * np.log10((target @ target) / (e @ e)))
