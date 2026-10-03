import numpy as np

def wiener_gain(noisy_power, noise_power, floor=0.0):
    raise NotImplementedError

def snr_db(clean, estimate) -> float:
    raise NotImplementedError

def si_sdr(reference, estimate) -> float:
    raise NotImplementedError
