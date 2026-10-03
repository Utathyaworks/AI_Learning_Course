import numpy as np

def stft_mag(x, n_fft, hop):
    x = np.asarray(x, float)
    n_frames = 1 + (len(x) - n_fft) // hop
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * w
    return np.abs(np.fft.rfft(frames, axis=1)).T
