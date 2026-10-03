import numpy as np
from submission import *  # noqa: F401,F403
def test_shape():
    S = stft_mag(np.zeros(1000), 256, 128)
    assert S.shape == (129, 1 + (1000 - 256) // 128)

def test_pure_tone_peak():
    sr, f = 8000, 1000
    t = np.arange(sr) / sr
    S = stft_mag(np.sin(2 * np.pi * f * t), 512, 256)
    peak_bin = np.argmax(S.mean(1))
    assert abs(peak_bin * sr / 512 - f) <= sr / 512

def test_first_frame_value():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(64) / 64)
    assert np.allclose(stft_mag(x, 64, 32)[:, 0], np.abs(np.fft.rfft(x[:64] * w)))
