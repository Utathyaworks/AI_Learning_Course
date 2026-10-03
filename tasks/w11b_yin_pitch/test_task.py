import numpy as np
from submission import *  # noqa: F401,F403
SR = 16000

def _tone(f, n=2048, harmonics=(1.0,)):
    t = np.arange(n) / SR
    return sum(a * np.sin(2 * np.pi * f * (k + 1) * t) for k, a in enumerate(harmonics))

def test_pure_tones():
    for f in (110.0, 220.0, 440.0):
        assert abs(yin_pitch(_tone(f), SR) - f) / f < 0.01

def test_no_octave_error_with_harmonics():
    x = _tone(150.0, harmonics=(0.4, 1.0, 0.8, 0.5))   # strong 2nd harmonic
    assert abs(yin_pitch(x, SR) - 150.0) / 150.0 < 0.02

def test_silence():
    assert yin_pitch(np.zeros(2048), SR) == 0.0
