# Short-time Fourier transform magnitude

Implement `stft_mag(x, n_fft, hop)`:

- Split `x` into frames of length `n_fft` starting at 0, hop, 2·hop, … (no padding; drop the incomplete tail).
- Multiply each frame by a periodic Hann window `w[n] = 0.5 − 0.5·cos(2πn / n_fft)`.
- Return the magnitude of the one-sided spectrum, shape `(n_fft // 2 + 1, n_frames)`.
You may use `np.fft.rfft` here.
