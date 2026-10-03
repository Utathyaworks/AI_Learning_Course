# Wiener gain, SNR and SI-SDR

Implement three audio-enhancement building blocks.

- `wiener_gain(noisy_power, noise_power, floor=0.0)`: per-bin gain using the maximum-likelihood a-priori SNR
  `ξ = max(P_noisy / P_noise − 1, 0)` and the Wiener rule `ξ / (1 + ξ)`, then clipped below at `floor`.
- `snr_db(clean, estimate)`: `10·log10(‖clean‖² / ‖clean − estimate‖²)`.
- `si_sdr(reference, estimate)`: scale-invariant SDR in dB — project the (zero-mean) estimate onto the (zero-mean)
  reference, `s_target = (⟨ŝ, s⟩ / ‖s‖²)·s`, `e = ŝ − s_target`, return `10·log10(‖s_target‖² / ‖e‖²)`.
