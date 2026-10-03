# YIN pitch estimation

Implement `yin_pitch(frame, sr, fmin=60.0, fmax=800.0, threshold=0.1)` returning the fundamental frequency in
Hz, or `0.0` for silence/unvoiced frames.

1. Difference function `d(τ) = Σ_j (x[j] − x[j+τ])²` over a fixed window `j = 0 … W−1` with `W = len(frame) − τ_max`.
2. Cumulative mean normalized difference `d′(0) = 1`, `d′(τ) = d(τ) · τ / Σ_{k=1..τ} d(k)`.
3. Search `τ ∈ [sr/fmax, sr/fmin]`: take the first τ with `d′(τ) < threshold`, then walk to the local minimum.
   If none falls below the threshold, use the global minimum only if `d′ < 0.5`, else return 0.
4. Refine τ with parabolic interpolation on `d′`, return `sr / τ`. Return 0 if the frame energy is ~0.
