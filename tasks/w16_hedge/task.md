# Hedge (multiplicative weights) and its regret

Implement `hedge(losses, eta)` for `losses` of shape (T, n) with entries in [0, 1].

Start with uniform weights. At round t, play the distribution `p_t ∝ w_t`, suffer expected loss `p_t · ℓ_t`, then
update `w ← w · exp(−eta · ℓ_t)`. Return `(P, regret)`: `P` (T, n) the distributions played and
`regret = Σ_t p_t·ℓ_t − min_i Σ_t ℓ_t[i]`. Keep it numerically stable for long horizons (normalize each round).
