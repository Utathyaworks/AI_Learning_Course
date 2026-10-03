# Scaled dot-product attention with masking

Implement `attention(Q, K, V, mask=None)` for batched inputs `Q` (..., Lq, d), `K` (..., Lk, d),
`V` (..., Lk, dv). Scores are `Q Kᵀ / √d`. `mask` is boolean, broadcastable to (..., Lq, Lk), with **True meaning
"may attend"**; masked-out scores get −∞ before the softmax. Return `(out, weights)`.
Also implement `causal_mask(L)` returning an (L, L) boolean lower-triangular mask.
