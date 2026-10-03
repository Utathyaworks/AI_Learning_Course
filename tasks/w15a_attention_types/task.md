# Attention types: additive, multi-head with masks, sliding window, cross and latent

Implement in NumPy (no batch dimension):

- `additive_attention(q, K, V, Wq, Wk, v)`: Bahdanau scores e_j = vᵀ tanh(Wq q + Wk k_j) for a single query q (d,),
  keys K (m, d); return `(context, weights)`.
- `make_mask(n, kind, window=None, prefix=None)`: boolean (n, n), True = may attend. `kind` is `"full"`,
  `"causal"`, `"window"` (causal and |i − j| < window) or `"prefix"` (positions < prefix see each other freely;
  later positions are causal).
- `multihead_attention(X, Wq, Wk, Wv, Wo, h, mask=None, Y=None)`: self-attention on X (n, d); if `Y` is given,
  cross-attention where queries come from X and keys/values from Y. Split d into h heads, scaled dot-product per
  head, concatenate, project with Wo. Masked positions get −∞ before the softmax.
- `latent_cross_attention(L, X, Wq, Wk, Wv, Wo, h)`: Perceiver-style — a small set of latents L (k, d) attends to a
  long input X (n, d); output (k, d). (It's cross-attention with queries from L.)
