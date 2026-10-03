# Sparse autoencoder: forward pass, loss and evaluation metrics

A sparse autoencoder (SAE) for activations X (n, d) with m latents:

f = ReLU((X − b_dec) W_enc + b_enc), X̂ = f W_dec + b_dec, with W_enc (d, m), W_dec (m, d).

Implement:
- `normalize_decoder(W_dec)`: rescale each latent's decoder row to unit L2 norm.
- `sae_forward(X, W_enc, b_enc, W_dec, b_dec)`: return `(f, X_hat)`.
- `sae_loss(X, f, X_hat, l1)`: mean over samples of ‖X − X̂‖² + l1 · ‖f‖₁.
- `topk_encode(pre, k)`: a top-k SAE's activation — keep the k largest pre-activations per row (after ReLU), zero
  the rest.
- `sae_metrics(X, f, X_hat)`: dict with `l0` (mean number of active latents per sample), `fve` (fraction of
  variance explained, 1 − Σ‖X − X̂‖² / Σ‖X − mean(X)‖²) and `dead` (fraction of latents never active).
