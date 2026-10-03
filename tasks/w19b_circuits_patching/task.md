# Circuits in a 1-layer attention-only transformer, and activation patching

A tiny 1-layer, attention-only transformer with one head, written in NumPy:

`forward(tokens, W)` with `W = dict(E, P, WQ, WK, WV, WO, U)`:
x0 = E[tokens] + P[:n]; scores = (x0 WQ)(x0 WK)ᵀ/√d_head with a causal mask; A = softmax(scores);
head = A (x0 WV) WO; x1 = x0 + head; logits = x1 U. Return `(logits, cache)` where
`cache = {"x0": x0, "A": A, "head": head}`.

Implement:
- `forward(tokens, W, patch=None)`: if `patch` is a dict like `{"head": array}`, use that array as the head output
  instead of computing it (activation patching).
- `qk_circuit(W)` = E WQ (E WK)ᵀ (vocab × vocab, ignoring positions) and `ov_circuit(W)` = E WV WO U (vocab × vocab).
- `direct_logit_attribution(cache, W, pos, token)`: split the logit of `token` at position `pos` into the
  contribution of the embedding path (x0 U) and of the head (head U); return `(emb, head)` — they sum to the logit.
- `patching_effect(clean, corrupt, W, pos, token)`: run the corrupted input with the head output patched in from
  the clean run; return the fraction of the clean-vs-corrupt logit difference for `token` at `pos` that patching
  restores: (patched − corrupt) / (clean − corrupt).
