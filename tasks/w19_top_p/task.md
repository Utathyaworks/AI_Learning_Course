# Temperature and nucleus (top-p) sampling filter

Implement `sampling_probs(logits, temperature=1.0, top_p=1.0)` for a 1-D logits vector.

1. Divide logits by `temperature` and apply a stable softmax.
2. Keep the smallest set of most-probable tokens whose cumulative probability is ≥ `top_p` (always keep at least
   the top token; break probability ties by lower index), zero out the rest, and renormalize.
Return the final probability vector.
