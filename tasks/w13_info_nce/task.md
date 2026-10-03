# Symmetric InfoNCE (NT-Xent style) loss

Implement `info_nce(z1, z2, temperature=0.1)` for two batches of embeddings (N, D) where row i of `z1` and
row i of `z2` are a positive pair.

- L2-normalize rows, compute similarities `S = z1 @ z2.T / temperature`.
- Loss = mean of the cross-entropy over rows (target = diagonal) and over columns, i.e.
  `0.5 * (CE(S, diag) + CE(S.T, diag))`.
Use a stable log-softmax.
