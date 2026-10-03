# Label propagation: the harmonic solution

Implement `label_propagation(W, labeled_idx, Y_l)`.

- `W`: symmetric (n, n) non-negative affinity matrix; `labeled_idx`: indices of labelled nodes; `Y_l`: one-hot
  labels (len(labeled_idx), C).
- Solve the harmonic problem: unlabeled scores `F_u = (D_uu − W_uu)^{-1} W_ul Y_l` with `D` the degree matrix.
- Return `F` of shape (n, C): labelled rows equal `Y_l`, unlabeled rows `F_u`. Rows sum to 1 when every
  unlabeled node is connected to a labelled one.
