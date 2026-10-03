# Gram–Schmidt orthonormalization

Implement classical Gram–Schmidt (with re-orthogonalization optional) for the columns of `A`.

- `A` has shape (m, n) with linearly independent columns, m ≥ n.
- Return `Q` of shape (m, n) with orthonormal columns spanning the same space, column by column,
  so that `A = Q R` for some upper-triangular `R` with positive diagonal.
- Do not call `np.linalg.qr`.
