# PCA from the covariance eigendecomposition

Implement `pca(X, k)` returning `(components, explained_variance, X_proj)`.

- Center `X` (n, d) by its column means.
- `components`: (k, d), rows are unit principal directions, ordered by decreasing variance.
- `explained_variance`: (k,), the variance along each direction (use the `n - 1` denominator).
- `X_proj`: (n, k), the centered data projected onto the components.
Use an eigendecomposition or SVD you call yourself; don't use scikit-learn.
