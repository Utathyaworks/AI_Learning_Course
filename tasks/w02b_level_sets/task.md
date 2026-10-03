# Surfaces and level sets: grids, critical points and quadratic shapes

Two-variable functions without calculus.

- `grid_eval(f, xs, ys)`: evaluate f(x, y) on the grid; return Z of shape (len(ys), len(xs)) with Z[i, j] = f(xs[j], ys[i])
  (the `np.meshgrid` default convention).
- `grid_argmin(Z, xs, ys)`: the (x, y) location of the smallest value in Z.
- `quadratic_shape(A)`: for a symmetric 2×2 matrix, classify the surface z = [x y] A [x y]ᵀ as `"bowl"` (both
  eigenvalues > 0), `"cap"` (both < 0), `"saddle"` (opposite signs) or `"trough"` (one eigenvalue ≈ 0, |λ| < 1e-12).
- `level_set_ellipse_axes(A, c)`: for a positive-definite A, the contour xᵀAx = c is an ellipse; return its semi-axis
  lengths sorted from long to short (√(c/λ) for each eigenvalue λ).
- `contour_crossings(z_line, level)`: indices i where the sampled values z_line cross `level` between i and i+1
  (strict sign change of z − level).
