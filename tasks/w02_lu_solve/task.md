# Solve Ax = b with LU and partial pivoting

Implement `lu_solve(A, b)`: factor `PA = LU` with partial pivoting, then solve by forward and back
substitution. `A` is (n, n) and invertible; `b` is (n,) or (n, k). Return `x` with the same trailing shape as `b`.
Do not call `np.linalg.solve`, `np.linalg.inv` or SciPy.
