# Rotations, quaternions and the pinhole camera

Implement:

- `rodrigues(axis, angle)`: 3×3 rotation matrix for a rotation by `angle` (radians, right-hand rule) about `axis`
  (normalize it first).
- `quat_to_matrix(q)`: rotation matrix of a quaternion `q = (w, x, y, z)` (normalize it first).
- `quat_multiply(p, q)`: Hamilton product, so that `quat_to_matrix(quat_multiply(p, q)) == R(p) @ R(q)`.
- `homogeneous(R, t)`: the 4×4 matrix `[[R, t], [0, 1]]`.
- `project(X, K, R, t)`: pinhole projection of world points X (N, 3) to pixels (N, 2): `x ~ K (R X + t)`, divide by depth.
