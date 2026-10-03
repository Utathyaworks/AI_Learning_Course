# Distances, hyperplanes and polygons

Implement loop-free NumPy geometry helpers.

- `pairwise_dist(X, Y)`: (N, M) Euclidean distances between rows of X (N, d) and Y (M, d), using
  ‖x−y‖² = ‖x‖² + ‖y‖² − 2x·y (clip tiny negatives to 0 before the square root).
- `hyperplane_signed_distance(X, w, b)`: signed distance of each row of X to the hyperplane w·x + b = 0.
- `point_line_distance(P, a, b)`: distance from each 2-D point in P (N, 2) to the **infinite** line through points a and b.
- `polygon_area(V)`: signed area of a polygon with vertices V (n, 2) in order (shoelace formula;
  counter-clockwise = positive).
- `convex_hull(P)`: the convex hull of 2-D points as an (h, 2) array in counter-clockwise order, starting from the
  lowest-x (then lowest-y) point, without collinear points on the edges (Andrew's monotone chain).
