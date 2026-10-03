# One k-means (Lloyd) iteration

Implement `kmeans_step(X, centers)` returning `(labels, new_centers, inertia)`.

- `labels[i]`: index of the nearest center (squared Euclidean; ties go to the lower index).
- `new_centers[j]`: mean of points assigned to j; if a cluster is empty, keep its old center.
- `inertia`: sum of squared distances of points to their **old** assigned center.
Vectorize the distance computation (no Python loop over points).
