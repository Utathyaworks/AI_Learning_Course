# Best decision-tree split by Gini impurity

Implement `best_split(x, y)` for one numeric feature `x` and integer class labels `y`.

Return `(threshold, weighted_gini)` for the split `x <= threshold` vs `x > threshold` that minimizes the
weighted Gini impurity of the two children. Thresholds are midpoints between consecutive distinct sorted values.
If all `x` are equal, return `(None, gini(y))`.
