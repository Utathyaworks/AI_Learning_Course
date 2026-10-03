# Exact Shapley values

Implement `shapley_values(value, n)` where `value` is a function taking a `frozenset` of player indices
(0..n−1) and returning a number. Compute exact Shapley values from the subset formula
`φ_i = Σ_{S ⊆ N∖{i}} |S|!(n−|S|−1)!/n! · (v(S∪{i}) − v(S))`. Return a NumPy array of length n.
(n will be at most 10.)
