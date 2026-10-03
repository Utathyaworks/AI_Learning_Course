# Value iteration on a tabular MDP

Implement `value_iteration(P, R, gamma, tol=1e-10)`.

- `P` (S, A, S): transition probabilities; `R` (S, A): expected immediate reward.
- Iterate the Bellman optimality operator until the max change is below `tol`.
Return `(V, policy)` where `policy[s]` is a greedy action (lowest index on ties).
