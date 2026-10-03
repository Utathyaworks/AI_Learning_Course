# Stable softmax cross-entropy and its gradient

Implement `softmax_cross_entropy(logits, y)` for logits of shape (N, C) and integer labels `y` (N,).

Return `(loss, dlogits)`: the **mean** cross-entropy over the batch and its gradient with respect to the logits.
It must stay finite for logits as large as ±1e4 (use the log-sum-exp trick).
