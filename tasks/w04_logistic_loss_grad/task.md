# Stable logistic loss and gradient

Implement `logistic_loss_grad(X, y, w, b, l2=0.0)` for binary labels `y ∈ {0, 1}`.

Return `(loss, dw, db)` where `loss` is the **mean** negative log-likelihood plus `0.5 * l2 * ||w||^2`.
It must be numerically stable for very large |logits| (no overflow warnings, no `inf`/`nan`).
