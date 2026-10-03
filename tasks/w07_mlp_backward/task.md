# Two-layer MLP: forward and backward

Implement `mlp_loss_grads(X, y, W1, b1, W2, b2)` for the network
`h = relu(X @ W1 + b1)`, `logits = h @ W2 + b2`, with mean softmax cross-entropy loss.

Return `(loss, grads)` where `grads` is a dict with keys `"W1", "b1", "W2", "b2"` matching the parameter shapes.
Derive the backward pass yourself (no autodiff library).
