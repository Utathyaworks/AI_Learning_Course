# Adam and AdamW update

Implement one optimizer step: `adam_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0, decoupled=True)`.

Return `(w_new, m_new, v_new)`. `t` is the step number starting at 1; apply bias correction.
- `decoupled=True` is **AdamW**: `w ← w − lr·weight_decay·w` applied separately from the Adam update.
- `decoupled=False` is **Adam + L2**: add `weight_decay·w` to the gradient before the moments.
