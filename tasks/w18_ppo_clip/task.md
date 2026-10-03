# PPO clipped surrogate loss and gradient

Implement `ppo_clip_loss(logp_new, logp_old, adv, eps=0.2)`.

With ratio `ρ = exp(logp_new − logp_old)`, the loss to **minimize** is
`−mean( min(ρ·A, clip(ρ, 1−eps, 1+eps)·A) )`. Return `(loss, dlogp_new, clip_fraction)` where
`dlogp_new` is the gradient of the loss w.r.t. `logp_new` and `clip_fraction` is the fraction of samples where
the clipped term is the one selected **and** differs from the unclipped one.
