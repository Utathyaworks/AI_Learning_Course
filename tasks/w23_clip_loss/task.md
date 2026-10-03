# CLIP contrastive loss with a learnable temperature

Implement `clip_loss(img, txt, logit_scale)` for paired embeddings `img`, `txt` (N, D).

L2-normalize both, form `logits = exp(logit_scale) · img @ txt.T`, and return
`(loss, d_logit_scale)`: loss = mean of image→text and text→image cross-entropies with the diagonal as targets;
`d_logit_scale` is the derivative of the loss with respect to `logit_scale`.
