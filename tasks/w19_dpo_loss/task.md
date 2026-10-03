# DPO loss and implicit reward accuracy

Implement `dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta)` for batches of sequence
log-probabilities (1-D arrays).

With implicit rewards `r_c = β(π_c − ref_c)` and `r_r = β(π_r − ref_r)`, the loss is
`mean(−log σ(r_c − r_r))`, computed stably. Return `(loss, reward_accuracy, grad_chosen, grad_rejected)` where
`reward_accuracy` is the fraction with `r_c > r_r`, and the gradients are w.r.t. `pi_chosen` and `pi_rejected`.
