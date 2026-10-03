# FixMatch pseudo-labels and unlabeled loss

Implement the unlabeled part of FixMatch.

`fixmatch_unlabeled_loss(probs_weak, logits_strong, threshold=0.95)`:
- `probs_weak` (N, C): predicted probabilities on weakly augmented images (treat as constants).
- Pseudo-label = argmax of `probs_weak`; mask = max probability ≥ threshold.
- Loss = cross-entropy of `logits_strong` against the pseudo-labels, **summed over masked examples and divided
  by N** (all unlabeled examples, not just the confident ones).
Return `(loss, mask, pseudo_labels)`. Use a numerically stable log-softmax.
