# CTC loss by the forward algorithm

Implement `ctc_loss(log_probs, target, blank=0)`.

- `log_probs` (T, C): per-frame log-probabilities (already log-softmaxed). `target`: list of label ids (no blanks).
- Return the negative log-likelihood `−log p(target | x)`, summing over all alignments that collapse to the target
  (merge repeats, then remove blanks), computed with the forward (α) recursion **in log space**.
- Return `inf` if no alignment is possible (target too long for T).
