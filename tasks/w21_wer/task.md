# Word error rate by edit distance

Implement `wer(reference, hypothesis)` for whitespace-tokenized strings.

WER = (substitutions + deletions + insertions) / (number of reference words), using the minimum-edit-distance
dynamic program over words. Return a float. If the reference is empty, return 0.0 when the hypothesis is also
empty and `inf` otherwise.
