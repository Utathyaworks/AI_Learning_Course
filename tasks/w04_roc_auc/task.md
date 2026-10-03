# ROC AUC from ranks

Implement `roc_auc(y_true, scores)` for binary labels without scikit-learn.

Use the rank (Mann–Whitney) formulation: AUC is the probability that a random positive scores higher than a random
negative, counting ties as one half. Handle ties correctly. Raise `ValueError` if only one class is present.
