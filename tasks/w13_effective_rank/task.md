# Effective rank of an embedding matrix

Implement `effective_rank(Z)` (Roy & Vetterli): center `Z` (n, d) by column means, take its singular
values `s`, form `p = s / s.sum()`, and return `exp(H(p))` with `H(p) = −Σ p log p` (treat 0·log 0 as 0).
A collapsed representation has effective rank near 1.
