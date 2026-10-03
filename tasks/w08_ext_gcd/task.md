# Extended Euclid, modular inverse and CRT

Implement without `pow(a, -1, m)` or any library number theory:

- `ext_gcd(a, b)` → `(g, x, y)` with `g = gcd(a, b) ≥ 0` and `a*x + b*y = g`.
- `mod_inverse(a, m)` → the inverse of `a` modulo `m` in `[0, m)`; raise `ValueError` if it doesn't exist.
- `crt(residues, moduli)` → the smallest non-negative `x` with `x ≡ r_i (mod m_i)` for pairwise-coprime moduli.
