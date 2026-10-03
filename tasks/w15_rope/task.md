# Rotary position embedding (RoPE)

Implement `apply_rope(x, positions, base=10000.0)` for `x` of shape (L, d) with even d.

Pair dimensions as (0,1), (2,3), …; pair k uses frequency `θ_k = base^(−2k/d)` and is rotated by angle
`positions[t] · θ_k`: `(x0, x1) → (x0 cos − x1 sin, x0 sin + x1 cos)`. Return the rotated array.
A correct implementation makes `apply_rope(q, [m]) · apply_rope(k, [n])` depend only on `m − n`.
