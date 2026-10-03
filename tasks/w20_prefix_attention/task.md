# Prefix tuning as extra keys and values

Prefix tuning prepends learned key/value vectors to every attention layer.

Implement `prefix_attention(Q, K, V, P_k, P_v)` for single-head attention: `Q` (Lq, d), `K` and `V` (L, d),
prefixes `P_k` and `P_v` (p, d). Attention runs over the concatenated keys `[P_k; K]` and values `[P_v; V]`
(no mask). Return `(out, prefix_weight)` where `prefix_weight` (Lq,) is the total attention mass each query puts
on the prefix positions.
