# Online-softmax (tiled) attention, KV cache and GQA

Single-head shapes: Q (n, d), K (m, d), V (m, dv).

- `attention(Q, K, V, causal=False)`: softmax(QKᵀ/√d)V; with `causal=True`, query i may attend to keys 0…i
  (assume n == m).
- `tiled_attention(Q, K, V, block=16)`: the same result, computed over key/value blocks with the **online softmax**
  (keep a running max m, running sum l and running output o per query; rescale when the max grows). Never build the
  full n × m score matrix — work one key block at a time.
- `decode_with_cache(Q, K, V)`: generate causally one position at a time, appending each step's k, v to a cache
  and attending only over the cache; return the (n, dv) outputs (must equal causal attention).
- `gqa_expand(K, n_q_heads)`: K of shape (n_kv_heads, m, d) → (n_q_heads, m, d), each KV head shared by
  n_q_heads / n_kv_heads consecutive query heads.
