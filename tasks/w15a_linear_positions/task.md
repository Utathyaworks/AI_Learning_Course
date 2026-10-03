# Linear attention as an RNN, RoPE, ALiBi and KV-cache memory

Implement:

- `linear_attention_quadratic(Q, K, V, causal=True)`: with feature map φ(x) = elu(x) + 1, output row i is
  Σ_{j≤i} (φ(q_i)·φ(k_j)) v_j / Σ_{j≤i} φ(q_i)·φ(k_j) (all j if not causal).
- `linear_attention_recurrent(Q, K, V)`: the same causal output computed as an RNN with state S = Σ φ(k_j) v_jᵀ
  (d × dv) and normalizer z = Σ φ(k_j), updated one step at a time.
- `rope(X, positions, base=10000.0)`: rotary embedding on X (n, d), d even — rotate each pair (x_{2i}, x_{2i+1}) by
  angle positions · base^(−2i/d).
- `alibi_bias(n, n_heads)`: (n_heads, n, n) causal bias −m_h·(i − j) for j ≤ i and −∞ for j > i, with slopes
  m_h = 2^(−8(h+1)/n_heads) for h = 0 … n_heads−1.
- `kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, bytes_per=2)`: memory to cache K and V for one sequence.
