# Symmetric int8 quantization

Implement per-tensor and per-channel **symmetric** int8 quantization.

- `quantize(x, axis=None)` → `(q, scale)`: `q` is `np.int8` in [-127, 127], `scale = max|x| / 127` (per tensor,
  or per slice along `axis`, kept broadcastable). Use round-half-to-even (`np.round`). If max|x| is 0, use scale 1.
- `dequantize(q, scale)` → float array `q * scale`.
The round-trip error per element must be at most `scale / 2`.
