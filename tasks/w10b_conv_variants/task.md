# Convolution arithmetic: dilated and transposed convolution

Single-channel 2-D operations (valid cross-correlation, no padding):

- `conv_out_size(n, k, stride=1, padding=0, dilation=1)`: the standard output-length formula.
- `dilated_conv2d(x, w, dilation)`: valid correlation of `x` with kernel `w` whose taps are spaced `dilation` apart.
- `strided_conv2d(x, w, stride)`: valid correlation sampled every `stride` pixels.
- `conv_transpose2d(y, w, stride, out_shape)`: the **adjoint** of `strided_conv2d` for an input of shape
  `out_shape`, so that `⟨strided_conv2d(x, w, s), y⟩ = ⟨x, conv_transpose2d(y, w, s, x.shape)⟩` for all x, y.
