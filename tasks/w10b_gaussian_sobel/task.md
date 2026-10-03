# Separable Gaussian blur and Sobel gradients

Implement classical filters for grayscale images (2-D float arrays).

- `gaussian_kernel1d(sigma)`: radius `r = ceil(3·sigma)`, weights `exp(−x²/2σ²)` for x = −r…r, normalized to sum 1.
- `gaussian_blur(img, sigma)`: separable blur (rows then columns) with **reflect** padding
  (`np.pad(..., mode="reflect")`, i.e. the edge pixel is not repeated). Output has the input's shape.
- `sobel(img)`: return `(gx, gy, magnitude)` using the 3×3 Sobel kernels as **correlation**
  (`gx` responds to left→right increases), with `np.pad(..., mode="symmetric")` padding.
