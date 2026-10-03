# RoI Align with bilinear sampling

Implement `roi_align(feature, box, out_size, sampling_ratio=2)` for a single-channel feature map (H, W),
following the `aligned=True` convention of torchvision:

- Pixel (i, j) has its value at continuous coordinates (x = j, y = i); shift the box by −0.5 so that box
  `(x1, y1, x2, y2)` in "pixel-edge" coordinates maps onto these centers.
- Split the box into `out_size × out_size` bins; in each bin take `sampling_ratio × sampling_ratio` evenly spaced
  sample points (at the centers of a regular sub-grid), bilinearly interpolate the feature at each, and average.
- Sample points outside `[−1, H] × [−1, W]` contribute 0; inside that range, clamp coordinates to the image.
