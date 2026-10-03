# Box encoding, IoU and GIoU

Boxes are `(x1, y1, x2, y2)` rows. Implement:

- `box_iou(a, b)`: (N, M) IoU matrix.
- `generalized_iou(a, b)`: element-wise GIoU for paired boxes of equal shape (N, 4):
  `IoU − |C \ (A ∪ B)| / |C|` with C the smallest enclosing box.
- `encode(boxes, anchors)`: Faster R-CNN deltas `(tx, ty, tw, th)` with `tx = (x − xa)/wa`, `ty = (y − ya)/ha`,
  `tw = log(w / wa)`, `th = log(h / ha)` using box centers and widths.
- `decode(deltas, anchors)`: the inverse, returning xyxy boxes.
