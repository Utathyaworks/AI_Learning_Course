# Average precision for object detection

Implement `average_precision(detections, ground_truth, iou_thr=0.5)` for a single class.

- `detections`: list of `(image_id, score, box)`; `ground_truth`: dict `image_id → array (G, 4)` of xyxy boxes.
- Sort detections by score (descending). Each detection matches the unmatched ground-truth box in its image with the
  highest IoU, if that IoU ≥ `iou_thr` (true positive); otherwise it is a false positive. A ground-truth box can be
  matched only once.
- Build the precision–recall curve, make precision monotonically non-increasing from the right (the envelope), and
  return the area under it using all recall change points (VOC 2010+ / "all-point" interpolation).
Return 0.0 if there are no ground-truth boxes.
