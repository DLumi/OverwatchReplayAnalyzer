import numpy as np


def percent_roi_to_pixels(frame_shape: tuple[int, int],
                          roi_pct: tuple[float, float, float, float]) \
        -> tuple[int, int, int, int]:
    """Convert (x1, y1, x2, y2) fractional ROI to (x, y, w, h) pixel coords."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = roi_pct
    x1 = int(round(x1 * w))
    y1 = int(round(y1 * h))
    x2 = int(round(x2 * w))
    y2 = int(round(y2 * h))
    return x1, y1, x2 - x1, y2 - y1


def non_max_suppression(matches: list, overlap_thresh: float = 0.3) -> list:
    if not matches:
        return []

    boxes = np.array([m.box for m in matches], dtype=np.float32)
    scores = np.array([m.score for m in matches], dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        overlap = inter / areas[order[1:]]
        order = order[1:][overlap <= overlap_thresh]

    return [matches[int(i)] for i in keep]
