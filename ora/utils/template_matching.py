from dataclasses import dataclass, field

import cv2
import numpy as np

from .scoring import edge_ncc


@dataclass
class MatchConfig:
    threshold: float = 0.8
    min_height_pct: float = 0.03
    max_height_pct: float = 0.08
    scale_steps: int = 12
    method: int = field(default_factory=lambda: cv2.TM_CCOEFF_NORMED)
    nms_overlap: float = 0.3
    mae_threshold: float = 28.0


def find_template_multiscale(
        frame: np.ndarray,
        template_base: np.ndarray,
        threshold: float = 0.8,
        roi_pct=None,
        scales=None,
        min_height_pct: float = 0.03,
        max_height_pct: float = 0.08,
        scale_steps: int = 12,
        method: int = cv2.TM_CCOEFF_NORMED,
        nms_overlap: float = 0.3,
        mask_base=None,
        mae_threshold: float = 28.0,
) -> list[dict]:
    if roi_pct is not None:
        rx, ry, rw, rh = percent_roi_to_pixels(frame.shape, roi_pct)
        search_img = frame[ry:ry + rh, rx:rx + rw]
    else:
        rx, ry = 0, 0
        search_img = frame

    if scales is None:
        scales = make_scales_for_frame(
            frame.shape,
            template_base.shape,
            min_height_pct=min_height_pct,
            max_height_pct=max_height_pct,
            steps=scale_steps,
        )

    candidates = []

    for scale in scales:
        template = cv2.resize(template_base, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        th, tw = template.shape[:2]

        if th < 3 or tw < 3:
            continue
        if th > search_img.shape[0] or tw > search_img.shape[1]:
            continue

        mask = None
        if mask_base is not None:
            mask = cv2.resize(mask_base, (tw, th), interpolation=cv2.INTER_NEAREST)
            visible = cv2.countNonZero(mask)
            if visible < 10:
                continue
            if visible / float(tw * th) < 0.15:
                continue

        result = cv2.matchTemplate(search_img, template, method, mask=mask)

        kernel = np.ones((3, 3), np.uint8)
        local_max = result == cv2.dilate(result, kernel)
        ys, xs = np.where((result >= threshold) & local_max)

        for x, y in zip(xs, ys):
            full_x = x + rx
            full_y = y + ry
            patch = search_img[y:y + th, x:x + tw]

            mae = 0
            if mask_base is not None:
                mae = edge_ncc(patch, template, mask)
                if mae < mae_threshold:
                    continue

            candidates.append({
                "score": float(result[y, x]),
                "mae": float(mae),
                "loc": (full_x, full_y),
                "scale": float(scale),
                "shape": (th, tw),
                "box": (full_x, full_y, tw, th),
            })

    return non_max_suppression(candidates, nms_overlap)


def percent_roi_to_pixels(frame_shape, roi_pct: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """Convert (x1, y1, x2, y2) fractional ROI to (x, y, w, h) pixel coords."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = roi_pct
    x1 = int(round(x1 * w))
    y1 = int(round(y1 * h))
    x2 = int(round(x2 * w))
    y2 = int(round(y2 * h))
    return x1, y1, x2 - x1, y2 - y1


def make_scales_for_frame(
        frame_shape,
        template_shape,
        min_height_pct: float = 0.03,
        max_height_pct: float = 0.08,
        steps: int = 12,
) -> np.ndarray:
    frame_h = frame_shape[0]
    template_h = template_shape[0]
    min_scale = (frame_h * min_height_pct) / template_h
    max_scale = (frame_h * max_height_pct) / template_h
    return np.linspace(min_scale, max_scale, steps)


def non_max_suppression(matches: list[dict], overlap_thresh: float = 0.3) -> list[dict]:
    if not matches:
        return []

    boxes = np.array([m["box"] for m in matches], dtype=np.float32)
    scores = np.array([m["score"] for m in matches], dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = boxes[:, 2] * boxes[:, 3]
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
