from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

from .box_proc import percent_roi_to_pixels, non_max_suppression
from .scoring import edge_ncc


class MatchResult(NamedTuple):
    box: tuple[int, int, int, int]   # XYXY
    center: tuple[int, int]
    score: float
    ncc: float
    scale: float
    shape: tuple[int, int]


@dataclass
class MatchConfig:
    """Config file for template matching

    Args:
        threshold: minimum matchTemplate score to keep a candidate
        min_height_pct: smallest possible height for the template, as a % of frame height
        max_height_pct: largest possible height for the template, as a % of frame height
        scale_steps: number of resizing steps between min and max
        method: matchTemplate's `method` argument
        nms_overlap: IoU overlap threshold for non-max suppression
        ncc_threshold: minimum edge NCC score; only applied when mask_base is provided;
                       higher => candidate matches template more"""

    threshold: float = 0.8
    min_height_pct: float = 0.03
    max_height_pct: float = 0.08
    scale_steps: int | None = 12
    method: int = cv2.TM_CCOEFF_NORMED
    nms_overlap: float = 0.3
    ncc_threshold: float = 0.3


def _extract_roi(frame: np.ndarray,
                 roi_pct: tuple[float, float, float, float] | None) \
        -> tuple[np.ndarray, int, int]:
    if roi_pct is not None:
        rx, ry, rw, rh = percent_roi_to_pixels(frame.shape, roi_pct)
        return frame[ry:ry + rh, rx:rx + rw], rx, ry
    return frame, 0, 0


def find_template_multiscale(
        frame: np.ndarray,
        template_base: np.ndarray,
        threshold: float = 0.8,
        roi_pct: tuple[float, float, float, float] | None = None,
        scales: list[float] | None = None,
        min_height_pct: float = 0.03,
        max_height_pct: float = 0.08,
        scale_steps: int | None = 12,
        method: int = cv2.TM_CCOEFF_NORMED,
        nms_overlap: float = 0.3,
        mask_base: np.ndarray | None = None,
        ncc_threshold: float = 0.3,
) -> list[MatchResult]:
    search_img, rx, ry = _extract_roi(frame, roi_pct)

    if scales is None and scale_steps:
        scales = _compute_resizing_scales(
            frame.shape,
            template_base.shape,
            min_height_pct=min_height_pct,
            max_height_pct=max_height_pct,
            steps=scale_steps,
        )
    else:
        raise ValueError(r'Could not calculate `scales` - `scale_steps` were not provided!')

    candidates = []
    for scale in scales:
        candidates.extend(_match_at_scale(search_img, template_base, scale,
                                          method, threshold, mask_base,
                                          ncc_threshold, rx, ry))
    return non_max_suppression(candidates, nms_overlap)


def _match_at_scale(
        search_img: np.ndarray,
        template_base: np.ndarray,
        scale: float,
        method: int,
        threshold: float,
        mask_base: np.ndarray | None,
        ncc_threshold: float,
        rx: int,
        ry: int,
) -> list[MatchResult]:
    template = cv2.resize(template_base, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    th, tw = template.shape[:2]

    if th < 3 or tw < 3:
        return []
    if th > search_img.shape[0] or tw > search_img.shape[1]:
        return []

    mask = None
    if mask_base is not None:
        mask = cv2.resize(mask_base, (tw, th), interpolation=cv2.INTER_NEAREST)
        visible = cv2.countNonZero(mask)
        if visible < 10 or visible / float(tw * th) < 0.15:
            return []

    result = cv2.matchTemplate(search_img, template, method, mask=mask)

    kernel = np.ones((3, 3), np.uint8)
    local_max = result == cv2.dilate(result, kernel)
    ys, xs = np.where((result >= threshold) & local_max)

    candidates = []
    for x, y in zip(xs, ys):
        patch = search_img[y:y + th, x:x + tw]

        ncc = 0.0
        if mask_base is not None:
            ncc = edge_ncc(patch, template, mask)
            if ncc < ncc_threshold:
                continue

        x1, y1 = x + rx, y + ry
        x2, y2 = x1 + tw, y1 + th
        candidates.append(MatchResult(
            box=(x1, y1, x2, y2),
            center=((x1 + x2) // 2, (y1 + y2) // 2),
            score=float(result[y, x]),
            ncc=float(ncc),
            scale=float(scale),
            shape=(th, tw),
        ))

    return candidates


def _compute_resizing_scales(
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


