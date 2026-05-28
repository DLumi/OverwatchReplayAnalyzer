import cv2
import numpy as np

from ...utils.template_matching import find_template_multiscale, MatchConfig, MatchResult
from ...utils.morphology import filter_blobs, prepare_template
from ...utils.scoring import binary_iou

ARROW_MATCH_CONFIG = MatchConfig(
    threshold=0.6,
    min_height_pct=0.085,
    max_height_pct=0.086,
    scale_steps=1,
)


def detect_killfeed_arrows(frame: np.ndarray, ref: np.ndarray,
                           debug: bool = False) -> list[MatchResult]:
    if debug:
        cv2.imshow("frame", frame)

    roi_gray_blurred = _compute_arrowness(frame, debug=debug)
    roi_binary = (roi_gray_blurred > 50).astype(np.uint8) * 255

    roi_binary, _ = filter_blobs(
        roi_binary,
        min_size_frac=0.0007,
        max_size_frac=0.01,
        fill_percent=0.2,
        large_blob_multiplier=0.0,
        max_horizontal_dominance=3.0,
        max_vertical_dominance=5.0,
    )

    if debug:
        cv2.imshow("roi_binary", roi_binary)

    _, ref_binary = prepare_template(ref, margin=(0, 0))

    candidates = find_template_multiscale(
        frame=roi_binary,
        template_base=ref_binary,
        threshold=ARROW_MATCH_CONFIG.threshold,
        min_height_pct=ARROW_MATCH_CONFIG.min_height_pct,
        max_height_pct=ARROW_MATCH_CONFIG.max_height_pct,
        scale_steps=ARROW_MATCH_CONFIG.scale_steps,
    )

    if not candidates:
        return []

    candidate_groups = _group_by_row(candidates, vertical_threshold=22)

    arrow_data = []
    for group in candidate_groups:
        candidates_with_iou = [
            (r, _arrow_iou(r, roi_binary, ref_binary))
            for r in group
        ]
        best, best_iou = max(candidates_with_iou,
                             key=lambda x: (round(x[1] / 0.03), x[0].score))

        if debug:
            print(f'best {best_iou=}')

        if best_iou < 0.4:
            continue

        arrow_data.append(best)

    arrow_data = _filter_vertical_outliers(arrow_data, max_distance=50)

    if debug:
        for result in arrow_data:
            cv2.circle(frame, result.center, radius=3, thickness=5,
                       color=(0, 255, 0), lineType=cv2.LINE_AA)

    return arrow_data


def _arrow_iou(res: MatchResult, frame: np.ndarray, ref: np.ndarray) -> float:
    patch = frame[res.box[1]:res.box[3], res.box[0]:res.box[2]]
    bw, bh = res.box[2] - res.box[0], res.box[3] - res.box[1]
    template_resized = cv2.resize(ref, (bw, bh), interpolation=cv2.INTER_NEAREST)
    return binary_iou(patch, template_resized)


def _compute_arrowness(frame: np.ndarray, debug: bool = False) -> np.ndarray:
    """Computes three heatmaps that target main arrow properties: whiteness,
    local domination (brighter than pixels in the immediate vicinity), and prevalence
    of diagonal shapes.

    Returns:
        a combined heatmap of weighted properties - (dominance ** 1.0) * (whiteness ** 0.4) * (edges_diagonal ** 0.6) ** (1 / 2)"""

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # whiteness
    b, g, r = cv2.split(frame.astype(np.int16))
    spread = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
    whiteness = np.exp(-spread.astype(np.float32) / 40) * 255
    whiteness = whiteness.astype(np.uint8)

    # local dominance
    local_min = cv2.erode(frame_gray, np.ones((5, 5), np.uint8))
    local_min = cv2.GaussianBlur(local_min, (0, 0), sigmaX=7)
    dominance = cv2.subtract(frame_gray, local_min)

    # diagonal edges
    gx = cv2.Sobel(frame_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame_gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    threshold = np.percentile(magnitude, 1)
    magnitude = np.clip(magnitude - threshold, 0, None)
    angle = np.arctan2(np.abs(gy), np.abs(gx))
    diagonalness = np.sin(2 * angle)
    edges_diagonal = magnitude * diagonalness
    edges_diagonal = cv2.normalize(edges_diagonal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    arrowness = (dominance.astype(np.float32) ** 1.0 *
                 whiteness.astype(np.float32) ** 0.4 *
                 edges_diagonal.astype(np.float32) ** 0.6) ** (1 / 2)
    arrowness = cv2.normalize(arrowness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if debug:
        cv2.imshow('whiteness', whiteness)
        cv2.imshow('dominance', dominance)
        cv2.imshow('edges_diagonal', edges_diagonal)
        cv2.imshow('arrowness', arrowness)

    return arrowness


def _group_by_row(candidates: list[MatchResult],
                  vertical_threshold: int = 10) -> list[list[MatchResult]]:
    """Group candidates into the rows within a certain pixel threshold"""

    candidates = sorted(candidates, key=lambda c: c.box[1])
    groups = []
    current_group = [candidates[0]]
    for c in candidates[1:]:
        if abs(c.box[1] - current_group[0].box[1]) <= vertical_threshold:
            current_group.append(c)
        else:
            groups.append(current_group)
            current_group = [c]
    groups.append(current_group)
    return groups


def _filter_vertical_outliers(arrow_data: list[MatchResult],
                               max_distance: int) -> list[MatchResult]:
    """Discard rows after a certain pixel distance"""

    if len(arrow_data) <= 1:
        return arrow_data
    arrow_data = sorted(arrow_data, key=lambda a: a.center[1])
    filtered = [arrow_data[0]]
    for current in arrow_data[1:]:
        if current.center[1] - filtered[-1].center[1] <= max_distance:
            filtered.append(current)
        else:
            break
    return filtered
