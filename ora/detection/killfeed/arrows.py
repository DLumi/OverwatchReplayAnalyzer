import cv2
import numpy as np

from ...utils.template_matching import find_template_multiscale, MatchConfig
from ...utils.morphology import filter_blobs, prepare_2d_portrait
from ...utils.scoring import binary_iou

ARROW_MATCH_CONFIG = MatchConfig(
    threshold=0.6,
    min_height_pct=0.085,
    max_height_pct=0.08,
    scale_steps=1,
)


def get_killfeed_entry_images(frame: np.ndarray, ref: np.ndarray,
                               entry_height_pc: float = 0.15,
                               debug: bool = False) \
        -> list[tuple[np.ndarray, tuple[int, int], tuple[int, ...]]]:
    h, w = frame.shape[:2]

    rx1 = int(w * 0.3)
    rx2 = int(w * 0.8)
    ry1 = int(h * 0.1)

    arrow_frame = frame[ry1:, rx1:rx2]
    arrow_data = detect_killfeed_arrows(arrow_frame, ref, debug=debug)

    if not arrow_data:
        return []

    arrow_data = [
        (
            (cx + rx1, cy + ry1),
            (bx + rx1, by + rx1, bw, bh),
        )
        for (cx, cy), (bx, by, bw, bh) in arrow_data
    ]

    arrow_centers = sorted([x[0] for x in arrow_data], key=lambda x: x[1], reverse=True)
    arrow_boxes = sorted([x[1] for x in arrow_data], key=lambda x: x[1], reverse=True)

    entry_h_halved = int(h * entry_height_pc / 2)

    kf_entries = []
    for i, ((x, y), (b_x, b_y, b_w, b_h)) in enumerate(zip(arrow_centers, arrow_boxes)):
        y1 = y - entry_h_halved
        y2 = y + entry_h_halved
        entry_cropped = frame[y1:y2, 0:w]
        kf_entries.append((entry_cropped, (x, y), (b_x, 0, b_w, entry_h_halved * 2)))
        if debug:
            cv2.imshow(f"cropped_entry_{i}", entry_cropped)

    return kf_entries


def detect_killfeed_arrows(frame: np.ndarray, ref: np.ndarray,
                            debug: bool = False) \
        -> list[tuple[tuple[int, int], tuple[int, ...]]]:
    if debug:
        cv2.imshow("frame", frame)

    roi_gray_blurred = compute_arrowness(frame, debug=debug)
    roi_edges_filled = (roi_gray_blurred > 50).astype(np.uint8) * 255

    roi_edges_filled, _ = filter_blobs(
        roi_edges_filled,
        min_size_frac=0.0007,
        max_size_frac=0.01,
        fill_percent=0.2,
        large_blob_multiplier=0.0,
        max_horizontal_dominance=3.0,
        max_vertical_dominance=5.0,
    )

    if debug:
        cv2.imshow("roi_edges_filled", roi_edges_filled)

    _, ref_edges_filled = prepare_2d_portrait(ref, margin=(0, 0))

    candidates = find_template_multiscale(
        frame=roi_edges_filled,
        template_base=ref_edges_filled,
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
        def candidate_key(res):
            patch = roi_edges_filled[res['box'][1]:res['box'][1] + res['box'][3],
                    res['box'][0]:res['box'][0] + res['box'][2]]
            template_resized = cv2.resize(ref_edges_filled, (res['box'][2], res['box'][3]),
                                          interpolation=cv2.INTER_NEAREST)
            iou = binary_iou(patch, template_resized)
            bucket = round(iou / 0.03)
            return bucket, res['score']

        best = max(group, key=candidate_key)
        x, y, w, h = best['box']

        patch = roi_edges_filled[y:y + h, x:x + w]
        template_resized = cv2.resize(ref_edges_filled, (w, h), interpolation=cv2.INTER_NEAREST)
        iou = binary_iou(patch, template_resized)

        if debug:
            print(f'best {iou=}')

        if iou < 0.4:
            continue

        center = int(x + w / 2), int(y + h / 2)
        arrow_data.append((center, best['box']))

    arrow_data = _filter_vertical_outliers(arrow_data, max_distance=50)

    if debug:
        for c, _ in arrow_data:
            cv2.circle(frame, c, radius=3, thickness=5, color=(0, 255, 0), lineType=cv2.LINE_AA)

    return arrow_data


def compute_arrowness(frame: np.ndarray, debug: bool = False) -> np.ndarray:
    """Score each pixel by how much it looks like part of a killfeed arrow."""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(frame.astype(np.int16))
    spread = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
    whiteness = np.exp(-spread.astype(np.float32) / 40) * 255
    whiteness = whiteness.astype(np.uint8)

    blur_hi = cv2.GaussianBlur(frame_gray, (0, 0), sigmaX=7)
    dog = cv2.subtract(frame_gray, blur_hi)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

    local_min = cv2.erode(frame_gray, np.ones((5, 5), np.uint8))
    local_min = cv2.GaussianBlur(local_min, (0, 0), sigmaX=7)
    dominance = cv2.subtract(frame_gray, local_min)

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
        cv2.imshow('dog', dog)
        cv2.imshow('dominance', dominance)
        cv2.imshow('edges_diagonal', edges_diagonal)
        cv2.imshow('arrowness', arrowness)

    return arrowness


def _group_by_row(candidates: list, vertical_threshold: int = 10) -> list:
    candidates = sorted(candidates, key=lambda c: c['box'][1])
    groups = []
    current_group = [candidates[0]]
    for c in candidates[1:]:
        if abs(c['box'][1] - current_group[0]['box'][1]) <= vertical_threshold:
            current_group.append(c)
        else:
            groups.append(current_group)
            current_group = [c]
    groups.append(current_group)
    return groups


def _filter_vertical_outliers(arrow_data: list, max_distance: int) -> list:
    if len(arrow_data) <= 1:
        return arrow_data
    arrow_data = sorted(arrow_data, key=lambda a: a[0][1])
    filtered = [arrow_data[0]]
    for current in arrow_data[1:]:
        if current[0][1] - filtered[-1][0][1] <= max_distance:
            filtered.append(current)
        else:
            break
    return filtered
