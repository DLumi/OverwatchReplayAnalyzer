import cv2
import numpy as np

from ...utils.morphology import filter_blobs


def detect_killfeed_presence(frame: np.ndarray) -> bool:
    """Killfeed visible = long horizontal lines around blueish/reddish objects + white text blobs near them."""
    raw_hor_lines = find_hor_edges_by_color(frame, min_len=12, percentile=95)
    line_mask, raw_lines_list = extend_and_merge_lines(raw_hor_lines)
    text_mask = find_killfeed_text_mask(frame)
    scores = [text_near_line_score(text_mask, line) for line in raw_lines_list]
    return any(score >= 0.05 for score in scores)


def find_killfeed_text_mask(frame: np.ndarray,
                             min_component_frac: float = 0.001,
                             max_component_frac: float = 0.02) -> np.ndarray:
    """Detect white text blobs in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = (gray > 200).astype("uint8") * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((2, 8), np.uint8), iterations=1)
    mask, _ = filter_blobs(
        mask,
        min_size_frac=min_component_frac,
        max_size_frac=max_component_frac,
        fill_percent=0.7,
        large_blob_multiplier=5.0,
        max_vertical_dominance=1.0,
        max_horizontal_dominance=11.0,
    )
    return mask


def find_hor_edges_by_color(frame: np.ndarray, percentile: int = 97, min_len: int = 12) -> np.ndarray:
    """Detect horizontal edges of potential killfeed entries via LAB color derivatives."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    dyL = np.abs(cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
    dya = np.abs(cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3))
    dyb = np.abs(cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3))

    strength = 0.3 * dyL + 1.2 * dya + 1.2 * dyb
    thr = np.percentile(strength, percentile)
    mask = (strength > thr).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def extend_and_merge_lines(edge_mask: np.ndarray,
                            extend_by: float = 1.5,
                            min_line_len: int = 30,
                            vert_thresh: int = 5) \
        -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]]]:
    h, w = edge_mask.shape[:2]
    line_segments = _extract_lines_from_edges(edge_mask, min_len=min_line_len)
    rows = _group_segments_by_y(line_segments, y_thr=vert_thresh)

    out = np.zeros_like(edge_mask)
    raw_lines = []

    for row in rows:
        merged = _merge_lines_in_row(row, w, extend_frac=extend_by)
        for x1, x2 in merged:
            line = (x1, row["cy"]), (x2, row["cy"])
            raw_lines.append(line)
            cv2.line(out, line[0], line[1], 255, 2)

    return out, raw_lines


def text_near_line_score(text_mask: np.ndarray, line,
                          horizontal_area: int = 20,
                          vertical_area: int = 24) -> float:
    h, w = text_mask.shape[:2]
    (x1, y), (x2, _) = line
    rx1 = max(0, x1 - horizontal_area)
    rx2 = min(w, x2 + horizontal_area)
    ry1 = max(0, y - vertical_area)
    ry2 = min(h, y + vertical_area)
    zone = text_mask[ry1:ry2, rx1:rx2]
    return cv2.countNonZero(zone) / max(zone.size, 1)


def _extract_lines_from_edges(edge_mask: np.ndarray, min_len: int = 25) -> list:
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    h = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, h_kernel)
    cnts, _ = cv2.findContours(h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segs = []
    for c in cnts:
        x, y, w, h_ = cv2.boundingRect(c)
        if w >= min_len and h_ <= 4:
            segs.append([x, y, x + w, y + h_])
    return segs


def _group_segments_by_y(segs: list, y_thr: int = 5) -> list:
    rows = []
    for x1, y1, x2, y2 in sorted(segs, key=lambda s: (s[1], s[0])):
        cy = (y1 + y2) // 2
        placed = False
        for row in rows:
            if abs(row["cy"] - cy) <= y_thr:
                row["segs"].append([x1, y1, x2, y2])
                row["cy"] = int(np.mean([(s[1] + s[3]) // 2 for s in row["segs"]]))
                placed = True
                break
        if not placed:
            rows.append({"cy": cy, "segs": [[x1, y1, x2, y2]]})
    return rows


def _merge_lines_in_row(row: dict, img_w: int, extend_frac: float = 0.5) -> list:
    intervals = []
    for x1, y1, x2, y2 in row["segs"]:
        length = x2 - x1
        ext = int(length * extend_frac)
        intervals.append([max(0, x1 - ext), min(img_w, x2 + ext)])
    intervals.sort()
    merged = []
    for a, b in intervals:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return merged
