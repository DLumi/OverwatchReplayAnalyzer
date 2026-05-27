import cv2
import numpy as np


def detect_killfeed_presence(frame: np.ndarray) -> bool:
    """Killfeed visible = long horizontal lines around blueish / reddish objects + white text blobs near them"""

    raw_hor_lines = find_hor_edges_by_color(frame, min_len=12, percentile=95)

    line_mask, raw_lines_list = extend_and_merge_lines(raw_hor_lines)

    text_mask = find_killfeed_text_mask(frame)

    scores = [text_near_line_score(text_mask, line) for line in raw_lines_list]
    is_found = any(score >= 0.05 for score in scores)

    return is_found


def find_killfeed_text_mask(frame: np.ndarray, min_component_frac=0.001, max_component_frac=0.02) -> np.ndarray:
    """Detect something that looks like text in the frame

    Args:
        frame: a BGR image to analyze
        min_component_frac: minimum size of a text blob, as a % of the frame size
        max_component_frac: maximum size of a text blob, as a % of the frame size

    Returns:
        a text blob mask
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # text is essentially bright blobs
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


def find_hor_edges_by_color(frame: np.ndarray, percentile=97, min_len=12) -> np.ndarray:
    """Detect horizontal edges of potential killfeed entries

    Args:
        frame: a BGR image
        percentile: 0-100, the higher the number the harsher the filtration
        min_len: minimum length of an edge line, in pixels

    Returns:
        a mask of horizontal edges"""

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # horizontal boundaries = vertical derivative
    dyL = np.abs(cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
    dya = np.abs(cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3))
    dyb = np.abs(cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3))

    # chroma-heavy edge map
    strength = 0.3 * dyL + 1.2 * dya + 1.2 * dyb

    thr = np.percentile(strength, percentile)
    mask = (strength > thr).astype(np.uint8) * 255

    # keep horizontal runs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return h


def extend_and_merge_lines(edge_mask: np.ndarray,
                           extend_by: float = 1.5,
                           min_line_len: int = 30,
                           vert_thresh: int = 5) \
        -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]]]:
    """
    Args:
        edge_mask: mask of horizontal edges
        extend_by: fraction of the line length
        min_line_len: in pixels
        vert_thresh: grace threshold to merge the lines, in pixels

    Returns:
        updated line map + raw lines (XYXY) list

    """


    h, w = edge_mask.shape[:2]

    line_segments = extract_lines_from_edges(edge_mask, min_len=min_line_len)
    rows = group_segments_by_y(line_segments, y_thr=vert_thresh)

    out = np.zeros_like(edge_mask)
    raw_lines = []

    for row in rows:
        merged = merge_lines_in_row(row, w, extend_frac=extend_by)

        for x1, x2 in merged:
            line = (x1, row["cy"]), (x2, row["cy"])
            raw_lines.append(line)
            cv2.line(out, line[0], line[1], 255, 2)

    return out, raw_lines


def extract_lines_from_edges(edge_mask, min_len=25):
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    h = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, h_kernel)

    cnts, _ = cv2.findContours(h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segs = []
    for c in cnts:
        x, y, w, h_ = cv2.boundingRect(c)
        if w >= min_len and h_ <= 4:
            segs.append([x, y, x + w, y + h_])

    return segs


def group_segments_by_y(segs, y_thr=5):
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


def merge_lines_in_row(row, img_w: int, extend_frac=0.5):
    intervals = []

    for x1, y1, x2, y2 in row["segs"]:
        length = x2 - x1
        ext = int(length * extend_frac)

        intervals.append([
            max(0, x1 - ext),
            min(img_w, x2 + ext)
        ])

    intervals.sort()

    merged = []
    for a, b in intervals:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    return merged


def text_near_line_score(text_mask, line, horizontal_area=20, vertical_area=24):
    """Measures text intersection with area around the line

    Args:
        text_mask: text blob mask
        line: (x,y) tuple
        horizontal_area: horizontal area to the left and to the right of the line, pixels
        vertical_area: vertical area below and above the line, pixels"""

    h, w = text_mask.shape[:2]
    (x1, y), (x2, _) = line

    rx1 = max(0, x1 - horizontal_area)
    rx2 = min(w, x2 + horizontal_area)
    ry1 = max(0, y - vertical_area)
    ry2 = min(h, y + vertical_area)

    zone = text_mask[ry1:ry2, rx1:rx2]

    # local text density, not relative to whole frame
    return cv2.countNonZero(zone) / max(zone.size, 1)


def filter_blobs(mask: np.ndarray,
                 min_size_frac: float = 0.001,
                 max_size_frac: float = 0.02,
                 fill_percent: float = 0.7,
                 large_blob_multiplier: float = 5.0,
                 max_vertical_dominance: float = 2.0,
                 max_horizontal_dominance: float = 11.0
                 ) -> np.ndarray:

    """Parametric mask filter; only blobs surviving the filter pass through

    Args:
        mask: mask array
        min_size_frac: minimum blob size, as a fraction of total mask area
        max_size_frac: maximum blob size, as a fraction of total mask area
        fill_percent: minimum bbox fill ratio required for a blob to survive
        large_blob_multiplier: a multiplier for min_size_frac after which the blob is considered large
        max_vertical_dominance: threshold for vertically elongated blobs
        max_horizontal_dominance: threshold for horizontally elongated blobs

    Returns:
        a refined mask array
    """

    # analyzing raw blobs and filter out non-text
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out = np.zeros_like(mask)
    components = []

    img_area = mask.shape[0] * mask.shape[1]

    for i in range(1, num_labels):  # skip background
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        area_frac = area / img_area

        bbox_area = w * h
        fill = area / bbox_area

        long_axis_ratio = max(w, h) / max(min(w, h), 1)

        bad = (
                # generally too big
                area_frac > max_size_frac or

                # generally too small
                area_frac < min_size_frac or

                # wide line-like blob
                (area_frac > min_size_frac * large_blob_multiplier
                 and w >= h
                 and long_axis_ratio > max_horizontal_dominance) or

                # tall line-like blob
                (area_frac > min_size_frac * large_blob_multiplier
                 and h > w
                 and long_axis_ratio > max_vertical_dominance) or

                # anything sparsely populated (lines, curves, etc)
                fill < fill_percent
        )

        if not bad:
            out[labels == i] = 255
            cx, cy = centroids[i]

            components.append({
                "box": (x, y, w, h),
                "center": (int(cx), int(cy)),
                "area": int(area),
            })
    return out, components


def fill_closed_edge_shapes(edges, min_area=5, mode=cv2.RETR_EXTERNAL):
    # edges: uint8 0/255
    cnts, _ = cv2.findContours(edges, mode, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(edges)

    for c in cnts:
        area = cv2.contourArea(c)
        if area >= min_area:
            cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)

    return filled
