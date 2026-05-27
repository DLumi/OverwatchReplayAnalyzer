import cv2
import numpy as np


def filter_blobs(mask: np.ndarray,
                 min_size_frac: float = 0.001,
                 max_size_frac: float = 0.02,
                 fill_percent: float = 0.7,
                 large_blob_multiplier: float = 5.0,
                 max_vertical_dominance: float = 2.0,
                 max_horizontal_dominance: float = 11.0
                 ) -> tuple[np.ndarray, list[dict]]:
    """Parametric mask filter; only blobs surviving the filter pass through.

    Args:
        mask: binary mask array
        min_size_frac: minimum blob size as a fraction of total mask area
        max_size_frac: maximum blob size as a fraction of total mask area
        fill_percent: minimum bbox fill ratio required for a blob to survive
        large_blob_multiplier: multiplier for min_size_frac above which a blob is "large"
        max_vertical_dominance: threshold for vertically elongated blobs
        max_horizontal_dominance: threshold for horizontally elongated blobs

    Returns:
        refined mask array, list of surviving blob dicts with box/center/area
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out = np.zeros_like(mask)
    components = []
    img_area = mask.shape[0] * mask.shape[1]

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        area_frac = area / img_area
        fill = area / (w * h)
        long_axis_ratio = max(w, h) / max(min(w, h), 1)

        bad = (
            area_frac > max_size_frac or
            area_frac < min_size_frac or
            (area_frac > min_size_frac * large_blob_multiplier
             and w >= h
             and long_axis_ratio > max_horizontal_dominance) or
            (area_frac > min_size_frac * large_blob_multiplier
             and h > w
             and long_axis_ratio > max_vertical_dominance) or
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


def fill_closed_edge_shapes(edges: np.ndarray, min_area: int = 5,
                            mode: int = cv2.RETR_EXTERNAL) -> np.ndarray:
    cnts, _ = cv2.findContours(edges, mode, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(edges)
    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
    return filled


def prepare_2d_portrait(image: np.ndarray,
                        margin: tuple[int, int] = (50, 50)) -> tuple[np.ndarray, np.ndarray | None]:
    """Crop margins and split image into BGR template and alpha mask.

    Args:
        image: BGR or BGRA image
        margin: (vertical, horizontal) margin to crop from each side

    Returns:
        (template_bgr, mask) — mask is None if image has no alpha channel
    """
    m_v, m_h = margin
    h, w = image.shape[:2]

    if image.ndim == 3 and image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
        mask = ((alpha > 10).astype(np.uint8) * 255)[m_v:h - m_v, m_h:w - m_h]
        template = bgr[m_v:h - m_v, m_h:w - m_h]
    else:
        template = image[m_v:h - m_v, m_h:w - m_h]
        mask = None

    return template, mask
