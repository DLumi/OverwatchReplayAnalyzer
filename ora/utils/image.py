import cv2
import numpy as np
from pathlib import Path


def crop(img: np.ndarray, pos_arr: tuple[int, int, int, int]) -> np.ndarray:
    """Crop image with [y, h, x, w] coords, clamping to image bounds."""
    y, h, x, w = pos_arr
    img_h, img_w = img.shape[:2]
    y1 = max(0, y)
    x1 = max(0, x)
    y2 = min(img_h, y + h)
    x2 = min(img_w, x + w)
    return img[y1:y2, x1:x2]


def read_image(path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Load an image from a Unicode path."""
    data = np.fromfile(Path(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return img
