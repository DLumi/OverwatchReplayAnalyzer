import cv2
import numpy as np


def binary_iou(patch: np.ndarray, template: np.ndarray) -> float:
    """Intersection over Union on two binary (0/255) images.

    Both inputs are thresholded at 127 before comparison.
    Returns 0..1, where 1 is a perfect overlap."""
    p = patch > 127
    t = template > 127
    intersection = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return intersection / (union + 1e-6)


def blob_ncc(patch: np.ndarray, template: np.ndarray) -> float:
    """Normalized Cross-Correlation on raw pixel intensities.

    Mean-centers both images before correlating, so it's invariant to
    global brightness offset. Returns -1..1, where 1 is a perfect match."""
    p = patch.astype(np.float32) / 255.0
    t = template.astype(np.float32) / 255.0
    p -= p.mean()
    t -= t.mean()
    denom = np.sqrt(np.sum(p ** 2) * np.sum(t ** 2)) + 1e-6
    return np.sum(p * t) / denom


def shape_similarity(patch: np.ndarray, template: np.ndarray) -> float:
    """Shape dissimilarity via Hu moments (lower = more similar).

    Compares the log-scaled Hu moment vectors of two binary images.
    Rotation/scale/translation invariant by construction.
    Returns 0..∞, where 0 is identical shape."""
    m1 = cv2.moments((patch > 127).astype(np.uint8))
    m2 = cv2.moments((template > 127).astype(np.uint8))
    hu1 = cv2.HuMoments(m1).flatten()
    hu2 = cv2.HuMoments(m2).flatten()
    hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
    hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)
    return np.sum(np.abs(hu1 - hu2))


def edge_ncc(patch: np.ndarray, template: np.ndarray,
             mask: np.ndarray = None) -> float:
    """NCC computed on Sobel edge responses rather than raw pixels.

    More robust to lighting/color differences than blob_ncc — two images
    with different brightness but the same structure will still score high.
    Optional mask zeros out edge responses outside the region of interest.
    Returns -1..1, where 1 is a perfect edge-structure match."""
    if len(patch.shape) == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    ksize = 3
    p_edge = cv2.Sobel(patch, cv2.CV_32F, 1, 1, ksize=ksize)
    t_edge = cv2.Sobel(template, cv2.CV_32F, 1, 1, ksize=ksize)

    if mask is not None:
        p_edge = p_edge * (mask / 255.0)
        t_edge = t_edge * (mask / 255.0)

    num = np.sum(p_edge * t_edge)
    denom = np.sqrt(np.sum(p_edge ** 2) * np.sum(t_edge ** 2)) + 1e-6
    return num / denom
