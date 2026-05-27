import cv2
import numpy as np


def binary_iou(patch: np.ndarray, template: np.ndarray) -> float:
    p = patch > 127
    t = template > 127
    intersection = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return intersection / (union + 1e-6)


def blob_ncc(patch: np.ndarray, template: np.ndarray) -> float:
    p = patch.astype(np.float32) / 255.0
    t = template.astype(np.float32) / 255.0
    p -= p.mean()
    t -= t.mean()
    denom = np.sqrt(np.sum(p ** 2) * np.sum(t ** 2)) + 1e-6
    return np.sum(p * t) / denom


def shape_similarity(patch: np.ndarray, template: np.ndarray) -> float:
    m1 = cv2.moments((patch > 127).astype(np.uint8))
    m2 = cv2.moments((template > 127).astype(np.uint8))
    hu1 = cv2.HuMoments(m1).flatten()
    hu2 = cv2.HuMoments(m2).flatten()
    hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
    hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)
    return np.sum(np.abs(hu1 - hu2))


def edge_ncc(patch: np.ndarray, template: np.ndarray, mask: np.ndarray = None) -> float:
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
