from typing import Literal
from itertools import repeat

import cv2
import numpy as np

from ...utils.template_matching import find_template_multiscale, MatchConfig, percent_roi_to_pixels
from ...utils.morphology import prepare_2d_portrait
from ...hero import Hero

KF_MAIN_CONFIG = MatchConfig(
    min_height_pct=0.8,
    max_height_pct=0.9,
    mae_threshold=0.3,
    threshold=0.92,
)

KF_ASSIST_CONFIG = MatchConfig(
    min_height_pct=0.6,
    max_height_pct=0.8,
    mae_threshold=0.45,
    threshold=0.88,
)

TEAM_COLORS = {0: np.array([40, 40, 220]),
               1: np.array([220, 220, 40])}


def detect_heroes_2d(frame: np.ndarray,
                     hero_list: list[Hero],
                     portrait_type: Literal['kf_main', 'kf_assist'],
                     debug: bool = False) -> list[tuple[Hero, tuple[int, int], tuple[int, ...], int]]:
    fr_h = frame.shape[0]

    if portrait_type == 'kf_main':
        config = KF_MAIN_CONFIG
        margin = (50, 50)
        steps = int((config.max_height_pct - config.min_height_pct) * fr_h)
        threshold = config.threshold
        method = cv2.TM_CCORR_NORMED
    elif portrait_type == 'kf_assist':
        config = KF_ASSIST_CONFIG
        margin = (20, 50)
        steps = int((config.max_height_pct - config.min_height_pct) * fr_h)
        threshold = config.threshold
        method = cv2.TM_CCORR_NORMED
    else:
        raise ValueError(f'portrait_type {portrait_type} not supported')

    search_params = {
        'min_height_pct': config.min_height_pct,
        'max_height_pct': config.max_height_pct,
        'scale_steps': steps,
        'mae_threshold': config.mae_threshold,
    }

    rx, ry, rw, rh = percent_roi_to_pixels(frame.shape, (0.08, 0.0, 0.92, 1.0))
    search_space = frame[ry:ry + rh, rx:rx + rw]

    detected_instances = []
    for hero in hero_list:
        template, mask = prepare_2d_portrait(hero.portrait2d, margin=margin)

        if mask is None:
            method = cv2.TM_CCOEFF_NORMED
            threshold = 0.8

        found_instances = find_template_multiscale(search_space, template, **search_params,
                                                   mask_base=mask,
                                                   method=method,
                                                   threshold=threshold)
        if found_instances:
            detected_instances.extend(zip(repeat(hero), found_instances))

    detected_heroes = []
    for hero, instance in detected_instances:
        if debug:
            cv2.circle(frame, instance['loc'], radius=3,
                       thickness=5, color=(0, 255, 0), lineType=cv2.LINE_AA)

        x, y, w, h = instance['box']
        x = rx + x
        center = (int(x + w / 2), int(y + h / 2))

        if portrait_type == 'kf_main':
            left_color, right_color = scan_side_color(frame, (x, y, w, h))
            team_color = np.mean((left_color, right_color), axis=0)
            team = closest_team(team_color)
        else:
            team = -1

        if debug and portrait_type == 'kf_main':
            swatch = np.zeros((100, 300, 3), dtype=np.uint8)
            swatch[:, :100] = left_color
            swatch[:, 100:] = right_color
            swatch[:, 200:] = team_color
            cv2.imshow(f"swatch_{hero}", swatch)

        detected_heroes.append((hero, center, (x, y, w, h), team))

    return sorted(detected_heroes, key=lambda x: x[1][0])


def scan_side_color(frame: np.ndarray, box, team_color_px_offset: int = 10,
                    scan_width: int = 2):
    x, y, w, h = map(int, box)
    H, W = frame.shape[:2]

    left_x1 = max(0, x - team_color_px_offset - scan_width)
    left_x2 = max(0, x - team_color_px_offset)
    right_x1 = min(W, x + w + team_color_px_offset)
    right_x2 = min(W, x + w + team_color_px_offset + scan_width)
    y1 = max(0, y)
    y2 = min(H, y + h)

    left_strip = frame[y1:y2, left_x1:left_x2]
    right_strip = frame[y1:y2, right_x1:right_x2]

    left_mean = left_strip.mean(axis=(0, 1)) if left_strip.size else None
    right_mean = right_strip.mean(axis=(0, 1)) if right_strip.size else None

    return left_mean, right_mean


def closest_team(color) -> int:
    color = np.array(color)
    team1 = np.linalg.norm(color - TEAM_COLORS[0])
    team2 = np.linalg.norm(color - TEAM_COLORS[1])
    return 0 if team1 < team2 else 1
