from typing import Literal

import cv2
import numpy as np

from .teams import get_team_idx
from ...utils.template_matching import find_template_multiscale, MatchConfig, MatchResult
from ...utils.box_proc import percent_roi_to_pixels
from ...utils.morphology import prepare_template
from ...hero import Hero

KF_MAIN_CONFIG = MatchConfig(
    min_height_pct=0.8,
    max_height_pct=0.9,
    ncc_threshold=0.3,
    threshold=0.92,
)

KF_ASSIST_CONFIG = MatchConfig(
    min_height_pct=0.6,
    max_height_pct=0.8,
    ncc_threshold=0.45,
    threshold=0.88,
)

_PORTRAIT_PARAMS: dict[str, tuple[MatchConfig, tuple[int, int]]] = {
    'kf_main':   (KF_MAIN_CONFIG,   (50, 50)),
    'kf_assist': (KF_ASSIST_CONFIG, (20, 50)),
}


def detect_heroes_2d(frame: np.ndarray,
                     hero_list: list[Hero],
                     portrait_type: Literal['kf_main', 'kf_assist'],
                     debug: bool = False) -> list[tuple[Hero, MatchResult, int]]:
    if portrait_type not in _PORTRAIT_PARAMS:
        raise ValueError(f'portrait_type {portrait_type!r} not supported')

    config, margin = _PORTRAIT_PARAMS[portrait_type]
    fr_h = frame.shape[0]
    steps = int((config.max_height_pct - config.min_height_pct) * fr_h)
    method = cv2.TM_CCORR_NORMED
    threshold = config.threshold

    search_params = {
        'min_height_pct': config.min_height_pct,
        'max_height_pct': config.max_height_pct,
        'scale_steps': steps,
        'ncc_threshold': config.ncc_threshold,
    }

    rx, ry, rw, rh = percent_roi_to_pixels(frame.shape, (0.08, 0.0, 0.92, 1.0))
    search_space = frame[ry:ry + rh, rx:rx + rw]

    detected_instances = []
    for hero in hero_list:
        template, mask = prepare_template(hero.portrait2d, margin=margin)
        _method = cv2.TM_CCOEFF_NORMED if mask is None else method
        _threshold = 0.8 if mask is None else threshold

        found_instances = find_template_multiscale(search_space, template, **search_params,
                                                   mask_base=mask,
                                                   method=_method,
                                                   threshold=_threshold)
        if found_instances:
            detected_instances.extend((hero, r) for r in found_instances)

    detected_heroes = []
    for hero, instance in detected_instances:
        if debug:
            cv2.circle(frame, instance.center, radius=3,
                       thickness=5, color=(0, 255, 0), lineType=cv2.LINE_AA)

        x1, y1, x2, y2 = instance.box
        instance = instance._replace(
            box=(x1 + rx, y1, x2 + rx, y2),
            center=(instance.center[0] + rx, instance.center[1]),
        )

        team = get_team_idx(frame, instance.box, debug=debug, hero=hero) \
            if portrait_type == 'kf_main' else -1

        detected_heroes.append((hero, instance, team))

    return sorted(detected_heroes, key=lambda x: x[1].center[0])
