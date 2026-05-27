from typing import Literal

import cv2
import numpy as np

from ...utils.template_matching import find_template_multiscale, MatchConfig
from ...utils.morphology import prepare_2d_portrait
from ...utils.image import read_image
from ...hero import Hero, Ability, AbilityCategory

_env = Hero(name='Environmental',
            abilities=[Ability(category='ability',
                               name='Environmental',
                               bind=None,
                               icon=read_image(
                                   r"C:\PyProjects\OverwatchDataAnalysis\images\ui\EnvironmentalKill.webp",
                                   cv2.IMREAD_UNCHANGED))])

ABILITY_MATCH_CONFIG = MatchConfig(
    threshold=0.7,
    min_height_pct=0.5,
    max_height_pct=0.7,
)

ULT_MATCH_CONFIG = MatchConfig(
    threshold=0.7,
    min_height_pct=0.7,
    max_height_pct=0.8,
)

ENV_MATCH_CONFIG = MatchConfig(
    threshold=0.7,
    min_height_pct=0.9,
    max_height_pct=1.0,
    scale_steps=2,
)


def get_used_ability(frame: np.ndarray, hero: Hero | None,
                     debug: bool = False) -> Ability | None:
    category = None
    if not hero:
        hero = _env
        category = 'environmental'

    prep = preprocess_for_ability_matching(frame, debug=debug)

    all_found = []
    for i, ability in enumerate(hero.abilities):
        _, mask = prepare_2d_portrait(ability.icon, margin=(0, 0))

        if ability.category == 'ultimate':
            mask = cv2.bitwise_not(mask)

        if debug:
            cv2.imshow(f'ab_{i}', mask)

        found_instances = detect_ability(
            frame=prep,
            ref_mask=mask,
            debug=debug,
            category=category or ability.category,
        )

        for instance in found_instances:
            all_found.append((ability, *instance))

    if debug:
        print(all_found[:3])

    if len(all_found) > 1:
        raise ValueError('Detected several abilities in KillFeed!')

    return all_found[0] if all_found else None


def detect_ability(frame: np.ndarray,
                   ref_mask: np.ndarray,
                   category: AbilityCategory | Literal['environmental'],
                   debug: bool = False) \
        -> list[tuple[tuple[int, int], tuple[int, ...], float]]:
    fr_h = frame.shape[0]

    if category == 'environmental':
        config = ENV_MATCH_CONFIG
        scale_steps = config.scale_steps
    elif category == 'ultimate':
        config = ULT_MATCH_CONFIG
        scale_steps = int(fr_h * config.max_height_pct) - int(fr_h * config.min_height_pct)
    else:
        config = ABILITY_MATCH_CONFIG
        scale_steps = int(fr_h * config.max_height_pct) - int(fr_h * config.min_height_pct)

    candidates = find_template_multiscale(
        frame=frame,
        template_base=ref_mask,
        min_height_pct=config.min_height_pct,
        max_height_pct=config.max_height_pct,
        scale_steps=scale_steps,
        threshold=config.threshold,
    )

    if not candidates:
        ref_mask = cv2.flip(ref_mask, 1)
        candidates = find_template_multiscale(
            frame=frame,
            template_base=ref_mask,
            min_height_pct=config.min_height_pct,
            max_height_pct=config.max_height_pct,
            scale_steps=scale_steps,
            threshold=config.threshold,
        )

    if not candidates:
        return []

    centers = []
    for res in candidates:
        x, y, w, h = res['box']
        center = int(x + w / 2), int(y + h / 2)
        centers.append((center, res['box'], res['score']))

    return centers


def preprocess_for_ability_matching(frame: np.ndarray, debug: bool = False) -> np.ndarray:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray_blurred = cv2.GaussianBlur(frame_gray, (0, 0), sigmaX=7, sigmaY=7)
    search_area = cv2.subtract(frame_gray, roi_gray_blurred)
    roi_gray_blurred = cv2.normalize(search_area, None, 0, 255, cv2.NORM_MINMAX)

    if debug:
        cv2.imshow("roi_gray_blurred", roi_gray_blurred)

    roi_edges_filled = (roi_gray_blurred > 30).astype(np.uint8) * 255

    if debug:
        cv2.imshow("roi_filled", roi_edges_filled)

    return roi_edges_filled
