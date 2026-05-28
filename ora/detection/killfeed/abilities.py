from typing import Literal

import cv2
import numpy as np

from ...utils.template_matching import find_template_multiscale, MatchConfig, MatchResult
from ...utils.morphology import prepare_template
from ...utils.image import read_image
from ...hero import Hero, Ability, AbilityCategory

_env = Hero(name='Environmental',
            abilities=[Ability(category='ability',
                               name='Environmental',
                               bind=None,
                               icon=read_image(
                                   r"C:\PyProjects\OverwatchDataAnalysis\images\ui\EnvironmentalKill.webp",
                                   cv2.IMREAD_UNCHANGED))])


_regular_ability = MatchConfig(
    threshold=0.7,
    min_height_pct=0.5,
    max_height_pct=0.7,
    scale_steps=None
)

ABILITIES_CONFIG = {
        'ability': _regular_ability,
        'ultimate': MatchConfig(
            threshold=0.7,
            min_height_pct=0.7,
            max_height_pct=0.8,
            scale_steps=None
        ),
        'environmental': MatchConfig(
            threshold=0.7,
            min_height_pct=0.9,
            max_height_pct=1.0,
            scale_steps=2,
        )}


def get_used_ability(frame: np.ndarray, hero: Hero | None,
                     debug: bool = False) -> tuple[Ability, MatchResult] | None:
    category = None
    if hero is None:
        hero = _env
        category = 'environmental'

    ability_mask = _compute_ability_mask(frame, debug=debug)

    all_found = []
    for i, ability in enumerate(hero.abilities):
        _, mask = prepare_template(ability.icon, margin=(0, 0))

        if ability.category == 'ultimate':
            mask = cv2.bitwise_not(mask)

        if debug:
            cv2.imshow(f'ab_{i}', mask)

        found_instances = _match_ability(
            frame=ability_mask,
            ref_mask=mask,
            debug=debug,
            category=category or ability.category,
        )

        for instance in found_instances:
            all_found.append((ability, instance))

    if debug:
        print(all_found[:3])

    if len(all_found) > 1:
        raise ValueError('Detected several abilities in KillFeed!')

    return all_found[0] if all_found else None


def _match_ability(frame: np.ndarray,
                   ref_mask: np.ndarray,
                   category: AbilityCategory | Literal['environmental'],
                   debug: bool = False) -> list[MatchResult]:
    fr_h = frame.shape[0]

    config = ABILITIES_CONFIG.get(category, _regular_ability)
    scale_steps = config.scale_steps or int((config.max_height_pct - config.min_height_pct) * fr_h)

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

    return candidates


def _compute_ability_mask(frame: np.ndarray, debug: bool = False) -> np.ndarray:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(frame_gray, (0, 0), sigmaX=7, sigmaY=7)
    dog = cv2.subtract(frame_gray, blurred)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

    if debug:
        cv2.imshow("dog", dog)

    mask = (dog > 30).astype(np.uint8) * 255

    if debug:
        cv2.imshow("ability_mask", mask)

    return mask
