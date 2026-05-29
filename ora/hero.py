from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .utils.image import read_image


AbilityCategory = Literal['weapon', 'ability', 'passive', 'ultimate']


@dataclass(frozen=True)
class Ability:
    category: AbilityCategory
    name: str
    icon: np.ndarray
    bind: str | None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ability):
            return NotImplemented
        return self.category == other.category and self.name == other.name

    def __hash__(self) -> int:
        return hash((self.category, self.name))

    def __repr__(self):
        n = self.name if self.name != 'Quick melee' else '✊'
        return f'Ability({n})'


@dataclass(frozen=True)
class Hero:
    name: str

    portrait2d: np.ndarray | None = None
    portrait3d: np.ndarray | None = None

    abilities: list[Ability] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hero):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self):
        return f'Hero({self.name})'


def populate_heroes():
    hero_images = {x.stem.rsplit('_', 1)[0]: read_image(x, flags=cv2.IMREAD_UNCHANGED) for x in
                   Path(r'C:\PyProjects\OverwatchDataAnalysis\images\heroes').iterdir()}

    qm = Ability(category='ability',
                 name='Quick melee',
                 icon=read_image(r"C:\PyProjects\OverwatchDataAnalysis\images\heroes_new\Quick_melee.webp",
                                 cv2.IMREAD_UNCHANGED),
                 bind='V')

    hero_list = []
    for hero in Path(r'C:\PyProjects\OverwatchDataAnalysis\images\heroes_new').iterdir():

        if not hero.is_dir():
            continue

        abilities = []
        for ability in (hero / 'abilities').iterdir():
            icon = read_image(ability, flags=cv2.IMREAD_UNCHANGED)
            ab_name, cat_raw, bind = ability.stem.split('@')
            if 'weapon' in cat_raw.lower():
                category = 'weapon'
            elif 'passive' in cat_raw.lower():
                category = 'passive'
            elif 'ultimate' in cat_raw.lower():
                category = 'ultimate'
            else:
                category = cat_raw.lower()
            abilities.append(Ability(category=category, name=ab_name, icon=icon, bind=bind))

        abilities.append(qm)

        hero_list.append(Hero(name=hero.name, abilities=abilities,
                              portrait2d=hero_images['_'.join(hero.name.split())]))

    return hero_list
