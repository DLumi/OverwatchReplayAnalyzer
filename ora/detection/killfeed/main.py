import cv2
import numpy as np

from dataclasses import dataclass, field as dc_field

from ...player import KFPlayer
from ...utils.template_matching import percent_roi_to_pixels
from .presence import detect_killfeed_presence
from .arrows import get_killfeed_entry_images
from .heroes import detect_heroes_2d
from .abilities import get_used_ability
from ...hero import Hero, Ability

_KF_ARROW_REF = cv2.imread(r"C:\PyProjects\OverwatchDataAnalysis\images\ui\killfeed_arrow2.png",
                           cv2.IMREAD_GRAYSCALE)


@dataclass
class KFParseContext:
    image: np.ndarray
    heroes: list[Hero]
    arrow_center: tuple[int, int]
    arrow_box: tuple[int, ...]
    frame_i: int
    row_i: int
    killer_team: int | None = None
    killed_box: tuple[int, ...] | None = None
    killer_box: tuple[int, ...] | None = None


@dataclass(frozen=False)
class KillFeedEntry:
    """Contains info about one killfeed row."""

    frame: int
    row: int

    player1: KFPlayer | None
    player2: KFPlayer
    assists: list[KFPlayer] = dc_field(default_factory=list)

    ability: Ability | None = None
    is_critical: bool = False
    is_environmental: bool = False

    @staticmethod
    def _players_equal(p1, p2):
        if p1 is None or p2 is None:
            return p1 is p2
        return (
                p1.hero == p2.hero
                and p1.team == p2.team
        )

    def __eq__(self, other):
        return (
                self._players_equal(self.player1, other.player1)
                and self._players_equal(self.player2, other.player2)
        )

    def __gt__(self, other):
        return (self.frame > other.frame) or (self.frame == other.frame and self.row > other.row)

    def __repr__(self):
        maybe_p1 = self.player1.hero if self.player1 else ''
        maybe_critical = '🎯' if self.is_critical else ''
        maybe_ability = self.ability if self.ability is not None else ''
        maybe_assists = '++[' + ''.join([f'{x.hero},' for x in self.assists]) + ']' \
            if self.assists else ''
        maybe_environmental = '🤸' if self.is_environmental else ''
        return (f'#{self.row}: {maybe_p1}{maybe_assists} '
                f'{maybe_ability}{maybe_critical}➜{maybe_environmental} {self.player2.hero}')

    @classmethod
    def from_image(cls, context: KFParseContext):

        # detect main kill pair
        killfeed_entry, context = cls._detect_main_part(context)

        if killfeed_entry is None:
            return None

        # process left side: assists / ability / critical
        if killfeed_entry.player1:
            killfeed_entry._detect_left_modifiers(context, debug=False)

        # process right side: environmental
        killfeed_entry._detect_right_modifiers(context)

        return killfeed_entry

    @classmethod
    def _detect_main_part(cls, context: KFParseContext, debug: bool = False):

        found_main = detect_heroes_2d(frame=context.image,
                                      hero_list=context.heroes,
                                      portrait_type='kf_main',
                                      debug=debug)
        roles = assign_roles(found_main, context.arrow_center, debug=debug)

        if not found_main:
            print(f'F#{context.frame_i}:{context.row_i} Heroes not found!')
            return None, context

        killed_hero, _, killed_hero_box, killed_hero_team = roles['object']
        killfeed_entry = cls(frame=context.frame_i,
                             row=context.row_i,
                             player1=None,
                             player2=KFPlayer(team=killed_hero_team,
                                              hero=killed_hero))

        context.killed_box = killed_hero_box

        if 'subject' not in roles:
            return killfeed_entry, context

        killer, _, killer_box, killer_team = roles['subject']
        context.killer_box = killer_box
        context.killer_team = killer_team

        killfeed_entry.player1 = KFPlayer(team=killer_team, hero=killer)
        return killfeed_entry, context

    def _detect_left_modifiers(self, context: KFParseContext, debug: bool = False):
        arrow_box_x, _, arrow_box_w, _ = context.arrow_box

        killer_box_x, _, killer_box_w, _ = context.killer_box
        right_side = killer_box_x + killer_box_w

        if (arrow_box_x - right_side) < arrow_box_w * 2.5:
            return

        special_roi = context.image[:, right_side:arrow_box_x]

        found_assist = detect_heroes_2d(frame=special_roi, hero_list=context.heroes,
                                        portrait_type='kf_assist', debug=True)

        if found_assist:
            found_assist = [(hero, center, box, context.killer_team) for (hero, center, box, team) in found_assist]
            self.assists = [KFPlayer(team=team, hero=hero) for (hero, _, _, team) in found_assist]

            _, _, assist_box, _ = found_assist[-1]
            assist_box_x, _, assist_box_w, _ = assist_box
            right_side = assist_box_x + assist_box_w
            region_right_bound = special_roi.shape[1]

            if (region_right_bound - right_side) < arrow_box_w * 1.5:
                return

            special_roi = special_roi[:, right_side:arrow_box_x]

        found_ability = get_used_ability(frame=special_roi, hero=self.player1.hero, debug=False)

        if found_ability:
            ability, _, ability_box, _ = found_ability
            self.ability = ability

            ability_box_x, _, ability_box_w, _ = ability_box
            right_side = ability_box_x + ability_box_w
            region_right_bound = special_roi.shape[1]

            if (region_right_bound - right_side) < arrow_box_w:
                return
            else:
                self.is_critical = True
        else:
            self.is_critical = True

    def _detect_right_modifiers(self, context: KFParseContext):
        arrow_box_x, _, arrow_box_w, _ = context.arrow_box
        arrow_box_right_side = arrow_box_x + arrow_box_w

        killer_box_x, _, _, _ = context.killed_box

        if (killer_box_x - arrow_box_right_side) < arrow_box_w * 2:
            return

        special_roi = context.image[:, arrow_box_right_side:killer_box_x]

        found_environmental = get_used_ability(frame=special_roi, hero=None, debug=False)

        if found_environmental:
            self.is_environmental = True


def assign_roles(heroes: list, arrow_center: tuple[int, int],
                 debug: bool = False) -> dict:
    if len(heroes) > 2:
        if debug:
            print(heroes)
        raise ValueError('Detected more than 2 heroes in KillFeed!')

    roles = {}
    for hero in heroes:
        hero_center_x = hero[1][0]
        role = 'subject' if hero_center_x < arrow_center[0] else 'object'
        if role in roles:
            print('WARNING! Overwriting roles')
        roles[role] = hero

    return roles


class KillFeed:
    """Killfeed object containing the info about the killfeed entries throughout the match."""

    def __init__(self, roi: list[float, float, float, float] = (0.73, 0.0, 1.0, 0.25)):
        self.entries = []
        self.roi = roi

    def _get_roi_image(self, frame: np.ndarray) -> np.ndarray:
        rx, ry, rw, rh = percent_roi_to_pixels(frame_shape=frame.shape[:2], roi_pct=self.roi)
        return frame[ry:ry + rh, rx:rx + rw]

    def update_from_frame(self, frame: np.ndarray):
        search_area = self._get_roi_image(frame=frame)
        if not detect_killfeed_presence(search_area):
            return

        image_entries = get_killfeed_entry_images(frame=search_area, ref=_KF_ARROW_REF)

        for img, arrow_center, arrow_box in image_entries:
            self.entries.append(KillFeedEntry.from_image(image=img,
                                                         arrow_center=arrow_center,
                                                         arrow_box=arrow_box,
                                                         heroes=None,
                                                         frame_i=None,
                                                         row_i=None))
