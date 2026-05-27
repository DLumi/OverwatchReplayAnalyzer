import cv2
import numpy as np

from dataclasses import dataclass, field as dc_field

from .player import KFPlayer
from .utils.template_matching import percent_roi_to_pixels, find_template_multiscale
from .utils.killfeed_detection import detect_killfeed_presence
from .utils.killfeed_processor import get_kilfeed_entry_images, assign_roles, get_used_ability, detect_heroes_2d
from .hero import Hero, Ability

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
    """Class of a Killfeed object.

    Contains info about one killfeed row. """

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
        # print(roles)

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
            # print('No assist or ability')
            return killfeed_entry, context

        # cv2.waitKey(0)
        killer, _, killer_box, killer_team = roles['subject']
        context.killer_box = killer_box
        context.killer_team = killer_team

        killfeed_entry.player1 = KFPlayer(team=killer_team, hero=killer)
        return killfeed_entry, context

    def _detect_left_modifiers(self, context: KFParseContext, debug: bool = False):
        arrow_box_x, _, arrow_box_w, _ = context.arrow_box

        # if space between killer hero portrait and arrow is higher than arrow width
        # => there must be assist / ability / critical / environmental modifier
        killer_box_x, _, killer_box_w, _ = context.killer_box
        right_side = killer_box_x + killer_box_w

        if (arrow_box_x - right_side) < arrow_box_w * 2.5:
            # print('No assist or ability')
            return

        # limiting the search space to increase chances of a good match
        special_roi = context.image[:, right_side:arrow_box_x]

        # cv2.imshow('special_roi', special_roi)

        found_assist = detect_heroes_2d(frame=special_roi, hero_list=context.heroes,
                                        portrait_type='kf_assist', debug=True)

        if found_assist:
            # reassign team colors for assists, cause detection by color is inconsistent,
            # and only killer can have assists anyway
            found_assist = [(hero, center, box, context.killer_team) for (hero, center, box, team) in found_assist]

            self.assists = [KFPlayer(team=team, hero=hero) for (hero, _, _, team) in found_assist]

            # cropping assists out of the search area to see if we need to look for any other modifiers
            _, _, assist_box, _ = found_assist[-1]  # furthest right
            assist_box_x, _, assist_box_w, _ = assist_box
            right_side = assist_box_x + assist_box_w
            region_right_bound = special_roi.shape[1]

            # print(region_right_bound - right_side, arrow_box_w)

            if (region_right_bound - right_side) < arrow_box_w * 1.5:
                # print('No ability, no critical')
                return

            special_roi = special_roi[:, right_side:arrow_box_x]
            # cv2.imshow('special_roi', special_roi)

        # --------------------------------------------------
        # Detecting ability / ultimate used
        # --------------------------------------------------

        # print('-----------------')
        found_ability = get_used_ability(frame=special_roi, hero=self.player1.hero, debug=False)

        if found_ability:
            ability, _, ability_box, _ = found_ability
            self.ability = ability

            # cropping ability out of the search area to see if we need to look for any other modifiers
            ability_box_x, _, ability_box_w, _ = ability_box
            right_side = ability_box_x + ability_box_w
            region_right_bound = special_roi.shape[1]

            # special_roi = special_roi[:, right_side:arrow_box_x]
            # cv2.imshow('special_roi', special_roi)

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
            # print('No environmental')
            return

        # limiting the search space to increase chances of a good match
        special_roi = context.image[:, arrow_box_right_side:killer_box_x]

        found_environmental = get_used_ability(frame=special_roi, hero=None, debug=False)

        if found_environmental:
            self.is_environmental = True

        return


class KillFeed:
    """Killfeed object containing the info about the killfeed entries throughout the match"""

    def __init__(self, roi: list[float, float, float, float] = (0.73, 0.0, 1.0, 0.25)):
        """Killfeed constructor

        Args:
            roi: XYXY coordinates to look for killfeed"""

        self.entries = []
        self.roi = roi

    def _get_roi_image(self, frame: np.ndarray) -> np.ndarray:
        rx, ry, rw, rh = percent_roi_to_pixels(frame_shape=frame.shape[:2], roi_pct=self.roi)
        return frame[ry:ry + rh, rx:rx + rw]

    def update_from_frame(self, frame: np.ndarray):
        search_area = self._get_roi_image(frame=frame)
        if not detect_killfeed_presence(search_area):
            return

        image_entries = get_kilfeed_entry_images(frame=search_area, ref=_KF_ARROW_REF)

        for img, arrow_center, arrow_box in image_entries:
            self.entries.append(KillFeedEntry.from_image(image=img,
                                                         arrow_center=arrow_center,
                                                         arrow_box=arrow_box,
                                                         heroes=None,
                                                         frame_i=None,
                                                         row_i=None))
