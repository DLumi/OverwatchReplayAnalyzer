from dataclasses import dataclass, field as dc_field, fields as dc_fields

import numpy as np

from .abilities import get_used_ability
from .heroes import detect_heroes_2d

from ...hero import Hero, Ability
from ...player import KFPlayer


_TEAM_COLOR = {1: '\033[94m', 0: '\033[91m'}  # bright blue, bright red
_RESET = '\033[0m'


def _c(text: str, team: int) -> str:
    return f"{_TEAM_COLOR.get(team, '')}{text}{_RESET}"


@dataclass
class KFParseContext:
    image: np.ndarray
    heroes: list[Hero]
    arrow_center: tuple[int, int]
    arrow_box: tuple[int, int, int, int]
    frame_i: int
    row_i: int
    killer_team: int | None = None
    killed_box: tuple[int, int, int, int] | None = None
    killer_box: tuple[int, int, int, int] | None = None


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
        maybe_p1 = _c(str(self.player1.hero), self.player1.team) if self.player1 else ''
        maybe_critical = '🎯' if self.is_critical else ''
        maybe_ability = self.ability if self.ability is not None else ''
        maybe_assists = '++[' + ''.join([f'{_c(str(x.hero), x.team)},' for x in self.assists]) + ']' \
            if self.assists else ''
        maybe_environmental = '🤸' if self.is_environmental else ''
        p2 = _c(str(self.player2.hero), self.player2.team)
        return (f'#{self.frame}:{self.row}: {maybe_p1}{maybe_assists} '
                f'{maybe_ability}{maybe_critical}➜{maybe_environmental} {p2}')

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

        if not found_main:
            print(f'F#{context.frame_i}:{context.row_i} Heroes not found!')
            return None, context

        roles = assign_roles(found_main, context.arrow_center, debug=debug)

        if 'object' not in roles:
            print(f'F#{context.frame_i}:{context.row_i} No object role assigned!')
            return None, context

        killed_hero, killed_match, killed_hero_team = roles['object']
        killfeed_entry = cls(frame=context.frame_i,
                             row=context.row_i,
                             player1=None,
                             player2=KFPlayer(team=killed_hero_team,
                                              hero=killed_hero))

        context.killed_box = killed_match.box

        if 'subject' not in roles:
            return killfeed_entry, context

        killer, killer_match, killer_team = roles['subject']
        context.killer_box = killer_match.box
        context.killer_team = killer_team

        killfeed_entry.player1 = KFPlayer(team=killer_team, hero=killer)
        return killfeed_entry, context

    def _detect_left_modifiers(self, context: KFParseContext, debug: bool = False):
        arrow_box_x, _, arrow_box_x2, _ = context.arrow_box
        arrow_box_w = arrow_box_x2 - arrow_box_x

        killer_box_x, _, killer_box_x2, _ = context.killer_box
        right_side = killer_box_x2

        if (arrow_box_x - right_side) < arrow_box_w * 2.5:
            return

        special_roi = context.image[:, right_side:arrow_box_x]

        found_assist = detect_heroes_2d(frame=special_roi, hero_list=context.heroes,
                                        portrait_type='kf_assist', debug=debug)

        if found_assist:
            found_assist = [(hero, match, context.killer_team) for (hero, match, team) in found_assist]
            self.assists = [KFPlayer(team=team, hero=hero) for (hero, _, team) in found_assist]

            _, assist_match, _ = found_assist[-1]
            assist_box_x, assist_box_x2 = assist_match.box[0], assist_match.box[2]
            right_side = assist_box_x2
            region_right_bound = special_roi.shape[1]

            if (region_right_bound - right_side) < arrow_box_w * 1.5:
                return

            special_roi = special_roi[:, right_side:arrow_box_x]

        found_ability = get_used_ability(frame=special_roi, hero=self.player1.hero, debug=False)

        if found_ability:
            ability, match = found_ability
            self.ability = ability

            ability_box_x, _, ability_box_x2, _ = match.box
            right_side = ability_box_x2
            region_right_bound = special_roi.shape[1]

            if (region_right_bound - right_side) < arrow_box_w:
                return
            else:
                self.is_critical = True
        else:
            self.is_critical = True

    def _detect_right_modifiers(self, context: KFParseContext):
        arrow_box_x, _, arrow_box_x2, _ = context.arrow_box
        arrow_box_w = arrow_box_x2 - arrow_box_x
        arrow_box_right_side = arrow_box_x2

        killer_box_x, _, _, _ = context.killed_box

        if (killer_box_x - arrow_box_right_side) < arrow_box_w * 2:
            return

        special_roi = context.image[:, arrow_box_right_side:killer_box_x]

        found_environmental = get_used_ability(frame=special_roi, hero=None, debug=False)

        if found_environmental:
            self.is_environmental = True

    def as_dict(self):
        return {'frame': self.frame,
                'row': self.row,
                'player1': self.player1.as_dict() if self.player1 is not None else None,
                'player2': self.player2.as_dict() if self.player2 is not None else None,
                'assists': [x.as_dict() for x in self.assists],
                'ability': self.ability.name if self.ability is not None else None,
                'is_critical': self.is_critical,
                'is_environmental': self.is_environmental}


def assign_roles(heroes: list, arrow_center: tuple[int, int],
                 debug: bool = False) -> dict:
    if len(heroes) > 2:
        if debug:
            print(heroes)
        raise ValueError('Detected more than 2 heroes in KillFeed!')

    roles = {}
    for hero in heroes:
        hero_center_x = hero[1].center[0]
        role = 'subject' if hero_center_x < arrow_center[0] else 'object'
        if role in roles:
            print('WARNING! Overwriting roles')
        roles[role] = hero

    return roles
