import cv2
import numpy as np
from collections import Counter, defaultdict

from . import KillFeedEntry
from .arrows import detect_killfeed_arrows
from .entry import _KF_ARROW_REF, KFParseContext

from ...hero import Ability
from ...player import KFPlayer
from ...utils.box_proc import percent_roi_to_pixels
from .presence import detect_killfeed_presence


class KillFeed:
    """Killfeed object containing the info about the killfeed entries throughout the match."""

    def __init__(self, roi: list[float, float, float, float] = (0.73, 0.0, 1.0, 0.25),
                 window_frames: int = 12):
        self.entries: list[KillFeedEntry] = []
        self.roi = roi
        self.window_frames = window_frames

    def _get_roi_image(self, frame: np.ndarray) -> np.ndarray:
        rx, ry, rw, rh = percent_roi_to_pixels(frame_shape=frame.shape[:2], roi_pct=self.roi)
        return frame[ry:ry + rh, rx:rx + rw]

    def deduplicate(self, debug: bool = False) -> None:
        self.entries = _deduplicate_entries(self.entries, self.window_frames, debug=debug)

    def update_from_frame(self, frame: np.ndarray, heroes: list, frame_i: int):
        search_area = self._get_roi_image(frame=frame)
        if not detect_killfeed_presence(search_area):
            return

        image_entries = get_killfeed_entry_images(frame=search_area, ref=_KF_ARROW_REF)

        for row_i, (img, arrow_center, arrow_box) in enumerate(image_entries):
            context = KFParseContext(
                image=img,
                heroes=heroes,
                arrow_center=arrow_center,
                arrow_box=arrow_box,
                frame_i=frame_i,
                row_i=row_i,
            )
            entry = KillFeedEntry.from_image(context)
            if entry is not None:
                self.entries.append(entry)


def get_killfeed_entry_images(frame: np.ndarray, ref: np.ndarray,
                               entry_height_pc: float = 0.15,
                               debug: bool = False) \
        -> list[tuple[np.ndarray, tuple[int, int], tuple[int, ...]]]:
    h, w = frame.shape[:2]

    rx1 = int(w * 0.3)
    rx2 = int(w * 0.8)
    ry1 = int(h * 0.1)

    arrow_frame = frame[ry1:, rx1:rx2]
    arrow_data = detect_killfeed_arrows(arrow_frame, ref, debug=debug)

    if not arrow_data:
        return []

    arrow_data = [
        r._replace(
            box=(r.box[0] + rx1, r.box[1] + ry1, r.box[2] + rx1, r.box[3] + ry1),
            center=(r.center[0] + rx1, r.center[1] + ry1),
        )
        for r in arrow_data
    ]
    arrow_data = sorted(arrow_data, key=lambda r: r.center[1], reverse=True)

    entry_h_halved = int(h * entry_height_pc / 2)

    kf_entries = []
    for i, r in enumerate(arrow_data):
        y = r.center[1]
        entry_cropped = frame[y - entry_h_halved:y + entry_h_halved, 0:w]
        kf_entries.append((entry_cropped, r.center, (r.box[0], 0, r.box[2], entry_h_halved * 2)))
        if debug:
            cv2.imshow(f"cropped_entry_{i}", entry_cropped)

    return kf_entries


def _deduplicate_entries(entries: list[KillFeedEntry], window_frames: int,
                         debug: bool = False) -> list[KillFeedEntry]:
    if not entries:
        return []

    sorted_entries = sorted(entries)
    n = len(sorted_entries)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Pre-compute res barriers: group same-team kill entries by victim within window_frames
    res_ranges = _build_res_ranges(sorted_entries, window_frames)

    # Stage 1: cluster by content within the window, blocked by res barriers
    for i, e1 in enumerate(sorted_entries):
        for j in range(i + 1, n):
            e2 = sorted_entries[j]
            if e2.frame - e1.frame > window_frames:
                break
            if _entries_match(e1, e2) and not _has_res_between(e1, e2, res_ranges):
                union(i, j)

    raw_clusters: dict[int, list[KillFeedEntry]] = defaultdict(list)
    for i, e in enumerate(sorted_entries):
        raw_clusters[find(i)].append(e)

    # Stage 2: split clusters where the same victim appears at multiple rows in one frame
    result = []
    for cluster in raw_clusters.values():
        for sub_cluster in _resolve_resurrections(cluster, debug=debug):
            result.append(_merge_cluster(sub_cluster, debug=debug))

    return sorted(result)


def _entries_match(e1: KillFeedEntry, e2: KillFeedEntry) -> bool:
    if e1.player2 != e2.player2:
        return False
    e1_is_res = e1.player1 is not None and e1.player1.team == e1.player2.team
    e2_is_res = e2.player1 is not None and e2.player1.team == e2.player2.team
    if e1_is_res != e2_is_res:
        return False
    if e1_is_res:
        # Two res entries: a different hero CAN rez in the same window, so player1 must match
        if e1.player1 is None or e2.player1 is None:
            return True
        return e1.player1 == e2.player1
    # Two death entries: same victim can't die to two different killers without a rez — player1 irrelevant
    return True


def _build_res_ranges(sorted_entries: list[KillFeedEntry],
                      window_frames: int) -> dict[KFPlayer, list[tuple[int, int]]]:
    """Group same-team kill entries (resurrections) by victim within window_frames.

    Returns a mapping from victim KFPlayer to a list of (min_frame, max_frame) ranges,
    one range per distinct resurrection event detected.
    """
    res_ranges: dict[KFPlayer, list[tuple[int, int]]] = defaultdict(list)

    by_event: dict[tuple[KFPlayer, KFPlayer], list[KillFeedEntry]] = defaultdict(list)
    for e in sorted_entries:
        if e.player1 is not None and e.player1.team == e.player2.team:
            by_event[(e.player1, e.player2)].append(e)

    for (_, victim), candidates in by_event.items():
        group_start = candidates[0].frame
        group_end = candidates[0].frame
        for e in candidates[1:]:
            if e.frame - group_end <= window_frames:
                group_end = e.frame
            else:
                res_ranges[victim].append((group_start, group_end))
                group_start = group_end = e.frame
        res_ranges[victim].append((group_start, group_end))

    return res_ranges


def _has_res_between(e1: KillFeedEntry, e2: KillFeedEntry,
                     res_ranges: dict[KFPlayer, list[tuple[int, int]]]) -> bool:
    """True if a resurrection of e1's victim is fully enclosed between e1 and e2's frames."""
    for f_start, f_end in res_ranges.get(e1.player2, []):
        if e1.frame < f_start and f_end < e2.frame:
            return True
    return False


def _resolve_resurrections(cluster: list[KillFeedEntry],
                           debug: bool = False) -> list[list[KillFeedEntry]]:
    """Split a cluster if the victim appears at multiple rows in the same frame.

    Row=0 entries at and after the split frame form a new sub-cluster (the new death);
    row>0 entries continue the old sub-cluster (the original death scrolled down).
    Recurses to handle multiple back-to-back resurrections.
    """
    frame_groups: dict[int, list[KillFeedEntry]] = defaultdict(list)
    for e in cluster:
        frame_groups[e.frame].append(e)

    split_frames = sorted(f for f, es in frame_groups.items() if len(es) > 1)
    if not split_frames:
        return [cluster]

    split_frame = split_frames[0]
    if debug:
        print(f'[KillFeed] Resurrection detected at frame {split_frame}: {cluster[0].player2}')

    min_row = min(e.row for e in frame_groups[split_frame])
    old = [e for e in cluster if e.frame < split_frame]
    old += [e for e in cluster if e.frame >= split_frame and e.row > min_row]
    new = [e for e in cluster if e.frame >= split_frame and e.row == min_row]

    result = []
    result.extend(_resolve_resurrections(old, debug=debug))
    result.extend(_resolve_resurrections(new, debug=debug))
    return result


def _merge_cluster(cluster: list[KillFeedEntry], debug: bool = False) -> KillFeedEntry:
    anchor = min(cluster, key=lambda e: (e.frame, e.row))
    n = len(cluster)

    player1 = _majority_player([e.player1 for e in cluster])
    player2 = _majority_player([e.player2 for e in cluster])
    assert player2 is not None, "Cluster has no player2 — should never happen"

    ability_votes = [e.ability for e in cluster if e.ability is not None]
    ability = _majority_ability(ability_votes, n)

    critical_votes = sum(e.is_critical for e in cluster)
    if critical_votes * 2 == n:
        print(f'[KillFeed] is_critical vote tie in cluster of {n} — defaulting to False')
    is_critical = critical_votes > n / 2

    env_votes = sum(e.is_environmental for e in cluster)
    if env_votes * 2 == n:
        print(f'[KillFeed] is_environmental vote tie in cluster of {n} — defaulting to False')
    is_environmental = env_votes > n / 2

    all_assist_heroes = {a.hero.name: a.hero for e in cluster for a in e.assists}
    assists = []
    for hero_name, hero in all_assist_heroes.items():
        count = sum(1 for e in cluster if any(a.hero.name == hero_name for a in e.assists))
        if count > n / 2:
            teams = [a.team for e in cluster for a in e.assists if a.hero.name == hero_name]
            assists.append(KFPlayer(team=Counter(teams).most_common(1)[0][0], hero=hero))

    merged = KillFeedEntry(
        frame=anchor.frame,
        row=anchor.row,
        player1=player1,
        player2=player2,
        assists=assists,
        ability=ability,
        is_critical=is_critical,
        is_environmental=is_environmental,
    )

    if debug and merged.player1 is not None and merged.player1.team == merged.player2.team:
        print(f'[KillFeed] Resurrection entry detected: {merged}')

    return merged


def _majority_player(players: list[KFPlayer | None]) -> KFPlayer | None:
    non_none = [p for p in players if p is not None]
    if not non_none:
        return None
    key_to_player: dict[tuple, KFPlayer] = {}
    counts: Counter = Counter()
    for p in non_none:
        k = (p.team, p.hero.name)
        counts[k] += 1
        key_to_player[k] = p
    return key_to_player[counts.most_common(1)[0][0]]


def _majority_ability(abilities: list[Ability], n: int) -> Ability | None:
    if not abilities:
        return None
    key_to_ability: dict[tuple, Ability] = {}
    counts: Counter = Counter()
    for a in abilities:
        k = (a.category, a.name)
        counts[k] += 1
        key_to_ability[k] = a
    top_key, top_count = counts.most_common(1)[0]
    if top_count > n / 2:
        return key_to_ability[top_key]
    if top_count * 2 == n:
        print(f'[KillFeed] Ability vote tie ({key_to_ability[top_key].name} {top_count}/{n}) — result is None')
    return None
