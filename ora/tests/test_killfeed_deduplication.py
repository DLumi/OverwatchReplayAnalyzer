import numpy as np
import pytest
from pytest_check import check

from ora.detection.killfeed.entry import KillFeedEntry
from ora.detection.killfeed.main import (
    _deduplicate_entries,
    _entries_match,
    _resolve_resurrections,
)
from ora.hero import Ability, Hero
from ora.player import KFPlayer

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_WIDOW = Hero(name='Widowmaker')
_REAPER = Hero(name='Reaper')
_MERCY = Hero(name='Mercy')
_ANA = Hero(name='Ana')
_LUCIO = Hero(name='Lucio')

_DUMMY_ICON = np.zeros((10, 10), dtype=np.uint8)
_ABILITY_A = Ability(category='weapon', name='Widow Kiss', icon=_DUMMY_ICON, bind='M1')
_ABILITY_B = Ability(category='ability', name='Venom Mine', icon=_DUMMY_ICON, bind='E')


def _entry(frame: int, row: int,
           p1_hero: Hero | None = _REAPER, p1_team: int = 0,
           p2_hero: Hero = _WIDOW, p2_team: int = 1,
           is_critical: bool = False,
           is_environmental: bool = False,
           ability: Ability | None = None,
           assists: list[KFPlayer] | None = None) -> KillFeedEntry:
    player1 = KFPlayer(team=p1_team, hero=p1_hero) if p1_hero is not None else None
    player2 = KFPlayer(team=p2_team, hero=p2_hero)
    return KillFeedEntry(
        frame=frame, row=row,
        player1=player1, player2=player2,
        is_critical=is_critical,
        is_environmental=is_environmental,
        ability=ability,
        assists=assists or [],
    )


# ---------------------------------------------------------------------------
# _entries_match
# ---------------------------------------------------------------------------

def test_match_identical():
    check.is_true(_entries_match(_entry(0, 0), _entry(1, 0)),
                  "identical entries should match")


def test_match_none_p1_is_compatible():
    e1 = _entry(0, 0, p1_hero=None)
    e2 = _entry(1, 0, p1_hero=_REAPER)
    check.is_true(_entries_match(e1, e2),
                  "None player1 should match any killer for the same victim")
    check.is_true(_entries_match(e2, e1),
                  "_entries_match should be symmetric for None player1")


def test_death_and_res_entry_do_not_match():
    death = _entry(0, 0, p1_hero=_REAPER, p1_team=0, p2_hero=_WIDOW, p2_team=1)
    res   = _entry(1, 0, p1_hero=_MERCY,  p1_team=1, p2_hero=_WIDOW, p2_team=1)
    check.is_false(_entries_match(death, res),
                   "death entry must not match res entry even for the same victim")
    check.is_false(_entries_match(res, death),
                   "res/death mismatch must be symmetric")


def test_no_match_different_victim():
    e1 = _entry(0, 0, p2_hero=_WIDOW)
    e2 = _entry(1, 0, p2_hero=_MERCY)
    check.is_false(_entries_match(e1, e2),
                   "entries with different victims must not match")


def test_match_different_killer_same_victim():
    # Different killers, same victim — misdetection scenario; should merge and majority-vote killer
    e1 = _entry(0, 0, p1_hero=_REAPER)
    e2 = _entry(1, 0, p1_hero=_ANA)
    check.is_true(_entries_match(e1, e2),
                  "different killers with same victim should match — misdetection case")


# ---------------------------------------------------------------------------
# Basic deduplication
# ---------------------------------------------------------------------------

def test_basic_dedup():
    entries = [_entry(f, 0) for f in range(8)]
    result = _deduplicate_entries(entries, window_frames=12)
    assert len(result) == 1, "8 identical entries should collapse to 1"
    check.equal(result[0].frame, 0, "merged entry should anchor to first observed frame")
    check.equal(result[0].row, 0, "merged entry should anchor to first observed row")


def test_empty():
    check.equal(_deduplicate_entries([], window_frames=12), [],
                "empty input should return empty list")


def test_outside_window_produces_two_entries():
    entries = [_entry(0, 0), _entry(20, 0)]
    result = _deduplicate_entries(entries, window_frames=12)
    check.equal(len(result), 2,
                "entries more than window_frames apart should not merge")


def test_two_distinct_events_not_merged():
    # Reaper kills Widow at frame 0, Ana kills Mercy at frame 5
    entries = [
        _entry(0, 0, p1_hero=_REAPER, p2_hero=_WIDOW),
        _entry(1, 0, p1_hero=_REAPER, p2_hero=_WIDOW),
        _entry(5, 0, p1_hero=_ANA, p2_hero=_MERCY),
        _entry(6, 0, p1_hero=_ANA, p2_hero=_MERCY),
    ]
    result = _deduplicate_entries(entries, window_frames=12)
    check.equal(len(result), 2,
                "events with different victims must not merge")


def test_output_sorted_by_frame_then_row():
    entries = [
        _entry(10, 0, p1_hero=_REAPER, p2_hero=_WIDOW),
        _entry(5, 1, p1_hero=_ANA, p2_hero=_MERCY),
        _entry(5, 0, p1_hero=_LUCIO, p2_hero=_REAPER, p2_team=0, p1_team=1),
    ]
    result = _deduplicate_entries(entries, window_frames=12)
    check.equal(len(result), 3, "three distinct events should produce three entries")
    frames_rows = [(e.frame, e.row) for e in result]
    check.equal(frames_rows, sorted(frames_rows),
                "output must be sorted by (frame, row)")


# ---------------------------------------------------------------------------
# Majority voting — scalar fields
# ---------------------------------------------------------------------------

def test_critical_majority_true():
    entries = [_entry(f, 0, is_critical=(f < 4)) for f in range(6)]  # 4 True, 2 False
    result = _deduplicate_entries(entries, window_frames=12)
    check.is_true(result[0].is_critical,
                  "is_critical should be True when 4 of 6 frames vote True")


def test_critical_majority_false():
    entries = [_entry(f, 0, is_critical=(f < 2)) for f in range(6)]  # 2 True, 4 False
    result = _deduplicate_entries(entries, window_frames=12)
    check.is_false(result[0].is_critical,
                   "is_critical should be False when only 2 of 6 frames vote True")


def test_environmental_majority():
    entries = [_entry(f, 0, is_environmental=(f < 4)) for f in range(6)]
    result = _deduplicate_entries(entries, window_frames=12)
    check.is_true(result[0].is_environmental,
                  "is_environmental should be True when 4 of 6 frames vote True")


def test_ability_majority():
    entries = (
        [_entry(f, 0, ability=_ABILITY_A) for f in range(4)] +
        [_entry(f + 4, 0, ability=_ABILITY_B) for f in range(2)]
    )
    result = _deduplicate_entries(entries, window_frames=12)
    check.is_not_none(result[0].ability,
                      "ability should be set when present in majority of frames")
    check.equal(result[0].ability.name, _ABILITY_A.name,
                "ability with the most votes should win")


def test_ability_none_when_minority():
    # 2 frames detect an ability, 4 don't → ability appears in minority → result is None
    entries = (
        [_entry(f, 0, ability=_ABILITY_A) for f in range(2)] +
        [_entry(f + 2, 0) for f in range(4)]
    )
    result = _deduplicate_entries(entries, window_frames=12)
    check.is_none(result[0].ability,
                  "ability appearing in minority of frames should be suppressed")


# ---------------------------------------------------------------------------
# player1=None (suicide→killer reveal)
# ---------------------------------------------------------------------------

def test_suicide_to_killer_reveals_player1():
    entries = (
        [_entry(f, 0, p1_hero=None) for f in range(2)] +
        [_entry(f + 2, 0, p1_hero=_REAPER) for f in range(4)]
    )
    result = _deduplicate_entries(entries, window_frames=12)
    assert len(result) == 1, "None-killer and Reaper entries for same victim should merge"
    check.is_not_none(result[0].player1,
                      "killer should be resolved from non-None majority")
    check.equal(result[0].player1.hero.name, _REAPER.name,
                "majority-voted killer should be Reaper")


def test_all_none_player1_stays_none():
    entries = [_entry(f, 0, p1_hero=None) for f in range(5)]
    result = _deduplicate_entries(entries, window_frames=12)
    assert len(result) == 1, "5 matching None-killer entries should merge into 1"
    check.is_none(result[0].player1,
                  "player1 should remain None when all frames have no killer detected")


# ---------------------------------------------------------------------------
# Assists
# ---------------------------------------------------------------------------

def test_assist_above_threshold_included():
    # Lucio assists in 4 of 6 frames (>50%)
    lucio = KFPlayer(team=0, hero=_LUCIO)
    entries = (
        [_entry(f, 0, assists=[lucio]) for f in range(4)] +
        [_entry(f + 4, 0, assists=[]) for f in range(2)]
    )
    result = _deduplicate_entries(entries, window_frames=12)
    check.is_true(any(a.hero.name == _LUCIO.name for a in result[0].assists),
                  "assist appearing in majority of frames should be included")


def test_assist_below_threshold_excluded():
    # Lucio assists in only 2 of 6 frames (≤50%)
    lucio = KFPlayer(team=0, hero=_LUCIO)
    entries = (
        [_entry(f, 0, assists=[lucio]) for f in range(2)] +
        [_entry(f + 2, 0, assists=[]) for f in range(4)]
    )
    result = _deduplicate_entries(entries, window_frames=12)
    check.is_false(any(a.hero.name == _LUCIO.name for a in result[0].assists),
                   "assist appearing in minority of frames should be excluded")


# ---------------------------------------------------------------------------
# Resurrection split (_resolve_resurrections)
# ---------------------------------------------------------------------------

def test_resurrection_splits_cluster():
    # frames 10-12: Widow at row 0 (first death)
    # frame 13: Widow at row 0 (new death) AND row 1 (old death scrolled down)
    # frames 14-15: Widow at row 0 (new) and row 1 (old)
    entries = (
        [_entry(f, 0) for f in range(10, 13)] +          # cluster A
        [_entry(13, 0), _entry(13, 1)] +                  # split frame
        [_entry(f, 0) for f in range(14, 16)] +           # cluster B (new death)
        [_entry(f, 1) for f in range(14, 16)]             # cluster A tail
    )
    result = _deduplicate_entries(entries, window_frames=12)
    assert len(result) == 2, "split-frame cluster should produce exactly 2 entries"
    frames = sorted(e.frame for e in result)
    check.equal(frames[0], 10, "older cluster should anchor at first observed frame")
    check.equal(frames[1], 13, "newer cluster should anchor at the split frame")


def test_resolve_resurrections_no_split():
    entries = [_entry(f, 0) for f in range(5)]
    sub_clusters = _resolve_resurrections(entries)
    check.equal(len(sub_clusters), 1,
                "single-row cluster should not be split")
    check.equal(len(sub_clusters[0]), 5,
                "all 5 entries should remain in one sub-cluster")


def test_resolve_resurrections_splits_correctly():
    # frame 5 has two entries for same victim → split
    cluster = [
        _entry(3, 0), _entry(4, 0),
        _entry(5, 0), _entry(5, 1),   # split frame
        _entry(6, 0), _entry(6, 1),
    ]
    sub_clusters = _resolve_resurrections(cluster)
    assert len(sub_clusters) == 2, "frame with multiple rows should split into 2 sub-clusters"
    rows_a = {e.row for e in sub_clusters[0]}
    rows_b = {e.row for e in sub_clusters[1]}
    check.equal(rows_b, {0},
                "newer sub-cluster should contain only row-0 entries")
    check.is_in(1, rows_a,
                "older sub-cluster should contain the scrolled-down row-1 entry")


# ---------------------------------------------------------------------------
# Resurrection entry logging (player1.team == player2.team)
# ---------------------------------------------------------------------------

def test_res_entry_logs_warning(capsys):
    # Mercy (team 1) resurrects Widow (team 1) — same team kill; debug must be True to print
    entries = [_entry(f, 0, p1_hero=_MERCY, p1_team=1, p2_hero=_WIDOW, p2_team=1)
               for f in range(3)]
    _deduplicate_entries(entries, window_frames=12, debug=True)
    captured = capsys.readouterr()
    check.is_in('[KillFeed] Resurrection entry detected', captured.out,
                "res entry should print a warning when debug=True")


def test_res_entry_no_output_without_debug(capsys):
    entries = [_entry(f, 0, p1_hero=_MERCY, p1_team=1, p2_hero=_WIDOW, p2_team=1)
               for f in range(3)]
    _deduplicate_entries(entries, window_frames=12, debug=False)
    captured = capsys.readouterr()
    check.equal(captured.out, '',
                "no output should be produced when debug=False")


# ---------------------------------------------------------------------------
# player1 majority — conflicting non-None values
# ---------------------------------------------------------------------------

def test_conflicting_player1_resolved_by_majority():
    # 4 frames correctly detect Reaper, 2 misdetect as Ana — same victim, so they cluster.
    # Majority vote picks Reaper.
    entries = (
        [_entry(f, 0, p1_hero=_REAPER) for f in range(4)] +
        [_entry(f + 4, 0, p1_hero=_ANA) for f in range(2)]
    )
    result = _deduplicate_entries(entries, window_frames=12)
    assert len(result) == 1, "entries with same victim but different killers should merge into 1"
    check.equal(result[0].player1.hero.name, _REAPER.name,
                "majority-voted killer (Reaper, 4 of 6) should win over minority (Ana, 2 of 6)")


# ---------------------------------------------------------------------------
# Team awareness
# ---------------------------------------------------------------------------

def test_different_teams_not_merged():
    # Same heroes but swapped teams (mirror match) — must stay as two separate entries
    entries = [
        _entry(0, 0, p1_hero=_REAPER, p1_team=0, p2_hero=_WIDOW, p2_team=1),
        _entry(1, 0, p1_hero=_REAPER, p1_team=1, p2_hero=_WIDOW, p2_team=0),
    ]
    result = _deduplicate_entries(entries, window_frames=12)
    check.equal(len(result), 2,
                "same heroes on swapped teams (mirror match) must not merge")


def test_stacked_events_collapse_correctly():
    # Simulates a realistic killfeed stack growing and shrinking over 4 frames:
    # fr1: [Widow→Ana r0]
    # fr2: [Reaper→Mercy r0, Widow→Ana r1]
    # fr3: [Lucio→Widow r0, Reaper→Mercy r1, Widow→Ana r2]
    # fr4: [Lucio→Widow r0, Reaper→Mercy r1]  (Widow→Ana scrolled off)
    # Must collapse to exactly 3 entries, anchored at their first appearance.
    entries = [
        _entry(1, 0, p1_hero=_WIDOW,  p1_team=1, p2_hero=_ANA,   p2_team=0),
        _entry(2, 0, p1_hero=_REAPER, p1_team=0, p2_hero=_MERCY,  p2_team=1),
        _entry(2, 1, p1_hero=_WIDOW,  p1_team=1, p2_hero=_ANA,   p2_team=0),
        _entry(3, 0, p1_hero=_LUCIO,  p1_team=0, p2_hero=_WIDOW,  p2_team=1),
        _entry(3, 1, p1_hero=_REAPER, p1_team=0, p2_hero=_MERCY,  p2_team=1),
        _entry(3, 2, p1_hero=_WIDOW,  p1_team=1, p2_hero=_ANA,   p2_team=0),
        _entry(4, 0, p1_hero=_LUCIO,  p1_team=0, p2_hero=_WIDOW,  p2_team=1),
        _entry(4, 1, p1_hero=_REAPER, p1_team=0, p2_hero=_MERCY,  p2_team=1),
    ]
    result = _deduplicate_entries(entries, window_frames=12)
    assert len(result) == 3, "stacked killfeed should collapse to exactly 3 events"
    check.equal(result[0].player1.hero.name, _WIDOW.name,  "first event killer should be Widow")
    check.equal(result[0].frame, 1,                        "first event should anchor at frame 1")
    check.equal(result[1].player1.hero.name, _REAPER.name, "second event killer should be Reaper")
    check.equal(result[1].frame, 2,                        "second event should anchor at frame 2")
    check.equal(result[2].player1.hero.name, _LUCIO.name,  "third event killer should be Lucio")
    check.equal(result[2].frame, 3,                        "third event should anchor at frame 3")


def test_overlapping_events_stay_independent():
    # Three events with overlapping frame ranges — cross-event entries must never merge
    # Widow→Ana (frames 0-8), Reaper→Mercy (frames 6-14), Lucio→Widow (frames 10-18)
    widow_ana    = [_entry(f, 0, p1_hero=_WIDOW,  p1_team=1, p2_hero=_ANA,   p2_team=0) for f in range(0, 9)]
    reaper_mercy = [_entry(f, 1, p1_hero=_REAPER, p1_team=0, p2_hero=_MERCY, p2_team=1) for f in range(6, 15)]
    lucio_widow  = [_entry(f, 2, p1_hero=_LUCIO,  p1_team=1, p2_hero=_WIDOW, p2_team=0) for f in range(10, 19)]
    result = _deduplicate_entries(widow_ana + reaper_mercy + lucio_widow, window_frames=12)
    assert len(result) == 3, "three overlapping but distinct events should remain independent"
    pairs = [(e.player1.hero.name, e.player2.hero.name) for e in result]
    check.is_in((_WIDOW.name,  _ANA.name),   pairs, "Widow→Ana event must be present")
    check.is_in((_REAPER.name, _MERCY.name), pairs, "Reaper→Mercy event must be present")
    check.is_in((_LUCIO.name,  _WIDOW.name), pairs, "Lucio→Widow event must be present")


def test_triple_death_with_resurrection_barriers():
    # Three consecutive Reaper→Widow kills separated by a Mercy rez and a self-rez.
    # Entries never coexist on screen (independent frames, all row 0).
    # Without res barriers all three Reaper→Widow clusters are within window=12
    # and would merge. Expected: 5 distinct entries.
    #   0-2:  Reaper → Widow  (death 1)
    #   3-4:  Mercy  → Widow  (mercy rez — same-team barrier)
    #   5-6:  Reaper → Widow  (death 2)
    #   7-8:  Widow  → Widow  (self-rez — same-team barrier)
    #   9-10: Reaper → Widow  (death 3)
    death_1   = [_entry(f, 0, p1_hero=_REAPER, p1_team=0, p2_hero=_WIDOW, p2_team=1) for f in range(0, 3)]
    mercy_res = [_entry(f, 0, p1_hero=_MERCY,  p1_team=1, p2_hero=_WIDOW, p2_team=1) for f in range(3, 5)]
    death_2   = [_entry(f, 0, p1_hero=_REAPER, p1_team=0, p2_hero=_WIDOW, p2_team=1) for f in range(5, 7)]
    self_res  = [_entry(f, 0, p1_hero=_WIDOW,  p1_team=1, p2_hero=_WIDOW, p2_team=1) for f in range(7, 9)]
    death_3   = [_entry(f, 0, p1_hero=_REAPER, p1_team=0, p2_hero=_WIDOW, p2_team=1) for f in range(9, 11)]

    result = _deduplicate_entries(
        death_1 + mercy_res + death_2 + self_res + death_3,
        window_frames=12,
    )
    assert len(result) == 5, "three deaths + two res entries should remain as 5 distinct events"
    death_results = [e for e in result if e.player1 is not None and e.player1.team != e.player2.team]
    check.equal(len(death_results), 3,
                "exactly 3 of the 5 entries should be death events")


def test_triple_death_one_frame():
    # Three consecutive Reaper→Widow kills separated by a Mercy rez and a self-rez.
    # All entries are on the same screen simultaneously (different rows, same frames).
    # Separation relies on _resolve_resurrections row-based splitting, not frame barriers.
    #   row 0: Reaper → Widow  (death 3, newest)
    #   row 1: Mercy  → Widow  (mercy rez)
    #   row 2: Reaper → Widow  (death 2)
    #   row 3: Widow  → Widow  (self-rez)
    #   row 4: Reaper → Widow  (death 1, oldest)
    death_1   = [_entry(f, 4, p1_hero=_REAPER, p1_team=0, p2_hero=_WIDOW, p2_team=1) for f in range(0, 3)]
    mercy_res = [_entry(f, 3, p1_hero=_MERCY,  p1_team=1, p2_hero=_WIDOW, p2_team=1) for f in range(0, 3)]
    death_2   = [_entry(f, 2, p1_hero=_REAPER, p1_team=0, p2_hero=_WIDOW, p2_team=1) for f in range(0, 3)]
    self_res  = [_entry(f, 1, p1_hero=_WIDOW,  p1_team=1, p2_hero=_WIDOW, p2_team=1) for f in range(0, 3)]
    death_3   = [_entry(f, 0, p1_hero=_REAPER, p1_team=0, p2_hero=_WIDOW, p2_team=1) for f in range(0, 3)]

    result = _deduplicate_entries(
        death_1 + mercy_res + death_2 + self_res + death_3,
        window_frames=12,
    )
    assert len(result) == 5, "three deaths + two res entries visible simultaneously should remain as 5 events"
    death_results = [e for e in result if e.player1 is not None and e.player1.team != e.player2.team]
    check.equal(len(death_results), 3,
                "exactly 3 of the 5 entries should be death events")


def test_res_barrier_splits_on_resurrection():
    # Reaper kills Widow at frames 0-5 (first death, scrolls off before frame 10)
    # Mercy rezzes Widow at frames 8-10 (res entries — same team, Widow team 1)
    # Reaper kills Widow again at frames 12-17 (second death, within window of first if no barrier)
    # Expected: 2 death clusters separated by the res barrier
    death_1 = [_entry(f, 0, p1_hero=_REAPER, p1_team=0, p2_hero=_WIDOW, p2_team=1)
               for f in range(0, 6)]
    res = [_entry(f, 0, p1_hero=_MERCY, p1_team=1, p2_hero=_WIDOW, p2_team=1)
           for f in range(8, 11)]
    death_2 = [_entry(f, 0, p1_hero=_REAPER, p1_team=0, p2_hero=_WIDOW, p2_team=1)
               for f in range(12, 18)]
    result = _deduplicate_entries(death_1 + res + death_2, window_frames=12)
    death_results = [e for e in result if e.player1 is not None and e.player1.team != e.player2.team]
    check.equal(len(death_results), 2,
                "resurrection barrier should prevent the two Reaper→Widow deaths from merging")
