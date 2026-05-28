from pathlib import Path

import cv2
import pytest
from pytest_check import check

from ora.detection.killfeed import KFParseContext, KillFeedEntry
from ora.detection.killfeed.main import get_killfeed_entry_images
from ora.tests.test_killfeed_cases import (TC_ENVIRONMENTAL, TC_CRITICAL, TC_ASSIST, TC_SINGLE,
                                           TC_ABILITIES, TC_ARROW_HARD,
                                           heroes)
from ora.utils.image import read_image

TESTING_FILES = Path(r'C:\Users\duff_\Desktop\test\test_cases')
# arrow_ref = read_image(r"C:\PyProjects\OverwatchDataAnalysis\images\ui\killfeed_arrow2.png",
#                        cv2.IMREAD_GRAYSCALE)
arrow_ref = read_image(r"C:\PyProjects\OverwatchDataAnalysis\images\ui\killfeed_arrow3.png",
                       cv2.IMREAD_UNCHANGED)


# def test_detect_killfeed_presence():
#     results = []
#     for category in TESTING_FILES.iterdir():
#         for image in category.iterdir():
#             im = read_image(image)
#             is_visible = detect_killfeed_presence(im)
#             results.append((image, is_visible, category.name == 'has_killfeed'))
#
#     preds = [pred for _, pred, gt in results]
#     gts = [gt for _, pred, gt in results]
#
#     recall = recall_score(gts, preds)
#
#     # print("Accuracy :", accuracy_score(gts, preds))
#     # print("Precision:", precision_score(gts, preds))
#     # print("Recall   :", recall)
#     # print("F1       :", f1_score(gts, preds))
#     #
#     # print("\nConfusion matrix:")
#     # print(confusion_matrix(gts, preds))
#     assert recall >= 0.92
#
#
# def test_detect_arrows():
#     results = []
#     for image in (TESTING_FILES / 'has_killfeed').iterdir():
#         im = read_image(image)
#         arrow_found = detect_killfeed_arrows(im, ref=arrow_ref)
#         results.append((image, bool(arrow_found), 1))
#
#     preds = [pred for _, pred, gt in results]
#     gts = [gt for _, pred, gt in results]
#
#     recall = recall_score(gts, preds)
#     assert recall >= 0.99


def entry_tester(im_name: str, entries: list[KillFeedEntry], image_paths: dict[str, Path]):
    image_path = image_paths[im_name]
    im = read_image(image_path)

    ent_im = get_killfeed_entry_images(frame=im, ref=arrow_ref,
                                      debug=False, entry_height_pc=0.15)

    # cv2.waitKey(0)

    check.equal(len(ent_im), len(entries), 'Different number of rows detected')

    for i, ((im, arrow_center, arrow_box), entry) in enumerate(zip(ent_im, entries)):
        c = KFParseContext(image=im, arrow_center=arrow_center,
                           arrow_box=arrow_box, frame_i=entry.frame, row_i=entry.row,
                           heroes=list(heroes.values()))

        # found_entry = None
        # try:
        found_entry = KillFeedEntry.from_image(c)
        # print(found_entry)
        # except Exception as e:
        #     print(e)

        # check.not_equal(found_entry, None, f'{entry.frame}:{entry.row} Heroes not found')

        are_killfeed_entries_identical(entry, found_entry, frame=entry.frame, row=i)


def are_killfeed_entries_identical(killfeed_entry1: KillFeedEntry, killfeed_entry2: KillFeedEntry,
                                   frame: int, row: int):
    # killer, killed, and teams
    check.equal(killfeed_entry1, killfeed_entry2, f'{frame}:{row} Detected heroes or teams are different')
    # everything else
    check.equal(killfeed_entry1.is_critical, killfeed_entry2.is_critical, f'{frame}:{row} Critical status is different')
    check.equal(killfeed_entry1.assists, killfeed_entry2.assists, f'{frame}:{row} Assists do not align')
    check.equal(killfeed_entry1.ability, killfeed_entry2.ability, f'{frame}:{row} Ability does not align')
    check.equal(killfeed_entry1.is_environmental, killfeed_entry2.is_environmental,
                f'{frame}:{row} Environmental status is different')


@pytest.mark.parametrize("im_names, entries", TC_SINGLE.items(), ids=TC_SINGLE.keys())
def test_solo_kills(im_names, entries):
    image_paths = {p.stem: p for p in (TESTING_FILES / 'has_killfeed').iterdir() if p.stem in im_names}

    entry_tester(im_name=im_names, entries=entries, image_paths=image_paths)


@pytest.mark.parametrize("im_names, entries", TC_ASSIST.items(), ids=TC_ASSIST.keys())
def test_assists(im_names, entries):
    image_paths = {p.stem: p for p in (TESTING_FILES / 'has_killfeed').iterdir() if p.stem in im_names}

    entry_tester(im_name=im_names, entries=entries, image_paths=image_paths)


@pytest.mark.parametrize("im_names, entries", TC_CRITICAL.items(), ids=TC_CRITICAL.keys())
def test_critical(im_names, entries):
    image_paths = {p.stem: p for p in (TESTING_FILES / 'has_killfeed').iterdir() if p.stem in im_names}

    entry_tester(im_name=im_names, entries=entries, image_paths=image_paths)


@pytest.mark.parametrize("im_names, entries", TC_ENVIRONMENTAL.items(), ids=TC_ENVIRONMENTAL.keys())
def test_environmental(im_names, entries):
    image_paths = {p.stem: p for p in (TESTING_FILES / 'has_killfeed').iterdir() if p.stem in im_names}

    entry_tester(im_name=im_names, entries=entries, image_paths=image_paths)


@pytest.mark.parametrize("im_names, entries", TC_ABILITIES.items(), ids=TC_ABILITIES.keys())
def test_abilities(im_names, entries):
    image_paths = {p.stem: p for p in (TESTING_FILES / 'has_killfeed').iterdir() if p.stem in im_names}

    entry_tester(im_name=im_names, entries=entries, image_paths=image_paths)


@pytest.mark.xfail(reason="hard case")
@pytest.mark.parametrize("im_names, entries", TC_ARROW_HARD.items(), ids=TC_ARROW_HARD.keys())
def test_arrows_hard(im_names, entries):
    image_paths = {p.stem: p for p in (TESTING_FILES / 'has_killfeed').iterdir() if p.stem in im_names}

    entry_tester(im_name=im_names, entries=entries, image_paths=image_paths)
