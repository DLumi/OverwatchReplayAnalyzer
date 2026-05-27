from typing import Literal
from itertools import repeat

import cv2
import numpy as np

from .killfeed_detection import filter_blobs, fill_closed_edge_shapes
from .template_matching import find_template_multiscale, percent_roi_to_pixels, edge_ncc
from ..hero import Hero, Ability, AbilityCategory
from ..player import KFPlayer
from .image import read_image

_env = Hero(name='Environmental',
            abilities=[Ability(category='ability',
                               name='Environmental',
                               bind=None,
                               icon=read_image(
                                   r"C:\PyProjects\OverwatchDataAnalysis\images\ui\EnvironmentalKill.webp",
                                   cv2.IMREAD_UNCHANGED))])


def get_kilfeed_entry_images(frame, ref, entry_height_pc: float = 0.15, debug: bool = False) \
        -> list[tuple[np.ndarray, tuple[int, int], tuple[int, ...]]]:
    h, w = frame.shape[:2]

    rx1 = int(w * 0.3)
    rx2 = int(w * 0.8)
    ry1 = int(h * 0.1)

    # print(ry1, h)

    arrow_frame = frame[ry1:, rx1:rx2]

    # cv2.imwrite(r'C:\Users\duff_\Desktop\test\crop.png', arrow_frame)

    # cv2.imshow('arrow_frame', arrow_frame)

    arrow_data = detect_killfeed_arrows(arrow_frame, ref, debug=debug)

    if not arrow_data:
        return []

    # readjusting coordinates back
    arrow_data = [
        (
            (cx + rx1, cy + ry1),
            (bx + rx1, by + rx1, bw, bh),
        )
        for (cx, cy), (bx, by, bw, bh) in arrow_data
    ]

    arrow_centers, arrow_boxes = [x[0] for x in arrow_data], [x[1] for x in arrow_data]

    arrow_centers = sorted(arrow_centers, key=lambda x: x[1], reverse=True)
    arrow_boxes = sorted(arrow_boxes, key=lambda x: x[1], reverse=True)

    entry_h_halved = int(h * entry_height_pc / 2)

    kf_entries = []
    for i, ((x, y), (b_x, b_y, b_w, b_h)) in enumerate(zip(arrow_centers, arrow_boxes)):
        y1 = y - entry_h_halved
        y2 = y + entry_h_halved
        entry_cropped = frame[y1: y2, 0: w]

        kf_entries.append((entry_cropped, (x, y), (b_x, 0, b_w, entry_h_halved * 2)))
        if debug:
            cv2.imshow(f"cropped_entry_{i}", entry_cropped)
    return kf_entries


def detect_killfeed_arrows(frame: np.ndarray, ref: np.ndarray, debug: bool = False) \
        -> list[tuple[tuple[int, int], tuple[int, ...]]]:
    # frame = mask_non_white(frame)

    if debug:
        cv2.imshow("frame", frame)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # roi_gray_blurred_2 = cv2.bilateralFilter(
    #     frame_gray,
    #     d=40,
    #     sigmaColor=20,
    #     sigmaSpace=100
    # )
    #
    # roi_edges = cv2.Canny(
    #     roi_gray_blurred_2,
    #     threshold1=50,
    #     threshold2=150,
    #     apertureSize=7,
    #     L2gradient=True,
    # )

    # cv2.imshow("roi_edges", roi_edges)

    # mask_non_white(frame)
    roi_gray_blurred = shit(frame, debug=debug)

    # roi_gray_blurred = cv2.GaussianBlur(frame_gray, (0, 0), sigmaX=7, sigmaY=7)
    # search_area = cv2.subtract(frame_gray, roi_gray_blurred)
    #
    # # attenuate non-white regions in the DoG result
    #
    # # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # # white_mask = cv2.inRange(hsv, (0, 0, 120), (180, 110, 255))
    #
    # roi_gray_blurred = cv2.normalize(search_area, None, 0, 255, cv2.NORM_MINMAX)
    #
    # if debug:
    #     # cv2.imshow("white_mask", white_mask)
    #     cv2.imshow("roi_gray_blurred", roi_gray_blurred)

    roi_edges_filled = (roi_gray_blurred > 50).astype(np.uint8) * 255  # 35

    # kernel = np.ones((2, 2), np.uint8)
    # roi_edges_filled = cv2.morphologyEx(roi_edges_filled, cv2.MORPH_OPEN, kernel)

    roi_edges_filled, _ = filter_blobs(
        roi_edges_filled,
        min_size_frac=0.0007,
        max_size_frac=0.01,
        fill_percent=0.2,
        large_blob_multiplier=0.0,
        max_horizontal_dominance=3.0,
        max_vertical_dominance=5.0,
    )

    if debug:
        cv2.imshow("roi_edges_filled", roi_edges_filled)

    # ref_gray_blurred = cv2.GaussianBlur(ref, (0, 0), sigmaX=7, sigmaY=7)
    # ref_gray_blurred = cv2.subtract(ref, ref_gray_blurred)
    # ref_edges_filled = (ref_gray_blurred > 35).astype(np.uint8) * 255
    # ref_edges_filled = shit(ref, debug=False)
    # ref_edges_filled = (ref_edges_filled > 100).astype(np.uint8) * 255
    _, ref_edges_filled = prepare_2d_portrait(ref, margin=(0, 0))

    # cv2.imshow("ref_edges_filled", ref_edges_filled)

    candidates = find_template_multiscale(frame=roi_edges_filled,
                                          template_base=ref_edges_filled,
                                          # mask_base=ref_edges_filled,
                                          min_height_pct=0.085, max_height_pct=0.08,
                                          scale_steps=1,
                                          # mae_threshold=0.6,
                                          threshold=0.6)  # .75, 0.7

    # print(len(candidates))

    if not candidates:
        return []

    candidate_groups = group_by_row(candidates, vertical_threshold=22)

    arrow_data = []
    for group in candidate_groups:
        # print('-----------')
        # if debug:
        #     for gn, g in enumerate(group):
        #         x, y, w, h = g['box']
        #         print(w, h)
        #         # print(g['score'])
        #         patch = roi_edges_filled[y:y + h, x:x + w]
        #         template_resized = cv2.resize(ref_edges_filled, (w, h), interpolation=cv2.INTER_NEAREST)
        #         cv2.imshow(f"template_resized_{gn}", template_resized)
        #         cv2.imshow(f"patch_{gn}", patch)
        #
        #         iou = binary_iou(patch, template_resized)  # recompute for the gate
        #         print('iou', iou)
        #         print('shape similarity', shape_similarity(patch, template_resized))
        #         cv2.circle(frame, g['box'][:2], radius=3, thickness=5, color=(0, 0, 255), lineType=cv2.LINE_AA)

        # group = [g for g in group if g['score'] > 0.65]

        # # pick best IoU within the row
        # best = max(group, key=lambda res: binary_iou(
        #     roi_edges_filled[res['box'][1]:res['box'][1] + res['box'][3],
        #     res['box'][0]:res['box'][0] + res['box'][2]],
        #     cv2.resize(ref_edges_filled, (res['box'][2], res['box'][3]),
        #                interpolation=cv2.INTER_NEAREST)
        # ))

        def candidate_key(res):
            patch = roi_edges_filled[res['box'][1]:res['box'][1] + res['box'][3],
                    res['box'][0]:res['box'][0] + res['box'][2]]
            template_resized = cv2.resize(ref_edges_filled, (res['box'][2], res['box'][3]),
                                          interpolation=cv2.INTER_NEAREST)
            iou = binary_iou(patch, template_resized)
            bucket = round(iou / 0.03)  # quantize to 0.02 buckets
            return (bucket, res['score'])

        best = max(group, key=candidate_key)

        # # pick best IoU within the row
        # best = min(group, key=lambda res: shape_similarity(
        #     roi_edges_filled[res['box'][1]:res['box'][1] + res['box'][3],
        #     res['box'][0]:res['box'][0] + res['box'][2]],
        #     cv2.resize(ref_edges_filled, (res['box'][2], res['box'][3]),
        #                interpolation=cv2.INTER_NEAREST)
        # ))

        x, y, w, h = best['box']

        patch = roi_edges_filled[y:y + h, x:x + w]
        template_resized = cv2.resize(ref_edges_filled, (w, h), interpolation=cv2.INTER_NEAREST)
        iou = binary_iou(patch, template_resized)  # recompute for the gate
        # iou = shape_similarity(patch, template_resized)  # recompute for the gate

        if debug:
            print(f'best {iou=}')

        if iou < 0.4:  # softer threshold now
            continue

        # if iou > 2:  # softer threshold now
        #     continue

        center = int(x + w / 2), int(y + h / 2)

        # if debug:
        # cv2.circle(frame, center, radius=3, thickness=5, color=(0, 0, 255), lineType=cv2.LINE_AA)
        arrow_data.append((center, best['box']))

    arrow_data = filter_vertical_outliers(arrow_data, max_distance=50)

    if debug:
        for c, _ in arrow_data:
            cv2.circle(frame, c, radius=3, thickness=5, color=(0, 255, 0), lineType=cv2.LINE_AA)

    # arrow_data = []
    # for res in candidates:
    #     if debug:
    #         print(res)
    #     x, y, w, h = res['box']
    #
    #     patch = roi_edges_filled[y:y + h, x:x + w]
    #     template_resized = cv2.resize(ref_edges_filled, (w, h), interpolation=cv2.INTER_NEAREST)
    #
    #     iou = binary_iou(patch, template_resized)
    #
    #     print(f'{iou=}')
    #     if iou < 0.47:  # tune this; 0.53
    #         continue
    #
    #     center = int(x + w / 2), int(y + h / 2)
    #
    #     if debug:
    #         # print(f'{iou=}')
    #         cv2.circle(frame, center, radius=3, thickness=5, color=(0, 255, 0), lineType=cv2.LINE_AA)
    #     arrow_data.append((center, res['box']))

    return arrow_data


def shape_similarity(patch, template):
    m1 = cv2.moments((patch > 127).astype(np.uint8))
    m2 = cv2.moments((template > 127).astype(np.uint8))
    hu1 = cv2.HuMoments(m1).flatten()
    hu2 = cv2.HuMoments(m2).flatten()
    # log scale comparison
    hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
    hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)
    return np.sum(np.abs(hu1 - hu2))


def mask_non_white(frame: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(frame.astype(np.int16))
    spread = (np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r))

    white_mask_new = (spread < 40).astype(np.uint8) * 255

    return cv2.bitwise_and(frame, frame, mask=white_mask_new)


def filter_vertical_outliers(arrow_data, max_distance):
    if len(arrow_data) <= 1:
        return arrow_data

    arrow_data = sorted(arrow_data, key=lambda a: a[0][1])  # sort by center y

    filtered = [arrow_data[0]]
    for current in arrow_data[1:]:
        if current[0][1] - filtered[-1][0][1] <= max_distance:
            filtered.append(current)
        else:
            break  # everything below this is too far

    return filtered


def group_by_row(candidates, vertical_threshold=10):
    candidates = sorted(candidates, key=lambda c: c['box'][1])  # sort by y
    groups = []
    current_group = [candidates[0]]

    for c in candidates[1:]:
        if abs(c['box'][1] - current_group[0]['box'][1]) <= vertical_threshold:
            current_group.append(c)
        else:
            groups.append(current_group)
            current_group = [c]
    groups.append(current_group)
    return groups


def shit(frame, debug: bool = False):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. whiteness
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # white_mask = cv2.inRange(hsv, (0, 0, 120), (180, 110, 255))

    b, g, r = cv2.split(frame.astype(np.int16))
    spread = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
    # whiteness = np.clip(255 - spread * 3, 0, 255).astype(np.uint8)
    # whiteness = (spread < 50).astype(np.uint8) * 255
    # whiteness = cv2.normalize(255 - spread, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    whiteness = np.exp(-spread.astype(np.float32) / 40) * 255
    whiteness = whiteness.astype(np.uint8)

    if debug:
        cv2.imshow('whiteness', whiteness)

    # 2. medium frequency (DoG)
    # blur_lo = cv2.GaussianBlur(frame_gray, (0, 0), sigmaX=3)
    blur_hi = cv2.GaussianBlur(frame_gray, (0, 0), sigmaX=7)
    dog = cv2.subtract(frame_gray, blur_hi)  # bandpass
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

    if debug:
        cv2.imshow('dog', dog)

    # 3. local brightness dominance
    # local_max = cv2.dilate(frame_gray, np.ones((5, 5), np.uint8))
    # dominance = (frame_gray.astype(np.float32) / (local_max.astype(np.float32) + 1e-6) * 255).astype(np.uint8)
    local_min = cv2.erode(frame_gray, np.ones((5, 5), np.uint8))
    local_min = cv2.GaussianBlur(local_min, (0, 0), sigmaX=7)

    if debug:
        cv2.imshow('local_min', local_min)

    dominance = cv2.subtract(frame_gray, local_min)

    if debug:
        cv2.imshow('dominance', dominance)

    # edges = cv2.Canny(frame_gray, 50, 150)
    # edges_blurred = cv2.GaussianBlur(edges, (0, 0), sigmaX=3)  # spread edge influence

    gx = cv2.Sobel(frame_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame_gray, cv2.CV_32F, 0, 1, ksize=3)

    # cv2.imshow('gx', gx)
    # cv2.imshow('gy', gy)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    threshold = np.percentile(magnitude, 1)  # adaptive, tune the percentile
    magnitude = np.clip(magnitude - threshold, 0, None)

    angle = np.arctan2(np.abs(gy), np.abs(gx))  # 0 = horizontal, pi/2 = vertical

    # diagonalness: peaks at pi/4 (45 degrees)
    diagonalness = np.sin(2 * angle)  # 0 at 0 and pi/2, 1 at pi/4

    edges_diagonal = magnitude * diagonalness
    edges_diagonal = cv2.normalize(edges_diagonal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if debug:
        cv2.imshow('edges_blurred', edges_diagonal)

    # combine
    # arrowness = (dog.astype(np.float32) *
    #              dominance.astype(np.float32)) ** (1 / 2)  # geometric mean
    # arrowness = (whiteness.astype(np.float32) *
    #              dog.astype(np.float32) *
    #              dominance.astype(np.float32)) ** (1 / 3)  # geometric mean
    arrowness = (dominance.astype(np.float32) ** 1.0 *
                 whiteness.astype(np.float32) ** 0.4 *
                 edges_diagonal.astype(np.float32) ** 0.6) ** (1 / 2)
    arrowness = cv2.normalize(arrowness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if debug:
        cv2.imshow('arrowness', arrowness)

    # arrowness = cv2.bitwise_and(arrowness, whiteness)
    #
    # cv2.imshow('arrowness_constrained', arrowness)

    return arrowness


def binary_iou(patch, template):
    # both already binary (0/255) at this point
    p = patch > 127
    t = template > 127
    intersection = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return intersection / (union + 1e-6)


def preprocess_for_ability_matching(frame, debug: bool = False):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi_gray_blurred = cv2.GaussianBlur(frame_gray, (0, 0), sigmaX=7, sigmaY=7)
    search_area = cv2.subtract(frame_gray, roi_gray_blurred)

    roi_gray_blurred = cv2.normalize(search_area, None, 0, 255, cv2.NORM_MINMAX)

    if debug:
        cv2.imshow("roi_gray_blurred", roi_gray_blurred)

    roi_edges_filled = (roi_gray_blurred > 30).astype(np.uint8) * 255

    if debug:
        cv2.imshow("roi_filled", roi_edges_filled)

    # roi_edges_filled, _ = filter_blobs(
    #     roi_edges_filled,
    #     min_size_frac=0.01,
    #     max_size_frac=0.1,
    #     fill_percent=0.2,
    #     large_blob_multiplier=0.0,
    #     max_horizontal_dominance=3.0,
    #     max_vertical_dominance=3.0,
    # )

    # if debug:
    #     cv2.imshow("roi_filled_filtered", roi_edges_filled)

    return roi_edges_filled


def detect_ability(frame: np.ndarray,
                   ref_mask: np.ndarray,
                   category: AbilityCategory | Literal['environmental'],
                   debug: bool = False) \
        -> list[tuple[tuple[int, int], tuple[int, ...], float]]:

    fr_h = frame.shape[0]

    if category == 'environmental':
        min_height_pct = 0.9
        max_height_pct = 1.0
        scale_steps = 2
    elif category == 'ultimate':
        min_height_pct = 0.7
        max_height_pct = 0.8
        scale_steps = int(fr_h * max_height_pct) - int((fr_h * min_height_pct))
    else:
        min_height_pct = 0.5
        max_height_pct = 0.7
        scale_steps = int(fr_h * max_height_pct) - int((fr_h * min_height_pct))

    candidates = find_template_multiscale(frame=frame,
                                          template_base=ref_mask,
                                          # mask_base=ref_mask,
                                          min_height_pct=min_height_pct, max_height_pct=max_height_pct,
                                          # method=cv2.TM_CCORR_NORMED,
                                          scale_steps=scale_steps,
                                          # mae_threshold=100,
                                          threshold=0.7,
                                          )

    if not candidates:
        ref_mask = cv2.flip(ref_mask, 1)
        candidates = find_template_multiscale(frame=frame,
                                              template_base=ref_mask,
                                              # mask_base=ref_mask,
                                              min_height_pct=min_height_pct, max_height_pct=max_height_pct,
                                              # method=cv2.TM_CCORR_NORMED,
                                              scale_steps=scale_steps,
                                              # mae_threshold=100,
                                              threshold=0.7)

    if not candidates:
        return []

    centers = []
    for res in candidates:
        # print(res)
        x, y, w, h = res['box']

        # print(candidates)
        center = int(x + w / 2), int(y + h / 2)
        # cv2.circle(search_img, center, radius=3, thickness=5, color=(0, 255, 0), lineType=cv2.LINE_AA)
        centers.append((center, res['box'], res['score']))

    return centers


def prepare_2d_portrait(image, margin: tuple[int, int] = (50, 50)) -> tuple[np.ndarray, np.ndarray]:
    m_v, m_h = margin
    h, w = image.shape[:2]

    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]

        mask_base = ((alpha > 10).astype(np.uint8) * 255)[m_v:h - m_v, m_h:w - m_h]
        template_base = bgr[m_v:h - m_v, m_h:w - m_h]
    else:
        template_base = image[m_v:h - m_v, m_h:w - m_h]
        mask_base = None

    return template_base, mask_base


def scan_side_color(frame, box, team_color_px_offset: int = 10, scan_width: int = 2):
    x, y, w, h = map(int, box)

    H, W = frame.shape[:2]

    # Keep scan regions inside image bounds
    left_x1 = max(0, x - team_color_px_offset - scan_width)
    left_x2 = max(0, x - team_color_px_offset)

    right_x1 = min(W, x + w + team_color_px_offset)
    right_x2 = min(W, x + w + team_color_px_offset + scan_width)

    y1 = max(0, y)
    y2 = min(H, y + h)

    # print('left', y1,y2, left_x1,left_x2)
    # print('right', y1,y2, right_x1,right_x2)

    left_strip = frame[y1:y2, left_x1:left_x2]
    right_strip = frame[y1:y2, right_x1:right_x2]

    left_mean = left_strip.mean(axis=(0, 1)) if left_strip.size else None
    right_mean = right_strip.mean(axis=(0, 1)) if right_strip.size else None

    return left_mean, right_mean


TEAM_COLORS = {0: np.array([40, 40, 220]),
               1: np.array([220, 220, 40])}


def closest_team(color):
    color = np.array(color)

    team1 = np.linalg.norm(color - TEAM_COLORS[0])
    team2 = np.linalg.norm(color - TEAM_COLORS[1])

    return 0 if team1 < team2 else 1


def detect_heroes_2d(frame, hero_list: list[Hero],
                     portrait_type: Literal['kf_main', 'kf_assist'],
                     debug: bool = False) -> list[tuple[Hero, tuple[int, int], tuple[int, ...], int]]:
    # print('--------------------')

    fr_h = frame.shape[0]

    detected_heroes = []
    detected_instances = []

    if portrait_type == 'kf_main':
        margin = (50, 50)
        min_h = 0.8
        max_h = 0.9
        steps = int((max_h - min_h) * fr_h)
        mae_threshold = 0.3
    elif portrait_type == 'kf_assist':
        margin = (20, 50)  # 20
        min_h = 0.6
        max_h = 0.8
        steps = int((max_h - min_h) * fr_h)
        mae_threshold = 0.45
    else:
        raise ValueError(f'portrait_type {portrait_type} not supported')

    search_params = {'min_height_pct': min_h,
                     'max_height_pct': max_h,
                     'scale_steps': steps,
                     # 'mae_threshold': 29,
                     'mae_threshold': mae_threshold,
                     # 'quality_threshold': 0.94,
                     }

    rx, ry, rw, rh = percent_roi_to_pixels(frame.shape, (0.08, 0.0, 0.92, 1.0))
    search_space = frame[ry:ry + rh, rx:rx + rw]

    for hero in hero_list:
        template, mask = prepare_2d_portrait(hero.portrait2d, margin=margin)

        if mask is not None:
            method = cv2.TM_CCORR_NORMED
            if portrait_type == 'kf_main':
                threshold = 0.92
            else:
                threshold = 0.88
        else:
            method = cv2.TM_CCOEFF_NORMED
            threshold = 0.8

        # if hero.name == 'Echo':
        #     # print(template.shape)
        #     cv2.imshow('template', template)
        #     cv2.imshow('search_space', search_space.copy())

        # print(hero.name)

        found_instances = find_template_multiscale(search_space, template, **search_params,
                                                   mask_base=mask,
                                                   method=method,
                                                   threshold=threshold)

        if found_instances:
            detected_instances.extend(zip(repeat(hero), found_instances))

    # print(detected_instances)

    for hero, instance in detected_instances:
        # print(hero)
        # print(instance)
        if debug:
            cv2.circle(frame, instance['loc'], radius=3,
                       thickness=5, color=(0, 255, 0), lineType=cv2.LINE_AA)

        x, y, w, h = instance['box']
        x = rx + x

        center = (int(x + w / 2), int(y + h / 2))

        if portrait_type == 'kf_main':
            left_color, right_color = scan_side_color(frame, (x, y, w, h))
            # print(left_color, right_color)
            team_color = np.mean((left_color, right_color), axis=0)
            # team_color = (255, 255, 255)
            team = closest_team(team_color)
        else:
            team = -1

        if debug and portrait_type == 'kf_main':
            # print((hero, center, team))

            swatch = np.zeros((100, 300, 3), dtype=np.uint8)
            swatch[:, :100] = left_color
            swatch[:, 100:] = right_color
            swatch[:, 200:] = team_color

            cv2.imshow(f"swatch_{hero}", swatch)

        detected_heroes.append((hero, center, (x, y, w, h), team))
    # print(detected_heroes)
    return sorted(detected_heroes, key=lambda x: x[1][0])


def get_used_ability(frame: np.ndarray, hero: Hero | None, debug: bool = False) -> Ability | None:
    category = None
    if not hero:
        hero = _env
        category = 'environmental'

    prep = preprocess_for_ability_matching(frame, debug=debug)
    # prep = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    all_found = []
    masks = []
    for i, ability in enumerate(hero.abilities):
        _, mask = prepare_2d_portrait(ability.icon, margin=(0, 0))

        if ability.category == 'ultimate':
            mask = cv2.bitwise_not(mask)

        if debug:
            cv2.imshow(f'ab_{i}', mask)

        # print(ability.name)

        found_instances = detect_ability(frame=prep,
                                         ref_mask=mask,
                                         debug=debug,
                                         # debug=False,
                                         category=category or ability.category
                                         )

        # if ability.name == 'Charged Shot':
        #     print('Biba!------------------')
        #     print(found_instances)
        #     cv2.waitKey(0)

        for instance in found_instances:
            masks.append(mask)
            all_found.append((ability, *instance))

    # def candidate_key(
    #         res, prep, mask):
    #     _, _, box, score = res
    #
    #     patch = prep[box[1]:box[1] + box[3],
    #             box[0]:box[0] + box[2]]
    #     template_resized = cv2.resize(mask, (box[2], box[3]),
    #                                   interpolation=cv2.INTER_NEAREST)
    #     # iou = binary_iou(patch, template_resized)
    #     iou = blob_ncc(patch, template_resized)  # FIXME post-match filtering
    #     print(iou, score)
    #     return iou
    #
    # all_found = [max([found], key=lambda res: candidate_key(res, prep, m))
    #              for found, m in zip(all_found, masks)
    #              if found]

    # all_found = [min([found], key=lambda res: candidate_key(res, prep, m))
    #              for found, m in zip(all_found, masks)
    #              if found]

    if debug:
        print(all_found[:3])

    if len(all_found) > 1:
        raise ValueError('Detected several abilities in KillFeed!')

    return all_found[0] if all_found else None


def blob_ncc(patch, template):
    p = patch.astype(np.float32) / 255.0
    t = template.astype(np.float32) / 255.0

    p -= p.mean()
    t -= t.mean()

    denom = np.sqrt(np.sum(p ** 2) * np.sum(t ** 2)) + 1e-6
    return np.sum(p * t) / denom


def assign_roles(heroes: list, arrow_center: tuple[int, int], debug: bool = False) \
        -> dict[str, tuple[Hero, tuple[int, int], tuple[int, ...], int]]:
    if len(heroes) > 2:
        if debug:
            print(heroes)
        raise ValueError('Detected more than 2 heroes in KillFeed!')

    roles = {}

    for hero in heroes:
        hero_center_x = hero[1][0]

        if hero_center_x < arrow_center[0]:
            role = 'subject'
        else:
            role = 'object'

        if role in roles:
            print('WARNING! Overwriting roles')
        roles[role] = hero

    return roles
