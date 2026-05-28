import cv2
import numpy as np

from ...hero import Hero

TEAM_COLORS = {0: np.array([40, 40, 220]),
               1: np.array([220, 220, 40])}


def get_team_idx(frame: np.ndarray, box: tuple[int, int, int, int],
                 debug: bool = False, hero: Hero | None = None) -> int:
    """Scan along the left and the right sides of the detected hero portrait to get a mean color
    of the killfeed box. NOTE: reliably works only for non-assists!"""

    left_color, right_color = _scan_sides_color(frame, box)
    if left_color is None or right_color is None:
        raise ValueError(f'team color scan returned empty strip for box {box}')
    team_color = np.mean((left_color, right_color), axis=0)

    if debug:
        swatch = np.zeros((100, 300, 3), dtype=np.uint8)
        swatch[:, :100] = left_color
        swatch[:, 100:] = right_color
        swatch[:, 200:] = team_color
        cv2.imshow(f"swatch_{hero}", swatch)

    return _closest_team_by_color(team_color)


def _scan_sides_color(frame: np.ndarray, box: tuple[int, int, int, int],
                      team_color_px_offset: int = 10,
                      scan_width: int = 2):
    bx1, by1, bx2, by2 = map(int, box)
    H, W = frame.shape[:2]

    left_x1 = max(0, bx1 - team_color_px_offset - scan_width)
    left_x2 = max(0, bx1 - team_color_px_offset)
    right_x1 = min(W, bx2 + team_color_px_offset)
    right_x2 = min(W, bx2 + team_color_px_offset + scan_width)
    y1 = max(0, by1)
    y2 = min(H, by2)

    left_strip = frame[y1:y2, left_x1:left_x2]
    right_strip = frame[y1:y2, right_x1:right_x2]

    left_mean = left_strip.mean(axis=(0, 1)) if left_strip.size else None
    right_mean = right_strip.mean(axis=(0, 1)) if right_strip.size else None

    return left_mean, right_mean


def _closest_team_by_color(color) -> int:
    color = np.array(color)
    team1 = np.linalg.norm(color - TEAM_COLORS[0])
    team2 = np.linalg.norm(color - TEAM_COLORS[1])
    return 0 if team1 < team2 else 1
