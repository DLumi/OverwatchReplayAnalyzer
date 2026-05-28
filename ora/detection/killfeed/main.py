import cv2
import numpy as np

from . import KillFeedEntry
from .arrows import detect_killfeed_arrows
from .entry import _KF_ARROW_REF

from ...utils.box_proc import percent_roi_to_pixels
from .presence import detect_killfeed_presence


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
