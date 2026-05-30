import numpy as np

from ...hero import Hero
from .entry import KillFeedEntry
from .main import process_killfeed_frame

_heroes: list[Hero] | None = None


def init_worker(heroes: list[Hero]) -> None:
    global _heroes
    _heroes = heroes


def killfeed_process_fn(roi: np.ndarray, frame_i: int) -> list[KillFeedEntry]:
    """FrameDispatcher expects process_fn(roi, frame_i) with no heroes argument.
    functools.partial would work syntactically but would pickle heroes with every
    task submission — one copy per frame across IPC. This wrapper reads heroes from
    the module-level _heroes instead, which is set once per worker process by
    init_worker at pool startup."""
    return process_killfeed_frame(roi, frame_i, _heroes)
