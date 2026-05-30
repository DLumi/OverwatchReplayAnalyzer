"""Parallel video frame dispatcher.

Each registered job declares an ROI, a sampling step, and a process_fn.
FrameDispatcher splits the video into contiguous worker chunks, and each
worker runs a fetch thread (decord get_batch + immediate ROI crop) feeding
a bounded queue (maxsize=2) to keep at most one sub-batch of full frames
in memory at a time. Results are collected per job and returned flat.

All frame indices are global (absolute position in the VOD file), so results
from different workers and different jobs share the same frame coordinate space.
"""

import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from queue import Queue
from threading import Thread

import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm

from ..utils.box_proc import percent_roi_to_pixels


@dataclass
class DispatchJob:
    """Configuration for one analysis pass over the video.

    step is in frames, not seconds — derive it as round(fps / target_fps).
    process_fn must be a top-level module-level function; lambdas and closures
    will be rejected at register() time because they can't be pickled for
    worker processes on Windows.
    """
    name: str
    roi_pct: tuple[float, float, float, float]
    step: int
    process_fn: Callable[[np.ndarray, int], list]


def _chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _crop_roi(frame: np.ndarray, roi_pct: tuple[float, float, float, float]) -> np.ndarray:
    # ascontiguousarray is required: the frame has already had its channel axis
    # reversed (RGB→BGR) and then spatially sliced, leaving a non-contiguous array
    # that some OpenCV operations refuse to accept.
    x, y, w, h = percent_roi_to_pixels(frame.shape[:2], roi_pct)
    return np.ascontiguousarray(frame[y:y + h, x:x + w])


def _worker_fn(video_path: str, start: int, end: int,
               jobs: list[DispatchJob], sub_batch_size: int) -> dict[str, list]:
    """Worker entry point — must be top-level so ProcessPoolExecutor can pickle it
    by reference on Windows (spawn model).

    Internally runs a fetch thread that reads sub-batches via decord get_batch,
    crops per-job ROIs, and drops the full frames before putting ROIs on the queue.
    The main thread consumes the queue and calls each job's process_fn.
    Heroes and other per-worker state are expected to be set by the pool initializer,
    not passed as arguments here.
    """
    from decord import VideoReader as _VR, cpu as _cpu

    vr = _VR(video_path, ctx=_cpu(0))

    job_frame_sets: dict[str, set[int]] = {
        job.name: set(range(start + (-start % job.step), end, job.step)) for job in jobs
    }
    all_indices = sorted(set.union(*job_frame_sets.values()))

    results: dict[str, list] = {job.name: [] for job in jobs}
    q: Queue = Queue(maxsize=2)

    def fetch() -> None:
        for sub in _chunks(all_indices, sub_batch_size):
            batch = vr.get_batch(sub).asnumpy()  # (N, H, W, C) RGB
            processed = []
            for frame_i, frame in zip(sub, batch):
                frame_bgr = frame[:, :, ::-1]  # decord outputs RGB; OpenCV expects BGR
                for job in jobs:
                    if frame_i in job_frame_sets[job.name]:
                        roi = _crop_roi(frame_bgr, job.roi_pct)
                        processed.append((job, frame_i, roi))
            del batch
            q.put(processed)
        q.put(None)

    t = Thread(target=fetch, daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is None:
            break
        for job, frame_i, roi in item:
            results[job.name].extend(job.process_fn(roi, frame_i))

    t.join()
    return results


class FrameDispatcher:
    """Reads a video and fans out cropped ROIs to registered analysis jobs in parallel.

    Workers are assigned contiguous frame ranges so each opens the video once and
    reads forward — no random seeking across the full file per frame.
    Per-worker shared state (e.g. hero reference images) should be loaded via
    initializer/initargs, which the pool calls once per worker at startup.
    """

    def __init__(self, video_path: str,
                 start_time: float = 0.0, end_time: float | None = None,
                 n_workers: int | None = None, chunks_per_worker: int = 4,
                 sub_batch_size: int = 8,
                 initializer: Callable | None = None, initargs: tuple = ()):
        self.video_path = video_path
        self.start_time = start_time
        self.end_time = end_time
        self.n_workers = n_workers or cpu_count()
        self.chunks_per_worker = chunks_per_worker
        self.sub_batch_size = sub_batch_size
        self.initializer = initializer
        self.initargs = initargs
        self._jobs: list[DispatchJob] = []

    def register(self, job: DispatchJob) -> None:
        """Add a job. Validates process_fn picklability immediately so the error
        surfaces here rather than as a cryptic PicklingError inside the executor."""
        try:
            pickle.dumps(job.process_fn)
        except (pickle.PicklingError, AttributeError, TypeError) as e:
            raise ValueError(
                f"DispatchJob '{job.name}': process_fn must be a top-level function, "
                f"not a lambda or closure. Got: {job.process_fn!r}"
            ) from e
        self._jobs.append(job)

    def run(self) -> dict[str, list]:
        """Process the video and return results as dict[job_name, flat list of results].

        Uses len(ranges) as max_workers rather than n_workers so short videos
        (fewer chunks than workers) don't spin up idle processes.
        """
        if not self._jobs:
            return {}

        vr = VideoReader(self.video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        del vr

        start_f = int(self.start_time * fps)
        end_f = int(self.end_time * fps) if self.end_time is not None else total_frames
        end_f = min(end_f, total_frames)

        if start_f >= end_f:
            return {job.name: [] for job in self._jobs}

        frame_range = end_f - start_f
        n_chunks = self.n_workers * self.chunks_per_worker
        chunk_size = max(1, frame_range // n_chunks)
        ranges = []
        for i in range(n_chunks):
            s = start_f + i * chunk_size
            e = start_f + (i + 1) * chunk_size if i < n_chunks - 1 else end_f
            if s < end_f:
                ranges.append((s, e))

        all_results: dict[str, list] = {job.name: [] for job in self._jobs}

        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=self.initializer,
            initargs=self.initargs,
        ) as executor:
            futures = {
                executor.submit(_worker_fn, self.video_path, s, e, self._jobs, self.sub_batch_size): (s, e)
                for s, e in ranges
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing frames"):
                for name, entries in future.result().items():
                    all_results[name].extend(entries)

        return all_results
