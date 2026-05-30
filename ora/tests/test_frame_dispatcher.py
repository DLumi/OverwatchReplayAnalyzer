import numpy as np
import pytest

from ora.detection.frame_dispatcher import DispatchJob, FrameDispatcher, _chunks


# --- _chunks ---

def test_chunks_splits_evenly():
    assert list(_chunks([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]


def test_chunks_with_remainder():
    assert list(_chunks([1, 2, 3, 4, 5, 6, 7], 3)) == [[1, 2, 3], [4, 5, 6], [7]]


def test_chunks_size_larger_than_list():
    assert list(_chunks([1, 2], 8)) == [[1, 2]]


def test_chunks_empty():
    assert list(_chunks([], 4)) == []


def test_chunks_size_one():
    assert list(_chunks([1, 2, 3], 1)) == [[1], [2], [3]]


# --- DispatchJob ---

def _dummy_fn(roi: np.ndarray, frame_i: int) -> list:
    return []


def test_dispatch_job_fields():
    job = DispatchJob(name="killfeed", roi_pct=(0.73, 0.0, 1.0, 0.25), step=15, process_fn=_dummy_fn)
    assert job.name == "killfeed"
    assert job.roi_pct == (0.73, 0.0, 1.0, 0.25)
    assert job.step == 15
    assert job.process_fn is _dummy_fn


# --- FrameDispatcher.register ---

def test_register_adds_job():
    dispatcher = FrameDispatcher("dummy.mp4")
    job = DispatchJob(name="killfeed", roi_pct=(0.73, 0.0, 1.0, 0.25), step=15, process_fn=_dummy_fn)
    dispatcher.register(job)
    assert len(dispatcher._jobs) == 1
    assert dispatcher._jobs[0].name == "killfeed"


def test_register_multiple_jobs():
    dispatcher = FrameDispatcher("dummy.mp4")
    dispatcher.register(DispatchJob("killfeed", (0.73, 0.0, 1.0, 0.25), 15, _dummy_fn))
    dispatcher.register(DispatchJob("healthbar", (0.0, 0.0, 0.5, 0.1), 8, _dummy_fn))
    assert len(dispatcher._jobs) == 2
    assert {j.name for j in dispatcher._jobs} == {"killfeed", "healthbar"}


def test_register_rejects_lambda():
    dispatcher = FrameDispatcher("dummy.mp4")
    job = DispatchJob(name="bad", roi_pct=(0.0, 0.0, 1.0, 1.0), step=10,
                      process_fn=lambda roi, i: [])
    with pytest.raises(ValueError, match="top-level function"):
        dispatcher.register(job)


def test_run_no_jobs_returns_empty():
    dispatcher = FrameDispatcher("dummy.mp4")
    assert dispatcher.run() == {}


# --- Integration test (requires test video) ---
# TODO (user): provide a short test video clip and uncomment below.
#
# VIDEO_PATH = r"C:\Users\duff_\Desktop\test\<clip>.mp4"
#
# @pytest.mark.integration
# def test_dispatcher_matches_sequential():
#     """Parallel dispatcher results must equal sequential KillFeed.update_from_frame."""
#     from ora.hero import populate_heroes
#     from ora.detection.killfeed import KillFeed
#     from ora.detection.killfeed.worker import init_worker, killfeed_process_fn
#     from ora.utils.video_loader import VideoLoader
#
#     heroes = populate_heroes()
#     step = 15
#     roi_pct = (0.73, 0.0, 1.0, 0.25)
#
#     # Sequential baseline
#     vl = VideoLoader(VIDEO_PATH)
#     kf_seq = KillFeed(roi=roi_pct)
#     frame_i = 0
#     while True:
#         frame = vl.get_frame_image(frame_i)
#         if frame is None:
#             break
#         kf_seq.update_from_frame(frame, heroes, frame_i)
#         frame_i += step
#     vl.close()
#     kf_seq.deduplicate()
#
#     # Parallel dispatcher
#     dispatcher = FrameDispatcher(VIDEO_PATH, n_workers=4, initializer=init_worker, initargs=(heroes,))
#     dispatcher.register(DispatchJob("killfeed", roi_pct, step, killfeed_process_fn))
#     raw = dispatcher.run()
#     kf_par = KillFeed(roi=roi_pct)
#     kf_par.entries = raw["killfeed"]
#     kf_par.deduplicate()
#
#     assert kf_par.entries == kf_seq.entries
