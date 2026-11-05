"""Заполнение провалов меняет только кадры без позы трекера и двигает ответ плавно."""
from types import SimpleNamespace

import numpy as np

from online.estimation.gap_fill import GapFill, triangulate_pair
from online.geometry import DEFAULT_TARGET
from online.tracker import Estimate


def make():
    tracker = SimpleNamespace(current_views={}, groups={}, segmenter=None)
    return tracker, GapFill(tracker, DEFAULT_TARGET, 30.0)


def drive(fill, frames, step=0.3):
    for i in range(frames):
        fill.step(i, Estimate(np.array([step * i, 0.0, 3.0, 0.0, 0.0]), "multi", {}, 0.9), {})


def test_tracker_pose_passes_through_unchanged():
    _, fill = make()
    estimate = Estimate(np.array([1.0, 2.0, 3.0, 0.1, 0.0]), "multi", {}, 0.9)
    assert fill.step(0, estimate, {}) is estimate


def test_gap_continues_recent_motion_on_the_road():
    _, fill = make()
    drive(fill, 30)
    start = fill.pose[:3].copy()
    out = [fill.step(30 + i, Estimate(None, "lost", {}, 0.0), {}) for i in range(30)]
    assert all(e.status == "predicted" and e.pose is not None for e in out)
    assert np.isclose(out[-1].pose[2], 3.0, atol=1e-6)            # высота дороги
    moved = out[-1].pose[:3] - start
    assert 8.5 < moved[0] < 9.5 and abs(moved[1]) < 1e-6           # 30 кадров × 0,3 м
    steps = [np.linalg.norm(b.pose[:3] - a.pose[:3]) for a, b in zip(out, out[1:])]
    assert max(steps) < 0.31


def test_first_filled_frame_continues_tracker_answer():
    _, fill = make()
    drive(fill, 30)
    last = fill.pose[:3].copy()
    first = fill.step(30, Estimate(None, "lost", {}, 0.0), {})
    assert np.linalg.norm(first.pose[:3] - last) < 0.31            # без скачка на входе в провал


def test_step_is_limited():
    _, fill = make()
    drive(fill, 5, step=0.0)
    before = fill.pose[:3].copy()
    fill.x[:2] += 50.0                                             # фильтр «увидел» машину далеко
    out = fill.step(5, Estimate(None, "lost", {}, 0.0), {})
    assert np.isclose(np.linalg.norm(out.pose[:3] - before), fill.step_limit())


def test_triangulation_of_two_rays():
    a = (np.array([0.0, 0.0, 10.0]), np.array([1.0, 0.0, -0.1]) / np.linalg.norm([1.0, 0.0, -0.1]))
    target = np.array([100.0, 0.0, 0.0])
    origin = np.array([0.0, 100.0, 10.0])
    b = (origin, (target - origin) / np.linalg.norm(target - origin))
    point, gap = triangulate_pair(a, b)
    assert np.allclose(point, target, atol=1e-6) and gap < 1e-6


def test_nothing_known_stays_empty():
    _, fill = make()
    assert fill.step(0, Estimate(None, "lost", {}, 0.0), {}).pose is None
