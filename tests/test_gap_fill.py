"""Заполнение провалов меняет только кадры без позы трекера и двигает ответ плавно."""
from types import SimpleNamespace

import numpy as np

from online.estimation.gap_fill import GapFill, triangulate_pair
from online.geometry import DEFAULT_TARGET, center_of
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
    start = center_of(fill.pose, DEFAULT_TARGET)
    out = [fill.step(30 + i, Estimate(None, "lost", {}, 0.0), {}) for i in range(30)]
    assert all(e.status == "predicted" and e.pose is not None for e in out)
    assert np.isclose(out[-1].pose[2], 3.0, atol=1e-6)            # высота дороги
    moved = center_of(out[-1].pose, DEFAULT_TARGET) - start
    assert 8.0 < moved[0] < 9.0 and abs(moved[1]) < 1e-6           # 30 кадров × ~0,3 м
    steps = [np.linalg.norm(center_of(b.pose, DEFAULT_TARGET) - center_of(a.pose, DEFAULT_TARGET))
             for a, b in zip(out, out[1:])]
    assert max(steps) < 0.31


def test_step_is_limited():
    _, fill = make()
    drive(fill, 5, step=0.0)
    before = center_of(fill.pose, DEFAULT_TARGET)
    pose = fill.move_to(before + np.array([50.0, 0.0, 0.0]), fill.pose)
    assert np.isclose(np.linalg.norm(center_of(pose, DEFAULT_TARGET) - before), fill.step_limit())


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
