"""Заполнение провалов меняет только кадры без позы трекера."""
from types import SimpleNamespace

import numpy as np

from online.estimation.gap_fill import GapFill
from online.geometry import DEFAULT_TARGET
from online.tracker import Estimate


def make():
    tracker = SimpleNamespace(current_views={}, groups={}, velocity=np.array([0.3, 0.0, 0.0]), segmenter=None)
    return tracker, GapFill(tracker, DEFAULT_TARGET, 30.0)


def test_tracker_pose_passes_through_unchanged():
    tracker, fill = make()
    estimate = Estimate(np.array([1.0, 2.0, 3.0, 0.1, 0.0]), "multi", {}, 0.9)
    assert fill.step(0, estimate, {}) is estimate


def test_gap_is_filled_with_ground_level_forecast():
    tracker, fill = make()
    fill.step(0, Estimate(np.array([0.0, 0.0, 3.0, 0.0, 0.0]), "multi", {}, 0.9), {})
    out = [fill.step(i, Estimate(None, "lost", {}, 0.0), {}) for i in range(1, 31)]
    assert all(e.status == "predicted" and e.pose is not None for e in out)
    assert np.isclose(out[-1].pose[2], 3.0)                 # высота дороги
    assert 7.0 < out[-1].pose[0] < 9.0                      # 30 кадров × 0,3 м с затуханием
    assert np.linalg.norm(fill.velocity) < 0.3


def test_nothing_known_stays_empty():
    tracker, fill = make()
    estimate = Estimate(None, "lost", {}, 0.0)
    assert fill.step(0, estimate, {}).pose is None
