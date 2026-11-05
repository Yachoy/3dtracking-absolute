"""Опции захвата ConsensusTracker: обрезанная краем маска и одна камера.

Проверка без нейросетей: силуэт фуры по эталону Way1 (кадр 80) проецируется в камеры 1 и 2;
в камере 2 его обрезает нижний край кадра — ровно случай, когда трекер не захватывал фуру.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.camera import Camera
from online.estimation.consensus import ConsensusTracker, SingleCameraGate
from online.geometry import DEFAULT_TARGET, Observation, Rot, View, center_of
from online.perception.vision import Instance

DATA = Path(__file__).resolve().parents[1] / "data" / "test"
REFERENCE = DATA / "object_transform_SM_AA_track.csv"


def reference_pose(frame):
    rows = {}
    for row in REFERENCE.read_text(encoding="cp1251").splitlines()[1:]:
        p = row.split(";")
        try:
            rows[int(p[0])] = np.array([float(p[1]), float(p[2]), float(p[3])]) / 100
        except (ValueError, IndexError):
            pass
    a, b = rows[frame - 5], rows[frame + 5]
    return np.r_[rows[frame], np.arctan2(b[1] - a[1], b[0] - a[0]), 0.0]


def silhouette_instance(view, pose, width=1920, height=1080):
    mask = np.zeros((height, width), np.uint8)
    for part in DEFAULT_TARGET.parts:
        uv, _ = view.project(np.asarray(part) @ Rot(*pose[3:]).T + pose[:3])
        cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)[:, 0].astype(float)
    return Instance(contour, 0.8, 7)


@pytest.mark.skipif(not REFERENCE.exists(), reason="нужны данные Way1 в data/test")
@pytest.mark.parametrize("allow", [False, True])
def test_clipped_pair_acquisition(allow):
    from online.run import sources

    tracks = sources(DATA)
    pose = reference_pose(80)
    tracker = ConsensusTracker(None, DEFAULT_TARGET, 30.0, len(tracks), acquire_clipped=allow)
    tracker.frame = 80
    groups = {}
    for key in (1, 2):
        view = View.of(Camera.from_track(tracks[key], 80, 1920, 1080), model=DEFAULT_TARGET)
        tracker.current_views[key] = view
        groups[key] = [Observation(View.of(view.camera, silhouette_instance(view, pose), DEFAULT_TARGET))]
    assert not groups[1][0].clipped and groups[2][0].clipped  # камера 2: нижний край кадра
    found, support = tracker.acquire(groups)
    if not allow:
        assert found is None and support == []
    else:
        assert found is not None and len(support) == 2
        error = np.linalg.norm(center_of(found, DEFAULT_TARGET) - center_of(pose, DEFAULT_TARGET))
        assert error < 3.0


def test_single_camera_gate_needs_persistence():
    gate = SingleCameraGate(5, 42.0, 30.0)
    results = [gate.update(f, 2, np.array([0.3 * f, 0.0, 3.0])) for f in range(5)]
    assert results == [False, False, False, False, True]
    assert not gate.update(5, 2, np.array([60.0, 0.0, 3.0]))   # скачок — серия заново
    assert not gate.update(6, 3, np.array([60.3, 0.0, 3.0]))   # другая камера — заново


def test_default_tracker_behaviour_unchanged():
    tracker = ConsensusTracker(None, DEFAULT_TARGET, 30.0, 5)
    assert not tracker.acquire_clipped and not tracker.single_camera
