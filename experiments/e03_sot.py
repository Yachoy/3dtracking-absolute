"""E03 — классические SOT-трекеры (CSRT / KCF / MOSSE) на каждой камере.

Постановка из обсуждения ТЗ: объект не обязан быть «фурой», его могут
заменить, поэтому класса нет. На первом кадре, где объект виден в камере,
ему даётся рамка (в эксперименте её роль играет проекция известной модели —
стенд-ин для «оператор обвёл объект»), дальше трекер ведёт её по внешнему
виду, без детектора и без обучения.

Структура намеренно разделена на две части:

1. `track_camera` — независимый проход по одной камере (медленный, идёт в
   отдельном процессе, 5 камер = 5 ядер);
2. `SotMultiCamera.run` — фузия наблюдений в 3D (дешёвая, последовательная).

3D собирается из рамок: низ рамки — точка касания земли, высота рамки даёт
дальность `d = fy·H/h` (H — априорная высота объекта). Наблюдения камер
смешиваются медианой (одна сбежавшая камера не тянет всех), между кадрами
без измерения — экстраполяция по последней скорости, не дольше `max_gap`.

    python -m tools.validate --exp e03_sot --param tracker=csrt
"""

from __future__ import annotations

import cv2
import numpy as np

from core.camera import Camera
from core.dataset import Scene, load_scene
from core.geometry import triangulate
from core.object_model import ObjectModel, heading_from_positions
from core.parallel import per_camera
from core.sot import SotTracker, union_box
from core.validation import Prediction
from experiments.base import Experiment


def _usable(box, width: float, height: float, min_w: float, min_h: float, margin: float) -> bool:
    """Рамка целиком в кадре и достаточно крупная, чтобы трекер взялся."""
    if box is None:
        return False
    x0, y0, x1, y1 = box
    return (
        x0 > margin and y0 > margin and x1 < width - margin and y1 < height - margin
        and (x1 - x0) >= min_w and (y1 - y0) >= min_h
    )


def track_camera(job) -> tuple[int, int | None, dict[int, tuple[float, float, float, float]]]:
    """Проход одной камеры: инициализация «оператором» и ведение SOT.

    Живёт в отдельном процессе, поэтому заново читает сцену и открывает своё
    видео — но возвращает наружу только словарь рамок по кадрам.
    """
    data_root, cam_id, opts = job
    scene = load_scene(data_root)
    model = ObjectModel()
    frames = sorted(scene.target)
    gt = {f: p.loc for f, p in scene.target.items()}
    heading = {f: heading_from_positions(gt, f) for f in frames}
    track = scene.cameras[cam_id]

    def model_box(frame: int):
        cam = Camera.from_track(track, frame, scene.width, scene.height)
        return union_box(model.boxes_2d(cam, gt[frame], heading[frame]))

    init_frame = None
    for f in frames:
        box = model_box(f)
        if _usable(box, scene.width, scene.height, opts["min_width"], opts["min_height"], opts["margin"]):
            init_frame = f
            break

    boxes: dict[int, tuple[float, float, float, float]] = {}
    if init_frame is None or track.video is None:
        return cam_id, init_frame, boxes

    tracker = SotTracker(opts["tracker"])
    cap = cv2.VideoCapture(str(track.video))
    idx = 0
    started = False
    try:
        for f in frames:
            while idx < f:
                if not cap.grab():
                    break
                idx += 1
            ok, img = cap.read()
            if not ok:
                break
            idx += 1
            if not started:
                if f != init_frame:
                    continue
                box = model_box(f)
                if not _usable(box, scene.width, scene.height, opts["min_width"], opts["min_height"], opts["margin"]):
                    continue
                tracker.init(img, box)
                started = True
                boxes[f] = tuple(float(v) for v in box)
                continue
            box = tracker.update(img)
            if box is None:
                # без 3D-оценки переинициализировать нечем: дальше в этой камере тишина
                break
            boxes[f] = tuple(float(v) for v in box)
    finally:
        cap.release()
    return cam_id, init_frame, boxes


def _anchor_causal(points: dict[int, np.ndarray], offset: np.ndarray, span: int = 6) -> dict[int, np.ndarray]:
    """Центр силуэта -> точка-ответ, курс только по прошлым кадрам.

    offset — где центр рамки относительно ответа в системе объекта:
    (вперёд, влево, вверх) в сантиметрах; ответ = центр − повёрнутый offset.
    """
    frames = sorted(points)
    out: dict[int, np.ndarray] = {}
    heading = 0.0
    for i, f in enumerate(frames):
        lo = max(0, i - span)
        d = points[f] - points[frames[lo]]
        if np.linalg.norm(d[:2]) > 50.0:
            heading = float(np.degrees(np.arctan2(d[1], d[0])))
        fwd = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading)), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        out[f] = points[f] - (fwd * offset[0] + up * offset[2])
    return out


class SotMultiCamera(Experiment):
    name = "e03_sot"

    def __init__(
        self,
        tracker: str = "csrt",
        min_width: int = 40,
        min_height: int = 16,
        margin: int = 6,
        answer_fwd_m: float = 10.0,
        answer_up_m: float = 0.0,
        sigma_px: float = 3.0,
        min_cameras: int = 1,
        measure: str = "size",
        object_height_m: float = 4.0,
        max_gap: int = 15,
        workers: int = 5,
        **kwargs,
    ) -> None:
        self.tracker_kind = tracker
        self.min_width = int(min_width)
        self.min_height = int(min_height)
        self.margin = int(margin)
        self.answer_fwd_m = float(answer_fwd_m)
        self.answer_up_m = float(answer_up_m)
        self.sigma_px = float(sigma_px)
        self.min_cameras = int(min_cameras)
        self.measure = measure
        self.object_height_cm = float(object_height_m) * 100.0
        self.max_gap = int(max_gap)
        self.workers = int(workers)

    # ------------------------------------------------------------------ run
    def run(self, scene: Scene) -> Prediction:
        cam_ids = sorted(scene.cameras)
        opts = {
            "tracker": self.tracker_kind,
            "min_width": self.min_width,
            "min_height": self.min_height,
            "margin": self.margin,
        }
        jobs = [(str(scene.root), i, opts) for i in cam_ids]
        passes = per_camera(track_camera, jobs, workers=self.workers)
        boxes = {i: b for i, _, b in passes}
        inits = {i: f for i, f, _ in passes}

        frames = sorted(scene.target)
        heights: dict[int, list[float]] = {i: [] for i in cam_ids}
        raw: dict[int, np.ndarray] = {}
        raw_cams: dict[int, int] = {}
        last_point: np.ndarray | None = None
        last_frame: int | None = None
        velocity = np.zeros(3)

        for frame in frames:
            cam_frame = {
                i: Camera.from_track(scene.cameras[i], frame, scene.width, scene.height)
                for i in cam_ids
            }
            point = None
            used = 0
            if self.measure == "size":
                est = []
                for i in cam_ids:
                    box = boxes[i].get(frame)
                    if box is None:
                        continue
                    cam = cam_frame[i]
                    heights[i].append(max(1.0, abs(box[3] - box[1])))
                    del heights[i][:-7]
                    h_px = float(np.median(heights[i]))
                    if h_px < 8.0:
                        continue
                    depth = cam.fy * self.object_height_cm / h_px  # см
                    u = 0.5 * (box[0] + box[2])
                    origin, direction = cam.ray(np.array([u, float(box[3])]))
                    est.append(origin + direction * depth)
                if len(est) >= self.min_cameras:
                    point = np.median(np.array(est), axis=0)
                    used = len(est)
            else:  # rays: чистая триангуляция по низу рамок, без априорной высоты
                origins, dirs = [], []
                for i in cam_ids:
                    box = boxes[i].get(frame)
                    if box is None:
                        continue
                    cam = cam_frame[i]
                    u = 0.5 * (box[0] + box[2])
                    origin, direction = cam.ray(np.array([u, float(box[3])]))
                    origins.append(origin)
                    dirs.append(direction)
                if len(origins) >= max(2, self.min_cameras):
                    point, _, used = self._robust(origins, dirs)
                    if used < max(2, self.min_cameras):
                        point = None

            # --- причинная фильтрация: прогноз только из прошлого
            if point is not None and abs(point[2]) > 15_000.0:
                point = None
            if point is not None and last_point is not None and last_frame is not None and frame > last_frame:
                dt_s = (frame - last_frame) / scene.fps
                predicted = last_point + velocity * dt_s
                if float(np.linalg.norm(point - predicted)) > 4_000.0:  # 40 м
                    point = None
            if point is not None:
                if last_point is not None and last_frame is not None and frame > last_frame:
                    dt_s = (frame - last_frame) / scene.fps
                    measured_v = (point - last_point) / dt_s
                    velocity = 0.7 * velocity + 0.3 * measured_v
                    speed = float(np.linalg.norm(velocity))
                    if speed > 4_000.0:
                        velocity *= 4_000.0 / speed
                last_point, last_frame = point, frame
                raw[frame] = point
                raw_cams[frame] = used
            elif last_point is not None and last_frame is not None and frame - last_frame <= self.max_gap:
                raw[frame] = last_point + velocity * ((frame - last_frame) / scene.fps)
                raw_cams[frame] = 0

        offset = np.array([self.answer_fwd_m * 100.0, 0.0, self.answer_up_m * 100.0])
        xyz = _anchor_causal(raw, offset)
        return Prediction(
            name=self.name,
            xyz=xyz,
            n_cams=raw_cams,
            meta={
                "tracker": self.tracker_kind,
                "measure": self.measure,
                "inits": inits,
                "object_height_m": self.object_height_cm / 100.0,
            },
        )

    @staticmethod
    def _robust(origins, dirs, limit_m: float = 25.0):
        """МНК по лучам с отбраковкой: луч, ушедший сильно мимо, не тянет точку."""
        sigmas = [1.0 / 500.0 for _ in origins]
        point, cov = triangulate(origins, dirs, sigmas)
        if len(origins) < 3:
            return point, cov, len(origins)
        res = np.array([
            float(np.linalg.norm((point - o) - d * float((point - o) @ d)))
            for o, d in zip(origins, dirs)
        ])
        keep = res <= max(limit_m * 100.0, 3.0 * float(np.median(res)))
        if keep.sum() >= 2 and (~keep).any():
            point, cov = triangulate(
                [o for o, k in zip(origins, keep) if k],
                [d for d, k in zip(dirs, keep) if k],
                [s for s, k in zip(sigmas, keep) if k],
            )
        return point, cov, int(keep.sum())
