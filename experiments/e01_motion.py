"""E01 — motion-детекция: боксы из изображения, а не из проекции GT.

Схема:
  1. фон = медиана по кадрам (для панорамирующей камеры кадры приводятся
     к общей ориентации), вычитание даёт маску «движущегося»;
  2. связные компоненты → повернутые 2D-боксы (без опоры на модель);
  3. кадры связываются не покадрово, а **согласием триангуляции**:
     перебираются комбинации кандидатов с разных камер, побеждает та,
     лучи которой лучше всего сходятся в одну 3D-точку. Так одинокая
     ложная детекция на дереве не может утащить траекторию;
  4. найденная точка сдвигается из центра бокса в якорь модели
     (передний нижний центр) по известному габариту и курсу движения.

Здесь нет ни GT-точки, ни 3D-модели в детекции: только изображения и
калибровка. Модель нужна лишь для перехода «центр объекта → якорь».
"""

from __future__ import annotations

import itertools

import numpy as np

from core.camera import Camera
from core.dataset import Scene
from core.detect import Detection, MotionDetector
from core.geometry import triangulate
from core.object_model import VehicleModel, anchor_offset, to_anchor
from core.validation import Prediction
from experiments.base import Experiment


def _ray_distance(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    delta = point - origin
    along = float(delta @ direction)
    return float(np.linalg.norm(delta - along * direction))


def _plausible(point: np.ndarray, cameras: list[Camera], max_range_m: float) -> bool:
    """Точка не за камерами, не под землёй и не в бесконечности."""
    if not np.isfinite(point).all():
        return False
    z = point[2] / 100.0
    if z < -20.0 or z > 400.0:
        return False
    return all(cam.distance(point) / 100.0 <= max_range_m for cam in cameras)


class MotionTriangulation(Experiment):
    name = "e01_motion"

    def __init__(
        self,
        threshold: int = 20,
        min_area: float = 40.0,
        max_area_frac: float = 0.05,
        scale: float = 0.5,
        samples: int = 64,
        sigma_px: float = 3.0,
        min_cameras: int = 2,
        max_candidates: int = 3,
        gate_px: float = 150.0,
        residual_limit_m: float = 25.0,
        max_range_m: float = 2000.0,
        aspect_range: tuple[float, float] = (0.45, 5.0),
        anchor_correct: bool = True,
        anchor_span: int = 12,
    ) -> None:
        self.threshold = int(threshold)
        self.min_area = float(min_area)
        self.max_area_frac = float(max_area_frac)
        self.scale = float(scale)
        self.samples = int(samples)
        self.sigma_px = float(sigma_px)
        self.min_cameras = int(min_cameras)
        self.max_candidates = int(max_candidates)
        self.gate_px = float(gate_px)
        self.residual_limit_m = float(residual_limit_m)
        self.max_range_m = float(max_range_m)
        self.aspect_range = aspect_range
        self.anchor_correct = bool(anchor_correct)
        self.anchor_span = int(anchor_span)

    # ------------------------------------------------------------------
    def _candidates(self, detections: list[Detection], gate: np.ndarray | None) -> list[Detection]:
        """Кандидаты кадра: форма похожа на транспорт, рядом с предсказанием."""
        out: list[Detection] = []
        for det in detections:
            w, h = det.size
            if h <= 0 or w <= 0:
                continue
            ratio = w / h
            if not (self.aspect_range[0] <= ratio <= self.aspect_range[1]):
                continue
            if gate is not None and np.linalg.norm(det.center - gate) > self.gate_px:
                continue
            out.append(det)
            if len(out) >= self.max_candidates:
                break
        return out

    def run(self, scene: Scene) -> Prediction:
        detectors = {
            idx: MotionDetector(
                scene,
                idx,
                threshold=self.threshold,
                min_area=self.min_area,
                max_area_frac=self.max_area_frac,
                scale=self.scale,
                samples=self.samples,
            )
            for idx in sorted(scene.cameras)
        }

        xyz_center: dict[int, np.ndarray] = {}
        n_cams: dict[int, int] = {}
        boxes: dict[int, dict[int, tuple]] = {}
        residuals: dict[int, float] = {}

        last_point: np.ndarray | None = None
        last_frame: int | None = None
        velocity = np.zeros(3)

        for frame in sorted(scene.target):
            cameras = {
                idx: Camera.from_track(track, frame, scene.width, scene.height)
                for idx, track in scene.cameras.items()
            }
            predicted = (
                last_point + velocity * (frame - last_frame) if last_point is not None and last_frame is not None else None
            )

            cand: dict[int, list[Detection]] = {}
            for idx in sorted(cameras):
                gate = None
                if predicted is not None:
                    uv, depth = cameras[idx].project(predicted)
                    if cameras[idx].in_image(uv, depth)[0]:
                        gate = uv[0]
                cand[idx] = self._candidates(detectors[idx].detect(frame), gate)

            cams_with = [i for i in sorted(cand) if cand[i]]
            best: tuple[float, np.ndarray, tuple[int, ...], list[Detection], float] | None = None
            for k in range(len(cams_with), self.min_cameras - 1, -1):
                for subset in itertools.combinations(cams_with, k):
                    for combo in itertools.product(*[cand[i] for i in subset]):
                        origins = np.stack([cameras[i].ray(d.center)[0] for i, d in zip(subset, combo)])
                        dirs = np.stack([cameras[i].ray(d.center)[1] for i, d in zip(subset, combo)])
                        sigmas = np.array([cameras[i].angular_sigma(self.sigma_px) for i in subset])
                        point, _ = triangulate(origins, dirs, sigmas)
                        if not _plausible(point, [cameras[i] for i in subset], self.max_range_m):
                            continue
                        res = float(
                            np.mean([_ray_distance(point, o, d) for o, d in zip(origins, dirs)])
                        ) / 100.0
                        if res > self.residual_limit_m:
                            continue
                        score = res - 0.5 * k  # больше камер — лучше при равной невязке
                        if best is None or score < best[0]:
                            best = (score, point, subset, list(combo), res)
                if best is not None:
                    break

            if best is None:
                if last_point is not None and last_frame is not None and frame - last_frame > 30:
                    last_point, velocity = None, np.zeros(3)
                continue

            _, point, subset, combo, res = best
            xyz_center[frame] = point
            n_cams[frame] = len(subset)
            residuals[frame] = res
            boxes[frame] = {int(i): d.rect for i, d in zip(subset, combo)}

            if last_point is not None and last_frame is not None and frame != last_frame:
                step = (point - last_point) / float(frame - last_frame)
                velocity = 0.6 * velocity + 0.4 * step
            last_point, last_frame = point, frame

        xyz = to_anchor(xyz_center, anchor_offset(VehicleModel()), span=self.anchor_span) if self.anchor_correct else xyz_center

        return Prediction(
            name=self.name,
            xyz=xyz,
            n_cams=n_cams,
            meta={
                "boxes": boxes,
                "threshold": self.threshold,
                "min_area": self.min_area,
                "sigma_px": self.sigma_px,
                "residual_m": residuals,
                "frames": len(xyz),
                "anchor_correct": self.anchor_correct,
                "centers": xyz_center,
            },
        )
