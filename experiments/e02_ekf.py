"""E02 — motion-детекция + фильтр Калмана по 3D.

Отличие от `e01_motion`: детекции те же, но кадры больше не решаются
независимо. Трек ведётся фильтром Калмана (постоянная скорость), который

  1. сглаживает шумные триангуляции между кадрами;
  2. даёт предсказание, по которому гейтятся кандидаты в каждой камере
     (вместо «последней точки»);
  3. отбрасывает измерения, не прошедшие порог хи-квадрат
     (расстояние Махаланобиса), — одиночный ложный бокс не уводит трек;
  4. продолжает траекторию на кадрах без измерения (по модели движения),
     поэтому покрытие растёт, хоть точность в пропусках и падает.

Измерение — не центр 2D-бокса, а **точка касания земли** (середина нижнего
края повернутого бокса). Центр силуэта зависит от ракурса и окклюзий, а
низ бокса — это колёса на земле, он устойчивее.

Идеи (EKF в 3D, измерение по низу рамки, хи-квадрат гейт, шум модели
движения 12 м/с²) повторяют ветку `baseline-online-tracker` и переносятся
в общий слой: детектор остаётся наш, метрики и видео — общие.
"""

from __future__ import annotations

import itertools

import numpy as np

from core.camera import Camera
from core.dataset import Scene
from core.detect import Detection, MotionDetector
from core.geometry import triangulate
from core.validation import Prediction
from experiments.base import Experiment
from experiments.e01_motion import MotionTriangulation, _plausible, _ray_distance

FPS = 30.0
CHI2_GATE = {1: 10.8, 2: 13.8, 3: 16.3}  # 99.9% для 1..3 измерений
HEADING_MIN_SPEED = 2.0  # м/с


class Ekf3D:
    """Постоянная скорость, [x y z vx vy vz] в сантиметрах и см/с."""

    def __init__(self, pos: np.ndarray, pos_var: float = 25.0**2):
        self.x = np.r_[pos, np.zeros(3)]
        self.P = np.diag([pos_var] * 3 + [1500.0**2] * 3)

    def predict(self, dt_s: float, accel_xy: float, accel_z: float) -> None:
        dt = float(dt_s)
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        G = np.vstack([np.eye(3) * dt**2 / 2.0, np.eye(3) * dt])
        q = (np.array([accel_xy, accel_xy, accel_z]) * 100.0) ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ np.diag(q) @ G.T

    def innovation(self, z: np.ndarray, cov: np.ndarray):
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        r = z - self.x[:3]
        S = H @ self.P @ H.T + cov
        return r, H, S, float(r @ np.linalg.solve(S, r))

    def update(self, r: np.ndarray, H: np.ndarray, S: np.ndarray) -> None:
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ r
        self.P = (np.eye(6) - K @ H) @ self.P


def _measure_point(det: Detection, mode: str) -> np.ndarray:
    return det.ground_point if mode == "ground" else det.center


class EkfMotion(MotionTriangulation):
    name = "e02_ekf"

    def __init__(
        self,
        measure: str = "ground",
        accel_xy: float = 12.0,
        accel_z: float = 1.0,
        offset_fwd_m: float = 11.3,
        offset_up_m: float = 1.8,
        noise_floor_m: float = 3.0,
        reset_misses: int = 15,
        smoothing: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.measure = measure
        self.accel_xy = float(accel_xy)
        self.accel_z = float(accel_z)
        self.offset_fwd_m = float(offset_fwd_m)
        self.offset_up_m = float(offset_up_m)
        self.noise_floor_m = float(noise_floor_m)
        self.reset_misses = int(reset_misses)

    # ------------------------------------------------------------------
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

        track: Ekf3D | None = None
        heading = 0.0
        xyz: dict[int, np.ndarray] = {}
        n_cams: dict[int, int] = {}
        boxes: dict[int, dict[int, tuple]] = {}
        residuals: dict[int, float] = {}
        measured: dict[int, np.ndarray] = {}
        gaps: dict[int, int] = {}

        last_frame: int | None = None
        misses = 0
        for frame in sorted(scene.target):
            cameras = {
                idx: Camera.from_track(tr, frame, scene.width, scene.height)
                for idx, tr in scene.cameras.items()
            }
            if track is not None and last_frame is not None:
                track.predict((frame - last_frame) / FPS, self.accel_xy, self.accel_z)

            predicted = track.x[:3] if track is not None else None

            cand: dict[int, list[Detection]] = {}
            for idx in sorted(cameras):
                gate = None
                if predicted is not None:
                    uv, depth = cameras[idx].project(predicted)
                    if cameras[idx].in_image(uv, depth)[0]:
                        gate = uv[0]
                cand[idx] = self._candidates(detectors[idx].detect(frame), gate)

            cams_with = [i for i in sorted(cand) if cand[i]]
            best: tuple[float, np.ndarray, np.ndarray, tuple[int, ...], list[Detection], float] | None = None
            for k in range(len(cams_with), self.min_cameras - 1, -1):
                for subset in itertools.combinations(cams_with, k):
                    for combo in itertools.product(*[cand[i] for i in subset]):
                        pts = [_measure_point(d, self.measure) for d in combo]
                        origins = np.stack([cameras[i].ray(p)[0] for i, p in zip(subset, pts)])
                        dirs = np.stack([cameras[i].ray(p)[1] for i, p in zip(subset, pts)])
                        sigmas = np.array([cameras[i].angular_sigma(self.sigma_px) for i in subset])
                        point, cov = triangulate(origins, dirs, sigmas)
                        if not _plausible(point, [cameras[i] for i in subset], self.max_range_m):
                            continue
                        res = float(np.mean([_ray_distance(point, o, d) for o, d in zip(origins, dirs)])) / 100.0
                        if res > self.residual_limit_m:
                            continue
                        score = res - 0.5 * k
                        if best is None or score < best[0]:
                            best = (score, point, cov, subset, list(combo), res)
                if best is not None:
                    break

            if best is None:
                if track is None:
                    continue
                misses += 1
                if misses >= self.reset_misses:
                    track, misses = None, 0
                    continue
                xyz[frame] = self._answer(track.x[:3], heading, 0.0)
                n_cams[frame] = 0
                gaps[frame] = frame - last_frame if last_frame is not None else 0
                last_frame = frame
                continue

            _, point, cov, subset, combo, res = best
            if track is None:
                track = Ekf3D(point, pos_var=25.0**2)
            else:
                # пол шума: ковариация триангуляции не учитывает смещение
                # центра рамки от ракурса, поэтому она слишком оптимистична
                noise = cov + np.eye(3) * (self.noise_floor_m * 100.0) ** 2
                r, H, S, chi2 = track.innovation(point, noise)
                if chi2 < CHI2_GATE[3]:
                    track.update(r, H, S)
                    misses = 0
                else:
                    res = float("nan")
                    misses += 1
                    if misses >= self.reset_misses:
                        track = None
                        continue

            speed = float(np.hypot(track.x[3], track.x[4])) / 100.0
            if speed > HEADING_MIN_SPEED:
                heading = float(np.arctan2(track.x[4], track.x[3]))
            xyz[frame] = self._answer(track.x[:3], heading, speed)
            n_cams[frame] = len(subset)
            boxes[frame] = {int(i): d.rect for i, d in zip(subset, combo)}
            measured[frame] = point
            residuals[frame] = res
            last_frame = frame

        return Prediction(
            name=self.name,
            xyz=xyz,
            n_cams=n_cams,
            meta={
                "boxes": boxes,
                "measure": self.measure,
                "accel_xy": self.accel_xy,
                "offset_fwd_m": self.offset_fwd_m,
                "offset_up_m": self.offset_up_m,
                "residual_m": residuals,
                "measured": measured,
                "gaps": gaps,
                "frames": len(xyz),
            },
        )

    # ------------------------------------------------------------------
    def _answer(self, pos: np.ndarray, heading: float, speed: float) -> np.ndarray:
        """Точка-ответ: сдвиг от измеренной точки к переду/верху объекта по курсу."""
        out = pos.copy()
        out[2] += self.offset_up_m * 100.0
        if speed > HEADING_MIN_SPEED or self.measure == "ground":
            out[0] += self.offset_fwd_m * 100.0 * np.cos(heading)
            out[1] += self.offset_fwd_m * 100.0 * np.sin(heading)
        return out
