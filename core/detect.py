"""Детекция движущегося объекта по вычитанию фона.

Камеры в сцене статичны (кроме CAM5, которая панорамирует), поэтому
фон можно собрать один раз: медиана по кадрам даёт «пустую» сцену без
фургона. Для панорамирующей камеры кадры предварительно приводятся к
ориентации опорного кадра поворотным гомографическим преобразованием
(вращение вокруг оптического центра не меняет внутреннюю геометрию).

Детекция — порог по модулю разницы с фоном, морфология, связные
компоненты. Результат — повернутый прямоугольник (`rect`) и обычный
bbox (`box`) в пикселях исходного кадра.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .dataset import CameraTrack, Scene


# --------------------------------------------------------------------- утилиты


def _matrix(cam: Camera, width: int, height: int) -> np.ndarray:
    """Матрица интринсиков камеры для изображения заданного размера."""
    sx, sy = width / cam.width, height / cam.height
    return np.array(
        [[cam.fx * sx, 0.0, width / 2.0], [0.0, cam.fy * sy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rotation(cam: Camera) -> np.ndarray:
    """Матрица «мир → камера»: строки right, down, forward."""
    fwd, right, down = cam.basis
    return np.stack([right, down, fwd])


def orientation_homography(
    src: Camera, dst: Camera, src_size: tuple[int, int], dst_size: tuple[int, int]
) -> np.ndarray:
    """Пиксели `src` → пиксели `dst` при чистом вращении камеры вокруг центра."""
    ks = _matrix(src, *src_size)
    kd = _matrix(dst, *dst_size)
    return kd @ _rotation(dst) @ _rotation(src).T @ np.linalg.inv(ks)


@dataclass(frozen=True)
class Detection:
    """Найденный объект в одном кадре одной камеры (пиксели исходного кадра)."""

    frame: int
    cam: int
    box: np.ndarray  # (4,) x0, y0, x1, y1
    rect: tuple[float, float, float, float, float]  # cx, cy, w, h, angle повернутого прямоугольника
    area: float

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0])

    @property
    def size(self) -> tuple[float, float]:
        return float(self.box[2] - self.box[0]), float(self.box[3] - self.box[1])

    @property
    def ground_point(self) -> np.ndarray:
        """Середина нижнего края повернутого прямоугольника — где объект касается земли."""
        c = self.corners()
        mids = [(c[i] + c[(i + 1) % 4]) / 2.0 for i in range(4)]
        return max(mids, key=lambda m: m[1])

    @property
    def vertical_extent(self) -> float:
        c = self.corners()
        return float(c[:, 1].max() - c[:, 1].min())

    def corners(self) -> np.ndarray:
        """Четыре угла повернутого прямоугольника."""
        cx, cy, w, h, angle = self.rect
        box = cv2.boxPoints(((cx, cy), (w, h), angle))
        return np.asarray(box, dtype=np.float64)


# ------------------------------------------------------------------- фон


class BackgroundModel:
    """Медианный фон камеры.

    Кадры разбиваются на группы по ориентации (при панорамировании),
    внутри группы все кадры приводятся к опорной ориентации поворотной
    гомографией, медиана даёт «пустую» сцену. Для кадра берётся группа
    с ближайшей опорной ориентацией, поэтому искажение пересчёта мало.
    Вместе с фоном хранится маска валидных пикселей (что реально видели
    кадры группы) — на краях она нулевая и там детекция отключается.
    """

    def __init__(
        self,
        track: CameraTrack,
        width: int,
        height: int,
        scale: float = 0.5,
        samples: int = 96,
        max_delta_deg: float = 15.0,
    ) -> None:
        if track.video is None:
            raise ValueError(f"у камеры {track.name} нет видео")
        self.track = track
        self.scale = scale
        self.width = int(round(width * scale))
        self.height = int(round(height * scale))
        self.full_width = width
        self.full_height = height

        frames = sorted(track.poses)
        poses = np.array([track.poses[f].rot for f in frames])
        yaw = np.degrees(np.unwrap(np.radians(poses[:, 2])))
        self.frames = np.array(frames)
        self.yaw = yaw

        # группы кадров с близкой ориентацией
        groups: list[list[int]] = []
        current = [0]
        for i in range(1, len(frames)):
            if abs(yaw[i] - yaw[current[0]]) > max_delta_deg:
                groups.append(current)
                current = [i]
            else:
                current.append(i)
        groups.append(current)

        self.refs: list[int] = []  # индекс группы -> кадр опорной ориентации
        self.plates: list[np.ndarray] = []
        self.valid: list[np.ndarray] = []
        cap = cv2.VideoCapture(str(track.video))
        for g in groups:
            ref = frames[g[len(g) // 2]]
            self.refs.append(ref)
            picks = [g[i] for i in np.linspace(0, len(g) - 1, min(samples, len(g))).astype(int)]
            ref_cam = Camera.from_track(track, ref, width, height)
            stack: list[np.ndarray] = []
            counts = np.zeros((self.height, self.width), np.float32)
            for f in dict.fromkeys(picks):
                image = _read(cap, f, scale)
                if image is None:
                    continue
                cam = Camera.from_track(track, f, width, height)
                h = orientation_homography(
                    cam, ref_cam, (self.width, self.height), (self.width, self.height)
                )
                warped = cv2.warpPerspective(
                    image, h, (self.width, self.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                stack.append(warped.astype(np.float32))
                corners = np.array(
                    [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]], dtype=np.float32
                ).reshape(-1, 1, 2)
                quad = cv2.perspectiveTransform(corners, h).reshape(-1, 2)
                quad = np.clip(quad, [-1e4, -1e4], [self.width + 1e4, self.height + 1e4]).astype(np.int32)
                cv2.fillConvexPoly(counts, quad, 1.0, lineType=cv2.LINE_8)
            if not stack:
                continue
            plate = np.median(np.stack(stack), axis=0).astype(np.uint8)
            self.plates.append(plate)
            self.valid.append(counts > 0)
        cap.release()
        if not self.plates:
            raise RuntimeError(f"{track.name}: не удалось собрать фон")
        self.group_index = [g[0] for g in groups]

    # ------------------------------------------------------------------
    def _group_for(self, frame: int) -> int:
        """Индекс группы с ближайшей опорной ориентацией."""
        if len(self.plates) == 1:
            return 0
        i = int(np.searchsorted(np.array([self.refs[k] for k in range(len(self.refs))]), frame, side="right") - 1)
        i = min(max(i, 0), len(self.plates) - 1)
        # сравним соседние группы по разнице курса
        best, best_d = i, None
        yaw_f = float(self.track.poses[frame].rot[2])
        for k in (i - 1, i):
            if 0 <= k < len(self.plates):
                ref = Camera.from_track(self.track, self.refs[k], self.full_width, self.full_height)
                d = abs(((yaw_f - float(ref.rot[2]) + 180) % 360) - 180)
                if best_d is None or d < best_d:
                    best, best_d = k, d
        return best

    def plate_at(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Фон, повёрнутый под ориентацию кадра, и маска валидных пикселей."""
        k = self._group_for(frame)
        cam = Camera.from_track(self.track, frame, self.full_width, self.full_height)
        ref = Camera.from_track(self.track, self.refs[k], self.full_width, self.full_height)
        valid = self.valid[k]
        plate = self.plates[k]
        if np.allclose(cam.rot, ref.rot, atol=1e-9):
            return plate, valid
        h = orientation_homography(ref, cam, (self.width, self.height), (self.width, self.height))
        warped = cv2.warpPerspective(
            plate, h, (self.width, self.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        warped_valid = cv2.warpPerspective(
            valid.astype(np.uint8) * 255, h, (self.width, self.height), flags=cv2.INTER_NEAREST
        )
        return warped, warped_valid > 0


def _read(cap: cv2.VideoCapture, frame: int, scale: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, image = cap.read()
    if not ok or image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


# --------------------------------------------------------------- детектор


class MotionDetector:
    """Детектор «всё, что отличается от фона» с фильтрами по площади и форме."""

    def __init__(
        self,
        scene: Scene,
        cam: int,
        threshold: int = 20,
        min_area: float = 40.0,
        max_area_frac: float = 0.25,
        scale: float = 0.5,
        samples: int = 64,
    ) -> None:
        self.scene = scene
        self.cam = cam
        self.threshold = threshold
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.scale = scale
        track = scene.cameras[cam]
        self.background = BackgroundModel(track, scene.width, scene.height, scale=scale, samples=samples)
        self.video = Path(track.video) if track.video is not None else None
        self._cap: cv2.VideoCapture | None = None
        self._next: int = -1

    # ------------------------------------------------------------ захват
    def _gray(self, frame: int, image: np.ndarray | None) -> np.ndarray | None:
        """Серый кадр в масштабе детекции: из видео (файл держим открытым) или из готового BGR."""
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self.scale != 1.0:
                gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            return gray
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.video))
            self._next = -1
        if self._cap is None or not self._cap.isOpened():
            return None
        if frame < self._next or self._next < 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            self._next = int(frame)
        while self._next < frame:
            if not self._cap.grab():
                return None
            self._next += 1
        ok, img = self._cap.read()
        if not ok:
            return None
        self._next += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return gray

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def detect(self, frame: int, image: np.ndarray | None = None) -> list[Detection]:
        """Все кандидаты в кадре, отсортированные по площади."""
        gray = self._gray(frame, image)
        if gray is None:
            return []

        plate, valid = self.background.plate_at(frame)
        diff = cv2.absdiff(gray, plate)
        mask = ((diff > self.threshold) & valid).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        limit = self.max_area_frac * mask.shape[0] * mask.shape[1]
        out: list[Detection] = []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area < self.min_area or area > limit:
                continue
            component = (labels[y : y + h, x : x + w] == k).astype(np.uint8)
            contur = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if not contur:
                continue
            (cx, cy), (rw, rh), angle = cv2.minAreaRect(contur[0])
            s = 1.0 / self.scale
            out.append(
                Detection(
                    frame=frame,
                    cam=self.cam,
                    box=np.array([x * s, y * s, (x + w) * s, (y + h) * s]),
                    rect=((cx + x) * s, (cy + y) * s, rw * s, rh * s, angle),
                    area=float(area) * s * s,
                )
            )
        out.sort(key=lambda d: d.area, reverse=True)
        return out

    def mask(self, frame: int, image: np.ndarray | None = None) -> np.ndarray:
        """Бинарная маска (в масштабе детекции) — для отладки и фитинга."""
        gray = self._gray(frame, image)
        if gray is None:
            return np.zeros((self.background.height, self.background.width), np.uint8)
        plate, valid = self.background.plate_at(frame)
        diff = cv2.absdiff(gray, plate)
        mask = ((diff > self.threshold) & valid).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
