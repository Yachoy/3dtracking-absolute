"""Модель камеры: интринсики из фокусного и сенсора, проекция, лучи."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import CameraTrack, Pose
from .geometry import euler_basis


@dataclass(frozen=True)
class Camera:
    index: int
    loc: np.ndarray  # (3,) см
    rot: np.ndarray  # (3,) roll, pitch, yaw, градусы
    focal_mm: float
    sensor_mm: tuple[float, float]
    width: int
    height: int
    intrinsics_px: tuple[float, float, float, float] | None = None

    # ---------------------------------------------------------------- базис
    @property
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return euler_basis(*self.rot)

    @property
    def forward(self) -> np.ndarray:
        return self.basis[0]

    @classmethod
    def from_track(cls, track: CameraTrack, frame: int, width: int, height: int) -> "Camera":
        pose: Pose = track.poses[frame]
        return cls(
            index=track.index,
            loc=pose.loc.copy(),
            rot=pose.rot.copy(),
            focal_mm=pose.focal_mm if pose.focal_mm is not None else track.focal_mm,  # зум по кадрам (Way2)
            sensor_mm=track.sensor_mm,
            width=width,
            height=height,
            intrinsics_px=track.intrinsics_px,
        )

    # ---------------------------------------------------------- интринсики
    @property
    def fx(self) -> float:
        if self.intrinsics_px is not None:
            return self.intrinsics_px[0]
        return self.width * self.focal_mm / self.sensor_mm[0]

    @property
    def fy(self) -> float:
        if self.intrinsics_px is not None:
            return self.intrinsics_px[1]
        return self.height * self.focal_mm / self.sensor_mm[1]

    @property
    def cx(self) -> float:
        if self.intrinsics_px is not None:
            return self.intrinsics_px[2]
        return self.width / 2.0

    @property
    def cy(self) -> float:
        if self.intrinsics_px is not None:
            return self.intrinsics_px[3]
        return self.height / 2.0

    @property
    def fpx(self) -> float:
        """Эффективное фокусное в пикселях (среднее по осям)."""
        return 0.5 * (self.fx + self.fy)

    @property
    def name(self) -> str:
        return f"CAM{self.index}"

    # ------------------------------------------------------------ проекции
    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Мир (N,3) см → пиксели (N,2), глубина (N,) см."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        fwd, right, down = self.basis
        q = pts - self.loc
        z = q @ fwd
        x = q @ right
        y = q @ down
        z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = self.cx + self.fx * x / z_safe
        v = self.cy + self.fy * y / z_safe
        return np.stack([u, v], axis=1), z

    def in_image(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        return (
            (depth > 1e-3)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < self.width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < self.height)
        )

    def project_box(self, corners: np.ndarray) -> np.ndarray | None:
        """2D-бокс (x0,y0,x1,y1) по 8 углам или None, если бокс не виден."""
        uv, depth = self.project(corners)
        mask = self.in_image(uv, depth)
        if mask.sum() < 2:
            return None
        pts = uv[mask]
        return np.array(
            [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=np.float64
        )

    def ray(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Пиксель (u,v) → (начало, единичное направление) луча."""
        fwd, right, down = self.basis
        u, v = float(uv[0]), float(uv[1])
        d = fwd + ((u - self.cx) / self.fx) * right + ((v - self.cy) / self.fy) * down
        n = np.linalg.norm(d)
        return self.loc.copy(), d / n

    def angular_sigma(self, sigma_px: float) -> float:
        return float(sigma_px) / self.fpx

    def distance(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(point) - self.loc))
