"""Геометрия сцены: базис UE-камеры, триангуляция, 3D-бокс объекта.

Соглашения UE: мировая система левосторонняя, X — вперёд, Y — вправо,
Z — вверх; Rotator = (roll, pitch, yaw) в градусах, yaw 0° смотрит в +X,
положительный yaw закручивает по часовой стрелке (к +Y).
Оптическая ось камеры — её forward, «право» и «низ» изображения
образуют вместе с forward ортонормированный базис.
"""

from __future__ import annotations

import numpy as np


def euler_basis(roll: float, pitch: float, yaw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (forward, right, down) камеры в мировых координатах."""
    r, p, y = np.radians([roll, pitch, yaw])
    forward = np.array(
        [np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)], dtype=np.float64
    )
    right0 = np.array([-np.sin(y), np.cos(y), 0.0])
    down0 = np.cross(right0, forward)
    # roll — поворот вокруг оптической оси
    right = np.cos(r) * right0 + np.sin(r) * down0
    down = np.cos(r) * down0 - np.sin(r) * right0
    return forward, right, down


def oriented_basis(yaw_deg: float, pitch_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Базис объекта: (вперёд, вправо, вверх) с наклоном по курсу и уклону дороги."""
    y = np.radians(yaw_deg)
    p = np.radians(pitch_deg)
    fwd = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
    right = np.array([-np.sin(y), np.cos(y), 0.0])
    up = np.cross(fwd, right)
    return fwd, right / np.linalg.norm(right), up / np.linalg.norm(up)


def box_corners(
    center: np.ndarray, yaw_deg: float, size: tuple[float, float, float], pitch_deg: float = 0.0
) -> np.ndarray:
    """8 углов 3D-бокса (см). size = (длина, ширина, высота)."""
    fwd, right, up = oriented_basis(yaw_deg, pitch_deg)
    corners = []
    for a in (-0.5, 0.5):
        for b in (-0.5, 0.5):
            for c in (-0.5, 0.5):
                corners.append(center + a * size[0] * fwd + b * size[1] * right + c * size[2] * up)
    return np.stack(corners)


def heading_from_velocity(velocity: np.ndarray) -> float:
    """Yaw движения в градусах (UE)."""
    return float(np.degrees(np.arctan2(velocity[1], velocity[0])))


def triangulate(
    origins: np.ndarray,
    directions: np.ndarray,
    angular_sigma: np.ndarray,
    iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Взвешенное МНК-пересечение лучей.

    Параметры
    ---------
    origins : (N, 3) — позиции камер, см
    directions : (N, 3) — единичные направления на объект
    angular_sigma : (N,) — угловая неопределённость луча, рад
                         (≈ σ_px / f_px со знаком и масштабом по пикселям)

    Возвращает точку (3,) и её ковариацию (3, 3) в см².
    Ковариация включает только поперечную неопределённость лучей:
    для каждой камеры Σ_i = (d_i·σ_θi)²·(I − d_i d_iᵀ), d_i — дальность.
    """
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    angular_sigma = np.asarray(angular_sigma, dtype=np.float64)

    point = origins.mean(axis=0)
    for _ in range(max(1, iterations)):
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for C, d, s in zip(origins, directions, angular_sigma):
            dist = float(np.linalg.norm(point - C))
            # поперечное смещение луча на дальности dist, см
            w = 1.0 / max(dist * s, 1e-9) ** 2
            P = np.eye(3) - np.outer(d, d)
            A += w * P
            b += w * (P @ C)
        point = np.linalg.pinv(A) @ b

    cov = np.linalg.pinv(A)
    return point, cov


def ground_intersect(origin: np.ndarray, direction: np.ndarray, z: float = 0.0) -> np.ndarray | None:
    """Пересечение луча с горизонтальной плоскостью z=const (для priors)."""
    if abs(direction[2]) < 1e-9:
        return None
    t = (z - origin[2]) / direction[2]
    if t <= 0:
        return None
    return origin + t * direction
