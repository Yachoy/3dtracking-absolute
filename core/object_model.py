"""Модель объекта: составные боксы фургона и точка привязки GT.

GT-точка в сцене — не центр фургона, а его **передний нижний центр**
(проверено наложением бокса на кадры CAM4/CAM5: при смещении
(−11 м по курсу, +2 м вверх) от центра бокс ложится на кузов).

Фургон — седельный тягач с прицепом, поэтому описывается не одним
параллелепипедом, а двумя частями: кабина ниже и короче, прицеп выше
и висит над рамой. Для отладочной отрисовки этого достаточно, чтобы
силуэт совпадал с объектом в кадре.

Габариты подобраны по рендерам: общая длина ≈20 м, ширина 2.5–2.55 м,
высота кабины 3.3 м, прицепа 4.0 м.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import heading_from_velocity, oriented_basis


@dataclass(frozen=True)
class VehiclePart:
    """Параллелепипед в системе объекта: +X вперёд, +Y вправо, +Z вверх.

    x_from/x_to отсчитываются от точки привязки (передний бампер), поэтому
    кузов уходит в отрицательные X.
    """

    x_from: float
    x_to: float
    width: float
    z_from: float
    z_to: float

    def corners(self, origin: np.ndarray, yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
        fwd, right, up = oriented_basis(yaw_deg, pitch_deg)
        pts = []
        for x in (self.x_from, self.x_to):
            for y in (-self.width / 2, self.width / 2):
                for z in (self.z_from, self.z_to):
                    pts.append(origin + x * fwd + y * right + z * up)
        return np.stack(pts)


@dataclass(frozen=True)
class VehicleModel:
    """Двухчастная модель: тягач + прицеп."""

    parts: tuple[VehiclePart, ...] = field(
        default_factory=lambda: (
            # тягач: кабина и рама
            VehiclePart(x_from=-660.0, x_to=0.0, width=250.0, z_from=0.0, z_to=330.0),
            # прицеп: висит над рамой, чуть шире и выше
            VehiclePart(x_from=-2000.0, x_to=-690.0, width=255.0, z_from=100.0, z_to=400.0),
        )
    )

    @property
    def length_cm(self) -> float:
        return float(max(abs(p.x_from) for p in self.parts))

    def corners(self, anchor: np.ndarray, yaw_deg: float, pitch_deg: float = 0.0) -> list[np.ndarray]:
        return [p.corners(anchor, yaw_deg, pitch_deg) for p in self.parts]

    def boxes_2d(self, camera, anchor: np.ndarray, yaw_deg: float, pitch_deg: float = 0.0) -> list[np.ndarray]:
        out = []
        for corners in self.corners(anchor, yaw_deg, pitch_deg):
            box = camera.project_box(corners)
            if box is not None:
                out.append(box)
        return out

    # совместимость со старым интерфейсом: один общий бокс
    def center(self, anchor: np.ndarray, heading_deg: float) -> np.ndarray:
        fwd, _, _ = oriented_basis(heading_deg)
        return anchor - (self.length_cm / 2) * fwd

    def box_2d(self, camera, anchor: np.ndarray, heading_deg: float, pitch_deg: float = 0.0):
        boxes = self.boxes_2d(camera, anchor, heading_deg, pitch_deg)
        if not boxes:
            return None
        arr = np.stack(boxes)
        return np.array([arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()])


ObjectModel = VehicleModel


def heading_from_positions(track: dict[int, np.ndarray], frame: int, span: int = 3) -> float:
    """Курс по соседним кадрам траектории; fallback — на ближайший кадр."""
    lo, hi = frame - span, frame + span
    if lo in track and hi in track:
        return heading_from_velocity(track[hi] - track[lo])
    for _ in range(span + 3):
        if lo in track and frame in track:
            return heading_from_velocity(track[frame] - track[lo])
        if hi in track and frame in track:
            return heading_from_velocity(track[hi] - track[frame])
        lo -= 1
        hi += 1
    return 0.0


def pitch_from_positions(track: dict[int, np.ndarray], frame: int, span: int = 4) -> float:
    """Уклон дороги под объектом, градусы (наклон бокса по ходу движения)."""
    lo, hi = frame - span, frame + span
    if lo in track and hi in track:
        d = track[hi] - track[lo]
        horizontal = math.hypot(d[0], d[1])
        if horizontal > 1e-6:
            return math.degrees(math.atan2(d[2], horizontal))
    return 0.0
