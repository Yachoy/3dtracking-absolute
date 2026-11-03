"""Сопровождение 2D-боксов по кадрам: привязка детекций к траектории.

Объект в сцене один, камеры статичны, между кадрами объект смещается
плавно. Поэтому достаточно жадной ассоциации с гейтом по расстоянию:
из всех кандидатов кадра берём ближайший к предсказанному положению
(последняя детекция + скорость), если он не дальше гейта. Иначе кадр
считается пропуском — лучше потерять кадр, чем прыгнуть на дерево.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import Detection


@dataclass
class BoxTracker:
    """Жадный трекер одной камеры с гейтом по смещению."""

    gate_px: float = 220.0
    max_miss: int = 5
    damp: float = 0.6
    last: Detection | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    miss: int = 0
    history: list[Detection] = field(default_factory=list)

    def _predict(self) -> np.ndarray | None:
        if self.last is None:
            return None
        return self.last.center + self.velocity * self.damp

    def update(self, detections: list[Detection], frame: int) -> Detection | None:
        if not detections:
            self.miss += 1
            if self.miss > self.max_miss:
                self.last = None
                self.velocity = np.zeros(2)
            return None

        predicted = self._predict()
        if predicted is None:
            # первая детекция — берём самую крупную
            choice = max(detections, key=lambda d: d.area)
        else:
            dists = [float(np.linalg.norm(d.center - predicted)) for d in detections]
            k = int(np.argmin(dists))
            if dists[k] > self.gate_px:
                self.miss += 1
                if self.miss > self.max_miss:
                    self.last = None
                    self.velocity = np.zeros(2)
                return None
            choice = detections[k]

        if self.last is not None and choice.frame != self.last.frame:
            step = choice.center - self.last.center
            self.velocity = 0.5 * self.velocity + 0.5 * step
        self.last = choice
        self.miss = 0
        self.history.append(choice)
        return choice


def robust_point(
    origins: np.ndarray,
    directions: np.ndarray,
    sigmas: np.ndarray,
    threshold_m: float = 10.0,
    rounds: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Триангуляция с отбраковкой лучей, не сходящихся в одной точке.

    Возвращает точку, ковариацию и булеву маску использованных лучей.
    """
    from .geometry import triangulate

    keep = np.ones(len(origins), dtype=bool)
    point, cov = triangulate(origins, directions, sigmas)
    for _ in range(rounds):
        if keep.sum() < 2:
            break
        d = directions[keep]
        o = origins[keep]
        deltas = point[None, :] - o
        proj = o + (deltas * d).sum(axis=1, keepdims=True) * d
        err = np.linalg.norm(deltas - (proj - o), axis=1)
        if err.size < 3:
            break
        limit = max(threshold_m, 3.0 * float(np.median(err)))
        good = err <= limit
        if good.all():
            break
        idx = np.flatnonzero(keep)
        keep[idx[~good]] = False
        point, cov = triangulate(origins[keep], directions[keep], sigmas[keep])
    return point, cov, keep
