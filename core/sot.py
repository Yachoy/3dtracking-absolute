"""Классические однокадровые трекеры (CSRT / KCF / MOSSE) из OpenCV.

Идея: объект не обязан быть «фурой» из известного класса. Оператор даёт
рамку на первом кадре, где объект виден в камере, — дальше трекер ведёт её
по внешнему виду. Никакого обучения на нашем датасете и никакого GT внутри.
"""

from __future__ import annotations

import cv2
import numpy as np

_KINDS = {
    "csrt": lambda: cv2.legacy.TrackerCSRT_create(),
    "kcf": lambda: cv2.legacy.TrackerKCF_create(),
    "mosse": lambda: cv2.legacy.TrackerMOSSE_create(),
    "mil": lambda: cv2.legacy.TrackerMIL_create(),
}


class SotTracker:
    """Обёртка над cv2.legacy.*: init(box_xywh) → update() → box_xyxy."""

    def __init__(self, kind: str = "csrt") -> None:
        if kind not in _KINDS:
            raise ValueError(f"неизвестный трекер {kind!r}, есть {sorted(_KINDS)}")
        self.kind = kind
        self._tracker = None
        self.lost = 0

    @property
    def active(self) -> bool:
        return self._tracker is not None

    def init(self, frame: np.ndarray, box: tuple[float, float, float, float]) -> None:
        """box — (x0, y0, x1, y1) в пикселях кадра."""
        x0, y0, x1, y1 = [float(v) for v in box]
        w, h = max(2.0, x1 - x0), max(2.0, y1 - y0)
        self._tracker = _KINDS[self.kind]()
        self._tracker.init(frame, (x0, y0, w, h))
        self.lost = 0

    def update(self, frame: np.ndarray) -> tuple[float, float, float, float] | None:
        if self._tracker is None:
            return None
        ok, box = self._tracker.update(frame)
        if not ok:
            self._tracker = None
            self.lost += 1
            return None
        x, y, w, h = box
        return (x, y, x + w, y + h)

    def reset(self) -> None:
        self._tracker = None


def union_box(boxes) -> tuple[float, float, float, float] | None:
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    arr = np.asarray(boxes, dtype=float)
    return (
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 2].max()),
        float(arr[:, 3].max()),
    )
