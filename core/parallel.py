"""Параллельный проход по камерам.

Камеры независимы вплоть до фузии: каждая читает своё видео и ведёт свой
трекер/детектор. Значит, самый дорогой этап — «пиксели в наблюдения» —
раскладывается по процессам, и 5 камер считаются на 5 ядрах вместо одного.

Процессы, а не потоки: OpenCV внутри отпускает GIL только на декодировании,
а сама арифметика (медианы фона, CSRT, пороги) гил держит. Модель данных
между процессами — маленькие словари `{кадр: рамка}`, поэтому копирования
видеопикселей нет.

    boxes = per_camera(track_camera, jobs, workers=5)
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable, Sequence


def default_workers(n_items: int, limit: int | None = None) -> int:
    """Сколько процессов имеет смысл поднять: по числу задач и ядер."""
    cpu = os.cpu_count() or 1
    want = n_items if limit is None or limit <= 0 else min(n_items, limit)
    return max(1, min(want, cpu))


def per_camera(fn: Callable[[Any], Any], jobs: Sequence[Any], workers: int | None = None) -> list[Any]:
    """Запускает `fn(job)` для каждой камеры, возвращает результаты по порядку.

    `fn` должна быть функцией верхнего уровня (её нужно запиклить в процесс),
    а `job` — маленьким кортежем: путь к данным, номер камеры, параметры.
    """
    n = default_workers(len(jobs), workers)
    if n <= 1 or len(jobs) <= 1:
        return [fn(job) for job in jobs]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
        return list(pool.map(fn, jobs))


def chunked(items: Iterable[Any], size: int) -> list[list[Any]]:
    """Разбивает последовательность на куски фиксированного размера."""
    out: list[list[Any]] = []
    cur: list[Any] = []
    for item in items:
        cur.append(item)
        if len(cur) >= size:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out
