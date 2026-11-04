"""Чтение датасета: сценарные CSV UE и пути к видео.

Источник данных — экспорт из Unreal Engine: `object_transform_<Actor>.csv`
(позиция/ориентация актора по кадрам) и `Way1.FinalImage.<Actor>.mp4`.

Формат CSV:
    Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;35
    0;-24701.96;6623.57;1200.0;0.0;0.0;-70.0;;;матрица;23,76*13,365мм
    1;...

Разделитель `;`, кодировка CP1251, единицы — сантиметры UE (X вперёд,
Y вправо, Z вверх), углы в градусах. Во второй строке камеры лежат
метаданные сенсора (`матрица;23,76*13,365мм`), в первой — фокусное.

Формат Way2 (scene2): фокусное — колонка на каждой строке, сенсор — отдельный файл
`matrix_size.txt` рядом с CSV (`23,76 mm * 13,365 mm`):
    Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;Focal_Length_mm
    0;-28487.34;5124.94;1200.0;0.0;1.6;-37.6;35.0
Фокус может меняться по кадрам (зум) — он хранится в `Pose.focal_mm`.

Данные в git не хранятся: `data/` в .gitignore, архив распаковывается
локально (см. README).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

TRACK_ACTOR = "SM_AA_track"
CAMERA_ACTORS = tuple(f"CineCameraActor{i}" for i in range(1, 6))

VIDEO_GLOB = "*.FinalImage.{actor}.mp4"  # Way1, Way2, …
SENSOR_FILE = "matrix_size.txt"
CSV_GLOB = "object_transform_{actor}.csv"


@dataclass(frozen=True)
class Pose:
    """Позиция и ориентация актора в одном кадре."""

    frame: int
    loc: np.ndarray  # (3,) см
    rot: np.ndarray  # (3,) roll, pitch, yaw — градусы
    focal_mm: float | None = None  # фокус этого кадра, если CSV задаёт его построчно (Way2)


@dataclass
class CameraTrack:
    """Временной ряд одной камеры."""

    index: int
    focal_mm: float
    sensor_mm: tuple[float, float]
    poses: dict[int, Pose]
    video: Path | None = None
    intrinsics_px: tuple[float, float, float, float] | None = None

    @property
    def name(self) -> str:
        return f"CAM{self.index}"

    @property
    def is_static(self) -> bool:
        locs = np.array([p.loc for p in self.poses.values()])
        return bool(np.allclose(locs, locs[0], atol=1e-6))


@dataclass
class Scene:
    """Одна сцена: GT-трек цели + камеры + видео."""

    root: Path
    target: dict[int, Pose]
    cameras: dict[int, CameraTrack]
    width: int = 1920
    height: int = 1080
    fps: float = 30.0

    @property
    def frames(self) -> list[int]:
        return sorted(self.target)

    def video_path(self, cam: int) -> Path | None:
        return self.cameras[cam].video

    def gt_array(self) -> tuple[np.ndarray, list[int]]:
        frames = self.frames
        return np.stack([self.target[f].loc for f in frames]), frames


def _split(line: str) -> list[str]:
    # Не используем strip(): он съедает значимые пустые хвостовые поля.
    return line.rstrip("\r\n").split(";")


def _parse_float(text: str) -> float:
    return float(text.replace(",", ".").replace("\ufeff", "").strip())


def _parse_sensor(text: str) -> tuple[float, float] | None:
    """«23,76*13,365мм», «23,76 mm * 13,365 mm» -> (23.76, 13.365)."""
    numbers = re.findall(r"\d+(?:[.,]\d+)?", text)
    if "*" not in text or len(numbers) != 2:
        return None
    return _parse_float(numbers[0]), _parse_float(numbers[1])


def read_sensor_file(directory: Path) -> tuple[float, float] | None:
    """Размер матрицы из `matrix_size.txt` рядом с CSV (формат Way2)."""
    path = Path(directory) / SENSOR_FILE
    if not path.is_file():
        return None
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return _parse_sensor(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    return None


def read_transform_csv(path: Path) -> tuple[dict[int, Pose], dict]:
    """Читает CSV трансформа. Возвращает (кадры, метаданные).

    Оба формата: Way1 (фокус в шапке, сенсор во второй строке) и Way2 (колонка
    `Focal_Length_mm` на каждой строке, сенсор в `matrix_size.txt` рядом с CSV).
    """
    path = Path(path)
    poses: dict[int, Pose] = {}
    meta: dict = {"focal_mm": None, "sensor_mm": None, "focal_per_frame": False}
    focal_col = None

    with open(path, encoding="cp1251", errors="replace") as fh:
        for lineno, raw in enumerate(fh):
            fields = _split(raw)
            if lineno == 0:
                # шапка: «...;Focal length;35» — ищем по подписи, а не по индексу:
                # число колонок зависит от наличия пустых полей в конце строки
                for pos, name in enumerate(fields):
                    label = name.strip().lower().replace(" ", "_")
                    if label == "focal_length" and pos + 1 < len(fields):
                        meta["focal_mm"] = _parse_float(fields[pos + 1])
                    elif label in ("focal_length_mm", "focal_mm"):
                        focal_col = pos  # Way2: фокус построчно
                continue
            for field in fields:
                # строка метаданных сенсора: «матрица;23,76*13,365мм»
                if "мм" in field and "*" in field:
                    sensor = _parse_sensor(field)
                    if sensor is not None:
                        meta["sensor_mm"] = sensor
            try:
                frame = int(fields[0])
                values = [_parse_float(v) for v in fields[1:7]]
            except (ValueError, IndexError):
                continue
            if len(values) < 3:
                continue
            focal = None
            if focal_col is not None and focal_col < len(fields) and fields[focal_col].strip():
                try:
                    focal = _parse_float(fields[focal_col])
                except ValueError:
                    focal = None
                if focal is not None and not (np.isfinite(focal) and focal > 0):
                    focal = None
            rot = list(values[3:6]) + [0.0] * (3 - len(values[3:6]))
            poses[frame] = Pose(
                frame=frame,
                loc=np.array(values[:3], dtype=np.float64),
                rot=np.array(rot[:3], dtype=np.float64),
                focal_mm=focal,
            )
    focals = [p.focal_mm for p in poses.values() if p.focal_mm is not None]
    if focals:
        if meta["focal_mm"] is None:
            meta["focal_mm"] = float(focals[0])
        meta["focal_per_frame"] = True
    if meta["sensor_mm"] is None:
        meta["sensor_mm"] = read_sensor_file(path.parent)
    return poses, meta


def load_scene(root: str | Path, width: int = 1920, height: int = 1080, fps: float = 30.0,
               require_gt: bool = True) -> Scene:
    """Загружает сцену из каталога с распакованным архивом.

    require_gt=False — для проверочного набора без трека объекта (только камеры и видео).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"нет каталога с данными: {root}")

    gt_path = root / CSV_GLOB.format(actor=TRACK_ACTOR)
    target = read_transform_csv(gt_path)[0] if gt_path.exists() else {}
    if not target and require_gt:
        raise RuntimeError(f"не найдены кадры GT в {root}")

    cameras: dict[int, CameraTrack] = {}
    found = sorted(int(m.group(1)) for p in root.glob("object_transform_CineCameraActor*.csv")
                   if (m := re.fullmatch(r"object_transform_CineCameraActor(\d+)\.csv", p.name)))
    for idx in found:  # Way1 — 5 камер, Way2 — 9
        actor = f"CineCameraActor{idx}"
        poses, meta = read_transform_csv(root / CSV_GLOB.format(actor=actor))
        if not poses:
            continue
        if meta["focal_mm"] is None or meta["sensor_mm"] is None:
            raise RuntimeError(f"в {actor} не найдены фокус/сенсор")
        videos = sorted(root.glob(VIDEO_GLOB.format(actor=actor)))
        video = videos[0] if len(videos) == 1 else None
        cameras[idx] = CameraTrack(
            index=idx,
            focal_mm=float(meta["focal_mm"]),
            sensor_mm=tuple(meta["sensor_mm"]),
            poses=poses,
            video=video,
        )

    return Scene(root=root, target=target, cameras=cameras, width=width, height=height, fps=fps)
