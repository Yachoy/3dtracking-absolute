"""Модель камеры Unreal Engine: поза на каждом кадре, проекция точек, лучи из пикселей.

Все расчёты внутри — в метрах. Во входных/выходных CSV — сантиметры (как в UE).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CM = 100.0
SENSOR_W_MM = 23.76
SENSOR_H_MM = 13.365


def read_table(path):
    """CSV из UE: ';' как разделитель, cp1251, в первой строке мусор в лишних колонках.

    Встречаются битые ячейки (буквы вместо цифр) — они становятся NaN.
    """
    df = pd.read_csv(path, sep=";", encoding="cp1251", dtype=str)
    df = df.loc[:, [c for c in df.columns if c.startswith(("Frame", "Loc_", "Rot_"))]]
    df = df.apply(lambda col: pd.to_numeric(col.str.replace(",", "."), errors="coerce"))
    return df.dropna(subset=["Frame"]).astype({"Frame": int}).set_index("Frame")


def read_focal_mm(path):
    with open(path, encoding="cp1251") as f:
        return float(f.readline().strip().split(";")[-1].replace(",", "."))


def ue_axes(roll, pitch, yaw):
    """Оси камеры в мире (forward, right, up) по формуле FRotationMatrix из UE. Углы в градусах."""
    r, p, y = np.radians([roll, pitch, yaw])
    sr, cr, sp, cp, sy, cy = np.sin(r), np.cos(r), np.sin(p), np.cos(p), np.sin(y), np.cos(y)
    fwd = np.array([cp * cy, cp * sy, sp])
    right = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp])
    up = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])
    return fwd, right, up


@dataclass
class CameraPose:
    """Поза одной камеры на одном кадре."""
    center: np.ndarray  # (3,) метры
    rot: np.ndarray     # (3,3) мир -> камера (строки: right, down, forward), OpenCV-конвенция
    fx: float
    fy: float
    cx: float
    cy: float

    def project(self, pts):
        """(N,3) мир, метры -> (N,2) пиксели и (N,) глубина вдоль оптической оси."""
        pc = (np.atleast_2d(pts) - self.center) @ self.rot.T
        z = pc[:, 2]
        uv = np.stack([self.cx + self.fx * pc[:, 0] / z, self.cy + self.fy * pc[:, 1] / z], axis=1)
        return uv, z

    def ray(self, u, v):
        """Единичный вектор луча в мире через пиксель (u, v)."""
        d_cam = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        d = self.rot.T @ d_cam
        return d / np.linalg.norm(d)


class Camera:
    """Камера с позами на всех кадрах (из object_transform_<name>.csv)."""

    def __init__(self, csv_path, width=1920, height=1080):
        self.name = Path(csv_path).stem.replace("object_transform_", "")
        self.table = read_table(csv_path)
        self.focal_mm = read_focal_mm(csv_path)
        self.width, self.height = width, height
        self.fx = self.focal_mm / SENSOR_W_MM * width
        self.fy = self.focal_mm / SENSOR_H_MM * height

    def pose(self, frame):
        row = self.table.loc[frame]
        fwd, right, up = ue_axes(row.Rot_Roll, row.Rot_Pitch, row.Rot_Yaw)
        return CameraPose(
            center=np.array([row.Loc_X, row.Loc_Y, row.Loc_Z]) / CM,
            rot=np.stack([right, -up, fwd]),
            fx=self.fx, fy=self.fy, cx=self.width / 2, cy=self.height / 2,
        )


def load_cameras(data_dir):
    paths = sorted(Path(data_dir).glob("object_transform_CineCameraActor*.csv"))
    return [Camera(p) for p in paths]


def load_ground_truth(data_dir):
    """Истинная траектория объекта, метры. None, если файла нет (проверочный набор)."""
    paths = [p for p in Path(data_dir).glob("object_transform_*.csv") if "CineCamera" not in p.name]
    return read_table(paths[0])[["Loc_X", "Loc_Y", "Loc_Z"]] / CM if paths else None


def video_path(data_dir, camera_name):
    return next(Path(data_dir).glob(f"*{camera_name}.mp4"))
