"""3D-модель объекта (фуры) из FBX: выпуклая оболочка, масштаб в сцене, где на ней точка-ответ.

Модель позволяет предсказать рамку объекта целиком для любой позиции и направления —
трекер сравнивает с рамкой YOLO все 4 края, а не одну точку.

Система координат модели здесь: (вперёд, вбок, вверх), метры, начало — на земле посередине фуры.

    python -m tracker.truck_model data/CargoTruck.fbx --fit data/test   # -> config/truck_model.json
"""
import argparse
import json
import struct
import zlib
from dataclasses import dataclass

import numpy as np

MODEL_FILE = "config/truck_model.json"


# ---------------- чтение бинарного FBX (7.x) ----------------
def _read_fbx(path):
    data = open(path, "rb").read()
    if not data.startswith(b"Kaydara FBX Binary"):
        raise ValueError("ожидается бинарный FBX")
    wide = struct.unpack("<I", data[23:27])[0] >= 7500
    scalars = {"Y": ("<h", 2), "C": ("<?", 1), "I": ("<i", 4), "F": ("<f", 4), "D": ("<d", 8), "L": ("<q", 8)}
    arrays = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "b"}

    def prop(p):
        t = chr(data[p])
        p += 1
        if t in scalars:
            fmt, n = scalars[t]
            return struct.unpack(fmt, data[p:p + n])[0], p + n
        if t in arrays:
            _, enc, clen = struct.unpack("<III", data[p:p + 12])
            raw = data[p + 12:p + 12 + clen]
            return np.frombuffer(zlib.decompress(raw) if enc else raw, arrays[t]), p + 12 + clen
        n = struct.unpack("<I", data[p:p + 4])[0]
        v = data[p + 4:p + 4 + n]
        return (v.decode("utf8", "replace") if t == "S" else v), p + 4 + n

    def node(p):
        if wide:
            end, nprops, _ = struct.unpack("<QQQ", data[p:p + 24])
            p += 24
        else:
            end, nprops, _ = struct.unpack("<III", data[p:p + 12])
            p += 12
        name = data[p + 1:p + 1 + data[p]].decode()
        p += 1 + data[p]
        if end == 0:
            return None, p
        props = []
        for _ in range(nprops):
            v, p = prop(p)
            props.append(v)
        kids = []
        while p < end:
            k, p = node(p)
            if k is None:
                break
            kids.append(k)
        return (name, props, kids), end

    top, p = {}, 27
    while p < len(data) - 200:
        n, p = node(p)
        if n is None:
            break
        top[n[0]] = n
    return top


def _p70(n):
    return {pr[1][0]: pr[1][4:] for k in n[2] if k[0] == "Properties70" for pr in k[2]}


def _euler_xyz(rx, ry, rz):
    a, b, c = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_fbx_vertices(path):
    """Все вершины всех мешей в координатах модели, метры, (вперёд, вбок, вверх).

    FBX здесь Y-вверх, Z-вперёд, сантиметры; трансформы деталей — Lcl Translation/Rotation/Scaling.
    """
    top = _read_fbx(path)
    unit_cm = float(_p70(top["GlobalSettings"]).get("UnitScaleFactor", [1.0])[0])
    objs = top["Objects"][2]
    models = {o[1][0]: o for o in objs if o[0] == "Model"}
    parent = {c[1][1]: c[1][2] for c in top["Connections"][2]}
    parts = []
    for o in objs:
        if o[0] != "Geometry":
            continue
        v = next(k for k in o[2] if k[0] == "Vertices")[1][0].reshape(-1, 3)
        pr = _p70(models[parent[o[1][0]]])
        S = np.array(pr.get("Lcl Scaling", [1, 1, 1]), float)
        R = _euler_xyz(*pr.get("Lcl Rotation", [0, 0, 0]))
        T = np.array(pr.get("Lcl Translation", [0, 0, 0]), float)
        parts.append((R @ (v * S).T).T + T)
    v = np.vstack(parts) * unit_cm / 100.0
    return np.c_[v[:, 2], -v[:, 0], v[:, 1]]


# ---------------- модель в сцене ----------------
@dataclass
class TruckModel:
    hull: np.ndarray        # (N,3) выпуклая оболочка, метры сцены, (вперёд, вбок, вверх) от центра на земле
    answer_offset: np.ndarray  # (вперёд, вбок, вверх) от центра до точки-ответа, метры сцены

    @classmethod
    def load(cls, path=MODEL_FILE):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        s = d["scale"]
        return cls(hull=np.array(d["hull_model_m"]) * s, answer_offset=np.array(d["answer_point_model_m"]) * s)

    def world_points(self, center, heading):
        c, s = np.cos(heading), np.sin(heading)
        P = self.hull
        return center + np.c_[P[:, 0] * c - P[:, 1] * s, P[:, 0] * s + P[:, 1] * c, P[:, 2]]

    def box(self, pose, center, heading):
        """Рамка (x1, y1, x2, y2) спроецированной модели или None, если модель за камерой."""
        uv, z = pose.project(self.world_points(center, heading))
        if (z <= 0.5).any():
            return None
        return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])

    def answer_point(self, center, heading):
        o = self.answer_offset
        c, s = np.cos(heading), np.sin(heading)
        return center + np.array([o[0] * c - o[1] * s, o[0] * s + o[1] * c, o[2]])


# ---------------- подгонка масштаба и точки-ответа по GT ----------------
def fit(hull_model, data_dir, det_csv, min_conf=0.5):
    """Подобрать масштаб модели и положение точки GT на ней так, чтобы проекция совпала с рамками YOLO."""
    import pandas as pd
    from scipy.optimize import least_squares

    from tracker.camera import load_cameras, load_ground_truth

    cams = {c.name: c for c in load_cameras(data_dir)}
    gt = load_ground_truth(data_dir).interpolate()
    vel = gt.rolling(9, center=True, min_periods=1).mean().diff()
    det = pd.read_csv(det_csv)
    det = det[(det.conf > min_conf) & (det.x1 > 3) & (det.x2 < 1917) & (det.y1 > 3) & (det.y2 < 1077)]
    samples = []
    for r in det.itertuples():
        v = vel.loc[r.frame].values
        if not np.isfinite(v).all() or np.hypot(v[0], v[1]) * 30 < 2:
            continue
        pose = cams[r.camera].pose(r.frame)
        (u, vv), = pose.project(gt.loc[r.frame].values)[0]
        w = r.x2 - r.x1
        if r.x1 - w <= u <= r.x2 + w and r.y1 - 50 <= vv <= r.y2 + 50:  # рамка — наш объект
            samples.append((pose, gt.loc[r.frame].values, np.arctan2(v[1], v[0]), np.array([r.x1, r.y1, r.x2, r.y2])))
    Y = np.array([s[3] for s in samples])
    size = np.c_[Y[:, 2] - Y[:, 0], Y[:, 3] - Y[:, 1]]

    def residuals(p):
        s, pf, pu = p
        m = TruckModel(hull_model * s, np.array([pf, 0, pu]) * s)
        boxes = [m.box(pose, g - (m.answer_point(g, h) - g), h) for pose, g, h, _ in samples]
        return ((np.array(boxes) - Y) / np.c_[size, size]).ravel()

    best = min((least_squares(residuals, [s0, pf0, 1.0], loss="soft_l1", f_scale=0.1)
                for s0 in (2.5, 3.5, 4.5) for pf0 in (-3.0, 0.0, 3.0)), key=lambda f: f.cost)
    return best.x, len(samples)


def main():
    from scipy.spatial import ConvexHull

    ap = argparse.ArgumentParser()
    ap.add_argument("fbx")
    ap.add_argument("--fit", metavar="DATA_DIR", required=True, help="набор с GT для подгонки масштаба")
    ap.add_argument("--detections", default="cache/detections.csv")
    ap.add_argument("--out", default=MODEL_FILE)
    args = ap.parse_args()

    v = load_fbx_vertices(args.fbx)
    hull = v[ConvexHull(v).vertices]
    print(f"модель: длина {np.ptp(v[:, 0]):.2f} м, ширина {np.ptp(v[:, 1]):.2f} м, высота {np.ptp(v[:, 2]):.2f} м, "
          f"оболочка {len(hull)} точек")
    (s, pf, pu), n = fit(hull, args.fit, args.detections)
    print(f"по {n} рамкам: масштаб {s:.3f}, точка-ответ на модели: {pf:.2f} м вперёд, {pu:.2f} м вверх "
          f"(в сцене {pf * s:.1f} и {pu * s:.1f} м); перед модели — {v[:, 0].max():.2f} м")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"source": str(args.fbx), "scale": s, "answer_point_model_m": [pf, 0.0, pu],
                   "hull_model_m": np.round(hull, 4).tolist()}, f, indent=1)
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()
