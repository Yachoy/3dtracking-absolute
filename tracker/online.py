"""Онлайн 3D-трекер: на кадре t используются только данные кадров <= t.

Один 3D-трек (EKF, постоянная скорость) — точка на земле под объектом.
Каждая камера даёт измерение: середина нижнего края рамки (u, v) и высота рамки h в пикселях.
- (u, v) — луч к точке на земле: нижний край рамки — это место, где колёса касаются земли,
  он гораздо меньше зависит от ракурса, чем центр рамки.
- h — дальность: высота грузовика одинакова с любого ракурса, h ≈ fy * H / дальность.
  Поэтому даже одна камера даёт полную 3D-точку, а не только луч.
EKF обновляется прямо по пикселям, сопоставление между камерами — через прогноз трека:
в каждой камере берём только детекцию, попавшую в зону, куда проецируется прогноз.
"""
import itertools
import json
from dataclasses import dataclass, field

import numpy as np

FPS = 30.0
INIT_REPROJ_PX = 25.0     # допуск перепроекции при поиске стартовой точки
INIT_MIN_CONF = 0.25
CONFIRM_UPDATES = 5       # столько кадров с измерениями, чтобы трек считался подтверждённым
TENTATIVE_MAX_MISS = 5
LOST_MAX_MISS = 90        # 3 с без измерений — трек сбрасывается
EDGE_PX = 3               # рамки, касающиеся края кадра, обрезаны — измерение смещено
CHI2_GATE = {2: 13.8, 3: 16.3}  # 99.9% для 2 и 3 степеней свободы


@dataclass
class TrackerConfig:
    offset: tuple = (0.0, 0.0, 0.0)   # (вперёд, влево, вверх) от точки трека до точки-ответа, метры
    object_height_m: float = 4.0      # высота объекта для оценки дальности по рамке
    use_size: bool = True             # использовать высоту рамки как измерение дальности
    single_camera_init: bool = True   # стартовать трек по одной камере (нужен use_size)
    single_init_min_conf: float = 0.5
    accel_xy: float = 12.0            # шум модели движения: насколько резко объект может ускоряться, м/с²
                                      # (подобрано на тесте: грузовик резко поворачивает, 3 — слишком мало)
    accel_z: float = 1.0

    @classmethod
    def load(cls, path, **overrides):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        cfg = cls(offset=tuple(data.get("offset_fwd_left_up_m", (0.0, 0.0, 0.0))),
                  object_height_m=data.get("object_height_m", 4.0))
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


@dataclass
class Detection:
    camera: int
    box: np.ndarray       # x1, y1, x2, y2
    conf: float

    @property
    def width(self):
        return self.box[2] - self.box[0]

    @property
    def height(self):
        return self.box[3] - self.box[1]

    @property
    def ground_point(self):
        """Середина нижнего края рамки — где объект касается земли."""
        return np.array([(self.box[0] + self.box[2]) / 2, self.box[3]])

    def measurement(self, use_size):
        return np.r_[self.ground_point, self.height] if use_size else self.ground_point

    def noise(self, use_size):
        # по горизонтали середина рамки «гуляет» вдоль длинного прицепа в зависимости от ракурса
        sig = [2.0 + 0.2 * self.width, 2.0 + 0.05 * self.height]
        if use_size:
            sig.append(2.0 + 0.1 * self.height)
        return np.diag(np.square(sig))


def triangulate(poses, uvs):
    """Точка, ближайшая ко всем лучам (МНК). poses — CameraPose, uvs — пиксели."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for pose, (u, v) in zip(poses, uvs):
        d = pose.ray(u, v)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ pose.center
    return np.linalg.lstsq(A, b, rcond=None)[0]


def point_from_size(pose, det, object_height_m):
    """Одна камера: дальность из высоты рамки, направление — луч через нижний край."""
    depth = pose.fy * object_height_m / det.height
    ray = pose.ray(*det.ground_point)
    return pose.center + ray * depth / (ray @ pose.rot[2])


def measurement_model(pose, p, object_height_m, use_size):
    """Ожидаемое измерение (u, v[, h]) для точки p и его якобиан по p."""
    pc = pose.rot @ (p - pose.center)
    x, y, z = pc
    pred = [pose.cx + pose.fx * x / z, pose.cy + pose.fy * y / z]
    jac = [[pose.fx / z, 0, -pose.fx * x / z**2],
           [0, pose.fy / z, -pose.fy * y / z**2]]
    if use_size:
        pred.append(pose.fy * object_height_m / z)
        jac.append([0, 0, -pose.fy * object_height_m / z**2])
    return np.array(pred), np.array(jac) @ pose.rot, z


@dataclass
class Track:
    x: np.ndarray                     # [X, Y, Z, Vx, Vy, Vz], метры и м/с
    P: np.ndarray
    updates: int = 0
    misses: int = 0
    heading: float = 0.0
    confirmed: bool = False
    used_cameras: list = field(default_factory=list)

    @classmethod
    def start(cls, pos, pos_var=4.0):
        P = np.diag([pos_var, pos_var, pos_var, 15.0**2, 15.0**2, 2.0**2])
        return cls(x=np.r_[pos, 0, 0, 0], P=P)

    def predict(self, accel_xy, accel_z, dt=1 / FPS):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = np.array([accel_xy, accel_xy, accel_z]) ** 2
        G = np.vstack([np.eye(3) * dt**2 / 2, np.eye(3) * dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ np.diag(q) @ G.T

    def innovation(self, pose, det, cfg):
        pred, J, z = measurement_model(pose, self.x[:3], cfg.object_height_m, cfg.use_size)
        if z <= 0:
            return None
        H = np.zeros((len(pred), 6))
        H[:, :3] = J
        r = det.measurement(cfg.use_size) - pred
        S = H @ self.P @ H.T + det.noise(cfg.use_size)
        return r, H, S, float(r @ np.linalg.solve(S, r))

    def update(self, r, H, S):
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ r
        self.P = (np.eye(6) - K @ H) @ self.P

    def refresh_heading(self, min_speed=1.0):
        if np.hypot(self.x[3], self.x[4]) > min_speed:
            self.heading = float(np.arctan2(self.x[4], self.x[3]))


class OnlineTracker3D:
    def __init__(self, cameras, cfg=None):
        self.cameras = cameras
        self.cfg = cfg or TrackerConfig()
        self.track = None

    # ---------- шаг по кадру ----------
    def step(self, frame, detections):
        """detections: список Detection только этого кадра. Возвращает (точка-ответ | None, статус)."""
        poses = [cam.pose(frame) for cam in self.cameras]
        detections = [d for d in detections if not self._touches_edge(d)]

        if self.track is not None:
            self.track.predict(accel_xy=self.cfg.accel_xy, accel_z=self.cfg.accel_z)
            self._update(poses, detections)
            if self._should_drop():
                self.track = None

        if self.track is None:
            self.track = self._initialize(poses, detections)

        if self.track is None or not self.track.confirmed:
            return None, "SEARCH"
        status = "TRACKING" if self.track.misses == 0 else "LOST"
        return self._answer_point(), status

    def _update(self, poses, detections):
        t = self.track
        gate = CHI2_GATE[3 if self.cfg.use_size else 2]
        t.used_cameras = []
        for cam_idx, pose in enumerate(poses):
            best = None
            for det in (d for d in detections if d.camera == cam_idx):
                inn = t.innovation(pose, det, self.cfg)
                if inn is not None and inn[3] < gate and (best is None or inn[3] < best[3]):
                    best = inn
            if best is not None:
                t.update(*best[:3])
                t.used_cameras.append(cam_idx)
        if t.used_cameras:
            t.updates += 1
            t.misses = 0
            t.confirmed |= t.updates >= CONFIRM_UPDATES
        else:
            t.misses += 1
        t.refresh_heading()

    def _should_drop(self):
        t = self.track
        return t.misses > (LOST_MAX_MISS if t.confirmed else TENTATIVE_MAX_MISS)

    # ---------- инициализация ----------
    def _initialize(self, poses, detections):
        """Сначала ищем согласованную пару детекций из разных камер, иначе — старт по одной камере."""
        strong = [d for d in detections if d.conf >= INIT_MIN_CONF]
        best = None
        for a, b in itertools.combinations(strong, 2):
            if a.camera == b.camera:
                continue
            p = triangulate([poses[a.camera], poses[b.camera]], [a.ground_point, b.ground_point])
            errs = [self._reproj_error(poses[d.camera], d, p) for d in (a, b)]
            if max(errs) > INIT_REPROJ_PX:
                continue
            support = self._support(poses, strong, p, exclude={a.camera, b.camera})
            score = (2 + support, -max(errs))
            if best is None or score > best[0]:
                best = (score, p)
        if best:
            return Track.start(best[1])

        if self.cfg.use_size and self.cfg.single_camera_init:
            cands = [d for d in strong if d.conf >= self.cfg.single_init_min_conf]
            if cands:
                det = max(cands, key=lambda d: d.conf)
                p = point_from_size(poses[det.camera], det, self.cfg.object_height_m)
                return Track.start(p, pos_var=25.0)
        return None

    @staticmethod
    def _reproj_error(pose, det, p):
        uv, z = pose.project(p)
        return np.inf if z[0] <= 0 else float(np.linalg.norm(uv[0] - det.ground_point))

    def _support(self, poses, detections, p, exclude):
        cams = {d.camera for d in detections
                if d.camera not in exclude and self._reproj_error(poses[d.camera], d, p) < INIT_REPROJ_PX * 2}
        return len(cams)

    # ---------- ответ ----------
    def _answer_point(self):
        h = self.track.heading
        fwd, left = np.array([np.cos(h), np.sin(h), 0]), np.array([-np.sin(h), np.cos(h), 0])
        o = self.cfg.offset
        return self.track.x[:3] + o[0] * fwd + o[1] * left + o[2] * np.array([0, 0, 1.0])

    def _touches_edge(self, det):
        cam = self.cameras[det.camera]
        x1, y1, x2, y2 = det.box
        return x1 < EDGE_PX or y1 < EDGE_PX or x2 > cam.width - EDGE_PX or y2 > cam.height - EDGE_PX
