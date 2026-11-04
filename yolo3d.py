"""Полный прогон: guided search + триангуляция + сдвиг пивота.

Логика на каждый кадр:
  1. по текущей 3D-оценке строится оранжевое окно поиска для каждой камеры;
  2. в окне вырезается кроп, увеличивается и отдаётся YOLO -> синий бокс;
  3. если камер с боксом >= 2 — триангуляция низов боксов (взвешенный МНК
     с ковариацией), если одна — луч + глубина из высоты бокса с
     анизотропной ковариацией, если ноль — только предсказание фильтра;
  4. фильтр Калмана (постоянная скорость, 3D) с гейтом по Махаланобису;
  5. сдвиг пивота: выход = точка кузова + offset_fwd по курсу + offset_up.

Запуск из любого места (пути считаются от файла):
    & 'C:\\Users\\asker\\PycharmProjects\\DescriptorModel\\.venv\\Scripts\\python.exe' `
      'C:\\Users\\asker\\PycharmProjects\\DescriptorModel\\ABC_EXP\\PELENG_TRACK_TASK\\repo\\tools\\full_run.py'
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.camera import Camera  # noqa: E402
from core.dataset import load_scene  # noqa: E402
from core.geometry import triangulate  # noqa: E402
from core.object_model import VehicleModel  # noqa: E402

W, H = 1920, 1080
CLASSES = ("car", "truck", "bus")
CAMS = (1, 2, 3, 4, 5)

GT_COLOR = (60, 220, 60)
WIN_COLOR = (0, 165, 255)      # оранжевый — окно поиска
DET_COLOR = (255, 120, 30)     # синий — детекция YOLO (BGR)
BORDER_COLOR = (255, 90, 0)
TXT = (235, 235, 235)


# --------------------------------------------------------------------------- #
# чтение кадров без потерь на seek
# --------------------------------------------------------------------------- #
class FrameReader:
    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        self.next_idx = 0

    def get(self, idx: int):
        if not self.cap.isOpened():
            return None
        if idx < self.next_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.next_idx = idx
        while self.next_idx < idx:
            if not self.cap.grab():
                return None
            self.next_idx += 1
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.next_idx = idx + 1
        return frame

    def release(self):
        self.cap.release()


# --------------------------------------------------------------------------- #
# геометрия и детекции
# --------------------------------------------------------------------------- #
def fit_tile(img, tw, th):
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((th, tw, 3), 16, np.uint8)
    dx, dy = (tw - nw) // 2, (th - nh) // 2
    out[dy:dy + nh, dx:dx + nw] = resized
    return out, scale, dx, dy


def best_box(result, names):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    best = None
    for b, c, k in zip(xyxy, conf, cls):
        if names[k] in CLASSES and (best is None or c > best[0]):
            best = (float(c), b.astype(np.float64))
    return best


def detect_in_window(detector, frame, window, conf, expect_w=None):
    x0, y0 = max(0, int(window[0])), max(0, int(window[1]))
    x1, y1 = min(W, int(window[2])), min(H, int(window[3]))
    if x1 - x0 < 16 or y1 - y0 < 12:
        return None
    crop = frame[y0:y1, x0:x1]
    # масштаб подбираем так, чтобы ожидаемый объект занимал ~360 px по ширине:
    # окно может быть большим (запас на неопределённость), но объект в нём мелкий
    s = 1.0
    if expect_w and expect_w > 4.0:
        s = float(np.clip(360.0 / expect_w, 1.0, 6.0))
    img = crop if s <= 1.01 else cv2.resize(crop, None, fx=s, fy=s,
                                            interpolation=cv2.INTER_CUBIC)
    size = int(np.clip(round(max(img.shape[:2]) / 32.0) * 32, 320, 960))
    hit = best_box(detector.predict(img, imgsz=size, conf=conf, verbose=False)[0],
                   detector.names)
    if hit is None and s > 1.01:
        # второй заход без апскейла: объект мог быть обрезан краем окна
        hit = best_box(detector.predict(crop, imgsz=640, conf=conf,
                                        verbose=False)[0], detector.names)
        s = 1.0
    if hit is None:
        return None
    c, b = hit
    box = np.array([b[0] / s + x0, b[1] / s + y0, b[2] / s + x0, b[3] / s + y0])
    return float(c), box


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def project_points(camera: Camera, points: np.ndarray):
    uv, depth = camera.project(points)
    ok = (depth > 50) & np.isfinite(uv).all(axis=1)
    if ok.sum() < 2:
        return None
    u, v = uv[ok, 0], uv[ok, 1]
    return np.array([u.min(), v.min(), u.max(), v.max()], dtype=np.float64)


def covariance_samples(point, cov, k=2.0):
    pts = [point]
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return np.array(pts)
    for i in range(3):
        sigma = float(np.sqrt(max(vals[i], 1.0)))
        step = vecs[:, i] * (k * sigma)
        pts.append(point + step)
        pts.append(point - step)
    return np.array(pts)


def extent_samples(point, heading=0.0, radius=1000.0, height=400.0, n=8):
    pts = []
    for k in range(n):
        a = 2.0 * np.pi * k / n
        off = np.array([radius * np.cos(a), radius * np.sin(a), 0.0])
        pts.append(point + off)
        pts.append(point + off + np.array([0.0, 0.0, height]))
    return np.array(pts)


def search_window(camera: Camera, point, cov, heading,
                  pad_frac=0.15, pad_px=6.0):
    box_a = project_points(camera, covariance_samples(point, cov))
    box_b = project_points(camera, extent_samples(point, heading))
    if os.environ.get("DBG_WIN"):
        sa = None if box_a is None else (round(box_a[2] - box_a[0]), round(box_a[3] - box_a[1]))
        sb = None if box_b is None else (round(box_b[2] - box_b[0]), round(box_b[3] - box_b[1]))
        pa = None if box_a is None else (round(box_a[0]), round(box_a[1]))
        pb = None if box_b is None else (round(box_b[0]), round(box_b[1]))
        print(f"   win {camera.name}: ellipse={sa}@{pa} model={sb}@{pb}")
    if box_a is None and box_b is None:
        return None, 0.0
    obj_w = 0.0 if box_b is None else float(box_b[2] - box_b[0])
    if box_a is None:
        box = box_b
    elif box_b is None:
        box = box_a
    else:
        box = np.array([min(box_a[0], box_b[0]), min(box_a[1], box_b[1]),
                        max(box_a[2], box_b[2]), max(box_a[3], box_b[3])])
    w = box[2] - box[0]
    h = box[3] - box[1]
    # тень на песке ниже машины: YOLO обводит её вместе с корпусом
    box = box + np.array([0.0, -0.20 * h, 0.0, 0.75 * h])
    w = box[2] - box[0]
    h = box[3] - box[1]
    box = box + np.array([-pad_frac * w - pad_px, -pad_frac * h - pad_px,
                           pad_frac * w + pad_px, pad_frac * h + pad_px])
    box[[0, 2]] = np.clip(box[[0, 2]], 0, W - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, H - 1)
    if box[2] - box[0] < 16 or box[3] - box[1] < 12:
        return None, 0.0
    return box, obj_w


# --------------------------------------------------------------------------- #
# фильтр Калмана: постоянная скорость в 3D
# --------------------------------------------------------------------------- #
class Kalman3D:
    def __init__(self, point, pos_sigma=300.0, vel_sigma=60.0, accel=0.6):
        self.x = np.zeros(6)
        self.x[:3] = point
        self.P = np.diag([pos_sigma ** 2] * 3 + [vel_sigma ** 2] * 3)
        self.accel = accel
        self.coast = 0

    def predict(self, dt=1.0):
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        q = self.accel ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * (q * dt ** 4 / 4.0)
        Q[:3, 3:] = np.eye(3) * (q * dt ** 3 / 2.0)
        Q[3:, :3] = Q[:3, 3:]
        Q[3:, 3:] = np.eye(3) * (q * dt ** 2)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.x[3:] = np.clip(self.x[3:], -90.0, 90.0)
        return self.x.copy()

    def update(self, z, R, gate=16.0) -> bool:
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        S = H @ self.P @ H.T + R
        y = z - H @ self.x
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        d2 = float(y @ S_inv @ y)
        if not np.isfinite(d2) or d2 > gate:
            self.coast += 1
            return False
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self.coast = 0
        return True

    @property
    def point(self):
        return self.x[:3].copy()

    @property
    def cov(self):
        return self.P[:3, :3].copy()

    @property
    def vel(self):
        return self.x[3:].copy()


class HeadingFilter:
    """CTRV-упрощение: курс и угловая скорость (градусы и градусы/кадр).

    Измерение курса — направление скорости КФ. Его точность тем ниже, чем
    меньше скорость и чем больше шум скорости, поэтому сигма измерения
    считается из ковариации скорости: sigma_h = sigma_v / |v|.
    Между измерениями курс идёт по угловой скорости, то есть на участках
    инерции поворот продолжается, а не замирает.
    """

    def __init__(self, heading=0.0, rate=0.0, h_sigma=40.0, rate_sigma=0.6):
        self.h = float(heading)
        self.w = float(rate)
        self.P = np.diag([h_sigma ** 2, rate_sigma ** 2])

    def predict(self, dt=1.0, accel=0.35):
        q = accel ** 2
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.array([[q * dt ** 3 / 3.0, q * dt ** 2 / 2.0],
                      [q * dt ** 2 / 2.0, q * dt]])
        self.h = self.h + self.w * dt
        self.P = F @ self.P @ F.T + Q
        self.w = float(np.clip(self.w, -4.0, 4.0))

    def update(self, h_meas, sigma_deg):
        s = self.P[0, 0] + sigma_deg ** 2
        K = self.P[:, 0] / s
        y = wrap180(h_meas - self.h)
        self.h = float(wrap180(self.h + K[0] * y))
        self.w = float(self.w + K[1] * y)
        self.P = self.P - np.outer(K, self.P[0, :])
        self.P = 0.5 * (self.P + self.P.T)
        return True


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "out" / "yolo-like"))
    ap.add_argument("--weights", default=str(ROOT / "yolo26l.pt"))
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=1001)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--offset-fwd", type=float, default=10.0, help="м, по курсу")
    ap.add_argument("--offset-up", type=float, default=4.0, help="м, вверх")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--heading-span", type=int, default=24, help="chord window for heading")
    ap.add_argument("--heading-mode", choices=("ctrv", "chord"), default="ctrv",
                    help="ctrv: курс+угловая скорость; chord: хорда по истории позиций")
    ap.add_argument("--heading-vel-span", type=int, default=10,
                    help="0: курс из скорости КФ; N: из смещения за N кадров")
    ap.add_argument("--accel", type=float, default=0.6,
                    help="шум ускорения в КФ, м/с² (меньше — глаже, но с запаздыванием)")
    ap.add_argument("--debug", type=int, default=0,
                    help="печатать подробности по первым N кадрам")
    ap.add_argument("--debug-crop", type=int, default=-1,
                    help="сохранить кропы окон на этом кадре")
    ap.add_argument("--roi", default=str(ROOT / "config" / "roi.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = load_scene(Path(args.data))
    model3d = VehicleModel()
    detector = YOLO(args.weights)
    roi_cfg = {}
    if Path(args.roi).exists():
        roi_cfg = json.loads(Path(args.roi).read_text(encoding="utf-8-sig"))

    # --- калибровки, которые живут между кадрами (sample-and-hold) ----------
    # эффективная высота бокса (см), своя на каждую камеру:
    # оценивается на многокамерных кадрах, в одиночных — удерживается
    h_eff = {1: 766.0, 2: 1109.0}
    h_eff_hist = {i: deque(maxlen=40) for i in CAMS}
    # рельеф: высота дороги как функция пройденного пути
    road = RoadModel()
    path_len = 0.0
    last_point = None

    readers = {}
    for idx in CAMS:
        path = scene.video_path(idx)
        if path is not None and Path(path).exists():
            readers[idx] = FrameReader(Path(path))

    frames = [f for f in range(args.lo, args.hi + 1) if f in scene.target]
    start_frame = frames[0] if frames else args.lo

    # стартовый кадр: ROI пользователя на первой камере
    start_box = None
    start_cam = 1
    if "cam1" in roi_cfg:
        start_box = np.array(roi_cfg["cam1"]["box"], dtype=np.float64)
        start_cam = 1

    kf: Kalman3D | None = None
    heading = 0.0
    heading_ready = False
    pos_hist = []
    last_conf: dict[int, float] = {}

    rows = []
    view_rows = []
    track = []
    errors = []

    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(str(out_dir / "validation.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))

    heading_ready = False
    pos_hist = []
    headf = None
    gt_all = scene.target
    t0 = time.time()

    bar = tqdm(frames, desc="full run", unit="frame", ncols=110)
    for f in bar:
        gt = np.asarray(gt_all[f].loc, dtype=np.float64)
        views = []

        for idx in CAMS:
            reader = readers.get(idx)
            if reader is None:
                continue
            frame = reader.get(f)
            if frame is None:
                continue
            cam = Camera.from_track(scene.cameras[idx], f, W, H)

            window = None
            det = None

            if kf is None:
                if start_box is not None and f == start_frame:
                    box0 = start_box.copy() if idx == start_cam else None
                else:
                    # старт без ROI: ищем якорь по всему кадру на любой камере
                    small = cv2.resize(frame, (1280, 720))
                    res = detector.predict(small, imgsz=1280, conf=max(args.conf, 0.20),
                                           verbose=False)[0]
                    got = best_box(res, getattr(detector, "names", {}))
                    box0 = None
                    if got is not None:
                        _, b0 = got
                        box0 = np.array([b0[0] * W / 1280.0, b0[1] * H / 720.0,
                                         b0[2] * W / 1280.0, b0[3] * H / 720.0])
                if box0 is not None:
                    det = (1.0, box0)
                    window = box0.copy()
                    kf = Kalman3D(_single_camera_point(cam, box0, h_eff.get(idx, 800.0)),
                          accel=args.accel)
            else:
                window, obj_w = search_window(cam, kf.point, kf.cov, heading)
                if window is not None and kf.coast >= 3:
                    grow = 1.8 if kf.coast < 10 else 2.8
                    cx, cy = 0.5 * (window[0] + window[2]), 0.5 * (window[1] + window[3])
                    hw, hh = 0.5 * (window[2] - window[0]) * grow, 0.5 * (window[3] - window[1]) * grow
                    window = np.array([cx - hw, cy - hh, cx + hw, cy + hh])
                    window[[0, 2]] = np.clip(window[[0, 2]], 0, W - 1)
                    window[[1, 3]] = np.clip(window[[1, 3]], 0, H - 1)
                if window is not None:
                    det = detect_in_window(detector, frame, window, args.conf, obj_w)

            # ---- если YOLO не увидел (задний/передний ракурс), ведём CSRT ----
            if det is not None:
                last_conf[idx] = det[0]
            views.append(dict(idx=idx, frame=frame, cam=cam, window=window, det=det))
            view_rows.append((f, idx, window, det))

        # ---------------- измерение -------------------------------------- #
        if args.debug_crop == f:
            for v in views:
                if v["window"] is None:
                    continue
                x0, y0, x1, y1 = [int(round(t)) for t in v["window"]]
                crop = v["frame"][y0:y1, x0:x1].copy()
                if v["det"] is not None:
                    b = v["det"][1]
                    cv2.rectangle(crop, (int(b[0] - x0), int(b[1] - y0)),
                                  (int(b[2] - x0), int(b[3] - y0)), (255, 0, 0), 2)
                    cv2.putText(crop, f"conf {v['det'][0]:.2f} h{v['det'][1][3]-v['det'][1][1]:.0f}",
                                (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(crop, "no det", (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imwrite(str(out_dir / f"crop_c{v['idx']}_f{f}.png"), crop)
        if args.debug and f <= args.debug:
            parts = []
            for v in views:
                w, d = v["window"], v["det"]
                ws = "none" if w is None else f"{w[2]-w[0]:.0f}x{w[3]-w[1]:.0f}"
                ds = "none" if d is None else f"{d[0]:.2f}/h{d[1][3]-d[1][1]:.0f}"
                parts.append(f"C{v['idx']} win={ws} det={ds}")
            kp = "none" if kf is None else np.round(kf.point / 100.0, 1)
            print(f"f{f} kf={kp} pt={np.round(kf.point / 100.0, 1) if kf else None} | " + " | ".join(parts))
        measured = [(v["cam"], _bottom_uv(v["det"][1]), v["det"][1])
                    for v in views if v["det"] is not None]
        touch = {v["idx"]: _touches(v["det"][1], v["window"])
                 for v in views if v["det"] is not None}
        windows = {v["idx"]: v["window"] for v in views}
        mode = "coast"
        z = None
        R = None
        if len(measured) >= 2:
            origins = np.array([c.loc for c, _, _ in measured])
            dirs = []
            sig = []
            for cam, uv, _ in measured:
                o, d = cam.ray(uv)
                dirs.append(d)
                sig.append(1.5 / cam.fpx)
            dirs = np.array(dirs)
            point, cov = triangulate(origins, dirs, np.array(sig))
            if np.all(np.isfinite(point)):
                z, R = point, _aniso_R(measured, point, touch)
                mode = f"tri{len(measured)}"
                # калибровки обновляем только по геометрически хорошим кадрам
                if _ray_angle(dirs) > 8.0:
                    _update_h_eff(h_eff, h_eff_hist, measured, point, touch)
                    road.add(path_len, float(point[2]))
        elif len(measured) == 1:
            cam, uv, box = measured[0]
            # одиночная камера: R из фьюжна «размер+земля» (своя глубина), геометрию не навязываем
            z, R, mode = _single_camera_measure(
                cam, box, touch.get(cam.index, False), kf, h_eff, road, path_len)
        else:
            mode = "coast"

        raw_err = None if z is None else float(np.linalg.norm(np.asarray(z) - gt)) / 100.0
        if kf is not None:
            kf.predict()
            if z is not None and R is not None:
                if not kf.update(z, R, gate=30.0):
                    mode += "+rejected"
            else:
                kf.coast += 1

            if last_point is not None:
                path_len += float(np.linalg.norm(kf.point - last_point))
            last_point = kf.point.copy()

            pos_hist.append(kf.point.copy())
            while len(pos_hist) > args.heading_span:
                pos_hist.pop(0)

            if args.heading_mode == "ctrv":
                if headf is None:
                    headf = HeadingFilter(heading=heading, rate=0.0)
                headf.predict()
                vel = kf.vel
                if args.heading_vel_span > 0 and len(pos_hist) > args.heading_vel_span:
                    # короткий базис по позициям КФ: меньше шума, чем мгновенная
                    # скорость, и меньше запаздывания, чем длинная хорда
                    chord = pos_hist[-1] - pos_hist[-1 - args.heading_vel_span]
                    base = float(np.hypot(chord[0], chord[1]))
                    if base > 1.0:
                        h_meas = float(np.degrees(np.arctan2(chord[1], chord[0])))
                        sp = float(np.sqrt(max(np.trace(kf.P[:3, :3]) / 3.0, 1.0)))
                        sig = float(np.clip(np.degrees(np.arctan2(2.0 * sp, base)), 2.0, 90.0))
                    else:
                        h_meas, sig = None, None
                else:
                    speed = float(np.hypot(vel[0], vel[1]))
                    if speed > 20.0:
                        h_meas = float(np.degrees(np.arctan2(vel[1], vel[0])))
                        sv = float(np.sqrt(max(np.trace(kf.P[3:, 3:]) / 3.0, 1.0)))
                        sig = float(np.clip(np.degrees(np.arctan2(sv, speed)), 2.0, 90.0))
                    else:
                        h_meas, sig = None, None
                if h_meas is not None:
                    if not heading_ready:
                        headf.h = h_meas
                        headf.w = 0.0
                        headf.P = np.diag([sig ** 2, 0.6 ** 2])
                        heading_ready = True
                    else:
                        headf.update(h_meas, sig)
                heading = headf.h
            elif len(pos_hist) >= 10:
                n = len(pos_hist)
                k = max(2, n // 4)
                chord = (np.mean(pos_hist[-k:], axis=0)
                         - np.mean(pos_hist[:k], axis=0))
                if float(np.hypot(chord[0], chord[1])) > 300.0:
                    chord_heading = float(np.degrees(np.arctan2(chord[1], chord[0])))
                    if not heading_ready:
                        heading = chord_heading
                        heading_ready = True

            body = kf.point
            fwd = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading)), 0.0])
            pivot = body + fwd * (args.offset_fwd * 100.0) + np.array([0.0, 0.0, args.offset_up * 100.0])
        else:
            pivot = np.zeros(3)

        err = float(np.linalg.norm(pivot - gt)) / 100.0 if kf is not None else float("nan")
        track.append((f, pivot.copy(), kf.point.copy() if kf is not None else None))
        errors.append((f, err, len(measured), mode))
        _g0 = gt_all.get(max(0, f - 3), gt_all[f]).loc
        _g1 = gt_all.get(min(max(gt_all), f + 3), gt_all[f]).loc
        hdg_gt = float(np.degrees(np.arctan2(_g1[1] - _g0[1], _g1[0] - _g0[0])))
        rows.append(dict(hdg_gt=hdg_gt, raw_err=raw_err, frame=f, x_cm=pivot[0], y_cm=pivot[1], z_cm=pivot[2],
                         err_m=err, n_cams=len(measured), mode=mode,
                         heading=heading, coast=kf.coast if kf else 0))

        # ---------------- рендер ------------------------------------------ #
        if writer is not None:
            canvas = _render(f, views, track, errors, model3d, scene, gt, pivot,
                             kf, mode, len(measured), heading, heading_ready)
            writer.write(canvas)

        if f % 25 == 0 and errors:
            recent = [e[1] for e in errors[-25:] if np.isfinite(e[1])]
            bar.set_postfix({"err": f"{np.median(recent):.2f}m" if recent else "n/a",
                             "cams": len(measured), "mode": mode})

    bar.close()
    if writer is not None:
        writer.release()
    for r in readers.values():
        r.release()

    # ---------------- отчёт ----------------------------------------------- #
    valid = [e[1] for e in errors if np.isfinite(e[1])]
    if valid:
        arr = np.array(valid)
        summary = dict(n=len(arr), coverage=len(arr) / max(1, len(frames)),
                       median_m=float(np.median(arr)), mean_m=float(arr.mean()),
                       p90_m=float(np.percentile(arr, 90)), max_m=float(arr.max()),
                       rmse_m=float(np.sqrt(np.mean(arr ** 2))))
        print("\n=== итог ===")
        for k, v in summary.items():
            print(f"{k:>10}: {v:.3f}" if isinstance(v, float) else f"{k:>10}: {v}")
        warm = [e[1] for e in errors if e[0] > 10 and np.isfinite(e[1])]
        if warm:
            print(f"без первых 10 кадров: median {np.median(warm):.2f} m, "
                  f"mean {np.mean(warm):.2f}, p90 {np.percentile(warm, 90):.2f}, "
                  f"max {max(warm):.2f}, rmse {np.sqrt(np.mean(np.square(warm))):.2f}")
        (out_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print("калибровки h_eff (см):", {k: round(v, 1) for k, v in sorted(h_eff.items())})
        print(f"дорога: z={road.a:.1f} см, уклон={road.b * 100:.2f}%")

    with open(out_dir / "track.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["Frame", "Loc_X", "Loc_Y", "Loc_Z"])
        for r in rows:
            w.writerow([r["frame"], f"{r['x_cm']:.2f}", f"{r['y_cm']:.2f}", f"{r['z_cm']:.2f}"])

    with open(out_dir / "views.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["frame", "cam", "win_x0", "win_y0", "win_x1", "win_y1",
                    "det_x0", "det_y0", "det_x1", "det_y1", "conf"])
        for fr, idx, win, det in view_rows:
            wb = [] if win is None else [f"{v:.1f}" for v in win]
            db = [] if det is None else [f"{v:.1f}" for v in det[1]]
            w.writerow([fr, idx] + wb + [""] * (4 - len(wb)) + db + [""] * (4 - len(db))
                       + ["" if det is None else f"{det[0]:.3f}"])

    with open(out_dir / "debug.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nвремя: {time.time() - t0:.0f} с")
    print(f"csv: {out_dir / 'track.csv'}")
    if writer is not None:
        print(f"видео: {out_dir / 'validation.mp4'}")
    return 0


# --------------------------------------------------------------------------- #
def _bottom_uv(box):
    return np.array([0.5 * (box[0] + box[2]), box[3]], dtype=np.float64)


def _box_from_uv(views, idx):
    for v in views:
        if v["idx"] == idx and v["det"] is not None:
            return v["det"][1]
    return None


def _single_camera_point(cam, box, h_eff_cm):
    uv = _bottom_uv(box)
    h_px = max(4.0, float(box[3] - box[1]))
    depth = cam.fy * h_eff_cm / h_px
    origin, direction = cam.ray(uv)
    return origin + direction * depth


def _touches(box, window, margin=3.0):
    if window is None:
        return False
    return bool(abs(box[0] - window[0]) < margin or abs(box[1] - window[1]) < margin
                or abs(box[2] - window[2]) < margin or abs(box[3] - window[3]) < margin)


ROAD_SIGMA = 50.0   # см, насколько может ошибаться высота дороги


class RoadModel:
    """Высота дороги z(s) — почти линейная функция пройденного пути."""

    def __init__(self, z0=0.0):
        self.samples = deque(maxlen=400)
        self.a, self.b, self.s0 = float(z0), 0.0, 0.0
        self.ready = False

    def add(self, s, z):
        if not (np.isfinite(z) and -300.0 < z < 1500.0):
            return
        self.samples.append((float(s), float(z)))
        if len(self.samples) < 8:
            return
        ss = np.array([p[0] for p in self.samples])
        zz = np.array([p[1] for p in self.samples])
        if float(np.ptp(ss)) > 2000.0:
            self.s0 = float(ss.mean())
            A = np.stack([np.ones_like(ss), ss - self.s0], 1)
            sol, *_ = np.linalg.lstsq(A, zz, rcond=None)
            self.a, self.b = float(sol[0]), float(np.clip(sol[1], -0.06, 0.06))
        else:
            self.a, self.b, self.s0 = float(np.median(zz)), 0.0, float(ss.mean())
        self.ready = True

    def height(self, s):
        return self.a + self.b * (float(s) - self.s0)


def _ray_angle(dirs):
    best = 0.0
    n = len(dirs)
    for i in range(n):
        for j in range(i + 1, n):
            c = float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
            best = max(best, float(np.degrees(np.arccos(c))))
    return best


def _aniso_R(measured, point, touch, floor_cm: float = 50.0) -> np.ndarray:
    """Ковариация измерения из геометрии: поперёк луча по пикселям, вдоль — по размеру бокса.

    Та же матрица точности, что в tools/gsmooth.py у глобального сглаживания:
        I = sum_k [ P_perp/sl^2 + ray ray^T/sd^2 ],  R = inv(I).
    Вдоль луча доверие маленькое: глубина по высоте бокса зависит от ракурса.
    """
    I = np.zeros((3, 3))
    p = np.asarray(point, float)
    for cam, uv, box in measured:
        o, d = cam.ray(uv)
        d = np.asarray(d, float)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            continue
        d = d / n
        dist = float(np.linalg.norm(p - o))
        w = float(box[2] - box[0])
        h = float(box[3] - box[1])
        size = float(np.sqrt(max(w * h, 1.0)))
        sl = 2.0 * dist / cam.fpx + 70.0
        sd = 4.0 * dist / max(size, 5.0) + 70.0
        if touch.get(cam.index, False):
            sl *= 2.0
            sd *= 4.0
        perp = np.eye(3) - np.outer(d, d)
        I += perp / sl ** 2 + np.outer(d, d) / sd ** 2
    I += np.eye(3) * 1e-9
    return np.linalg.inv(I)


def _single_camera_measure(cam, box, clipped, kf, h_eff, road, path_len):
    """Одиночная камера: фьюжн «размер бокса» и «пересечение луча с дорогой»."""
    h_px = float(box[3] - box[1])
    if h_px < 8.0:
        return None, None, "single-small"
    o, d = cam.ray(_bottom_uv(box))
    if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
        return None, None, "single-nan"
    fwd = cam.basis[0]
    cos_axis = max(float(np.dot(d, fwd)), 1e-3)
    depth_kf = float(np.dot(kf.point - cam.loc, fwd)) if kf is not None else 0.0

    cand = []                                   # (tag, t по лучу, sigma_t)
    he = h_eff.get(cam.index)
    if he is not None and not clipped and h_px > 25.0:
        depth = cam.fy * he / h_px
        # грузовик не бывает в 2 раза дальше/ближе прошлого кадра
        ok = depth_kf <= 200.0 or abs(depth - depth_kf) <= 0.45 * depth_kf + 400.0
        if ok and depth_kf > 500.0:
            # резкое уменьшение бокса = фура ушла за гребень/частично закрыта,
            # её высота больше не равна габариту, а значит и не мера дальности
            exp_h = cam.fy * he / depth_kf
            if not (0.82 < h_px / max(exp_h, 1.0) < 1.22):
                ok = False
        if ok:
            cand.append(("size", depth / cos_axis, max(depth * 3.0 / h_px, 50.0)))

    if d[2] < -1e-6:                            # луч смотрит вниз — есть земля
        t = (road.height(path_len) - o[2]) / d[2]
        if 300.0 < t < 3.0e4:
            cand.append(("ground", t, max(ROAD_SIGMA / abs(d[2]), 50.0)))

    if not cand:
        return None, None, "single-none"

    wsum = sum(1.0 / s ** 2 for _, _, s in cand)
    t = sum(tt / s ** 2 for _, tt, s in cand) / wsum
    sigma_t = float(np.sqrt(1.0 / wsum))
    point = o + d * t

    tag = "+".join(sorted({c[0] for c in cand}))
    if len(cand) == 2:
        (_, t1, s1), (_, t2, s2) = cand
        if abs(t1 - t2) > 3.0 * float(np.hypot(s1, s2)):
            tag += "!"                          # размер и земля расходятся
    sigma_p = max(2.0 * t / cam.fpx, 120.0)
    R = sigma_t ** 2 * np.outer(d, d) + sigma_p ** 2 * (np.eye(3) - np.outer(d, d))
    return point, R, "single:" + tag + ("*" if clipped else "")


def _update_h_eff(h_eff, hist, measured, point, touch):
    """Обновляет эффективную высоту бокса только по полным, не обрезанным боксам."""
    for cam, _uv, box in measured:
        if touch.get(cam.index):
            continue
        h_px = float(box[3] - box[1])
        if h_px < 25.0:
            continue
        depth = float(np.dot(point - cam.loc, cam.basis[0]))
        if depth <= 0:
            continue
        sample = h_px * depth / cam.fy
        if not (100.0 < sample < 2600.0):
            continue
        hist[cam.index].append(sample)
        cur = h_eff.get(cam.index)
        if cur is None:
            if len(hist[cam.index]) >= 4:
                h_eff[cam.index] = float(np.median(hist[cam.index]))
        else:
            h_eff[cam.index] = float(0.8 * cur + 0.2 * sample)


# --------------------------------------------------------------------------- #
def _render(f, views, track, errors, model3d, scene, gt, pivot, kf, mode,
            n_cams, heading, heading_ready):
    tw, th = 640, 360
    canvas = np.full((H, W, 3), 14, np.uint8)

    for slot, idx in enumerate(CAMS):
        view = next((v for v in views if v["idx"] == idx), None)
        col, row = slot % 3, slot // 3
        ox, oy = col * tw, row * th
        if view is None:
            cv2.putText(canvas, f"CAM{idx}: no frame", (ox + 10, oy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
            continue
        tile, scale, dx, dy = fit_tile(view["frame"], tw, th)
        off = np.array([dx, dy, dx, dy])
        if view["window"] is not None:
            b = view["window"] * scale + off
            cv2.rectangle(tile, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), WIN_COLOR, 2)
        if view["det"] is not None:
            b = view["det"][1] * scale + off
            cv2.rectangle(tile, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), DET_COLOR, 2)
        head = f"CAM{idx}"
        if view["det"] is not None:
            head += f"  det {view['det'][0]:.2f}"
        cv2.putText(tile, head, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    TXT, 1, cv2.LINE_AA)
        if view["det"] is not None:
            cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), BORDER_COLOR, 4)
        canvas[oy:oy + th, ox:ox + tw] = tile

    # топ-вид
    ox, oy = 2 * tw, th
    top = np.full((th, tw, 3), 12, np.uint8)
    pts = np.array([p[1][:2] for p in track]) / 100.0
    if len(pts) > 1:
        lo = pts.min(axis=0) - 60
        hi = pts.max(axis=0) + 60
        span = np.maximum(hi - lo, 1.0)
        sc = min((tw - 40) / span[0], (th - 40) / span[1])
        def m2p(p):
            return (int(20 + (p[0] - lo[0]) * sc), int(th - 20 - (p[1] - lo[1]) * sc))
        gt_pts = np.array([scene.target[q].loc[:2] for q, _, _ in track]) / 100.0
        for i in range(1, len(gt_pts)):
            cv2.line(top, m2p(gt_pts[i - 1]), m2p(gt_pts[i]), GT_COLOR, 1)
        for i in range(1, len(pts)):
            cv2.line(top, m2p(pts[i - 1]), m2p(pts[i]), (60, 60, 255), 1)
        cv2.circle(top, m2p(gt_pts[-1]), 4, GT_COLOR, -1)
        cv2.drawMarker(top, m2p(pts[-1]), (60, 60, 255), cv2.MARKER_CROSS, 10, 2)
        for idx in CAMS:
            cam = Camera.from_track(scene.cameras[idx], f, W, H)
            c = m2p(cam.loc[:2] / 100.0)
            cv2.rectangle(top, (c[0] - 3, c[1] - 3), (c[0] + 3, c[1] + 3), (150, 150, 150), -1)
            cv2.putText(top, f"C{idx}", (c[0] + 5, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(top, "3D: GT green, estimate red", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, TXT, 1, cv2.LINE_AA)
    canvas[oy:oy + th, ox:ox + tw] = top

    # нижняя панель: график ошибки + статус
    by = 2 * th
    panel = np.full((H - by, W, 3), 10, np.uint8)
    gw = 1100
    errs = np.array([e[1] if np.isfinite(e[1]) else 0.0 for e in errors])
    if len(errs) > 1:
        ymax = max(2.0, float(np.percentile(errs, 95)) * 1.15)
        gh = H - by - 40
        for i in range(1, len(errs)):
            x0 = int(20 + (i - 1) * (gw - 20) / max(1, len(errs) - 1))
            x1 = int(20 + i * (gw - 20) / max(1, len(errs) - 1))
            y0 = int(by + 20 + gh - min(errs[i - 1], ymax) / ymax * gh)
            y1 = int(by + 20 + gh - min(errs[i], ymax) / ymax * gh)
            cv2.line(panel, (x0, y0 - by), (x1, y1 - by), (90, 200, 255), 1)
        med = float(np.median(errs))
        ym = int(by + 20 + gh - min(med, ymax) / ymax * gh) - by
        cv2.line(panel, (20, ym), (gw, ym), (60, 220, 60), 1)
        cv2.putText(panel, f"median {med:.2f} m", (gw - 190, ym - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1, cv2.LINE_AA)
        cv2.putText(panel, f"error vs GT (m), scale {ymax:.1f}", (22, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TXT, 1, cv2.LINE_AA)

    lines = [f"frame {f}   t={f / 30.0:.2f}s",
             f"mode: {mode}   cams with det: {n_cams}",
             f"est (m): {pivot[0] / 100:9.1f} {pivot[1] / 100:9.1f} {pivot[2] / 100:7.1f}",
             f"GT  (m): {gt[0] / 100:9.1f} {gt[1] / 100:9.1f} {gt[2] / 100:7.1f}",
             f"error: {np.linalg.norm(pivot - gt) / 100:.2f} m",
             f"heading: {heading:+.1f} deg" + ("" if heading_ready else "  (warm-up)")]
    if kf is not None:
        lines.append(f"coast: {kf.coast}")
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (gw + 30, 40 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, TXT, 1, cv2.LINE_AA)
    cv2.putText(panel, "orange = search window, blue = YOLO detection, frame border = camera sees the truck",
                (22, H - by - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)

    canvas[by:H, 0:W] = panel
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
