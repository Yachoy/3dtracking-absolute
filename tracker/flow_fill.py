"""Заполнение пропусков YOLO 2D-трекером по оптическому потоку (в духе Median Flow).

Для каждой камеры:
- вперёд: от последней рамки YOLO ведём рамку по потоку на следующих кадрах, пока YOLO молчит
  (рамка известна в тот же кадр — онлайн);
- назад: когда YOLO снова нашла объект после пропуска, ведём рамку от неё назад по прошедшим кадрам
  (рамка для кадра f становится известна только на кадре `known_at` > f).

Движение камеры вычитается: центры камер неподвижны, камера 5 только поворачивается, поэтому смещение
фона между кадрами — точная гомография K·R_to·R_fromᵀ·K⁻¹ из известных поз. Точки внутри рамки,
которые двигаются как фон, отбрасываются — рамку ведут только точки движущегося объекта.

    python -m tracker.flow_fill data/test --out cache/flow_boxes.csv
"""
import argparse
from collections import deque

import cv2
import numpy as np
import pandas as pd

from tracker.camera import load_cameras, video_path

FLOW_SCALE = 0.5          # поток считаем на кадре 960x540
SEED_CONF = 0.25          # рамки YOLO, от которых стартуем
MAX_AGE = 90              # не дальше 3 с от рамки YOLO
MIN_POINTS = 8
FB_MAX_PX = 1.0           # ошибка «вперёд-назад», пикселей (в масштабе FLOW_SCALE)
MOVING_PX = 0.4           # остаточное (после вычета движения камеры) смещение точки объекта
STILL_PX = 1.0            # если объект только что ехал быстрее — без движущихся точек считаем его потерянным
LK = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def ego_homography(pose_from, pose_to, scale):
    """Как сдвигается фон (точки «на бесконечности») между кадрами при чистом повороте камеры."""
    K = np.array([[pose_from.fx * scale, 0, pose_from.cx * scale],
                  [0, pose_from.fy * scale, pose_from.cy * scale],
                  [0, 0, 1]])
    return K @ pose_to.rot @ pose_from.rot.T @ np.linalg.inv(K)


def propagate(img_from, img_to, box, H_ego, prev_shift=None):
    """Перенести рамку (в масштабе FLOW_SCALE) с img_from на img_to.

    Возвращает (рамка, доля надёжных точек, смещение) или None — объект потерян.
    prev_shift — смещение на прошлом шаге: если объект ехал, а движущихся точек больше нет,
    значит он перекрыт или ушёл, и рамка иначе «прилипла» бы к фону.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 4:
        return None
    gx, gy = np.meshgrid(np.linspace(x1 + 0.1 * w, x2 - 0.1 * w, 12), np.linspace(y1 + 0.1 * h, y2 - 0.1 * h, 8))
    p0 = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)

    p1, st1, _ = cv2.calcOpticalFlowPyrLK(img_from, img_to, p0, None, **LK)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(img_to, img_from, p1, None, **LK)
    p0, p1, p0b = p0.reshape(-1, 2), p1.reshape(-1, 2), p0b.reshape(-1, 2)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p0 - p0b, axis=1) < FB_MAX_PX)

    bg = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H_ego).reshape(-1, 2)  # куда ушёл бы фон
    residual = np.linalg.norm(p1 - bg, axis=1)
    moving = ok & (residual > MOVING_PX)
    pts = moving if moving.sum() >= MIN_POINTS else None
    if pts is None:
        # объекта не видно в движении: либо он стоит, либо перекрыт/ушёл
        if ok.mean() < 0.7 or (prev_shift is not None and np.linalg.norm(prev_shift) > STILL_PX):
            return None
        pts = ok
    a, b = p0[pts], p1[pts]
    shift = np.median(b - a, axis=0)
    i, j = np.triu_indices(len(a), 1)
    da, db = np.linalg.norm(a[i] - a[j], axis=1), np.linalg.norm(b[i] - b[j], axis=1)
    keep = da > 2
    s = float(np.clip(np.median(db[keep] / da[keep]), 0.9, 1.1)) if keep.any() else 1.0
    cx, cy = (x1 + x2) / 2 + shift[0], (y1 + y2) / 2 + shift[1]
    score = float(pts.sum() / len(pts))
    return np.array([cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2]), score, shift


def seeds_by_frame(dets, camera):
    """Лучшая рамка YOLO на кадре (conf >= SEED_CONF)."""
    d = dets[(dets.camera == camera) & (dets.conf >= SEED_CONF)]
    d = d.loc[d.groupby("frame").conf.idxmax()]
    return {int(r.frame): (np.array([r.x1, r.y1, r.x2, r.y2]), float(r.conf)) for r in d.itertuples()}


def fill_camera(cam, path, seeds, n_frames=None):
    cap = cv2.VideoCapture(str(path))
    buf = deque(maxlen=MAX_AGE + 1)  # (frame, gray, pose) — только прошлые кадры
    rows = []
    cur, cur_age, cur_conf, cur_shift = None, 0, 0.0, None
    frame = 0
    while n_frames is None or frame < n_frames:
        ok, img = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(img, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        pose = cam.pose(frame)

        if frame in seeds:
            box, conf = seeds[frame]
            # назад: YOLO вернулась после пропуска — идём по буферу прошлых кадров
            if buf and buf[-1][0] not in seeds:
                b, prev_gray, prev_pose, shift = box * FLOW_SCALE, gray, pose, None
                for f_back, g_back, p_back in reversed(buf):
                    if f_back in seeds or frame - f_back > MAX_AGE:
                        break
                    res = propagate(prev_gray, g_back, b, ego_homography(prev_pose, p_back, FLOW_SCALE), shift)
                    if res is None:
                        break
                    b, score, shift = res
                    rows.append((cam.name, f_back, *(b / FLOW_SCALE), conf * score, "bwd", frame - f_back, frame))
                    prev_gray, prev_pose = g_back, p_back
            cur, cur_age, cur_conf, cur_shift = box * FLOW_SCALE, 0, conf, None
        elif cur is not None and buf and cur_age < MAX_AGE:
            # вперёд: YOLO молчит — ведём последнюю рамку по потоку
            _, prev_gray, prev_pose = buf[-1]
            res = propagate(prev_gray, gray, cur, ego_homography(prev_pose, pose, FLOW_SCALE), cur_shift)
            if res is None:
                cur = None
            else:
                cur, score, cur_shift = res
                cur_age += 1
                rows.append((cam.name, frame, *(cur / FLOW_SCALE), cur_conf * score, "fwd", cur_age, frame))
        buf.append((frame, gray, pose))
        frame += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--detections", default="cache/detections.csv")
    ap.add_argument("--out", default="cache/flow_boxes.csv")
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args()

    dets = pd.read_csv(args.detections)
    rows = []
    for cam in load_cameras(args.data_dir):
        r = fill_camera(cam, video_path(args.data_dir, cam.name), seeds_by_frame(dets, cam.name), args.max_frames)
        n_fwd = sum(x[7] == "fwd" for x in r)
        print(f"{cam.name}: вперёд {n_fwd}, назад {len(r) - n_fwd} рамок")
        rows += r
    cols = ["camera", "frame", "x1", "y1", "x2", "y2", "conf", "source", "age", "known_at"]
    pd.DataFrame(rows, columns=cols).to_csv(args.out, index=False)
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()
