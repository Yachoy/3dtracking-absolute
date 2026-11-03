"""Детекция транспорта на всех кадрах всех камер, результат кэшируется в CSV.

Каждый кадр обрабатывается независимо, так что кэш не нарушает онлайн-режим:
трекер всё равно читает детекции кадра t только на шаге t.
"""
import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from tracker.camera import load_cameras, video_path

# COCO: 2 car, 5 bus, 7 truck — грузовик издалека детектор часто путает с bus/car
VEHICLE_CLASSES = [2, 5, 7]


def detect_video(model, path, imgsz, conf, batch=8, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    rows, frames, idx = [], [], []
    frame_no = 0

    def flush():
        for f, res in zip(idx, model.predict(frames, imgsz=imgsz, conf=conf, classes=VEHICLE_CLASSES, verbose=False)):
            for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
                rows.append((f, x1, y1, x2, y2, c, int(k)))
        frames.clear(), idx.clear()

    while max_frames is None or frame_no < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img), idx.append(frame_no)
        if len(frames) == batch:
            flush()
        frame_no += 1
    if frames:
        flush()
    return pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "conf", "cls"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="cache/detections.csv")
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    parts = []
    for cam in load_cameras(args.data_dir):
        t0 = time.time()
        df = detect_video(model, video_path(args.data_dir, cam.name), args.imgsz, args.conf, max_frames=args.max_frames)
        df.insert(0, "camera", cam.name)
        parts.append(df)
        print(f"{cam.name}: {len(df)} детекций, {df.frame.nunique()} кадров с детекциями, {time.time() - t0:.0f} с")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_csv(args.out, index=False)
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()
