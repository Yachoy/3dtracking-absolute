"""Одна команда: папка с видео + CSV камер -> CSV с 3D-траекторией объекта.

    python run.py data/test                       # трекинг по кэшу детекций
    python run.py data/test --calibrate           # подобрать высоту объекта и сдвиг до точки-ответа по GT
    python evaluate.py data/test out/trajectory.csv
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from tracker.camera import CM, load_cameras, load_ground_truth
from tracker.online import Detection, OnlineTracker3D, TrackerConfig

CONFIG_FILE = "config/tracker.json"


def detections_by_frame(det_csv, cameras):
    df = pd.read_csv(det_csv)
    cam_idx = {c.name: i for i, c in enumerate(cameras)}
    out = {}
    for row in df.itertuples(index=False):
        out.setdefault(int(row.frame), []).append(
            Detection(cam_idx[row.camera], np.array([row.x1, row.y1, row.x2, row.y2]), float(row.conf)))
    return out


def track(cameras, dets, cfg):
    tracker = OnlineTracker3D(cameras, cfg)
    rows = []
    for frame in cameras[0].table.index:  # строго по порядку, на шаге t — только детекции кадра t
        point, status = tracker.step(frame, dets.get(frame, []))
        t = tracker.track
        p = point if point is not None else np.full(3, np.nan)
        rows.append({
            "Frame": frame, "Loc_X": p[0] * CM, "Loc_Y": p[1] * CM, "Loc_Z": p[2] * CM,
            "status": status,
            "cameras": "".join(str(c + 1) for c in t.used_cameras) if t else "",
            "heading": t.heading if t else np.nan,
            "cx": t.x[0] if t else np.nan, "cy": t.x[1] if t else np.nan, "cz": t.x[2] if t else np.nan,
        })
    return pd.DataFrame(rows)


def calibrate_height(cameras, dets, gt, pad_m=2.0):
    """Высота объекта: медиана h_px * дальность / fy по рамкам, в которые попадает проекция GT."""
    heights = []
    for frame, frame_dets in dets.items():
        if frame not in gt.index or gt.loc[frame].isna().any():
            continue
        for d in frame_dets:
            pose = cameras[d.camera].pose(frame)
            uv, z = pose.project(gt.loc[frame].values)
            pad = pose.fx * pad_m / max(z[0], 1.0)
            x1, y1, x2, y2 = d.box
            if z[0] > 0 and x1 - pad <= uv[0, 0] <= x2 + pad and y1 - pad <= uv[0, 1] <= y2 + pad:
                heights.append(d.height * z[0] / pose.fy)
    return float(np.median(heights))


def calibrate_offset(result, gt):
    """Медианный сдвиг от точки трека до точки GT в системе (вперёд, влево, вверх)."""
    df = result.merge(gt.add_suffix("_gt"), left_on="Frame", right_index=True, how="left").dropna(
        subset=["cx", "Loc_X_gt", "Loc_Y_gt", "Loc_Z_gt"])
    df = df[(df.status == "TRACKING") & (df.cameras.astype(str).str.len() >= 2)]  # только кадры с ≥2 камерами
    d = df[["Loc_X_gt", "Loc_Y_gt", "Loc_Z_gt"]].values - df[["cx", "cy", "cz"]].values
    h = df.heading.values
    fwd = d[:, 0] * np.cos(h) + d[:, 1] * np.sin(h)
    left = -d[:, 0] * np.sin(h) + d[:, 1] * np.cos(h)
    return (float(np.median(fwd)), float(np.median(left)), float(np.median(d[:, 2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--detections", default="cache/detections.csv")
    ap.add_argument("--out", default="out/trajectory.csv")
    ap.add_argument("--calibrate", action="store_true", help="подобрать параметры по GT и сохранить в " + CONFIG_FILE)
    ap.add_argument("--no-single-init", action="store_true",
                    help="не стартовать трек по одной камере: меньше покрытие, но точнее на покрытых кадрах")
    args = ap.parse_args()

    cameras = load_cameras(args.data_dir)
    if not Path(args.detections).exists():
        raise SystemExit(f"нет {args.detections} — сначала: python -m tracker.detect {args.data_dir}")
    dets = detections_by_frame(args.detections, cameras)
    cfg = TrackerConfig.load(CONFIG_FILE, single_camera_init=not args.no_single_init)

    if args.calibrate:
        gt = load_ground_truth(args.data_dir)
        cfg = replace(cfg, object_height_m=calibrate_height(cameras, dets, gt), offset=(0.0, 0.0, 0.0))
        cfg = replace(cfg, offset=calibrate_offset(track(cameras, dets, cfg), gt))
        Path(CONFIG_FILE).parent.mkdir(exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"object_height_m": cfg.object_height_m, "offset_fwd_left_up_m": cfg.offset}, f, indent=2)
        print(f"высота объекта: {cfg.object_height_m:.2f} м, сдвиг (вперёд, влево, вверх): {np.round(cfg.offset, 2)} м")

    result = track(cameras, dets, cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result[["Frame", "Loc_X", "Loc_Y", "Loc_Z"]].to_csv(args.out, sep=";", index=False, float_format="%.4f")
    result.to_csv(Path(args.out).with_name("debug.csv"), index=False)
    print("сохранено:", args.out, "| статусы:", result.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
