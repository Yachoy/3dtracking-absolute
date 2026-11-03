"""Одна команда: папка с видео + CSV камер -> CSV с 3D-траекторией объекта.

    python run.py data/test                       # трекинг по кэшу детекций
    python run.py data/test --calibrate           # подобрать высоту объекта и сдвиг до точки-ответа по GT
    python run.py data/test --flow                # + рамки 2D-трекера по потоку между детекциями YOLO
    python run.py data/test --flow --replay       # + пересчёт трека по рамкам обратного прохода
    python evaluate.py data/test out/trajectory.csv
"""
import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from tracker.camera import CM, load_cameras, load_ground_truth
from tracker.online import Detection, OnlineTracker3D, TrackerConfig
from tracker.truck_model import MODEL_FILE, TruckModel

CONFIG_FILE = "config/tracker.json"


def load_config(use_model=True, **overrides):
    """Параметры трекера; 3D-модель объекта подключается, если есть config/truck_model.json."""
    cfg = TrackerConfig.load(CONFIG_FILE, **overrides)
    if use_model and Path(MODEL_FILE).exists():
        cfg = replace(cfg, model=TruckModel.load(MODEL_FILE))
    return cfg


def detections_by_frame(det_csv, cameras, flow_csv=None):
    """{кадр: [Detection]}. Рамки YOLO известны в свой кадр, рамки потока — с known_at из flow_csv."""
    cam_idx = {c.name: i for i, c in enumerate(cameras)}
    out = {}
    for row in pd.read_csv(det_csv).itertuples(index=False):
        out.setdefault(int(row.frame), []).append(
            Detection(cam_idx[row.camera], np.array([row.x1, row.y1, row.x2, row.y2]), float(row.conf),
                      known_at=int(row.frame)))
    if flow_csv:
        for row in pd.read_csv(flow_csv).itertuples(index=False):
            out.setdefault(int(row.frame), []).append(
                Detection(cam_idx[row.camera], np.array([row.x1, row.y1, row.x2, row.y2]), float(row.conf),
                          source=row.source, age=int(row.age), known_at=int(row.known_at)))
    return out


def _row(frame, point, status, t):
    p = point if point is not None else np.full(3, np.nan)
    return {
        "Frame": frame, "Loc_X": p[0] * CM, "Loc_Y": p[1] * CM, "Loc_Z": p[2] * CM,
        "status": status,
        "cameras": "".join(str(c + 1) for c in t.used_cameras) if t else "",
        "heading": t.heading if t else np.nan,
        "cx": t.x[0] if t else np.nan, "cy": t.x[1] if t else np.nan, "cz": t.x[2] if t else np.nan,
    }


def track(cameras, dets, cfg, replay=False):
    """Онлайн-проход: на шаге t видны только рамки с known_at <= t.

    replay=True: когда на кадре t становятся известны рамки обратного прохода для кадров a..t-1,
    трек откатывается к состоянию после кадра a-1 и пересчитывается до t. Выданные раньше ответы
    не меняются (возвращается online), но состояние на кадре t точнее. Пересчитанные ответы для
    прошлых кадров возвращаются отдельно (delayed) — это уже не строгий онлайн.
    """
    tracker = OnlineTracker3D(cameras, cfg)
    replay_from = {}  # known_at -> самый ранний кадр, для которого тогда появляются рамки
    for f, ds in dets.items():
        for d in ds:
            if d.known_at > f:
                replay_from[d.known_at] = min(replay_from.get(d.known_at, f), f)

    def visible(frame, now):
        return [d for d in dets.get(frame, []) if d.known_at <= now]

    snapshots, online, delayed = {}, [], {}
    for t in cameras[0].table.index:
        if replay and t in replay_from:
            a = replay_from[t]
            tracker.track = copy.deepcopy(snapshots.get(a - 1))
            for f in range(a, t):
                point, status = tracker.step(f, visible(f, t))
                delayed[f] = _row(f, point, status, tracker.track)
                snapshots[f] = copy.deepcopy(tracker.track)
        point, status = tracker.step(t, visible(t, t))
        online.append(_row(t, point, status, tracker.track))
        delayed[t] = online[-1]
        snapshots[t] = copy.deepcopy(tracker.track)
    online = pd.DataFrame(online)
    return (online, pd.DataFrame([delayed[f] for f in online.Frame])) if replay else online


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
    ap.add_argument("--no-model", action="store_true", help="не использовать 3D-модель объекта (" + MODEL_FILE + ")")
    ap.add_argument("--no-single-init", action="store_true",
                    help="не стартовать трек по одной камере: меньше покрытие, но точнее на покрытых кадрах")
    ap.add_argument("--flow", nargs="?", const="cache/flow_boxes.csv",
                    help="добавить рамки 2D-трекера по потоку (python -m tracker.flow_fill)")
    ap.add_argument("--replay", action="store_true",
                    help="пересчитывать трек по рамкам обратного прохода; дополнительно пишет *_delayed.csv")
    args = ap.parse_args()

    cameras = load_cameras(args.data_dir)
    if not Path(args.detections).exists():
        raise SystemExit(f"нет {args.detections} — сначала: python -m tracker.detect {args.data_dir}")
    yolo_dets = detections_by_frame(args.detections, cameras)
    dets = detections_by_frame(args.detections, cameras, args.flow) if args.flow else yolo_dets
    cfg = load_config(not args.no_model, single_camera_init=not args.no_single_init)
    print("3D-модель объекта:", "да" if cfg.model is not None else "нет")

    if args.calibrate:
        gt = load_ground_truth(args.data_dir)
        cfg = replace(cfg, object_height_m=calibrate_height(cameras, yolo_dets, gt), offset=(0.0, 0.0, 0.0))
        # сдвиг нужен только режиму без модели (с моделью точка-ответ берётся из неё)
        cfg = replace(cfg, offset=calibrate_offset(track(cameras, yolo_dets, replace(cfg, model=None)), gt))
        Path(CONFIG_FILE).parent.mkdir(exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"object_height_m": cfg.object_height_m, "offset_fwd_left_up_m": cfg.offset}, f, indent=2)
        print(f"высота объекта: {cfg.object_height_m:.2f} м, сдвиг (вперёд, влево, вверх): {np.round(cfg.offset, 2)} м")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.replay:
        result, delayed = track(cameras, dets, cfg, replay=True)
        delayed_path = Path(args.out).with_name(Path(args.out).stem + "_delayed.csv")
        delayed[["Frame", "Loc_X", "Loc_Y", "Loc_Z"]].to_csv(delayed_path, sep=";", index=False, float_format="%.4f")
        print("с задержкой (не строгий онлайн):", delayed_path)
    else:
        result = track(cameras, dets, cfg)
    result[["Frame", "Loc_X", "Loc_Y", "Loc_Z"]].to_csv(args.out, sep=";", index=False, float_format="%.4f")
    result.to_csv(Path(args.out).with_name("debug.csv"), index=False)
    print("сохранено:", args.out, "| статусы:", result.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
