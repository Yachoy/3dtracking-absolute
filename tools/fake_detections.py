"""Синтетические детекции из GT — чтобы отлаживать трекер без нейросети.

Рамка = проекция центра грузовика (5 м назад от точки GT, 1.7 м вверх) + шум,
часть детекций выкидывается (перекрытия), добавляются ложные рамки.

    python -m tools.fake_detections data/test --out cache/fake_detections.csv
"""
import argparse

import numpy as np
import pandas as pd

from tracker.camera import load_cameras, load_ground_truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="cache/fake_detections.csv")
    ap.add_argument("--noise-px", type=float, default=3.0)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--false-per-frame", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cams = load_cameras(args.data_dir)
    gt = load_ground_truth(args.data_dir).interpolate()
    vel = gt.diff().bfill().values
    rows = []
    for i, frame in enumerate(gt.index):
        h = vel[i, :2] / (np.linalg.norm(vel[i, :2]) + 1e-9)
        center = gt.loc[frame].values + np.r_[-5.0 * h, 1.7]
        for cam in cams:
            pose = cam.pose(frame)
            uv, z = pose.project(center)
            if z[0] > 0 and rng.random() > args.drop:
                u, v = uv[0] + rng.normal(0, args.noise_px, 2)
                w, hh = cam.fx * 9.0 / z[0], cam.fy * 3.5 / z[0]
                if 0 < u - w / 2 and u + w / 2 < cam.width and 0 < v - hh / 2 and v + hh / 2 < cam.height:
                    rows.append((cam.name, frame, u - w / 2, v - hh / 2, u + w / 2, v + hh / 2, rng.uniform(0.3, 0.9), 7))
            for _ in range(rng.poisson(args.false_per_frame)):
                u, v, s = rng.uniform(100, cam.width - 100), rng.uniform(100, cam.height - 100), rng.uniform(10, 80)
                rows.append((cam.name, frame, u - s, v - s / 2, u + s, v + s / 2, rng.uniform(0.1, 0.6), 2))
    pd.DataFrame(rows, columns=["camera", "frame", "x1", "y1", "x2", "y2", "conf", "cls"]).to_csv(args.out, index=False)
    print("сохранено:", args.out, len(rows))


if __name__ == "__main__":
    main()
