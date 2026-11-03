"""Качество детектора по камерам относительно GT.

Для каждой камеры: сколько кадров объект в поле зрения, сколько из них нашёл детектор
(проекция GT внутри рамки), средний сдвиг центра рамки от проекции, ложные рамки.

    python -m tools.diagnose data/test cache/detections.csv
"""
import argparse

import numpy as np
import pandas as pd

from tracker.camera import load_cameras, load_ground_truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("det_csv")
    ap.add_argument("--pad-m", type=float, default=2.0, help="на сколько метров расширять рамку при проверке попадания GT")
    args = ap.parse_args()

    cams = load_cameras(args.data_dir)
    gt = load_ground_truth(args.data_dir).interpolate()
    dets = pd.read_csv(args.det_csv)
    rows = []
    for cam in cams:
        d_cam = dets[dets.camera == cam.name].groupby("frame")
        in_view = hit = false = 0
        offsets, sizes = [], []
        for frame in gt.index:
            uv, z = cam.pose(frame).project(gt.loc[frame].values)
            u, v = uv[0]
            visible = z[0] > 0 and 0 <= u < cam.width and 0 <= v < cam.height
            in_view += visible
            boxes = d_cam.get_group(frame) if frame in d_cam.groups else None
            if boxes is None:
                continue
            centers = np.c_[(boxes.x1 + boxes.x2) / 2, (boxes.y1 + boxes.y2) / 2]
            dist = np.linalg.norm(centers - uv[0], axis=1)
            # точка GT — перед кабины, она лежит на краю рамки: проверяем попадание в расширенную рамку
            pad = cam.fx * args.pad_m / max(z[0], 1.0)
            near = ((boxes.x1 - pad <= u) & (u <= boxes.x2 + pad) & (boxes.y1 - pad <= v) & (v <= boxes.y2 + pad)).values
            if visible and near.any():
                hit += 1
                j = int(np.argmin(dist))
                offsets.append(centers[j] - uv[0])
                sizes.append(boxes.x2.values[j] - boxes.x1.values[j])
            false += int((~near).sum())
        off = np.median(offsets, axis=0) if offsets else (np.nan, np.nan)
        rows.append({
            "камера": cam.name, "фокус": cam.focal_mm, "в кадре": in_view, "найдено": hit,
            "recall %": round(100 * hit / max(in_view, 1), 1),
            "сдвиг u,px": round(off[0], 1), "сдвиг v,px": round(off[1], 1),
            "ширина рамки, px (медиана)": round(np.median(sizes), 1) if sizes else np.nan,
            "чужих рамок": false,
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
