"""Метрики прогона без эталона (ground truth): правдоподобие траектории и выборочные засечки.

Когда эталона нет (Way2), точность всё равно можно оценить тремя независимыми способами:

1. Режимы трекера. Доля кадров, где фуру видят 2+ камеры (`multi`), и самая длинная серия без
   свежего наблюдения (`flow`/`forecast`/`motion`): если трекер сотни кадров едет только по
   оптическому потоку, он, скорее всего, ведёт ложный захват.
2. Физика. Фура не ездит со скоростью 600 м/с и не уходит на 40 м под землю. Считаем скорость,
   ускорение, скачки между кадрами и разброс высоты относительно медианы.
3. Засечки. На нескольких кадрах вручную (или другим детектором) отмечаем центр машины в двух
   камерах; лучи из калиброванных камер пересекаются в 3D-точке — это независимая оценка
   положения. Зазор между лучами показывает точность самой засечки. Формат файла засечек:
       {"checks": [{"frame": 100, "points": {"1": [945, 735], "2": [930, 540]}}]}
   Пиксели — в исходном разрешении видео камеры.

Запуск:
    python tools/no_gt_metrics.py --out out/way2 --data data/way2 --checks data/way2_checks.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.camera import Camera  # noqa: E402
from online.run import sources  # noqa: E402

FRESH = {"multi", "mono"}


def longest_run(flags):
    best = cur = start = best_start = 0
    for i, flag in enumerate(flags):
        if flag:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    return best, best_start


def triangulate(rays):
    """Точка, ближайшая ко всем лучам (МНК), и наибольшее расстояние от неё до луча."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in rays:
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o
    p = np.linalg.solve(A, b)
    gap = max(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (p - o)) for o, d in rays)
    return p, gap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="каталог прогона с replay.json")
    ap.add_argument("--data", type=Path, help="каталог данных (нужен для засечек)")
    ap.add_argument("--checks", type=Path, help="JSON с засечками по двум и более камерам")
    ap.add_argument("--max-speed", type=float, default=40.0, help="порог скорости, м/с (40 м/с = 144 км/ч)")
    ap.add_argument("--height-band", type=float, default=5.0, help="допуск высоты от медианы, м")
    ap.add_argument("--window", type=float, default=0.5, help="окно для скорости и ускорения, с")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    replay = json.loads((args.out / "replay.json").read_text(encoding="utf-8"))
    frames = replay["frames"]
    fps = float(replay.get("metadata", {}).get("fps", 30.0))
    n = len(frames)
    xyz = np.array([np.array(f["xyz"], float) / 100 if f.get("xyz") and f["xyz"][0] is not None
                    else np.full(3, np.nan) for f in frames])
    ok = np.isfinite(xyz).all(axis=1)
    status = [f["status"] for f in frames]
    counts = {s: status.count(s) for s in sorted(set(status))}
    stale, stale_start = longest_run([s not in FRESH for s in status])

    idx = np.flatnonzero(ok)
    # скорость по окну 0,5 с: покадровое дрожание ±0,5 м само по себе даёт «20 м/с» на 30 кадр/с
    w = max(1, int(round(args.window * fps)))
    a, b = idx[:-w], idx[w:]
    speed = np.linalg.norm(xyz[b] - xyz[a], axis=1) / ((b - a) / fps)
    accel = np.abs(speed[w:] - speed[:-w]) / (w / fps) if len(speed) > w else np.zeros(0)
    jumps = np.linalg.norm(np.diff(xyz[idx], axis=0), axis=1)
    z = xyz[idx, 2]
    z_med = float(np.median(z)) if len(z) else float("nan")

    report = {
        "frames": n,
        "coverage": round(float(ok.mean()), 4),
        "status": counts,
        "multi_share": round(counts.get("multi", 0) / n, 4),
        "longest_without_detection": {"frames": int(stale), "start": int(stale_start)},
        "speed_mps": {"median": round(float(np.median(speed)), 2), "p95": round(float(np.percentile(speed, 95)), 2),
                      "max": round(float(speed.max()), 2)} if len(speed) else None,
        "frames_over_max_speed": int((speed > args.max_speed).sum()),
        "accel_mps2_p95": round(float(np.percentile(accel, 95)), 2) if len(accel) else None,
        "jumps_over_5m": int((jumps > 5.0).sum()),
        "max_jump_m": round(float(jumps.max()), 2) if len(jumps) else None,
        "height_m": {"median": round(z_med, 2), "p5": round(float(np.percentile(z, 5)), 2),
                     "p95": round(float(np.percentile(z, 95)), 2)} if len(z) else None,
        "frames_off_height_band": int((np.abs(z - z_med) > args.height_band).sum()),
    }

    if args.checks:
        if args.data is None:
            raise SystemExit("--checks требует --data (калибровка камер)")
        tracks = sources(args.data)
        out = []
        for check in json.loads(args.checks.read_text(encoding="utf-8"))["checks"]:
            frame = int(check["frame"])
            rays = []
            for key, uv in check["points"].items():
                camera = Camera.from_track(tracks[int(key)], frame, args.width, args.height)
                o, d = camera.ray(np.asarray(uv, float))
                rays.append((o / 100, d))
            point, gap = triangulate(rays)
            est = xyz[frame] if frame < n else np.full(3, np.nan)
            out.append({"frame": frame, "cameras": sorted(map(int, check["points"])),
                        "triangulated_m": [round(float(v), 2) for v in point], "ray_gap_m": round(float(gap), 2),
                        "estimate_m": None if np.isnan(est).any() else [round(float(v), 2) for v in est],
                        "difference_m": None if np.isnan(est).any() else round(float(np.linalg.norm(est - point)), 2),
                        "status": status[frame] if frame < n else None})
        diffs = [c["difference_m"] for c in out if c["difference_m"] is not None]
        report["checks"] = out
        if diffs:
            report["checks_mae_m"] = round(float(np.mean(diffs)), 2)
            report["checks_max_m"] = round(float(np.max(diffs)), 2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
