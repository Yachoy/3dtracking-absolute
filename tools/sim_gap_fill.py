"""Офлайн-проверка GapFill без видеокарты.

На вход — replay прогона БЕЗ заполнения (`--no-fill-gaps`, или колонка primary из shadow-лога);
наблюдения в провалах — его `tracked_2d_boxes` (маски SAM2). Слабые детекции YOLO и отброшенные
маски трекера в replay не пишутся, поэтому здесь не моделируются.

Печатает покрытие, шаги > 10 м по точке ответа (как в trajectory.csv), скачки на возвратах
трекера, а с `--reference` — ошибку по эталону отдельно для кадров трекера и для заполненных.
Параметры модуля можно менять: `DECAY=0.99 TAKE_OVER=forecast …`

    python tools/sim_gap_fill.py --replay out/way1-base/replay.json --data data/test \
        --reference data/test/object_transform_SM_AA_track.csv
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.camera import Camera  # noqa: E402
from online.estimation import gap_fill as G  # noqa: E402
from online.geometry import DEFAULT_TARGET as M, View  # noqa: E402
from online.perception.vision import Instance  # noqa: E402
from online.run import sources  # noqa: E402
from online.tracker import Estimate  # noqa: E402


def box_instance(b):
    x1, y1, x2, y2 = b
    return Instance(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float), 0.5, 7)


def read_reference(path):
    gt = {}
    for row in Path(path).read_text(encoding="cp1251").splitlines()[1:]:
        p = row.split(";")
        try:
            gt[int(p[0])] = np.array([float(p[1]), float(p[2]), float(p[3])]) / 100
        except (ValueError, IndexError):
            pass
    return gt


def runs(frames):
    out = []
    for i in sorted(frames):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--compare", type=Path, help="другой replay для сравнения ошибки на тех же кадрах")
    ap.add_argument("params", nargs="*", help="NAME=VALUE для констант online/estimation/gap_fill.py")
    args = ap.parse_args()
    for kv in args.params:
        name, value = kv.split("=", 1)
        old = getattr(G, name)
        if isinstance(old, tuple):
            kind = type(old[0]) if old else str
            setattr(G, name, tuple(kind(v) for v in value.split(",") if v))
        else:
            setattr(G, name, type(old)(value))

    replay = json.loads(args.replay.read_text(encoding="utf-8"))["frames"]
    tracks = sources(args.data)
    width, height = 1920, 1080
    tracker = SimpleNamespace(current_views={}, groups={}, segmenter=None)
    fill = G.GapFill(tracker, M, 30.0)
    course, reliable = 0.0, []
    out, status = {}, {}
    for f in replay:
        i = f["frame"]
        tracker.current_views = {k: View.of(Camera.from_track(t, i, width, height), model=M)
                                 for k, t in tracks.items() if i in t.poses}
        pose = None
        if f["xyz"][0] is not None:
            here = np.array(f["xyz"]) / 100
            if f.get("yaw_pitch_roll_deg"):
                yaw, pitch, _ = np.deg2rad(f["yaw_pitch_roll_deg"])
            else:  # для наземной машины курс в replay не пишется — по надёжным кадрам траектории
                if f["status"] in ("multi", "mono"):
                    reliable.append((i, here))
                past = [p for j, p in reliable if i - 45 <= j <= i - 15]
                if past and np.linalg.norm(here[:2] - past[0][:2]) > 5:
                    course = np.arctan2(*(here[:2] - past[0][:2])[::-1])
                yaw, pitch = course, 0.0
            pose = np.r_[here, yaw, pitch]
        tracked = {int(k): box_instance(b) for k, b in (f.get("tracked_2d_boxes") or {}).items()}
        est = fill.step(i, Estimate(pose, f["status"] if pose is not None else "lost", {}, 0.0, tracked), {})
        if est.pose is not None:
            out[i], status[i] = np.asarray(est.pose[:3], float), est.status

    n = len(replay)
    base = {f["frame"]: np.array(f["xyz"]) / 100 for f in replay if f["xyz"][0] is not None}
    filled = {i for i in out if status[i] in G.FILLED}
    print(f"покрытие: было {len(base)}/{n}, стало {len(out)}/{n}; заполнено кадров {len(filled)}")
    jumps = [(i, status.get(i - 1), round(float(np.linalg.norm(out[i] - out[i - 1])), 1))
             for i in sorted(out) if i - 1 in out and np.linalg.norm(out[i] - out[i - 1]) > 10]
    base_jumps = [i for i in sorted(base) if i - 1 in base and np.linalg.norm(base[i] - base[i - 1]) > 10]
    print(f"шагов > 10 м: база {len(base_jumps)}, с заполнением {len(jumps)}: {jumps}")
    if args.reference:
        gt = read_reference(args.reference)
        err = lambda xyz, i: float(np.linalg.norm(xyz - gt[i]))
        other = None
        if args.compare:
            other = {f["frame"]: np.array(f["xyz"]) / 100 for f in
                     json.loads(args.compare.read_text(encoding="utf-8"))["frames"] if f["xyz"][0] is not None}
        for name, frames in (("кадры трекера", [i for i in out if i not in filled and i in gt]),
                             ("заполненные", [i for i in filled if i in gt]),
                             ("все", [i for i in out if i in gt])):
            if not frames:
                continue
            e = np.array([err(out[i], i) for i in frames])
            line = f"{name}: {len(frames)} кадров, MAE {e.mean():.2f} м, медиана {np.median(e):.2f}, макс {e.max():.1f}"
            if other is not None:
                o = np.array([err(other[i], i) for i in frames if i in other])
                line += f" | {args.compare.name} на тех же: MAE {o.mean():.2f}"
            print(line)
        for a, b in runs(filled):
            e = [err(out[i], i) for i in range(a, b + 1) if i in gt]
            print(f"  заполнено {a}–{b}: MAE {np.mean(e):.2f} м, макс {np.max(e):.1f}, "
                  f"на выходе {e[-1]:.1f} м")


if __name__ == "__main__":
    main()
