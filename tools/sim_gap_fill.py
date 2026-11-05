"""Офлайн-проверка GapFill без видеокарты: replay прогона с --no-fill-gaps, наблюдения — его tracked_2d_boxes.

Считает скачок на возврате трекера, наибольший шаг внутри провала и все шаги > 10 м.
Параметры модуля можно менять: DECAY=0.99 PULL_SINGLE=0.35 …
    python tools/sim_gap_fill.py . out/way2-base/replay.json data/way2
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
from core.camera import Camera  # noqa: E402
from online.estimation import gap_fill as G  # noqa: E402
from online.geometry import DEFAULT_TARGET as M, View  # noqa: E402
from online.perception.vision import Instance  # noqa: E402
from online.run import sources  # noqa: E402
from online.tracker import Estimate  # noqa: E402

replay = json.loads(Path(sys.argv[2]).read_text())["frames"]
tracks = sources(Path(sys.argv[3]))
for kv in sys.argv[4:]:
    k, v = kv.split("=")
    setattr(G, k, type(getattr(G, k))(v))


def views(frame):
    return {k: View.of(Camera.from_track(t, frame, 1920, 1080), model=M) for k, t in tracks.items() if frame in t.poses}


def box_instance(b):
    x1, y1, x2, y2 = b
    return Instance(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float), 0.5, 7)


tracker = SimpleNamespace(current_views={}, groups={}, segmenter=None)
course = 0.0
reliable = []
fill = G.GapFill(tracker, M, 30.0)
out, status = {}, {}
for f in replay:
    i = f["frame"]
    tracker.current_views = views(i)
    pose = None
    if f["xyz"][0] is not None:
        if f.get("yaw_pitch_roll_deg"):
            yaw, pitch, _ = np.deg2rad(f["yaw_pitch_roll_deg"])
        else:  # для наземной машины курс в replay не пишется — берём его по надёжным кадрам траектории
            here = np.array(f["xyz"]) / 100
            if f["status"] in ("multi", "mono"):
                reliable.append((i, here))
            past = [p for j, p in reliable if i - 45 <= j <= i - 15]
            if past and np.linalg.norm(here[:2] - past[0][:2]) > 5:
                course = np.arctan2(*(here[:2] - past[0][:2])[::-1])
            yaw, pitch = course, 0.0
        pose = np.r_[np.array(f["xyz"]) / 100, yaw, pitch]
    tracked = {int(k): box_instance(b) for k, b in (f.get("tracked_2d_boxes") or {}).items()}
    est = fill.step(i, Estimate(pose, f["status"] if pose is not None else "lost", {}, 0.0, tracked), {})
    if est.pose is not None:
        out[i], status[i] = np.asarray(est.pose[:3], float), est.status  # точка ответа, как в trajectory.csv
for a, b in ((815, 1041), (1418, 1499), (1907, 1932)):
    steps = [np.linalg.norm(out[i] - out[i - 1]) for i in range(a, min(b + 2, len(replay))) if i in out and i - 1 in out]
    ret = np.linalg.norm(out[b + 1] - out[b]) if b + 1 in out and b in out else None
    weak = sum(status.get(i) == "weak" for i in range(a, b + 1))
    print(f"провал {a}-{b}: скачок на возврате {ret if ret is None else round(ret, 1)} м, "
          f"макс шаг {max(steps):.2f} м, weak {weak}/{b - a + 1}")
allsteps = [np.linalg.norm(out[i] - out[i - 1]) for i in out if i - 1 in out]
print("шагов > 10 м всего:", sum(s > 10 for s in allsteps))
base = {f["frame"]: np.array(f["xyz"]) / 100 for f in replay if f["xyz"][0] is not None}
bc = base
print("база: шагов > 10 м:", [i for i in bc if i - 1 in bc and np.linalg.norm(bc[i] - bc[i - 1]) > 10])
print("с заполнением:", [(i, status.get(i - 1), round(float(np.linalg.norm(out[i] - out[i - 1])), 1)) for i in out
                         if i - 1 in out and np.linalg.norm(out[i] - out[i - 1]) > 10])
