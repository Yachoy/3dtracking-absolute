"""E01 — реальный причинный конвейер без GT: вычитание фона -> многокамерный EKF.

1. Детекция движения в каждой камере (`bgsub_ekf/detect.py`); для поворотной
   CAM5 фон переносится гомографией чистого поворота из известной ориентации.
2. 3D-трекер (`bgsub_ekf/fuse.py`): EKF [x,y,z,vx,vy,vz], измерение — центр
   рамки блоба в каждой камере, поэтому работает при 1..N видящих камерах;
   совместная ассоциация блобов по камерам, несколько гипотез, перепривязка
   при дрейфе, у обрезанных краем кадра рамок — только надёжная координата.
3. Точка привязки GT: центр + (L + tau*v) по курсу (L~половина длины, tau —
   запаздывание оценки).

Строго причинно: кадр t использует только кадры <= t.
"""

from __future__ import annotations

import queue
import threading
import time

import cv2
import numpy as np

from core.dataset import Scene
from core.validation import Prediction
from experiments.base import Experiment
from experiments.bgsub_ekf.cams import TrackCamera
from experiments.bgsub_ekf.detect import Detector
from experiments.bgsub_ekf.fuse import Tracker


class BgSubEKF(Experiment):
    name = "e01_bgsub_ekf"

    def __init__(self, offset_cm: float = 800.0, lag_s: float = 0.4, video_offset: int = 0, **_) -> None:
        self.offset_cm = float(offset_cm)
        self.lag_s = float(lag_s)
        self.video_offset = int(video_offset)

    def run(self, scene: Scene) -> Prediction:
        idx = sorted(i for i in scene.cameras if scene.cameras[i].video is not None)
        cams = [TrackCamera(scene.cameras[i], scene.width, scene.height) for i in idx]
        caps = [cv2.VideoCapture(str(scene.cameras[i].video)) for i in idx]
        dets = [Detector(c) for c in cams]
        tracker = Tracker(cams, offset=(self.offset_cm, 0.0), lag=self.lag_s)

        def worker(k, qin, qout):  # один поток на камеру (OpenCV отпускает GIL)
            while True:
                f = qin.get()
                if f is None:
                    return
                ok, img = caps[k].read()
                qout.put((k, dets[k].step(img, f) if ok else None))

        qins = [queue.Queue() for _ in cams]
        qout: queue.Queue = queue.Queue()
        for k in range(len(cams)):
            threading.Thread(target=worker, args=(k, qins[k], qout), daemon=True).start()

        xyz: dict[int, np.ndarray] = {}
        n_cams: dict[int, int] = {}
        t0 = time.time()
        f = 0
        while True:
            for q in qins:
                q.put(f)
            res = dict(qout.get() for _ in cams)
            if any(res[k] is None for k in range(len(cams))):
                break
            out, _, used = tracker.step(f, [res[k] for k in range(len(cams))])
            if out is not None:
                xyz[f + self.video_offset] = out
                n_cams[f + self.video_offset] = int(used)
            if f % 100 == 0:
                print(f"  кадр {f}: камер {used}, {f / (time.time() - t0 + 1e-9):.1f} fps", flush=True)
            f += 1
        for q in qins:
            q.put(None)

        return Prediction(
            name=self.name,
            xyz=xyz,
            n_cams=n_cams,
            meta={
                "offset_cm": self.offset_cm,
                "lag_s": self.lag_s,
                "frames": len(xyz),
                "fps": round(f / (time.time() - t0), 2),
                "mean_cameras_used": float(np.mean(list(n_cams.values()))) if n_cams else 0.0,
            },
        )
