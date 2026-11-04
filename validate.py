"""Валидатор: собирает видео по сохранённому прогону, не запуская детектор.

Читает три файла из каталога прогона (`--run`):
    views.csv  — окно поиска и детекция YOLO на каждый кадр и камеру
    track.csv  — итоговая 3D-траектория (пивот, см)
    debug.csv  — режим измерения, число камер, курс (необязателен: без него
                 режим выводится из views.csv)

и рисует то же, что и `yolo3d.py` во время прогона: 5 плиток камер (оранжевое
окно поиска, синий бокс YOLO, синяя рамка кадра если объект найден), вид
сверху с GT и оценкой, панель с графиком ошибки и статусом.

Заодно считает метрики по GT и пишет `metrics.json` в каталог прогона.

Примеры:
    python validate.py --run out/yolo-like
    python validate.py --run out/yolo-like --video my.mp4 --lo 200 --hi 400
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.camera import Camera  # noqa: E402
from core.dataset import load_scene  # noqa: E402

W, H = 1920, 1080
CAMS = (1, 2, 3, 4, 5)

GT_COLOR = (60, 220, 60)
WIN_COLOR = (0, 165, 255)      # оранжевый — окно поиска
DET_COLOR = (255, 120, 30)     # синий — детекция YOLO (BGR)
BORDER_COLOR = (255, 90, 0)    # синяя рамка «камера видит фургон»
TXT = (235, 235, 235)
DIM = (170, 170, 170)


# --------------------------------------------------------------------------- #
# чтение сохранённого прогона
# --------------------------------------------------------------------------- #
def read_views(path: Path) -> dict[int, dict[int, dict]]:
    """frame -> cam -> {"window": box|None, "det": (conf, box)|None}."""
    out: dict[int, dict[int, dict]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            f = int(r["frame"])
            idx = int(r["cam"])
            win = None
            if r["win_x0"]:
                win = np.array([float(r["win_x0"]), float(r["win_y0"]),
                                float(r["win_x1"]), float(r["win_y1"])])
            det = None
            if r["det_x0"] and r["conf"]:
                box = np.array([float(r["det_x0"]), float(r["det_y0"]),
                                float(r["det_x1"]), float(r["det_y1"])])
                det = (float(r["conf"]), box)
            out.setdefault(f, {})[idx] = {"window": win, "det": det}
    return out


def read_track(path: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            out[int(r["Frame"])] = np.array([float(r["Loc_X"]), float(r["Loc_Y"]),
                                             float(r["Loc_Z"])])
    return out


def read_debug(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[int(r["frame"])] = r
    return out


# --------------------------------------------------------------------------- #
# рисование
# --------------------------------------------------------------------------- #
def fit_tile(img: np.ndarray, tw: int, th: int):
    """Вписывает кадр в плитку с сохранением пропорций (поля по краям)."""
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    tile = np.full((th, tw, 3), 14, np.uint8)
    dx, dy = (tw - nw) // 2, (th - nh) // 2
    tile[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return tile, s, dx, dy


def draw_frame(f, frames, views, track, errors, scene, gt, dbg):
    tw, th = 640, 360
    canvas = np.full((H, W, 3), 14, np.uint8)

    for slot, idx in enumerate(CAMS):
        col, row = slot % 3, slot // 3
        ox, oy = col * tw, row * th
        img = frames.get(idx)
        if img is None:
            cv2.putText(canvas, f"CAM{idx}: no frame", (ox + 10, oy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
            continue
        tile, scale, dx, dy = fit_tile(img, tw, th)
        off = np.array([dx, dy, dx, dy])
        view = views.get(f, {}).get(idx, {})
        win, det = view.get("window"), view.get("det")
        if win is not None:
            b = win * scale + off
            cv2.rectangle(tile, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), WIN_COLOR, 2)
        if det is not None:
            b = det[1] * scale + off
            cv2.rectangle(tile, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), DET_COLOR, 2)
        head = f"CAM{idx}" + (f"  det {det[0]:.2f}" if det is not None else "")
        cv2.putText(tile, head, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TXT, 1, cv2.LINE_AA)
        if det is not None:
            cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), BORDER_COLOR, 4)
        canvas[oy:oy + th, ox:ox + tw] = tile

    # ---------------- вид сверху ------------------------------------------- #
    ox, oy = 2 * tw, th
    top = np.full((th, tw, 3), 12, np.uint8)
    pts = np.array([track[q][:2] for q in sorted(track) if q <= f]) / 100.0
    if len(pts) > 1:
        lo = pts.min(axis=0) - 60
        hi = pts.max(axis=0) + 60
        span = np.maximum(hi - lo, 1.0)
        sc = min((tw - 40) / span[0], (th - 40) / span[1])

        def m2p(p):
            return (int(20 + (p[0] - lo[0]) * sc), int(th - 20 - (p[1] - lo[1]) * sc))

        gt_pts = np.array([scene.target[q].loc[:2] for q in sorted(track) if q <= f]) / 100.0
        for i in range(1, len(gt_pts)):
            cv2.line(top, m2p(gt_pts[i - 1]), m2p(gt_pts[i]), GT_COLOR, 1)
        for i in range(1, len(pts)):
            cv2.line(top, m2p(pts[i - 1]), m2p(pts[i]), (60, 60, 255), 1)
        cv2.circle(top, m2p(gt_pts[-1]), 4, GT_COLOR, -1)
        cv2.drawMarker(top, m2p(pts[-1]), (60, 60, 255), cv2.MARKER_CROSS, 10, 2)
        for idx in CAMS:
            cam = Camera.from_track(scene.cameras[idx], f, W, H)
            c = m2p(cam.loc[:2] / 100.0)
            cv2.rectangle(top, (c[0] - 3, c[1] - 3), (c[0] + 3, c[1] + 3), (150, 150, 150), -1)
            cv2.putText(top, f"C{idx}", (c[0] + 5, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, DIM, 1, cv2.LINE_AA)
    cv2.putText(top, "3D: GT green, estimate red", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, TXT, 1, cv2.LINE_AA)
    canvas[oy:oy + th, ox:ox + tw] = top

    # ---------------- нижняя панель ---------------------------------------- #
    by = 2 * th
    panel = np.full((H - by, W, 3), 10, np.uint8)
    gw = 1100
    errs = np.array([e for _, e in errors if np.isfinite(e)])
    if len(errs) > 1:
        ymax = max(2.0, float(np.percentile(errs, 95)) * 1.15)
        gh = by - 40
        for i in range(1, len(errs)):
            x0 = int(20 + (i - 1) * (gw - 20) / max(1, len(errs) - 1))
            x1 = int(20 + i * (gw - 20) / max(1, len(errs) - 1))
            y0 = int(20 + gh - min(errs[i - 1], ymax) / ymax * gh)
            y1 = int(20 + gh - min(errs[i], ymax) / ymax * gh)
            cv2.line(panel, (x0, y0), (x1, y1), (90, 200, 255), 1)
        med = float(np.median(errs))
        ym = int(20 + gh - min(med, ymax) / ymax * gh)
        cv2.line(panel, (20, ym), (gw, ym), GT_COLOR, 1)
        cv2.putText(panel, f"median {med:.2f} m", (gw - 190, ym - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(panel, f"error vs GT (m), scale {ymax:.1f}", (22, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TXT, 1, cv2.LINE_AA)

    row = dbg.get(f, {})
    pred = track.get(f)
    n_det = sum(1 for v in views.get(f, {}).values() if v["det"] is not None)
    mode = row.get("mode") or ("tri" if n_det >= 2 else "single" if n_det == 1 else "coast")
    lines = [f"frame {f}   t={f / 30.0:.2f}s",
             f"mode: {mode}   cams with det: {n_det}",
             f"est (m): {pred[0] / 100:9.1f} {pred[1] / 100:9.1f} {pred[2] / 100:7.1f}" if pred is not None else "est: n/a",
             f"GT  (m): {gt[0] / 100:9.1f} {gt[1] / 100:9.1f} {gt[2] / 100:7.1f}",
             f"error: {np.linalg.norm(pred - gt) / 100:.2f} m" if pred is not None else "",
             f"heading: {float(row.get('heading') or 0.0):+.1f} deg",
             f"coast: {row.get('coast', 0)}"]
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (gw + 30, 40 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, TXT, 1, cv2.LINE_AA)
    cv2.putText(panel, "orange = search window, blue = YOLO detection, frame border = camera sees the truck",
                (22, by - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)
    canvas[by:H, 0:W] = panel
    return canvas


# --------------------------------------------------------------------------- #
# метрики
# --------------------------------------------------------------------------- #
def summarise(track: dict[int, np.ndarray], scene) -> dict:
    errs, frames = [], []
    for f in sorted(track):
        if f in scene.target:
            errs.append(float(np.linalg.norm(track[f] - scene.target[f].loc)) / 100.0)
            frames.append(f)
    e = np.array(errs)
    blocks = []
    for lo in range(0, 1001, 100):
        m = [(f, x) for f, x in zip(frames, errs) if lo <= f < lo + 100]
        if m:
            blocks.append({"range": f"{lo}-{lo + 100}", "n": len(m),
                           "median": round(float(np.median([x for _, x in m])), 2)})
    return {
        "frames": len(e),
        "coverage": round(len(e) / max(1, len(scene.target)), 4),
        "median_m": round(float(np.median(e)), 3),
        "mean_m": round(float(e.mean()), 3),
        "rmse_m": round(float(np.sqrt((e ** 2).mean())), 3),
        "p90_m": round(float(np.percentile(e, 90)), 3),
        "max_m": round(float(e.max()), 3),
        "share_lt_2m": round(float((e < 2).mean()), 3),
        "share_lt_5m": round(float((e < 5).mean()), 3),
        "blocks": blocks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="видео по сохранённому прогону")
    ap.add_argument("--run", default=str(ROOT / "out" / "yolo-like"),
                    help="каталог прогона с views.csv и track.csv")
    ap.add_argument("--data", default=str(ROOT / "data"), help="каталог с видео и CSV")
    ap.add_argument("--video", default="", help="выходное видео (по умолчанию <run>/validation.mp4)")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=1001)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    views = read_views(run / "views.csv")
    track = read_track(run / "track.csv")
    dbg = read_debug(run / "debug.csv")
    scene = load_scene(Path(args.data))

    print(f"кадров в track: {len(track)}, в views: {len(views)}, данных: {len(scene.target)}")
    summary = summarise(track, scene)
    print(f"медиана {summary['median_m']:.2f} м | MAE {summary['mean_m']:.2f} | "
          f"RMSE {summary['rmse_m']:.2f} | p90 {summary['p90_m']:.2f} | max {summary['max_m']:.2f}")
    (run / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.no_video:
        return 0

    out = Path(args.video) if args.video else run / "validation.mp4"
    frames_total = [f for f in sorted(track) if args.lo <= f <= args.hi]
    if not frames_total:
        print("нет кадров в диапазоне")
        return 1

    caps, pos = {}, {}
    for i in CAMS:
        cap = cv2.VideoCapture(str(scene.video_path(i)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames_total[0])
        caps[i], pos[i] = cap, frames_total[0]

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    errors: list[tuple[int, float]] = []
    for f in tqdm(frames_total, desc="render", unit="fr"):
        frames = {}
        for i, cap in caps.items():
            while pos[i] < f:                     # кадры в CSV могут идти с пропусками
                cap.grab()
                pos[i] += 1
            ok, fr = cap.read()
            pos[i] += 1
            frames[i] = fr if ok else None
        gt = scene.target[f].loc
        pred = track[f]
        errors.append((f, float(np.linalg.norm(pred - gt)) / 100.0))
        writer.write(draw_frame(f, frames, views, track, errors, scene, gt, dbg))

    writer.release()
    for cap in caps.values():
        cap.release()
    print(f"видео: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
