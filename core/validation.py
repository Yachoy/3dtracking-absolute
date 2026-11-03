"""Общий механизм валидации: композит «3D слева / камеры справа» + видео.

Раскладка кадра (1920×1080):
    ┌───────────────────────────┬──────────────┬──────────────┐
    │                           │  CAM1        │  CAM2        │
    │   3D-вид: GT и pred       ├──────────────┼──────────────┤
    │   + траектории            │  CAM3        │  CAM4        │
    ├───────────────────────────┼──────────────┼──────────────┤
    │   график ошибки по кадрам │  CAM5        │  сводка      │
    └───────────────────────────┴──────────────┴──────────────┘

На камерах: зелёный — GT 2D (проекция GT 3D), красный — предсказание.
Эксперимент отдаёт только 3D-траекторию, всё остальное рисует валидатор —
поэтому любой новый подход получает одинаковую визуализацию и метрики.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .dataset import Scene
from .metrics import ErrorTable, build_errors
from .object_model import ObjectModel, heading_from_positions, pitch_from_positions
from .scene3d import BG, GREEN, RED, Scene3D, WHITE

CANVAS_W, CANVAS_H = 1920, 1080
LEFT_W = 720
PAD = 12
TILE_W = (CANVAS_W - LEFT_W - 3 * PAD) // 2
TILE_H = int(round(TILE_W * 9 / 16))
COL_X = [LEFT_W + PAD, LEFT_W + 2 * PAD + TILE_W]
ROW_Y = [40, 40 + TILE_H + 8, 40 + 2 * (TILE_H + 8)]

GT_COLOR = (60, 220, 60)
PRED_COLOR = (60, 60, 255)
TEXT = (225, 225, 225)
MUTED = (150, 150, 150)


@dataclass
class Prediction:
    """Результат эксперимента: 3D-траектория и служебные данные."""

    name: str
    xyz: dict[int, np.ndarray]
    n_cams: dict[int, int] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _json_default(obj):
    """numpy-типы в JSON: массивы и скаляры из метаданных экспериментов."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_default(v) for k, v in obj.items()}
    return str(obj)


def _text(img, text, org, scale=0.5, color=TEXT, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _tile_caption(img, text, color=TEXT):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (18, 18, 20), -1)
    _text(img, text, (8, 18), 0.5, color, 1)


def _draw_boxes(img, box, color, label=None, scale=(1.0, 1.0), label_at="top"):
    if box is None:
        return
    box = np.asarray(box, dtype=np.float64) * np.array([scale[0], scale[1], scale[0], scale[1]])
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    if label:
        ty = max(36, y0 - 5) if label_at == "top" else min(img.shape[0] - 5, y1 + 15)
        _text(img, label, (x0 + 3, ty), 0.45, color, 1)


def _draw_rect(img, rect, color, label=None, scale=(1.0, 1.0)):
    """Повернутый прямоугольник (cx, cy, w, h, angle) из детекции."""
    if rect is None:
        return
    cx, cy, w, h, angle = rect
    cx, cy = cx * scale[0], cy * scale[1]
    w, h = w * scale[0], h * scale[1]
    pts = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.int32)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    if label:
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        _text(img, label, (int(x0) + 3, max(36, int(y0) - 5)), 0.45, color, 1)


def _draw_error_chart(w, h, frames, errors, current, summary) -> np.ndarray:
    img = np.full((h, w, 3), BG, dtype=np.uint8)
    _text(img, "3D ERROR vs GT (m)", (12, 20), 0.5, TEXT)
    plot = (50, 40, w - 20, h - 34)
    x0, y0, x1, y1 = plot
    cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 64), 1)

    vals = np.array([e if np.isfinite(e) else np.nan for e in errors])
    finite = vals[np.isfinite(vals)]
    if finite.size:
        vmax = max(0.5, float(np.nanpercentile(finite, 99)) * 1.15)
    else:
        vmax = 1.0
    _text(img, f"{vmax:.1f}", (8, y0 + 5), 0.38, MUTED)
    _text(img, "0", (8, y1 + 3), 0.38, MUTED)

    n = len(frames)
    if n > 1:
        fmin, fmax = frames[0], frames[-1]
        span = max(fmax - fmin, 1)
        pts = []
        for f, v in zip(frames, vals):
            if not np.isfinite(v):
                pts.append(None)
                continue
            x = int(x0 + (f - fmin) / span * (x1 - x0))
            y = int(y1 - min(v / vmax, 1.0) * (y1 - y0))
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            if a and b:
                cv2.line(img, a, b, (90, 160, 240), 1, cv2.LINE_AA)
        idx = min(range(n), key=lambda i: abs(frames[i] - current))
        if pts[idx]:
            cv2.circle(img, pts[idx], 4, (240, 200, 90), -1, cv2.LINE_AA)

    s = summary
    _text(img, f"RMSE {s.get('rmse_m', float('nan')):.2f}   med {s.get('median_m', float('nan')):.2f}   "
              f"p95 {s.get('p95_m', float('nan')):.2f}   max {s.get('max_m', float('nan')):.2f}",
          (x0, h - 8), 0.45, TEXT)
    return img


class Validator:
    """Собирает общий отчёт по любому эксперименту."""

    def __init__(
        self,
        scene: Scene,
        experiment_name: str = "experiment",
        object_model: ObjectModel | None = None,
        video_offset: int = 0,
        out_dir: str | Path = "out",
        max_frames: int | None = None,
        use_pitch: bool = False,
    ) -> None:
        self.scene = scene
        self.experiment_name = experiment_name
        self.model = object_model or ObjectModel()
        self.video_offset = video_offset
        self.out_dir = Path(out_dir) / experiment_name
        self.max_frames = max_frames
        # наклон бокса по уклону траектории: GT-Z шумный, по умолчанию выключен
        self.use_pitch = use_pitch

    # ------------------------------------------------------------------ run
    def run(self, pred: Prediction, render_video: bool = True) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        gt = {f: p.loc for f, p in self.scene.target.items()}
        frames = [f for f in sorted(gt) if pred.xyz.get(f) is not None]
        if self.max_frames:
            frames = frames[: self.max_frames]

        table = build_errors({f: gt[f] for f in frames}, pred.xyz, pred.n_cams)
        summary = table.summary()
        summary["experiment"] = pred.name
        summary["meta"] = pred.meta

        if render_video:
            self._render_video(frames, gt, pred, table, summary)

        self._write_reports(table, summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, ensure_ascii=False, indent=2))
        return summary

    # --------------------------------------------------------------- отчёты
    def _write_reports(self, table: ErrorTable, summary: dict) -> None:
        (self.out_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
        )
        rows = table.as_rows()
        if rows:
            keys = list(rows[0])
            with open(self.out_dir / "errors.csv", "w", encoding="utf-8") as fh:
                fh.write(",".join(keys) + "\n")
                for row in rows:
                    fh.write(",".join(f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k]) for k in keys) + "\n")

    # ---------------------------------------------------------------- видео
    def _render_video(self, frames, gt, pred, table, summary) -> None:
        scene = self.scene
        cams: dict[int, Camera] = {
            i: Camera.from_track(t, frames[0], scene.width, scene.height) for i, t in scene.cameras.items()
        }
        scene3d = Scene3D(gt, pred.xyz, list(cams.values()))
        chart_h = CANVAS_H - 40 - scene3d.h - PAD
        chart_frames = table.frames
        chart_errors = table.errors

        caps = {}
        next_idx: dict[int, int] = {}
        for i, track in scene.cameras.items():
            if track.video is not None:
                cap = cv2.VideoCapture(str(track.video))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0] + self.video_offset)
                caps[i] = cap
                next_idx[i] = frames[0] + self.video_offset
        last_img: dict[int, np.ndarray] = {}

        def grab(cam_idx: int, target: int) -> np.ndarray | None:
            """Кадр видео с нужным индексом (пропуски в GT не сбивают поток)."""
            cap = caps.get(cam_idx)
            if cap is None or not cap.isOpened():
                return None
            while next_idx[cam_idx] < target:
                if not cap.grab():
                    return None
                next_idx[cam_idx] += 1
            ok, frame_img = cap.read()
            if not ok:
                return None
            next_idx[cam_idx] += 1
            return frame_img

        video_path = self.out_dir / "validation.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), scene.fps, (CANVAS_W, CANVAS_H))

        gt_track = gt
        pred_track = pred.xyz
        for k, frame in enumerate(frames):
            canvas = np.full((CANVAS_H, CANVAS_W, 3), (32, 32, 36), dtype=np.uint8)

            # ---- левая часть: 3D + график
            left = scene3d.render(frame, gt_track[frame], pred_track.get(frame))
            canvas[40 : 40 + scene3d.h, 0:LEFT_W] = left
            _text(canvas, f"{self.experiment_name}   frame {frame}   t={frame / scene.fps:5.2f}s", (12, 28), 0.55, TEXT)

            err = table.errors[k] if k < len(table.errors) else float("nan")
            chart = _draw_error_chart(LEFT_W, chart_h, chart_frames, chart_errors, frame, summary)
            canvas[40 + scene3d.h + PAD : CANVAS_H - 40, 0:LEFT_W] = chart[: CANVAS_H - 40 - (40 + scene3d.h + PAD), :]

            # ---- правая часть: камеры
            for slot, cam_idx in enumerate(sorted(scene.cameras)):
                col, row = slot % 2, slot // 2
                if slot >= 5:
                    break
                x, y = COL_X[col], ROW_Y[row]
                cam = Camera.from_track(scene.cameras[cam_idx], frame, scene.width, scene.height)
                frame_img = grab(cam_idx, frame + self.video_offset)
                if frame_img is not None:
                    last_img[cam_idx] = frame_img
                img = last_img.get(cam_idx)
                if img is None:
                    tile = np.full((TILE_H, TILE_W, 3), 40, dtype=np.uint8)
                    _tile_caption(tile, f"CAM{cam_idx}")
                    canvas[y : y + TILE_H, x : x + TILE_W] = tile
                    continue
                tile = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)

                gt_pt = gt_track[frame]
                pred_pt = pred_track.get(frame)
                h_gt = heading_from_positions(gt_track, frame)
                p_gt = pitch_from_positions(gt_track, frame) if self.use_pitch else 0.0
                h_pred = heading_from_positions(pred_track, frame) if pred_pt is not None else h_gt
                p_pred = (
                    pitch_from_positions(pred_track, frame) if (pred_pt is not None and self.use_pitch) else p_gt
                )
                scale = (TILE_W / scene.width, TILE_H / scene.height)
                for i, box in enumerate(self.model.boxes_2d(cam, gt_pt, h_gt, p_gt)):
                    _draw_boxes(tile, box, GT_COLOR, "GT" if i == 0 else None, scale, "bottom")
                detected = pred.meta.get("boxes", {}).get(frame, {}).get(cam_idx)
                if detected is not None:
                    _draw_rect(tile, detected, PRED_COLOR, "DET", scale)
                elif pred_pt is not None:
                    for i, box in enumerate(self.model.boxes_2d(cam, pred_pt, h_pred, p_pred)):
                        _draw_boxes(tile, box, PRED_COLOR, "PRED" if i == 0 else None, scale, "top")

                dist = cam.distance(gt_pt) / 100.0
                vis = cam.project(gt_pt)[0]
                visible = cam.in_image(vis, cam.project(gt_pt)[1])[0]
                _tile_caption(tile, f"CAM{cam_idx}  f={cam.focal_mm:.0f}mm  d={dist:.0f}m" + ("" if visible else "   [not visible]"))
                canvas[y : y + TILE_H, x : x + TILE_W] = tile

            # ---- сводка
            x, y = COL_X[1], ROW_Y[2]
            panel = np.full((TILE_H, TILE_W, 3), 22, dtype=np.uint8)
            _text(panel, "SUMMARY", (14, 26), 0.58, TEXT, 2)
            lines = [
                f"frame   {frame} / {frames[-1]}",
                f"time    {frame / scene.fps:6.2f} s",
                f"error   {err:6.2f} m",
                f"cameras {pred.n_cams.get(frame, 0)} / {len(scene.cameras)}",
            ]
            for i, line in enumerate(lines):
                _text(panel, line, (14, 54 + i * 20), 0.46, TEXT)
            _text(panel, "GT (m)", (14, 144), 0.44, GT_COLOR)
            g = gt_track[frame] / 100.0
            _text(panel, f"x {g[0]:8.1f}  y {g[1]:8.1f}  z {g[2]:6.1f}", (14, 164), 0.44, TEXT)
            _text(panel, "PRED (m)", (14, 190), 0.44, PRED_COLOR)
            p = pred_track.get(frame)
            if p is not None:
                p = p / 100.0
                _text(panel, f"x {p[0]:8.1f}  y {p[1]:8.1f}  z {p[2]:6.1f}", (14, 210), 0.44, TEXT)
            else:
                _text(panel, "no estimate", (14, 210), 0.44, MUTED)
            s = summary
            _text(panel, "RUN", (14, 240), 0.42, MUTED)
            _text(panel, f"RMSE {s.get('rmse_m', float('nan')):.2f}  p95 {s.get('p95_m', float('nan')):.2f}", (14, 260), 0.44, TEXT)
            _text(panel, f"median {s.get('median_m', float('nan')):.2f}   n={s.get('n', 0)}", (14, 280), 0.44, TEXT)
            _text(panel, "green = GT 2D    red = detection/prediction", (14, 312), 0.38, MUTED)
            canvas[y : y + TILE_H, x : x + TILE_W] = panel

            writer.write(canvas)
            if k % 100 == 0:
                print(f"  render {k}/{len(frames)}")

        writer.release()
        for cap in caps.values():
            cap.release()
        print(f"видео: {video_path}")
