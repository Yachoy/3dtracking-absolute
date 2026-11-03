"""Пять отдельных видео: на каждом кадре камеры — только известные данные.

Что рисуется (и только оно):
  - имя камеры, фокусное, разрешение;
  - поза камеры из CSV (x, y, z, yaw, pitch, roll);
  - кадр и время;
  - точка GT из object_transform_SM_AA_track.csv: её проекция перекрестием,
    мировые координаты и расстояние до камеры;
  - если точка вне кадра — стрелка у границы в её сторону и подпись.

Никаких боксов, моделей и предсказаний: это «эталонная разметка», с которой
можно сверять любой алгоритм.

    python -m tools.annotate_cameras --data data --out out/annotated
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.camera import Camera  # noqa: E402
from core.dataset import load_scene  # noqa: E402

W, H = 1920, 1080
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREY = (160, 160, 160)
CYAN = (255, 220, 60)
ORANGE = (60, 170, 255)


def _text(img, text, org, scale=0.62, color=WHITE, thick=1, plate=True):
    """Текст с затемнённой подложкой: читается на любом фоне."""
    x, y = int(org[0]), int(org[1])
    if plate:
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
        x0, y0 = max(x - 10, 0), max(y - th - 10, 0)
        x1, y1 = min(x + tw + 10, img.shape[1]), min(y + base + 8, img.shape[0])
        if x1 > x0 and y1 > y0:
            roi = img[y0:y1, x0:x1]
            roi[:] = (roi * 0.35).astype(np.uint8)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _crosshair(img, uv, color=CYAN, size=16, thick=2):
    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.circle(img, (u, v), size + 7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(img, (u, v), size + 7, color, 1, cv2.LINE_AA)
    cv2.line(img, (u - size, v), (u + size, v), color, thick, cv2.LINE_AA)
    cv2.line(img, (u, v - size), (u, v + size), color, thick, cv2.LINE_AA)


def _edge_arrow(img, uv, color=ORANGE):
    """Стрелка у границы кадра туда, где объект, и возвращает точку для подписи."""
    center = np.array([W / 2.0, H / 2.0], dtype=float)
    d = np.array([uv[0], uv[1]], dtype=float) - center
    n = np.linalg.norm(d)
    if n < 1e-6:
        d, n = np.array([1.0, 0.0]), 1.0
    d /= n
    tx = (W / 2.0 - 70.0) / abs(d[0]) if abs(d[0]) > 1e-6 else np.inf
    ty = (H / 2.0 - 70.0) / abs(d[1]) if abs(d[1]) > 1e-6 else np.inf
    tip = center + d * min(tx, ty)
    tail = tip - d * 80.0
    p0, p1 = tuple(np.int32(tail)), tuple(np.int32(tip))
    cv2.arrowedLine(img, p0, p1, (0, 0, 0), 9, cv2.LINE_AA, tipLength=0.45)
    cv2.arrowedLine(img, p0, p1, color, 5, cv2.LINE_AA, tipLength=0.45)
    return tip - d * 105.0


def render_camera(scene, cam_index, out_path: Path, show_map: bool = True) -> None:
    track = scene.cameras[cam_index]
    cap = cv2.VideoCapture(str(track.video))
    if not cap.isOpened():
        raise RuntimeError(f"не открывается {track.video}")

    poses = {f: pose for f, pose in track.poses.items()}
    frames = sorted(f for f in scene.target if f in poses)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(scene.fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    idx = -1
    try:
        for frame in frames:
            while idx < frame - 1:
                if not cap.grab():
                    break
                idx += 1
            ok, img = cap.read()
            idx += 1
            if not ok:
                break

            cam = Camera.from_track(track, frame, scene.width, scene.height)
            pose = poses[frame]
            point = scene.target[frame].loc
            dist_m = cam.distance(point) / 100.0
            uv, depth = cam.project(point)
            uv, depth = uv[0], float(depth[0])

            visible = depth > 0 and 0 <= uv[0] < W and 0 <= uv[1] < H

            # --- известные данные о камере ---
            _text(img, f"CAM{cam_index}  f={track.focal_mm:g} mm   {scene.width}x{scene.height}  {scene.fps:g} fps",
                  (28, 46), 0.78)
            _text(img, f"pos  x {pose.loc[0] / 100:8.2f}  y {pose.loc[1] / 100:8.2f}  z {pose.loc[2] / 100:7.2f} m",
                  (28, 82), 0.62)
            _text(img, f"yaw {pose.rot[2]:7.2f}   pitch {pose.rot[1]:6.2f}   roll {pose.rot[0]:6.2f} deg",
                  (28, 112), 0.62)

            # --- известные данные о объекте ---
            _text(img, f"GT  x {point[0] / 100:8.2f}  y {point[1] / 100:8.2f}  z {point[2] / 100:7.2f} m",
                  (28, H - 96), 0.72)
            _text(img, f"distance {dist_m:7.1f} m", (28, H - 62), 0.72,
                  color=CYAN if visible else ORANGE)
            _text(img, f"frame {frame}   t {frame / scene.fps:6.2f} s", (28, H - 26), 0.62, GREY)

            if not visible:
                _text(img, "object outside this camera", (28, 152), 0.62, ORANGE)

            # --- проекция точки GT ---
            if visible:
                _crosshair(img, uv)
                _text(img, f"GT ({point[0] / 100:.1f}, {point[1] / 100:.1f}, {point[2] / 100:.1f})  {dist_m:.0f} m",
                      (int(uv[0]) + 34, int(uv[1]) - 18), 0.62, CYAN)
            else:
                label_at = _edge_arrow(img, uv)
                _text(img, f"GT {dist_m:.0f} m", (int(label_at[0]) - 40, int(label_at[1])), 0.6, ORANGE)

            if show_map:
                _draw_map(img, scene, cam_index, frame, point, uv, visible)

            proc.stdin.write(img.tobytes())
        proc.stdin.close()
    finally:
        cap.release()
        proc.wait()


def _draw_map(img, scene, cam_index, frame, point, uv, visible) -> None:
    """Маленькая карта сверху справа: камеры, их секторы и точка сейчас."""
    size = 300
    pad = 24
    x0, y0 = W - size - pad, H - size - pad
    roi = img[y0:y0 + size, x0:x0 + size]
    cv2.rectangle(roi, (0, 0), (size, size), (18, 18, 18), -1)
    cv2.rectangle(roi, (0, 0), (size, size), (70, 70, 70), 1)

    pts = [scene.cameras[i].poses[frame].loc / 100.0 for i in sorted(scene.cameras)]
    pts = np.array(pts)
    allp = np.vstack([pts, point.reshape(1, 3) / 100.0])
    lo = allp[:, :2].min(axis=0) - 40
    hi = allp[:, :2].max(axis=0) + 40
    span = max(hi - lo)

    def to_px(p):
        u = (p[0] - lo[0]) / span * (size - 40) + 20
        v = (p[1] - lo[1]) / span * (size - 40) + 20
        return int(size - u), int(size - v)  # +Y вверх на карте

    for i in sorted(scene.cameras):
        c = scene.cameras[i].poses[frame].loc / 100.0
        p = to_px(c)
        color = CYAN if i == cam_index else (110, 110, 110)
        cv2.circle(roi, p, 4, color, -1, cv2.LINE_AA)
        cv2.putText(roi, f"{i}", (p[0] + 6, p[1] - 4), FONT, 0.4, color, 1, cv2.LINE_AA)

    trail = [to_px(scene.target[f].loc[:2] / 100.0) for f in sorted(scene.target) if f <= frame]
    if len(trail) > 1:
        cv2.polylines(roi, [np.array(trail, dtype=np.int32)], False, (90, 120, 160), 1, cv2.LINE_AA)

    gp = to_px(point / 100.0)
    cv2.drawMarker(roi, gp, CYAN if visible else ORANGE, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    cv2.putText(roi, "GT", (gp[0] + 9, gp[1] - 8), FONT, 0.38, WHITE, 1, cv2.LINE_AA)
    cv2.putText(roi, "top view, m", (8, size - 8), FONT, 0.4, GREY, 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out/annotated")
    ap.add_argument("--cams", default="", help="через запятую, по умолчанию все")
    ap.add_argument("--no-map", action="store_true")
    args = ap.parse_args()

    scene = load_scene(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cams = [int(c) for c in args.cams.split(",") if c.strip()] or sorted(scene.cameras)
    for i in cams:
        path = out_dir / f"cam{i}.mp4"
        print(f"cam{i}: {path}", flush=True)
        render_camera(scene, i, path, show_map=not args.no_map)
    print("готово:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
