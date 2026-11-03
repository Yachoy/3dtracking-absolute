"""Демо-видео: слева вид сверху (GT зелёным, трек красным), справа 5 камер с рамками YOLO.

Трекер прогоняется прямо здесь, кадр за кадром (онлайн), чтобы показать, какие рамки он взял.

    python -m tools.demo_video data/test --out out/demo.mp4
    python -m tools.demo_video data/test --flow --out out/demo_flow.mp4   # + рамки потока
"""
import argparse

import cv2
import imageio_ffmpeg
import numpy as np

from run import detections_by_frame, load_config
from tracker.camera import SENSOR_W_MM, load_cameras, load_ground_truth, video_path
from tracker.online import OnlineTracker3D

W, H = 1920, 1080
MAP_W = 960                      # левая половина — карта
TILE_W, TILE_H = 480, 270        # 5 камер сеткой 2x3 справа, 6-я ячейка + низ — инфо
SCALE = TILE_W / 1920

GREEN, RED, YELLOW, ORANGE, CYAN = (60, 200, 60), (40, 40, 230), (0, 220, 255), (0, 140, 255), (230, 220, 60)
WHITE, GREY, DARK, CAM_BLUE = (240, 240, 240), (120, 120, 120), (28, 28, 28), (230, 160, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class MapView:
    """Вид сверху (как в редакторе UE: X вправо, Y вниз), масштаб — по GT с запасом."""

    def __init__(self, gt, margin_m=40):
        lo, hi = np.nanmin(gt[:, :2], axis=0) - margin_m, np.nanmax(gt[:, :2], axis=0) + margin_m
        self.scale = min((MAP_W - 40) / (hi[0] - lo[0]), (H - 120) / (hi[1] - lo[1]))
        size = (hi - lo) * self.scale
        self.origin = np.array([(MAP_W - size[0]) / 2, 80 + (H - 120 - size[1]) / 2]) - lo * self.scale
        self.lo, self.hi = lo, hi

    def px(self, xy):
        return tuple(np.round(np.asarray(xy)[:2] * self.scale + self.origin).astype(int))

    def clamp(self, xy):
        return np.clip(xy[:2], self.lo, self.hi)

    def background(self, gt):
        img = np.full((H, MAP_W, 3), DARK, np.uint8)
        step = 50
        for x in np.arange(np.ceil(self.lo[0] / step) * step, self.hi[0], step):
            cv2.line(img, self.px((x, self.lo[1])), self.px((x, self.hi[1])), (45, 45, 45), 1)
        for y in np.arange(np.ceil(self.lo[1] / step) * step, self.hi[1], step):
            cv2.line(img, self.px((self.lo[0], y)), self.px((self.hi[0], y)), (45, 45, 45), 1)
        pts = np.array([self.px(p) for p in gt if not np.isnan(p).any()])
        cv2.polylines(img, [pts], False, (40, 90, 40), 1, cv2.LINE_AA)
        cv2.putText(img, "Top view (grid 50 m)", (20, 40), FONT, 0.9, WHITE, 2, cv2.LINE_AA)
        return img


def draw_camera_on_map(img, view, cam, pose, idx):
    """Камера и её поле зрения; если камера за пределами карты — значок на краю."""
    c = pose.center
    inside = np.all(c[:2] >= view.lo) and np.all(c[:2] <= view.hi)
    p = view.px(view.clamp(c))
    fwd = pose.rot[2]
    yaw = np.arctan2(fwd[1], fwd[0])
    half = np.arctan(SENSOR_W_MM / 2 / cam.focal_mm)
    if inside:
        for a in (yaw - half, yaw + half):
            end = c[:2] + 400 * np.array([np.cos(a), np.sin(a)])
            cv2.line(img, p, view.px(end), (90, 70, 40), 1, cv2.LINE_AA)
    cv2.circle(img, p, 7, CAM_BLUE, -1, cv2.LINE_AA)
    label_x = p[0] + 9 if p[0] < MAP_W - 50 else p[0] - 45  # у правого края подпись слева от значка
    cv2.putText(img, f"C{idx + 1}" + ("" if inside else "*"), (label_x, p[1] - 9), FONT, 0.6, CAM_BLUE, 2, cv2.LINE_AA)


def draw_tile(frame_img, pose, dets, used, gt_pt, pred_pt, idx):
    tile = cv2.resize(frame_img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    for d in dets:
        x1, y1, x2, y2 = (d.box * SCALE).astype(int)
        is_used = any(d is u for u in used)
        color = ORANGE if is_used else (YELLOW if d.source == "yolo" else CYAN)
        cv2.rectangle(tile, (x1, y1), (x2, y2), color, 3 if is_used else 1)
        text = f"{d.conf:.2f}" if d.source == "yolo" else f"flow +{d.age}"
        cv2.putText(tile, text, (x1, max(y1 - 4, 12)), FONT, 0.4, color, 1, cv2.LINE_AA)
    for pt, color in ((gt_pt, GREEN), (pred_pt, RED)):
        if pt is None or np.isnan(pt).any():
            continue
        uv, z = pose.project(pt)
        if z[0] > 0:
            u, v = (uv[0] * SCALE).astype(int)
            if 0 <= u < TILE_W and 0 <= v < TILE_H:
                cv2.circle(tile, (u, v), 6, color, 2, cv2.LINE_AA)
    label = f"CAM {idx + 1}" + ("  [used]" if used else "")
    cv2.rectangle(tile, (0, 0), (len(label) * 11 + 12, 24), (0, 0, 0), -1)
    cv2.putText(tile, label, (6, 17), FONT, 0.55, ORANGE if used else WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), (60, 60, 60), 1)
    return tile


def draw_info(frame, status, used_cams, err, errs):
    img = np.full((H - 2 * TILE_H, 960, 3), DARK, np.uint8)  # 3-й ряд сетки + полоса под ним; слева ляжет CAM 5
    x0 = TILE_W + 20  # 6-я ячейка справа от CAM 5
    status_color = {"TRACKING": GREEN, "LOST": ORANGE, "SEARCH": GREY}[status]
    lines = [
        (f"Frame {frame}", WHITE),
        (f"Status: {status}", status_color),
        (f"Cameras used: {', '.join(str(c + 1) for c in used_cams) or '-'}", WHITE),
        (f"Error now: {err:.1f} m" if err is not None else "Error now: -", WHITE),
        (f"MAE so far: {np.mean(errs):.1f} m" if errs else "MAE so far: -", WHITE),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (x0, 40 + 42 * i), FONT, 0.8, color, 2, cv2.LINE_AA)
    legend = [(GREEN, "GT trajectory / GT point"), (RED, "Online 3D track (frames <= t only)"),
              (YELLOW, "YOLO26 detection"), (CYAN, "Optical-flow box (frames since YOLO)"),
              (ORANGE, "Box used by tracker"), (CAM_BLUE, "Camera (* = off map)")]
    for i, (color, text) in enumerate(legend):
        y = TILE_H + 40 + 36 * i
        cv2.rectangle(img, (20, y - 14), (44, y + 4), color, -1)
        cv2.putText(img, text, (56, y), FONT, 0.75, WHITE, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--detections", default="cache/detections.csv")
    ap.add_argument("--out", default="out/demo.mp4")
    ap.add_argument("--flow", nargs="?", const="cache/flow_boxes.csv", help="показать и использовать рамки потока (вперёд)")
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args()

    cams = load_cameras(args.data_dir)
    gt_df = load_ground_truth(args.data_dir)
    gt = gt_df.interpolate().values if gt_df is not None else None
    dets = detections_by_frame(args.detections, cams, args.flow)
    tracker = OnlineTracker3D(cams, load_config())
    caps = [cv2.VideoCapture(str(video_path(args.data_dir, c.name))) for c in cams]

    view = MapView(gt if gt is not None else np.array([[-200, -200, 0], [200, 200, 0]]))
    base = view.background(gt) if gt is not None else view.background(np.empty((0, 3)))
    writer = imageio_ffmpeg.write_frames(args.out, (W, H), fps=30, codec="libx264", pix_fmt_in="bgr24",
                                         quality=7, macro_block_size=8)
    writer.send(None)

    pred_path, errs = [], []
    frames = cams[0].table.index[: args.max_frames]
    for t, frame in enumerate(frames):
        imgs = [cap.read()[1] for cap in caps]
        frame_dets = [d for d in dets.get(frame, []) if d.known_at <= frame]  # без обратного прохода: онлайн
        point, status = tracker.step(frame, frame_dets)
        tr = tracker.track
        used = tr.used_detections if tr is not None else []
        gt_pt = gt[t] if gt is not None else None
        err = float(np.linalg.norm(point - gt_pt)) if point is not None and gt_pt is not None else None
        if err is not None:
            errs.append(err)
        pred_path.append(view.px(point) if point is not None else None)

        canvas = np.full((H, W, 3), DARK, np.uint8)
        m = base.copy()
        poses = [c.pose(frame) for c in cams]
        for i, (c, pose) in enumerate(zip(cams, poses)):
            draw_camera_on_map(m, view, c, pose, i)
        if gt is not None:
            cv2.polylines(m, [np.array([view.px(p) for p in gt[: t + 1]])], False, GREEN, 3, cv2.LINE_AA)
            cv2.circle(m, view.px(gt_pt), 9, GREEN, -1, cv2.LINE_AA)
        for a, b in zip(pred_path, pred_path[1:]):
            if a is not None and b is not None:
                cv2.line(m, a, b, RED, 2, cv2.LINE_AA)
        if point is not None:
            cv2.circle(m, view.px(point), 9, RED, -1, cv2.LINE_AA)
            if gt_pt is not None:
                cv2.line(m, view.px(point), view.px(gt_pt), WHITE, 1, cv2.LINE_AA)
        canvas[:, :MAP_W] = m

        canvas[2 * TILE_H:, MAP_W:] = draw_info(frame, status, tr.used_cameras if tr else [], err, errs)
        for i, (img, pose) in enumerate(zip(imgs, poses)):
            if img is None:
                continue
            cam_used = [d for d in used if d.camera == i]
            tile = draw_tile(img, pose, [d for d in frame_dets if d.camera == i], cam_used, gt_pt, point, i)
            r, c = divmod(i, 2)
            canvas[r * TILE_H:(r + 1) * TILE_H, MAP_W + c * TILE_W:MAP_W + (c + 1) * TILE_W] = tile
        writer.send(np.ascontiguousarray(canvas))
        if t % 100 == 0:
            print(f"кадр {frame}/{len(frames)}")
    writer.close()
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()
