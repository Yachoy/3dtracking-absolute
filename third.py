# -*- coding: utf-8 -*-
"""
third.py - background subtraction, with YOLO-seg as the fallback when it fails,
then the same 3D pipeline as first.py.

Logic per camera and frame:
  1. background subtraction (first.py machinery: static median / pose
     background, crop around the predicted position, one merged region);
  2. the region is checked against the predicted 3D position.  If it is missing
     or inconsistent (projection residual > 60 px), the frame is re-detected
     with YOLO11s-seg on a crop around the prediction, upscaled 2x so that the
     small distant truck is inside the network's comfortable scale range;
  3. all observations -> multi-view triangulation -> constant-acceleration
     Kalman in WCS.

Output: out/third_grid.mp4 (5 cameras + top view; blue = bg subtraction,
magenta = YOLO segmentation, red = GT projection, yellow = the 3D estimate)
and out/third_pred.csv in the same format as the ground truth.

Usage:  python third.py [--nocache] [--nosave]
"""
import os
import sys
import time
import pickle
import numpy as np
import cv2
from ultralytics import YOLO

import first as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MODEL = os.path.join(HERE, "yolo11s-seg.pt")
VEHICLES = {2, 5, 7}            # car, bus, truck (COCO)
CONF = 0.18
ROI_HALF = 260                  # crop half-size around the prediction, px
ROI_ZOOM = 2.0                  # crop upscale before the network
RES_OK = 60.0                   # px: region == 3D estimate


def yolo_region(model, img, box=None):
    """Segment a vehicle.  box=(x0,y0,x1,y1) limits the search."""
    x0 = y0 = 0
    src = img
    if box is not None:
        H, W = img.shape[:2]
        x0 = max(0, int(box[0]))
        y0 = max(0, int(box[1]))
        x1 = min(W, int(box[2]))
        y1 = min(H, int(box[3]))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        src = cv2.resize(img[y0:y1, x0:x1], None, fx=ROI_ZOOM, fy=ROI_ZOOM,
                         interpolation=cv2.INTER_CUBIC)
    r = model.predict(src, imgsz=960, conf=CONF, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    best = None
    for i in range(len(r.boxes)):
        cls = int(r.boxes.cls[i].cpu().numpy())
        if cls not in VEHICLES:
            continue
        b = r.boxes.xyxy[i].cpu().numpy().astype(np.float64) / (ROI_ZOOM if box else 1.0)
        b += np.array([x0, y0, x0, y0])
        area = (b[2] - b[0]) * (b[3] - b[1])
        if best is None or area > best[0]:
            best = (area, b, float(r.boxes.conf[i]), i)
    if best is None:
        return None
    area, b, conf, i = best
    if r.masks is None:
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        poly = None
    else:
        poly = (r.masks.xy[i] / (ROI_ZOOM if box else 1.0)
                + np.array([x0, y0])).astype(np.int32)
        m = cv2.moments(poly)
        if m["m00"] > 0:
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        else:
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return dict(cx=float(cx), cy=float(cy), bbox=tuple(float(v) for v in b),
                area_w=max(area / 4.0, 1.0), conf=conf, poly=poly)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cams = [F.Camera(c) for c in F.CAMS]
    for c in cams:
        c.build_background()
    gt = F.load_gt()
    print("[bg] %.1fs" % (time.time() - t0))

    det_bg = F.detect_all(cams, os.path.join(OUT, "third_det_bg.pkl"),
                          rebuild="--nocache" in sys.argv)
    nb = sum(det_bg[ci][f] is not None for ci in range(len(cams)) for f in range(F.NFRAMES))
    print("[bg-det] %d camera-frames" % nb)

    pos1, _, got1, _ = F.track(cams, det_bg)
    e = pos1 - gt
    print("[bg-track] fixes=%d RMSE=(%.1f, %.1f, %.1f) cm, median %.1f m"
          % (got1.sum(), *np.sqrt((e ** 2).mean(0)),
             np.median(np.linalg.norm(e, axis=1)) / 100))

    # ---- YOLO fallback for the frames the background subtraction lost ------
    cache = os.path.join(OUT, "third_det_yolo.pkl")
    det = None
    if os.path.exists(cache) and "--nocache" not in sys.argv:
        with open(cache, "rb") as fh:
            det = pickle.load(fh)
        print("[yolo] cache loaded")
    if det is None:
        det = [[None] * F.NFRAMES for _ in cams]
        model = YOLO(MODEL)
        caps = [cv2.VideoCapture(c.video) for c in cams]
        ny = 0
        for f in range(F.NFRAMES):
            for ci, cap in enumerate(caps):
                ok, img = cap.read()
                if not ok:
                    continue
                d = det_bg[ci][f]
                pred_uv = None
                p = cams[ci].project(f, pos1[f])
                if p is not None and 0 <= p[0] < F.W and 0 <= p[1] < F.H:
                    pred_uv = p
                bad = d is None or (pred_uv is not None and
                                    np.hypot(d["cx"] - pred_uv[0],
                                             d["cy"] - pred_uv[1]) > RES_OK)
                if not bad:
                    d2 = dict(d)
                    d2["src"] = "bg"
                    det[ci][f] = d2
                    continue
                roi = None
                if pred_uv is not None:
                    roi = (pred_uv[0] - ROI_HALF, pred_uv[1] - ROI_HALF,
                           pred_uv[0] + ROI_HALF, pred_uv[1] + ROI_HALF)
                r = yolo_region(model, img, roi)
                if r is not None:
                    r["src"] = "yolo"
                    det[ci][f] = r
                    ny += 1
            if f % 200 == 0:
                print("[yolo] f%4d  yolo detections %d  %.1fs" % (f, ny, time.time() - t0))
        for cap in caps:
            cap.release()
        with open(cache, "wb") as fh:
            pickle.dump(det, fh)
    ny = sum(det[ci][f] is not None and det[ci][f].get("src") == "yolo"
             for ci in range(len(cams)) for f in range(F.NFRAMES))
    print("[yolo] fallback detections: %d (%.1fs)" % (ny, time.time() - t0))

    pos, vel, got, _ = F.track(cams, det)
    sm = F.rts(pos, got)
    for name, tr in (("causal", pos), ("rts", sm)):
        e = tr - gt
        print("[%s] fixes=%d RMSE=(%.1f, %.1f, %.1f) cm | 3D=%.1f m | median=%.1f m"
              % (name, got.sum(), *np.sqrt((e ** 2).mean(0)),
                 np.linalg.norm(np.sqrt((e ** 2).mean(0))) / 100,
                 np.median(np.linalg.norm(e, axis=1)) / 100))

    out_csv = os.path.join(OUT, "third_pred.csv")
    with open(out_csv, "w", encoding="utf-8") as fh:
        fh.write("Frame;Loc_X;Loc_Y;Loc_Z\n")
        for f in range(F.NFRAMES):
            fh.write("%d;%.4f;%.4f;%.4f\n" % (f, pos[f, 0], pos[f, 1], pos[f, 2]))
    print("[out]", out_csv)

    if "--nosave" not in sys.argv:
        render(cams, det, pos, gt, os.path.join(OUT, "third_grid.mp4"))
    print("[done] %.1fs" % (time.time() - t0))


def render(cams, det, pos, gt, path):
    caps = [cv2.VideoCapture(c.video) for c in cams]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 1080))
    x0, x1, y0, y1 = -62000, 42000, -42000, 22000
    used = np.zeros((F.NFRAMES,), int)

    def to_map(p):
        return (int((p[1] - y0) / (y1 - y0) * 620),
                int((1 - (p[0] - x0) / (x1 - x0)) * 340) + 8)

    for f in range(F.NFRAMES):
        cells = []
        n_used = 0
        for ci, cap in enumerate(caps):
            ok, img = cap.read()
            if not ok:
                img = np.zeros((F.H, F.W, 3), np.uint8)
            tile = cv2.resize(img, (640, 360))
            d = det[ci][f]
            if d is not None:
                if d.get("src") == "yolo" and d.get("poly") is not None:
                    poly = (np.asarray(d["poly"], np.float64) / 3.0).astype(np.int32)
                    lay = tile.copy()
                    cv2.fillPoly(lay, [poly], (255, 0, 255))
                    tile = cv2.addWeighted(lay, 0.35, tile, 0.65, 0)
                    cv2.polylines(tile, [poly], True, (255, 0, 255), 2)
            if d is not None:
                b = [int(v / 3) for v in d["bbox"]]
                col = (0, 255, 255) if d.get("src") == "yolo" else (255, 160, 0)
                cv2.rectangle(tile, (b[0], b[1]), (b[2], b[3]), col, 2)
            pj = cams[ci].project(f, gt[f])
            if pj is not None:
                cv2.drawMarker(tile, (int(pj[0] / 3), int(pj[1] / 3)), (0, 0, 255),
                               cv2.MARKER_CROSS, 16, 2)
            ep = cams[ci].project(f, pos[f])
            if ep is not None:
                cv2.drawMarker(tile, (int(ep[0] / 3), int(ep[1] / 3)), (0, 255, 255),
                               cv2.MARKER_TILTED_CROSS, 16, 2)
                if d is not None and np.hypot(d["cx"] - ep[0], d["cy"] - ep[1]) < RES_OK:
                    n_used += 1
            src = "-" if d is None else d.get("src", "bg")
            head = "cam%d f%03d  %s" % (cams[ci].cid, f, src)
            cv2.putText(tile, head, (6, 18), 0, 0.45, (0, 0, 0), 3)
            cv2.putText(tile, head, (6, 18), 0, 0.45,
                        (0, 255, 0) if d is not None else (150, 180, 220), 1)
            cells.append(tile)
        used[f] = n_used
        mp = np.full((360, 640, 3), 25, np.uint8)
        cv2.putText(mp, "top view  GT=red  est=yellow  cams used=%d" % n_used,
                    (8, 20), 0, 0.45, (200, 200, 200), 1)
        cv2.polylines(mp, [np.array([to_map(p) for p in gt], np.int32)], False,
                      (60, 60, 160), 1)
        epts = np.array([to_map(p) for p in pos[:f + 1]], np.int32)
        if len(epts) > 1:
            cv2.polylines(mp, [epts], False, (0, 220, 255), 1)
        for c in cams:
            pc = to_map(c.pose[f, 1:4])
            cv2.drawMarker(mp, pc, (255, 120, 0), cv2.MARKER_TRIANGLE_UP, 12, 2)
            cv2.putText(mp, str(c.cid), (pc[0] + 6, pc[1] + 4), 0, 0.45, (255, 120, 0), 1)
        cv2.drawMarker(mp, to_map(gt[f]), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.drawMarker(mp, to_map(pos[f]), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(mp, "err=%.1f m" % (np.linalg.norm(pos[f] - gt[f]) / 100.0),
                    (8, 344), 0, 0.5, (0, 255, 255), 1)
        cells.append(mp)
        vw.write(np.vstack([np.hstack(cells[0:2]), np.hstack(cells[2:4]),
                            np.hstack(cells[4:6])]))
    vw.release()
    for cap in caps:
        cap.release()
    print("[viz]", path)


if __name__ == "__main__":
    main()
