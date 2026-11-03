# -*- coding: utf-8 -*-
"""
second.py - detection of the only moving object by optical flow.

Why: for a camera that rotates, a background model is fragile (the pose in the
CSV and the rendered image disagree by ~10% in scale), but the *motion* is
easy.  Everything static obeys one global homography, the truck does not.
So the detector is:

    1. Shi-Tomasi corners on frame f-1 (FAST-like, uniform over the image)
    2. LK optical flow f-1 -> f
    3. RANSAC homography fitted to ALL flow vectors  = camera-induced motion
    4. residual of every point against that model; outliers = anomalous motion
    5. the outliers are merged into clusters; the biggest cluster is the truck
    6. its bounding box is the detection

Two classes of points appear naturally: static ones (fit the global model) and
the ones that move synchronously in their own direction (the truck, and its
shadow).  No background image, no feature matching against a reference, no
dependence on the intrinsics being perfect - RANSAC absorbs the small mismatch
between the CSV pose and the render.

The grid video puts all five cameras next to each other plus the top view, the
same layout as first.py:
    blue dots    - static points (consistent with the global model)
    magenta dots - anomalous points
    yellow box   - merged cluster = detection
    red cross    - ground-truth projection (evaluation only, not used)

Usage:
    python second.py             # -> out/flow_grid.mp4
    python second.py --still 620 # single frame -> out/flow_still_620.png
"""
import os
import sys
import time
import numpy as np
import cv2

DATA = r"C:\Users\misha\PycharmProjects\CudaTest\hakTrack\data"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

W, H = 1920, 1080
SENSOR_W = 23.76
WW, WH = 960, 540
NFRAMES = 1001
CAMS = [1, 2, 3, 4, 5]

RES_PX = 3.0          # absolute floor for the flow residual, px
MIN_PTS = 6           # min anomalous points for a detection
GRID = 16             # clustering grid cell, px (working resolution)
MERGE = 2             # dilation of the grid, cells (~MERGE*GRID px gap)

LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
FT = dict(maxCorners=1200, qualityLevel=0.01, minDistance=7, blockSize=7)


# ------------------------------------------------------------------ camera --
def _basis(roll, pitch, yaw):
    r, p, y = np.radians([roll, pitch, yaw])
    F = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
    R0 = np.array([-np.sin(y), np.cos(y), 0.0])
    U0 = np.cross(F, R0)
    R = R0 * np.cos(r) + U0 * np.sin(r)
    U = U0 * np.cos(r) - R0 * np.sin(r)
    return np.column_stack([F, R, U])


def _read_cam_csv(path):
    lines = open(path, encoding="cp1251").read().splitlines()
    hdr = lines[0].split(";")
    foc = float(hdr[10]) if len(hdr) > 10 and hdr[10].strip() else 35.0
    rows = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        v = []
        for c in ln.split(";")[:7]:
            try:
                v.append(float(c))
            except ValueError:
                v.append(float("nan"))          # Excel date-coerced cells
        rows.append(v)
    a = np.array(rows, float)
    idx = np.arange(len(a))
    for c in range(a.shape[1]):
        bad = ~np.isfinite(a[:, c])
        if bad.any():
            a[bad, c] = np.interp(idx[bad], idx[~bad], a[~bad, c])
    return foc, a


class Camera:
    def __init__(self, cid):
        self.cid = cid
        self.foc, self.pose = _read_cam_csv(
            os.path.join(DATA, "object_transform_CineCameraActor%d.csv" % cid))
        self.video = os.path.join(DATA, "Way1.FinalImage.CineCameraActor%d.mp4" % cid)
        n = len(self.pose)
        self.fpx = self.foc * W / SENSOR_W
        self.K = np.array([[self.fpx, 0.0, W / 2.0],
                           [0.0, self.fpx, H / 2.0],
                           [0.0, 0.0, 1.0]])
        self.Proj = np.zeros((n, 3, 4))
        for i in range(n):
            Fv, Rv, Uv = _basis(*self.pose[i, 4:7]).T
            M = np.vstack([Rv, -Uv, Fv])          # pinhole: z=fwd, x=right, y=down
            C = self.pose[i, 1:4]
            self.Proj[i] = self.K @ np.hstack([M, (-M @ C)[:, None]])

    def project(self, i, P3):
        i = min(i, len(self.Proj) - 1)
        X = self.Proj[i] @ np.append(P3, 1.0)
        if X[2] <= 1e-6:
            return None
        return X[0] / X[2], X[1] / X[2]


def load_gt():
    raw = np.genfromtxt(os.path.join(DATA, "object_transform_SM_AA_track.csv"),
                        delimiter=";", skip_header=1, encoding="cp1251")
    fr = raw[:NFRAMES, 0]
    gt = raw[:NFRAMES, 1:4].astype(float)
    for c in range(3):
        bad = ~np.isfinite(gt[:, c])
        if bad.any():
            gt[bad, c] = np.interp(fr[bad], fr[~bad], gt[~bad, c])
    return gt


# ---------------------------------------------------------------- detector --
def flow_detect(prev_gray, gray):
    """Corners + LK + RANSAC global motion -> anomalous cluster box.

    Returns (pts_static, pts_anom, box or None).
    """
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FT)
    if p0 is None or len(p0) < 30:
        return np.empty((0, 2)), np.empty((0, 2)), None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK)
    ok = st.ravel() == 1
    p0, p1 = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
    if len(p0) < 30:
        return p0, np.empty((0, 2)), None
    Hm, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 2.0)
    if Hm is None:
        return p0, np.empty((0, 2)), None
    inl = inl.ravel().astype(bool)
    pred = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    res = np.linalg.norm(p1 - pred, axis=1)
    scale = 1.4826 * np.median(res[inl]) + 0.5
    anom = (~inl) | (res > max(RES_PX, 3.0 * scale))
    pts = p1[anom]
    if len(pts) < MIN_PTS:
        return p1[~anom], pts, None
    # cluster on a coarse grid: the truck is one blob, speckle is not
    gh, gw = WH // GRID + 1, WW // GRID + 1
    grid = np.zeros((gh, gw), np.uint8)
    np.add.at(grid, (np.clip(pts[:, 1] // GRID, 0, gh - 1).astype(int),
                     np.clip(pts[:, 0] // GRID, 0, gw - 1).astype(int)), 1)
    grid = cv2.dilate(grid, np.ones((2 * MERGE + 1, 2 * MERGE + 1), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((grid > 0).astype(np.uint8), 8)
    if n < 2:
        return p1[~anom], pts, None
    j = 1 + int(np.argmax(stats[1:, 4]))
    m = lab == j
    ci = np.clip(pts[:, 0] // GRID, 0, gw - 1).astype(int)
    cj = np.clip(pts[:, 1] // GRID, 0, gh - 1).astype(int)
    sel = m[cj, ci]
    if sel.sum() < MIN_PTS:
        return p1[~anom], pts, None
    q = pts[sel]
    x0, y0 = q.min(0) - 4
    x1, y1 = q.max(0) + 4
    box = (int(x0), int(y0), int(x1), int(y1))
    return p1[~anom], q, box


# ------------------------------------------------------------------- video --
def render(gt, cam_ids=CAMS, path=None, still=None):
    cams = [Camera(c) for c in cam_ids]
    path = path or os.path.join(OUT, "flow_grid.mp4")
    caps = [cv2.VideoCapture(c.video) for c in cams]
    vw = None if still is not None else cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 1080))

    x0, x1, y0, y1 = -62000, 42000, -42000, 22000

    def to_map(p):
        return (int((p[1] - y0) / (y1 - y0) * 620),
                int((1 - (p[0] - x0) / (x1 - x0)) * 340) + 8)

    prev_gray = [None] * len(cams)
    n_det = 0
    t0 = time.time()
    for f in range(NFRAMES):
        cells = []
        for ci, cap in enumerate(caps):
            ok, img = cap.read()
            if not ok:
                img = np.zeros((H, W, 3), np.uint8)
            gray = cv2.cvtColor(cv2.resize(img, (WW, WH)), cv2.COLOR_BGR2GRAY)
            if prev_gray[ci] is None:
                static, anom, box = np.empty((0, 2)), np.empty((0, 2)), None
            else:
                static, anom, box = flow_detect(prev_gray[ci], gray)
            prev_gray[ci] = gray

            tile = cv2.resize(img, (640, 360))
            sc = 640.0 / WW
            for x, y in static:
                cv2.circle(tile, (int(x * sc), int(y * sc)), 1, (255, 190, 60), -1)
            for x, y in anom:
                cv2.circle(tile, (int(x * sc), int(y * sc)), 2, (255, 0, 255), -1)
            if box is not None:
                n_det += 1
                b = [int(v * sc) for v in box]          # working res -> tile
                cv2.rectangle(tile, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
            pj = cams[ci].project(f, gt[f])
            if pj is not None:
                cv2.drawMarker(tile, (int(pj[0] / 3), int(pj[1] / 3)), (0, 0, 255),
                               cv2.MARKER_CROSS, 16, 2)
            head = "cam%d f%03d  %s" % (cams[ci].cid, f,
                                        "no det" if box is None else "det %d pts" % len(anom))
            cv2.putText(tile, head, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(tile, head, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0) if box is not None else (160, 200, 255), 1)
            cells.append(tile)

        mp = np.full((360, 640, 3), 25, np.uint8)
        cv2.putText(mp, "top view  GT=red", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)
        cv2.polylines(mp, [np.array([to_map(p) for p in gt], np.int32)], False,
                      (60, 60, 160), 1)
        for c in cams:
            pc = to_map(c.pose[f, 1:4])
            cv2.drawMarker(mp, pc, (255, 120, 0), cv2.MARKER_TRIANGLE_UP, 12, 2)
            cv2.putText(mp, str(c.cid), (pc[0] + 6, pc[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)
        cv2.drawMarker(mp, to_map(gt[f]), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cells.append(mp)

        grid = np.vstack([np.hstack(cells[0:2]), np.hstack(cells[2:4]),
                          np.hstack(cells[4:6])])
        if still is not None and f >= still:
            ok, buf = cv2.imencode(".png", grid)
            p = os.path.join(OUT, "flow_still_%d.png" % still)
            open(p, "wb").write(buf.tobytes())
            print("[still]", p)
            break
        if vw is not None:
            vw.write(grid)
        if f % 100 == 0:
            print("[flow] f%4d  detections so far %d  %.1fs" % (f, n_det, time.time() - t0))
    if vw is not None:
        vw.release()
        print("[out]", path, " %.1fs" % (time.time() - t0))
    for cap in caps:
        cap.release()


def main():
    os.makedirs(OUT, exist_ok=True)
    gt = load_gt()
    if "--still" in sys.argv:
        f = int(sys.argv[sys.argv.index("--still") + 1])
        render(gt, still=f)
    else:
        render(gt)


if __name__ == "__main__":
    main()
