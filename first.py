# -*- coding: utf-8 -*-
"""
3D tracking of the tracked actor (SM_AA truck) from 5 cameras.

Pipeline (causal, online - frame f uses only frames <= f):
  1. per camera: background model. Static cameras: median over an early window.
     A camera whose pose in the CSV is animated gets a *propagated* background:
     the background is registered frame-to-frame (ORB homography) and warped
     into the current view, so a rotating camera still yields a valid mask.
  2. per camera: ONE motion region (the truck is the only moving thing) found
     inside a CROP around the predicted position; growth of the box per frame
     is limited and the region must be dense, so foliage flicker / whole-image
     misalignment cannot drag the box away.
  3. multi-view DLT triangulation (+ covariance from the reprojection
     Jacobian) -> constant-velocity Kalman with Huber weighting -> output.

Usage:
    python first.py            # track, write out/pred_track.csv, print metrics
    python first.py --viz      # + out/grid.mp4 (5 cameras + top view)
    python first.py --nocache  # rebuild the detection cache
"""
import os
import sys
import time
import pickle
import numpy as np
import cv2

DATA = r"C:\Users\misha\PycharmProjects\CudaTest\hakTrack\data"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

W, H = 1920, 1080
SENSOR_W = 23.76
WW, WH = 960, 540           # working resolution for detection
SC = W / WW                 # working-res -> full-res factor (2.0)
NFRAMES = 1001
CAMS = [1, 2, 3, 4, 5]

Z_THR = 4.5                 # adaptive threshold, in robust sigmas
ABS_THR = 8                 # ... and at least this many grey levels
MIN_AREA = 20               # min component area at working resolution
DENSITY = 0.15              # min moving pixels / box area
MAX_GROW = 1.55             # max box size growth per frame (side ratio)
MAX_AREA_GROW = 3.0         # max silhouette area growth per frame
MODE = "bright"             # "bright": only brighter-than-bg motion (truck, no shadow)
PIX_SIGMA = 6.0             # assumed 2D centroid noise (px at 1080p)


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


class PoseBG:
    """Background of a rotating camera, stored by VIEW DIRECTION, not by pixel.

    The reference is an equirectangular panorama (azimuth x elevation) in world
    coordinates, sampled at the same angular resolution as the working image.
    Every pixel of every frame is mapped to (az, el) through its own pose and
    intrinsics, so the background is accumulated over directions: the camera
    may pan as far as it likes, the same patch of dune always lands on the same
    panorama texel.  No feature matching - the pose is known.

    The panorama is a median over n time-separated samples per texel, which is
    what removes the truck (it passes any given direction only briefly).
    """

    def __init__(self, cam, n=8):
        self.cam = cam
        self.n = n
        yaw = np.unwrap(np.deg2rad(cam.pose[:, 6]))
        fovx = 2.0 * np.arctan(W / (2.0 * cam.fpx))
        fovy = 2.0 * np.arctan(H / (2.0 * cam.fpx))
        self.az0 = float(yaw.min()) - 0.6 * fovx
        self.az1 = float(yaw.max()) + 0.6 * fovx
        self.el0, self.el1 = -0.9 * fovy, 0.9 * fovy
        self.PW = int(np.ceil((self.az1 - self.az0) / fovx * WW))
        self.PH = int(np.ceil((self.el1 - self.el0) / fovy * WH))
        ys, xs = np.mgrid[0:self.PH, 0:self.PW]
        az = self.az0 + (xs + 0.5) / self.PW * (self.az1 - self.az0)
        el = self.el1 - (ys + 0.5) / self.PH * (self.el1 - self.el0)
        self.d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                           np.sin(el)], -1).astype(np.float32)
        self.pano = np.zeros((self.PH, self.PW, 3), np.uint8)
        self.seen = np.zeros((self.PH, self.PW), np.float32)
        self.ok = True

    def _maps_to_pano(self, M, w, h):
        """pixel (u,v) of the current view -> panorama texel."""
        sx, sy = w / float(W), h / float(H)
        fx, fy = self.cam.fpx * sx, self.cam.fpx * sy
        ys, xs = np.mgrid[0:h, 0:w]
        dx = (xs - w / 2.0) / fx
        dy = (ys - h / 2.0) / fy
        dz = np.ones_like(dx)
        n = np.sqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = dx / n, dy / n, dz / n
        wx = M[0, 0] * dx + M[1, 0] * dy + M[2, 0] * dz      # d = M^T v
        wy = M[0, 1] * dx + M[1, 1] * dy + M[2, 1] * dz
        wz = M[0, 2] * dx + M[1, 2] * dy + M[2, 2] * dz
        az = np.arctan2(wy, wx)
        az = self.az0 + ((az - self.az0) % (2 * np.pi))      # unwrap to the sweep
        el = np.arcsin(np.clip(wz, -1, 1))
        px = ((az - self.az0) / (self.az1 - self.az0) * self.PW - 0.5)
        py = ((self.el1 - el) / (self.el1 - self.el0) * self.PH - 0.5)
        return px.astype(np.float32), py.astype(np.float32)

    def _maps_to_view(self, M, w, h):
        """panorama texel -> where it lands in the current view."""
        dx, dy, dz = self.d[..., 0], self.d[..., 1], self.d[..., 2]
        cx = M[0, 0] * dx + M[0, 1] * dy + M[0, 2] * dz
        cy = M[1, 0] * dx + M[1, 1] * dy + M[1, 2] * dz
        cz = M[2, 0] * dx + M[2, 1] * dy + M[2, 2] * dz
        sx, sy = w / float(W), h / float(H)
        fx, fy = self.cam.fpx * sx, self.cam.fpx * sy
        z = np.where(np.abs(cz) < 1e-6, 1e-6, cz)
        mx = (fx * cx / z + w / 2.0).astype(np.float32)
        my = (fy * cy / z + h / 2.0).astype(np.float32)
        ok = (cz > 1e-3) & (mx >= 0) & (mx < w) & (my >= 0) & (my < h)
        return mx, my, ok

    def collect(self, cam, step=3):
        """median over n time-separated samples of every direction."""
        n = self.n
        buf = np.zeros((n, self.PH, self.PW, 3), np.uint8)
        filled = np.zeros((n, self.PH, self.PW), bool)
        cnt = np.zeros((self.PH, self.PW), np.int32)
        cap = cv2.VideoCapture(cam.video)
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                small = cv2.resize(img, (WW, WH))
                mx, my, vis = self._maps_to_view(cam._M(i), WW, WH)
                snap = cv2.remap(small, mx, my, cv2.INTER_LINEAR)
                # one sample per texel per epoch: the n slots are spread over the
                # whole clip, so the median sees the same direction at n
                # different times and a passing truck outvotes itself
                k = min(n - 1, (i * n) // max(NFRAMES, 1))
                buf[k][vis] = snap[vis]
                filled[k][vis] = True
                cnt += vis
            i += 1
        cap.release()
        med = np.zeros((self.PH, self.PW, 3), np.float32)
        stack = buf.astype(np.float32)
        order = np.arange(n)[:, None, None]
        for py in range(0, self.PH, 64):
            sl = slice(py, min(py + 64, self.PH))
            blk = stack[:, sl]
            okm = filled[:, sl]
            blk = np.where(okm[..., None], blk, np.nan)
            with np.errstate(all="ignore"):
                med[sl] = np.nanmedian(blk, axis=0)
        have = cnt > 0
        self.pano = np.where(have[..., None], med, 0).astype(np.uint8)
        self.seen = have.astype(np.float32)

    def step(self, f, small):
        M = self.cam._M(f)
        h, w = small.shape[:2]
        px, py = self._maps_to_pano(M, w, h)
        bg = cv2.remap(self.pano, px, py, cv2.INTER_LINEAR).astype(np.float32)
        sg = cv2.remap(self.cam.sig_max.astype(np.float32), px, py,
                       cv2.INTER_LINEAR, borderValue=1e6)
        seen = cv2.remap(self.seen, px, py, cv2.INTER_LINEAR) > 0.5
        cur = small.astype(np.float32)
        bg[~seen] = cur[~seen]                    # unknown -> claim static
        self.ok = bool(seen.mean() > 0.25)
        return bg, sg, seen


class Camera:
    def __init__(self, cid):
        self.cid = cid
        self.foc, self.pose = _read_cam_csv(
            os.path.join(DATA, "object_transform_CineCameraActor%d.csv" % cid))
        self.video = os.path.join(DATA, "Way1.FinalImage.CineCameraActor%d.mp4" % cid)
        self.fpx = self.foc * W / SENSOR_W
        self.K = np.array([[self.fpx, 0.0, W / 2.0],
                           [0.0, self.fpx, H / 2.0],
                           [0.0, 0.0, 1.0]])
        n = len(self.pose)
        self.C = self.pose[:, 1:4].astype(np.float64)
        self.Proj = np.zeros((n, 3, 4))
        for i in range(n):
            Fv, Rv, Uv = _basis(*self.pose[i, 4:7]).T
            M = np.vstack([Rv, -Uv, Fv])          # pinhole: z=fwd, x=right, y=down
            self.Proj[i] = self.K @ np.hstack([M, (-M @ self.C[i])[:, None]])
        yaw = self.pose[:, 6]
        self.static = np.abs(yaw - yaw[0]) < 0.75
        self.bg_hi = NFRAMES - 1
        if not self.static.all():
            self.bg_hi = int(np.where(~self.static)[0][0]) - 2
        self.mbg = None
        self.pano = None

    def _M(self, i):
        """World-direction -> pinhole camera axes (rows x=right, y=down, z=fwd)."""
        i = min(i, len(self.pose) - 1)
        Fv, Rv, Uv = _basis(*self.pose[i, 4:7]).T
        return np.vstack([Rv, -Uv, Fv])

    def project(self, i, P3):
        i = min(i, len(self.Proj) - 1)
        X = self.Proj[i] @ np.append(P3, 1.0)
        if X[2] <= 1e-6:
            return None
        return X[0] / X[2], X[1] / X[2]

    def usable(self, i):
        i = min(i, len(self.static) - 1)
        if self.static[i]:
            return True
        return self.mbg is not None and self.mbg.ok

    def build_background(self):
        cap = cv2.VideoCapture(self.video)
        frames = np.unique(np.linspace(0, max(self.bg_hi, 2), 80).astype(int))
        buf = []
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if ok:
                buf.append(cv2.resize(img, (WW, WH)))
        cap.release()
        st = np.stack(buf).astype(np.float32)
        self.bg = np.median(st, axis=0)
        self.sigma = 1.4826 * np.median(np.abs(st - self.bg), axis=0) + 3.0
        self.sig_max = self.sigma.max(2).astype(np.float32)

    def bg_for(self, i, small):
        i = min(i, len(self.static) - 1)
        if self.static[i]:
            return self.bg, self.sig_max, None
        if self.mbg is None:
            self.mbg = PoseBG(self)
            if self.pano is None:
                self.mbg.collect(self)
                self.pano = (self.mbg.pano, self.mbg.seen)
            else:
                self.mbg.pano, self.mbg.seen = self.pano[0], self.pano[1]
        return self.mbg.step(i, small)


# ------------------------------------------------------------- detection ----
def _mask(crop, bgc, sigc, blur=0):
    """Positive (brighter) motion only: the truck is white, its cast shadow is
    dark.  Using |diff| merges the shadow into the region and drags the centroid
    ~10 m backwards, so the signed difference is used instead."""
    if blur:
        crop = cv2.GaussianBlur(crop, (blur | 1, blur | 1), 0)
        bgc = cv2.GaussianBlur(bgc, (blur | 1, blur | 1), 0)
    d = crop.astype(np.float32) - bgc
    if MODE == "abs":
        dm = np.abs(d).max(2)
    else:
        dm = d.max(2)
    m = ((dm / sigc > Z_THR) & (dm > ABS_THR)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def find_region(small, bg, sig_max, uv=None, half=None, prev=None, blur=0):
    """One motion region, searched inside a crop around `uv` (working res)."""
    if uv is None:                                    # full-image search
        x0, y0, x1, y1 = 0, 0, WW, WH
    else:
        half = int(np.clip(half if half else 60, 30, 260))
        x0 = int(np.clip(uv[0] - half, 0, WW - 1))
        x1 = int(np.clip(uv[0] + half, 1, WW))
        y0 = int(np.clip(uv[1] - half, 0, WH - 1))
        y1 = int(np.clip(uv[1] + half, 1, WH))
    crop = small[y0:y1, x0:x1]
    m = _mask(crop, bg[y0:y1, x0:x1], sig_max[y0:y1, x0:x1], blur)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    comps = [i for i in range(1, n) if stats[i, 4] >= MIN_AREA]
    if not comps:
        return None
    if uv is None:
        comps.sort(key=lambda i: -stats[i, 4])
    else:                                             # nearest to the prediction
        px, py = uv[0] - x0, uv[1] - y0
        comps.sort(key=lambda i: np.hypot(cent[i][0] - px, cent[i][1] - py))
        lim = 0.7 * half
        comps = [i for i in comps if np.hypot(cent[i][0] - px, cent[i][1] - py) <= lim]
        if not comps:
            return None
    main = comps[0]
    x, y, w, h, a = stats[main]
    box = [x, y, x + w, y + h]
    used = [main]
    if prev is not None:                              # bounded merge of neighbours
        lim = MAX_GROW
        for i in comps[1:]:
            bx, by, bw, bh, ba = stats[i]
            gap = 10
            if not (bx < box[2] + gap and bx + bw > box[0] - gap and
                    by < box[3] + gap and by + bh > box[1] - gap):
                continue
            nx0, ny0 = min(box[0], bx), min(box[1], by)
            nx1, ny1 = max(box[2], bx + bw), max(box[3], by + bh)
            if (nx1 - nx0) > lim * max(box[2] - box[0], 8) or \
               (ny1 - ny0) > lim * max(box[3] - box[1], 8):
                continue
            used.append(i)
            box = [nx0, ny0, nx1, ny1]
    sel = np.isin(lab, used)
    ms = cv2.moments(sel.astype(np.uint8), binaryImage=True)
    if ms["m00"] <= 0:
        return None
    area = float(ms["m00"])
    bw, bh = box[2] - box[0], box[3] - box[1]
    dens = area / max(bw * bh, 1)
    if dens < DENSITY:
        return None
    if prev is not None:
        if area > MAX_AREA_GROW * prev["area_w"] or area < prev["area_w"] / MAX_AREA_GROW:
            return None
        if bw > MAX_GROW * prev["w_w"] + 6 or bh > MAX_GROW * prev["h_w"] + 6:
            return None
    cx = x0 + ms["m10"] / ms["m00"]
    cy = y0 + ms["m01"] / ms["m00"]
    return dict(cx=cx * SC, cy=cy * SC,
                bbox=((x0 + box[0]) * SC, (y0 + box[1]) * SC,
                      (x0 + box[2]) * SC, (y0 + box[3]) * SC),
                area=area * SC * SC, dens=float(dens),
                w_w=float(bw), h_w=float(bh), area_w=area)


def filter_area(det, frac=0.35, span=6):
    """Causal sanity filter on region size (rule: a box side may not change by
    more than the scaling law allows inside one or two frames)."""
    for ci in range(len(det)):
        hist = []
        for f in range(NFRAMES):
            d = det[ci][f]
            if d is None:
                continue
            if len(hist) >= 3:
                med = float(np.median([a for _, a in hist[-span:]]))
                if d["area_w"] < frac * med or d["area_w"] > med / frac:
                    det[ci][f] = None
                    continue
            hist.append((f, d["area_w"]))
    return det


def detect_hinted(cams, pos, cache, half=110, rebuild=False):
    """Second detection pass: the search window is placed by the 3D estimate,
    so the region can never drift onto foliage and never chain its own error."""
    if cache and os.path.exists(cache) and not rebuild:
        with open(cache, "rb") as fh:
            det = pickle.load(fh)
        print("[det2] cache loaded", cache)
        return det
    det = [[None] * NFRAMES for _ in cams]
    for c in cams:                      # each pass replays the pose background
        c.mbg = None
    caps = [cv2.VideoCapture(c.video) for c in cams]
    for f in range(NFRAMES):
        for ci, cap in enumerate(caps):
            ok, img = cap.read()
            if not ok:
                continue
            small = cv2.resize(img, (WW, WH))
            cam = cams[ci]
            bg, sg, _vm = cam.bg_for(f, small)
            bl = 0 if cam.static[f] else 7
            pj = cam.project(f, pos[f])
            if pj is None or not (0 <= pj[0] < W and 0 <= pj[1] < H):
                continue
            uv = np.array([pj[0] / SC, pj[1] / SC])
            prev = det[ci][f - 1] if f > 0 else None
            d = find_region(small, bg, sg, uv, half, prev, bl)
            if d is None:
                d = find_region(small, bg, sg, uv, int(half * 1.7), None, bl)
            det[ci][f] = d
    for cap in caps:
        cap.release()
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump(det, fh)
    return det


def detect_all(cams, cache, rebuild=False):
    det = None
    if cache and os.path.exists(cache) and not rebuild:
        with open(cache, "rb") as fh:
            det = pickle.load(fh)
        print("[det] cache loaded", cache)
        return det
    det = [[None] * NFRAMES for _ in cams]
    for c in cams:                      # each pass replays the pose background
        c.mbg = None
    caps = [cv2.VideoCapture(c.video) for c in cams]
    # image-space state per camera: position, velocity, box size, miss counter
    st = [dict(uv=None, vel=np.zeros(2), prev=None, miss=0, box=None) for _ in cams]
    for f in range(NFRAMES):
        for ci, cap in enumerate(caps):
            ok, img = cap.read()
            if not ok:
                continue
            small = cv2.resize(img, (WW, WH))
            cam = cams[ci]
            if cam.static[min(f, len(cam.static) - 1)] and f > cam.bg_hi:
                continue
            bg, sg, _vm = cam.bg_for(f, small)
            bl = 0 if cam.static[f] else 7
            s = st[ci]
            if s["uv"] is None or s["miss"] > 25:          # (re)acquire
                d = find_region(small, bg, sg, None, None, None, bl)
                if d is not None:
                    s.update(uv=np.array([d["cx"] / SC, d["cy"] / SC]), vel=np.zeros(2),
                             prev=d, miss=0, box=d["bbox"])
            else:
                pred = s["uv"] + s["vel"]
                half = int(np.clip(1.4 * max(s["prev"]["w_w"], s["prev"]["h_w"]) + 14,
                                   34, 240))
                d = find_region(small, bg, sg, pred, half, s["prev"], bl)
                if d is None:                              # widen once
                    d = find_region(small, bg, sg, pred, int(half * 1.6), s["prev"], bl)
                if d is not None:
                    uvn = np.array([d["cx"] / SC, d["cy"] / SC])
                    s["vel"] = 0.6 * s["vel"] + 0.4 * (uvn - s["uv"])
                    s["uv"], s["prev"], s["miss"] = uvn, d, 0
                else:
                    s["miss"] += 1
                    s["uv"] = s["uv"] + s["vel"]
            det[ci][f] = d if (s["miss"] <= 2) else None
    for cap in caps:
        cap.release()
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump(det, fh)
    return det


# --------------------------------------------------------- triangulation ----
def triangulate(obs):
    A = np.zeros((2 * len(obs), 4))
    for k, (P, u, v) in enumerate(obs):
        A[2 * k] = u * P[2] - P[0]
        A[2 * k + 1] = v * P[2] - P[1]
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def tri_cov(obs, X3, pix_sigma=PIX_SIGMA):
    """3x3 covariance of the triangulated point from the reprojection Jacobian."""
    J = np.zeros((2 * len(obs), 3))
    r = np.zeros(2 * len(obs))
    for k, (P, u, v) in enumerate(obs):
        x = P @ np.append(X3, 1.0)
        if x[2] <= 1e-6:
            return np.eye(3) * 1e8
        pu, pv = x[0] / x[2], x[1] / x[2]
        sx = (P[0, :3] - pu * P[2, :3]) / x[2]
        sy = (P[1, :3] - pv * P[2, :3]) / x[2]
        J[2 * k] = sx
        J[2 * k + 1] = sy
        r[2 * k] = u - pu
        r[2 * k + 1] = v - pv
    JtJ = J.T @ J + np.eye(3) * 1e-9
    C = np.linalg.inv(JtJ) * pix_sigma ** 2
    return C


def reproj(obs, X3):
    out = []
    for P, u, v in obs:
        x = P @ np.append(X3, 1.0)
        out.append(1e4 if x[2] <= 1e-6 else np.hypot(x[0] / x[2] - u, x[1] / x[2] - v))
    return out


# ---------------------------------------------------------------- Kalman ---
class KF3:
    """Constant-velocity Kalman over 3D position (cm) and velocity (cm/frame)."""

    def __init__(self, p0, q=60.0):
        self.x = np.concatenate([np.asarray(p0, float), np.zeros(3)])
        self.P = np.diag(np.concatenate([np.full(3, 400.0 ** 2), np.full(3, 60.0 ** 2)]))
        self.F = np.block([[np.eye(3), np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])
        qm = np.array([[0.25, 0.5], [0.5, 1.0]]) * q
        self.Q = np.kron(qm, np.eye(3))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z, R):
        S = self.H @ self.P @ self.H.T + R
        d = z - self.H @ self.x
        d2 = float(d @ np.linalg.solve(S, d))          # Mahalanobis^2
        if d2 > 49.0:                                   # implausible jump -> ignore
            return self.x.copy(), d2, False
        if d2 > 16.0:                                   # borderline -> Huber weight
            R = R * (d2 / 16.0)
            S = self.H @ self.P @ self.H.T + R
        Kg = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ d
        self.P = (np.eye(6) - Kg @ self.H) @ self.P
        vmax = 110.0                                    # cm/frame  (~33 m/s, GT max ~32)
        sp = np.linalg.norm(self.x[3:])
        if sp > vmax:
            self.x[3:] *= vmax / sp
        return self.x.copy(), d2, True


class KFCA:
    """Constant-acceleration Kalman (p, v, a) in cm / cm per frame.

    The generator drives the truck with keyframes joined by cubic segments, so
    between observations the motion is locally quadratic.  A CV filter coasts in
    a straight line and loses the turn (the truck is often seen by one camera
    only for several seconds), a CA filter follows the curve.
    """

    def __init__(self, p0, jerk=2.0):
        self.x = np.concatenate([np.asarray(p0, float), np.zeros(6)])
        self.P = np.diag(np.concatenate([np.full(3, 400.0 ** 2),
                                         np.full(3, 40.0 ** 2),
                                         np.full(3, 4.0 ** 2)]))
        F1 = np.array([[1.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        self.F = np.kron(F1, np.eye(3))
        H1 = np.array([[1.0, 0.0, 0.0]])
        self.H = np.kron(H1, np.eye(3))
        # continuous white-jerk model, sampled with dt = 1 frame
        qm = np.array([[0.25, 0.5, 0.5], [0.5, 1.0, 1.0], [0.5, 1.0, 1.0]]) * jerk
        self.Q = np.kron(qm, np.eye(3))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z, R):
        S = self.H @ self.P @ self.H.T + R
        d = z - self.H @ self.x
        d2 = float(d @ np.linalg.solve(S, d))
        if d2 > 64.0:
            return self.x, d2, False
        if d2 > 16.0:
            R = R * (d2 / 16.0)
            S = self.H @ self.P @ self.H.T + R
        Kg = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ d
        self.P = (np.eye(9) - Kg @ self.H) @ self.P
        return self.x, d2, True


def robust_smooth(z, got, half=6, min_n=4, max_age=8):
    """Causal robust local-quadratic fit of the triangulated positions.

    The trajectory is piecewise cubic in frame time, i.e. locally a constant
    acceleration (a quadratic), so a quadratic fit over the last ~11 frames is
    the physically right smoother; IRLS rejects the occasional bad
    triangulation without latching the track onto it (no corners).
    Points older than `max_age` frames are dropped, otherwise a long detection
    gap (~2 s) makes the quadratic extrapolate and the track jumps.
    """
    out = np.full_like(z, np.nan)
    for f in np.where(got)[0]:
        sel = [i for i in range(max(0, f - half), f + 1)
               if got[i] and (f - i) <= max_age]
        if len(sel) < 2:
            out[f] = z[f]
            continue
        t = np.array(sel, float) - f
        if len(sel) < min_n:
            A = np.stack([np.ones_like(t), t], 1)          # linear fit
            coef, *_ = np.linalg.lstsq(A, z[sel], rcond=None)
            out[f] = coef[0]
            continue
        A = np.stack([np.ones_like(t), t, 0.5 * t * t], 1)
        w = np.ones(len(sel))
        for _ in range(3):
            Aw = A * w[:, None]
            coef, *_ = np.linalg.lstsq(Aw, z[sel] * w[:, None], rcond=None)
            res = np.linalg.norm(z[sel] - A @ coef, axis=1)
            s = max(1.4826 * np.median(res), 50.0)
            w = 1.0 / (1.0 + (res / (2.0 * s)) ** 2)
        out[f] = coef[0]
    return out


# ------------------------------------------------------------------ track --
def track(cams, det, verbose=False):
    n = len(cams)
    raw = np.zeros((NFRAMES, 3))
    rawok = np.zeros(NFRAMES, bool)
    for f in range(NFRAMES):
        obs = []
        for ci, cam in enumerate(cams):
            d = det[ci][f]
            if d is None:
                continue
            obs.append((cam.Proj[f], d["cx"], d["cy"]))
        if len(obs) < 2:
            continue
        best = None
        cur = list(obs)
        while len(cur) >= 2:                      # drop the worst camera until consistent
            X = triangulate(cur)
            if X is None or not np.isfinite(X).all():
                break
            e = np.array(reproj(cur, X))
            if e.max() < 60.0:
                best = X
                break
            if len(cur) == 2:
                break
            cur.pop(int(np.argmax(e)))
        if best is None:
            continue
        raw[f] = best
        rawok[f] = True
    zs = robust_smooth(raw, rawok)
    if verbose:
        print("[track] raw fixes=%d  smoothed=%d"
              % (rawok.sum(), np.isfinite(zs).all(1).sum()))

    kf = None
    pos = np.zeros((NFRAMES, 3))
    vel = np.zeros((NFRAMES, 3))
    got = np.zeros(NFRAMES, bool)
    for f in range(NFRAMES):
        if kf is not None:
            kf.predict()
        if np.isfinite(zs[f]).all():
            if kf is None:
                kf = KFCA(zs[f])
            else:
                before = kf.x.copy()
                kf.update(zs[f], np.eye(3) * 300.0 ** 2)
                d = kf.x[:3] - before[:3]
                nd = np.linalg.norm(d)
                if nd > 250.0:                       # converge, do not snap
                    kf.x[:3] = before[:3] + d * (250.0 / nd)
            got[f] = True
        if kf is not None:
            sp = np.linalg.norm(kf.x[3:6])
            if sp > 110.0:                           # 33 m/s, GT max ~32
                kf.x[3:6] *= 110.0 / sp
            ac = np.linalg.norm(kf.x[6:])
            if ac > 3.0:
                kf.x[6:] *= 3.0 / ac
            pos[f] = kf.x[:3]
            vel[f] = kf.x[3:6]
    f0 = int(np.argmax(got)) if got.any() else 0
    if got.any():
        pos[:f0] = pos[f0]
    return pos, vel, got, f0


def rts(pos, got, q=60.0):
    idx = np.where(got)[0]
    if len(idx) < 10:
        return pos.copy()
    kf = None
    xs, Ps, xp, Pp = [], [], [], []
    for i in idx:
        if kf is None:
            kf = KFCA(pos[i])
            xp.append(kf.x.copy())
            Pp.append(kf.P.copy())
        else:
            p, P = kf.predict()
            xp.append(p)
            Pp.append(P)
            kf.update(pos[i], np.eye(3) * 300.0 ** 2)[0]
        xs.append(kf.x.copy())
        Ps.append(kf.P.copy())
    xs = np.array(xs)
    out = xs[:, :3].copy()
    C = xs[-1].copy()
    for k in range(len(idx) - 2, -1, -1):
        G = Ps[k] @ kf.F.T @ np.linalg.pinv(Pp[k + 1])
        C = xs[k] + G @ (C - xp[k + 1])
        out[k] = C[:3]
    res = pos.copy()
    res[idx] = out
    return res


# --------------------------------------------------------------- metrics ---
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


# ------------------------------------------------------------- visualise ---
def _tile(img, w=640, h=360):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def visualize(cams, det, pos, gt, path, res_ok=60.0):
    """Diagnostic grid: image + segmentation mask + box + triangulation check.

    Per camera cell:
      magenta tint   - the motion mask that produced the region
      blue box/star  - the region that goes into the triangulation
      yellow x       - projection of the fused 3D estimate
      red +          - projection of the ground-truth 3D point
      header         - det/none, 2D residual to the estimate, USED/SKIP
    The residual is exactly what decides the camera's vote in triangulate(),
    so a camera tagged SKIP is the one the solver threw away.
    """
    for c in cams:                      # re-propagate the moving background
        c.mbg = None
    caps = [cv2.VideoCapture(c.video) for c in cams]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 1080))
    x0, x1 = -62000, 42000
    y0, y1 = -42000, 22000

    def to_map(p):
        return (int((p[1] - y0) / (y1 - y0) * 620),
                int((1 - (p[0] - x0) / (x1 - x0)) * 340) + 8)

    for f in range(NFRAMES):
        cells = []
        votes = 0
        for ci, cap in enumerate(caps):
            ok, img = cap.read()
            if not ok:
                img = np.zeros((H, W, 3), np.uint8)
            cam = cams[ci]
            small = cv2.resize(img, (WW, WH), interpolation=cv2.INTER_AREA)
            bg, sg, _vm = cam.bg_for(f, small)
            bl = 0 if cam.static[f] else 7
            tile = _tile(img)
            d = det[ci][f]

            # ---- segmentation mask, drawn only around the region / prediction
            if d is not None:
                cx, cy = d["cx"], d["cy"]
            else:
                ep0 = cam.project(f, pos[f])
                cx, cy = (ep0 if ep0 is not None else (W / 2, H / 2))
            wx, wy = cx / SC, cy / SC
            half = 110
            lx = int(np.clip(wx - half, 0, WW - 1))
            hx = int(np.clip(wx + half, 1, WW))
            ly = int(np.clip(wy - half, 0, WH - 1))
            hy = int(np.clip(wy + half, 1, WH))
            m = _mask(small[ly:hy, lx:hx], bg[ly:hy, lx:hx],
                      sg[ly:hy, lx:hx], 0 if cam.static[f] else 7)
            if m.any():
                s2 = int(SC)
                full = np.zeros((H, W), np.uint8)
                full[ly * s2:hy * s2, lx * s2:hx * s2] = cv2.resize(
                    m, ((hx - lx) * s2, (hy - ly) * s2),
                    interpolation=cv2.INTER_NEAREST)
                mmt = cv2.resize(full, (640, 360), interpolation=cv2.INTER_NEAREST)
                sel = mmt.astype(bool)
                tint = tile.copy()
                tint[sel] = (0.45 * tint[sel] + np.array([255, 0, 255]) * 0.55)
                tile = tint
                cnts, _ = cv2.findContours(mmt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(tile, cnts, -1, (255, 0, 255), 1)

            if d is not None:
                b = [int(v / 3) for v in d["bbox"]]
                cv2.rectangle(tile, (b[0], b[1]), (b[2], b[3]), (255, 160, 0), 2)
                cv2.drawMarker(tile, (int(d["cx"] / 3), int(d["cy"] / 3)),
                               (255, 160, 0), cv2.MARKER_STAR, 12, 1)
            pj = cam.project(f, gt[f])
            if pj is not None:
                cv2.drawMarker(tile, (int(pj[0] / 3), int(pj[1] / 3)), (0, 0, 255),
                               cv2.MARKER_CROSS, 16, 2)
            ep = cam.project(f, pos[f])
            if ep is not None:
                cv2.drawMarker(tile, (int(ep[0] / 3), int(ep[1] / 3)), (0, 255, 255),
                               cv2.MARKER_TILTED_CROSS, 16, 2)

            res = -1.0
            if d is not None and ep is not None:
                res = float(np.hypot(d["cx"] - ep[0], d["cy"] - ep[1]))
            used = res >= 0 and res < res_ok
            votes += int(used)
            head = "cam%d f%03d  %s" % (cam.cid, f, "no-det" if d is None else "det")
            if d is not None:
                head += "  res=%3.0fpx  %s" % (res, "USED" if used else "SKIP")
                if not used:
                    head += " (inconsistent)"
            col = (0, 255, 0) if used else (120, 200, 255)
            cv2.putText(tile, head, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(tile, head, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
            if not cam.usable(f):
                cv2.putText(tile, "moving cam: bg model", (6, 344),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            cells.append(tile)

        # ---------------------------------------------------------------- map
        mp = np.full((360, 640, 3), 25, np.uint8)
        cv2.putText(mp, "top view  GT=red   est=yellow   cams used=%d" % votes,
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        gpts = np.array([to_map(p) for p in gt], np.int32)
        cv2.polylines(mp, [gpts], False, (60, 60, 160), 1)
        epts = np.array([to_map(p) for p in pos[:f + 1]], np.int32)
        if len(epts) > 1:
            cv2.polylines(mp, [epts], False, (0, 220, 255), 1)
        for c in cams:
            pc = to_map(c.C[f])
            cv2.drawMarker(mp, pc, (255, 120, 0), cv2.MARKER_TRIANGLE_UP, 12, 2)
            cv2.putText(mp, str(c.cid), (pc[0] + 6, pc[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)
        cv2.drawMarker(mp, to_map(gt[f]), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.drawMarker(mp, to_map(pos[f]), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
        err = float(np.linalg.norm(pos[f] - gt[f])) / 100.0
        cv2.putText(mp, "err=%.1f m" % err, (8, 344),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cells.append(mp)
        grid = np.vstack([np.hstack(cells[0:2]), np.hstack(cells[2:4]),
                          np.hstack(cells[4:6])])
        vw.write(grid)
    vw.release()
    for cap in caps:
        cap.release()


# ------------------------------------------------------------------- main --
def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cams = [Camera(c) for c in CAMS]
    for c in cams:
        c.build_background()
    gt = load_gt()
    print("[bg] %.1fs; cam5 static until frame %d" % (time.time() - t0, cams[4].bg_hi))

    det = detect_all(cams, os.path.join(OUT, "det_cache_v3.pkl"),
                     rebuild="--nocache" in sys.argv)
    nreg = sum(det[ci][f] is not None for ci in range(len(cams)) for f in range(NFRAMES))
    print("[det] %d/%d camera-frames with a region, %.1fs"
          % (nreg, len(cams) * NFRAMES, time.time() - t0))

    pos, vel, got, f0 = track(cams, det, verbose=True)
    sm = rts(pos, got)
    for name, tr in (("causal", pos), ("rts", sm)):
        e = tr - gt
        print("[%s] fixes=%d RMSE=(%.1f, %.1f, %.1f) cm | 3D=%.1f m | median=%.1f m"
              % (name, got.sum(), *np.sqrt((e ** 2).mean(0)),
                 np.linalg.norm(np.sqrt((e ** 2).mean(0))) / 100,
                 np.median(np.linalg.norm(e, axis=1)) / 100))

    # ---- pass 2: re-detect with the window placed by the 3D estimate --------
    det2 = detect_hinted(cams, sm, os.path.join(OUT, "det_cache_v3b.pkl"),
                         rebuild="--nocache" in sys.argv)
    nreg = sum(det2[ci][f] is not None for ci in range(len(cams)) for f in range(NFRAMES))
    for ci in range(len(cams)):
        ins = tot = 0
        for f in range(NFRAMES):
            p = cams[ci].project(f, gt[f])
            d = det2[ci][f]
            if p is None or not (0 <= p[0] < W and 0 <= p[1] < H):
                continue
            tot += 1
            ins += bool(d is not None and d["bbox"][0] <= p[0] <= d["bbox"][2]
                        and d["bbox"][1] <= p[1] <= d["bbox"][3])
        print("[det2] cam%d regions=%d  GT-in-box %d/%d = %.0f%%"
              % (cams[ci].cid, sum(det2[ci][f] is not None for f in range(NFRAMES)),
                 ins, tot, 100.0 * ins / max(tot, 1)))
    print("[det2] %d/%d camera-frames with a region, %.1fs"
          % (nreg, len(cams) * NFRAMES, time.time() - t0))

    pos2, vel2, got2, _ = track(cams, det2)
    sm2 = rts(pos2, got2)
    for name, tr in (("causal2", pos2), ("rts2", sm2)):
        e = tr - gt
        print("[%s] fixes=%d RMSE=(%.1f, %.1f, %.1f) cm | 3D=%.1f m | median=%.1f m"
              % (name, got2.sum(), *np.sqrt((e ** 2).mean(0)),
                 np.linalg.norm(np.sqrt((e ** 2).mean(0))) / 100,
                 np.median(np.linalg.norm(e, axis=1)) / 100))
    if got2.sum() > got.sum():
        sm, det = pos2, det2      # causal track is the deliverable (online task)

    out_csv = os.path.join(OUT, "pred_track.csv")
    with open(out_csv, "w", encoding="utf-8") as fh:
        fh.write("Frame;Loc_X;Loc_Y;Loc_Z\n")
        for f in range(NFRAMES):
            fh.write("%d;%.4f;%.4f;%.4f\n" % (f, sm[f, 0], sm[f, 1], sm[f, 2]))
    print("[out]", out_csv)

    if "--viz" in sys.argv:
        visualize(cams, det, sm, gt, os.path.join(OUT, "grid.mp4"))
        print("[viz] grid.mp4")
    print("[done] %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

