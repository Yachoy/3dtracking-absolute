# -*- coding: utf-8 -*-
"""
boxFrustum.py - object position from the intersection of the box frustums,
built the way the user draws it in 2D.

For every camera the bounding box is a pyramid; its cross-section by the ground
plane is a quadrilateral (a "triangle" when the box is narrow).  The algorithm:

  1. project each box onto the horizontal plane z = z_ref -> a convex quad;
  2. clip the quads against each other (Sutherland-Hodgman) -> the intersection
     polygon (the blue triangle of the sketch);
  3. the area centroid of that polygon is the horizontal position (green dot);
  4. at that (x, y) scan z and keep the interval where the point projects inside
     EVERY box; the middle of that interval lifts the point by half the height
     of the intersection solid.

No volume sampling, no dependence on a single pixel: the lateral position comes
from the box walls, the height from the vertical extent of the intersection.

Detections: the cached boxes of first.py (out/det_cache_v3b.pkl).
Feed-forward: measurement -> constant-acceleration 3D Kalman -> CSV + video.

Usage:
    python boxFrustum.py            # metrics
    python boxFrustum.py --viz      # + out/frustum.mp4
"""
import os
import sys
import time
import pickle
import numpy as np
import cv2

import first as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
NFRAMES = F.NFRAMES
CAMS = F.CAMS
DET = os.path.join(OUT, "det_cache_v3b.pkl")

FAR = 2.0e6          # cm, for rays that do not hit the plane (above the horizon)
Z_STEP = 20.0        # cm, vertical scan step
MARGIN = 2.0         # px tolerance for "inside the box"


# ------------------------------------------------------------ 2D geometry ---
def _area2(poly):
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _centroid(poly):
    a = _area2(poly)
    if abs(a) < 1e-9:
        return np.mean(np.asarray(poly, float), axis=0), 0.0
    cx = cy = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        cr = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cr
        cy += (y1 + y2) * cr
    return np.array([cx / (6 * a), cy / (6 * a)]), abs(a)


def _inside(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0


def _isect(s, e, a, b):
    x1, y1, x2, y2 = s[0], s[1], e[0], e[1]
    x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return list(e)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]


def clip_poly(subject, clipper):
    """Sutherland-Hodgman: subject clipped by a convex clipper."""
    if not subject or len(clipper) < 3:
        return []
    cl = list(clipper)
    if _area2(cl) < 0:
        cl.reverse()
    out = list(subject)
    for i in range(len(cl)):
        a, b = cl[i], cl[(i + 1) % len(cl)]
        inp = out
        out = []
        if not inp:
            return []
        s = inp[-1]
        for e in inp:
            if _inside(e, a, b):
                if not _inside(s, a, b):
                    out.append(_isect(s, e, a, b))
                out.append(e)
            elif _inside(s, a, b):
                out.append(_isect(s, e, a, b))
            s = e
    return out


def floor_tri(cam, f, bbox, z_plane, t_max=None):
    """The box pyramid as a TRIANGLE on the ground plane.

    Apex = the camera's own ground point, the two sides = the rays through the
    left and right box edges (taken at the bottom row of the box, the most
    stable part near the object's ground contact).  The far ends are clipped at
    t_max: at 100-600 m the horizontal rays are ill-conditioned, half a pixel
    tilts the hit point by kilometres.
    """
    x0, y0, x1, y1 = bbox
    Kinv = np.linalg.inv(cam.K)
    M = cam._M(f)
    C = cam.C[f]
    tmax = FAR if t_max is None else float(t_max)
    tmin = 0.0 if t_max is None else 0.45 * float(t_max)
    pts = []
    for u in (x0, x1):
        d = M.T @ (Kinv @ np.array([u, y1, 1.0]))
        n = np.linalg.norm(d)
        if d[2] > 1e-4 * n:
            t = (z_plane - C[2]) / d[2]
        else:
            t = tmax
        t = min(max(t, tmin), tmax)
        p = C + t * d
        pts.append([p[0], p[1]])
    return [[C[0], C[1]], pts[0], pts[1]]


def z_interval(cams, det, f, x, y, ids=None):
    """Vertical interval at (x,y) where the point projects inside every box."""
    zs = np.arange(-800.0, 4000.0, Z_STEP)
    P = np.stack([np.full_like(zs, x), np.full_like(zs, y), zs], 1)
    ok = np.ones(len(zs), bool)
    for ci, cam in enumerate(cams):
        if ids is not None and ci not in ids:
            continue
        d = det[ci][f]
        if d is None:
            continue
        b = d["bbox"]
        Xc = cam.Proj[f] @ np.hstack([P, np.ones((len(P), 1))]).T
        z = Xc[2]
        zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u, v = Xc[0] / zz, Xc[1] / zz
        ok &= ((z > 0) & (u >= b[0] - MARGIN) & (u <= b[2] + MARGIN) &
               (v >= b[1] - MARGIN) & (v <= b[3] + MARGIN))
    if not ok.any():
        return None
    return float(zs[ok].min()), float(zs[ok].max())


def ray_plane(cam, f, uv, z_plane):
    Kinv = np.linalg.inv(cam.K)
    d = cam._M(f).T @ (Kinv @ np.array([uv[0], uv[1], 1.0]))
    if abs(d[2]) < 1e-9:
        return None
    t = (z_plane - cam.C[f][2]) / d[2]
    if t <= 0:
        return None
    return cam.C[f] + t * d


def measure(cams, det, f, z_plane=350.0, pred3=None):
    """Pick the camera combination whose intersection is the most compact.

    Every pair/triple of boxes is tried, the wedges are clipped and the result
    is scored by the perimeter of its oriented bounding box (w + h).  A narrow
    "cone" has a big h, a fat wedge a big w, so the minimum is the best
    conditioned solution.  A single camera falls back to ray x plane.
    """
    ids = [ci for ci in range(len(cams)) if det[ci][f] is not None]
    if len(ids) == 1 and pred3 is not None:
        ci = ids[0]                      # last best modification: one ray + plane
        d = det[ci][f]
        X = ray_plane(cams[ci], f, (d["cx"], d["cy"]), float(pred3[2]))
        if X is None:
            return None
        return dict(X=X, poly=None, area=0.0, z0=X[2], z1=X[2],
                    half_h=60.0, ncam=1, single=True, score=0.0)
    if len(ids) < 2:
        return None

    t_max = None
    if pred3 is not None:
        d = [np.linalg.norm(pred3 - cams[ci].C[f]) for ci in ids]
        t_max = float(1.5 * np.median(d))
    tris = {ci: floor_tri(cams[ci], f, det[ci][f]["bbox"], z_plane, t_max)
            for ci in ids}

    combos = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            combos.append((ids[a], ids[b]))
    if len(ids) >= 3:
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                for c in range(b + 1, len(ids)):
                    combos.append((ids[a], ids[b], ids[c]))

    best = None
    for combo in combos:
        poly = tris[combo[0]]
        bad = False
        for ci in combo[1:]:
            poly = clip_poly(poly, tris[ci])
            if not poly or len(poly) < 3:
                bad = True
                break
        if bad:
            continue
        ctr, area = _centroid(poly)
        if area < 400.0:                       # cm^2: slivers are useless
            continue
        _rect = cv2.minAreaRect(np.array(poly, np.float32))
        rw, rh = _rect[1]
        score = float(rw + rh)
        if best is None or score < best[0]:
            best = (score, poly, ctr, area, list(combo))

    if best is None:
        return None
    score, poly, ctr, area, combo = best
    zi = z_interval(cams, det, f, ctr[0], ctr[1], combo)
    if zi is None:
        return None
    z0, z1 = zi
    return dict(X=np.array([ctr[0], ctr[1], 0.5 * (z0 + z1)]),
                poly=np.array(poly), area=area, z0=z0, z1=z1,
                half_h=0.5 * (z1 - z0), ncam=len(combo), single=False,
                combo=combo, score=score)


def wall_point(centre, target, poly):
    """First crossing of the segment centre->target with the polygon boundary."""
    c = np.asarray(centre, float)
    t = np.asarray(target, float)
    best = None
    for i in range(len(poly)):
        a = np.asarray(poly[i], float)
        b = np.asarray(poly[(i + 1) % len(poly)], float)
        e = b - a
        d = t - c
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-12:
            continue
        w = a - c
        u = (w[0] * e[1] - w[1] * e[0]) / den          # along c->t
        v = (w[0] * d[1] - w[1] * d[0]) / den          # along a->b
        if -1e-9 <= u <= 1.0 + 1e-9 and -1e-9 <= v <= 1.0 + 1e-9:
            if best is None or u < best:
                best = u
    if best is None:
        return c
    return c + best * (t - c)


def ema_angle(prev, new, a):
    d = np.angle(np.exp(1j * (new - prev)))
    return prev + a * d


# ---------------------------------------------------------------- tracker ---
def cast_clip(xy, theta, dist, polys, step=20.0):
    """Cast from xy along theta and stop at the last point inside ANY polygon."""
    if dist <= 0:
        return None
    d = np.array([np.cos(theta), np.sin(theta)])
    best = None
    t = 0.0
    while t <= dist:
        p = xy + t * d
        inside = False
        for poly in polys:
            if poly is not None and len(poly) >= 3:
                if cv2.pointPolygonTest(np.asarray(poly, np.float32),
                                        (float(p[0]), float(p[1])), False) >= 0:
                    inside = True
                    break
        if inside:
            best = p
        elif best is not None:
            break
        t += step
    return best


def track(cams, det, gamma=2.0, verbose=True,
          alpha_pos=0.7, alpha_v=0.4, alpha_w=0.35, alpha_z=0.3, beta_c=0.8,
          use_global_kf=True):
    """Two-level tracker.

    Local (polar, relative to the object itself): speed v and turn rate w are
    smoothed by EMA and used as a constant-velocity Kalman prediction of the
    heading THETA = theta + w.  The ray is cast from the previous position along
    THETA for gamma*v, and may stay inside ANY one of the frustum triangles (not
    necessarily all of them), so the motion is extended as far as the geometry
    allows.

    Global: the resulting point (blended with the intersection centroid) feeds a
    constant-acceleration Kalman in world coordinates.
    """
    pos = np.zeros((NFRAMES, 3))
    hits = np.zeros(NFRAMES, bool)
    halfh = np.zeros(NFRAMES)
    area = np.zeros(NFRAMES)
    score = np.zeros(NFRAMES)
    ncam = np.zeros(NFRAMES, int)
    speed = np.zeros(NFRAMES)
    z_ref = 350.0
    xy = None
    v = 0.0
    w = 0.0
    th = 0.0
    zf = None
    kf = None

    for f in range(NFRAMES):
        m = measure(cams, det, f, z_ref,
                    None if xy is None else np.array([xy[0], xy[1], z_ref]))
        if m is not None:
            C = np.asarray(m["X"][:2], float)
            poly = m.get("poly")
            if xy is None:
                xy = C.copy()
                meas = C
            else:
                th_pred = th + w                         # polar prediction
                rays = [poly] if poly is not None else []
                q = cast_clip(xy, th_pred, gamma * max(v, 1.0), rays)
                if q is None:
                    q = C
                meas = (1 - beta_c) * q + beta_c * C
                new = xy + alpha_pos * (meas - xy)
                dxy = new - xy
                sp = float(np.linalg.norm(dxy))
                if sp > 1e-3:
                    th_meas = float(np.arctan2(dxy[1], dxy[0]))
                    v = (1 - alpha_v) * v + alpha_v * sp
                    w = (1 - alpha_w) * w + alpha_w * np.angle(np.exp(1j * (th_meas - th)))
                    th = th_meas
                xy = new
            zf = m["X"][2] if zf is None else zf + alpha_z * (m["X"][2] - zf)
            z_ref = 0.8 * z_ref + 0.2 * float(m["X"][2])
            hits[f] = True
            halfh[f] = m["half_h"]
            area[f] = m["area"]
            score[f] = m["score"]
            ncam[f] = m["ncam"]
        else:
            if xy is not None:
                th = th + w                          # extrapolate in polar space
                xy = xy + v * np.array([np.cos(th), np.sin(th)])
        if xy is None:
            continue
        z = zf if zf is not None else z_ref
        if use_global_kf:
            if kf is None:
                kf = F.KFCA(np.array([xy[0], xy[1], z]))
            else:
                kf.predict()
                kf.update(np.array([xy[0], xy[1], z]), np.eye(3) * 400.0 ** 2)
                kf.x[:3] = np.array([xy[0], xy[1], z])      # follow the geometry
                pos[f] = kf.x[:3]
                speed[f] = np.linalg.norm(kf.x[3:6])
                continue
        pos[f] = np.array([xy[0], xy[1], z])
        speed[f] = v
    if verbose:
        print("[frustum] measured %d/%d  gamma=%.2f  median speed %.1f cm/frame"
              % (hits.sum(), NFRAMES, gamma,
                 np.median(speed[hits]) if hits.any() else 0))
    return pos, None, hits, halfh


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cams = [F.Camera(c) for c in CAMS]
    gt = F.load_gt()
    with open(DET, "rb") as fh:
        det = pickle.load(fh)
    pos, vel, hits, halfh = track(cams, det)
    e = pos - gt
    rmse = np.sqrt((e ** 2).mean(0))
    print("[3d] RMSE=(%.1f,%.1f,%.1f) cm | 3D=%.1f m | median=%.1f m | p90=%.1f m"
          % (*rmse, np.linalg.norm(rmse) / 100,
             np.median(np.linalg.norm(e, axis=1)) / 100,
             np.percentile(np.linalg.norm(e, axis=1), 90) / 100))
    out_csv = os.path.join(OUT, "frustum_pred.csv")
    with open(out_csv, "w", encoding="utf-8") as fh:
        fh.write("Frame;Loc_X;Loc_Y;Loc_Z\n")
        for f in range(NFRAMES):
            fh.write("%d;%.4f;%.4f;%.4f\n" % (f, pos[f, 0], pos[f, 1], pos[f, 2]))
    print("[out]", out_csv, "%.1fs" % (time.time() - t0))
    if "--viz" in sys.argv:
        render(cams, det, pos, gt, halfh, os.path.join(OUT, "frustum.mp4"))


def render(cams, det, pos, gt, halfh, path):
    x0, x1, y0, y1 = -62000, 42000, -42000, 22000

    def to_map(p):
        return (int((p[1] - y0) / (y1 - y0) * 620),
                int((1 - (p[0] - x0) / (x1 - x0)) * 340) + 8)

    caps = [cv2.VideoCapture(c.video) for c in cams]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1920, 720))
    nmeas = 0
    for f in range(NFRAMES):
        cells = []
        zp = float(pos[f][2]) if f else 350.0
        m = measure(cams, det, f, zp, pos[f])
        poly = None
        if m is not None and m.get("poly") is not None:
            poly = np.array([to_map((p[0], p[1], 0)) for p in m["poly"]], np.int32)
        nmeas += m is not None
        for ci, cap in enumerate(caps):
            ok, img = cap.read()
            if not ok:
                img = np.zeros((F.H, F.W, 3), np.uint8)
            tile = cv2.resize(img, (640, 360))
            d = det[ci][f]
            if d is not None:
                b = [int(v / 3) for v in d["bbox"]]
                cv2.rectangle(tile, (b[0], b[1]), (b[2], b[3]), (255, 160, 0), 2)
            if m is not None and m.get("poly") is not None:
                Q = np.hstack([m["poly"], np.full((len(m["poly"]), 1), m["X"][2]),
                               np.ones((len(m["poly"]), 1))])
                Xc = cams[ci].Proj[f] @ Q.T
                zz = np.where(np.abs(Xc[2]) < 1e-6, 1e-6, Xc[2])
                pts = np.stack([(Xc[0] / zz) / 3, (Xc[1] / zz) / 3], 1).astype(np.int32)
                cv2.polylines(tile, [pts], True, (255, 0, 255), 2)
            pj = cams[ci].project(f, gt[f])
            if pj is not None:
                cv2.drawMarker(tile, (int(pj[0] / 3), int(pj[1] / 3)), (0, 0, 255),
                               cv2.MARKER_CROSS, 16, 2)
            ep = cams[ci].project(f, pos[f])
            if ep is not None:
                cv2.drawMarker(tile, (int(ep[0] / 3), int(ep[1] / 3)), (0, 255, 255),
                               cv2.MARKER_TILTED_CROSS, 16, 2)
            head = "cam%d f%03d bbox=%s" % (cams[ci].cid, f,
                                            "no" if d is None else "yes")
            cv2.putText(tile, head, (6, 18), 0, 0.45, (0, 0, 0), 3)
            cv2.putText(tile, head, (6, 18), 0, 0.45, (255, 255, 255), 1)
            cells.append(tile)

        mp = np.full((360, 640, 3), 25, np.uint8)
        cv2.putText(mp, "top view  GT=red  est=yellow  halfH=%.0f cm  area=%.0f m2"
                    % (halfh[f], 0 if m is None else m["area"] / 1e4),
                    (8, 20), 0, 0.42, (200, 200, 200), 1)
        for ci in range(len(cams)):
            if det[ci][f] is None:
                continue
            q = floor_tri(cams[ci], f, det[ci][f]["bbox"], zp,
                          float(np.linalg.norm(pos[f] - cams[ci].C[f]) * 1.5))
            pts = np.array([to_map((p[0], p[1], 0)) for p in q], np.int32)
            cv2.polylines(mp, [pts], True, (90, 90, 90), 1)
        if poly is not None:
            cv2.fillPoly(mp, [poly], (60, 30, 90))
            cv2.polylines(mp, [poly], True, (255, 0, 255), 2)
            c = to_map((m["X"][0], m["X"][1], 0))
            cv2.drawMarker(mp, c, (0, 255, 0), cv2.MARKER_STAR, 12, 2)
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
        vw.write(np.vstack([np.hstack(cells[0:3]), np.hstack(cells[3:6])]))
    vw.release()
    for cap in caps:
        cap.release()
    print("[viz] %s  (frames with a polygon: %d/%d)" % (path, nmeas, NFRAMES))


if __name__ == "__main__":
    main()
