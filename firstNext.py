# -*- coding: utf-8 -*-
"""
firstNext.py - first.py's tracker, plus the single-camera fix.

The algorithm is deliberately IDENTICAL to first.py:
  * two detection passes (the second searches a crop placed by the 3D estimate),
    both cached on disk so an experiment starts in a second;
  * DLT triangulation with iterative rejection of the worst camera;
  * causal robust local-quadratic smoothing of the triangulated positions;
  * constant-acceleration Kalman in world coordinates.
The ONLY addition: when the smoothed measurement is missing but exactly one
camera still has a box, the measurement becomes the intersection of that camera
ray with the horizontal plane z = (last triangulated height).  That is the case
the user saw as "the yellow cross runs away from its own box": without it the
filter just extrapolates and its projection leaves the box.

Usage:
    python firstNext.py             # track + metrics
    python firstNext.py --viz       # + out/firstnext.mp4
    python firstNext.py --nocache   # rebuild the box caches
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

DET1 = os.path.join(OUT, "det_bbox.pkl")
DET2 = os.path.join(OUT, "det_bbox2.pkl")
BGC = os.path.join(OUT, "bg_cache.npz")


# ------------------------------------------------------------------ caches --
def get_cameras(rebuild=False):
    cams = [F.Camera(c) for c in CAMS]
    if os.path.exists(BGC) and not rebuild:
        z = np.load(BGC)
        for i, c in enumerate(cams):
            c.bg, c.sigma, c.sig_max = z["bg%d" % i], z["sigma%d" % i], z["sigmax%d" % i]
        print("[bg] cache")
    else:
        for c in cams:
            c.build_background()
        np.savez_compressed(BGC, **{
            **{"bg%d" % i: c.bg for i, c in enumerate(cams)},
            **{"sigma%d" % i: c.sigma for i, c in enumerate(cams)},
            **{"sigmax%d" % i: c.sig_max for i, c in enumerate(cams)}})
        print("[bg] built + cached")
    return cams


def _detect(cams, hint=None, cache=None, rebuild=False):
    """One detection pass. hint = per-camera, per-frame predicted pixel (or None)."""
    if cache and os.path.exists(cache) and not rebuild:
        with open(cache, "rb") as fh:
            det = pickle.load(fh)
        print("[det] cache", os.path.basename(cache))
        return det
    det = [[None] * NFRAMES for _ in cams]
    for c in cams:
        c.mbg = None
    caps = [cv2.VideoCapture(c.video) for c in cams]
    st = [dict(uv=None, vel=np.zeros(2), prev=None, miss=0) for _ in cams]
    for f in range(NFRAMES):
        for ci, cap in enumerate(caps):
            ok, img = cap.read()
            if not ok:
                continue
            cam = cams[ci]
            if not cam.static[min(f, len(cam.static) - 1)]:
                continue                      # moving camera: pose != render
            small = cv2.resize(img, (F.WW, F.WH))
            bg, sg, _ = cam.bg_for(f, small)
            s = st[ci]
            d = None
            if hint is not None and hint[ci][f] is not None:
                p = hint[ci][f]
                uv = np.array([p[0] / F.SC, p[1] / F.SC])
                d = F.find_region(small, bg, sg, uv, 110,
                                  det[ci][f - 1] if f > 0 else None, 0)
                if d is None:
                    d = F.find_region(small, bg, sg, uv, 190, None, 0)
            elif s["uv"] is None or s["miss"] > 25:
                d = F.find_region(small, bg, sg, None, None, None, 0)
                if d is not None:
                    s.update(uv=np.array([d["cx"] / F.SC, d["cy"] / F.SC]),
                             vel=np.zeros(2), prev=d, miss=0)
            else:
                pred = s["uv"] + s["vel"]
                h = int(np.clip(1.4 * max(s["prev"]["w_w"], s["prev"]["h_w"]) + 14, 34, 240))
                d = F.find_region(small, bg, sg, pred, h, s["prev"], 0)
                if d is None:
                    d = F.find_region(small, bg, sg, pred, int(h * 1.6), s["prev"], 0)
                if d is not None:
                    uvn = np.array([d["cx"] / F.SC, d["cy"] / F.SC])
                    s["vel"] = 0.6 * s["vel"] + 0.4 * (uvn - s["uv"])
                    s["uv"], s["prev"], s["miss"] = uvn, d, 0
                else:
                    s["miss"] += 1
                    s["uv"] = s["uv"] + s["vel"]
            if hint is None:
                det[ci][f] = d if s["miss"] <= 2 else None
            else:
                det[ci][f] = d
    for cap in caps:
        cap.release()
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump(det, fh)
    return det


def hints_from(cams, pos):
    h = [[None] * NFRAMES for _ in cams]
    for f in range(NFRAMES):
        for ci, cam in enumerate(cams):
            p = cam.project(f, pos[f])
            if p is not None and 0 <= p[0] < F.W and 0 <= p[1] < F.H:
                h[ci][f] = p
    return h


# ------------------------------------------------------- single-camera fix --
def ray_plane(cam, f, uv, z_plane):
    Kinv = np.linalg.inv(cam.K)
    v = Kinv @ np.array([uv[0], uv[1], 1.0])
    d = cam._M(f).T @ v
    if abs(d[2]) < 1e-6:
        return None
    t = (z_plane - cam.C[f][2]) / d[2]
    if t <= 0:
        return None
    return cam.C[f] + t * d


# ------------------------------------------------------------------ track --
def track(cams, det, use_ray=True, verbose=True):
    raw = np.zeros((NFRAMES, 3))
    rawok = np.zeros(NFRAMES, bool)
    for f in range(NFRAMES):
        obs = [(cam.Proj[f], det[ci][f]["cx"], det[ci][f]["cy"])
               for ci, cam in enumerate(cams) if det[ci][f] is not None]
        if len(obs) < 2:
            continue
        cur = list(obs)
        while len(cur) >= 2:
            X = F.triangulate(cur)
            if X is None or not np.isfinite(X).all():
                break
            e = np.array(F.reproj(cur, X))
            if e.max() < 60.0:
                raw[f] = X
                rawok[f] = True
                break
            if len(cur) == 2:
                break
            cur.pop(int(np.argmax(e)))

    zs = F.robust_smooth(raw, rawok)

    kf = None
    pos = np.zeros((NFRAMES, 3))
    vel = np.zeros((NFRAMES, 3))
    hits = np.zeros(NFRAMES, bool)
    ray = np.zeros(NFRAMES, bool)
    z_ref = None
    for f in range(NFRAMES):
        if kf is not None:
            kf.predict()
        z = zs[f] if np.isfinite(zs[f]).all() else None
        r_noise = 300.0
        if z is None and use_ray and kf is not None:
            seen = [ci for ci in range(len(cams)) if det[ci][f] is not None]
            if len(seen) == 1:
                ci = seen[0]
                d = det[ci][f]
                zp = float(kf.x[2]) if z_ref is None else z_ref
                Xp = ray_plane(cams[ci], f, (d["cx"], d["cy"]), zp)
                if Xp is not None and np.linalg.norm(Xp - kf.x[:3]) < 3000:
                    z = Xp
                    r_noise = 600.0
                    ray[f] = True
        if z is not None:
            if kf is None:
                kf = F.KFCA(z)
            else:
                before = kf.x[:3].copy()
                _, d2, ok = kf.update(z, np.eye(3) * r_noise ** 2)
                if ok:
                    dd = kf.x[:3] - before
                    if np.linalg.norm(dd) > 250.0:
                        kf.x[:3] = before + dd * (250.0 / np.linalg.norm(dd))
                    hits[f] = True
                    if not ray[f]:
                        z_ref = float(raw[f][2])
        if kf is not None:
            sp = np.linalg.norm(kf.x[3:6])
            if sp > 110.0:
                kf.x[3:6] *= 110.0 / sp
            ac = np.linalg.norm(kf.x[6:])
            if ac > 3.0:
                kf.x[6:] *= 3.0 / ac
            pos[f] = kf.x[:3]
            vel[f] = kf.x[3:6]
    if verbose:
        print("[track] fixes %d/%d (ray1 %d), raw triangulations %d"
              % (hits.sum(), NFRAMES, ray.sum(), rawok.sum()))
    return pos, vel, hits


# ------------------------------------------------------------- consistency --
def consistency(cams, det, pos):
    inside = total = 0
    worst = []
    for f in range(NFRAMES):
        for ci, cam in enumerate(cams):
            d = det[ci][f]
            if d is None:
                continue
            p = cam.project(f, pos[f])
            if p is None:
                continue
            total += 1
            ok = (d["bbox"][0] <= p[0] <= d["bbox"][2] and
                  d["bbox"][1] <= p[1] <= d["bbox"][3])
            inside += ok
            if not ok:
                worst.append((f, cam.cid, float(np.hypot(p[0] - d["cx"], p[1] - d["cy"]))))
    worst.sort(key=lambda x: -x[2])
    return inside, total, worst[:5]


# -------------------------------------------------------------------- main --
def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rebuild = "--nocache" in sys.argv
    cams = get_cameras(rebuild)
    gt = F.load_gt()

    # the SAME detections first.py ends up with, so the comparison is honest
    same = os.path.join(OUT, "det_cache_v3b.pkl")
    if os.path.exists(same) and not rebuild:
        with open(same, "rb") as fh:
            det2 = pickle.load(fh)
        print("[det] first.py cache reused:", os.path.basename(same))
    else:
        det1 = _detect(cams, None, DET1, rebuild)
        pos1, _, _ = track(cams, det1, use_ray=False, verbose=False)
        det2 = _detect(cams, hints_from(cams, pos1), DET2, rebuild)
    n = sum(det2[ci][f] is not None for ci in range(len(cams)) for f in range(NFRAMES))
    print("[det] pass2: %d/%d camera-frames with a box (%.1fs)"
          % (n, len(cams) * NFRAMES, time.time() - t0))

    posA, _, hitA = track(cams, det2, use_ray=False)
    posB, _, hitB = track(cams, det2, use_ray=True)
    for name, tr, hit in (("no-ray (== first.py)", posA, hitA),
                          ("with ray1 fix", posB, hitB)):
        e = tr - gt
        rmse = np.sqrt((e ** 2).mean(0))
        ins, tot, worst = consistency(cams, det2, tr)
        print("[%s] fixes=%d RMSE=(%.1f,%.1f,%.1f) cm | 3D=%.1f m | median=%.1f m"
              " | in-box %.1f%%"
              % (name, hit.sum(), *rmse, np.linalg.norm(rmse) / 100,
                 np.median(np.linalg.norm(e, axis=1)) / 100, 100.0 * ins / max(tot, 1)))
        if name.startswith("with"):
            print("   worst in-box misses:", worst)

    out_csv = os.path.join(OUT, "firstnext_pred.csv")
    with open(out_csv, "w", encoding="utf-8") as fh:
        fh.write("Frame;Loc_X;Loc_Y;Loc_Z\n")
        for f in range(NFRAMES):
            fh.write("%d;%.4f;%.4f;%.4f\n" % (f, posB[f, 0], posB[f, 1], posB[f, 2]))
    print("[out]", out_csv)

    if "--viz" in sys.argv:
        render(cams, det2, posB, gt, os.path.join(OUT, "firstnext.mp4"))
    print("[done] %.1fs" % (time.time() - t0))


def render(cams, det, pos, gt, path):
    caps = [cv2.VideoCapture(c.video) for c in cams]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1920, 720))
    x0, x1, y0, y1 = -62000, 42000, -42000, 22000

    def to_map(p):
        return (int((p[1] - y0) / (y1 - y0) * 620),
                int((1 - (p[0] - x0) / (x1 - x0)) * 340) + 8)

    for f in range(NFRAMES):
        cells = []
        for ci, cap in enumerate(caps):
            ok, img = cap.read()
            if not ok:
                img = np.zeros((F.H, F.W, 3), np.uint8)
            tile = cv2.resize(img, (640, 360))
            d = det[ci][f]
            ep = cams[ci].project(f, pos[f])
            if d is not None:
                b = [int(v / 3) for v in d["bbox"]]
                inside = ep is not None and (d["bbox"][0] <= ep[0] <= d["bbox"][2]
                                             and d["bbox"][1] <= ep[1] <= d["bbox"][3])
                cv2.rectangle(tile, (b[0], b[1]), (b[2], b[3]),
                              (0, 200, 0) if inside else (255, 160, 0), 2)
            pj = cams[ci].project(f, gt[f])
            if pj is not None:
                cv2.drawMarker(tile, (int(pj[0] / 3), int(pj[1] / 3)), (0, 0, 255),
                               cv2.MARKER_CROSS, 16, 2)
            if ep is not None:
                cv2.drawMarker(tile, (int(ep[0] / 3), int(ep[1] / 3)), (0, 255, 255),
                               cv2.MARKER_TILTED_CROSS, 16, 2)
            head = "cam%d f%03d bbox=%s" % (cams[ci].cid, f,
                                            "no" if d is None else "yes")
            cv2.putText(tile, head, (6, 18), 0, 0.45, (0, 0, 0), 3)
            cv2.putText(tile, head, (6, 18), 0, 0.45, (255, 255, 255), 1)
            cells.append(tile)

        mp = np.full((360, 640, 3), 25, np.uint8)
        cv2.putText(mp, "top view  GT=red  est=yellow", (8, 20), 0, 0.45, (200, 200, 200), 1)
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
    print("[viz]", path)


if __name__ == "__main__":
    main()
