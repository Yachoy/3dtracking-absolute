"""Причинный многокамерный 3D-трекер (EKF).

Stage 2: causal multi-camera 3D tracking.
EKF, state = [x,y,z,vx,vy,vz] of the object's visual centre (cm, cm/s).
Measurement: blob centroid (u,v) in each camera = projection of the centre.
Output point = centre + model offset (pivot of the 3D model) along heading/up."""
import numpy as np, pickle, itertools

DT = 1/30.
ACC = [400.]
ACCZ = [130.]
VZ_SIG = [60.]
class EKF:
    def __init__(s, p0, acc=400.):
        s.x = np.r_[p0, 0, 0, 0].astype(float)
        s.P = np.diag([300.**2]*3 + [1500.**2]*3)
        s.acc = acc
    def predict(s):
        F = np.eye(6); F[:3, 3:] = np.eye(3)*DT
        G = np.vstack([np.eye(3)*DT**2/2, np.eye(3)*DT])
        q = np.diag([s.acc**2, s.acc**2, ACCZ[0]**2])
        s.x = F @ s.x; s.P = F @ s.P @ F.T + G @ q @ G.T
    def vz_prior(s, sig):
        # ground vehicle: vertical speed ~ 0 (pseudo-measurement) -> makes single-camera depth observable
        H = np.zeros((1, 6)); H[0, 5] = 1
        S = H @ s.P @ H.T + sig**2; K = s.P @ H.T / S
        s.x = s.x + (K * (0 - s.x[5])).ravel(); s.P = (np.eye(6) - K @ H) @ s.P
    def hjac(s, Pm):
        X = np.append(s.x[:3], 1); p = Pm @ X
        uv = p[:2]/p[2]
        J = (Pm[:2, :3]*p[2] - np.outer(p[:2], Pm[2, :3]))/p[2]**2
        H = np.hstack([J, np.zeros((2, 3))])
        return uv, H, p[2]
    def update(s, z, uv, H, sig):
        R = np.eye(2)*sig**2
        S = H @ s.P @ H.T + R; K = s.P @ H.T @ np.linalg.inv(S)
        s.x = s.x + K @ (z - uv); s.P = (np.eye(6) - K @ H) @ s.P

def triangulate(Ps, uvs):
    A = []
    for P, (u, v) in zip(Ps, uvs):
        A += [u*P[2]-P[0], v*P[2]-P[1]]
    X = np.linalg.svd(np.array(A))[2][-1]; return X[:3]/X[3]

def reproj(P, X):
    p = P @ np.append(X, 1); return p[:2]/p[2], p[2]

def at_border(b, m=4):
    return b['x'] < m or b['y'] < m or b['x']+b['w'] > 1920-m or b['y']+b['h'] > 1080-m

def valid_axes(b, m=4):
    ax = []
    if not (b['x'] < m or b['x']+b['w'] > 1920-m): ax.append(0)
    if not (b['y'] < m or b['y']+b['h'] > 1080-m): ax.append(1)
    return ax

def sigma_px(b):
    return 0.12*np.sqrt(b['w']*b['h']) + 3.

def stacked_update(x, P, Ps, chosen, apply_to=None):
    """Joint EKF update with blobs chosen in several cameras. Returns new x,P and post-fit cost."""
    zs, hs, Hs, Rs = [], [], [], []
    for i, b in chosen:
        Pm = Ps[i]; X = np.append(x[:3], 1); p = Pm @ X
        uv = p[:2]/p[2]; J = (Pm[:2, :3]*p[2] - np.outer(p[:2], Pm[2, :3]))/p[2]**2
        Hfull = np.hstack([J, np.zeros((2, 3))]); sg = sigma_px(b)**2
        axes = valid_axes(b)   # truncated by frame border -> only the coordinate parallel to that border is reliable
        for ax in axes:
            zs.append([b['cx'], b['cy']][ax]); hs.append(uv[ax]); Hs.append(Hfull[ax:ax+1]); Rs.append(sg)
    if not zs: return x, P, 0.
    z = np.array(zs); h = np.array(hs); H = np.vstack(Hs); R = np.diag(Rs)
    S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
    r = z - h; x2 = x + K @ r; P2 = (np.eye(6) - K @ H) @ P
    nis = r @ np.linalg.solve(S, r)
    return x2, P2, nis

def gated_update(ekf, Ps, blobs, gate, cam_bonus=12., max_cand=3):
    # candidates per camera (Mahalanobis gate on prediction)
    cands = []
    for i in range(len(Ps)):
        uv, H, d = ekf.hjac(Ps[i]); c = []
        if d > 0:
            for k, b in enumerate(blobs[i]):
                sg = sigma_px(b); S = H @ ekf.P @ H.T + np.eye(2)*sg**2
                r = np.array([b['cx'], b['cy']]) - uv; m = r @ np.linalg.solve(S, r)
                if m < gate: c.append((m - 2*np.log(b['area']), k, b))
        c.sort(key=lambda t: t[0]); cands.append([(i, k, b) for _, k, b in c[:max_cand]])
    best = (0., [])
    opts = [[None] + c for c in cands]
    for combo in itertools.product(*opts):
        ch = [t for t in combo if t is not None]
        if not ch: continue
        _, _, nis = stacked_update(ekf.x, ekf.P, Ps, [(i, b) for i, k, b in ch])
        cost = nis - cam_bonus*len(ch) - sum(np.log(b['area']) for _, _, b in ch)
        if cost < best[0]: best = (cost, ch)
    ch = best[1]
    if ch:
        ekf.x, ekf.P, _ = stacked_update(ekf.x, ekf.P, Ps, [(i, b) for i, k, b in ch])
    return len(ch), [(i, k) for i, k, b in ch]

def spawn(Ps, blobs, claimed):
    new = []
    for i, j in itertools.combinations(range(len(Ps)), 2):
        for ki, bi in enumerate(blobs[i]):
            if (i, ki) in claimed or bi['area'] < 100 or at_border(bi): continue
            for kj, bj in enumerate(blobs[j]):
                if (j, kj) in claimed or bj['area'] < 100 or at_border(bj): continue
                X = triangulate([Ps[i], Ps[j]], [(bi['cx'], bi['cy']), (bj['cx'], bj['cy'])])
                e = []
                for k, b in ((i, bi), (j, bj)):
                    uv, d = reproj(Ps[k], X); e.append(np.linalg.norm(uv-[b['cx'], b['cy']])/sigma_px(b) if d > 0 else 1e9)
                if max(e) < 2 and -2000 < X[2] < 5000: new.append(X)
    return new

class Track:
    def __init__(s, X, f):
        s.ekf = EKF(X, ACC[0]); s.score = 0.; s.miss = 0; s.age = 0; s.born = f; s.dist = 0.

class Tracker:
    """Causal multi-camera 3D tracker. step(f, blobs_per_cam) -> (output_xyz or None, centre, n_cams)."""
    def __init__(s, cams, offset=(800., 0.), gate=16., acc=1500., accz=300., vzsig=150., merge=1.0,
                 reanchor=True, reanchor_dist=4000., lag=0.4):
        s.cams, s.offset, s.gate, s.merge, s.reanchor, s.reanchor_dist = cams, offset, gate, merge, reanchor, reanchor_dist
        ACC[0] = acc; ACCZ[0] = accz; VZ_SIG[0] = vzsig
        s.lag = lag; s.tracks = []; s.cur = None; s.heading = np.array([1., 0, 0])
    def step(s, f, raw_blobs):
        cams, gate = s.cams, s.gate
        Ps = [c.P(f) for c in cams]
        blobs = [merge_blobs(b, k=s.merge) if s.merge else b for b in raw_blobs]
        allclaimed = []
        for t in sorted(s.tracks, key=lambda t: -t.score):
            t.ekf.predict(); t.age += 1
            t.ekf.vz_prior(VZ_SIG[0])
            u, cl = gated_update(t.ekf, Ps, blobs, gate)
            t.used = u; t.claimed = cl; allclaimed += cl
            t.score = 0.97*t.score + u + (0.5 if u >= 2 else 0)
            t.miss = 0 if u else t.miss + 1
            t.dist += np.linalg.norm(t.ekf.x[3:])*DT
            if t.miss > 15: t.ekf.x[3:] *= 0.95
        cur = s.cur
        tracks = [t for t in s.tracks if t is cur or not (t.miss > 20 or (t.age > 30 and t.score < 3))]
        keep = []
        for t in sorted(tracks, key=lambda t: -t.score):
            if all(np.linalg.norm(t.ekf.x[:3]-k.ekf.x[:3]) > 500 for k in keep) or t is cur: keep.append(t)
        tracks = keep
        if len(tracks) < 20:
            for X in spawn(Ps, blobs, allclaimed): tracks.append(Track(X, f))
        conf = [t for t in tracks if t.age >= 10 and t.dist > 300]
        if conf:
            best = max(conf, key=lambda t: t.score)
            if cur is None or cur not in tracks or best.score > 1.5*cur.score + 2: cur = best
        s.tracks, s.cur = tracks, cur
        if cur is None: return None, None, 0
        if cur.used <= 1 and s.reanchor:
            mine = cur.claimed; best = None
            for i, j in itertools.combinations(range(len(cams)), 2):
                for ki, bi in enumerate(blobs[i]):
                    for kj, bj in enumerate(blobs[j]):
                        if mine and (i, ki) not in mine and (j, kj) not in mine: continue
                        if min(bi['area'], bj['area']) < 300 or at_border(bi) or at_border(bj): continue
                        X = triangulate([Ps[i], Ps[j]], [(bi['cx'], bi['cy']), (bj['cx'], bj['cy'])])
                        e = []
                        for k, b in ((i, bi), (j, bj)):
                            uv, d = reproj(Ps[k], X); e.append(np.linalg.norm(uv-[b['cx'], b['cy']])/sigma_px(b) if d > 0 else 1e9)
                        dist = np.linalg.norm(X - cur.ekf.x[:3])
                        if max(e) < 1.5 and dist < s.reanchor_dist and abs(X[2]-cur.ekf.x[2]) < 1000 and (best is None or dist < best[0]):
                            best = (dist, X)
            if best is not None and best[0] > 300:
                cur.ekf.x[:3] = best[1]; cur.ekf.P[:3, :3] = np.eye(3)*200.**2
                cur.ekf.P[3:, 3:] = np.maximum(cur.ekf.P[3:, 3:], np.eye(3)*800.**2); cur.used = 2
        v = cur.ekf.x[3:5]; sp = np.linalg.norm(v)
        if sp > 150: s.heading = np.r_[v/sp, 0]
        c = cur.ekf.x[:3].copy()
        # model pivot = visual centre + (half-length L + estimation lag tau * speed) along heading
        along = s.offset[0] + s.lag*sp
        return c + along*s.heading + np.array([0, 0, s.offset[1]]), c, cur.used

def run(det, cams, nframes, **kw):
    tr = Tracker(cams, **kw)
    out = np.full((nframes, 3), np.nan); centre = np.full((nframes, 3), np.nan); nused = np.zeros(nframes, int)
    for f in range(nframes):
        o, c, u = tr.step(f, [det[i][f] if f < len(det[i]) else [] for i in range(len(cams))])
        if o is not None: out[f], centre[f], nused[f] = o, c, u
    return out, centre, nused


def merge_blobs(blobs, k=0.6, min_area=60):
    """Union blobs whose boxes are close (fragmented vehicle: cab/trailer, occluding trunks)."""
    bs = [dict(b) for b in blobs if b['area'] >= min_area]
    changed = True
    while changed and len(bs) > 1:
        changed = False
        for a in range(len(bs)):
            for c in range(a+1, len(bs)):
                A, B = bs[a], bs[c]
                gx = max(0, max(A['x'], B['x']) - min(A['x']+A['w'], B['x']+B['w']))
                gy = max(0, max(A['y'], B['y']) - min(A['y']+A['h'], B['y']+B['h']))
                lim = k*max(A['w'], A['h'], B['w'], B['h'])
                if gx <= lim and gy <= lim:
                    x0, y0 = min(A['x'], B['x']), min(A['y'], B['y'])
                    x1, y1 = max(A['x']+A['w'], B['x']+B['w']), max(A['y']+A['h'], B['y']+B['h'])
                    bs[a] = dict(x=x0, y=y0, w=x1-x0, h=y1-y0, cx=(x0+x1)/2, cy=(y0+y1)/2, area=A['area']+B['area'])
                    bs.pop(c); changed = True; break
            if changed: break
    for b in bs:  # use box centre (robust to holes) for merged/any blobs
        b['cx'], b['cy'] = b['x']+b['w']/2, b['y']+b['h']/2
    return bs
