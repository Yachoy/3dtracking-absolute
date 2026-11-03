"""Причинный детектор движения для одной камеры.

Вычитание фона на половинном разрешении. Для поворотной камеры фон переносится
гомографией чистого поворота H = K R_t R_{t-1}^T K^-1 (ориентация известна из CSV)
и уточняется фазовой корреляцией. Пиксели, неподвижные > STATIC_N кадров,
сразу уходят в фон: это убирает «призраки» на старте и след за объектом.
"""
import cv2
import numpy as np

S = 0.5        # масштаб обработки
STATIC_N = 4   # кадров неподвижности до поглощения фоном


class Detector:
    """Stateful causal detector for one camera: call step(frame_bgr, frame_idx) -> blobs."""
    def __init__(s, cam, alpha=0.03, thr=18):
        s.cam, s.alpha, s.thr = cam, alpha, thr
        s.K = np.array([[cam.fx*S, 0, cam.cx*S], [0, cam.fy*S, cam.cy*S], [0, 0, 1]]); s.Ki = np.linalg.inv(s.K)
        s.bg = None; s.ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    def step(s, img, f):
        sm = cv2.GaussianBlur(cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_AREA), (5, 5), 0).astype(np.float32)
        _, Rm = s.cam.pose(f); hw = (sm.shape[1], sm.shape[0])
        if s.bg is None:
            s.bg = sm.copy(); s.prev = sm.copy(); s.cnt = np.zeros(sm.shape[:2], np.float32); s.Rprev = Rm
        elif not np.allclose(Rm, s.Rprev):
            H = s.K @ Rm @ s.Rprev.T @ s.Ki   # pure-rotation homography from known camera orientation
            g0 = cv2.warpPerspective(cv2.cvtColor(s.prev, cv2.COLOR_BGR2GRAY), H, hw, borderMode=cv2.BORDER_REPLICATE)
            (dx, dy), resp = cv2.phaseCorrelate(g0, cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY))
            if resp > 0.1 and abs(dx) < 5 and abs(dy) < 5:
                H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]]) @ H
            valid = cv2.warpPerspective(np.ones(sm.shape[:2], np.float32), H, hw) > 0.99
            s.bg = cv2.warpPerspective(s.bg, H, hw, borderMode=cv2.BORDER_REPLICATE); s.bg[~valid] = sm[~valid]
            s.prev = cv2.warpPerspective(s.prev, H, hw, borderMode=cv2.BORDER_REPLICATE)
            s.cnt = cv2.warpPerspective(s.cnt, H, hw, flags=cv2.INTER_NEAREST); s.cnt[~valid] = 0
        s.Rprev = Rm
        still = np.abs(sm - s.prev).max(axis=2) < 4
        s.cnt = np.where(still, s.cnt + 1, 0); s.prev = sm
        absorb = s.cnt > STATIC_N; s.bg[absorb] = sm[absorb]
        m = (np.abs(sm - s.bg).max(axis=2) > s.thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, s.ker); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, s.ker, iterations=2)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        blobs = [dict(cx=cen[i][0]/S, cy=cen[i][1]/S, x=st[i][0]/S, y=st[i][1]/S, w=st[i][2]/S, h=st[i][3]/S, area=st[i][4]/S/S)
                 for i in range(1, n) if st[i][4] >= 25]
        fg = m.astype(bool)[..., None]; a = s.alpha
        s.bg = np.where(fg, s.bg*(1-a*0.2) + sm*a*0.2, s.bg*(1-a) + sm*a)
        return blobs
