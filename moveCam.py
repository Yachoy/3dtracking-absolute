# -*- coding: utf-8 -*-
"""
moveCam.py - the moving camera only (cam5, fragment 575..1000).

Idea: compare NEURAL FEATURES instead of pixels.  Pixels of a dune change by
30-40 grey levels from one frame to the next just because the camera panned and
the shading changed; the early features of the segmentation network are much
more stable.  So:

  1. every frame goes through the first blocks of yolo11s-seg;
  2. the previous frame's feature map is warped into the current view by the
     camera rotation (K M_f M_prev^T K^-1 mapped into the feature grid) with
     bilinear sampling on the GPU;
  3. the residual |F_f - warp(F_prev)| is normalised per frame (median / MAD ->
     z-score) so the heat map is readable;
  4. z > Z_THR is the "something moves differently here" mask; the largest
     cluster after morphology is the detection.

Panels (3x2, 640x360 each):
   source | pixel residual (z) | feature residual (z)
   feature mask + box | feature map as RGB | metrics
"""
import os
import sys
import time
import numpy as np
import cv2
import torch

import first as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
WEIGHTS = os.path.join(HERE, "yolo11s-seg.pt")

CAM = 5
F0, F1 = 575, 1000
LAST = 5                  # blocks 0..LAST: P4 (stride 16, 256 ch)
Z_THR = 4.0
AREA_MIN = 80             # working-res px
IMG_W, IMG_H = 960, 544   # network input (multiple of 32)
WW, WH = F.WW, F.WH


class Feat:
    """First blocks of the yolo11-seg backbone, on the GPU."""

    def __init__(self, weights, last=LAST):
        from ultralytics import YOLO
        seq = YOLO(weights).model.model
        self.blocks = torch.nn.Sequential(*[seq[i] for i in range(last + 1)])
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blocks.to(self.dev).eval()
        with torch.no_grad():
            probe = self.blocks(torch.zeros(1, 3, IMG_H, IMG_W).to(self.dev))
        self.s = IMG_H // probe.shape[2]
        self.ch = probe.shape[1]
        print("[feat] blocks 0..%d: %d ch, stride %d, device %s"
              % (last, self.ch, self.s, self.dev))

    @torch.no_grad()
    def __call__(self, bgr):
        img = cv2.resize(bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(np.ascontiguousarray(img[..., ::-1])).to(self.dev).float() / 255.0
        t = t.permute(2, 0, 1)[None]
        return self.blocks(t)[0]


def feat_warp_grid(fe, Hm_img, hf, wf, dev):
    """Bilinear sampling grid bringing the previous feature map into view."""
    s = fe.s
    S = np.array([[s, 0, s / 2.0 - 0.5], [0, s, s / 2.0 - 0.5], [0, 0, 1.0]])
    Hs = np.linalg.inv(S) @ Hm_img @ S
    ys, xs = np.mgrid[0:hf, 0:wf].astype(np.float64)
    p = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 0)
    q = np.linalg.inv(Hs) @ p
    qx = (q[0] / q[2] * 2.0 / max(wf - 1, 1) - 1.0).reshape(hf, wf)
    qy = (q[1] / q[2] * 2.0 / max(hf - 1, 1) - 1.0).reshape(hf, wf)
    g = np.stack([qx, qy], -1)[None]
    return torch.from_numpy(g).to(dev).float()


def zmap(d):
    d = d - np.median(d)
    s = 1.4826 * np.median(np.abs(d)) + 1e-6
    return d / s


def colour(z):
    v = np.clip(z / 8.0, 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def panel(name, img, extra=""):
    p = np.ascontiguousarray(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    txt = name if not extra else "%s   %s" % (name, extra)
    cv2.putText(p, txt, (6, 18), 0, 0.45, (0, 0, 0), 3)
    cv2.putText(p, txt, (6, 18), 0, 0.45, (255, 255, 255), 1)
    return cv2.resize(p, (640, 360), interpolation=cv2.INTER_AREA)


def image_homography(cam, i, j):
    """pixels of frame i -> pixels of frame j (pure rotation, pose from the CSV)."""
    s = IMG_W / float(F.W)
    Ks = np.array([[cam.fpx * s, 0, IMG_W / 2.0],
                   [0, cam.fpx * (IMG_H / float(F.H)), IMG_H / 2.0],
                   [0, 0, 1.0]])
    Hm = Ks @ (cam._M(j) @ cam._M(i).T) @ np.linalg.inv(Ks)
    return Hm / Hm[2, 2]


def main():
    os.makedirs(OUT, exist_ok=True)
    cam = F.Camera(CAM)
    fe = Feat(WEIGHTS)
    dev = fe.dev
    t0 = time.time()
    cap = cv2.VideoCapture(cam.video)
    vw = cv2.VideoWriter(os.path.join(OUT, "movecam_feat.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1920, 720))
    prev_f = None
    prev_img = None
    prev_i = None
    areas = []
    f = 0
    while True:
        ok, img = cap.read()
        if not ok or f > F1:
            break
        if f < F0:
            f += 1
            continue
        big = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        ft = fe(img)

        if prev_f is not None:
            Hij = image_homography(cam, prev_i, f)
            hf, wf = ft.shape[1], ft.shape[2]
            g = feat_warp_grid(fe, Hij, hf, wf, dev)
            wprev = torch.nn.functional.grid_sample(
                prev_f[None], g, mode="bilinear", padding_mode="border",
                align_corners=True)[0]
            z = zmap((ft - wprev).abs().mean(0).cpu().numpy())
            wp = cv2.warpPerspective(prev_img, Hij, (IMG_W, IMG_H))
            zp = zmap(np.abs(big.astype(np.float32) - wp.astype(np.float32)).max(2))
        else:
            z = np.zeros((ft.shape[1], ft.shape[2]), np.float32)
            zp = np.zeros((IMG_H, IMG_W), np.float32)

        zm = cv2.resize(z, (WW, WH), interpolation=cv2.INTER_CUBIC)
        zm = cv2.GaussianBlur(zm, (9, 9), 0)
        m = (zm > Z_THR).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        box = None
        mask = np.zeros_like(m)
        if n > 1:
            j = 1 + int(np.argmax(stats[1:, 4]))
            if stats[j, 4] >= AREA_MIN:
                mask = (lab == j).astype(np.uint8)
                box = stats[j, :4]
        areas.append(0 if box is None else int(mask.sum()))

        mask_vis = cv2.cvtColor(cv2.resize(mask * 255, (IMG_W, IMG_H),
                                          interpolation=cv2.INTER_NEAREST),
                                cv2.COLOR_GRAY2BGR)
        if box is not None:
            b = [int(v * 2) for v in box]
            cv2.rectangle(mask_vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        fvis = ft[:3].cpu().numpy().transpose(1, 2, 0)
        fvis = (fvis - fvis.min()) / (np.ptp(fvis) + 1e-6)
        fvis = (fvis * 255).astype(np.uint8)

        met = np.full((360, 640, 3), 25, np.uint8)
        cv2.putText(met, "feature residual, P4: %d ch, stride %d" % (fe.ch, fe.s),
                    (8, 22), 0, 0.5, (200, 200, 200), 1)
        cv2.putText(met, "z > %.1f -> mask" % Z_THR, (8, 52), 0, 0.5, (200, 200, 200), 1)
        if box is not None:
            cv2.putText(met, "cluster %dx%d px, area %d" %
                        (box[2] * 2, box[3] * 2, mask.sum() * 4),
                        (8, 92), 0, 0.55, (0, 255, 0), 1)
        else:
            cv2.putText(met, "cluster: none", (8, 92), 0, 0.55, (120, 120, 120), 1)
        cv2.putText(met, "frame %d" % f, (8, 130), 0, 0.5, (0, 255, 255), 1)

        cells = [panel("source", img),
                 panel("pixel residual (pose-warped)", colour(zp)),
                 panel("feature residual", colour(zm)),
                 panel("feature mask", mask_vis),
                 panel("feature map ch0-2 (P4)", fvis),
                 met]
        vw.write(np.vstack([np.hstack(cells[0:3]), np.hstack(cells[3:6])]))
        prev_f = ft
        prev_img = big
        prev_i = f
        if f % 50 == 0:
            print("[movecam] f%4d  %.1fs" % (f, time.time() - t0))
        f += 1
    cap.release()
    vw.release()
    print("[movecam] ->", os.path.join(OUT, "movecam_feat.mp4"))
    a = np.array(areas)
    print("frames with a cluster: %d/%d  median area %d px"
          % ((a > 0).sum(), len(a), int(np.median(a)) if len(a) else 0))


if __name__ == "__main__":
    main()
