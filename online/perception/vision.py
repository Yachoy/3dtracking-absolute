"""Current-image instance segmentation, optionally transported from t-1 by KLT."""
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Instance:
    contour: np.ndarray
    score: float
    label: int
    flow: bool = False
    source: str = "yolo"
    affine: np.ndarray | None = None

    @property
    def box(self):
        return np.r_[self.contour.min(axis=0), self.contour.max(axis=0)]


class Segmenter:
    """Ultralytics YOLO26 end-to-end ONNX segmenter."""

    def __init__(self, path, threads=3, labels=(), exclude=()):
        self.labels = labels
        self.exclude = exclude
        if "CUDAExecutionProvider" in ort.get_available_providers() and hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        for gpu in ("DmlExecutionProvider", "CUDAExecutionProvider"):  # DirectML — любая видеокарта на Windows
            if gpu in ort.get_available_providers():
                providers.insert(0, gpu)
        self.model = ort.InferenceSession(str(path), options, providers=providers)
        ort.disable_telemetry_events()
        self.input = self.model.get_inputs()[0].name
        self.size = self.model.get_inputs()[0].shape[-1]

    def infer(self, image, roi, local=False):
        x, y, x2, y2 = roi
        crop = image[y:y2, x:x2]
        h, w = crop.shape[:2]
        scale = self.size / max(h, w)
        rw, rh = round(w * scale), round(h * scale)
        scaled = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:rh, :rw] = scaled
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255, swapRB=True)
        results, proto = self.model.run(None, {self.input: blob})
        proto = proto[0]
        found = []
        for row in results[0]:
            label = int(row[5])
            confidence = float(row[4])
            if confidence < (0.055 if local else 0.20) or label in self.exclude:
                continue
            a, b, c, d = row[:4]
            if c - a < 12 or d - b < 10:
                continue
            ph, pw = proto.shape[1:]
            a0, b0 = np.maximum(np.floor(np.array([a, b]) * pw / self.size).astype(int) - 1, 0)
            c0, d0 = np.minimum(np.ceil(np.array([c, d]) * ph / self.size).astype(int) + 2, [pw, ph])
            small = np.einsum("i,ijk->jk", row[6:], proto[:, b0:d0, a0:c0])
            if not small.size:
                continue
            logits = cv2.resize(small, None, fx=self.size/pw, fy=self.size/ph)
            xx, yy = np.arange(logits.shape[1]) + a0 * self.size/pw, np.arange(logits.shape[0]) + b0 * self.size/ph
            mask = (logits > 0) & (xx[None] >= a-2) & (xx[None] <= min(c+2, rw-1))
            mask &= (yy[:, None] >= b-2) & (yy[:, None] <= min(d+2, rh-1))
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) < 50:
                continue
            points = (contour[:, 0].astype(float) + [xx[0], yy[0]]) / scale + [x, y]
            found.append(Instance(points, confidence, label))
        return found

    def search(self, image, expected=None):
        height, width = image.shape[:2]
        if expected is None:
            side = min(height, width)
            span = max(height, width) - side
            n = max(1, int(np.ceil(span / (0.8 * side))))
            tiles = [int(round(i)) for i in np.linspace(0, span, n + 1)] if span else [0]
            rois = [(i, 0, i + side, side) for i in tiles] if width >= height else [
                (0, i, side, i + side) for i in tiles
            ]
        else:
            center = (expected[:2] + expected[2:]) / 2
            side = int(min(height, width, max(176, 1.85 * max(expected[2:] - expected[:2]) + 30)))
            x, y = np.clip(np.round(center - side / 2).astype(int), 0, [width-side, height-side])
            rois = [(x, y, x + side, y + side)]
        found = [item for roi in rois for item in self.infer(image, roi, local=expected is not None)]
        return sorted(found, key=lambda item: item.score * (1 if not self.labels or item.label in self.labels else 0.55), reverse=True)


def transport(previous, current, instance):
    """Forward/backward checked affine flow inside the previous instance."""
    height, width = previous.shape
    box = instance.box
    pad = max(16, min(120, int(max(box[2:] - box[:2]) * 0.35)))
    x1, y1 = np.maximum(np.floor(box[:2]).astype(int) - pad, 0)
    x2, y2 = np.minimum(np.ceil(box[2:]).astype(int) + pad + 1, [width, height])
    a, b = previous[y1:y2, x1:x2], current[y1:y2, x1:x2]
    mask = np.zeros_like(a)
    cv2.fillPoly(mask, [np.round(instance.contour - [x1, y1]).astype(np.int32)], 255)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    points = cv2.goodFeaturesToTrack(a, 45, 0.009, 3, mask=mask)
    if points is None or len(points) < 5:
        return None
    opts = dict(winSize=(17, 17), maxLevel=2)
    moved, valid, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, **opts)
    back, valid_back, _ = cv2.calcOpticalFlowPyrLK(b, a, moved, None, **opts)
    keep = valid[:, 0].astype(bool) & valid_back[:, 0].astype(bool)
    keep &= np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 1.2
    if keep.sum() < 5:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(points[keep], moved[keep], ransacReprojThreshold=1.5)
    if matrix is None or inliers.sum() < 5 or not 0.88 <= np.linalg.norm(matrix[:, 0]) <= 1.14:
        return None
    origin = np.array([x1, y1])
    matrix[:, 2] += origin - matrix[:, :2] @ origin
    contour = instance.contour @ matrix[:, :2].T + matrix[:, 2]
    return Instance(contour, instance.score, instance.label, True, instance.source, matrix)

class PromptSegmenter:
    """Refine a CAD-guided box against pixels of the current frame."""

    def __init__(self, model, threads):
        from experiments.sam2_model.sam2seg import SAM2

        self.model = SAM2(threads, model_path=model)

    def search(self, image, expected, precise=False):
        height, width = image.shape[:2]
        center = (expected[:2] + expected[2:]) / 2
        extent = np.maximum(expected[2:] - expected[:2], 16)
        radius = np.maximum(extent * (0.85 if precise else 1.12), 36 if precise else 52)
        x, y = np.maximum(np.floor(center - radius).astype(int), 0)
        x2, y2 = np.minimum(np.ceil(center + radius).astype(int), [width, height])
        if x2-x < 32 or y2-y < 32:
            return None
        offset = np.array([x, y])
        corner = np.array([x2-x-1, y2-y-1])
        pad = np.minimum(extent * (0.02 if precise else np.array([0.31, 0.20])), 55)
        lo = np.clip(expected[:2] - offset - pad, 0, corner)
        hi = np.clip(expected[2:] - offset + pad, 0, corner)
        if np.min(hi-lo) < 10:
            return None
        crop = image[y:y2, x:x2]
        side = max(crop.shape[:2])
        crop = cv2.copyMakeBorder(crop, 0, side - crop.shape[0], 0, side - crop.shape[1],
                                  cv2.BORDER_REPLICATE)
        features = self.model.encode(crop)
        if precise:
            return self.select_mask(features, lo, hi, expected, offset, y2-y, x2-x)
        mask, quality = self.model.decode(features, [lo, hi], [2, 3])
        if quality < 0.64:
            return None
        mask = mask[:y2-y, :x2-x].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 70:
            return None
        return Instance(contour[:, 0].astype(float) + offset, float(quality), -1, source="sam")

    def select_mask(self, features, lo, hi, expected, offset, height, width):
        best = None
        for mask, quality in self.model.decode_candidates(features, [lo, hi], [2, 3]):
            if quality < 0.45:
                continue
            contours, _ = cv2.findContours(mask[:height, :width].astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                if cv2.contourArea(contour) < 55:
                    continue
                points = contour[:, 0].astype(float) + offset
                box = np.r_[points.min(axis=0), points.max(axis=0)]
                common = np.maximum(np.minimum(box[2:], expected[2:]) - np.maximum(box[:2], expected[:2]), 0).prod()
                union = (box[2:] - box[:2]).prod() + (expected[2:] - expected[:2]).prod() - common
                agreement = common / max(union, 1)
                score = agreement * quality
                if agreement >= 0.32 and (best is None or score > best[0]):
                    best = score, Instance(points, quality, -1, source="sam")
        return None if best is None else best[1]

def contour_motion(previous, current, contour, transform=None):
    """Estimate displacement in the current object after removing camera rotation."""
    height, width = current.shape
    lo = np.maximum(np.floor(contour.min(axis=0)).astype(int) - 18, 0)
    hi = np.minimum(np.ceil(contour.max(axis=0)).astype(int) + 19, [width, height])
    if np.any(hi - lo < 18):
        return None
    x, y = lo
    x2, y2 = hi
    now = current[y:y2, x:x2]
    if transform is None:
        before = previous[y:y2, x:x2]
    else:
        shift = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
        before = cv2.warpPerspective(previous, shift @ transform, (x2-x, y2-y),
                                     borderMode=cv2.BORDER_REPLICATE)
    mask = np.zeros_like(now)
    cv2.fillPoly(mask, [np.round(contour - lo).astype(np.int32)], 255)
    points = cv2.goodFeaturesToTrack(now, 45, 0.012, 4, mask=mask)
    if points is None or len(points) < 5:
        return None
    past, flags, _ = cv2.calcOpticalFlowPyrLK(now, before, points, None,
                                             winSize=(15, 15), maxLevel=2)
    keep = flags[:, 0].astype(bool)
    if keep.sum() < 5:
        return None
    moved = points[keep, 0] - past[keep, 0]
    return np.median(moved, axis=0)
