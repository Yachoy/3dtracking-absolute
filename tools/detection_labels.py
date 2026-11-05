"""Где фура видна, а детектор её не находит, и разметка для дообучения YOLO.

Положение фуры берётся из эталона (`--reference`, Way1) или из надёжных кадров трекера
(`--replay`, статус `multi` — псевдоразметка для сцен без эталона, Way2). CAD-модель ставится в эту
позу и проецируется в каждую камеру. Проекция не знает о закрытии горой или деревом, поэтому
видимость проверяет SAM2 по рамке проекции: фура считается видимой, если маска SAM2 лежит внутри
расширенного силуэта CAD и покрывает не меньше `--min-visible` его площади.

Режимы:
  recall  — для видимых фур запускает YOLO так же, как трекер `--global-seg` (весь кадр, 960),
            и по окну вокруг фуры (640); печатает долю найденных при порогах 0,05 / 0,16 / 0,30
            по камерам и размерам, сохраняет вырезки пропусков в `--out/misses`.
  dataset — пишет датасет YOLO-seg (класс 7 «truck» COCO, чтобы веса оставались совместимы с
            `--global-seg` и фильтром классов 2/5/7): маска = SAM2 ∩ силуэт CAD; часть кадров
            обучающей выборки дублируется с синтетическим дымом/туманом и пониженным контрастом.

    python tools/detection_labels.py recall --data data/test --reference data/test/object_transform_SM_AA_track.csv \
        --sam models/sam2/sam2.1_hiera_tiny --weights models/yolo26l-seg.pt --out out/recall-way1
    python tools/detection_labels.py dataset --data data/test --reference data/test/object_transform_SM_AA_track.csv \
        --sam models/sam2/sam2.1_hiera_tiny --out datasets/peleng --split train --name way1
    python tools/detection_labels.py dataset --data data/way2 --replay out/way2-base/replay.json \
        --sam models/sam2/sam2.1_hiera_tiny --out datasets/peleng --split val --name way2
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.camera import Camera  # noqa: E402
from online.geometry import DEFAULT_TARGET, Rot, View  # noqa: E402
from online.run import sources  # noqa: E402
from online.target import load_target  # noqa: E402

TRUCK = 7  # COCO «truck»
COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]


def read_reference(path):
    gt = {}
    for row in Path(path).read_text(encoding="cp1251").splitlines()[1:]:
        p = row.split(";")
        try:
            gt[int(p[0])] = np.array([float(p[1]), float(p[2]), float(p[3])]) / 100
        except (ValueError, IndexError):
            pass
    return gt


def read_replay(path):
    """Надёжные позы трекера (многокамерные засечки) — псевдоразметка без эталона."""
    frames = json.loads(Path(path).read_text(encoding="utf-8"))["frames"]
    return {f["frame"]: np.array(f["xyz"]) / 100 for f in frames
            if f["xyz"][0] is not None and f["status"] == "multi"}


def poses(points, window=5):
    """Курс по траектории (в эталоне и replay его нет): направление движения за ±window кадров."""
    keys = sorted(points)
    out, course = {}, 0.0
    for i in keys:
        a = points.get(i - window, points[i])
        b = points.get(i + window, points[i])
        if np.linalg.norm(b[:2] - a[:2]) > 1.0:
            course = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
        out[i] = np.r_[points[i], course, 0.0]
    return out


def silhouette(view, model, pose, shape):
    """Маска проекции CAD (объединение выпуклых оболочек частей) и её рамка."""
    matrix = Rot(*pose[3:]).T
    mask = np.zeros(shape, np.uint8)
    box = None
    for part in model.parts:
        uv, depth = view.project(np.asarray(part) @ matrix + pose[:3])
        if depth.min() < 1:
            return None, None
        hull = cv2.convexHull(np.round(np.clip(uv, -5000, 5000)).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        box = np.r_[lo, hi] if box is None else np.r_[np.minimum(box[:2], lo), np.maximum(box[2:], hi)]
    return mask, box


MIN_IOU = 0.55       # маска SAM2 по форме совпадает с силуэтом CAD
MIN_SAM_SCORE = 0.80
MAX_TEXTURE = 55.0   # средний модуль лапласиана внутри маски: листва даёт 67–137, борт фуры 11–40
                     # (Way1: дерево перед закрытой фурой SAM2 охотно заливает по рамке проекции)


def visible_mask(prompt, image, sil, box, min_visible):
    """Маска SAM2 по рамке проекции, если фура действительно видна; иначе None."""
    h, w = image.shape[:2]
    clipped = np.clip(box, [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
    if min(clipped[2:] - clipped[:2]) < 10 or sil.sum() < 80:
        return None
    found = prompt.search(image, clipped, precise=True)
    if found is None or found.score < MIN_SAM_SCORE:
        return None
    sam = np.zeros((h, w), np.uint8)
    cv2.fillPoly(sam, [np.round(found.contour).astype(np.int32)], 1)
    size = int(max(8, 0.08 * max(clipped[2:] - clipped[:2])))
    grown = cv2.dilate(sil, np.ones((2 * size + 1, 2 * size + 1), np.uint8))
    inside = (sam & grown).sum() / max(sam.sum(), 1)
    covered = (sam & sil).sum() / max(sil.sum(), 1)
    iou = (sam & sil).sum() / max((sam | sil).sum(), 1)
    if inside < 0.80 or covered < min_visible or iou < MIN_IOU:
        return None
    core = cv2.erode(sam, np.ones((5, 5), np.uint8))
    if core.any():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))[core > 0].mean() > MAX_TEXTURE:
            return None  # под маской листва или скалы, а не борт машины
    return sam & grown


def smoke(image, rng):
    """Синтетический дым/туман: размытое случайное поле серо-белого цвета и пониженный контраст."""
    h, w = image.shape[:2]
    field = np.zeros((h, w), np.float32)
    for scale in (8, 32, 128):
        small = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
        field += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * (scale / 128)
    field = cv2.GaussianBlur(field, (0, 0), 25)
    field = (field - field.min()) / max(np.ptp(field), 1e-6)
    alpha = np.clip(field * rng.uniform(0.35, 0.85), 0, 0.85)[..., None]
    tone = rng.uniform(170, 235)
    out = image.astype(np.float32) * (1 - alpha) + tone * alpha
    contrast = rng.uniform(0.6, 1.0)
    out = (out - out.mean()) * contrast + out.mean()
    return np.clip(out, 0, 255).astype(np.uint8)


def iou(a, b):
    common = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    return common / max((a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - common, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("recall", "dataset"))
    ap.add_argument("--data", type=Path, required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference", type=Path, help="эталон (CSV трека объекта)")
    source.add_argument("--replay", type=Path, help="replay трекера: позы кадров multi как псевдоэталон")
    ap.add_argument("--target", type=Path, help="CAD-цель (по умолчанию — встроенная фура)")
    ap.add_argument("--sam", type=Path, required=True, help="SAM2 ONNX, например models/sam2/sam2.1_hiera_tiny")
    ap.add_argument("--weights", type=Path, help="YOLO-seg для режима recall")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=3, help="каждый N-й кадр")
    ap.add_argument("--min-visible", type=float, default=0.35, help="доля силуэта, покрытая маской SAM2")
    ap.add_argument("--split", choices=("train", "val"), default="train")
    ap.add_argument("--name", default="scene", help="префикс файлов сцены в датасете")
    ap.add_argument("--smoke", type=float, default=0.4, help="доля обучающих кадров с копией в дыму")
    ap.add_argument("--negatives", type=float, default=0.1, help="доля кадров без видимой фуры")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frames", help="диапазон кадров a:b (для проверки на нескольких кадрах)")
    args = ap.parse_args()

    if sys.platform == "win32":  # cuDNN из PyTorch должен загрузиться раньше ONNX Runtime, иначе SAM2 уходит на CPU
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
    from online.perception.vision import PromptSegmenter

    model = load_target(args.target) if args.target else DEFAULT_TARGET
    points = read_reference(args.reference) if args.reference else read_replay(args.replay)
    posed = poses(points)
    tracks = sources(args.data)
    caps = {key: cv2.VideoCapture(str(track.video)) for key, track in tracks.items()}
    prompt = PromptSegmenter(args.sam, args.threads)
    yolo = None
    if args.mode == "recall":
        from ultralytics import YOLO
        yolo = YOLO(str(args.weights))
    rng = np.random.default_rng(0)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "dataset":
        for sub in ("images", "labels"):
            (args.out / sub / args.split).mkdir(parents=True, exist_ok=True)
    stats = []
    selected = sorted(posed)
    if args.frames:
        first, last = (int(v) for v in args.frames.split(":"))
        selected = [f for f in selected if first <= f <= last]
    for frame in selected[::args.stride]:
        pose = posed[frame]
        for key, track in tracks.items():
            if frame not in track.poses:
                continue
            caps[key].set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = caps[key].read()
            if not ok:
                continue
            h, w = image.shape[:2]
            view = View.of(Camera.from_track(track, frame, w, h), model=model)
            sil, box = silhouette(view, model, pose, (h, w))
            in_view = box is not None and box[2] > 0 and box[3] > 0 and box[0] < w and box[1] < h
            mask = visible_mask(prompt, image, sil, box, args.min_visible) if in_view else None
            if args.mode == "recall":
                if mask is None:
                    continue
                ys, xs = np.nonzero(mask)
                truth = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
                best_full = best_crop = 0.0
                result = yolo.predict(image, imgsz=960, conf=0.01, classes=list(model.labels) or None,
                                      max_det=10, verbose=False)[0]
                for b, s in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    if iou(b, truth) >= 0.5:
                        best_full = max(best_full, float(s))
                side = int(min(h, w, max(320, 2.0 * max(truth[2:] - truth[:2]))))
                cx, cy = ((truth[:2] + truth[2:]) / 2).astype(int)
                x0, y0 = int(np.clip(cx - side // 2, 0, w - side)), int(np.clip(cy - side // 2, 0, h - side))
                crop = yolo.predict(image[y0:y0 + side, x0:x0 + side], imgsz=640, conf=0.01,
                                    classes=list(model.labels) or None, max_det=10, verbose=False)[0]
                for b, s in zip(crop.boxes.xyxy.cpu().numpy(), crop.boxes.conf.cpu().numpy()):
                    if iou(b + [x0, y0, x0, y0], truth) >= 0.5:
                        best_crop = max(best_crop, float(s))
                size = float(max(truth[2:] - truth[:2]))
                clipped = bool(truth[0] < 4 or truth[1] < 4 or truth[2] > w - 5 or truth[3] > h - 5)
                stats.append((frame, key, size, clipped, best_full, best_crop))
                if best_full < 0.16:
                    (args.out / "misses").mkdir(exist_ok=True)
                    pad = int(0.3 * size) + 20
                    x1, y1 = np.maximum(truth[:2].astype(int) - pad, 0)
                    x2, y2 = np.minimum(truth[2:].astype(int) + pad, [w, h])
                    shot = image[y1:y2, x1:x2].copy()
                    cv2.imwrite(str(args.out / "misses" / f"f{frame:04d}_cam{key}_full{best_full:.2f}_crop{best_crop:.2f}.jpg"), shot)
            else:
                keep_negative = mask is None and rng.random() < args.negatives
                if mask is None and not keep_negative:
                    continue
                lines = []
                if mask is not None:
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if cv2.contourArea(contour) < 40:
                            continue
                        poly = contour[:, 0].astype(float) / [w, h]
                        lines.append(f"{TRUCK} " + " ".join(f"{v:.5f}" for v in poly.ravel()))
                stem = f"{args.name}_f{frame:04d}_cam{key}"
                variants = [(stem, image)]
                if args.split == "train" and lines and rng.random() < args.smoke:
                    variants.append((stem + "_smoke", smoke(image, rng)))
                for name, picture in variants:
                    cv2.imwrite(str(args.out / "images" / args.split / f"{name}.jpg"), picture,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    (args.out / "labels" / args.split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
                stats.append((frame, key, bool(lines)))
    if args.mode == "recall":
        rows = np.array([(s[2], s[3], s[4], s[5]) for s in stats], float)
        cams = np.array([s[1] for s in stats])
        print(f"видимых фур (кадр × камера): {len(rows)}")
        for name, column in (("весь кадр 960 (как --global-seg)", 2), ("окно вокруг фуры 640", 3)):
            print(name + ": " + ", ".join(f"≥{t}: {np.mean(rows[:, column] >= t):.0%}" for t in (0.05, 0.16, 0.30)))
        for key in sorted(set(cams)):
            sel = cams == key
            print(f"  камера {key}: {sel.sum()} видимых, найдено при 0,16: {np.mean(rows[sel, 2] >= 0.16):.0%} "
                  f"(окно: {np.mean(rows[sel, 3] >= 0.16):.0%})")
        for lo, hi in ((0, 60), (60, 150), (150, 400), (400, 4000)):
            sel = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
            if sel.any():
                print(f"  размер {lo}–{hi} px: {sel.sum()}, найдено при 0,16: {np.mean(rows[sel, 2] >= 0.16):.0%}")
        sel = rows[:, 1] > 0
        if sel.any():
            print(f"  обрезана краем кадра: {sel.sum()}, найдено при 0,16: {np.mean(rows[sel, 2] >= 0.16):.0%}")
        print(f"вырезки пропусков: {args.out / 'misses'}")
    else:
        positives = sum(1 for s in stats if s[2])
        print(f"{args.split}: {positives} кадров с фурой, {len(stats) - positives} без неё → {args.out}")
        yaml = args.out / "data.yaml"
        if not yaml.exists():
            yaml.write_text("path: " + str(args.out.resolve()).replace("\\", "/") + "\ntrain: images/train\nval: images/val\n"
                            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(COCO)), encoding="utf-8")


if __name__ == "__main__":
    main()
