"""Луч на кабину: дополнительный ракурс, когда объект видит одна камера (или ни одной).

Белый прицеп на снегу детектор и SAM часто не видят, а кабину крупная YOLO находит
(камера 5, кадры 0–19 Way1: YOLO26l — «truck 0.29» ровно по кабине). Кабина и есть точка-ответ,
поэтому луч из такой камеры через центр найденной кабины — сильное измерение.

На каждом кадре со статусом mono/forecast/motion/flow:
1. для камер без наблюдения модель ставится в текущую позу, проецируется кабина и вся машина;
2. в окне вокруг ожидаемой машины YOLO ищет транспорт размером с кабину рядом с её проекцией;
3. поза поправляется взвешенным МНК: старая оценка точна поперёк луча своей камеры и неточна
   вдоль (дальность), новая камера даёт луч на центр кабины с точностью в несколько пикселей.
Используются только текущий кадр и состояние трекера — режим остаётся причинным.

По умолчанию луч работает только до первой многокамерной засечки: пока дальность известна лишь по
одной камере, луч — единственный второй ракурс (Way1, кадры 0–21: 5.32 → 0.81 м). Позже дальность держит
модель движения, а дальние кабины на камерах 3–4 YOLO путает: поправки 7–11 м с подъёмом на 2 м
(Way1, кадры 645–684 и 781–798) портят ответ, в сумме MAE 3.78 → 4.02 м при работе на всём ролике.
"""
import numpy as np

from online.geometry import Rot, View

VEHICLE_LABELS = (2, 5, 7, 8)  # car, bus, truck, boat — дальнюю кабину YOLO называет по-разному
EDGE_PX = 3


class CabRay:
    def __init__(self, weights, model, conf=0.08, imgsz=640, max_shift_m=12.0, sigma_px=4.0, until_multi=True):
        from ultralytics import YOLO

        self.detector = YOLO(str(weights))
        self.model = model
        cab = model.reference_part if model.reference_part is not None else model.vertices
        self.cab = np.asarray(cab, float)
        self.cab_center = (self.cab.min(axis=0) + self.cab.max(axis=0)) / 2
        self.conf, self.imgsz = conf, imgsz
        self.max_shift_m, self.sigma_px = max_shift_m, sigma_px
        self.applied = None  # (камера, сдвиг, м) на последнем кадре — для replay.json
        self.until_multi = until_multi
        self.seen_multi = False  # была ли уже засечка двумя и более камерами

    def detect(self, image, view, pose):
        """Центр кабины в кадре камеры или None."""
        truck = view.box(pose)
        R = Rot(*pose[3:])
        uv, depth = view.project(self.cab @ R.T + pose[:3])
        if truck is None or depth.min() < 1:
            return None
        cab = np.r_[uv.min(axis=0), uv.max(axis=0)]
        h, w = image.shape[:2]
        if cab[0] < EDGE_PX or cab[1] < EDGE_PX or cab[2] > w - EDGE_PX or cab[3] > h - EDGE_PX:
            return None  # кабина должна целиком попадать в кадр
        side = int(min(h, w, max(176, 1.85 * max(truck[2:] - truck[:2]) + 30)))
        center = (truck[:2] + truck[2:]) / 2
        x, y = np.clip(np.round(center - side / 2).astype(int), 0, [w - side, h - side])
        result = self.detector.predict(image[y:y + side, x:x + side], imgsz=self.imgsz, conf=self.conf,
                                       classes=list(VEHICLE_LABELS), verbose=False)[0]
        cab_size = max(cab[2:] - cab[:2])
        # прогноз по одной камере сдвинут на метры (десятки пикселей) — допуск в размер всей машины
        margin = max(max(truck[2:] - truck[:2]), 30)
        reach = np.r_[truck[:2] - margin, truck[2:] + margin]
        best = None
        for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            box = box + [x, y, x, y]
            if box[0] < EDGE_PX or box[1] < EDGE_PX or box[2] > w - EDGE_PX or box[3] > h - EDGE_PX:
                continue  # обрезанная краем кадра рамка — центр смещён
            mid = (box[:2] + box[2:]) / 2
            size = max(box[2:] - box[:2])
            if not (reach[0] <= mid[0] <= reach[2] and reach[1] <= mid[1] <= reach[3]):
                continue
            if not 0.4 * cab_size <= size <= 2.2 * cab_size:  # не вся фура и не полхолма
                continue
            # ближе к ожидаемой кабине — лучше
            closeness = np.exp(-np.linalg.norm(mid - (cab[:2] + cab[2:]) / 2) / max(cab_size, 8))
            value = float(score) * closeness
            if best is None or value > best[0]:
                best = value, mid
        return None if best is None else best[1]

    def step(self, pose, status, used, cameras, images):
        """Поправленная поза (или та же). used — камеры, давшие наблюдение на этом кадре."""
        self.applied = None
        if len(used) > 1:
            self.seen_multi = True
        if self.until_multi and self.seen_multi:
            return pose
        if pose is None or status not in {"mono", "forecast", "motion", "flow"} or len(used) > 1:
            return pose
        R = Rot(*pose[3:])
        offset = R @ self.cab_center  # от точки-ответа до центра кабины
        p = pose[:3]
        A, b = np.zeros((3, 3)), np.zeros(3)
        # априорная точность текущей оценки
        if used:
            key = next(iter(used))
            origin = cameras[key].loc / 100
            ray = (p - origin) / np.linalg.norm(p - origin)
            along = np.outer(ray, ray)
            dist = np.linalg.norm(p - origin)
            W = along / max(2.0, 0.02 * dist) ** 2 + (np.eye(3) - along) / 0.4**2
        else:
            W = np.eye(3) / 3.0**2
        A += W
        b += W @ p
        rays = []
        for key, camera in cameras.items():
            if key in used or images.get(key) is None:
                continue
            view = View.of(camera, model=self.model)
            mid = self.detect(images[key], view, pose)
            if mid is None:
                continue
            origin, direction = view.ray(mid)
            dist = np.linalg.norm(p + offset - origin)
            M = (np.eye(3) - np.outer(direction, direction)) / (dist * self.sigma_px / camera.fpx) ** 2
            # центр кабины x + offset должен лежать на луче: M (x + offset − origin) = 0
            A += M
            b += M @ (origin - offset)
            rays.append(key)
        if not rays:
            return pose
        x = np.linalg.solve(A, b)
        shift = x - p
        if np.linalg.norm(shift) > self.max_shift_m:
            return pose
        corrected = pose.copy()
        corrected[:3] = x
        self.applied = (rays, float(np.linalg.norm(shift)))
        return corrected
