"""Ответ на каждом кадре: заполнение провалов, когда трекер не выдаёт координат.

`ConsensusTracker` после секунды без засечки двумя камерами объявляет потерю и не выдаёт ничего
(Way2: кадры 0–46, 815–1041, 1418–1499, 1907–1932 — 396 кадров без координат).

GapFill работает только с ВЫХОДОМ: состояние трекера не меняется, поэтому кадры, где трекер выдаёт
позу, остаются бит в бит прежними. На кадре без позы ответ даёт фильтр Калмана по точке ответа
на плоскости дороги (x, y, vx, vy). Ведём именно точку ответа, а не центр CAD: она у кабины, в 10 м
от центра, и при смене курса центр поворачивался бы вокруг неё, давая скачки ответа до 20 м.
Курс поворачивается к направлению движения плавно, не больше `TURN_DEG` за кадр.

* Прогноз — постоянная скорость с шумом ускорения (`ACCEL_SIGMA`). Начальная скорость — наклон
  траектории за последнюю секунду надёжных кадров трекера. Высота — уровень дороги по кадрам `multi`.
* Измерения по камерам, каждое со своей точностью:
  - направление на центр рамки (по горизонтали) — надёжно; если рамка обрезана краем кадра,
    вместо центра берётся её видимая сторона против крайней точки проекции CAD;
  - дальность по высоте рамки — слабо: на Way2 высота рамки в среднем 0,87 высоты проекции CAD
    при разбросе 0,62–1,15, поэтому поправка на 0,87 и погрешность 30 %;
  - две и больше камер фильтр складывает сам — получается триангуляция.
  Источники: 2D-маски SAM2 (`tracked_2d`), маски трекера, которые он не принял (для захвата ему
  нужны две камеры), и YOLO с порогом 0,05 около проекции прогноза — дым и пыль роняют уверенность
  YOLO на белой фуре до 0,05–0,15. Последние два источника — только если рамка была и на прошлом
  кадре (защита от одиночных ложных срабатываний).
* Измерение, несовместимое с прогнозом с учётом обеих неопределённостей (расстояние Махаланобиса
  больше 3σ), отбрасывается. Так ошибочная маска не утаскивает ответ (v2: 172 м на кадре 1042).
* Ответ не сдвигается за кадр дальше `max(1,5 м, 2 × предельная скорость / fps)`.
* До первого захвата — триангуляция двух свежих детекций, иначе одна уверенная детекция; дальше
  тот же фильтр с большой начальной неопределённостью.

Используются только текущий и прошлые кадры — режим остаётся причинным.
"""
from collections import deque
from itertools import combinations

import numpy as np

from online.geometry import Observation, Rot, View, center_of, with_center
from online.tracker import Estimate, box_overlap

FILLED = ("predicted", "weak", "guess")
HISTORY_S = 1.0          # окно для начальной скорости по надёжным кадрам трекера
ACCEL_SIGMA = 1.5        # м/с² — шум ускорения в прогнозе (повороты, разгон)
BLIND_DECAY = 1.0        # затухание скорости вслепую: 1 — постоянная скорость (на Way2 лучше по сумме стыков 1042 и 1500)
RESET_SIGMA = {          # м — неопределённость позы трекера по тому, как она получена
    "multi": 0.5, "mask_multi": 1.0, "cad_points": 1.0, "cad_flow": 1.5, "cad_region": 1.5,
    "mono": 2.5, "flow": 3.0, "motion": 3.0, "mask_large": 3.0,
}
RESET_VEL_SIGMA = 1.5    # м/с
TAKE_OVER = ("forecast",)  # кадры трекера без наблюдений, которые заполнение берёт на себя
                           # (Way1, 383–411: прогноз трекера уехал на 44 м, а маски видны в 3 камерах)
INFLATE = 2.0            # во сколько раз растёт неопределённость, если все наблюдения кадра отброшены
INFLATE_MAX_M = 60.0
REACQUIRE_FRAMES = 3     # кадров подряд, когда две камеры согласны между собой, но не с фильтром
REACQUIRE_SIGMA = 2.0
GUESS_POS_SIGMA = 15.0
GUESS_VEL_SIGMA = 6.0
CENTER_PX = 6.0          # погрешность направления на центр рамки: max(6 px, 12 % ширины)
CENTER_REL = 0.12
EDGE_REL = 0.15          # видимая сторона обрезанной рамки: 15 % ширины проекции
HEIGHT_RATIO = 0.87      # высота рамки / высота проекции CAD (медиана по Way2)
HEIGHT_REL = 0.30
HEIGHT_MODE = "window"   # "window": высота рамки в обе стороны, если отношение к ожидаемой в HEIGHT_WINDOW;
                         # "closer": только если рамка больше ожидаемой; "off" — не брать
HEIGHT_WINDOW = (0.75, 1.5)  # рамка сильно ниже ожидаемой — скорее закрытие (дерево, пыль), чем дальность
WEAK_SCALE = 1.5         # слабые детекции и отброшенные маски — погрешность в 1,5 раза больше
GATE = {1: 8.8, 2: 11.6}  # χ² на 99,7 % — порог Махаланобиса
WEAK_CONF = 0.05
NEAR_PROJECTION = 0.30
GUESS_SCORE = 0.45
GUESS_OVERLAP = 0.55
BELOW_CAMERAS_M = 25.0
SCENE_REACH = 1.2
SCENE_MARGIN_M = 50.0
RELIABLE = ("multi", "mono", "mask_multi")
TURN_DEG = 1.5           # наибольший поворот курса за кадр в заполнении
TURN_SPEED = 3.0         # м/с — ниже этой скорости курс по движению не меняем
EDGE_PX = 4


def scene_plausible(pose, views, model):
    """Наземная цель внутри зоны, которую снимают камеры, и не глубоко под ними.

    Way2, кадр 6: огонь в камере 1 по размеру маски дал ~1 км (761 м от центра расстановки при её
    радиусе 468 м) и 44 м ниже камер."""
    if not model.ground_vehicle or len(views) < 3:
        return True
    locs = np.array([view.camera.loc for view in views.values()]) / 100
    center = center_of(pose, model)
    middle = locs[:, :2].mean(axis=0)
    radius = np.linalg.norm(locs[:, :2] - middle, axis=1).max()
    if np.linalg.norm(center[:2] - middle) > SCENE_REACH * radius + SCENE_MARGIN_M:
        return False
    return center[2] >= locs[:, 2].min() - BELOW_CAMERAS_M


def triangulate_pair(a, b):
    """Середина общего перпендикуляра двух лучей и расстояние между ними."""
    (oa, ra), (ob, rb) = a, b
    cosine = ra @ rb
    separation = 1 - cosine ** 2
    if separation < 0.01:
        return None, np.inf
    offset = ob - oa
    p, q = offset @ ra, offset @ rb
    ta, tb = (p - cosine * q) / separation, (cosine * p - q) / separation
    if min(ta, tb) <= 2:
        return None, np.inf
    pa, pb = oa + ra * ta, ob + rb * tb
    return (pa + pb) / 2, float(np.linalg.norm(pa - pb))


class GapFill:
    def __init__(self, tracker, model, fps):
        self.tracker = tracker
        self.model = model
        self.fps = fps
        self.pose = None             # последний выданный ответ
        self.template = None         # поза-шаблон (курс, наклон) последнего кадра трекера
        self.yaw = None
        self.x = None                # [x, y, vx, vy] точки ответа, м и м/кадр
        self.P = None
        self.answer_z = None         # высота точки ответа на дороге
        self.history = deque(maxlen=max(2, round(HISTORY_S * fps)))
        self.previous_boxes = {}     # рамки-кандидаты прошлого кадра по камерам
        self.disagree = 0            # кадров подряд, когда две камеры согласны, а фильтр — нет
        self.gap = 0
        self.filled = 0

    # ------------------------------------------------------------------ helpers
    def step_limit(self):
        return max(1.5, 2 * self.model.speed / self.fps)

    def inside_scene(self, pose):
        return scene_plausible(pose, self.tracker.current_views, self.model)

    def pose_at(self, xy):
        """Поза из точки ответа на дороге и текущего курса."""
        pose = self.template.copy()
        pose[0], pose[1] = xy
        if self.answer_z is not None and self.model.ground_vehicle:
            pose[2] = self.answer_z
        if self.yaw is not None:
            pose[3] = self.yaw
        return pose

    def turn(self):
        """Курс плавно к направлению движения."""
        if not self.model.ground_vehicle or np.linalg.norm(self.x[2:]) * self.fps < TURN_SPEED:
            return
        course = np.arctan2(self.x[3], self.x[2])
        change = (course - self.yaw + np.pi) % (2 * np.pi) - np.pi
        self.yaw += float(np.clip(change, -np.radians(TURN_DEG), np.radians(TURN_DEG)))

    def reset(self, pose, velocity, pos_sigma, vel_sigma):
        self.template = np.asarray(pose, float).copy()
        self.yaw = float(self.template[3]) if len(self.template) > 3 else None
        self.x = np.r_[self.template[:2], velocity[:2]]
        self.P = np.diag([pos_sigma ** 2] * 2 + [(vel_sigma / self.fps) ** 2] * 2)

    def remember(self, frame, pose, status):
        """Кадр трекера с наблюдением: фильтр начинается заново от его позы."""
        point = np.asarray(pose[:3], float)
        if status in RELIABLE:
            self.history.append((frame, point))
        if status == "multi":
            self.answer_z = point[2] if self.answer_z is None else 0.8 * self.answer_z + 0.2 * point[2]
        velocity = np.zeros(3)
        if len(self.history) >= 4:
            frames = np.array([f for f, _ in self.history], float)
            centers = np.array([c for _, c in self.history])
            velocity = np.polyfit(frames - frames[-1], centers, 1)[0]
            limit = self.model.speed / self.fps
            velocity *= min(1.0, limit / max(np.linalg.norm(velocity[:2]), 1e-9))
        self.reset(pose, velocity, RESET_SIGMA.get(status, 2.0), RESET_VEL_SIGMA)
        self.disagree = 0

    # ---------------------------------------------------------- measurements
    def candidates(self, images, predicted, tracked_2d):
        """[(камера, наблюдение, надёжное ли)] — маски SAM2, маски трекера, слабые детекции."""
        near = {}
        for key, view in self.tracker.current_views.items():
            box = view.box(predicted)
            if box is not None:
                near[key] = box
        found = []
        for key, instance in (tracked_2d or {}).items():
            view = self.tracker.current_views.get(key)
            if view is not None:
                found.append((key, Observation(View.of(view.camera, instance, self.model)), True))
        weak = []
        for key, group in (getattr(self.tracker, "groups", None) or {}).items():
            weak.extend((key, ob) for ob in group if not ob.view.instance.flow)
        windows = []
        for key, view in self.tracker.current_views.items():
            box = near.get(key)
            if box is None or key not in images or key in (tracked_2d or {}):
                continue
            width, height = view.camera.width, view.camera.height
            if box[2] < 0 or box[3] < 0 or box[0] > width or box[1] > height or min(box[2:] - box[:2]) < 8:
                continue
            side = int(min(height, width, max(320, 3.0 * max(box[2:] - box[:2]))))
            middle = (np.clip(box[:2], 0, [width, height]) + np.clip(box[2:], 0, [width, height])) / 2
            x, y = np.clip(np.round(middle - side / 2).astype(int), 0, [width - side, height - side])
            windows.append((key, images[key][y:y + side, x:x + side], (x, y, side, False)))
        segmenter = getattr(self.tracker, "segmenter", None)
        if windows and segmenter is not None and hasattr(segmenter, "detect"):
            for key, instances in segmenter.detect(windows, 640, WEAK_CONF).items():
                view = self.tracker.current_views[key]
                weak.extend((key, Observation(View.of(view.camera, item, self.model))) for item in instances)
        labels = self.model.labels
        boxes = {}
        for key, ob in weak:
            label = ob.view.instance.label
            if labels and label >= 0 and label not in labels:
                continue
            if key not in near or box_overlap(ob.box, near[key]) < NEAR_PROJECTION:
                continue
            boxes.setdefault(key, []).append(ob.box)
            # одиночные срабатывания не берём: рамка должна быть и на прошлом кадре
            if any(box_overlap(ob.box, old) >= 0.3 for old in self.previous_boxes.get(key, ())):
                found.append((key, ob, False))
        for key, ob, reliable in found:
            if reliable:
                boxes.setdefault(key, []).append(ob.box)
        self.previous_boxes = boxes
        return found

    def measurement_model(self, observation):
        """(h, измерение, погрешности, прогноз измерения) для одного наблюдения или None."""
        view = observation.view
        width, height = view.camera.width, view.camera.height
        box = observation.box
        clip_left, clip_right = box[0] < EDGE_PX, box[2] > width - EDGE_PX
        clip_top, clip_bottom = box[1] < EDGE_PX, box[3] > height - EDGE_PX
        if (clip_left and clip_right) or min(box[2:] - box[:2]) < 6:
            return None
        vertices = self.model.vertices
        use_height = not (clip_top or clip_bottom)

        def h(xy):
            pose = self.pose_at(xy)
            uv, depth = view.project(vertices @ Rot(*pose[3:]).T + pose[:3])
            if depth.min() < 1:
                return None
            if clip_left:
                values = [uv[:, 0].max()]
            elif clip_right:
                values = [uv[:, 0].min()]
            else:
                values = [view.project(center_of(pose, self.model)[None])[0][0, 0]]
            if use_height:
                values.append(HEIGHT_RATIO * (uv[:, 1].max() - uv[:, 1].min()))
            return np.array(values), uv

        first = h(self.x[:2])
        if first is None:
            return None
        predicted, uv = first
        proj_width = max(uv[:, 0].max() - uv[:, 0].min(), 1.0)
        if clip_left or clip_right:
            measured = [box[2] if clip_left else box[0]]
            sigma = [max(CENTER_PX, EDGE_REL * proj_width)]
        else:
            measured = [(box[0] + box[2]) / 2]
            sigma = [max(CENTER_PX, CENTER_REL * (box[2] - box[0]))]
        if use_height:
            measured.append(box[3] - box[1])
            sigma.append(max(CENTER_PX, HEIGHT_REL * (box[3] - box[1])))
        return h, np.array(measured), np.array(sigma), predicted

    def update(self, observation, reliable):
        """Шаг фильтра по одному наблюдению; True, если принято."""
        model = self.measurement_model(observation)
        if model is None:
            return False
        h, z, sigma, predicted = model
        if not reliable:
            sigma = sigma * WEAK_SCALE
        delta = 0.3
        J = np.zeros((len(z), 4))
        for axis in range(2):
            step = np.zeros(2)
            step[axis] = delta
            plus, minus = h(self.x[:2] + step), h(self.x[:2] - step)
            if plus is None or minus is None:
                return False
            J[:, axis] = (plus[0] - minus[0]) / (2 * delta)
        residual = z - predicted
        ratio = z[1] / max(predicted[1], 1e-6) if len(z) == 2 else 1.0
        if len(z) == 2 and (HEIGHT_MODE == "off" or (HEIGHT_MODE == "closer" and residual[1] < 0)
                            or (HEIGHT_MODE == "window" and not HEIGHT_WINDOW[0] <= ratio <= HEIGHT_WINDOW[1])):
            # рамка ниже ожидаемой — это может быть закрытие (дерево, пыль), а не дальность:
            # закрытие делает рамку только меньше, поэтому верим высоте, лишь когда фура «ближе»
            z, predicted, sigma, J, residual = z[:1], predicted[:1], sigma[:1], J[:1], residual[:1]
        S = J @ self.P @ J.T + np.diag(sigma ** 2)
        distance = float(residual @ np.linalg.solve(S, residual))
        if distance > GATE.get(len(z), 11.6):
            return False
        K = self.P @ J.T @ np.linalg.inv(S)
        x = self.x + K @ residual
        if not self.inside_scene(self.pose_at(x[:2])):
            return False
        self.x = x
        self.P = (np.eye(4) - K @ J) @ self.P
        speed = np.linalg.norm(self.x[2:])
        limit = self.model.speed / self.fps
        if speed > limit:
            self.x[2:] *= limit / speed
        return True

    def predict(self):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = 1.0
        q = (ACCEL_SIGMA / self.fps ** 2) ** 2
        block = q * np.array([[0.25, 0.5], [0.5, 1.0]])
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = block
        Q[np.ix_([1, 3], [1, 3])] = block
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # --------------------------------------------------------------------- step
    def step(self, frame, estimate, images):
        if estimate.pose is not None and (estimate.status not in TAKE_OVER or self.x is None):
            self.remember(frame, estimate.pose, estimate.status)
            self.pose = np.asarray(estimate.pose, float).copy()
            self.gap = 0
            return estimate

        self.gap += 1
        if self.x is None:
            guess = self.initial_guess()
            if guess is None:
                return estimate
            self.reset(guess, np.zeros(3), GUESS_POS_SIGMA, GUESS_VEL_SIGMA)
            self.pose = guess
            self.filled += 1
            return Estimate(guess.copy(), "guess", {}, 0.0, estimate.tracked_2d)

        self.predict()
        self.turn()
        predicted = self.pose_at(self.x[:2])
        used = {}
        candidates = sorted(self.candidates(images, predicted, estimate.tracked_2d), key=lambda item: not item[2])
        rejected = 0
        for key, ob, reliable in candidates:
            if key in used:
                continue  # одно наблюдение на камеру за кадр, надёжные первыми
            if self.update(ob, reliable):
                used[key] = ob.view.instance
            else:
                rejected += 1
        if not used and rejected:
            self.recover(candidates)
        if not used:  # совсем вслепую скорость затухает: машина могла притормозить или свернуть
            self.x[2:] *= BLIND_DECAY
        pose = self.pose_at(self.x[:2])
        if self.pose is not None:  # плавность: точка ответа не дальше step_limit за кадр
            move = pose[:3] - self.pose[:3]
            limit = self.step_limit()
            if np.linalg.norm(move) > limit:
                pose[:3] = self.pose[:3] + move * limit / np.linalg.norm(move)
        self.pose = pose
        self.filled += 1
        status = "weak" if used else "predicted"
        return Estimate(pose.copy(), status, used, 0.3 if used else 0.0, estimate.tracked_2d)

    def recover(self, candidates):
        """Все наблюдения кадра отброшены: фильтр мог уехать сам (стартовал с ошибочной позы).

        Неопределённость растёт, и, если две камеры несколько кадров подряд сходятся в одной точке,
        фильтр перезапускается от их триангуляции."""
        position = self.P[:2, :2]
        if np.sqrt(np.trace(position) / 2) < INFLATE_MAX_M:
            self.P[:2, :2] = position * INFLATE ** 2
        best = None
        for (ka, a, _), (kb, b, _) in combinations(candidates, 2):
            if ka == kb or a.clipped or b.clipped:
                continue
            point, gap = triangulate_pair(a.view.ray(a.center), b.view.ray(b.center))
            if point is None or gap > 0.3 * self.model.size:
                continue
            pose = self.pose_at(point[:2])
            if not self.inside_scene(pose):
                continue
            quality = min(a.overlap(pose), b.overlap(pose))
            if best is None or quality > best[0]:
                best = quality, point
        if best is None:
            self.disagree = 0
            return
        self.disagree += 1
        if self.disagree >= REACQUIRE_FRAMES:
            # точка — центр рамок, а фильтр ведёт точку ответа: переводим через текущую позу
            pose = with_center(self.pose_at(self.x[:2]), best[1], self.model)
            self.x[:2] = pose[:2]
            self.P[:2, :2] = np.eye(2) * REACQUIRE_SIGMA ** 2
            self.disagree = 0

    def initial_guess(self):
        """До первого захвата: две камеры с согласованными лучами, иначе одна уверенная детекция."""
        groups = getattr(self.tracker, "groups", None) or {}
        fresh = [(key, ob) for key, obs in groups.items() for ob in obs
                 if not ob.view.instance.flow and not ob.clipped and min(ob.size) >= 12]
        best = None
        for (ka, a), (kb, b) in combinations(fresh, 2):
            if ka == kb:
                continue
            point, gap = triangulate_pair(a.view.ray(a.center), b.view.ray(b.center))
            if point is None or gap > 0.3 * self.model.size:
                continue
            pose = with_center(self.tracker.fit_pose([a, b], None), point, self.model)
            if not self.inside_scene(pose):
                continue
            quality = min(a.overlap(pose), b.overlap(pose))
            if quality >= 0.25 and (best is None or 2 + quality > best[0]):
                best = 2 + quality, pose
        if best is None:
            for key, ob in fresh:
                if ob.view.instance.score < GUESS_SCORE or min(ob.size) < 18:
                    continue
                pose = self.tracker.fit_pose([ob], None)
                if not self.inside_scene(pose) or ob.overlap(pose) < GUESS_OVERLAP:
                    continue
                if best is None or ob.view.instance.score > best[0]:
                    best = ob.view.instance.score, pose
        return None if best is None else best[1]
