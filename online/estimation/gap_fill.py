"""Ответ на каждом кадре: заполнение провалов, когда трекер не выдаёт координат.

`ConsensusTracker` после секунды без засечки двумя камерами объявляет потерю и не выдаёт ничего
(Way2: кадры 0–46, 815–1041, 1418–1499, 1907–1932 — 396 кадров без координат). Пустой кадр — худший
ответ: любая разумная оценка ближе к фуре, чем её отсутствие.

GapFill работает только с ВЫХОДОМ: состояние трекера не меняется, поэтому кадры, где трекер выдаёт
позу, остаются бит в бит прежними. На кадре без позы:
1. если фура раньше уже была найдена — прогноз от последней позы: скорость затухает
   (`DECAY` за кадр), высота держится на уровне дороги по последним многокамерным засечкам;
2. прогноз уточняется слабыми измерениями одной камеры: маски, которые трекер нашёл, но отбросил
   (для захвата ему нужны две камеры), и YOLO с низким порогом (0,05) в окне вокруг проекции
   прогноза — дым и пыль на белой фуре роняют уверенность до 0,05–0,15 (Way2, кадр 816, камера 4).
   Измерение принимается, только если CAD в подобранной позе совпадает с маской и поза недалеко
   от прогноза;
3. если фура ещё ни разу не найдена — оценка по одной уверенной детекции (score ≥ 0,45).
Каждая оценка проверяется на правдоподобие: внутри зоны камер и не глубоко под ними.
Используются только текущий и прошлые кадры — режим остаётся причинным.
"""
import numpy as np

from online.geometry import Observation, View, center_of, optimize, with_center
from online.tracker import Estimate, box_overlap

FILLED = ("predicted", "weak", "guess")
DECAY = 0.995           # затухание скорости за кадр в прогнозе (полжизни ~4,6 с на 30 кадр/с)
WEAK_CONF = 0.05
WEAK_OVERLAP = 0.50     # силуэт CAD против маски
GUESS_SCORE = 0.45
GUESS_OVERLAP = 0.65
BELOW_CAMERAS_M = 25.0
SCENE_REACH = 1.2
SCENE_MARGIN_M = 50.0


class GapFill:
    def __init__(self, tracker, model, fps):
        self.tracker = tracker
        self.model = model
        self.fps = fps
        self.pose = None          # последний выданный ответ (трекера или свой)
        self.velocity = np.zeros(3)  # м/кадр
        self.altitude = None
        self.gap = 0
        self.filled = 0

    # ------------------------------------------------------------------ helpers
    def inside_scene(self, pose):
        views = self.tracker.current_views
        if not self.model.ground_vehicle or len(views) < 3:
            return True
        locs = np.array([view.camera.loc for view in views.values()]) / 100
        center = center_of(pose, self.model)
        middle = locs[:, :2].mean(axis=0)
        radius = np.linalg.norm(locs[:, :2] - middle, axis=1).max()
        if np.linalg.norm(center[:2] - middle) > SCENE_REACH * radius + SCENE_MARGIN_M:
            return False
        return center[2] >= locs[:, 2].min() - BELOW_CAMERAS_M

    def limit(self):
        """Насколько измерение может отойти от прогноза: растёт с длиной провала."""
        speed = self.model.speed / self.fps
        return min(60.0, max(4.0, 0.30 * self.model.size) + 0.5 * speed * self.gap)

    def fit_single(self, observation, prior, sigma):
        yaw, pitch = prior[3:5]
        pose = optimize([observation], prior, prior, yaw, pitch, [sigma] * 3,
                        yaw_window=0.8, model=self.model)[0]
        if self.altitude is not None and self.model.ground_vehicle and abs(pose[2] - self.altitude) > 3.0:
            return None
        return pose

    def weak_candidates(self, images, predicted):
        """Маски трекера и слабые детекции YOLO около проекции прогноза, по камерам."""
        found = {}
        for key, group in (getattr(self.tracker, "groups", None) or {}).items():
            found[key] = [ob for ob in group if not ob.view.instance.flow]
        windows = []
        for key, view in self.tracker.current_views.items():
            box = view.box(predicted)
            if box is None or key not in images:
                continue
            width, height = view.camera.width, view.camera.height
            if box[2] < 0 or box[3] < 0 or box[0] > width or box[1] > height or min(box[2:] - box[:2]) < 8:
                continue
            image = images[key]
            side = int(min(height, width, max(320, 3.0 * max(box[2:] - box[:2]))))
            center = (np.clip(box[:2], 0, [width, height]) + np.clip(box[2:], 0, [width, height])) / 2
            x, y = np.clip(np.round(center - side / 2).astype(int), 0, [width - side, height - side])
            windows.append((key, image[y:y + side, x:x + side], (x, y, side, False)))
        segmenter = getattr(self.tracker, "segmenter", None)
        if windows and segmenter is not None and hasattr(segmenter, "detect"):
            for key, instances in segmenter.detect(windows, 640, WEAK_CONF).items():
                view = self.tracker.current_views[key]
                found.setdefault(key, []).extend(Observation(View.of(view.camera, item, self.model))
                                                 for item in instances)
        return found

    # --------------------------------------------------------------------- step
    def step(self, frame, estimate, images):
        if estimate.pose is not None:
            fresh = estimate.status in ("multi", "mask_multi")
            if fresh:
                self.altitude = estimate.pose[2] if self.altitude is None else 0.8 * self.altitude + 0.2 * estimate.pose[2]
            velocity = getattr(self.tracker, "velocity", None)
            if velocity is not None and estimate.status != "forecast":
                self.velocity = np.asarray(velocity, float).copy()
            self.pose = np.asarray(estimate.pose, float).copy()
            self.gap = 0
            return estimate

        self.gap += 1
        if self.pose is None:
            guess = self.initial_guess()
            if guess is None:
                return estimate
            self.pose, self.velocity = guess, np.zeros(3)
            self.filled += 1
            return Estimate(guess.copy(), "guess", {}, 0.0, estimate.tracked_2d)

        predicted = self.pose.copy()
        self.velocity *= DECAY
        predicted[:3] += self.velocity
        if self.altitude is not None and self.model.ground_vehicle:
            predicted[2] = self.altitude
        measured = self.measure(images, predicted)
        if measured is not None:
            pose, key, instance, quality = measured
            change = center_of(pose, self.model) - center_of(self.pose, self.model)
            self.velocity += 0.2 * (change - self.velocity)
            limit = self.model.speed / self.fps
            self.velocity *= min(1.0, limit / max(np.linalg.norm(self.velocity), 1e-9))
            self.pose = pose
            self.filled += 1
            return Estimate(pose.copy(), "weak", {key: instance}, quality, estimate.tracked_2d)
        self.pose = predicted
        self.filled += 1
        return Estimate(predicted.copy(), "predicted", {}, 0.0, estimate.tracked_2d)

    def measure(self, images, predicted):
        limit = self.limit()
        sigma = max(1.0, 0.5 * limit)
        best = None
        for key, observations in self.weak_candidates(images, predicted).items():
            view = self.tracker.current_views[key]
            expected = view.box(predicted)
            for ob in observations:
                if self.model.labels and ob.view.instance.label not in self.model.labels and ob.view.instance.label >= 0:
                    continue
                if expected is not None and box_overlap(ob.box, expected) < 0.05 and self.gap < 15:
                    continue  # в начале провала прогноз точен: маска должна быть рядом
                # старт подгонки — на луче маски, на дальности прогноза
                origin, ray = ob.view.ray(ob.center)
                depth = np.linalg.norm(center_of(predicted, self.model) - origin)
                seed = with_center(predicted, origin + ray * depth, self.model)
                pose = self.fit_single(ob, seed, sigma)
                if pose is None or not self.inside_scene(pose):
                    continue
                if np.linalg.norm(center_of(pose, self.model) - center_of(predicted, self.model)) > limit:
                    continue
                quality = ob.overlap(pose)
                if quality < WEAK_OVERLAP:
                    continue
                score = quality * np.sqrt(max(ob.view.instance.score, 1e-3)) * (0.7 if ob.clipped else 1.0)
                if best is None or score > best[0]:
                    best = score, pose, key, ob.view.instance, quality
        return None if best is None else best[1:]

    def initial_guess(self):
        best = None
        for key, group in (getattr(self.tracker, "groups", None) or {}).items():
            for ob in group:
                if ob.clipped or ob.view.instance.flow or ob.view.instance.score < GUESS_SCORE or min(ob.size) < 18:
                    continue
                pose = self.tracker.fit_pose([ob], None)
                if not self.inside_scene(pose) or ob.overlap(pose) < GUESS_OVERLAP:
                    continue
                if best is None or ob.view.instance.score > best[0]:
                    best = ob.view.instance.score, pose
        return None if best is None else best[1]
