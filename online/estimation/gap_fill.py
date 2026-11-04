"""Ответ на каждом кадре: заполнение провалов, когда трекер не выдаёт координат.

`ConsensusTracker` после секунды без засечки двумя камерами объявляет потерю и не выдаёт ничего
(Way2: кадры 0–46, 815–1041, 1418–1499, 1907–1932 — 396 кадров без координат).

GapFill работает только с ВЫХОДОМ: состояние трекера не меняется, поэтому кадры, где трекер выдаёт
позу, остаются бит в бит прежними. На кадре без позы ответ строится так:

1. Прогноз от последнего ответа. Скорость — наклон траектории за последнюю секунду надёжных кадров
   трекера (а не сглаженная скорость трекера, которая в прогнозе затухает), высота — уровень дороги
   по засечкам двумя и более камерами.
2. Поправка по наблюдениям. Кандидаты: 2D-маски SAM2, которые трекер сопровождает после пропуска
   YOLO (`tracked_2d`), маски, найденные трекером, но отброшенные (ему для захвата нужны две
   камеры), и YOLO с порогом 0,05 в окне вокруг проекции прогноза — дым и пыль на белой фуре роняют
   уверенность до 0,05–0,15 (Way2, кадр 816, камера 4).
   * две камеры с согласованными лучами — триангуляция центра;
   * одна камера — направление на фуру надёжно, дальность нет: берём ближайшую к прогнозу точку
     луча, то есть правим только поперечную к лучу составляющую. Скорость прогноза на таких
     поправках не учится.
   Кандидат принимается, если силуэт CAD в новой позе совпадает с маской.
3. Плавность. Ответ тянется к измерению частично и не сдвигается за кадр дальше, чем позволяет
   скорость машины (`step_limit`): никаких рывков на десятки метров внутри провала.
4. До первого захвата — триангуляция двух свежих детекций с согласованными лучами, а если вторая
   камера фуру не видит — оценка по одной уверенной детекции.

Каждая оценка проверяется на правдоподобие: внутри зоны камер и не глубоко под ними.
Используются только текущий и прошлые кадры — режим остаётся причинным.
"""
from collections import deque
from itertools import combinations

import numpy as np

from online.geometry import Observation, View, center_of, with_center
from online.tracker import Estimate, box_overlap

FILLED = ("predicted", "weak", "guess")
DECAY = 0.99             # затухание скорости за кадр в прогнозе (полжизни ~2,3 с на 30 кадр/с)
HISTORY_S = 1.0          # окно для скорости по надёжным кадрам трекера
WEAK_CONF = 0.05
MATCH_OVERLAP = 0.35     # силуэт CAD против маски для поправки
NEAR_PROJECTION = 0.30   # слабая детекция / отброшенная маска: IoU с проекцией прогноза
GUESS_SCORE = 0.45
GUESS_OVERLAP = 0.55
PULL_SINGLE = 0.35       # доля поправки к измерению одной камеры за кадр
PULL_PAIR = 0.6          # и к триангуляции двумя камерами
BELOW_CAMERAS_M = 25.0
SCENE_REACH = 1.2
SCENE_MARGIN_M = 50.0
RELIABLE = ("multi", "mono", "mask_multi")


def closest_point(origin, ray, point):
    return origin + ray * max(0.0, float((point - origin) @ ray))


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
        self.velocity = np.zeros(3)  # м/кадр, центр CAD
        self.altitude = None
        self.history = deque(maxlen=max(2, round(HISTORY_S * fps)))
        self.gap = 0
        self.filled = 0

    # ------------------------------------------------------------------ helpers
    def step_limit(self):
        """Наибольший сдвиг ответа за кадр: как у трекера — две предельные скорости, не меньше 1,5 м."""
        return max(1.5, 2 * self.model.speed / self.fps)

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

    def road_center_z(self, pose):
        if self.altitude is None:
            return None
        return center_of(np.r_[pose[:2], self.altitude, pose[3:]], self.model)[2]

    def remember(self, frame, pose, status):
        if status in RELIABLE:
            self.history.append((frame, center_of(pose, self.model)))
            if status == "multi":
                self.altitude = pose[2] if self.altitude is None else 0.8 * self.altitude + 0.2 * pose[2]
            if len(self.history) >= 4:
                frames = np.array([f for f, _ in self.history], float)
                centers = np.array([c for _, c in self.history])
                slope = np.polyfit(frames - frames[-1], centers, 1)[0]
                if self.model.ground_vehicle:
                    slope[2] = 0.0
                limit = self.model.speed / self.fps
                self.velocity = slope * min(1.0, limit / max(np.linalg.norm(slope), 1e-9))

    def candidates(self, images, predicted, tracked_2d):
        """Наблюдения по камерам: 2D-маски SAM2, маски трекера и слабые детекции около прогноза."""
        found = {}
        near = {}  # проекция прогноза по камерам: слабые детекции и отброшенные маски должны быть рядом
        for key, view in self.tracker.current_views.items():
            box = view.box(predicted) if predicted is not None else None
            if box is not None:
                near[key] = box
        for key, instance in (tracked_2d or {}).items():
            view = self.tracker.current_views.get(key)
            if view is not None:
                found.setdefault(key, []).append(Observation(View.of(view.camera, instance, self.model)))
        for key, group in (getattr(self.tracker, "groups", None) or {}).items():
            found.setdefault(key, []).extend(ob for ob in group if not ob.view.instance.flow
                                             and key in near and box_overlap(ob.box, near[key]) >= NEAR_PROJECTION)
        windows = []
        for key, view in self.tracker.current_views.items():
            box = view.box(predicted) if predicted is not None else None
            if box is None or key not in images or key in found:
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
                found.setdefault(key, []).extend(Observation(View.of(view.camera, item, self.model))
                                                 for item in instances
                                                 if key in near and box_overlap(item.box, near[key]) >= NEAR_PROJECTION)
        labels = self.model.labels
        return {key: [ob for ob in obs if not labels or ob.view.instance.label < 0 or ob.view.instance.label in labels]
                for key, obs in found.items()}

    def single_point(self, observation, center):
        """Ближайшая к прогнозу точка луча: правится только поперечная к лучу составляющая.

        Дальность по одной камере не берём: ни по размеру маски (пыль прячет часть фуры), ни по
        пересечению луча с дорогой (рельеф и неточная высота — Way2, кадры 853–901: ответ уехал
        по лучу камеры 4 назад на 40 м, а потом прогноз выучил эту ложную скорость)."""
        origin, ray = observation.view.ray(observation.center)
        return closest_point(origin, ray, center)

    def measure(self, images, predicted, tracked_2d):
        """Лучшая поправка: (точка центра, доля, камеры, наблюдение) или None."""
        center = center_of(predicted, self.model)
        groups = self.candidates(images, predicted, tracked_2d)
        reach = max(6.0, 0.6 * self.model.size) + 0.5 * self.gap * np.linalg.norm(self.velocity) + 0.15 * self.gap
        best = None
        flat = [(key, ob) for key, obs in groups.items() for ob in obs]
        for (ka, a), (kb, b) in combinations(flat, 2):
            if ka == kb or a.clipped or b.clipped:
                continue
            point, gap = triangulate_pair(a.view.ray(a.center), b.view.ray(b.center))
            if point is None or gap > 0.3 * self.model.size or np.linalg.norm(point - center) > reach:
                continue
            pose = with_center(predicted, point, self.model)
            if not self.inside_scene(pose):
                continue
            quality = min(a.overlap(pose), b.overlap(pose))
            if quality >= MATCH_OVERLAP and (best is None or 1 + quality > best[0]):
                best = 1 + quality, point, PULL_PAIR, (ka, kb), a, quality
        if best is not None:
            return best[1:]
        for key, ob in flat:
            point = self.single_point(ob, center)
            if np.linalg.norm(point - center) > reach:
                continue
            pose = with_center(predicted, point, self.model)
            if not self.inside_scene(pose):
                continue
            quality = ob.overlap(pose)
            score = quality * np.sqrt(max(ob.view.instance.score, 1e-3)) * (0.7 if ob.clipped else 1.0)
            if quality >= MATCH_OVERLAP and (best is None or score > best[0]):
                best = score, point, PULL_SINGLE, (key,), ob, quality
        return None if best is None else best[1:]

    def move_to(self, target_center, base_pose):
        """Сдвиг ответа к цели с ограничением на кадр."""
        current = center_of(self.pose, self.model)
        step = target_center - current
        limit = self.step_limit()
        if np.linalg.norm(step) > limit:
            step *= limit / np.linalg.norm(step)
        pose = with_center(base_pose, current + step, self.model)
        if self.model.ground_vehicle and np.linalg.norm(self.velocity[:2]) * self.fps > 1.0:
            pose[3] = np.arctan2(self.velocity[1], self.velocity[0])  # курс по движению
            pose = with_center(pose, current + step, self.model)
        return pose

    # --------------------------------------------------------------------- step
    def step(self, frame, estimate, images):
        if estimate.pose is not None:
            self.remember(frame, estimate.pose, estimate.status)
            self.pose = np.asarray(estimate.pose, float).copy()
            self.gap = 0
            return estimate

        self.gap += 1
        if self.pose is None:
            guess = self.initial_guess()
            if guess is None:
                return estimate
            self.pose = guess
            self.filled += 1
            return Estimate(guess.copy(), "guess", {}, 0.0, estimate.tracked_2d)

        self.velocity *= DECAY
        predicted = self.pose.copy()
        predicted = with_center(predicted, center_of(predicted, self.model) + self.velocity, self.model)
        road = self.road_center_z(predicted)
        if road is not None and self.model.ground_vehicle:
            target = center_of(predicted, self.model)
            target[2] = road
            predicted = with_center(predicted, target, self.model)
        measured = self.measure(images, predicted, estimate.tracked_2d)
        if measured is None:
            target = center_of(predicted, self.model)
            status, views, quality = "predicted", {}, 0.0
        else:
            point, pull, keys, ob, quality = measured
            target = center_of(predicted, self.model) + pull * (point - center_of(predicted, self.model))
            if len(keys) > 1:  # скорость учим только на триангуляции: одна камера даёт лишь направление
                change = target - center_of(self.pose, self.model)
                self.velocity += 0.10 * (change - self.velocity)
                if self.model.ground_vehicle:
                    self.velocity[2] = 0.0
                limit = self.model.speed / self.fps
                self.velocity *= min(1.0, limit / max(np.linalg.norm(self.velocity), 1e-9))
            status, views = "weak", {key: ob.view.instance for key in keys[:1]}
        self.pose = self.move_to(target, predicted)
        self.filled += 1
        return Estimate(self.pose.copy(), status, views, quality, estimate.tracked_2d)

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
            pose = self.tracker.fit_pose([a, b], None)
            pose = with_center(pose, point, self.model)
            if not self.inside_scene(pose):
                continue
            quality = min(a.overlap(pose), b.overlap(pose))
            score = 2 + quality
            if quality >= 0.25 and (best is None or score > best[0]):
                best = score, pose
        if best is None:
            for key, ob in fresh:
                if ob.view.instance.score < GUESS_SCORE or min(ob.size) < 18:
                    continue
                pose = self.tracker.fit_pose([ob], None)
                if not self.inside_scene(pose) or ob.overlap(pose) < GUESS_OVERLAP:
                    continue
                if best is None or ob.view.instance.score > best[0]:
                    best = ob.view.instance.score, pose
        if best is None:
            return None
        pose = best[1]
        if self.pose is not None:  # плавно и для серии первых оценок
            current = center_of(self.pose, self.model)
            step = center_of(pose, self.model) - current
            limit = self.step_limit()
            if np.linalg.norm(step) > limit:
                pose = with_center(pose, current + step * limit / np.linalg.norm(step), self.model)
        return pose
