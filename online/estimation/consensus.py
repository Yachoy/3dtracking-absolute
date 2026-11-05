"""Acquire and track one rigid target from calibrated, synchronized views."""
from itertools import combinations

import cv2
import numpy as np

from online.geometry import Observation, View, angle_diff, center_of, fit, intersects, linear_seed, optimize, with_center
from online.perception.vision import Instance, transport
from online.tracker import Estimate, box_overlap


class GlobalSegmenter:
    def __init__(self, weights, labels):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.labels = list(labels) or None

    def __call__(self, images):
        windows = [(key, image, None) for key, image in images.items()]
        return self.detect(windows, 960, 0.16)

    def overhead(self, images):
        return self.tiles(images, rotate=True)

    def tiles(self, images, rotate=False):
        windows = []
        for key, image in images.items():
            height, width = image.shape[:2]
            side = max(1, min(width, round(height * 2 / 3)))
            step = max(1, round(side * 0.55))
            xs = sorted({*range(0, width - side + 1, step), width - side})
            ys = sorted({*range(0, height - side + 1, step), height - side})
            for y in ys:
                for x in xs:
                    tile = image[y:y + side, x:x + side]
                    if rotate:
                        tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
                    windows.append((key, tile, (x, y, side, rotate)))
        return self.detect(windows, 768, 0.16)

    def regions(self, images, boxes):
        windows = []
        for key, box in boxes.items():
            image = images[key]
            height, width = image.shape[:2]
            side = min(height, width, round(max(160, 2.0 * max(box[2:] - box[:2]))))
            center = (box[:2] + box[2:]) / 2
            x, y = np.clip(np.round(center - side / 2).astype(int), 0, [width - side, height - side])
            windows.append((key, image[y:y + side, x:x + side], (x, y, side, False)))
        return self.detect(windows, 640, 0.30)

    def detect(self, windows, image_size, confidence):
        detections = {}
        for start in range(0, len(windows), 4):
            batch = windows[start:start + 4]
            results = self.model.predict([image for _, image, _ in batch], imgsz=image_size, rect=True,
                                         classes=self.labels, conf=confidence, max_det=5, verbose=False)
            for (key, _, tile), result in zip(batch, results):
                if result.masks is None:
                    continue
                boxes = result.boxes.cpu().numpy()
                for polygon, score, label in zip(result.masks.xy, boxes.conf, boxes.cls):
                    if len(polygon) < 3 or min(np.ptp(polygon, axis=0)) < 5:
                        continue
                    if tile is not None:
                        x, y, side, rotate = tile
                        if np.any(polygon.min(axis=0) < 3) or np.any(polygon.max(axis=0) > side - 4):
                            continue
                        if rotate:
                            polygon = np.column_stack((polygon[:, 1], side - 1 - polygon[:, 0]))
                        polygon = polygon + [x, y]
                    item = Instance(np.asarray(polygon, dtype=float), float(score), int(label))
                    detections.setdefault(key, []).append(item)
        for key, instances in detections.items():
            unique = []
            for item in sorted(instances, key=lambda instance: instance.score, reverse=True):
                if all(box_overlap(item.box, other.box) < 0.65 for other in unique):
                    unique.append(item)
            detections[key] = unique[:5]
        return detections


def compatible_rays(first, second, size):
    origin, ray = first.view.ray(first.center)
    other, direction = second.view.ray(second.center)
    cosine = ray @ direction
    separation = 1 - cosine ** 2
    if separation < 0.01:
        return False
    offset = other - origin
    a, b = offset @ ray, offset @ direction
    depth = (a - cosine * b) / separation
    other_depth = (cosine * a - b) / separation
    if min(depth, other_depth) <= 0:
        return False
    gap = np.linalg.norm(origin + ray * depth - other - direction * other_depth)
    return gap <= 0.30 * size


class SingleCameraGate:
    """Захват по одной камере — только если уверенная детекция держится несколько кадров подряд
    в одной камере и движется не быстрее машины (одиночные ложные срабатывания не проходят)."""

    def __init__(self, frames, speed, fps):
        self.frames = frames
        self.limit = speed / fps
        self.key = None
        self.history = []

    def update(self, frame, key, center):
        last = self.history[-1] if self.history else None
        if (self.key != key or last is None or frame - last[0] > 2
                or np.linalg.norm(center - last[1]) > self.limit * (frame - last[0]) + 1.5):
            self.key, self.history = key, []
        self.history = [*self.history, (frame, np.asarray(center, float))][-self.frames:]
        return len(self.history) >= self.frames

    def reset(self):
        self.key, self.history = None, []


class ConsensusTracker:
    def __init__(self, segmenter, model, fps, camera_count, video_masks=None,
                 acquire_clipped=False, single_camera=False):
        # acquire_clipped: захват по паре камер, где одна маска обрезана краем кадра
        #   (Way1, кадры 0-99: фуру уверенно видят камеры 1 и 2, но в камере 2 её режет нижний край);
        # single_camera: захват по одной камере в многокамерной сцене
        #   (Way1, кадры 232-312: фуру видит только камера 2, остальные закрыты горой и деревьями).
        self.acquire_clipped = acquire_clipped
        self.single_camera = single_camera
        self.single_gate = SingleCameraGate(max(3, round(fps / 6)), model.speed, fps)
        self.last_pose = None
        self.last_pose_frame = -1
        self.segmenter = segmenter
        self.model = model
        self.fps = fps
        self.camera_count = camera_count
        self.current_views = {}
        self.pose = None
        self.velocity = np.zeros(3)
        self.frame = -1
        self.last_seen = -1
        self.last_multi = -1
        self.masks = {}
        self.video_masks = video_masks
        self.image_tracks = {}
        self.mask_horizon = max(1, round(2.5 * fps))
        self.mask_pose = None
        self.mask_velocity = np.zeros(3)
        self.mask_frame = -1
        self.previous_output = None
        self.reverse_streak = 0
        self.search_period = max(1, round(fps / 3))

    def observations(self, cameras, images, prior):
        detections = self.segmenter({key: images[key] for key in cameras})
        if self.frame - self.last_multi > 8 and detections:
            steep = {key: images[key] for key, camera in cameras.items()
                     if camera.forward[2] < -0.65 and key not in detections}
            detections.update(self.segmenter.overhead(steep))
        if len(detections) < 2 and self.frame - self.last_seen >= self.search_period and self.frame % self.search_period == 0:
            detections.update(self.segmenter.tiles({key: images[key] for key in cameras if key not in detections}))
        if prior is not None and self.frame % 3 == 0:
            unmatched = {}
            matched = 0
            for key, view in self.current_views.items():
                expected = view.box(prior)
                if not intersects(expected, view.camera.width, view.camera.height) or min(expected[2:] - expected[:2]) < 8:
                    continue
                if any(box_overlap(item.box, expected) >= 0.35 for item in detections.get(key, ())):
                    matched += 1
                else:
                    unmatched[key] = expected
            if matched < 2 and unmatched:
                keys = sorted(unmatched, key=lambda key: np.prod(unmatched[key][2:] - unmatched[key][:2]), reverse=True)
                for key, instances in self.segmenter.regions(images, {key: unmatched[key] for key in keys[:4]}).items():
                    recovered = [item for item in instances if box_overlap(item.box, unmatched[key]) >= 0.35]
                    detections[key] = [*detections.get(key, ()), *recovered]
        observations = {}
        for key, view in self.current_views.items():
            instances = detections.get(key, [])
            previous = self.masks.get(key)
            if previous is not None and prior is not None:
                gray, mask, age = previous
                expected = view.box(prior)
                if expected is not None and age < 10 and not any(
                        box_overlap(instance.box, expected) > 0.35 for instance in instances):
                    carried = transport(gray, cv2.cvtColor(images[key], cv2.COLOR_BGR2GRAY), mask)
                    if carried is not None and box_overlap(carried.box, expected) > 0.35:
                        instances = [*instances, carried]
            if instances:
                observations[key] = [Observation(View.of(view.camera, item, self.model)) for item in instances]
        return observations

    def track_image_masks(self, estimate, cameras, images):
        self.video_masks.discard_missing(self.frame, cameras.keys() & self.image_tracks.keys())
        tracks = {}
        for key in cameras:
            image = images[key]
            detection = estimate.views.get(key)
            if detection is not None and not detection.flow:
                tracks[key] = self.frame, image, detection
                self.video_masks.sessions.pop(key, None)
                continue
            previous = self.image_tracks.get(key)
            if previous is None:
                continue
            last_detection, previous_image, previous_mask = previous
            if self.frame - last_detection > self.mask_horizon or image.shape != previous_image.shape:
                continue
            if key not in self.video_masks.sessions:
                self.video_masks.step(self.frame - 1, key, previous_image, previous_mask)
            mask = self.video_masks.step(self.frame, key, image, None)
            if mask is not None and box_overlap(mask.box, previous_mask.box) >= 0.45:
                tracks[key] = last_detection, image, mask
                estimate.tracked_2d[key] = mask
        self.image_tracks = tracks

    @staticmethod
    def support(pose, groups, threshold):
        accepted = []
        for observations in groups.values():
            best = None
            for observation in observations:
                box = observation.view.box(pose)
                if box is None or box_overlap(box, observation.box) < 0.3:
                    continue
                overlap = observation.overlap(pose)
                if overlap >= threshold and (best is None or overlap > best[1]):
                    best = observation, overlap
            if best is not None:
                accepted.append(best)
        return accepted

    def fit_pose(self, observations, prior):
        if prior is None or not self.model.ground_vehicle:
            return fit(observations, prior, 0.0, 0.0, 0, 0.8)[0]
        yaw, pitch = prior[3:]
        sigma = max(0.6, 0.12 * self.model.size)
        pose = optimize(observations, prior, prior, yaw, pitch, [sigma] * 3,
                        yaw_window=0.8, model=self.model)[0]
        if self.reverse_streak >= 6 and len(observations) >= 2:
            seed = self.reverse(pose)
            candidate = optimize(observations, seed, seed, seed[3], seed[4], [sigma] * 3,
                                 yaw_window=0.8, model=self.model)[0]
            before = np.mean([ob.overlap(pose) for ob in observations])
            after = np.mean([ob.overlap(candidate) for ob in observations])
            if after >= before - 0.04:
                return candidate
        return pose

    def reverse(self, pose):
        return with_center(np.r_[pose[:3], pose[3] + np.pi, -pose[4]], center_of(pose, self.model), self.model)

    def update_course(self, pose, fresh, reset):
        if not self.model.ground_vehicle or reset or not fresh:
            self.reverse_streak = 0
            return
        speed = np.linalg.norm(self.velocity[:2])
        course = np.arctan2(self.velocity[1], self.velocity[0])
        opposite = speed > 5 / self.fps and abs(angle_diff(course, pose[3])) > 2.35
        self.reverse_streak = self.reverse_streak + 1 if opposite else 0

    def acquire(self, groups):
        best = None
        for first, second in combinations(groups.values(), 2):
            for a in first:
                for b in second:
                    if a.view.instance.flow or b.view.instance.flow:
                        continue
                    if (a.clipped and b.clipped) or ((a.clipped or b.clipped) and not self.acquire_clipped):
                        continue
                    if not compatible_rays(a, b, self.model.size):
                        continue
                    pose = self.fit_pose([a, b], None)
                    support = self.support(pose, groups, 0.55)
                    fresh = [(ob, iou) for ob, iou in support if not ob.view.instance.flow]
                    if len(fresh) < 2:
                        continue
                    score = sum(iou * np.sqrt(ob.view.instance.score) for ob, iou in fresh)
                    if best is None or score > best[0]:
                        best = score, pose, fresh
        return (best[1], best[2]) if best else (None, [])

    def single_camera_ok(self, observation, candidate):
        """Захват по одной камере в многокамерной сцене: место правдоподобно, недалеко от последней
        позы с наблюдением и держится несколько кадров подряд."""
        from online.estimation.gap_fill import scene_plausible

        if not scene_plausible(candidate, self.current_views, self.model):
            self.single_gate.reset()
            return False
        center = center_of(candidate, self.model)
        if self.last_pose is not None:
            elapsed = (self.frame - self.last_pose_frame) / self.fps
            reach = self.model.speed * elapsed + max(15.0, 0.6 * self.model.size)
            if np.linalg.norm(center - center_of(self.last_pose, self.model)) > reach:
                self.single_gate.reset()
                return False
        return self.single_gate.update(self.frame, observation.view.camera.index, center)

    def step(self, frame, cameras, images):
        estimate = self.estimate_pose(frame, cameras, images)
        if self.video_masks is not None:
            self.track_image_masks(estimate, cameras, images)
            estimate = self.recover_mask_pose(estimate)
            self.previous_output = estimate.pose
        return estimate

    def recover_mask_pose(self, estimate):
        if estimate.views:
            self.mask_pose = estimate.pose.copy()
            self.mask_velocity = self.velocity.copy()
            self.mask_frame = self.frame
            return estimate
        gap = self.frame - self.mask_frame
        if self.mask_pose is None or gap > max(1, round(self.fps / 2)):
            return estimate
        observations = [Observation(View.of(self.current_views[key].camera, mask, self.model))
                        for key, mask in estimate.tracked_2d.items()]
        prior = self.mask_pose.copy()
        prior[:3] += gap * self.mask_velocity
        best = None
        for a, b in combinations(observations, 2):
            if a.clipped or b.clipped or not compatible_rays(a, b, self.model.size):
                continue
            pair = [a, b]
            poses = []
            if self.model.ground_vehicle:
                sigma = max(0.6, 0.10 * self.model.size)
                seeds = [prior]
                center = linear_seed(pair)
                if center is not None:
                    seeds.append(with_center(prior, center, self.model))
                poses = [optimize(pair, seed, prior, prior[3], prior[4], [sigma] * 3,
                                  yaw_window=0.35, model=self.model)[0] for seed in seeds]
            else:
                poses = [fit(pair, prior, 0.0, 0.0, 0, 0.8)[0]]
            limit = max(1.0, 1.5 * gap * self.model.speed / self.fps)
            for pose in poses:
                overlap = [ob.overlap(pose) for ob in pair]
                score = min(overlap)
                weak_view = score < 0.52
                if weak_view and (score < 0.42 or max(overlap) < 0.62):
                    continue
                if np.linalg.norm(center_of(pose, self.model) - center_of(prior, self.model)) > limit:
                    continue
                if best is None or score > best[0]:
                    best = score, pose, pair
        if best is None:
            return estimate
        score, pose, pair = best
        velocity = (center_of(pose, self.model) - center_of(self.mask_pose, self.model)) / gap
        self.mask_velocity += 0.25 * (velocity - self.mask_velocity)
        speed_limit = self.model.speed / self.fps
        self.mask_velocity *= min(1, speed_limit / max(np.linalg.norm(self.mask_velocity), 1e-9))
        self.mask_pose, self.mask_frame = pose, self.frame
        if estimate.pose is None:
            if self.previous_output is not None and np.linalg.norm(
                    pose[:3] - self.previous_output[:3]) > max(1.5, 2 * speed_limit):
                return estimate
            views = {ob.view.camera.index: ob.view.instance for ob in pair}
            return Estimate(pose, "mask_multi", views, score, estimate.tracked_2d)
        return estimate

    def estimate_pose(self, frame, cameras, images):
        if frame != self.frame + 1:
            raise ValueError("Expected consecutive frame numbers")
        self.frame = frame
        self.current_views = {key: View.of(camera, model=self.model) for key, camera in cameras.items()}
        previous = self.pose
        prior = None if previous is None else previous.copy()
        if prior is not None:
            prior[:3] += self.velocity
        groups = self.observations(cameras, images, prior)
        self.groups = groups  # для заполнения провалов (online/estimation/gap_fill.py)
        matched = self.support(prior, groups, 0.3) if prior is not None else []
        pose, accepted = None, []
        if matched:
            candidate = self.fit_pose([ob for ob, _ in matched], prior)
            checked = self.support(candidate, groups, 0.48)
            if checked and np.linalg.norm(center_of(candidate, self.model) - center_of(prior, self.model)) <= max(
                    1.5, self.model.speed * 2 / self.fps):
                pose, accepted = candidate, checked

        fresh = sum(not ob.view.instance.flow for ob, _ in accepted)
        reset = False
        if fresh < 2 and len(groups) >= 2 and (prior is None or frame - self.last_multi > 8 or not accepted):
            candidate, support = self.acquire(groups)
            if support:
                if (self.model.ground_vehicle and prior is not None and self.reverse_streak < 6
                        and abs(angle_diff(candidate[3], prior[3])) > np.pi / 2):
                    flipped = self.reverse(candidate)
                    alternatives = [(ob, iou) for ob, iou in self.support(flipped, groups, 0.55)
                                    if not ob.view.instance.flow]
                    if len(alternatives) >= len(support) and np.mean([iou for _, iou in alternatives]) >= (
                            np.mean([iou for _, iou in support]) - 0.04):
                        candidate, support = flipped, alternatives
                reset = previous is None or np.linalg.norm(center_of(candidate, self.model) -
                            center_of(prior, self.model)) > max(1.5, self.model.speed * 2 / self.fps)
                pose, accepted = candidate, support
                fresh = len(support)

        if previous is None and pose is None and (self.camera_count == 1 or self.single_camera):
            strong = [ob for group in groups.values() for ob in group
                      if not ob.clipped and ob.view.instance.score >= 0.50 and min(ob.size) >= 18]
            if strong:
                observation = max(strong, key=lambda ob: ob.view.instance.score)
                candidate = self.fit_pose([observation], None)
                overlap = observation.overlap(candidate)
                if overlap >= 0.68 and (self.camera_count == 1 or self.single_camera_ok(observation, candidate)):
                    pose, accepted, fresh = candidate, [(observation, overlap)], 1
            elif self.single_camera:
                self.single_gate.reset()

        if fresh >= 2:
            self.last_multi = frame
        masks = {}
        views = {}
        for ob, _ in accepted:
            key = ob.view.camera.index
            instance = ob.view.instance
            age = self.masks[key][2] + 1 if instance.flow else 0
            masks[key] = cv2.cvtColor(images[key], cv2.COLOR_BGR2GRAY), instance, age
            views[key] = instance
        self.masks = masks
        if pose is not None:
            if fresh:
                self.last_seen = frame
                self.last_pose, self.last_pose_frame = pose.copy(), frame
            if reset or previous is None:
                self.velocity.fill(0)
            else:
                delta = center_of(pose, self.model) - center_of(previous, self.model)
                self.velocity += 0.25 * (delta - self.velocity)
                limit = self.model.speed / self.fps
                self.velocity *= min(1, limit / max(np.linalg.norm(self.velocity), 1e-9))
            self.update_course(pose, fresh, reset)
            self.pose = pose
            status = "multi" if fresh >= 2 else "mono" if fresh else "flow"
            return Estimate(pose, status, views, float(np.mean([iou for _, iou in accepted])))
        if prior is not None and frame - self.last_seen <= round(self.fps):
            self.pose = prior
            self.velocity *= 0.95
            self.reverse_streak = 0
            return Estimate(prior, "forecast", {}, 0.0)
        self.pose = None
        self.velocity.fill(0)
        self.reverse_streak = 0
        return Estimate(None, "lost", {}, 0.0)
