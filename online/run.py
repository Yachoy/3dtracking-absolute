"""Run on a synchronized directory of Unreal exports. Ground truth stays out of inference."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np

from core.camera import Camera
from core.dataset import CameraTrack, Pose, read_transform_csv
from online.geometry import DEFAULT_TARGET, View, intersects
from online.estimation.kinematics import FlowJumpGuard
from online.estimation.mono_depth import MonoDepth
from online import smooth
from online.target import load_target
from online.tracker import Tracker
from online.perception.vision import PromptSegmenter, Segmenter


def sources(root):
    pattern = re.compile(r"object_transform_(CineCameraActor(\d+))\.csv$")
    result = {}
    for path in sorted(root.glob("object_transform_CineCameraActor*.csv")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        actor, number = match.groups()
        videos = sorted(root.glob(f"*{actor}.mp4"))
        if len(videos) != 1:
            raise ValueError(f"{actor}: expected one matching MP4, found {len(videos)}")
        poses, meta = read_transform_csv(path)
        poses = {f: p for f, p in poses.items() if np.isfinite(np.r_[p.loc, p.rot]).all()}
        if not poses or not meta["focal_mm"] or not meta["sensor_mm"]:
            raise ValueError(f"{actor}: incomplete calibration")
        key = int(number)
        result[key] = CameraTrack(key, meta["focal_mm"], meta["sensor_mm"], poses, videos[0])
    if not result:
        raise ValueError(f"No camera input in {root}")
    return result


def camera_manifest(path):
    cameras = {}
    spec = json.loads(path.read_text(encoding="utf-8"))
    scale = float(spec.get("location_meters_per_unit", 0.01))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("location_meters_per_unit must be positive")
    for entry in spec["cameras"]:
        key = entry["id"]
        if type(key) is not int or key < 1 or key in cameras:
            raise ValueError("Each camera needs a unique positive integer id")
        poses, meta = read_transform_csv(path.parent / entry["poses"])
        poses = {f: Pose(f, p.loc * (scale / 0.01), p.rot) for f, p in poses.items()
                 if np.isfinite(np.r_[p.loc, p.rot]).all()}
        intrinsics = entry.get("intrinsics_px")
        focal = entry.get("focal_mm", meta["focal_mm"])
        sensor = entry.get("sensor_mm", meta["sensor_mm"])
        if intrinsics is not None:
            intrinsics = tuple(map(float, intrinsics))
            if len(intrinsics) != 4 or not np.isfinite(intrinsics).all() or min(intrinsics[:2]) <= 0:
                raise ValueError(f"Camera {key}: expected [fx, fy, cx, cy] in pixels")
        elif (focal is None or sensor is None or len(sensor) != 2
              or not np.isfinite([focal, *sensor]).all() or min(focal, *sensor) <= 0):
            raise ValueError(f"Camera {key}: missing camera intrinsics")
        if not poses:
            raise ValueError(f"Camera {key}: no valid poses")
        cameras[key] = CameraTrack(key, focal, sensor, poses, path.parent / entry["video"], intrinsics)
    if not cameras:
        raise ValueError("Specify at least one camera")
    return cameras


def seed_box(value):
    try:
        camera, box = value.split(":", 1)
        camera = int(camera)
        box = np.asarray(box.split(","), dtype=float)
        if camera > 0 and box.shape == (4,) and np.isfinite(box).all() and min(box[2:] - box[:2]) > 0:
            return camera, box
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Expected ID:X1,Y1,X2,Y2 in original image pixels")


def draw(img, key, camera, estimate, model=DEFAULT_TARGET, view=None):
    out = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
    item = estimate.views.get(key)
    if item is not None:
        contour = item.contour
        contour = np.round(contour * [480/img.shape[1], 270/img.shape[0]]).astype(np.int32)
        cv2.polylines(out, [contour], True, (240, 199, 126) if item.source == "motion" else (83, 240, 136), 1, cv2.LINE_AA)
    if camera is not None and estimate.pose is not None:
        box = (view if view is not None else View.of(camera, model=model)).box(estimate.pose)
        if intersects(box, img.shape[1], img.shape[0]):
            scaled = box * np.tile([480/img.shape[1], 270/img.shape[0]], 2)
            box = np.clip(np.round(scaled).astype(int), [0, 0, 0, 0], [479, 269, 479, 269])
            color = (83, 240, 136) if key in estimate.views else (47, 182, 251)
            cv2.rectangle(out, tuple(box[:2]), tuple(box[2:]), color, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (480, 22), (31, 23, 19), -1)
    cv2.putText(out, f"CAM {key:02d}  |  " + ("OBSERVED" if key in estimate.views else "NO OBSERVATION"),
                (9, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 240, 247), 1, cv2.LINE_AA)
    return out


def grid(frames, cameras, estimate, model=DEFAULT_TARGET, views=None):
    views = views or {}
    tiles = [draw(img, key, cameras.get(key), estimate, model, views.get(key)) for key, img in frames.items()]
    cols = min(3, len(tiles))
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], (31, 23, 19)))
    return np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles), cols)])


def run(args):
    cv2.setNumThreads(2)
    protected = args.robust or args.adaptive
    tracks = camera_manifest(args.cameras) if args.cameras is not None else sources(args.data)
    model = load_target(args.target) if args.target is not None else DEFAULT_TARGET
    seed_boxes = dict(args.seed_box)
    if seed_boxes.keys() - tracks.keys() or len(seed_boxes) != len(args.seed_box):
        raise ValueError("Seed boxes must refer to distinct cameras in the input")
    if (args.pico_source is None) != (args.pico_checkpoint is None):
        raise ValueError("PicoPose needs both --pico-source and --pico-checkpoint")
    if args.pico_subpixel and args.pico_source is None:
        raise ValueError("--pico-subpixel needs PicoPose")
    if (args.gotrack_source is None) != (args.gotrack_checkpoint is None):
        raise ValueError("GoTrack needs both source and checkpoint")
    if (args.tapnext_source is None) != (args.tapnext_checkpoint is None):
        raise ValueError("TAPNext needs both source and checkpoint")
    modern = any((args.gotrack_source, args.roma_source, args.tapnext_source, args.unified))
    if args.appearance_target is not None and args.pico_source is None and not args.surface_pose and not modern:
        raise ValueError("--appearance-target needs a CAD refinement backend")
    if args.joint_pose and args.pico_source is None:
        raise ValueError("--joint-pose needs PicoPose")
    if args.recover_second_view and args.pico_source is None:
        raise ValueError("--recover-second-view needs PicoPose")
    if (args.joint_motion_check or args.joint_spatial_check) and not args.joint_pose:
        raise ValueError("--joint-motion-check and --joint-spatial-check need --joint-pose")
    appearance = load_target(args.appearance_target) if args.appearance_target is not None else model
    if modern and appearance.triangles is None:
        raise ValueError("Modern pose estimation requires a triangular CAD target")
    if args.surface_pose:
        if appearance.triangles is None:
            raise ValueError("--surface-pose needs a triangular --target or --appearance-target")
    args.out.mkdir(parents=True, exist_ok=True)
    caps = {key: cv2.VideoCapture(str(track.video)) for key, track in tracks.items()}
    ready = {key: cap for key, cap in caps.items() if cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) > 0}
    if not ready or (not protected and len(ready) != len(caps)):
        raise ValueError("Could not open every camera stream")
    fps = float(next(iter(ready.values())).get(cv2.CAP_PROP_FPS))
    if protected:
        rates = Counter(round(cap.get(cv2.CAP_PROP_FPS), 2) for cap in ready.values())
        fps = rates.most_common(1)[0][0]
    aligned = {key: cap for key, cap in ready.items() if abs(cap.get(cv2.CAP_PROP_FPS) - fps) <= 0.01}
    if not protected and len(aligned) != len(caps):
        raise ValueError("Camera frame rates differ")
    counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in aligned.values()]
    if args.pico_source is not None or modern or args.cab_ray is not None:
        # Windows: cuDNN из PyTorch загружается раньше пакета nvidia-cudnn для ONNX Runtime, иначе DLL конфликтуют
        import torch  # noqa: F401
    segmenter = Segmenter(args.model, args.threads, labels=model.labels,
                          exclude=(0,) if model.ground_vehicle else ())
    prompt = PromptSegmenter(args.sam, args.threads) if args.sam is not None else None
    print("YOLO providers:", segmenter.model.get_providers(), flush=True)
    if prompt is not None:
        print("SAM providers:", prompt.model.enc.get_providers(), flush=True)
    video_masks = None
    if args.sam_video is not None:
        from online.perception.video_masks import VideoMasks

        video_masks = VideoMasks(args.sam_video, revision=args.sam_video_revision)
    tracker = Tracker(segmenter, fps, args.cadence, prompt, model, seed_boxes,
                      robust=args.robust, adaptive=args.adaptive, video_masks=video_masks)
    tracker.camera_ids.update(tracks)
    mono_depth = MonoDepth(model, fps) if args.mono_depth else None
    flow_jump_guard = FlowJumpGuard(model, fps) if args.guard_flow_jumps else None
    cab_ray = None
    if args.cab_ray is not None:
        from online.estimation.cab_ray import CabRay

        cab_ray = CabRay(args.cab_ray, model, until_multi=not args.cab_ray_always)
    surface_pose = None
    if args.surface_pose:
        from online.estimation.surface_pose import SurfacePose

        surface_pose = SurfacePose(appearance, fps)
    previous_single = None
    previous_recovery_baseline = None
    refiner = None
    if args.pico_source is not None:
        from online.perception.appearance import PicoPose

        refiner = PicoPose(args.pico_source, args.pico_checkpoint, appearance, subpixel=args.pico_subpixel)
    pipeline = None
    if modern:
        from online.pipeline import MeasurementPipeline

        pipeline = MeasurementPipeline.from_options(args, appearance, fps)
    records, processing, missing = [], [], {}
    smoother = smooth.FixedLag(args.lag, args.history, args.method) if args.lag is not None else None
    delayed = []
    writer = None
    with ExitStack() as stack:
        for cap in caps.values():
            stack.callback(cap.release)
        stream = stack.enter_context(open(args.out / "trajectory.csv", "w", newline=""))
        table = csv.writer(stream, delimiter=";")
        table.writerow(["Frame", "Loc_X", "Loc_Y", "Loc_Z"])
        if smoother is not None:
            lag_stream = stack.enter_context((args.out / "trajectory-lag.csv").open("w", newline=""))
            lag_table = csv.writer(lag_stream, delimiter=";")
            lag_table.writerow(smooth.CSV_HEADER)
        for frame in range(args.limit or max(counts)):
            images = {}
            for key, cap in aligned.items():
                ok, image = cap.read()
                if ok:
                    images[key] = image
            if not images and not protected:
                break
            cameras = {}
            for key, image in images.items():
                track = tracks[key]
                if frame in track.poses:
                    cameras[key] = Camera.from_track(track, frame, image.shape[1], image.shape[0])
                else:
                    missing[key] = missing.get(key, 0) + 1
            start = time.perf_counter()
            estimate = tracker.step(frame, cameras, images)
            if (any(refinement is not None for refinement in (refiner, mono_depth, surface_pose, flow_jump_guard, pipeline))
                    and estimate.pose is not None):
                views = [view if tracker.frame_guard else View.of(view.camera, tracker.memory[key].mask, model)
                         for key, view in tracker.current_views.items()
                         if tracker.memory[key].frame == frame and tracker.memory[key].mask is not None]
            else:
                views = []
            if refiner is not None:
                joint = args.joint_pose and estimate.status in {"mono", "multi"}
                estimate.pose = refiner.refine(estimate.pose, images, views, estimate.views, joint=joint,
                                               spatial_check=args.joint_spatial_check)
                if args.joint_motion_check:
                    from online.estimation.joint_pose import motion_supports

                    if refiner.joint_applied and not motion_supports(
                            frame, previous_single, refiner.single_pose, estimate.pose, views, tracker.memory):
                        estimate.pose = refiner.single_pose
                        refiner.joint_applied = False
                    previous_single = refiner.single_pose
            mono_changed = False
            if mono_depth is not None:
                accepted = [view for view in views if view.camera.index in estimate.views]
                metric = refiner is not None and (refiner.joint_applied or refiner.single_applied)
                candidate = mono_depth.step(frame, estimate.pose, estimate.status, estimate.confidence,
                                            accepted, tracker.memory, metric_measurement=metric)
                mono_changed = candidate is not estimate.pose
                estimate.pose = candidate
            if surface_pose is not None:
                estimate.pose, surface_cameras = surface_pose.step(
                    frame, estimate.pose, estimate.status, estimate.confidence, estimate.views, views, tracker.memory)
            if flow_jump_guard is not None:
                checked = flow_jump_guard.step(estimate.pose, estimate.status, estimate.confidence, estimate.views, views)
                flow_jump_changed = checked is not estimate.pose
                estimate.pose = checked
            if cab_ray is not None and estimate.pose is not None:
                corrected = cab_ray.step(estimate.pose, estimate.status, estimate.views, cameras, images)
                # поправляем только ответ: если сдвигать и состояние трекера, выходной фильтр потом
                # долго сходится к многокамерной оценке (Way1, кадры 0–134: 1.61 м против 1.77–2.02 м)
                estimate.pose = corrected
            if args.recover_second_view:
                previous = previous_recovery_baseline
                previous_recovery_baseline = estimate.pose
                recovered_view = None
                if estimate.pose is not None and len(estimate.views) == 1:
                    primary = next(iter(estimate.views))
                    estimate.pose, recovered_view = refiner.recover_second_view(
                        frame, previous, estimate.pose, images, views, primary, tracker.memory)
                if recovered_view is not None:
                    key = recovered_view.camera.index
                    estimate.views = {**estimate.views, key: recovered_view.instance}
                    estimate.status = "multi"
            if pipeline is not None:
                estimate.pose, measurement = pipeline.step(frame, estimate.pose, images, views, tracker.memory)
                if measurement is not None:
                    confirmed = {view.camera.index: view.instance for view in views
                                 if view.camera.index in measurement.cameras}
                    estimate.views = {**estimate.views, **confirmed}
                    estimate.status = "multi" if len(estimate.views) > 1 else "mono"
                if args.unified and estimate.pose is not None and measurement is not None:
                    tracker.accept_measurement(estimate.pose, pipeline.state.velocity)
            processing.append(time.perf_counter() - start)
            xyz = [round(float(x) * 100, 4) for x in estimate.pose[:3]] if estimate.pose is not None else [None] * 3
            table.writerow([frame, *xyz])
            stream.flush()
            if smoother is not None:
                point = smoother.push(frame, xyz)
                if point is not None:
                    delayed.append(point)
                    lag_table.writerow(smooth.csv_row(point))
                    lag_stream.flush()
            record = {"frame": frame, "xyz": xyz, "status": estimate.status,
                      "cams": list(estimate.views), "confidence": round(estimate.confidence, 3)}
            if args.joint_pose:
                record["joint_applied"] = refiner.joint_applied
            if mono_depth is not None:
                record["mono_depth_applied"] = mono_changed
            if cab_ray is not None and cab_ray.applied is not None:
                record["cab_ray"] = {"cams": cab_ray.applied[0], "shift_m": round(cab_ray.applied[1], 2)}
            if surface_pose is not None:
                record["surface_cams"] = list(surface_cameras)
            if flow_jump_guard is not None:
                record["flow_jump_guard"] = flow_jump_changed
            if args.recover_second_view and recovered_view is not None:
                record["recovered_camera"] = recovered_view.camera.index
            if pipeline is not None:
                record["measurement_cameras"] = list(measurement.cameras) if measurement else []
                record["measurement_source"] = measurement.source if measurement else None
                record["reprojection_px"] = measurement.reprojection_px if measurement else None
                if pipeline.state is not None:
                    covariance = pipeline.state.reference_covariance()
                    record["position_std_m"] = (np.sqrt(np.maximum(np.diag(covariance), 0)).tolist()
                                                 if covariance is not None else None)
            if protected:
                record["available_cams"] = sorted(tracker.current_views)
                record["guard_active"] = tracker.frame_guard
                record["camera_health"] = {key: tracker.camera_health.get(key, "missing")
                                           for key in tracks if key not in tracker.current_views
                                           or key in tracker.camera_health}
            if not model.ground_vehicle:
                record["yaw_pitch_roll_deg"] = (np.rad2deg(estimate.pose[3:]).round(5).tolist()
                                                 if estimate.pose is not None else None)
            records.append(record)
            if args.video:
                fixed = {}
                for key, cap in caps.items():
                    img = images.get(key)
                    if img is None:
                        img = np.zeros((270, 480, 3), dtype=np.uint8)
                    fixed[key] = img
                mosaic = grid(fixed, cameras, estimate, model, tracker.current_views)
                if writer is None:
                    writer = cv2.VideoWriter(str(args.out / "control-raw.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, mosaic.shape[1::-1])
                    if not writer.isOpened():
                        raise RuntimeError("Could not open output MP4")
                    stack.callback(writer.release)
                writer.write(mosaic)
            if frame % 50 == 0:
                print(f"{frame:5d} {estimate.status:8s} {list(estimate.views)} {xyz} "
                      f"{1000*np.mean(processing[-50:]):.0f} ms/frame", flush=True)
    if not records:
        raise RuntimeError("No video frames were decoded")
    metadata = {"fps": fps, "frames": len(records), "cameras": len(caps),
                "mean_ms": round(float(np.mean(processing) * 1000), 2),
                "p95_ms": round(float(np.percentile(processing, 95) * 1000), 2),
                "skipped_calibration": missing, "object": model.name,
                "sequence": args.cameras.stem if args.cameras is not None else
                ("Way1" if model.ground_vehicle else args.data.name)}
    if video_masks is not None:
        metadata["sam_video_revision"] = video_masks.revision
    (args.out / "replay.json").write_text(json.dumps({"metadata": metadata, "frames": records}),
                                         encoding="utf-8")
    if smoother is not None:
        smooth.write_replay(args.out, smoother, records, delayed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data", type=Path, help="Directory of Unreal exports")
    inputs.add_argument("--cameras", type=Path, help="JSON camera and video manifest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/yolo26n-seg.onnx"))
    parser.add_argument("--sam", type=Path, help="Optional SAM 2 ONNX filename prefix")
    parser.add_argument("--target", type=Path, help="Metric CAD target JSON (default: Way1 CargoTruck)")
    parser.add_argument("--seed-box", type=seed_box, action="append", default=[],
                        help="Optional target box at startup, ID:X1,Y1,X2,Y2 (repeatable)")
    parser.add_argument("--pico-source", type=Path, help="Checkout of the official PicoPose source")
    parser.add_argument("--pico-checkpoint", type=Path, help="Published PicoPose checkpoint")
    parser.add_argument("--pico-subpixel", action="store_true", help="Keep continuous PicoPose correspondences for geometric estimation")
    parser.add_argument("--appearance-target", type=Path, help="Optional full mesh for a silhouette profile")
    parser.add_argument("--joint-pose", action="store_true", help="Refine one world pose using PicoPose matches from several cameras")
    parser.add_argument("--joint-motion-check", action="store_true", help="Require two camera flows to support joint corrections")
    parser.add_argument("--joint-spatial-check", action="store_true", help="Validate joint corrections on separate image regions")
    parser.add_argument("--recover-second-view", action="store_true", help="Restore a second view with independent motion and reprojection checks")
    parser.add_argument("--mono-depth", action="store_true", help="Carry verified depth through short single-camera intervals")
    parser.add_argument("--surface-pose", action="store_true", help="Refine pose from causal tracks attached to the CAD surface")
    parser.add_argument("--guard-flow-jumps", action="store_true", help="Reject implausible jumps without fresh detections")
    parser.add_argument("--cab-ray", type=Path, help="YOLO weights (e.g. yolo26l.pt): ray to the cab from cameras without a mask")
    parser.add_argument("--cab-ray-always", action="store_true",
                        help="keep the cab ray after the first multi-camera fix (default: only before it)")
    parser.add_argument("--renderer", choices=("numpy", "pytorch3d"), default="numpy", help="Visible CAD rasterizer for GoTrack")
    parser.add_argument("--gotrack-source", type=Path, help="Official GoTrack source checkout")
    parser.add_argument("--gotrack-checkpoint", type=Path, help="Official GoTrack checkpoint")
    parser.add_argument("--roma-source", type=Path, help="Official RoMa v2 source checkout")
    parser.add_argument("--tapnext-source", type=Path, help="Official TAPNet source checkout")
    parser.add_argument("--tapnext-checkpoint", type=Path, help="TAPNext NPZ checkpoint")
    parser.add_argument("--sam-video", help="Hugging Face SAM2 Video checkpoint or local directory")
    parser.add_argument("--sam-video-revision", help="Pinned revision for a custom Hugging Face SAM2 checkpoint")
    parser.add_argument("--unverified-measurements", action="store_true", help="Research mode: allow geometric candidates without independent temporal verification")
    parser.add_argument("--sequential-models", action="store_true", help="Park inactive neural backends on CPU to bound GPU memory")
    parser.add_argument("--unified", action="store_true", help="Use a generic causal SE(3) state with covariance")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--cadence", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--limit", type=int)
    guards = parser.add_mutually_exclusive_group()
    guards.add_argument("--robust", action="store_true", help="Apply strict tracking checks to every frame")
    guards.add_argument("--adaptive", action="store_true", help="Apply strict tracking checks after a detected camera fault")
    parser.add_argument("--lag", type=smooth.lag_frames, help="Also emit positions with this fixed frame lag")
    parser.add_argument("--history", type=smooth.lag_frames, help="Number of past frames for smoothing (default: same as lag)")
    parser.add_argument("--method", choices=smooth.METHODS, default="linear", help="Trajectory smoothing method")
    parser.add_argument("--video", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
