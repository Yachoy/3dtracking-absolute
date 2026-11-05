"""Run on a synchronized directory of Unreal exports. Ground truth stays out of inference."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

from core.camera import Camera
from core.dataset import CameraTrack, Pose, read_sensor_size, read_transform_csv
from core.video_sync import video_timelines
from online.geometry import DEFAULT_TARGET, View, center_of, intersects
from online.estimation.kinematics import FlowJumpGuard
from online.estimation.mono_depth import MonoDepth
from online.estimation.speed import SpeedMeter
from online import smooth
from online.target import load_target
from online.tracker import Tracker
from online.estimation.gap_fill import FILLED
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
        sensor = meta["sensor_mm"] or read_sensor_size(root)
        if not poses or not meta["focal_mm"] or not sensor:
            raise ValueError(f"{actor}: incomplete calibration")
        key = int(number)
        result[key] = CameraTrack(key, meta["focal_mm"], sensor, poses, videos[0],
                                  focal_by_frame=meta["focal_by_frame"])
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
        else:
            sensor = sensor or read_sensor_size(path.parent)
            if (focal is None or sensor is None or len(sensor) != 2
                    or not np.isfinite([focal, *sensor]).all() or min(focal, *sensor) <= 0):
                raise ValueError(f"Camera {key}: missing camera intrinsics")
        if not poses:
            raise ValueError(f"Camera {key}: no valid poses")
        focal_by_frame = {} if "focal_mm" in entry else meta["focal_by_frame"]
        cameras[key] = CameraTrack(key, focal, sensor, poses, path.parent / entry["video"], intrinsics, focal_by_frame)
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
    tracked = estimate.tracked_2d.get(key)
    item = tracked or estimate.views.get(key)
    if item is not None:
        contour = item.contour
        contour = np.round(contour * [480/img.shape[1], 270/img.shape[0]]).astype(np.int32)
        color = (240, 155, 214) if tracked else ((240, 199, 126) if item.source == "motion" else (83, 240, 136))
        cv2.polylines(out, [contour], True, color, 1, cv2.LINE_AA)
    if camera is not None and estimate.pose is not None:
        box = (view if view is not None else View.of(camera, model=model)).box(estimate.pose)
        if intersects(box, img.shape[1], img.shape[0]):
            scaled = box * np.tile([480/img.shape[1], 270/img.shape[0]], 2)
            box = np.clip(np.round(scaled).astype(int), [0, 0, 0, 0], [479, 269, 479, 269])
            color = (83, 240, 136) if key in estimate.views else (47, 182, 251)
            cv2.rectangle(out, tuple(box[:2]), tuple(box[2:]), color, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (480, 22), (31, 23, 19), -1)
    status = "TRACKED 2D" if tracked else ("OBSERVED" if key in estimate.views else "NO OBSERVATION")
    cv2.putText(out, f"CAM {key:02d}  |  {status}",
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
    protected = args.robust or args.adaptive or args.global_seg is not None
    tracks = camera_manifest(args.cameras) if args.cameras is not None else sources(args.data)
    use_pts = args.sync == "pts" or (args.sync == "auto" and any(track.focal_by_frame for track in tracks.values()))
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
    if args.cab_ray_always and args.cab_ray is None:
        raise ValueError("--cab-ray-always needs --cab-ray")
    if args.cab_ray is not None and (not model.labels or not args.cab_ray.is_file()):
        raise ValueError("--cab-ray needs local YOLO weights and target detector labels")
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
    timelines = {}
    if use_pts:
        aligned = ready
        counts = {key: int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for key, cap in ready.items()}
        fps, timelines = video_timelines({key: tracks[key].video for key in ready}, counts)
        length = max(max(frames) for frames in timelines.values()) + 1
    else:
        fps = float(next(iter(ready.values())).get(cv2.CAP_PROP_FPS))
        if protected:
            rates = Counter(round(cap.get(cv2.CAP_PROP_FPS), 2) for cap in ready.values())
            fps = rates.most_common(1)[0][0]
        aligned = {key: cap for key, cap in ready.items() if abs(cap.get(cv2.CAP_PROP_FPS) - fps) <= 0.01}
        if not protected and len(aligned) != len(caps):
            raise ValueError("Camera frame rates differ")
        length = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in aligned.values())
    if args.global_seg is None:
        if sys.platform == "win32" and (args.pico_source is not None or modern or args.cab_ray is not None):
            import torch  # Load PyTorch's cuDNN before ONNX Runtime on Windows.
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
    if args.global_seg is not None:
        from online.estimation.consensus import ConsensusTracker, GlobalSegmenter

        tracker = ConsensusTracker(GlobalSegmenter(args.global_seg, model.labels), model, fps, len(tracks),
                                   video_masks=video_masks, acquire_clipped=args.acquire_clipped,
                                   single_camera=args.single_camera)
    else:
        tracker = Tracker(segmenter, fps, args.cadence, prompt, model, seed_boxes,
                          robust=args.robust, adaptive=args.adaptive, video_masks=video_masks)
        tracker.camera_ids.update(tracks)
    gap_fill = None
    if args.global_seg is not None and not args.no_fill_gaps:
        from online.estimation.gap_fill import GapFill

        gap_fill = GapFill(tracker, model, fps)
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
    speed_meter = SpeedMeter(fps, model.speed)
    writer = None
    with ExitStack() as stack:
        for cap in caps.values():
            stack.callback(cap.release)
        stream = stack.enter_context(open(args.out / "trajectory.csv", "w", newline=""))
        table = csv.writer(stream, delimiter=";")
        table.writerow(["Frame", "Loc_X", "Loc_Y", "Loc_Z", "Speed_m_s", "Speed_km_h"])
        if smoother is not None:
            lag_stream = stack.enter_context((args.out / "trajectory-lag.csv").open("w", newline=""))
            lag_table = csv.writer(lag_stream, delimiter=";")
            lag_table.writerow(smooth.CSV_HEADER)
        for frame in range(min(args.limit, length) if args.limit is not None else length):
            images = {}
            for key, cap in aligned.items():
                if use_pts and frame not in timelines[key]:
                    continue
                ok, image = cap.read()
                if ok:
                    images[key] = image
                elif use_pts:
                    raise ValueError(f"Camera {key}: failed to decode frame {frame} at its timestamp")
            if not images and not protected and not use_pts:
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
            if gap_fill is not None:
                estimate = gap_fill.step(frame, estimate, images)  # только выход: состояние трекера не меняется
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
            if cab_ray is not None:
                ray_views = {key: view for key, view in tracker.current_views.items()
                             if tracker.camera_health.get(key) != "outlier"}
                # Refine only the emitted pose, after every stateful update.
                estimate.pose = cab_ray.step(estimate.pose, estimate.status, estimate.views, ray_views, images)
            processing.append(time.perf_counter() - start)
            xyz = [round(float(x) * 100, 4) for x in estimate.pose[:3]] if estimate.pose is not None else [None] * 3
            center = (center_of(estimate.pose, model)
                      if estimate.pose is not None and estimate.status not in {"forecast", "flow", "mask_multi", *FILLED} else None)
            speed = speed_meter.step(frame, center)
            speed_mps = round(speed, 4) if speed is not None else None
            speed_kmh = round(speed * 3.6, 3) if speed is not None else None
            table.writerow([frame, *xyz, speed_mps, speed_kmh])
            stream.flush()
            if smoother is not None:
                point = smoother.push(frame, xyz)
                if point is not None:
                    delayed.append(point)
                    lag_table.writerow(smooth.csv_row(point))
                    lag_stream.flush()
            record = {"frame": frame, "xyz": xyz, "status": estimate.status,
                      "cams": list(estimate.views), "confidence": round(estimate.confidence, 3),
                      "speed_mps": speed_mps, "speed_kmh": speed_kmh}
            if estimate.tracked_2d:
                record["tracked_2d_boxes"] = {key: item.box.round(1).tolist()
                                              for key, item in estimate.tracked_2d.items()}
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
            if args.robust or args.adaptive:
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
                speed_text = "--" if speed_kmh is None else f"{speed_kmh:.1f} km/h"
                label = f"SPEED {speed_text}"
                x = mosaic.shape[1] - cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0] - 12
                cv2.putText(mosaic, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 240, 247), 1, cv2.LINE_AA)
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
    first_video = next(iter(tracks.values())).video.stem
    sequence = args.cameras.stem if args.cameras is not None else (
        first_video.split(".FinalImage.")[0] if ".FinalImage." in first_video else args.data.name)
    metadata = {"fps": fps, "frames": len(records), "cameras": len(caps),
                "mean_ms": round(float(np.mean(processing) * 1000), 2),
                "p95_ms": round(float(np.percentile(processing, 95) * 1000), 2),
                "skipped_calibration": missing, "object": model.name, "sequence": sequence,
                "synchronization": "pts" if use_pts else "index"}
    if use_pts:
        metadata["video_frames_missing"] = {key: sum(frame not in frames for frame in range(len(records)))
                                            for key, frames in timelines.items()}
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
    parser.add_argument("--sync", choices=("auto", "index", "pts"), default="auto",
                        help="Align videos by frame index or presentation timestamp")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/yolo26n-seg.onnx"))
    parser.add_argument("--global-seg", type=Path, help="Use global segmentation and cross-camera CAD consensus")
    parser.add_argument("--acquire-clipped", action="store_true",
                        help="With --global-seg: allow acquisition from a camera pair where one mask touches the frame edge")
    parser.add_argument("--single-camera", action="store_true",
                        help="With --global-seg: allow acquisition from one camera when a confident detection persists")
    parser.add_argument("--no-fill-gaps", action="store_true",
                        help="With --global-seg: leave frames without a tracker pose empty instead of predicting them")
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
    parser.add_argument("--cab-ray", type=Path, help="Local YOLO weights for an extra ray to the target or its reference part")
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
    args = parser.parse_args()
    if (args.acquire_clipped or args.single_camera) and args.global_seg is None:
        parser.error("--acquire-clipped and --single-camera need --global-seg")
    if args.global_seg is not None:
        legacy_options = (args.sam, args.seed_box, args.pico_source, args.pico_checkpoint, args.pico_subpixel,
                          args.appearance_target, args.joint_pose, args.joint_motion_check, args.joint_spatial_check,
                          args.recover_second_view, args.mono_depth, args.surface_pose, args.guard_flow_jumps,
                          args.cab_ray, args.cab_ray_always, args.gotrack_source, args.roma_source, args.tapnext_source,
                          args.gotrack_checkpoint, args.tapnext_checkpoint, args.unified,
                          args.robust, args.adaptive)
        if not args.global_seg.is_file():
            parser.error("--global-seg requires a local segmentation checkpoint")
        if any(legacy_options):
            parser.error("--global-seg replaces the legacy tracker and its refinements")
    run(args)


if __name__ == "__main__":
    main()
