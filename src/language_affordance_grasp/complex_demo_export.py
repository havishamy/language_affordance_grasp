from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from language_affordance_grasp.projection import load_intrinsics
from language_affordance_grasp.titration_targets import (
    SAM3D_TO_OPENCV,
    load_mesh,
    resolve_world_from_opencv_camera,
)


SUPPORTED_OBJECTS = ["wide_mouth_bottle", "beaker", "dropper"]
SUPPORTED_AFFORDANCES = {
    "wide_mouth_bottle": ["stopper", "body_grasp"],
    "beaker": ["mouth"],
    "dropper": ["bulb"],
}
OBJECT_COLORS = {
    "wide_mouth_bottle": np.asarray([255, 150, 40], dtype=np.uint8),
    "beaker": np.asarray([70, 220, 120], dtype=np.uint8),
    "dropper": np.asarray([50, 180, 255], dtype=np.uint8),
}
REGION_COLORS = {
    ("wide_mouth_bottle", "stopper"): np.asarray([255, 255, 255], dtype=np.float32),
    ("wide_mouth_bottle", "body_grasp"): np.asarray([255, 130, 30], dtype=np.float32),
    ("beaker", "mouth"): np.asarray([255, 215, 0], dtype=np.float32),
    ("dropper", "bulb"): np.asarray([255, 0, 255], dtype=np.float32),
}


class ContractError(RuntimeError):
    def __init__(
        self,
        stage: str,
        code: str,
        message: str,
        object_name: str | None = None,
        region: str | None = None,
    ) -> None:
        super().__init__(message)
        self.payload = {
            "stage": stage,
            "object": object_name,
            "region": region,
            "code": code,
            "message": message,
        }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def parse_json_value(value: str, label: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractError("input_validation", "INVALID_JSON_FIELD", f"{label} is not valid JSON: {exc}") from exc


def validate_request_shape(objects: Any, regions: Any) -> None:
    if objects != SUPPORTED_OBJECTS:
        raise ContractError(
            "input_validation",
            "UNSUPPORTED_OBJECT_REQUEST",
            f"requested_objects must be exactly {SUPPORTED_OBJECTS}, got {objects!r}",
        )
    if regions != SUPPORTED_AFFORDANCES:
        raise ContractError(
            "input_validation",
            "UNSUPPORTED_AFFORDANCE_REQUEST",
            f"requested_affordance_regions must be exactly {SUPPORTED_AFFORDANCES}, got {regions!r}",
        )


def _load_mapping(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContractError("input_validation", "INVALID_JSON_FILE", f"Unable to parse {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("input_validation", "INVALID_JSON_FILE", f"{label} must contain a JSON object")
    return payload


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    objects = parse_json_value(args.requested_objects_json, "requested_objects")
    regions = parse_json_value(args.requested_affordance_regions_json, "requested_affordance_regions")
    validate_request_shape(objects, regions)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", args.capture_id):
        raise ContractError(
            "input_validation",
            "INVALID_CAPTURE_ID",
            "capture_id must contain 1 to 128 letters, digits, underscores, dots, or hyphens",
        )
    if not np.isfinite(args.depth_scale) or not (0.0 < args.depth_scale <= 1.0):
        raise ContractError("input_validation", "INVALID_DEPTH_SCALE", "depth_scale must be finite and in (0, 1]")

    try:
        rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
        depth = np.asarray(Image.open(args.depth))
    except Exception as exc:
        raise ContractError("input_validation", "INVALID_IMAGE", f"Unable to decode RGB/depth input: {exc}") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ContractError("input_validation", "INVALID_RGB", f"RGB must be uint8 HxWx3, got {rgb.shape} {rgb.dtype}")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ContractError("input_validation", "INVALID_DEPTH", f"Depth must be uint16 HxW, got {depth.shape} {depth.dtype}")
    if rgb.shape[:2] != depth.shape:
        raise ContractError(
            "input_validation",
            "IMAGE_SIZE_MISMATCH",
            f"RGB shape {rgb.shape[:2]} does not match depth shape {depth.shape}",
        )

    camera = _load_mapping(args.intrinsics, "camera JSON")
    required_camera = ("width", "height", "fx", "fy", "cx", "cy")
    missing = [key for key in required_camera if key not in camera]
    if missing:
        raise ContractError("input_validation", "INVALID_INTRINSICS", f"camera JSON is missing {missing}")
    height, width = depth.shape
    if int(camera["width"]) != width or int(camera["height"]) != height:
        raise ContractError(
            "input_validation",
            "INTRINSICS_SIZE_MISMATCH",
            f"camera size {(camera['height'], camera['width'])} does not match image size {(height, width)}",
        )
    intrinsics_values = np.asarray([camera[key] for key in ("fx", "fy", "cx", "cy")], dtype=np.float64)
    if not np.isfinite(intrinsics_values).all() or np.any(intrinsics_values[:2] <= 0.0):
        raise ContractError("input_validation", "INVALID_INTRINSICS", "Camera intrinsics must be finite with fx/fy > 0")
    camera_scale = camera.get("depth_scale_m_per_unit")
    if camera_scale is not None and not np.isclose(float(camera_scale), args.depth_scale, rtol=1e-6, atol=1e-12):
        raise ContractError(
            "input_validation",
            "DEPTH_SCALE_MISMATCH",
            f"Form depth_scale {args.depth_scale} does not match camera JSON value {camera_scale}",
        )

    world_ee = _load_mapping(args.world_ee, "T_world_ee JSON")
    if "T_world_ee" not in world_ee:
        raise ContractError("input_validation", "INVALID_WORLD_EE", "T_world_ee JSON must contain T_world_ee")
    transform = np.asarray(world_ee["T_world_ee"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ContractError("input_validation", "INVALID_WORLD_EE", "T_world_ee must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ContractError("input_validation", "INVALID_WORLD_EE", "T_world_ee must have homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-2) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=2e-2
    ):
        raise ContractError("input_validation", "INVALID_WORLD_EE", "T_world_ee rotation must be orthonormal")

    result = {
        "capture_id": args.capture_id,
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "depth_dtype": str(depth.dtype),
        "depth_raw_range": [int(depth.min()), int(depth.max())],
        "depth_scale": float(args.depth_scale),
        "requested_objects": objects,
        "requested_affordance_regions": regions,
    }
    if args.output:
        write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _load_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if mask.shape != shape:
        raise ContractError("affordance_validation", "MASK_SIZE_MISMATCH", f"Mask {path} has shape {mask.shape}, expected {shape}")
    return mask


def validate_affordance(args: argparse.Namespace) -> dict[str, Any]:
    heatmap = np.asarray(np.load(args.heatmap), dtype=np.float64)
    if heatmap.ndim == 3:
        heatmap = heatmap[..., 0]
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ContractError(
            "affordance_validation",
            "INVALID_HEATMAP",
            f"Affordance heatmap must be a finite 2D array, got {heatmap.shape}",
            args.object,
            args.region,
        )
    heatmap = np.maximum(heatmap, 0.0)
    target = _load_mask(args.target_mask, heatmap.shape)
    other = np.zeros(heatmap.shape, dtype=bool)
    for path in args.other_mask:
        other |= _load_mask(path, heatmap.shape)
    other &= ~target
    if not np.any(target) or float(heatmap.max()) <= 0.0:
        raise ContractError(
            "affordance_validation",
            "EMPTY_AFFORDANCE",
            "Target mask or affordance heatmap is empty",
            args.object,
            args.region,
        )

    y_peak, x_peak = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    ys, xs = np.where(target)
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    relative_y = (float(y_peak) - y_min) / max(y_max - y_min, 1)
    desired = target.copy()
    if args.region == "stopper":
        desired[int(round(y_min + 0.30 * (y_max - y_min))) :, :] = False
    elif args.region == "mouth":
        desired[int(round(y_min + 0.25 * (y_max - y_min))) :, :] = False
    elif args.region == "bulb":
        desired[int(round(y_min + 0.45 * (y_max - y_min))) :, :] = False
    elif args.region == "body_grasp":
        desired[: int(round(y_min + 0.30 * (y_max - y_min))), :] = False
        desired[int(round(y_min + 0.80 * (y_max - y_min))) :, :] = False
    target_peak = float(heatmap[target].max())
    other_peak = float(heatmap[other].max()) if np.any(other) else 0.0
    valid = bool(target[y_peak, x_peak] and desired[y_peak, x_peak] and target_peak > other_peak)
    metrics = {
        "object": args.object,
        "region": args.region,
        "argmax_uv": [int(x_peak), int(y_peak)],
        "argmax_inside_target": bool(target[y_peak, x_peak]),
        "argmax_inside_expected_vertical_band": bool(desired[y_peak, x_peak]),
        "argmax_relative_y": relative_y,
        "target_peak": target_peak,
        "other_peak": other_peak,
        "valid": valid,
    }
    write_json(Path(args.output), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not valid:
        raise ContractError(
            "affordance_validation",
            "AFFORDANCE_OUTSIDE_REQUESTED_REGION",
            f"Affordance validation failed: {metrics}",
            args.object,
            args.region,
        )
    return metrics


def validate_affordance_separation(args: argparse.Namespace) -> dict[str, Any]:
    first = np.asarray(np.load(args.first_heatmap), dtype=np.float64)
    second = np.asarray(np.load(args.second_heatmap), dtype=np.float64)
    if first.ndim == 3:
        first = first[..., 0]
    if second.ndim == 3:
        second = second[..., 0]
    if first.shape != second.shape or first.ndim != 2:
        raise ContractError(
            "affordance_validation",
            "HEATMAP_SIZE_MISMATCH",
            f"Heatmap shapes differ: {first.shape} vs {second.shape}",
            args.object,
            f"{args.first_region}_vs_{args.second_region}",
        )
    mask = _load_mask(args.target_mask, first.shape)
    first[~np.isfinite(first)] = 0.0
    second[~np.isfinite(second)] = 0.0
    first_values = first[mask]
    second_values = second[mask]
    first_active = mask & (first >= np.quantile(first_values, 1.0 - args.top_fraction))
    second_active = mask & (second >= np.quantile(second_values, 1.0 - args.top_fraction))
    intersection = int(np.logical_and(first_active, second_active).sum())
    union = int(np.logical_or(first_active, second_active).sum())
    top_iou = float(intersection / max(union, 1))
    first_y, first_x = np.unravel_index(int(np.argmax(first)), first.shape)
    second_y, second_x = np.unravel_index(int(np.argmax(second)), second.shape)
    ys = np.where(mask)[0]
    height = max(int(ys.max() - ys.min() + 1), 1)
    vertical_gap = float(abs(second_y - first_y) / height)
    result = {
        "object": args.object,
        "first_region": args.first_region,
        "second_region": args.second_region,
        "first_argmax_uv": [int(first_x), int(first_y)],
        "second_argmax_uv": [int(second_x), int(second_y)],
        "top_fraction": float(args.top_fraction),
        "top_region_iou": top_iou,
        "vertical_peak_gap_fraction": vertical_gap,
        "max_iou": float(args.max_iou),
        "min_vertical_gap": float(args.min_vertical_gap),
        "valid": bool(top_iou <= args.max_iou and vertical_gap >= args.min_vertical_gap),
    }
    write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise ContractError(
            "affordance_validation",
            "AFFORDANCE_REGIONS_NOT_SEPARATED",
            f"Affordance regions overlap too strongly: {result}",
            args.object,
            f"{args.first_region}_vs_{args.second_region}",
        )
    return result


def quaternion_wxyz_to_matrix(value: Any) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("rotation_quaternion_wxyz must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-10:
        raise ValueError("rotation quaternion has zero norm")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def sample_mesh_points(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        points, _ = trimesh.sample.sample_surface(mesh, int(count))
    finally:
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float64)


def downsample_points(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(points), size=maximum, replace=False))
    return points[indices]


def canonical_model_id(mesh: trimesh.Trimesh, rotation: np.ndarray, translation: np.ndarray, scale: float) -> str:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    canonical = ((vertices - translation[None, :]) @ rotation) / scale
    canonical = np.round(canonical, decimals=6).astype("<f4", copy=False)
    faces = np.asarray(mesh.faces, dtype="<i4")
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    digest.update(np.asarray(faces.shape, dtype="<i8").tobytes())
    digest.update(faces.tobytes())
    return f"sam3d-generated-sha256:{digest.hexdigest()}"


def export_point_cloud(path: Path, points: np.ndarray, color: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError(f"Cannot export invalid point cloud {path}: shape={points.shape}")
    colors = np.tile(np.asarray(color, dtype=np.uint8)[None, :], (len(points), 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.points.PointCloud(points.astype(np.float32), colors=colors).export(path)


def export_masked_depth_clouds(
    args: argparse.Namespace,
    output_dir: Path,
    world_from_sam3d: np.ndarray,
) -> dict[str, dict[str, Any]]:
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"), dtype=np.uint8)
    depth_raw = np.asarray(Image.open(args.depth))
    if depth_raw.ndim != 2:
        raise ContractError("depth_cloud_export", "INVALID_DEPTH", f"Expected 2D depth image, got {depth_raw.shape}")
    depth = depth_raw.astype(np.float64) * float(args.depth_scale)
    intrinsics = load_intrinsics(args.intrinsics)
    if rgb.shape[:2] != depth.shape:
        raise ContractError(
            "depth_cloud_export",
            "IMAGE_SIZE_MISMATCH",
            f"RGB shape {rgb.shape[:2]} does not match depth shape {depth.shape}",
        )
    height, width = depth.shape
    ys, xs = np.indices((height, width), dtype=np.float64)
    z = depth
    x = -((xs - float(intrinsics[0, 2])) / float(intrinsics[0, 0])) * z
    y = -((ys - float(intrinsics[1, 2])) / float(intrinsics[1, 1])) * z
    organized = np.stack([x, y, z], axis=-1)
    mask_paths = {
        "wide_mouth_bottle": args.wide_mouth_bottle_mask,
        "beaker": args.beaker_mask,
        "dropper": args.dropper_mask,
    }
    depth_dir = output_dir / "00_segmentation" / "depth_clouds"
    depth_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, Any]] = {}
    for object_name, mask_path in mask_paths.items():
        mask = _load_mask(mask_path, depth.shape)
        valid = mask & np.isfinite(depth) & (depth > 0.0)
        points_camera = organized[valid]
        colors = rgb[valid]
        if len(points_camera) < 50:
            raise ContractError(
                "depth_cloud_export",
                "NO_VALID_DEPTH_POINTS",
                f"Only {len(points_camera)} valid masked depth points",
                object_name,
            )
        points_world = transform_points(points_camera, world_from_sam3d)
        camera_path = depth_dir / f"{object_name}_depth_camera.ply"
        world_path = depth_dir / f"{object_name}_depth_world.ply"
        trimesh.points.PointCloud(points_camera.astype(np.float32), colors=colors).export(camera_path)
        trimesh.points.PointCloud(points_world.astype(np.float32), colors=colors).export(world_path)
        summary[object_name] = {
            "valid_point_count": int(len(points_camera)),
            "depth_camera_artifact": camera_path.relative_to(output_dir).as_posix(),
            "depth_world_artifact": world_path.relative_to(output_dir).as_posix(),
            "bounds_camera": np.stack([points_camera.min(axis=0), points_camera.max(axis=0)]).tolist(),
            "bounds_world": np.stack([points_world.min(axis=0), points_world.max(axis=0)]).tolist(),
        }
    write_json(
        depth_dir / "depth_clouds.json",
        {
            "depth_scale": float(args.depth_scale),
            "invalid_depth_rule": "raw depth 0 is excluded",
            "camera_frame": "SAM3D camera: x left, y up, z forward",
            "world_frame": "base",
            "objects": summary,
        },
    )
    return summary


def load_region(path: Path, object_name: str, region: str) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        pack = np.load(path)
        points = np.asarray(pack["points"], dtype=np.float64)
        scores = np.asarray(pack["scores"], dtype=np.float64).reshape(-1) if "scores" in pack else None
    except Exception as exc:
        raise ContractError(
            "affordance_region_export",
            "INVALID_REGION_FILE",
            f"Unable to read projected affordance points from {path}: {exc}",
            object_name,
            region,
        ) from exc
    finite = np.isfinite(points).all(axis=1) if points.ndim == 2 and points.shape[1] == 3 else np.zeros(0, dtype=bool)
    if scores is not None:
        if len(scores) != len(points):
            raise ContractError(
                "affordance_region_export",
                "INVALID_REGION_FILE",
                f"Point/score length mismatch in {path}",
                object_name,
                region,
            )
        finite &= np.isfinite(scores)
    if points.ndim != 2 or points.shape[1] != 3 or not np.any(finite):
        raise ContractError(
            "affordance_region_export",
            "NO_VALID_WORLD_POINTS",
            f"No finite projected affordance points in {path}",
            object_name,
            region,
        )
    return points[finite], None if scores is None else scores[finite]


def make_overlays(args: argparse.Namespace, output_dir: Path) -> None:
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"), dtype=np.float32)
    segmentation = rgb.copy()
    for object_name, path in (
        ("wide_mouth_bottle", args.wide_mouth_bottle_mask),
        ("beaker", args.beaker_mask),
        ("dropper", args.dropper_mask),
    ):
        mask = _load_mask(path, rgb.shape[:2])
        segmentation[mask] = 0.55 * segmentation[mask] + 0.45 * OBJECT_COLORS[object_name]
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(segmentation, 0, 255).astype(np.uint8)).save(overlay_dir / "segmentation_overlay.png")

    affordance = rgb.copy()
    for object_name, region, path in (
        ("wide_mouth_bottle", "stopper", args.stopper_heatmap),
        ("wide_mouth_bottle", "body_grasp", args.body_grasp_heatmap),
        ("beaker", "mouth", args.beaker_mouth_heatmap),
        ("dropper", "bulb", args.bulb_heatmap),
    ):
        heatmap = np.asarray(np.load(path), dtype=np.float32)
        if heatmap.ndim == 3:
            heatmap = heatmap[..., 0]
        heatmap[~np.isfinite(heatmap)] = 0.0
        maximum = float(heatmap.max())
        normalized = heatmap / maximum if maximum > 0 else np.zeros_like(heatmap)
        alpha = (0.5 * normalized)[..., None]
        color = REGION_COLORS[(object_name, region)]
        affordance = affordance * (1.0 - alpha) + color[None, None, :] * alpha
    Image.fromarray(np.clip(affordance, 0, 255).astype(np.uint8)).save(overlay_dir / "affordance_overlay.png")


def export_results(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = output_dir / "failure.json"
    try:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.job_id):
            raise ContractError(
                "result_export",
                "INVALID_JOB_ID",
                "job_id must contain only letters, digits, underscores, or hyphens",
            )
        if (
            args.model_point_count <= 0
            or not (256 <= args.max_region_points <= 4096)
            or not np.isfinite(args.depth_scale)
            or not (0.0 < args.depth_scale <= 1.0)
        ):
            raise ContractError(
                "result_export",
                "INVALID_EXPORT_LIMIT",
                "model_point_count must be positive, max_region_points must be in [256, 4096], and depth_scale in (0, 1]",
            )
        objects = parse_json_value(args.requested_objects_json, "requested_objects")
        regions = parse_json_value(args.requested_affordance_regions_json, "requested_affordance_regions")
        validate_request_shape(objects, regions)
        world_from_opencv, calibration_info = resolve_world_from_opencv_camera(args.calibration, args.world_ee)
        world_from_sam3d = world_from_opencv @ SAM3D_TO_OPENCV
        if not np.isfinite(world_from_sam3d).all():
            raise ContractError("result_export", "INVALID_CALIBRATION_CHAIN", "World calibration chain is non-finite")
        depth_clouds = export_masked_depth_clouds(args, output_dir, world_from_sam3d)

        object_inputs = {
            "wide_mouth_bottle": (args.wide_mouth_bottle_mesh, args.wide_mouth_bottle_pose, 44),
            "beaker": (args.beaker_mesh, args.beaker_pose, 43),
            "dropper": (args.dropper_mesh, args.dropper_pose, 42),
        }
        region_inputs = {
            ("wide_mouth_bottle", "stopper"): args.stopper_region,
            ("wide_mouth_bottle", "body_grasp"): args.body_grasp_region,
            ("beaker", "mouth"): args.beaker_mouth_region,
            ("dropper", "bulb"): args.bulb_region,
        }
        (output_dir / "models").mkdir(parents=True, exist_ok=True)
        result_objects: dict[str, Any] = {}
        world_scene_meshes: list[trimesh.Trimesh] = []
        for object_name, (mesh_path, pose_path, seed) in object_inputs.items():
            try:
                mesh = load_mesh(mesh_path)
                pose = json.loads(Path(pose_path).read_text(encoding="utf-8"))
                scale = float(pose["uniform_scale"])
                rotation = quaternion_wxyz_to_matrix(pose["rotation_quaternion_wxyz"])
                translation = np.asarray(pose["translation_pytorch3d"], dtype=np.float64)
            except Exception as exc:
                raise ContractError(
                    "model_export", "INVALID_MODEL_POSE", f"Invalid mesh or pose for {object_name}: {exc}", object_name
                ) from exc
            if not np.isfinite(scale) or scale <= 0.0 or translation.shape != (3,) or not np.isfinite(translation).all():
                raise ContractError("model_export", "INVALID_MODEL_POSE", "Scale/translation is invalid", object_name)

            sam3d_from_object = np.eye(4, dtype=np.float64)
            sam3d_from_object[:3, :3] = rotation
            sam3d_from_object[:3, 3] = translation
            world_from_object = world_from_sam3d @ sam3d_from_object
            if not np.isfinite(world_from_object).all():
                raise ContractError("model_export", "INVALID_MODEL_POSE", "T_world_object is non-finite", object_name)

            canonical_vertices = ((np.asarray(mesh.vertices) - translation[None, :]) @ rotation) / scale
            canonical_mesh = mesh.copy()
            canonical_mesh.vertices = canonical_vertices
            canonical_artifact = Path("models") / f"{object_name}_canonical.ply"
            canonical_mesh.export(output_dir / canonical_artifact)

            camera_mesh_artifact = Path("models") / f"{object_name}_camera_mesh.ply"
            mesh.export(output_dir / camera_mesh_artifact)
            world_mesh = mesh.copy()
            world_mesh.apply_transform(world_from_sam3d)
            world_mesh.visual.face_colors = np.tile(
                np.concatenate([OBJECT_COLORS[object_name], np.asarray([255], dtype=np.uint8)])[None, :],
                (len(world_mesh.faces), 1),
            )
            world_mesh_artifact = Path("models") / f"{object_name}_world_mesh.ply"
            world_mesh_glb_artifact = Path("models") / f"{object_name}_world_mesh.glb"
            world_mesh.export(output_dir / world_mesh_artifact)
            world_mesh.export(output_dir / world_mesh_glb_artifact)
            world_scene_meshes.append(world_mesh)

            camera_points = sample_mesh_points(mesh, args.model_point_count, seed)
            world_points = transform_points(camera_points, world_from_sam3d)
            artifact = Path("models") / f"{object_name}_world.ply"
            export_point_cloud(output_dir / artifact, world_points, OBJECT_COLORS[object_name])
            object_payload: dict[str, Any] = {
                "model_id": canonical_model_id(mesh, rotation, translation, scale),
                "T_world_object": world_from_object.tolist(),
                "scale": [scale, scale, scale],
                "model_points_world_artifact": artifact.as_posix(),
                "model_canonical_artifact": canonical_artifact.as_posix(),
                "model_mesh_camera_artifact": camera_mesh_artifact.as_posix(),
                "model_mesh_world_artifact": world_mesh_artifact.as_posix(),
                "model_mesh_world_glb_artifact": world_mesh_glb_artifact.as_posix(),
                "observed_depth_camera_artifact": depth_clouds[object_name]["depth_camera_artifact"],
                "observed_depth_world_artifact": depth_clouds[object_name]["depth_world_artifact"],
                "canonical_definition": (
                    "SAM3D generated local mesh after LOCAL_TO_PYTORCH3D conversion; content-addressed per exact "
                    "generated mesh. The canonical origin and axes are not guaranteed to match across different captures."
                ),
                "affordance_regions": {},
            }
            for requested_region in regions.get(object_name, []):
                camera_region, region_scores = load_region(
                    Path(region_inputs[(object_name, requested_region)]), object_name, requested_region
                )
                world_region = transform_points(camera_region, world_from_sam3d)
                if len(world_region) > args.max_region_points:
                    rng = np.random.default_rng(seed + len(requested_region))
                    selected_indices = np.sort(
                        rng.choice(len(world_region), size=args.max_region_points, replace=False)
                    )
                    world_region = world_region[selected_indices]
                    if region_scores is not None:
                        region_scores = region_scores[selected_indices]
                region_artifact = Path("affordance_regions") / object_name / f"{requested_region}_world.ply"
                region_npz_artifact = Path("affordance_regions") / object_name / f"{requested_region}_world.npz"
                export_point_cloud(output_dir / region_artifact, world_region, REGION_COLORS[(object_name, requested_region)])
                npz_payload: dict[str, np.ndarray] = {"points_world": world_region.astype(np.float32)}
                if region_scores is not None:
                    npz_payload["scores"] = region_scores.astype(np.float32)
                (output_dir / region_npz_artifact).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output_dir / region_npz_artifact, **npz_payload)
                object_payload["affordance_regions"][requested_region] = {
                    "points_world": world_region.tolist(),
                    "points_world_artifact": region_artifact.as_posix(),
                    "points_world_npz_artifact": region_npz_artifact.as_posix(),
                }
            result_objects[object_name] = object_payload

        scene_world = trimesh.util.concatenate(tuple(world_scene_meshes))
        scene_world_ply = Path("models") / "three_object_scene_world.ply"
        scene_world_glb = Path("models") / "three_object_scene_world.glb"
        scene_world.export(output_dir / scene_world_ply)
        scene_world.export(output_dir / scene_world_glb)
        make_overlays(args, output_dir)
        calibration_bytes = Path(args.calibration).read_bytes()
        payload = {
            "job_id": args.job_id,
            "capture_id": args.capture_id,
            "frame": "world",
            "units": "meters",
            "model_transform_convention": "p_world = T_world_object @ [scale * p_object, 1]",
            "requested_objects": objects,
            "objects": result_objects,
            "artifacts": {
                "three_object_scene_world_ply": scene_world_ply.as_posix(),
                "three_object_scene_world_glb": scene_world_glb.as_posix(),
                "segmentation_overlay": "overlays/segmentation_overlay.png",
                "affordance_overlay": "overlays/affordance_overlay.png",
                "pipeline_log": "pipeline.log",
            },
            "calibration": {
                "id": f"sha256:{hashlib.sha256(calibration_bytes).hexdigest()}",
                "ee_frame": "tool0",
                "world_frame": "base",
                "details": calibration_info,
            },
            "coordinate_conventions": {
                "opencv_camera": "x right, y down, z forward",
                "sam3d_camera": "x left, y up, z forward",
                "T_opencv_from_sam3d": SAM3D_TO_OPENCV.tolist(),
                "T_world_from_sam3d": world_from_sam3d.tolist(),
            },
        }
        write_json(output_dir / "result.json", payload)
        if failure_path.exists():
            failure_path.unlink()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    except ContractError as exc:
        write_json(failure_path, exc.payload)
        raise
    except Exception as exc:
        error = ContractError("result_export", "UNEXPECTED_EXPORT_ERROR", str(exc))
        write_json(failure_path, error.payload)
        raise error from exc


def write_failure(args: argparse.Namespace) -> None:
    path = Path(args.output)
    if path.exists():
        return
    write_json(
        path,
        {
            "stage": args.stage,
            "object": args.object or None,
            "region": args.region or None,
            "code": args.code,
            "message": args.message,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation and result export for the three-object complex demo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-inputs")
    validate.add_argument("--rgb", required=True)
    validate.add_argument("--depth", required=True)
    validate.add_argument("--intrinsics", required=True)
    validate.add_argument("--world-ee", required=True)
    validate.add_argument("--depth-scale", type=float, required=True)
    validate.add_argument("--capture-id", required=True)
    validate.add_argument("--requested-objects-json", required=True)
    validate.add_argument("--requested-affordance-regions-json", required=True)
    validate.add_argument("--output")

    affordance = subparsers.add_parser("validate-affordance")
    affordance.add_argument("--heatmap", required=True)
    affordance.add_argument("--target-mask", required=True)
    affordance.add_argument("--other-mask", action="append", default=[])
    affordance.add_argument("--object", choices=SUPPORTED_OBJECTS, required=True)
    affordance.add_argument("--region", choices=["stopper", "body_grasp", "mouth", "bulb"], required=True)
    affordance.add_argument("--output", required=True)

    separation = subparsers.add_parser("validate-separation")
    separation.add_argument("--first-heatmap", required=True)
    separation.add_argument("--second-heatmap", required=True)
    separation.add_argument("--target-mask", required=True)
    separation.add_argument("--object", choices=SUPPORTED_OBJECTS, required=True)
    separation.add_argument("--first-region", required=True)
    separation.add_argument("--second-region", required=True)
    separation.add_argument("--top-fraction", type=float, default=0.15)
    separation.add_argument("--max-iou", type=float, default=0.45)
    separation.add_argument("--min-vertical-gap", type=float, default=0.12)
    separation.add_argument("--output", required=True)

    export = subparsers.add_parser("export-results")
    export.add_argument("--output-dir", required=True)
    export.add_argument("--job-id", required=True)
    export.add_argument("--capture-id", required=True)
    export.add_argument("--requested-objects-json", required=True)
    export.add_argument("--requested-affordance-regions-json", required=True)
    export.add_argument("--calibration", required=True)
    export.add_argument("--world-ee", required=True)
    export.add_argument("--rgb", required=True)
    export.add_argument("--depth", required=True)
    export.add_argument("--intrinsics", required=True)
    export.add_argument("--depth-scale", type=float, required=True)
    export.add_argument("--wide-mouth-bottle-mask", required=True)
    export.add_argument("--beaker-mask", required=True)
    export.add_argument("--dropper-mask", required=True)
    export.add_argument("--wide-mouth-bottle-mesh", required=True)
    export.add_argument("--beaker-mesh", required=True)
    export.add_argument("--dropper-mesh", required=True)
    export.add_argument("--wide-mouth-bottle-pose", required=True)
    export.add_argument("--beaker-pose", required=True)
    export.add_argument("--dropper-pose", required=True)
    export.add_argument("--stopper-region", required=True)
    export.add_argument("--body-grasp-region", required=True)
    export.add_argument("--beaker-mouth-region", required=True)
    export.add_argument("--bulb-region", required=True)
    export.add_argument("--stopper-heatmap", required=True)
    export.add_argument("--body-grasp-heatmap", required=True)
    export.add_argument("--beaker-mouth-heatmap", required=True)
    export.add_argument("--bulb-heatmap", required=True)
    export.add_argument("--model-point-count", type=int, default=50000)
    export.add_argument("--max-region-points", type=int, default=4096)

    failure = subparsers.add_parser("write-failure")
    failure.add_argument("--output", required=True)
    failure.add_argument("--stage", required=True)
    failure.add_argument("--object", default="")
    failure.add_argument("--region", default="")
    failure.add_argument("--code", required=True)
    failure.add_argument("--message", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-inputs":
        validate_inputs(args)
    elif args.command == "validate-affordance":
        validate_affordance(args)
    elif args.command == "validate-separation":
        validate_affordance_separation(args)
    elif args.command == "export-results":
        export_results(args)
    else:
        write_failure(args)


if __name__ == "__main__":
    main()
