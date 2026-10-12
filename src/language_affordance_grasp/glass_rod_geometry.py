from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from language_affordance_grasp.config import write_json
from language_affordance_grasp.projection import load_intrinsics
from language_affordance_grasp.titration_targets import resolve_world_from_opencv_camera


class GeometryError(RuntimeError):
    def __init__(self, stage: str, code: str, message: str) -> None:
        super().__init__(message)
        self.payload = {
            "stage": stage,
            "object": "glass_rod",
            "region": None,
            "code": code,
            "message": message,
        }


def _finite_list(value: np.ndarray) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise GeometryError("geometry_export", "NONFINITE_GEOMETRY", "Geometry contains NaN or Inf")
    return [float(item) for item in array.tolist()]


def _load_detection_confidence(path: str | Path) -> float:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    detections = payload.get("detections", [])
    if not detections:
        raise GeometryError("object_segmentation", "GLASS_ROD_NOT_DETECTED", "No glass-rod detection was exported")
    return float(max(float(item["confidence"]) for item in detections))


def _principal_axis(mask: np.ndarray, endpoint_quantile: float) -> dict[str, Any]:
    yx = np.argwhere(mask)
    if len(yx) < 20:
        raise GeometryError("geometry_estimation", "MASK_TOO_SMALL", f"Glass-rod mask has only {len(yx)} pixels")
    uv = yx[:, ::-1].astype(np.float64)
    center = uv.mean(axis=0)
    centered = uv - center
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major = eigenvectors[:, order[0]]
    major /= np.linalg.norm(major) + 1e-12
    if major[0] < 0.0 or (abs(major[0]) < 1e-12 and major[1] < 0.0):
        major = -major
    projections = centered @ major
    low = float(np.quantile(projections, endpoint_quantile))
    high = float(np.quantile(projections, 1.0 - endpoint_quantile))
    endpoint_a = center + low * major
    endpoint_b = center + high * major
    lambda_major = max(float(eigenvalues[order[0]]), 0.0)
    lambda_minor = max(float(eigenvalues[order[1]]), 1e-12)
    return {
        "center_uv": center,
        "axis_uv": major,
        "endpoint_a_uv": endpoint_a,
        "endpoint_b_uv": endpoint_b,
        "length_px": high - low,
        "width_sigma_px": math.sqrt(lambda_minor),
        "elongation": math.sqrt(lambda_major / lambda_minor),
        "area_px": int(len(uv)),
        "bbox_xyxy": [
            int(uv[:, 0].min()),
            int(uv[:, 1].min()),
            int(uv[:, 0].max()) + 1,
            int(uv[:, 1].max()) + 1,
        ],
    }


def _pixels_to_camera(uv: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    u = uv[:, 0].astype(np.float64)
    v = uv[:, 1].astype(np.float64)
    z = depth_m.astype(np.float64)
    x = (u - float(intrinsics[0, 2])) * z / float(intrinsics[0, 0])
    y = (v - float(intrinsics[1, 2])) * z / float(intrinsics[1, 1])
    return np.stack([x, y, z], axis=1)


def _table_points(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    intrinsics: np.ndarray,
    bbox_xyxy: list[int],
    padding: int,
    max_points: int = 20000,
) -> np.ndarray:
    height, width = mask.shape
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    yy, xx = np.mgrid[y1:y2:2, x1:x2:2]
    raw = depth[yy, xx]
    valid = (raw > 0) & (~mask[yy, xx])
    uv = np.stack([xx[valid], yy[valid]], axis=1)
    z = raw[valid].astype(np.float64) * float(depth_scale)
    finite = np.isfinite(z) & (z > 0.05) & (z < 5.0)
    uv = uv[finite]
    z = z[finite]
    if len(z) > max_points:
        indices = np.linspace(0, len(z) - 1, max_points).round().astype(np.int64)
        uv = uv[indices]
        z = z[indices]
    return _pixels_to_camera(uv, z, intrinsics)


def _fit_plane_ransac(points: np.ndarray, threshold_m: float, iterations: int = 300) -> tuple[np.ndarray, float, np.ndarray, float]:
    if len(points) < 100:
        raise GeometryError("table_plane", "INSUFFICIENT_TABLE_DEPTH", f"Only {len(points)} table-depth points available")
    rng = np.random.default_rng(20261012)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        inliers = np.abs(points @ normal + offset) <= threshold_m
        if int(inliers.sum()) > int(best.sum()):
            best = inliers
    if int(best.sum()) < max(80, int(0.25 * len(points))):
        raise GeometryError(
            "table_plane",
            "TABLE_PLANE_FIT_FAILED",
            f"Best table plane has only {int(best.sum())}/{len(points)} inliers",
        )
    inlier_points = points[best]
    center = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal) + 1e-12
    offset = -float(normal @ center)
    distances = np.abs(inlier_points @ normal + offset)
    rmse = float(np.sqrt(np.mean(distances**2)))
    return normal, offset, best, rmse


def _intersect_pixel_with_plane(uv: np.ndarray, intrinsics: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    ray = np.asarray(
        [
            (float(uv[0]) - float(intrinsics[0, 2])) / float(intrinsics[0, 0]),
            (float(uv[1]) - float(intrinsics[1, 2])) / float(intrinsics[1, 1]),
            1.0,
        ],
        dtype=np.float64,
    )
    denominator = float(normal @ ray)
    if abs(denominator) < 1e-9:
        raise GeometryError("geometry_estimation", "RAY_PARALLEL_TO_TABLE", "Endpoint ray is parallel to table plane")
    distance = -float(offset) / denominator
    if not np.isfinite(distance) or distance <= 0.0:
        raise GeometryError("geometry_estimation", "INVALID_ENDPOINT_DEPTH", f"Endpoint plane depth is {distance}")
    return distance * ray


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([np.asarray(point, dtype=np.float64), [1.0]])
    return (transform @ homogeneous)[:3]


def _draw_overlay(
    rgb_path: str | Path,
    output_path: Path,
    endpoint_a_uv: np.ndarray,
    endpoint_b_uv: np.ndarray,
    grasp_a_uv: np.ndarray,
    grasp_b_uv: np.ndarray,
) -> None:
    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    a = tuple(float(value) for value in endpoint_a_uv)
    b = tuple(float(value) for value in endpoint_b_uv)
    draw.line((*a, *b), fill=(0, 255, 255), width=4)
    for uv, color, label in (
        (endpoint_a_uv, (255, 80, 80), "A"),
        (endpoint_b_uv, (80, 255, 80), "B"),
        (grasp_a_uv, (255, 180, 0), "GA"),
        (grasp_b_uv, (255, 180, 0), "GB"),
    ):
        u, v = float(uv[0]), float(uv[1])
        radius = 6
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline=color, width=3)
        draw.text((u + 8, v - 8), label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def solve_geometry(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", args.capture_id):
        raise GeometryError("input_validation", "INVALID_CAPTURE_ID", "capture_id contains unsupported characters")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.job_id):
        raise GeometryError("input_validation", "INVALID_JOB_ID", "job_id contains unsupported characters")
    for name in ("depth_scale", "rod_radius_m", "grasp_inset_m", "min_length_m", "max_length_m"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise GeometryError("input_validation", "INVALID_PARAMETER", f"{name} must be finite and positive")
    if args.min_length_m >= args.max_length_m:
        raise GeometryError("input_validation", "INVALID_PARAMETER", "min_length_m must be less than max_length_m")

    rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    depth = np.asarray(Image.open(args.depth))
    mask = np.asarray(Image.open(args.mask).convert("L")) > 0
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise GeometryError("input_validation", "INVALID_DEPTH", f"Depth must be uint16 HxW, got {depth.dtype} {depth.shape}")
    if rgb.shape[:2] != depth.shape or mask.shape != depth.shape:
        raise GeometryError("input_validation", "IMAGE_SIZE_MISMATCH", "RGB, depth, and mask sizes must match")

    intrinsics = load_intrinsics(args.intrinsics).astype(np.float64)
    confidence = _load_detection_confidence(args.detections)
    geometry_2d = _principal_axis(mask, args.endpoint_quantile)
    if geometry_2d["area_px"] < args.min_mask_area:
        raise GeometryError(
            "geometry_validation",
            "MASK_TOO_SMALL",
            f"Glass-rod mask has {geometry_2d['area_px']} pixels; expected at least {args.min_mask_area}",
        )
    if geometry_2d["elongation"] < args.min_elongation:
        raise GeometryError(
            "geometry_validation",
            "MASK_NOT_ELONGATED",
            f"Glass-rod mask elongation {geometry_2d['elongation']:.3f} is below {args.min_elongation}",
        )

    table_points = _table_points(
        mask,
        depth,
        args.depth_scale,
        intrinsics,
        geometry_2d["bbox_xyxy"],
        args.table_padding_px,
    )
    plane_normal_camera, plane_offset_camera, plane_inliers, plane_rmse = _fit_plane_ransac(
        table_points, args.table_plane_threshold_m
    )
    endpoint_a_camera = _intersect_pixel_with_plane(
        geometry_2d["endpoint_a_uv"], intrinsics, plane_normal_camera, plane_offset_camera
    )
    endpoint_b_camera = _intersect_pixel_with_plane(
        geometry_2d["endpoint_b_uv"], intrinsics, plane_normal_camera, plane_offset_camera
    )

    world_from_camera, calibration_info = resolve_world_from_opencv_camera(args.calibration, args.world_ee)
    plane_normal_world = world_from_camera[:3, :3] @ plane_normal_camera
    plane_normal_world /= np.linalg.norm(plane_normal_world) + 1e-12
    if float(plane_normal_world @ np.asarray([0.0, 0.0, 1.0])) < 0.0:
        plane_normal_world = -plane_normal_world
    endpoint_a_world = _transform_point(world_from_camera, endpoint_a_camera) + args.rod_radius_m * plane_normal_world
    endpoint_b_world = _transform_point(world_from_camera, endpoint_b_camera) + args.rod_radius_m * plane_normal_world
    axis_world = endpoint_b_world - endpoint_a_world
    length_m = float(np.linalg.norm(axis_world))
    if not (args.min_length_m <= length_m <= args.max_length_m):
        raise GeometryError(
            "geometry_validation",
            "IMPLAUSIBLE_ROD_LENGTH",
            f"Estimated glass-rod length {length_m:.6f} m is outside [{args.min_length_m}, {args.max_length_m}]",
        )
    axis_world /= length_m
    center_world = 0.5 * (endpoint_a_world + endpoint_b_world)
    grasp_inset_m = min(float(args.grasp_inset_m), 0.25 * length_m)
    grasp_near_a_world = endpoint_a_world + grasp_inset_m * axis_world
    grasp_near_b_world = endpoint_b_world - grasp_inset_m * axis_world
    image_fraction = grasp_inset_m / length_m
    grasp_near_a_uv = geometry_2d["endpoint_a_uv"] + image_fraction * (
        geometry_2d["endpoint_b_uv"] - geometry_2d["endpoint_a_uv"]
    )
    grasp_near_b_uv = geometry_2d["endpoint_b_uv"] - image_fraction * (
        geometry_2d["endpoint_b_uv"] - geometry_2d["endpoint_a_uv"]
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_relative = Path("glass_rod") / "geometry_overlay.png"
    _draw_overlay(
        args.rgb,
        output_dir / overlay_relative,
        geometry_2d["endpoint_a_uv"],
        geometry_2d["endpoint_b_uv"],
        grasp_near_a_uv,
        grasp_near_b_uv,
    )
    endpoint_pixel_scale = float(np.mean([endpoint_a_camera[2], endpoint_b_camera[2]]) / intrinsics[0, 0])
    endpoint_uncertainty_m = float(plane_rmse + 2.0 * endpoint_pixel_scale)
    result = {
        "schema_version": "stirring-geometry-v1",
        "job_id": args.job_id,
        "capture_id": args.capture_id,
        "frame": "world",
        "units": "meters",
        "assumptions": {
            "glass_rod_lies_on_table": True,
            "endpoint_a_b_have_no_task_semantics": True,
            "centerline_is_offset_from_table_by_rod_radius_m": float(args.rod_radius_m),
        },
        "world_up": [0.0, 0.0, 1.0],
        "world_down": [0.0, 0.0, -1.0],
        "glass_rod": {
            "class_id": 12,
            "class_name": "glass rod",
            "segmentation_confidence": confidence,
            "mask_artifact": "00_segmentation/glass_rod/mask.png",
            "mask_probability_artifact": "00_segmentation/glass_rod/mask_prob.png",
            "segmentation_overlay_artifact": "00_segmentation/glass_rod/overlay.png",
            "geometry_overlay_artifact": overlay_relative.as_posix(),
            "endpoint_a_pixel_uv": _finite_list(geometry_2d["endpoint_a_uv"]),
            "endpoint_b_pixel_uv": _finite_list(geometry_2d["endpoint_b_uv"]),
            "endpoint_a_world": _finite_list(endpoint_a_world),
            "endpoint_b_world": _finite_list(endpoint_b_world),
            "center_world": _finite_list(center_world),
            "axis_world_unoriented": _finite_list(axis_world),
            "length_m": length_m,
            "grasp_candidates": [
                {
                    "id": "grasp_near_a",
                    "grasp_point_world": _finite_list(grasp_near_a_world),
                    "grasp_point_pixel_uv": _finite_list(grasp_near_a_uv),
                    "opposite_tip_id": "endpoint_b",
                    "opposite_tip_world": _finite_list(endpoint_b_world),
                    "rod_direction_grasp_to_tip_world": _finite_list(axis_world),
                },
                {
                    "id": "grasp_near_b",
                    "grasp_point_world": _finite_list(grasp_near_b_world),
                    "grasp_point_pixel_uv": _finite_list(grasp_near_b_uv),
                    "opposite_tip_id": "endpoint_a",
                    "opposite_tip_world": _finite_list(endpoint_a_world),
                    "rod_direction_grasp_to_tip_world": _finite_list(-axis_world),
                },
            ],
            "geometry_source": "mask_pca_plus_table_plane_intersection",
            "quality": {
                "valid": True,
                "mask_area_px": int(geometry_2d["area_px"]),
                "mask_elongation": float(geometry_2d["elongation"]),
                "mask_length_px": float(geometry_2d["length_px"]),
                "mask_width_sigma_px": float(geometry_2d["width_sigma_px"]),
                "table_depth_point_count": int(len(table_points)),
                "table_plane_inlier_count": int(plane_inliers.sum()),
                "table_plane_inlier_fraction": float(plane_inliers.mean()),
                "table_plane_rmse_m": plane_rmse,
                "endpoint_uncertainty_m": endpoint_uncertainty_m,
            },
        },
        "calibration": {
            "details": calibration_info,
            "T_world_opencv_camera": world_from_camera.tolist(),
        },
    }
    write_json(output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover a table-top glass rod's endpoints and grasp candidates.")
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--world-ee", required=True)
    parser.add_argument("--depth-scale", type=float, required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rod-radius-m", type=float, default=0.0025)
    parser.add_argument("--grasp-inset-m", type=float, default=0.025)
    parser.add_argument("--min-length-m", type=float, default=0.08)
    parser.add_argument("--max-length-m", type=float, default=0.40)
    parser.add_argument("--min-mask-area", type=int, default=100)
    parser.add_argument("--min-elongation", type=float, default=4.0)
    parser.add_argument("--endpoint-quantile", type=float, default=0.01)
    parser.add_argument("--table-padding-px", type=int, default=35)
    parser.add_argument("--table-plane-threshold-m", type=float, default=0.004)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    try:
        solve_geometry(args)
    except GeometryError as exc:
        write_json(output_dir / "failure.json", exc.payload)
        raise
    except Exception as exc:
        error = GeometryError("geometry_estimation", "UNEXPECTED_GEOMETRY_ERROR", f"{type(exc).__name__}: {exc}")
        write_json(output_dir / "failure.json", error.payload)
        raise error from exc


if __name__ == "__main__":
    main()
