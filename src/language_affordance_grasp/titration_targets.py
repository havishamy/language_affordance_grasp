from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from language_affordance_grasp.config import write_json
from language_affordance_grasp.projection import load_intrinsics


SAM3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)

OBJECT_COLORS = {
    "dropper": np.asarray([50, 180, 255, 255], dtype=np.uint8),
    "beaker": np.asarray([70, 220, 120, 255], dtype=np.uint8),
    "table": np.asarray([150, 150, 150, 180], dtype=np.uint8),
}

TARGET_COLORS = {
    "dropper_bulb": np.asarray([255, 0, 255, 255], dtype=np.uint8),
    "beaker_mouth": np.asarray([255, 215, 0, 255], dtype=np.uint8),
}


def _as_transform(value: Any, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{label} must be a 4x4 matrix, got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"{label} must have homogeneous last row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-2):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-2):
        raise ValueError(f"{label} rotation determinant must be +1")
    return transform


def _matrix_from_mapping(mapping: dict[str, Any], keys: tuple[str, ...]) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        if key in mapping:
            return _as_transform(mapping[key], key), key
    return None, None


def load_transform_file(path: str | Path, preferred_keys: tuple[str, ...] = ()) -> tuple[np.ndarray, str]:
    transform_path = Path(path).expanduser().resolve()
    if transform_path.suffix.lower() == ".npy":
        return _as_transform(np.load(transform_path), transform_path.name), "matrix_file"
    if transform_path.suffix.lower() == ".json":
        with transform_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        if isinstance(payload, list):
            return _as_transform(payload, transform_path.name), "matrix_file"
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a matrix or mapping in {transform_path}")
        matrix, key = _matrix_from_mapping(payload, preferred_keys + ("matrix", "transform", "T"))
        if matrix is None or key is None:
            available = ", ".join(sorted(payload))
            raise ValueError(f"No supported transform key found in {transform_path}; available keys: {available}")
        return matrix, key
    return _as_transform(np.loadtxt(transform_path), transform_path.name), "matrix_file"


def resolve_world_from_opencv_camera(
    calibration_path: str | Path,
    world_ee_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    calibration_path = Path(calibration_path).expanduser().resolve()
    direct_keys = ("T_world_cam", "T_base_cam", "T_world_camera", "camera_to_world")
    ee_cam_keys = ("T_ee_cam", "T_gripper_cam", "T_gripper_camera", "camera_to_gripper")
    cam_ee_keys = ("T_cam_ee", "T_cam_gripper", "gripper_to_camera")
    world_ee_keys = ("T_world_ee", "T_base_ee", "T_base_gripper", "gripper_to_world")

    payload: dict[str, Any] | None = None
    if calibration_path.suffix.lower() == ".json":
        with calibration_path.open("r", encoding="utf-8") as file_handle:
            loaded = json.load(file_handle)
        if isinstance(loaded, dict):
            payload = loaded

    if payload is not None:
        direct, direct_key = _matrix_from_mapping(payload, direct_keys)
        if direct is not None:
            return direct, {
                "mode": "direct_world_camera",
                "calibration_path": str(calibration_path),
                "source_key": direct_key,
            }

        ee_cam, ee_cam_key = _matrix_from_mapping(payload, ee_cam_keys)
        if ee_cam is None:
            cam_ee, cam_ee_key = _matrix_from_mapping(payload, cam_ee_keys)
            if cam_ee is not None:
                ee_cam = np.linalg.inv(cam_ee)
                ee_cam_key = f"inverse({cam_ee_key})"
        embedded_world_ee, embedded_world_ee_key = _matrix_from_mapping(payload, world_ee_keys)
        if world_ee_path:
            world_ee, world_ee_key = load_transform_file(world_ee_path, world_ee_keys)
            world_ee_source = str(Path(world_ee_path).expanduser().resolve())
        else:
            world_ee, world_ee_key = embedded_world_ee, embedded_world_ee_key
            world_ee_source = str(calibration_path)
        if ee_cam is not None and world_ee is not None:
            return _as_transform(world_ee @ ee_cam, "T_world_cam"), {
                "mode": "eye_in_hand",
                "calibration_path": str(calibration_path),
                "eye_in_hand_key": ee_cam_key,
                "world_ee_path": world_ee_source,
                "world_ee_key": world_ee_key,
            }
        if ee_cam is not None:
            raise ValueError(
                "Eye-in-hand calibration provides T_ee_cam but no capture-time T_world_ee. "
                "Add T_world_ee/T_base_ee to the calibration JSON or pass --world-ee."
            )

    calibration, calibration_key = load_transform_file(
        calibration_path,
        direct_keys + ee_cam_keys + cam_ee_keys,
    )
    if world_ee_path:
        world_ee, world_ee_key = load_transform_file(world_ee_path, world_ee_keys)
        return _as_transform(world_ee @ calibration, "T_world_cam"), {
            "mode": "eye_in_hand_matrix_files",
            "calibration_path": str(calibration_path),
            "eye_in_hand_key": calibration_key,
            "world_ee_path": str(Path(world_ee_path).expanduser().resolve()),
            "world_ee_key": world_ee_key,
        }

    return calibration, {
        "mode": "direct_world_camera_matrix_file",
        "calibration_path": str(calibration_path),
        "source_key": calibration_key,
    }


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    point_h = np.concatenate([np.asarray(point, dtype=np.float64), [1.0]])
    return (transform @ point_h)[:3]


def project_sam3d_point(point: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(point, dtype=np.float64)
    if z <= 0.0:
        raise ValueError(f"Cannot project target with non-positive depth {z}")
    u = -(x / z) * float(intrinsics[0, 0]) + float(intrinsics[0, 2])
    v = -(y / z) * float(intrinsics[1, 1]) + float(intrinsics[1, 2])
    return np.asarray([u, v], dtype=np.float64)


def export_target_overlay(
    rgb_path: str | Path,
    intrinsics_path: str | Path,
    output_path: Path,
    dropper_center: np.ndarray,
    beaker_center: np.ndarray,
) -> tuple[str, list[float], list[float]]:
    image = Image.open(rgb_path).convert("RGB")
    intrinsics = load_intrinsics(intrinsics_path)
    dropper_uv = project_sam3d_point(dropper_center, intrinsics)
    beaker_uv = project_sam3d_point(beaker_center, intrinsics)
    draw = ImageDraw.Draw(image)
    radius = 8
    for uv, color, label in (
        (dropper_uv, tuple(TARGET_COLORS["dropper_bulb"][:3]), "dropper bulb"),
        (beaker_uv, tuple(TARGET_COLORS["beaker_mouth"][:3]), "beaker mouth"),
    ):
        u, v = float(uv[0]), float(uv[1])
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline=color, width=4)
        draw.line((u - radius - 3, v, u + radius + 3, v), fill=color, width=2)
        draw.line((u, v - radius - 3, u, v + radius + 3), fill=color, width=2)
        draw.text((u + radius + 4, v - radius), label, fill=color)
    image.save(output_path)
    return str(output_path), dropper_uv.tolist(), beaker_uv.tolist()


def extract_target_center(region_npz_path: str | Path, top_quantile: float) -> dict[str, Any]:
    pack = np.load(region_npz_path)
    points = np.asarray(pack["points"], dtype=np.float64)
    scores = np.asarray(pack["scores"], dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(scores):
        raise ValueError(f"Invalid affordance region arrays in {region_npz_path}")
    finite = np.isfinite(points).all(axis=1) & np.isfinite(scores)
    points = points[finite]
    scores = scores[finite]
    if len(points) == 0:
        raise ValueError(f"No finite projected affordance points in {region_npz_path}")

    threshold = float(np.asarray(pack["threshold"]).reshape(-1)[0]) if "threshold" in pack else 0.0
    threshold_mask = scores >= threshold
    if not np.any(threshold_mask):
        raise ValueError(f"No affordance points meet threshold {threshold:.4f} in {region_npz_path}")
    threshold_points = points[threshold_mask]
    threshold_scores = scores[threshold_mask]
    score_cutoff = float(np.quantile(threshold_scores, float(top_quantile)))
    selected = threshold_scores >= score_cutoff
    selected_points = threshold_points[selected]
    selected_scores = threshold_scores[selected]
    weights = np.maximum(selected_scores - score_cutoff, 1e-4)
    center = np.average(selected_points, axis=0, weights=weights)
    centered = selected_points - center[None, :]
    rms_radius = float(np.sqrt(np.average(np.sum(centered * centered, axis=1), weights=weights)))
    bounds = np.stack([selected_points.min(axis=0), selected_points.max(axis=0)], axis=0)
    return {
        "center_sam3d_camera": center.tolist(),
        "threshold": threshold,
        "top_quantile": float(top_quantile),
        "score_cutoff": score_cutoff,
        "projected_count": int(len(points)),
        "above_threshold_count": int(len(threshold_points)),
        "center_support_count": int(len(selected_points)),
        "center_support_rms_radius_m": rms_radius,
        "center_support_bounds_sam3d_camera": bounds.tolist(),
        "center_support_score_mean": float(selected_scores.mean()),
        "center_support_score_max": float(selected_scores.max()),
    }


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh_like = trimesh.load(str(path), process=False)
    if isinstance(mesh_like, trimesh.Scene):
        if not mesh_like.geometry:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(tuple(item.copy() for item in mesh_like.geometry.values()))
    elif isinstance(mesh_like, trimesh.Trimesh):
        mesh = mesh_like.copy()
    else:
        raise TypeError(f"Expected triangle mesh in {path}, got {type(mesh_like)}")
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh is empty in {path}")
    return mesh


def color_mesh(mesh: trimesh.Trimesh, color: np.ndarray) -> trimesh.Trimesh:
    result = mesh.copy()
    result.visual.face_colors = np.tile(color[None, :], (len(result.faces), 1))
    return result


def marker_sphere(center: np.ndarray, radius: float, color: np.ndarray) -> trimesh.Trimesh:
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=float(radius))
    sphere.apply_translation(np.asarray(center, dtype=np.float64))
    sphere.visual.face_colors = np.tile(color[None, :], (len(sphere.faces), 1))
    return sphere


def export_scene(
    path: Path,
    dropper_mesh: trimesh.Trimesh,
    beaker_mesh: trimesh.Trimesh,
    table_mesh: trimesh.Trimesh | None,
    dropper_center: np.ndarray,
    beaker_center: np.ndarray,
    marker_radius: float,
    transform: np.ndarray | None = None,
) -> str:
    parts: list[trimesh.Trimesh] = []
    if table_mesh is not None:
        parts.append(color_mesh(table_mesh, OBJECT_COLORS["table"]))
    parts.extend([
        color_mesh(dropper_mesh, OBJECT_COLORS["dropper"]),
        color_mesh(beaker_mesh, OBJECT_COLORS["beaker"]),
        marker_sphere(dropper_center, marker_radius, TARGET_COLORS["dropper_bulb"]),
        marker_sphere(beaker_center, marker_radius, TARGET_COLORS["beaker_mouth"]),
    ])
    combined = trimesh.util.concatenate(tuple(parts))
    if transform is not None:
        combined.apply_transform(transform)
    combined.export(path)
    return str(path)


def validate_mask(mask_path: str | Path, rgb_path: str | Path, label: str, min_area: int) -> dict[str, Any]:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"{label} mask shape {mask.shape} does not match RGB shape {rgb.shape[:2]}")
    area = int(mask.sum())
    if area < int(min_area):
        raise ValueError(f"{label} mask has only {area} pixels; expected at least {min_area}")
    result = {
        "label": label,
        "mask_path": str(Path(mask_path).resolve()),
        "shape": list(mask.shape),
        "area_px": area,
        "fraction": float(area / mask.size),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def normalize_intrinsics_file(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    with input_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected intrinsics mapping in {input_path}")

    source_key = "root"
    source = payload
    for candidate in ("aligned_depth_intrinsics", "color_intrinsics", "intrinsics"):
        if candidate in payload and isinstance(payload[candidate], dict):
            source_key = candidate
            source = payload[candidate]
            break
    if "K" in payload:
        matrix = np.asarray(payload["K"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"K must be 3x3 in {input_path}")
        fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
        source_key = "K"
    else:
        fx = float(source["fx"])
        fy = float(source.get("fy", fx))
        cx = float(source.get("cx", source.get("ppx")))
        cy = float(source.get("cy", source.get("ppy")))
    canonical = {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "source_path": str(input_path),
        "source_key": source_key,
    }
    if "width" in source:
        canonical["width"] = int(source["width"])
    if "height" in source:
        canonical["height"] = int(source["height"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, canonical)
    print(json.dumps(canonical, ensure_ascii=False, indent=2))
    return canonical


def validate_calibration(calibration_path: str | Path, world_ee_path: str | Path | None) -> dict[str, Any]:
    transform, info = resolve_world_from_opencv_camera(calibration_path, world_ee_path)
    result = {
        "valid": True,
        "calibration": info,
        "T_world_opencv_camera": transform.tolist(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def export_depth_clouds(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"), dtype=np.uint8)
    depth_raw = np.asarray(Image.open(args.depth))
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    depth = depth_raw.astype(np.float32) * float(args.depth_scale)
    intrinsics = load_intrinsics(args.intrinsics)
    height, width = depth.shape
    if rgb.shape[:2] != (height, width):
        raise ValueError(f"RGB shape {rgb.shape[:2]} does not match depth shape {(height, width)}")
    ys, xs = np.indices((height, width), dtype=np.float32)
    z = depth
    x = -((xs - float(intrinsics[0, 2])) / float(intrinsics[0, 0])) * z
    y = -((ys - float(intrinsics[1, 2])) / float(intrinsics[1, 1])) * z
    organized_points = np.stack([x, y, z], axis=-1)

    outputs: dict[str, Any] = {
        "coordinate_convention": "SAM3D camera: x=-(u-cx)/fx*z, y=-(v-cy)/fy*z, z=depth",
        "depth_scale": float(args.depth_scale),
        "objects": {},
    }
    for label, mask_path in (("dropper", args.dropper_mask), ("beaker", args.beaker_mask)):
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape != (height, width):
            raise ValueError(f"{label} mask shape {mask.shape} does not match depth shape {(height, width)}")
        valid = mask & np.isfinite(depth) & (depth > 0.0)
        points = organized_points[valid].astype(np.float32)
        colors = rgb[valid]
        if len(points) < int(args.min_points):
            raise ValueError(f"{label} has only {len(points)} valid masked depth points")
        cloud_path = output_dir / f"{label}_depth_camera.ply"
        trimesh.points.PointCloud(points, colors=colors).export(cloud_path)
        outputs["objects"][label] = {
            "mask_path": str(Path(mask_path).resolve()),
            "valid_depth_point_count": int(len(points)),
            "depth_camera_ply": str(cloud_path),
            "bounds_sam3d_camera": np.stack([points.min(axis=0), points.max(axis=0)]).tolist(),
        }
    write_json(output_dir / "depth_clouds.json", outputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return outputs


def load_point_cloud(path: str | Path) -> trimesh.points.PointCloud:
    cloud_like = trimesh.load(str(path), process=False)
    if not isinstance(cloud_like, trimesh.points.PointCloud):
        raise TypeError(f"Expected point cloud in {path}, got {type(cloud_like)}")
    return cloud_like.copy()


def export_model_surface_clouds(
    output_dir: Path,
    dropper_mesh: trimesh.Trimesh,
    beaker_mesh: trimesh.Trimesh,
    world_from_sam3d: np.ndarray,
    sample_count: int,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    random_state = np.random.get_state()
    try:
        for seed, label, mesh in ((42, "dropper", dropper_mesh), (43, "beaker", beaker_mesh)):
            np.random.seed(seed)
            points, _ = trimesh.sample.sample_surface(mesh, int(sample_count))
            colors = np.tile(OBJECT_COLORS[label][:3][None, :], (len(points), 1))
            camera_cloud = trimesh.points.PointCloud(points.astype(np.float32), colors=colors)
            camera_path = output_dir / f"{label}_model_surface_camera.ply"
            camera_cloud.export(camera_path)
            world_cloud = camera_cloud.copy()
            world_cloud.apply_transform(world_from_sam3d)
            world_path = output_dir / f"{label}_model_surface_world.ply"
            world_cloud.export(world_path)
            paths[f"{label}_model_surface_camera_ply"] = str(camera_path)
            paths[f"{label}_model_surface_world_ply"] = str(world_path)
    finally:
        np.random.set_state(random_state)
    return paths


def export_world_depth_cloud(path: str | Path, output_path: Path, world_from_sam3d: np.ndarray) -> str:
    cloud = load_point_cloud(path)
    cloud.apply_transform(world_from_sam3d)
    cloud.export(output_path)
    return str(output_path)


def finalize_targets(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    world_from_opencv, calibration_info = resolve_world_from_opencv_camera(
        args.calibration,
        args.world_ee,
    )
    world_from_sam3d = world_from_opencv @ SAM3D_TO_OPENCV

    dropper = extract_target_center(args.dropper_region, args.top_quantile)
    beaker = extract_target_center(args.beaker_region, args.top_quantile)
    dropper_camera = np.asarray(dropper["center_sam3d_camera"], dtype=np.float64)
    beaker_camera = np.asarray(beaker["center_sam3d_camera"], dtype=np.float64)
    for result, center_camera in ((dropper, dropper_camera), (beaker, beaker_camera)):
        result["center_opencv_camera"] = transform_point(SAM3D_TO_OPENCV, center_camera).tolist()
        result["center_world"] = transform_point(world_from_sam3d, center_camera).tolist()

    dropper_mesh = load_mesh(args.dropper_mesh)
    beaker_mesh = load_mesh(args.beaker_mesh)
    table_mesh = load_mesh(args.table_mesh) if args.table_mesh else None
    model_cloud_paths = export_model_surface_clouds(
        output_dir,
        dropper_mesh,
        beaker_mesh,
        world_from_sam3d,
        args.model_point_count,
    )
    camera_scene_path = output_dir / "titration_scene_camera.ply"
    world_scene_path = output_dir / "titration_scene_world.ply"
    export_scene(
        camera_scene_path,
        dropper_mesh,
        beaker_mesh,
        table_mesh,
        dropper_camera,
        beaker_camera,
        args.marker_radius,
    )
    export_scene(
        world_scene_path,
        dropper_mesh,
        beaker_mesh,
        table_mesh,
        dropper_camera,
        beaker_camera,
        args.marker_radius,
        transform=world_from_sam3d,
    )

    targets_only_camera = trimesh.util.concatenate((
        marker_sphere(dropper_camera, args.marker_radius, TARGET_COLORS["dropper_bulb"]),
        marker_sphere(beaker_camera, args.marker_radius, TARGET_COLORS["beaker_mouth"]),
    ))
    targets_only_world = targets_only_camera.copy()
    targets_only_world.apply_transform(world_from_sam3d)
    targets_only_camera_path = output_dir / "targets_only_camera.ply"
    targets_only_world_path = output_dir / "targets_only_world.ply"
    targets_only_camera.export(targets_only_camera_path)
    targets_only_world.export(targets_only_world_path)

    payload = {
        "coordinate_conventions": {
            "sam3d_camera": "x=-(u-cx)/fx*z, y=-(v-cy)/fy*z, z=depth",
            "opencv_camera": "x right, y down, z forward",
            "world_transform": "T_world_point = T_world_opencv_camera @ diag(-1,-1,1,1) @ point_sam3d_camera",
        },
        "calibration": calibration_info,
        "T_world_opencv_camera": world_from_opencv.tolist(),
        "T_opencv_camera_from_sam3d_camera": SAM3D_TO_OPENCV.tolist(),
        "T_world_from_sam3d_camera": world_from_sam3d.tolist(),
        "dropper_bulb": dropper,
        "beaker_mouth": beaker,
        "colors_rgba": {
            "dropper_object": OBJECT_COLORS["dropper"].tolist(),
            "beaker_object": OBJECT_COLORS["beaker"].tolist(),
            "table": OBJECT_COLORS["table"].tolist(),
            "dropper_bulb_marker": TARGET_COLORS["dropper_bulb"].tolist(),
            "beaker_mouth_marker": TARGET_COLORS["beaker_mouth"].tolist(),
        },
        "outputs": {
            "titration_scene_camera_ply": str(camera_scene_path),
            "titration_scene_world_ply": str(world_scene_path),
            "targets_only_camera_ply": str(targets_only_camera_path),
            "targets_only_world_ply": str(targets_only_world_path),
            **model_cloud_paths,
        },
    }
    if args.rgb and args.intrinsics:
        overlay_path, dropper_uv, beaker_uv = export_target_overlay(
            args.rgb,
            args.intrinsics,
            output_dir / "target_centers_overlay.png",
            dropper_camera,
            beaker_camera,
        )
        dropper["center_pixel_uv"] = dropper_uv
        beaker["center_pixel_uv"] = beaker_uv
        payload["outputs"]["target_centers_overlay_png"] = overlay_path
    if args.dropper_depth_cloud:
        payload["outputs"]["dropper_depth_camera_ply"] = str(Path(args.dropper_depth_cloud).resolve())
        payload["outputs"]["dropper_depth_world_ply"] = export_world_depth_cloud(
            args.dropper_depth_cloud,
            output_dir / "dropper_depth_world.ply",
            world_from_sam3d,
        )
    if args.beaker_depth_cloud:
        payload["outputs"]["beaker_depth_camera_ply"] = str(Path(args.beaker_depth_cloud).resolve())
        payload["outputs"]["beaker_depth_world_ply"] = export_world_depth_cloud(
            args.beaker_depth_cloud,
            output_dir / "beaker_depth_world.ply",
            world_from_sam3d,
        )
    json_path = output_dir / "titration_targets.json"
    write_json(json_path, payload)
    text_lines = [
        "Coordinates are in meters.",
        f"dropper_bulb_center_world: {' '.join(f'{v:.9f}' for v in dropper['center_world'])}",
        f"beaker_mouth_center_world: {' '.join(f'{v:.9f}' for v in beaker['center_world'])}",
        "dropper_bulb_marker_rgba: 255 0 255 255",
        "beaker_mouth_marker_rgba: 255 215 0 255",
    ]
    centers_txt_path = output_dir / "target_centers_world.txt"
    centers_txt_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    payload["outputs"]["targets_json"] = str(json_path)
    payload["outputs"]["target_centers_world_txt"] = str(centers_txt_path)
    write_json(json_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate masks and export two-object titration targets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-mask")
    validate.add_argument("--mask", required=True)
    validate.add_argument("--rgb", required=True)
    validate.add_argument("--label", required=True)
    validate.add_argument("--min-area", type=int, default=100)

    normalize_intrinsics = subparsers.add_parser("normalize-intrinsics")
    normalize_intrinsics.add_argument("--input", required=True)
    normalize_intrinsics.add_argument("--output", required=True)

    validate_calibration_parser = subparsers.add_parser("validate-calibration")
    validate_calibration_parser.add_argument("--calibration", required=True)
    validate_calibration_parser.add_argument("--world-ee", default=None)

    backproject = subparsers.add_parser("backproject")
    backproject.add_argument("--rgb", required=True)
    backproject.add_argument("--depth", required=True)
    backproject.add_argument("--intrinsics", required=True)
    backproject.add_argument("--dropper-mask", required=True)
    backproject.add_argument("--beaker-mask", required=True)
    backproject.add_argument("--output-dir", required=True)
    backproject.add_argument("--depth-scale", type=float, default=0.001)
    backproject.add_argument("--min-points", type=int, default=50)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--dropper-region", required=True)
    finalize.add_argument("--beaker-region", required=True)
    finalize.add_argument("--dropper-mesh", required=True)
    finalize.add_argument("--beaker-mesh", required=True)
    finalize.add_argument("--table-mesh", default=None)
    finalize.add_argument("--dropper-depth-cloud", default=None)
    finalize.add_argument("--beaker-depth-cloud", default=None)
    finalize.add_argument("--calibration", required=True)
    finalize.add_argument("--world-ee", default=None)
    finalize.add_argument("--output-dir", required=True)
    finalize.add_argument("--top-quantile", type=float, default=0.5)
    finalize.add_argument("--marker-radius", type=float, default=0.008)
    finalize.add_argument("--model-point-count", type=int, default=50000)
    finalize.add_argument("--rgb", default=None)
    finalize.add_argument("--intrinsics", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-mask":
        validate_mask(args.mask, args.rgb, args.label, args.min_area)
    elif args.command == "normalize-intrinsics":
        normalize_intrinsics_file(args.input, args.output)
    elif args.command == "validate-calibration":
        validate_calibration(args.calibration, args.world_ee)
    elif args.command == "backproject":
        export_depth_clouds(args)
    else:
        finalize_targets(args)


if __name__ == "__main__":
    main()
