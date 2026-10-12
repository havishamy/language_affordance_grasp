from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from language_affordance_grasp.config import write_json


@dataclass
class ProjectionResult:
    heatmap_path: str
    mesh_path: str
    intrinsics_path: str
    sample_count: int
    visible_count: int
    threshold: float
    active_count: int
    active_center_camera: list[float] | None
    active_bounds_camera: list[list[float]] | None
    affordance_surface_path: str | None
    affordance_region_path: str | None
    region_npz_path: str
    workspace_json_path: str


def load_intrinsics(path: str | Path) -> np.ndarray:
    with Path(path).open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    if "K" in data:
        intrinsics = np.asarray(data["K"], dtype=np.float32)
    elif "intrinsics" in data:
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
    else:
        fx = float(data["fx"])
        fy = float(data.get("fy", fx))
        cx = float(data["cx"])
        cy = float(data["cy"])
        intrinsics = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics, got {intrinsics.shape}")
    return intrinsics


def load_heatmap(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        heatmap = np.load(path)
    elif path.suffix.lower() == ".npz":
        pack = np.load(path)
        key = "affordance" if "affordance" in pack else pack.files[0]
        heatmap = pack[key]
    else:
        heatmap = np.asarray(Image.open(path)).astype(np.float32)
    if heatmap.ndim == 3:
        heatmap = heatmap[..., 0]
    heatmap = heatmap.astype(np.float32)
    heatmap[~np.isfinite(heatmap)] = 0.0
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value > min_value:
        heatmap = (heatmap - min_value) / (max_value - min_value)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)
    return heatmap


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh_like = trimesh.load(str(path), process=False)
    if isinstance(mesh_like, trimesh.Scene):
        if not mesh_like.geometry:
            raise ValueError(f"No geometry found in {path}")
        mesh = trimesh.util.concatenate(tuple(geom.copy() for geom in mesh_like.geometry.values()))
    elif isinstance(mesh_like, trimesh.Trimesh):
        mesh = mesh_like.copy()
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh_like)}")
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {path}")
    return mesh


def project_points_camera(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.clip(points[:, 2], 1e-8, None)
    u = -(points[:, 0] / z) * float(intrinsics[0, 0]) + float(intrinsics[0, 2])
    v = -(points[:, 1] / z) * float(intrinsics[1, 1]) + float(intrinsics[1, 2])
    return np.stack([u, v], axis=1).astype(np.float32)


def bilinear_sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip(uv[:, 0], 0.0, width - 1.0)
    y = np.clip(uv[:, 1], 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    return (wa * image[y0, x0] + wb * image[y0, x1] + wc * image[y1, x0] + wd * image[y1, x1]).astype(np.float32)


def visible_facing_mask(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    view_dirs = -points
    view_dirs = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8)
    return np.sum(normals * view_dirs, axis=1) > 0.0


def heat_colors(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(scores, 0.0, 1.0)
    red = np.clip(1.8 * scores, 0.0, 1.0)
    green = np.clip(1.8 * (1.0 - np.abs(scores - 0.5) * 2.0), 0.0, 1.0)
    blue = np.clip(1.8 * (1.0 - scores), 0.0, 1.0)
    return np.stack([red, green, blue], axis=1) * 255.0


def export_point_cloud(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> str | None:
    if len(points) == 0:
        return None
    cloud = trimesh.points.PointCloud(points.astype(np.float32), colors=None if colors is None else colors.astype(np.uint8))
    cloud.export(str(path))
    return str(path)


def compute_threshold(scores: np.ndarray, threshold_min: float, threshold_quantile: float) -> float:
    if len(scores) == 0:
        return float(threshold_min)
    return max(float(threshold_min), float(np.quantile(scores, float(threshold_quantile))))


def expand_active_region(points: np.ndarray, active: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0 or not np.any(active) or len(points) == 0:
        return active
    active_points = points[active]
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(active_points)
        distances, _ = tree.query(points, k=1)
        return active | (distances <= radius)
    except Exception:
        expanded = active.copy()
        for start in range(0, len(points), 512):
            chunk = points[start : start + 512]
            distances = np.linalg.norm(chunk[:, None, :] - active_points[None, :, :], axis=2)
            expanded[start : start + len(chunk)] |= distances.min(axis=1) <= radius
        return expanded


def project_affordance_to_mesh(
    heatmap_path: str | Path,
    mesh_path: str | Path,
    intrinsics_path: str | Path,
    output_dir: str | Path,
    sample_count: int = 80000,
    threshold_min: float = 0.35,
    threshold_quantile: float = 0.85,
    keep_visible_only: bool = True,
    expansion_radius: float = 0.0,
) -> ProjectionResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    heatmap = load_heatmap(heatmap_path)
    intrinsics = load_intrinsics(intrinsics_path)
    mesh = load_mesh(mesh_path)

    points, face_indices = trimesh.sample.sample_surface(mesh, int(sample_count))
    normals = mesh.face_normals[face_indices].astype(np.float32)
    points = points.astype(np.float32)
    uv = project_points_camera(points, intrinsics)
    height, width = heatmap.shape[:2]
    inside = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= width - 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= height - 1.0)
        & (points[:, 2] > 1e-6)
    )
    if keep_visible_only:
        inside &= visible_facing_mask(points, normals)
    points = points[inside]
    normals = normals[inside]
    uv = uv[inside]
    scores = bilinear_sample(heatmap, uv) if len(points) else np.zeros((0,), dtype=np.float32)

    threshold = compute_threshold(scores, threshold_min, threshold_quantile)
    active = scores >= threshold
    active = expand_active_region(points, active, float(expansion_radius))
    active_points = points[active]
    active_normals = normals[active]
    active_scores = scores[active]
    active_uv = uv[active]

    surface_path = export_point_cloud(output_path / "affordance_surface_camera.ply", points, heat_colors(scores))
    region_path = export_point_cloud(output_path / "affordance_region_camera.ply", active_points, heat_colors(active_scores))
    region_npz_path = output_path / "affordance_region_camera.npz"
    np.savez_compressed(
        region_npz_path,
        points=active_points.astype(np.float32),
        normals=active_normals.astype(np.float32),
        scores=active_scores.astype(np.float32),
        uv=active_uv.astype(np.float32),
        threshold=np.asarray([threshold], dtype=np.float32),
    )

    if len(active_points):
        center = np.average(active_points, axis=0, weights=np.maximum(active_scores, 1e-4)).astype(np.float32)
        bounds = np.stack([active_points.min(axis=0), active_points.max(axis=0)], axis=0).astype(np.float32)
        center_list = center.tolist()
        bounds_list = bounds.tolist()
    else:
        center_list = None
        bounds_list = None

    workspace_path = output_path / "workspace_3d.json"
    result = ProjectionResult(
        heatmap_path=str(heatmap_path),
        mesh_path=str(mesh_path),
        intrinsics_path=str(intrinsics_path),
        sample_count=int(sample_count),
        visible_count=int(len(points)),
        threshold=float(threshold),
        active_count=int(len(active_points)),
        active_center_camera=center_list,
        active_bounds_camera=bounds_list,
        affordance_surface_path=surface_path,
        affordance_region_path=region_path,
        region_npz_path=str(region_npz_path),
        workspace_json_path=str(workspace_path),
    )
    write_json(workspace_path, asdict(result))
    return result
