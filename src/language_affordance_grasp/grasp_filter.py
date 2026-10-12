from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from language_affordance_grasp.config import write_json


@dataclass
class FilteredGrasp:
    index: int
    final_score: float
    graspnet_score: float
    affordance_score: float
    distance_score: float
    task_score: float
    width: float
    height: float
    depth: float
    translation_camera: list[float]
    rotation_matrix_camera: list[list[float]]
    center_distance: float
    left_contact_distance: float
    right_contact_distance: float
    approach_angle_base_deg: float | None


def load_grasp_array(path: str | Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 17:
        raise ValueError(f"Expected GraspNet npy shape (N, 17), got {arr.shape}")
    return arr


def load_region(path: str | Path) -> dict[str, np.ndarray]:
    pack = np.load(path)
    points = np.asarray(pack["points"], dtype=np.float64)
    scores = np.asarray(pack["scores"], dtype=np.float64)
    normals = np.asarray(pack["normals"], dtype=np.float64) if "normals" in pack else np.zeros_like(points)
    return {"points": points, "scores": scores, "normals": normals}


def nearest_region_distance(query_points: np.ndarray, region_points: np.ndarray, region_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(region_points) == 0:
        return np.full((len(query_points),), np.inf), np.zeros((len(query_points),), dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(region_points)
        distances, indices = tree.query(query_points, k=1)
        return distances.astype(np.float64), region_scores[indices].astype(np.float64)
    except Exception:
        distances = []
        scores = []
        for query in query_points:
            all_distances = np.linalg.norm(region_points - query[None, :], axis=1)
            index = int(np.argmin(all_distances))
            distances.append(float(all_distances[index]))
            scores.append(float(region_scores[index]))
        return np.asarray(distances), np.asarray(scores)


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-9)


def load_transform(path: str | Path | None) -> np.ndarray | None:
    if path is None:
        return None
    transform_path = Path(path)
    transform = np.load(transform_path) if transform_path.suffix == ".npy" else np.loadtxt(transform_path)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {transform.shape}")
    return transform


def approach_angle_base(rotation_camera: np.ndarray, t_base_cam: np.ndarray | None) -> float | None:
    if t_base_cam is None:
        return None
    approach_camera = rotation_camera[:, 0]
    approach_base = normalize(t_base_cam[:3, :3] @ approach_camera)
    down_base = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    cosine = float(np.clip(np.dot(approach_base, down_base), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def infer_task_score(instruction: str, rotation_camera: np.ndarray, translation: np.ndarray, active_center: np.ndarray | None) -> float:
    text = instruction.lower()
    score = 1.0
    approach = normalize(rotation_camera[:, 0])
    closing = normalize(rotation_camera[:, 1])
    if "open" in text or "lid" in text or "cap" in text:
        score *= 1.10
        if active_center is not None:
            radial = translation - active_center
            radial[2] = 0.0
            if np.linalg.norm(radial) > 1e-6:
                tangent_alignment = abs(float(np.dot(closing, normalize(radial))))
                score *= 0.85 + 0.30 * tangent_alignment
        score *= 0.90 + 0.20 * abs(float(approach[2]))
    if "hold" in text or "grasp" in text or "pick" in text:
        score *= 1.05
    return float(score)


def grasp_to_txt_block(index: int, grasp: np.ndarray) -> str:
    score = grasp[0]
    width = grasp[1]
    height = grasp[2]
    depth = grasp[3]
    rotation = grasp[4:13].reshape(3, 3)
    translation = grasp[13:16]
    lines = [
        f"grasp {index}",
        f"score {score:.6f}",
        f"width {width:.6f} height {height:.6f} depth {depth:.6f}",
        f"translation {translation[0]:.6f} {translation[1]:.6f} {translation[2]:.6f}",
        "rotation",
    ]
    lines.extend(" ".join(f"{value:.6f}" for value in row) for row in rotation)
    return "\n".join(lines) + "\n"


def save_grasps_txt(path: str | Path, grasps: np.ndarray) -> None:
    with Path(path).open("w", encoding="utf-8") as file_handle:
        for index, grasp in enumerate(grasps):
            file_handle.write(grasp_to_txt_block(index, grasp))
            file_handle.write("\n")


def filter_grasps_by_affordance(
    raw_grasp_path: str | Path,
    region_npz_path: str | Path,
    output_dir: str | Path,
    instruction: str,
    filter_config: dict[str, Any],
    t_base_cam_path: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    grasps = load_grasp_array(raw_grasp_path)
    region = load_region(region_npz_path)
    region_points = region["points"]
    region_scores = region["scores"]
    active_center = None
    if len(region_points):
        active_center = np.average(region_points, axis=0, weights=np.maximum(region_scores, 1e-4))

    t_base_cam = load_transform(t_base_cam_path)
    max_width = float(filter_config.get("max_width", 0.09))
    min_score = float(filter_config.get("min_score", 0.0))
    max_center_distance = float(filter_config.get("max_center_distance", 0.045))
    max_contact_distance = float(filter_config.get("max_contact_distance", 0.04))
    scale = max(float(filter_config.get("distance_score_scale", 0.03)), 1e-6)
    angle_thresh = float(filter_config.get("approach_angle_thresh_deg", 65.0))
    top_k = int(filter_config.get("top_k", 20))

    ranked: list[tuple[float, np.ndarray, FilteredGrasp]] = []
    for index, grasp in enumerate(grasps):
        graspnet_score = float(grasp[0])
        width = float(grasp[1])
        if graspnet_score < min_score or width > max_width:
            continue
        rotation = grasp[4:13].reshape(3, 3)
        translation = grasp[13:16]
        closing = normalize(rotation[:, 1])
        left_contact = translation + closing * (0.5 * width)
        right_contact = translation - closing * (0.5 * width)
        distances, scores = nearest_region_distance(np.stack([translation, left_contact, right_contact]), region_points, region_scores)
        center_distance = float(distances[0])
        left_distance = float(distances[1])
        right_distance = float(distances[2])
        contact_distance = min(left_distance, right_distance)
        if center_distance > max_center_distance and contact_distance > max_contact_distance:
            continue
        angle_deg = approach_angle_base(rotation, t_base_cam)
        if angle_deg is not None and angle_deg > angle_thresh:
            continue
        affordance_score = float(max(scores))
        distance_score = math.exp(-min(center_distance, contact_distance) / scale)
        task_score = infer_task_score(instruction, rotation, translation, active_center)
        final_score = float(graspnet_score * (0.30 + 0.70 * affordance_score) * distance_score * task_score)
        ranked.append((final_score, grasp.copy(), FilteredGrasp(
            index=index,
            final_score=final_score,
            graspnet_score=graspnet_score,
            affordance_score=affordance_score,
            distance_score=float(distance_score),
            task_score=task_score,
            width=width,
            height=float(grasp[2]),
            depth=float(grasp[3]),
            translation_camera=translation.tolist(),
            rotation_matrix_camera=rotation.tolist(),
            center_distance=center_distance,
            left_contact_distance=left_distance,
            right_contact_distance=right_distance,
            approach_angle_base_deg=angle_deg,
        )))

    ranked.sort(key=lambda item: item[0], reverse=True)
    ranked = ranked[:top_k]
    selected_arrays = np.stack([item[1] for item in ranked], axis=0) if ranked else np.zeros((0, 17), dtype=np.float64)
    filtered_npy = output_path / "grasp_candidates_filtered.npy"
    filtered_txt = output_path / "grasp_candidates_filtered.txt"
    filtered_json = output_path / "grasp_candidates_filtered.json"
    np.save(filtered_npy, selected_arrays)
    save_grasps_txt(filtered_txt, selected_arrays)
    payload = {
        "raw_grasp_path": str(raw_grasp_path),
        "region_npz_path": str(region_npz_path),
        "instruction": instruction,
        "input_count": int(len(grasps)),
        "filtered_count": int(len(ranked)),
        "best_grasp": asdict(ranked[0][2]) if ranked else None,
        "grasps": [asdict(item[2]) for item in ranked],
        "filtered_npy_path": str(filtered_npy),
        "filtered_txt_path": str(filtered_txt),
    }
    write_json(filtered_json, payload)
    return payload
