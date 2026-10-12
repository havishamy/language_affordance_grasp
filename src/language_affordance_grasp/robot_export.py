from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import numpy as np

from language_affordance_grasp.config import write_json


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-9)


def rotation_matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    angle = float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    axis = np.asarray([
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
    ], dtype=np.float64)
    axis = axis / (2.0 * np.sin(angle) + 1e-9)
    return axis * angle


def load_transform(path: str | Path) -> np.ndarray:
    transform_path = Path(path)
    transform = np.load(transform_path) if transform_path.suffix == ".npy" else np.loadtxt(transform_path)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {transform.shape}")
    return transform


def build_tcp_pose(t_base_grasp: np.ndarray, tool_length: float, grasp_depth_offset: float) -> np.ndarray:
    target = t_base_grasp.copy()
    approach = normalize(target[:3, 0])
    closing = normalize(target[:3, 1])
    lateral = normalize(np.cross(approach, closing))
    closing = normalize(np.cross(lateral, approach))
    target[:3, 3] = target[:3, 3] + approach * float(grasp_depth_offset)
    tcp_z = approach
    tcp_x = closing
    tcp_y = normalize(np.cross(tcp_z, tcp_x))
    tcp_x = normalize(np.cross(tcp_y, tcp_z))
    r_tcp = np.stack([tcp_x, tcp_y, tcp_z], axis=1)
    tcp_pos = target[:3, 3] - tcp_z * float(tool_length)
    return np.concatenate([tcp_pos, rotation_matrix_to_rotvec(r_tcp)])


def export_selected_grasp(
    filtered_json_path: str | Path,
    output_dir: str | Path,
    robot_config: dict[str, Any],
    t_base_cam_path: str | Path | None = None,
    grasp_index: int = 0,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with Path(filtered_json_path).open("r", encoding="utf-8") as file_handle:
        filtered = json.load(file_handle)
    grasps = filtered.get("grasps", [])
    if not grasps:
        payload = {"success": False, "error": "No filtered grasps available."}
        write_json(output_path / "selected_grasp.json", payload)
        return payload
    if grasp_index < 0 or grasp_index >= len(grasps):
        raise IndexError(f"grasp_index {grasp_index} out of range [0, {len(grasps)})")

    selected = grasps[grasp_index]
    t_cam_grasp = np.eye(4, dtype=np.float64)
    t_cam_grasp[:3, :3] = np.asarray(selected["rotation_matrix_camera"], dtype=np.float64)
    t_cam_grasp[:3, 3] = np.asarray(selected["translation_camera"], dtype=np.float64)
    filtered_txt_path = filtered.get("filtered_txt_path")
    execute_args = [
        "python",
        "/home/dsj/code/graspnet-baseline/robot_execute_grasp_txt.py",
        "--grasp_file",
        str(filtered_txt_path) if filtered_txt_path else "",
        "--grasp_index",
        str(grasp_index),
        "--dry_run",
    ]
    if t_base_cam_path:
        execute_args.extend(["--t_base_cam_path", str(t_base_cam_path)])
    payload = {
        "success": True,
        "selected_index": grasp_index,
        "selected_grasp": selected,
        "T_cam_grasp": t_cam_grasp.tolist(),
        "T_base_cam_path": str(t_base_cam_path) if t_base_cam_path else None,
        "T_base_grasp": None,
        "tcp_pose_base": None,
        "pregrasp_tcp_pose_base": None,
        "retreat_tcp_pose_base": None,
        "lift_tcp_pose_base": None,
        "robot_dry_run_command": " ".join(shlex.quote(item) for item in execute_args if item),
    }
    if t_base_cam_path:
        t_base_cam = load_transform(t_base_cam_path)
        t_base_grasp = t_base_cam @ t_cam_grasp
        tcp_pose = build_tcp_pose(
            t_base_grasp,
            tool_length=float(robot_config.get("tool_length", 0.175)),
            grasp_depth_offset=float(robot_config.get("grasp_depth_offset", 0.0)),
        )
        approach = normalize(t_base_grasp[:3, 0])
        pregrasp = tcp_pose.copy()
        pregrasp[:3] -= approach * float(robot_config.get("pregrasp_distance", 0.10))
        retreat = tcp_pose.copy()
        retreat[:3] -= approach * float(robot_config.get("retreat_distance", 0.08))
        lift = retreat.copy()
        lift[2] += float(robot_config.get("lift_distance", 0.10))
        payload.update({
            "T_base_grasp": t_base_grasp.tolist(),
            "tcp_pose_base": tcp_pose.tolist(),
            "pregrasp_tcp_pose_base": pregrasp.tolist(),
            "retreat_tcp_pose_base": retreat.tolist(),
            "lift_tcp_pose_base": lift.tolist(),
        })
    write_json(output_path / "selected_grasp.json", payload)
    return payload
