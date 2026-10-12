from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from language_affordance_grasp.process import build_python_command, run_command


def run_yolov5_segmentation(
    config: dict[str, Any],
    rgb_path: str,
    output_dir: str | Path,
    class_name: str | None = None,
    class_id: int | None = None,
    selection: str | None = None,
    device: str | None = None,
    dry_run: bool = False,
) -> Path:
    yolo_repo = Path(config["paths"]["yolo_repo"])
    yolo_config = config["yolo"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    exporter_path = Path(__file__).resolve().parent / "yolo_mask_export.py"
    command_args = [
        "python",
        str(exporter_path),
        "--repo",
        str(yolo_repo),
        "--weights",
        str(yolo_config["weights"]),
        "--rgb",
        str(rgb_path),
        "--output-dir",
        str(output_path),
        "--selection",
        selection or str(yolo_config.get("selection", "best")),
        "--imgsz",
        str(yolo_config.get("imgsz", 640)),
        "--conf-thres",
        str(yolo_config.get("conf_thres", 0.25)),
        "--iou-thres",
        str(yolo_config.get("iou_thres", 0.45)),
        "--max-det",
        str(yolo_config.get("max_det", 1000)),
        "--device",
        device if device is not None else str(yolo_config.get("device", "")),
        "--mask-thresh",
        str(yolo_config.get("mask_thresh", 0.5)),
    ]
    if yolo_config.get("data"):
        command_args.extend(["--data", str(yolo_config["data"])])
    resolved_class_name = class_name if class_name is not None else yolo_config.get("class_name")
    resolved_class_id = class_id if class_id is not None else yolo_config.get("class_id")
    if resolved_class_id is not None:
        command_args.extend(["--class-id", str(resolved_class_id)])
    elif resolved_class_name:
        command_args.extend(["--class-name", str(resolved_class_name)])
    command = build_python_command(config["envs"].get("yolo"), command_args)
    run_command(command, cwd=yolo_repo, dry_run=dry_run)
    return output_path / "mask.png"


def run_fastsam_affordance(
    config: dict[str, Any],
    rgb_path: str,
    instruction: str,
    output_dir: str | Path,
    device: str | None = None,
    dry_run: bool = False,
) -> Path:
    fastsam_repo = Path(config["paths"]["fastsam_repo"])
    fastsam_config = config["fastsam"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    command = build_python_command(
        config["envs"].get("fastsam"),
        [
            "python",
            "-m",
            "affordance.infer_chem_hova",
            "--checkpoint",
            str(fastsam_config["checkpoint"]),
            "--rgb",
            str(rgb_path),
            "--instruction",
            instruction,
            "--output-dir",
            str(output_path),
            "--device",
            device or str(fastsam_config.get("device", "cuda")),
        ],
    )
    run_command(command, cwd=fastsam_repo, dry_run=dry_run)
    return output_path / "affordance_heatmap.npy"


def run_sam3d_recovery(
    config: dict[str, Any],
    rgb_path: str,
    depth_path: str,
    mask_path: str,
    table_mask_path: str | None,
    intrinsics_path: str,
    output_dir: str | Path,
    device: str | None = None,
    dry_run: bool = False,
) -> Path:
    sam3d_repo = Path(config["paths"]["sam3d_repo"])
    sam3d_config = config["sam3d"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    command_args = [
        "python",
        "recover_glass_pose_notable_graspnet.py",
        "--rgb",
        str(rgb_path),
        "--depth",
        str(depth_path),
        "--mask",
        str(mask_path),
        "--intrinsics-json",
        str(intrinsics_path),
        "--output-dir",
        str(output_path),
        "--sam3d-config",
        str(sam3d_config["config"]),
        "--depth-scale",
        str(sam3d_config.get("depth_scale", 0.001)),
        "--device",
        device or str(sam3d_config.get("device", "cuda")),
        "--seed",
        str(sam3d_config.get("seed", 42)),
        "--optimize-steps",
        str(sam3d_config.get("optimize_steps", 120)),
        "--combined-point-samples",
        str(sam3d_config.get("combined_point_samples", 120000)),
        "--table-point-ratio",
        str(sam3d_config.get("table_point_ratio", 0.35)),
    ]
    if table_mask_path:
        command_args.extend(["--table-mask", str(table_mask_path)])
    else:
        command_args.extend(["--table-mask", ""])
    command = build_python_command(config["envs"].get("sam3d"), command_args)
    run_command(command, cwd=sam3d_repo, dry_run=dry_run)
    return output_path / "pose_result.json"


def read_pose_result(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def run_graspnet(
    config: dict[str, Any],
    cloud_path: str,
    rgb_path: str | None,
    output_dir: str | Path,
    dry_run: bool = False,
) -> Path:
    graspnet_repo = Path(config["paths"]["graspnet_repo"])
    graspnet_config = config["graspnet"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_grasp_path = output_path / "grasp_candidates_raw.npy"
    command_args = [
        "python",
        "demo_me.py",
        "--checkpoint_path",
        str(graspnet_config["checkpoint"]),
        "--input_mode",
        "cloud",
        "--cloud_path",
        str(cloud_path),
        "--output_cloud_path",
        str(output_path / "grasp_candidates_raw_visualization.ply"),
        "--output_grasp_path",
        str(raw_grasp_path),
        "--num_point",
        str(graspnet_config.get("num_point", 20000)),
        "--num_view",
        str(graspnet_config.get("num_view", 300)),
        "--collision_thresh",
        str(graspnet_config.get("collision_thresh", 0.01)),
        "--voxel_size",
        str(graspnet_config.get("voxel_size", 0.01)),
        "--mesh_point_num",
        str(graspnet_config.get("mesh_point_num", 50000)),
        "--mesh_sample_mode",
        str(graspnet_config.get("mesh_sample_mode", "poisson")),
        "--top_k",
        str(graspnet_config.get("top_k_raw", 200)),
    ]
    if rgb_path:
        command_args.extend(["--rgb_path", str(rgb_path)])
    command = build_python_command(config["envs"].get("graspnet"), command_args)
    run_command(command, cwd=graspnet_repo, dry_run=dry_run)
    return raw_grasp_path
