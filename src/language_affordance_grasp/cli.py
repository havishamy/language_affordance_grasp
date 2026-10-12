from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from language_affordance_grasp.adapters import (
    read_pose_result,
    run_fastsam_affordance,
    run_graspnet,
    run_sam3d_recovery,
    run_yolov5_segmentation,
)
from language_affordance_grasp.config import load_config, require_path, write_json
from language_affordance_grasp.grasp_filter import filter_grasps_by_affordance
from language_affordance_grasp.robot_export import export_selected_grasp


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_run_dir(output_dir: str | None) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser()
    else:
        path = Path("/home/dsj/code/language_affordance_grasp/runs") / timestamp()
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="/home/dsj/code/language_affordance_grasp/configs/default.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print external commands without executing heavy models.")


def add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--mask", default=None, help="Object mask PNG. If omitted, YOLOv5 segmentation is used.")
    parser.add_argument("--table-mask", default=None)
    parser.add_argument("--intrinsics-json", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--t-base-cam", default=None)
    parser.add_argument("--yolo-class-name", default=None, help="Class name for automatic YOLO mask export.")
    parser.add_argument("--yolo-class-id", type=int, default=None, help="Class id for automatic YOLO mask export.")
    parser.add_argument("--segmentation-selection", choices=["best", "union"], default=None)


def validate_inputs(args: argparse.Namespace, config: dict[str, Any]) -> None:
    require_path(args.rgb, "RGB image")
    require_path(args.depth, "Depth image")
    if args.mask:
        require_path(args.mask, "Object mask")
    elif not getattr(args, "pose_result", None):
        require_path(config["paths"]["yolo_repo"], "YOLO repository")
        require_path(config["yolo"]["weights"], "YOLO segmentation checkpoint")
        if config["yolo"].get("data"):
            require_path(config["yolo"]["data"], "YOLO data YAML")
    require_path(args.intrinsics_json, "Intrinsics JSON")
    if args.table_mask:
        require_path(args.table_mask, "Table mask")
    if args.t_base_cam:
        require_path(args.t_base_cam, "T_base_cam")


def expected_pose_paths(pose_result_path: Path) -> tuple[Path, Path]:
    sam3d_dir = pose_result_path.parent
    return sam3d_dir / "optimized_mesh_camera.ply", sam3d_dir / "combined_table_object_camera_pointcloud.ply"


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    validate_inputs(args, config)
    run_dir = ensure_run_dir(args.output_dir)
    segmentation_dir = run_dir / "00_segmentation"
    affordance_dir = run_dir / "01_affordance"
    sam3d_dir = run_dir / "02_sam3d"
    projection_dir = run_dir / "03_projection"
    graspnet_dir = run_dir / "04_graspnet"
    robot_dir = run_dir / "05_robot"
    write_json(run_dir / "run_inputs.json", {key: value for key, value in vars(args).items() if key != "func"})
    write_json(run_dir / "run_config.json", config)

    mask_path = Path(args.mask) if args.mask else None
    if mask_path is None and not args.pose_result:
        mask_path = run_yolov5_segmentation(
            config=config,
            rgb_path=args.rgb,
            output_dir=segmentation_dir,
            class_name=args.yolo_class_name,
            class_id=args.yolo_class_id,
            selection=args.segmentation_selection,
            dry_run=args.dry_run,
        )

    heatmap_path = Path(args.heatmap) if args.heatmap else run_fastsam_affordance(
        config=config,
        rgb_path=args.rgb,
        instruction=args.instruction,
        output_dir=affordance_dir,
        dry_run=args.dry_run,
    )

    if args.pose_result:
        pose_result_path = Path(args.pose_result)
    else:
        pose_result_path = run_sam3d_recovery(
            config=config,
            rgb_path=args.rgb,
            depth_path=args.depth,
            mask_path=str(mask_path),
            table_mask_path=args.table_mask,
            intrinsics_path=args.intrinsics_json,
            output_dir=sam3d_dir,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        object_mesh_path, cloud_path = expected_pose_paths(pose_result_path)
        projection_result = None
    else:
        require_path(heatmap_path, "Affordance heatmap")
        require_path(pose_result_path, "SAM3D pose_result.json")
        pose_result = read_pose_result(pose_result_path)
        object_mesh_path = Path(pose_result["optimized_mesh_camera_ply"])
        cloud_path = Path(pose_result["combined_pointcloud_camera_ply"])
        projection_cfg = config["projection"]
        from language_affordance_grasp.projection import project_affordance_to_mesh

        projection_result = project_affordance_to_mesh(
            heatmap_path=heatmap_path,
            mesh_path=object_mesh_path,
            intrinsics_path=args.intrinsics_json,
            output_dir=projection_dir,
            sample_count=int(projection_cfg.get("sample_count", 80000)),
            threshold_min=float(projection_cfg.get("threshold_min", 0.35)),
            threshold_quantile=float(projection_cfg.get("threshold_quantile", 0.85)),
            keep_visible_only=bool(projection_cfg.get("keep_visible_only", True)),
            expansion_radius=float(projection_cfg.get("expansion_radius", 0.0)),
        )

    raw_grasp_path = Path(args.raw_grasps) if args.raw_grasps else run_graspnet(
        config=config,
        cloud_path=str(cloud_path),
        rgb_path=args.rgb,
        output_dir=graspnet_dir,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        summary = {
            "dry_run": True,
            "run_dir": str(run_dir),
            "expected_mask_path": str(mask_path) if mask_path else None,
            "expected_heatmap_path": str(heatmap_path),
            "expected_pose_result_path": str(pose_result_path),
            "expected_object_mesh_path": str(object_mesh_path),
            "expected_cloud_path": str(cloud_path),
            "expected_raw_grasp_path": str(raw_grasp_path),
        }
        write_json(run_dir / "summary.json", summary)
        return summary

    require_path(raw_grasp_path, "Raw GraspNet grasps")
    filtered = filter_grasps_by_affordance(
        raw_grasp_path=raw_grasp_path,
        region_npz_path=projection_result.region_npz_path,
        output_dir=graspnet_dir,
        instruction=args.instruction,
        filter_config=config["filter"],
        t_base_cam_path=args.t_base_cam,
    )
    selected = export_selected_grasp(
        filtered_json_path=graspnet_dir / "grasp_candidates_filtered.json",
        output_dir=robot_dir,
        robot_config=config["robot"],
        t_base_cam_path=args.t_base_cam,
        grasp_index=int(args.grasp_index),
    )
    summary = {
        "dry_run": False,
        "run_dir": str(run_dir),
        "mask_path": str(mask_path) if mask_path else None,
        "heatmap_path": str(heatmap_path),
        "pose_result_path": str(pose_result_path),
        "object_mesh_path": str(object_mesh_path),
        "cloud_path": str(cloud_path),
        "projection": asdict(projection_result),
        "raw_grasp_path": str(raw_grasp_path),
        "filtered": filtered,
        "selected": selected,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def project_command(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    output_dir = ensure_run_dir(args.output_dir)
    projection_cfg = config["projection"]
    from language_affordance_grasp.projection import project_affordance_to_mesh

    result = project_affordance_to_mesh(
        heatmap_path=args.heatmap,
        mesh_path=args.mesh,
        intrinsics_path=args.intrinsics_json,
        output_dir=output_dir,
        sample_count=int(args.sample_count or projection_cfg.get("sample_count", 80000)),
        threshold_min=float(args.threshold_min or projection_cfg.get("threshold_min", 0.35)),
        threshold_quantile=float(args.threshold_quantile or projection_cfg.get("threshold_quantile", 0.85)),
        keep_visible_only=not args.keep_backfaces,
        expansion_radius=float(args.expansion_radius if args.expansion_radius is not None else projection_cfg.get("expansion_radius", 0.0)),
    )
    return asdict(result)


def filter_command(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    output_dir = ensure_run_dir(args.output_dir)
    return filter_grasps_by_affordance(
        raw_grasp_path=args.raw_grasps,
        region_npz_path=args.region_npz,
        output_dir=output_dir,
        instruction=args.instruction,
        filter_config=config["filter"],
        t_base_cam_path=args.t_base_cam,
    )


def export_command(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    output_dir = ensure_run_dir(args.output_dir)
    return export_selected_grasp(
        filtered_json_path=args.filtered_json,
        output_dir=output_dir,
        robot_config=config["robot"],
        t_base_cam_path=args.t_base_cam,
        grasp_index=args.grasp_index,
    )


def segment_command(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    require_path(args.rgb, "RGB image")
    require_path(config["paths"]["yolo_repo"], "YOLO repository")
    require_path(config["yolo"]["weights"], "YOLO segmentation checkpoint")
    if config["yolo"].get("data"):
        require_path(config["yolo"]["data"], "YOLO data YAML")
    output_dir = ensure_run_dir(args.output_dir)
    mask_path = run_yolov5_segmentation(
        config=config,
        rgb_path=args.rgb,
        output_dir=output_dir,
        class_name=args.yolo_class_name,
        class_id=args.yolo_class_id,
        selection=args.segmentation_selection,
        dry_run=args.dry_run,
    )
    return {
        "dry_run": bool(args.dry_run),
        "mask_path": str(mask_path),
        "mask_prob_path": str(output_dir / "mask_prob.png"),
        "overlay_path": str(output_dir / "overlay.png"),
        "detections_path": str(output_dir / "detections.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Language affordance to SAM3D to GraspNet pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full offline pipeline.")
    add_common(run_parser)
    add_inputs(run_parser)
    run_parser.add_argument("--heatmap", default=None, help="Reuse an existing affordance_heatmap.npy.")
    run_parser.add_argument("--pose-result", default=None, help="Reuse an existing SAM3D pose_result.json.")
    run_parser.add_argument("--raw-grasps", default=None, help="Reuse an existing GraspNet .npy output.")
    run_parser.add_argument("--grasp-index", type=int, default=0)
    run_parser.set_defaults(func=run_pipeline)

    segment_parser = subparsers.add_parser("segment", help="Export a YOLOv5 segmentation mask for one object class.")
    add_common(segment_parser)
    segment_parser.add_argument("--rgb", required=True)
    segment_parser.add_argument("--yolo-class-name", default=None)
    segment_parser.add_argument("--yolo-class-id", type=int, default=None)
    segment_parser.add_argument("--segmentation-selection", choices=["best", "union"], default=None)
    segment_parser.set_defaults(func=segment_command)

    project_parser = subparsers.add_parser("project", help="Project 2D affordance onto a camera-frame mesh.")
    add_common(project_parser)
    project_parser.add_argument("--heatmap", required=True)
    project_parser.add_argument("--mesh", required=True)
    project_parser.add_argument("--intrinsics-json", required=True)
    project_parser.add_argument("--sample-count", type=int, default=None)
    project_parser.add_argument("--threshold-min", type=float, default=None)
    project_parser.add_argument("--threshold-quantile", type=float, default=None)
    project_parser.add_argument("--keep-backfaces", action="store_true")
    project_parser.add_argument("--expansion-radius", type=float, default=None)
    project_parser.set_defaults(func=project_command)

    filter_parser = subparsers.add_parser("filter", help="Filter GraspNet grasps by projected affordance region.")
    add_common(filter_parser)
    filter_parser.add_argument("--raw-grasps", required=True)
    filter_parser.add_argument("--region-npz", required=True)
    filter_parser.add_argument("--instruction", required=True)
    filter_parser.add_argument("--t-base-cam", default=None)
    filter_parser.set_defaults(func=filter_command)

    export_parser = subparsers.add_parser("export", help="Export selected camera/base/TCP grasp JSON.")
    add_common(export_parser)
    export_parser.add_argument("--filtered-json", required=True)
    export_parser.add_argument("--t-base-cam", default=None)
    export_parser.add_argument("--grasp-index", type=int, default=0)
    export_parser.set_defaults(func=export_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
