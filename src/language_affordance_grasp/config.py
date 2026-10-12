from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "fastsam_repo": "/home/dsj/code/FastSAM",
        "yolo_repo": "/home/dsj/code/yolo_3D",
        "sam3d_repo": "/home/dsj/code/sam-3d-objects",
        "graspnet_repo": "/home/dsj/code/graspnet-baseline",
    },
    "envs": {
        "yolo": "",
        "fastsam": "fastsam-annot",
        "sam3d": "",
        "graspnet": "",
    },
    "yolo": {
        "weights": "/home/dsj/code/yolo_3D/runs_edge_p2_sim/train-seg/exp/weights/best.pt",
        "data": "/home/dsj/dataset/mydata/server/mydata.yaml",
        "class_name": "Glass dropper bottle",
        "class_id": None,
        "selection": "best",
        "imgsz": 640,
        "conf_thres": 0.25,
        "iou_thres": 0.45,
        "max_det": 1000,
        "device": "",
        "mask_thresh": 0.5,
    },
    "fastsam": {
        "checkpoint": "/home/dsj/code/FastSAM/affordance_runs/chem_hova_fullimg_e5/best.pt",
        "device": "cuda",
    },
    "sam3d": {
        "config": "checkpoints/hf/pipeline.yaml",
        "device": "cuda",
        "depth_scale": 0.001,
        "seed": 42,
        "optimize_steps": 120,
        "combined_point_samples": 120000,
        "table_point_ratio": 0.35,
    },
    "projection": {
        "sample_count": 80000,
        "threshold_min": 0.35,
        "threshold_quantile": 0.85,
        "expansion_radius": 0.01,
        "keep_visible_only": True,
    },
    "graspnet": {
        "checkpoint": "/home/dsj/code/graspnet-baseline/checkpoint-rs.tar",
        "num_point": 20000,
        "num_view": 300,
        "collision_thresh": 0.01,
        "voxel_size": 0.01,
        "mesh_point_num": 50000,
        "mesh_sample_mode": "poisson",
        "top_k_raw": 200,
    },
    "filter": {
        "top_k": 20,
        "min_score": 0.0,
        "max_width": 0.09,
        "max_center_distance": 0.045,
        "max_contact_distance": 0.04,
        "distance_score_scale": 0.03,
        "approach_angle_thresh_deg": 65.0,
    },
    "robot": {
        "tool_length": 0.175,
        "grasp_depth_offset": 0.0,
        "pregrasp_distance": 0.10,
        "retreat_distance": 0.08,
        "lift_distance": 0.10,
    },
}


def _deep_merge(base_config: dict[str, Any], override_config: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base_config)
    for key, value in override_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as file_handle:
        if suffix == ".json":
            return json.load(file_handle)
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML config files.") from exc
        loaded = yaml.safe_load(file_handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected config mapping in {path}")
    return loaded


def load_config(path: str | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        config = _deep_merge(config, _load_mapping(Path(path)))
    return config


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)


def require_path(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved
