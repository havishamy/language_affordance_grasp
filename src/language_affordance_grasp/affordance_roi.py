from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from language_affordance_grasp.config import write_json


METHODS = ("expanded_crop", "mask_others", "crop_mask_others")


def load_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if mask.shape != shape:
        raise ValueError(f"Mask {path} has shape {mask.shape}, expected {shape}")
    return mask


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Target mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expanded_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    expansion: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    height, width = image_shape
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = int(round(box_width * float(expansion)))
    pad_y = int(round(box_height * float(expansion)))
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def inpaint_other_objects(rgb: np.ndarray, other_mask: np.ndarray, dilation_px: int) -> np.ndarray:
    kernel_size = max(1, int(dilation_px) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(other_mask.astype(np.uint8) * 255, kernel, iterations=1)
    return cv2.inpaint(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dilated, 5.0, cv2.INPAINT_TELEA)[..., ::-1]


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"), dtype=np.uint8)
    shape = rgb.shape[:2]
    target_mask = load_mask(args.target_mask, shape)
    other_mask = np.zeros(shape, dtype=bool)
    for path in args.other_mask:
        other_mask |= load_mask(path, shape)
    other_mask &= ~target_mask

    target_bbox = mask_bbox(target_mask)
    crop_bbox = expanded_bbox(target_bbox, shape, args.expansion)
    prepared = rgb.copy()
    if args.method in {"mask_others", "crop_mask_others"} and np.any(other_mask):
        prepared = inpaint_other_objects(prepared, other_mask, args.other_dilation)
    if args.method in {"expanded_crop", "crop_mask_others"}:
        x1, y1, x2, y2 = crop_bbox
        model_input = prepared[y1:y2, x1:x2]
        remap_bbox: list[int] | None = [x1, y1, x2, y2]
    else:
        model_input = prepared
        remap_bbox = None

    model_input_path = output_dir / "model_input.png"
    Image.fromarray(model_input).save(model_input_path)
    Image.fromarray(prepared).save(output_dir / "prepared_full_rgb.png")
    Image.fromarray(target_mask.astype(np.uint8) * 255).save(output_dir / "target_mask.png")
    Image.fromarray(other_mask.astype(np.uint8) * 255).save(output_dir / "other_objects_mask.png")
    metadata = {
        "method": args.method,
        "rgb_path": str(Path(args.rgb).resolve()),
        "target_mask_path": str(Path(args.target_mask).resolve()),
        "other_mask_paths": [str(Path(path).resolve()) for path in args.other_mask],
        "original_height": int(shape[0]),
        "original_width": int(shape[1]),
        "target_bbox_xyxy": list(target_bbox),
        "crop_bbox_xyxy": list(crop_bbox),
        "remap_bbox_xyxy": remap_bbox,
        "expansion": float(args.expansion),
        "other_dilation_px": int(args.other_dilation),
        "model_input_path": str(model_input_path),
        "model_input_shape": list(model_input.shape[:2]),
    }
    write_json(output_dir / "prepare_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def remap_heatmap(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    local = np.asarray(np.load(args.local_heatmap), dtype=np.float32)
    if local.ndim == 3:
        local = local[..., 0]
    height = int(metadata["original_height"])
    width = int(metadata["original_width"])
    remap_bbox = metadata.get("remap_bbox_xyxy")
    if remap_bbox is None:
        if local.shape != (height, width):
            local = cv2.resize(local, (width, height), interpolation=cv2.INTER_LINEAR)
        full = local
    else:
        x1, y1, x2, y2 = [int(value) for value in remap_bbox]
        crop_height, crop_width = y2 - y1, x2 - x1
        if local.shape != (crop_height, crop_width):
            local = cv2.resize(local, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)
        full = np.zeros((height, width), dtype=np.float32)
        full[y1:y2, x1:x2] = local
    full[~np.isfinite(full)] = 0.0

    heatmap_path = output_dir / "affordance_heatmap.npy"
    np.save(heatmap_path, full.astype(np.float32))
    maximum = float(full.max())
    normalized = full / maximum if maximum > 0.0 else np.zeros_like(full)
    cv2.imwrite(str(output_dir / "affordance_heatmap.png"), np.clip(normalized * 255.0, 0, 255).astype(np.uint8))
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"), dtype=np.uint8)
    color = cv2.applyColorMap(np.clip(normalized * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    alpha = (0.55 * normalized)[..., None]
    overlay = np.clip(rgb * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    max_y, max_x = np.unravel_index(int(np.argmax(full)), full.shape)
    cv2.circle(overlay, (int(max_x), int(max_y)), 7, (255, 255, 255), -1)
    Image.fromarray(overlay).save(output_dir / "affordance_overlay.png")
    result = {
        **metadata,
        "local_heatmap_path": str(Path(args.local_heatmap).resolve()),
        "heatmap_path": str(heatmap_path),
        "raw_min": float(full.min()),
        "raw_max": maximum,
        "raw_mean": float(full.mean()),
        "argmax_uv": [int(max_x), int(max_y)],
    }
    write_json(output_dir / "affordance_metadata.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def desired_region(rgb: np.ndarray, target_mask: np.ndarray, target_type: str) -> tuple[np.ndarray, str]:
    x1, y1, x2, y2 = mask_bbox(target_mask)
    if target_type == "dropper":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        red = (((hsv[..., 0] <= 15) | (hsv[..., 0] >= 165)) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 45))
        red &= target_mask
        if int(red.sum()) >= 30:
            return red, "red_pixels_inside_target"
    top_limit = y1 + max(3, int(round((y2 - y1) * 0.32)))
    top = target_mask.copy()
    top[top_limit:, :] = False
    return top, "top_32_percent_of_target_bbox"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output).expanduser().resolve()
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"), dtype=np.uint8)
    target = load_mask(args.target_mask, rgb.shape[:2])
    other = np.zeros(rgb.shape[:2], dtype=bool)
    for path in args.other_mask:
        other |= load_mask(path, rgb.shape[:2])
    other &= ~target
    desired, desired_method = desired_region(rgb, target, args.target_type)
    heatmap = np.asarray(np.load(args.heatmap), dtype=np.float64)
    heatmap[~np.isfinite(heatmap)] = 0.0
    heatmap = np.maximum(heatmap, 0.0)
    total = float(heatmap.sum()) + 1e-12
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    desired_y, desired_x = np.where(desired)
    if len(desired_x):
        expected_uv = np.asarray([desired_x.mean(), desired_y.mean()])
        distance = float(np.linalg.norm(np.asarray([x, y]) - expected_uv))
    else:
        expected_uv = np.asarray([np.nan, np.nan])
        distance = float("inf")
    result = {
        "method": args.method,
        "target_type": args.target_type,
        "heatmap_path": str(Path(args.heatmap).resolve()),
        "argmax_uv": [int(x), int(y)],
        "argmax_inside_target": bool(target[y, x]),
        "argmax_inside_desired": bool(desired[y, x]),
        "desired_region_method": desired_method,
        "desired_region_center_uv": expected_uv.tolist(),
        "argmax_distance_to_desired_center_px": distance,
        "raw_max": float(heatmap.max()),
        "target_peak": float(heatmap[target].max()) if np.any(target) else 0.0,
        "other_peak": float(heatmap[other].max()) if np.any(other) else 0.0,
        "target_mass_fraction": float(heatmap[target].sum() / total),
        "desired_mass_fraction": float(heatmap[desired].sum() / total),
        "other_mass_fraction": float(heatmap[other].sum() / total),
        "target_to_other_peak_ratio": float(
            (heatmap[target].max() + 1e-12) / (heatmap[other].max() + 1e-12)
        ) if np.any(target) and np.any(other) else None,
    }
    result["valid"] = bool(
        result["argmax_inside_target"]
        and result["argmax_inside_desired"]
        and result["target_peak"] > result["other_peak"]
    )
    write_json(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_valid and not result["valid"]:
        raise RuntimeError(
            f"Affordance quality check failed for {args.target_type}: "
            f"argmax={result['argmax_uv']}, inside_target={result['argmax_inside_target']}, "
            f"inside_desired={result['argmax_inside_desired']}, "
            f"target_peak={result['target_peak']:.6g}, other_peak={result['other_peak']:.6g}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare object-isolated affordance inputs and remap heatmaps.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--method", choices=METHODS, required=True)
    prepare_parser.add_argument("--rgb", required=True)
    prepare_parser.add_argument("--target-mask", required=True)
    prepare_parser.add_argument("--other-mask", action="append", default=[])
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--expansion", type=float, default=0.35)
    prepare_parser.add_argument("--other-dilation", type=int, default=8)

    remap_parser = subparsers.add_parser("remap")
    remap_parser.add_argument("--metadata", required=True)
    remap_parser.add_argument("--local-heatmap", required=True)
    remap_parser.add_argument("--rgb", required=True)
    remap_parser.add_argument("--output-dir", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--method", required=True)
    evaluate_parser.add_argument("--target-type", choices=["dropper", "beaker", "jar"], required=True)
    evaluate_parser.add_argument("--rgb", required=True)
    evaluate_parser.add_argument("--target-mask", required=True)
    evaluate_parser.add_argument("--other-mask", action="append", default=[])
    evaluate_parser.add_argument("--heatmap", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--require-valid", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "remap":
        remap_heatmap(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
