from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a single YOLOv5-seg instance mask as PNG.")
    parser.add_argument("--repo", required=True, help="Path to the YOLOv5 repository.")
    parser.add_argument("--weights", required=True, help="YOLOv5 segmentation checkpoint path.")
    parser.add_argument("--data", default="", help="Dataset YAML with class names.")
    parser.add_argument("--rgb", required=True, help="Input RGB image path.")
    parser.add_argument("--output-dir", required=True, help="Directory for mask outputs.")
    parser.add_argument("--class-name", default=None, help="Target class name, e.g. 'beaker'.")
    parser.add_argument("--class-id", type=int, default=None, help="Target class id. Overrides --class-name.")
    parser.add_argument("--selection", choices=["best", "union"], default="best")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="")
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YOLO data YAML files.") from exc
    with Path(path).open("r", encoding="utf-8") as file_handle:
        loaded = yaml.safe_load(file_handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {path}")
    return loaded


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def resolve_class_id(class_name: str | None, class_id: int | None, names: dict[int, str]) -> int | None:
    if class_id is not None:
        return int(class_id)
    if not class_name:
        return None
    wanted = class_name.strip().lower().replace("_", " ")
    for index, name in names.items():
        normalized = name.strip().lower().replace("_", " ")
        if normalized == wanted:
            return index
    available = ", ".join(f"{idx}:{name}" for idx, name in sorted(names.items()))
    raise ValueError(f"Class name not found: {class_name}. Available classes: {available}")


def import_yolo(repo: str | Path) -> dict[str, Any]:
    repo_path = Path(repo).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.segment.general import process_mask, scale_image
    from utils.torch_utils import select_device, smart_inference_mode

    return {
        "DetectMultiBackend": DetectMultiBackend,
        "letterbox": letterbox,
        "check_img_size": check_img_size,
        "non_max_suppression": non_max_suppression,
        "scale_boxes": scale_boxes,
        "process_mask": process_mask,
        "scale_image": scale_image,
        "select_device": select_device,
        "smart_inference_mode": smart_inference_mode,
    }


def unpack_segmentation_output(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, (list, tuple)):
        if len(output) == 2 and isinstance(output[0], (list, tuple)) and len(output[0]) >= 2:
            return output[0][0], output[0][1]
        if len(output) >= 2 and torch.is_tensor(output[0]) and torch.is_tensor(output[1]):
            return output[0], output[1]
    raise TypeError("Unsupported model output format for YOLOv5 segmentation inference.")


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = np.array(
        [
            [255, 56, 56],
            [255, 157, 151],
            [255, 112, 31],
            [255, 178, 29],
            [207, 210, 49],
            [72, 249, 10],
            [146, 204, 23],
            [61, 219, 134],
            [26, 147, 52],
            [0, 212, 187],
            [44, 153, 168],
            [0, 194, 255],
            [52, 69, 147],
            [100, 115, 255],
            [0, 24, 236],
            [132, 56, 255],
            [82, 0, 133],
            [203, 56, 255],
            [255, 149, 200],
            [255, 55, 199],
        ],
        dtype=np.uint8,
    )
    color = palette[int(class_id) % len(palette)]
    return int(color[2]), int(color[1]), int(color[0])


def write_empty_outputs(output_dir: Path, image_bgr: np.ndarray, metadata: dict[str, Any]) -> None:
    zero = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.imwrite(str(output_dir / "mask.png"), zero)
    cv2.imwrite(str(output_dir / "mask_prob.png"), zero)
    cv2.imwrite(str(output_dir / "overlay.png"), image_bgr)
    with (output_dir / "detections.json").open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, ensure_ascii=False, indent=2)


def export_mask(args: argparse.Namespace) -> Path:
    yolo = import_yolo(args.repo)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_names = normalize_names(load_yaml(args.data).get("names", {})) if args.data else {}
    image_bgr = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read RGB image: {args.rgb}")

    device = yolo["select_device"](args.device)
    model = yolo["DetectMultiBackend"](args.weights, device=device, data=args.data or None)
    stride, model_names, pt = model.stride, normalize_names(model.names), model.pt
    names = data_names or model_names
    target_class_id = resolve_class_id(args.class_name, args.class_id, names)
    imgsz = yolo["check_img_size"](args.imgsz, s=stride)

    image = yolo["letterbox"](image_bgr, imgsz, stride=stride, auto=pt)[0]
    image = image.transpose((2, 0, 1))[::-1]
    image = np.ascontiguousarray(image)
    image_tensor = torch.from_numpy(image).to(model.device).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)

    output = model(image_tensor)
    predictions, protos = unpack_segmentation_output(output)
    classes = [target_class_id] if target_class_id is not None else None
    predictions = yolo["non_max_suppression"](
        predictions,
        args.conf_thres,
        args.iou_thres,
        classes=classes,
        max_det=args.max_det,
        nm=32,
    )
    detections = predictions[0]

    metadata: dict[str, Any] = {
        "rgb": str(args.rgb),
        "weights": str(args.weights),
        "data": str(args.data) if args.data else None,
        "target_class_id": target_class_id,
        "target_class_name": names.get(target_class_id) if target_class_id is not None else None,
        "selection": args.selection,
        "detections": [],
        "selected_indices": [],
        "mask_path": str(output_dir / "mask.png"),
        "mask_prob_path": str(output_dir / "mask_prob.png"),
        "overlay_path": str(output_dir / "overlay.png"),
    }
    if len(detections) == 0:
        write_empty_outputs(output_dir, image_bgr, metadata)
        return output_dir / "mask.png"

    raw_masks = yolo["process_mask"](protos[0].float(), detections[:, 6:], detections[:, :4], image_tensor.shape[2:], upsample=True)
    raw_masks = raw_masks.permute(1, 2, 0).float().cpu().numpy()
    scaled_masks = yolo["scale_image"](image_tensor.shape[2:], raw_masks, image_bgr.shape)
    if scaled_masks.ndim == 2:
        scaled_masks = scaled_masks[:, :, None]
    detections[:, :4] = yolo["scale_boxes"](image_tensor.shape[2:], detections[:, :4], image_bgr.shape).round()

    detection_rows = detections[:, :6].detach().cpu().numpy()
    for index, row in enumerate(detection_rows):
        x1, y1, x2, y2, confidence, class_value = row.tolist()
        class_index = int(class_value)
        metadata["detections"].append(
            {
                "index": index,
                "class_id": class_index,
                "class_name": names.get(class_index, str(class_index)),
                "confidence": float(confidence),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "area_px": int((scaled_masks[:, :, index] > args.mask_thresh).sum()),
            }
        )

    if args.selection == "best":
        selected = [int(np.argmax(detection_rows[:, 4]))]
    else:
        selected = list(range(len(detection_rows)))
    metadata["selected_indices"] = selected

    selected_probs = scaled_masks[:, :, selected]
    if selected_probs.ndim == 2:
        selected_probs = selected_probs[:, :, None]
    mask_prob = selected_probs.max(axis=2).astype(np.float32)
    mask_binary = (mask_prob > float(args.mask_thresh)).astype(np.uint8) * 255

    overlay = image_bgr.copy()
    color_mask = np.zeros_like(overlay)
    for index in selected:
        class_id_for_color = int(detection_rows[index, 5])
        color_mask[scaled_masks[:, :, index] > args.mask_thresh] = color_for_class(class_id_for_color)
    overlay = np.where(mask_binary[..., None] > 0, cv2.addWeighted(overlay, 0.65, color_mask, 0.35, 0.0), overlay)
    for index, row in enumerate(detection_rows):
        x1, y1, x2, y2, confidence, class_value = row.tolist()
        class_index = int(class_value)
        color = color_for_class(class_index)
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{names.get(class_index, class_index)} {confidence:.2f}"
        cv2.putText(overlay, label, (int(x1), max(0, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(str(output_dir / "mask.png"), mask_binary)
    cv2.imwrite(str(output_dir / "mask_prob.png"), np.clip(mask_prob * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "overlay.png"), overlay)
    with (output_dir / "detections.json").open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, ensure_ascii=False, indent=2)
    return output_dir / "mask.png"


def main() -> None:
    args = parse_args()
    mask_path = export_mask(args)
    print(mask_path)


if __name__ == "__main__":
    main()
