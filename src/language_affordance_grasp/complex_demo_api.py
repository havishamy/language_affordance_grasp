from __future__ import annotations

import hmac
import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from language_affordance_grasp.complex_demo_export import (
    ContractError,
    SUPPORTED_AFFORDANCES,
    SUPPORTED_OBJECTS,
    validate_inputs,
    validate_request_shape,
    write_json,
)
from language_affordance_grasp.titration_api import (
    ApiSettings as TitrationApiSettings,
    build_bundle as build_titration_bundle,
    save_upload,
    tail_text,
    utc_now,
)


Workflow = Literal["titration", "complex_demo"]


@dataclass(frozen=True)
class ApiSettings:
    titration_run_root: Path
    complex_run_root: Path
    titration_pipeline_script: Path
    complex_pipeline_script: Path
    calibration_path: Path
    api_token: str
    max_image_bytes: int = 64 * 1024 * 1024
    max_json_bytes: int = 2 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "ApiSettings":
        repo = Path(os.environ.get("LAG_REPO", "/home/dsj/code/language_affordance_grasp"))
        calibration = os.environ.get("LAG_EYE_IN_HAND_CALIBRATION", "")
        return cls(
            titration_run_root=Path(
                os.environ.get("LAG_TITRATION_RUN_ROOT", os.environ.get("LAG_API_RUN_ROOT", str(repo / "runs" / "api_jobs")))
            ),
            complex_run_root=Path(os.environ.get("LAG_COMPLEX_DEMO_RUN_ROOT", str(repo / "runs" / "complex_demo_jobs"))),
            titration_pipeline_script=Path(
                os.environ.get("LAG_TITRATION_PIPELINE", str(repo / "scripts" / "run_titration_two_object.sh"))
            ),
            complex_pipeline_script=Path(
                os.environ.get("LAG_COMPLEX_DEMO_PIPELINE", str(repo / "scripts" / "run_complex_demo_three_object.sh"))
            ),
            calibration_path=Path(calibration) if calibration else Path(""),
            api_token=os.environ.get("LAG_API_TOKEN", ""),
            max_image_bytes=int(os.environ.get("LAG_API_MAX_IMAGE_BYTES", str(64 * 1024 * 1024))),
            max_json_bytes=int(os.environ.get("LAG_API_MAX_JSON_BYTES", str(2 * 1024 * 1024))),
        )

    def validate(self) -> None:
        TitrationApiSettings(
            run_root=self.titration_run_root,
            pipeline_script=self.titration_pipeline_script,
            calibration_path=self.calibration_path,
            api_token=self.api_token,
            max_image_bytes=self.max_image_bytes,
            max_json_bytes=self.max_json_bytes,
        ).validate()
        if not self.complex_pipeline_script.is_file():
            raise RuntimeError(f"Complex-demo pipeline script not found: {self.complex_pipeline_script}")
        if self.titration_run_root.resolve() == self.complex_run_root.resolve():
            raise RuntimeError("Titration and complex-demo run roots must be different")


class PipelineJobError(RuntimeError):
    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(str(error.get("message", "Pipeline job failed")))
        self.error = error


class UnifiedJobManager:
    """One worker queue for both endpoint families, so GPU jobs never overlap."""

    def __init__(self, settings: ApiSettings):
        self.settings = settings
        self._roots = {
            "titration": settings.titration_run_root,
            "complex_demo": settings.complex_run_root,
        }
        for root in self._roots.values():
            root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[Workflow, dict[str, dict[str, Any]]] = {"titration": {}, "complex_demo": {}}
        self._lock = threading.RLock()
        self._queue: queue.Queue[tuple[Workflow, str]] = queue.Queue()
        self._load_existing_jobs()
        self._worker = threading.Thread(target=self._worker_loop, name="unified-gpu-worker", daemon=True)
        self._worker.start()

    def job_dir(self, workflow: Workflow, job_id: str) -> Path:
        try:
            normalized = str(uuid.UUID(job_id))
        except ValueError as exc:
            raise KeyError(job_id) from exc
        return self._roots[workflow] / normalized

    def _status_path(self, workflow: Workflow, job_id: str) -> Path:
        return self.job_dir(workflow, job_id) / "status.json"

    def _load_existing_jobs(self) -> None:
        for workflow, root in self._roots.items():
            for status_path in root.glob("*/status.json"):
                try:
                    payload = json.loads(status_path.read_text(encoding="utf-8"))
                    job_id = str(uuid.UUID(str(payload["job_id"])))
                except Exception:
                    continue
                if payload.get("state") in {"queued", "running", "uploading"}:
                    payload.update(
                        state="interrupted",
                        updated_at=utc_now(),
                        error={
                            "stage": "queue",
                            "object": None,
                            "region": None,
                            "code": "SERVER_RESTARTED",
                            "message": "API server restarted while this job was pending or running; resubmit the capture.",
                        }
                        if workflow == "complex_demo"
                        else "API server restarted while this job was pending or running; resubmit the capture.",
                    )
                    write_json(status_path, payload)
                self._jobs[workflow][job_id] = payload

    def create_job(self, workflow: Workflow, metadata: dict[str, Any]) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        job_dir = self.job_dir(workflow, job_id)
        (job_dir / "inputs").mkdir(parents=True)
        payload = {
            "job_id": job_id,
            "workflow": workflow,
            "state": "uploading",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "metadata": metadata,
            "error": None,
            "result": None,
        }
        with self._lock:
            self._jobs[workflow][job_id] = payload
            self._persist(workflow, payload)
        return json.loads(json.dumps(payload))

    def enqueue(self, workflow: Workflow, job_id: str) -> dict[str, Any]:
        payload = self.update(workflow, job_id, state="queued")
        self._queue.put((workflow, job_id))
        return payload

    def get(self, workflow: Workflow, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs[workflow]:
                raise KeyError(job_id)
            return json.loads(json.dumps(self._jobs[workflow][job_id]))

    def update(self, workflow: Workflow, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs[workflow]:
                raise KeyError(job_id)
            self._jobs[workflow][job_id].update(changes)
            self._jobs[workflow][job_id]["updated_at"] = utc_now()
            self._persist(workflow, self._jobs[workflow][job_id])
            return json.loads(json.dumps(self._jobs[workflow][job_id]))

    def remove_failed_upload(self, workflow: Workflow, job_id: str) -> None:
        shutil.rmtree(self.job_dir(workflow, job_id), ignore_errors=True)
        with self._lock:
            self._jobs[workflow].pop(job_id, None)

    def _persist(self, workflow: Workflow, payload: dict[str, Any]) -> None:
        write_json(self._status_path(workflow, payload["job_id"]), payload)

    def _worker_loop(self) -> None:
        while True:
            workflow, job_id = self._queue.get()
            try:
                if workflow == "titration":
                    self._run_titration(job_id)
                else:
                    self._run_complex_demo(job_id)
            except PipelineJobError as exc:
                self.update(workflow, job_id, state="failed", error=exc.error, failed_at=utc_now())
            except Exception as exc:
                if workflow == "complex_demo":
                    error: Any = {
                        "stage": "api_worker",
                        "object": None,
                        "region": None,
                        "code": "UNEXPECTED_WORKER_ERROR",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                else:
                    error = f"{type(exc).__name__}: {exc}"
                self.update(workflow, job_id, state="failed", error=error, failed_at=utc_now())
            finally:
                self._queue.task_done()

    def _run_subprocess(self, workflow: Workflow, job_id: str, command: list[str]) -> None:
        job_dir = self.job_dir(workflow, job_id)
        self.update(workflow, job_id, state="running", command=command, started_at=utc_now())
        worker_log = job_dir / "worker.log"
        with worker_log.open("wb") as log_handle:
            process = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        if process.returncode == 0:
            return
        if workflow == "complex_demo":
            failure_path = job_dir / "output" / "failure.json"
            if failure_path.is_file():
                try:
                    error = json.loads(failure_path.read_text(encoding="utf-8"))
                except Exception:
                    error = None
            else:
                error = None
            if not isinstance(error, dict):
                error = {
                    "stage": "pipeline",
                    "object": None,
                    "region": None,
                    "code": "PIPELINE_EXIT_NONZERO",
                    "message": f"Pipeline exited with code {process.returncode}: {tail_text(worker_log, 40)}",
                }
            raise PipelineJobError(error)
        raise RuntimeError(f"pipeline exited with code {process.returncode}\n{tail_text(worker_log, 80)}")

    def _run_titration(self, job_id: str) -> None:
        job = self.get("titration", job_id)
        job_dir = self.job_dir("titration", job_id)
        input_dir = job_dir / "inputs"
        output_dir = job_dir / "output"
        metadata = job["metadata"]
        command = [
            str(self.settings.titration_pipeline_script),
            "--rgb", str(input_dir / "rgb.png"),
            "--depth", str(input_dir / "depth.png"),
            "--intrinsics", str(input_dir / "camera.json"),
            "--calibration", str(self.settings.calibration_path),
            "--world-ee", str(input_dir / "T_world_ee.json"),
            "--depth-scale", str(metadata["depth_scale"]),
            "--output-dir", str(output_dir),
        ]
        if metadata.get("gpu") is not None:
            command.extend(["--gpu", str(metadata["gpu"])])
        self._run_subprocess("titration", job_id, command)
        targets_path = output_dir / "04_targets" / "titration_targets.json"
        if not targets_path.is_file():
            raise FileNotFoundError(f"Pipeline completed without target output: {targets_path}")
        targets = json.loads(targets_path.read_text(encoding="utf-8"))
        result = {
            "units": "meters",
            "frame": "world",
            "dropper_bulb_center_world": targets["dropper_bulb"]["center_world"],
            "beaker_mouth_center_world": targets["beaker_mouth"]["center_world"],
            "dropper_bulb_center_pixel_uv": targets["dropper_bulb"].get("center_pixel_uv"),
            "beaker_mouth_center_pixel_uv": targets["beaker_mouth"].get("center_pixel_uv"),
            "targets_json": "output/04_targets/titration_targets.json",
            "overlay": "output/04_targets/target_centers_overlay.png",
        }
        self.update("titration", job_id, state="completed", result=result, completed_at=utc_now(), error=None)

    def _run_complex_demo(self, job_id: str) -> None:
        job = self.get("complex_demo", job_id)
        job_dir = self.job_dir("complex_demo", job_id)
        input_dir = job_dir / "inputs"
        output_dir = job_dir / "output"
        metadata = job["metadata"]
        command = [
            str(self.settings.complex_pipeline_script),
            "--rgb", str(input_dir / "rgb.png"),
            "--depth", str(input_dir / "depth.png"),
            "--intrinsics", str(input_dir / "camera.json"),
            "--calibration", str(self.settings.calibration_path),
            "--world-ee", str(input_dir / "T_world_ee.json"),
            "--depth-scale", str(metadata["depth_scale"]),
            "--capture-id", metadata["capture_id"],
            "--job-id", job_id,
            "--requested-objects-json", json.dumps(metadata["requested_objects"], separators=(",", ":")),
            "--requested-affordance-regions-json", json.dumps(
                metadata["requested_affordance_regions"], separators=(",", ":")
            ),
            "--output-dir", str(output_dir),
        ]
        if metadata.get("gpu") is not None:
            command.extend(["--gpu", str(metadata["gpu"])])
        self._run_subprocess("complex_demo", job_id, command)
        result_path = output_dir / "result.json"
        if not result_path.is_file():
            raise PipelineJobError(
                {
                    "stage": "result_export",
                    "object": None,
                    "region": None,
                    "code": "MISSING_RESULT_JSON",
                    "message": f"Pipeline completed without {result_path}",
                }
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_complex_result(output_dir, result, job_id, metadata)
        completed_at = utc_now()
        job = self.get("complex_demo", job_id)
        job_document = {
            **job,
            "state": "completed",
            "updated_at": completed_at,
            "completed_at": completed_at,
            "error": None,
            "result": {"result_json": "output/result.json"},
        }
        write_json(output_dir / "job.json", job_document)
        build_complex_bundle(job_dir, "results")
        self.update(
            "complex_demo",
            job_id,
            state="completed",
            result={"result_json": "output/result.json"},
            completed_at=completed_at,
            error=None,
        )


COMPLEX_REQUIRED_ARTIFACTS = (
    "result.json",
    "job.json",
    "pipeline.log",
    "models/wide_mouth_bottle_world.ply",
    "models/beaker_world.ply",
    "models/dropper_world.ply",
    "models/wide_mouth_bottle_canonical.ply",
    "models/beaker_canonical.ply",
    "models/dropper_canonical.ply",
    "models/wide_mouth_bottle_camera_mesh.ply",
    "models/beaker_camera_mesh.ply",
    "models/dropper_camera_mesh.ply",
    "models/wide_mouth_bottle_world_mesh.ply",
    "models/beaker_world_mesh.ply",
    "models/dropper_world_mesh.ply",
    "models/wide_mouth_bottle_world_mesh.glb",
    "models/beaker_world_mesh.glb",
    "models/dropper_world_mesh.glb",
    "models/three_object_scene_world.ply",
    "models/three_object_scene_world.glb",
    "00_segmentation/depth_clouds/depth_clouds.json",
    "00_segmentation/depth_clouds/wide_mouth_bottle_depth_camera.ply",
    "00_segmentation/depth_clouds/wide_mouth_bottle_depth_world.ply",
    "00_segmentation/depth_clouds/beaker_depth_camera.ply",
    "00_segmentation/depth_clouds/beaker_depth_world.ply",
    "00_segmentation/depth_clouds/dropper_depth_camera.ply",
    "00_segmentation/depth_clouds/dropper_depth_world.ply",
    "affordance_regions/wide_mouth_bottle/stopper_world.ply",
    "affordance_regions/wide_mouth_bottle/stopper_world.npz",
    "affordance_regions/wide_mouth_bottle/body_grasp_world.ply",
    "affordance_regions/wide_mouth_bottle/body_grasp_world.npz",
    "affordance_regions/beaker/mouth_world.ply",
    "affordance_regions/beaker/mouth_world.npz",
    "affordance_regions/dropper/bulb_world.ply",
    "affordance_regions/dropper/bulb_world.npz",
    "overlays/segmentation_overlay.png",
    "overlays/affordance_overlay.png",
)


def _safe_output_artifact(output_dir: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe artifact path: {relative}")
    path = (output_dir / relative_path).resolve()
    if output_dir.resolve() not in path.parents:
        raise ValueError(f"Artifact escapes output directory: {relative}")
    return path


def validate_complex_result(
    output_dir: Path,
    result: dict[str, Any],
    job_id: str,
    metadata: dict[str, Any],
) -> None:
    if result.get("job_id") != job_id or result.get("capture_id") != metadata["capture_id"]:
        raise ValueError("result.json job_id/capture_id does not match the queued job")
    if result.get("frame") != "world" or result.get("units") != "meters":
        raise ValueError("result.json must use world coordinates in meters")
    if result.get("requested_objects") != SUPPORTED_OBJECTS:
        raise ValueError("result.json requested_objects does not match the contract")
    objects = result.get("objects")
    if not isinstance(objects, dict) or list(objects) != SUPPORTED_OBJECTS:
        raise ValueError("result.json objects are missing or out of contract order")
    for object_name in SUPPORTED_OBJECTS:
        item = objects[object_name]
        transform = item.get("T_world_object")
        scale = item.get("scale")
        if not item.get("model_id") or not isinstance(transform, list) or not isinstance(scale, list):
            raise ValueError(f"Incomplete model metadata for {object_name}")
        artifact = item.get("model_points_world_artifact")
        if not isinstance(artifact, str) or not _safe_output_artifact(output_dir, artifact).is_file():
            raise ValueError(f"Missing model artifact for {object_name}: {artifact}")
        for artifact_key in (
            "model_canonical_artifact",
            "model_mesh_camera_artifact",
            "model_mesh_world_artifact",
            "model_mesh_world_glb_artifact",
        ):
            artifact = item.get(artifact_key)
            if not isinstance(artifact, str) or not _safe_output_artifact(output_dir, artifact).is_file():
                raise ValueError(f"Missing {artifact_key} for {object_name}: {artifact}")
        for artifact_key in ("observed_depth_camera_artifact", "observed_depth_world_artifact"):
            artifact = item.get(artifact_key)
            if not isinstance(artifact, str) or not _safe_output_artifact(output_dir, artifact).is_file():
                raise ValueError(f"Missing {artifact_key} for {object_name}: {artifact}")
        expected_regions = metadata["requested_affordance_regions"].get(object_name, [])
        actual_regions = item.get("affordance_regions", {})
        if list(actual_regions) != expected_regions:
            raise ValueError(f"Affordance regions for {object_name} do not match request")
        for region_name in expected_regions:
            points = actual_regions[region_name].get("points_world")
            if not isinstance(points, list) or not points:
                raise ValueError(f"Empty world affordance points for {object_name}/{region_name}")
            artifact = actual_regions[region_name].get("points_world_artifact")
            if not isinstance(artifact, str) or not _safe_output_artifact(output_dir, artifact).is_file():
                raise ValueError(f"Missing region artifact for {object_name}/{region_name}")
            npz_artifact = actual_regions[region_name].get("points_world_npz_artifact")
            if not isinstance(npz_artifact, str) or not _safe_output_artifact(output_dir, npz_artifact).is_file():
                raise ValueError(f"Missing region NPZ artifact for {object_name}/{region_name}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _best_alignment_artifacts(output_dir: Path) -> set[Path]:
    selected: set[Path] = set()
    for pose_path in (output_dir / "02_sam3d").glob("*/pose_result.json"):
        selected_for_object = False
        try:
            payload = json.loads(pose_path.read_text(encoding="utf-8"))
            value = payload.get("alignment_best_step_ply")
            if value:
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = pose_path.parent / candidate
                candidate = candidate.resolve()
                candidate.relative_to(output_dir.resolve())
                if candidate.is_file():
                    selected.add(candidate)
                    selected_for_object = True
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        if not selected_for_object:
            available_steps = sorted((pose_path.parent / "alignment_steps").glob("alignment_step_*.ply"))
            if available_steps:
                selected.add(available_steps[-1].resolve())
    return selected


def _select_auditable_result_files(output_dir: Path) -> list[tuple[Path, str]]:
    best_alignments = _best_alignment_artifacts(output_dir)
    selected: list[tuple[Path, str]] = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir)
        if relative.parts[0] in {".config", ".matplotlib"}:
            continue
        if relative.as_posix() == "bundle_manifest.json":
            continue
        if "alignment_steps" in relative.parts and path.resolve() not in best_alignments:
            continue
        selected.append((path, relative.as_posix()))
    return selected


def build_complex_bundle(job_dir: Path, scope: str) -> Path:
    if scope not in {"results", "all"}:
        raise HTTPException(status_code=400, detail="scope must be 'results' or 'all'")
    output_dir = job_dir / "output"
    bundle_path = job_dir / f"{scope}_bundle.zip"
    legacy_bundle_available = False
    if bundle_path.is_file():
        try:
            with zipfile.ZipFile(bundle_path) as archive:
                if scope == "all" or "bundle_manifest.json" in archive.namelist():
                    return bundle_path
                legacy_bundle_available = True
        except zipfile.BadZipFile:
            pass
    selected: list[tuple[Path, str]] = []
    if scope == "results":
        for relative in COMPLEX_REQUIRED_ARTIFACTS:
            path = _safe_output_artifact(output_dir, relative)
            if not path.is_file():
                if legacy_bundle_available:
                    return bundle_path
                raise FileNotFoundError(f"Required result artifact is missing: {relative}")
        selected = _select_auditable_result_files(output_dir)
        manifest = {
            "scope": "results",
            "generated_at": utc_now(),
            "policy": {
                "included": "All useful pipeline outputs, including lightweight optimization overlays",
                "excluded": [
                    ".config/**",
                    ".matplotlib/**",
                    "02_sam3d/*/alignment_steps/* except the recorded best PLY (or latest available fallback)",
                ],
                "full_unfiltered_bundle": "Use scope=all",
            },
            "file_count_excluding_manifest": len(selected),
            "files": [
                {
                    "path": archive_name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path, archive_name in sorted(selected, key=lambda pair: pair[1])
            ],
        }
        manifest_path = output_dir / "bundle_manifest.json"
        write_json(manifest_path, manifest)
        selected.append((manifest_path, "bundle_manifest.json"))
    else:
        for path in output_dir.rglob("*"):
            if path.is_file():
                selected.append((path, path.relative_to(output_dir).as_posix()))
        for path in (job_dir / "inputs").rglob("*"):
            if path.is_file():
                selected.append((path, (Path("inputs") / path.relative_to(job_dir / "inputs")).as_posix()))
        status_path = job_dir / "status.json"
        if status_path.is_file():
            selected.append((status_path, "status.json"))
        worker_log = job_dir / "worker.log"
        if worker_log.is_file():
            selected.append((worker_log, "worker.log"))
    temporary = bundle_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, archive_name in sorted(selected, key=lambda pair: pair[1]):
            archive.write(path, arcname=archive_name)
    temporary.replace(bundle_path)
    return bundle_path


def _decode_request_json(value: str, label: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{label} must be valid JSON: {exc}") from exc


def _validate_common_form(depth_scale: float, gpu: int | None) -> None:
    if not (0.0 < depth_scale <= 1.0):
        raise HTTPException(status_code=400, detail="depth_scale must be in (0, 1]")
    if gpu is not None and not (0 <= gpu <= 63):
        raise HTTPException(status_code=400, detail="gpu must be a non-negative physical GPU index")


async def _save_four_uploads(
    settings: ApiSettings,
    manager: UnifiedJobManager,
    workflow: Workflow,
    job_id: str,
    rgb: UploadFile,
    depth: UploadFile,
    camera: UploadFile,
    world_ee: UploadFile,
) -> dict[str, int]:
    input_dir = manager.job_dir(workflow, job_id) / "inputs"
    return {
        "rgb": await save_upload(rgb, input_dir / "rgb.png", settings.max_image_bytes),
        "depth": await save_upload(depth, input_dir / "depth.png", settings.max_image_bytes),
        "camera": await save_upload(camera, input_dir / "camera.json", settings.max_json_bytes),
        "world_ee": await save_upload(world_ee, input_dir / "T_world_ee.json", settings.max_json_bytes),
    }


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings.from_env()
    app = FastAPI(title="Language Affordance Unified Titration and Complex Demo API", version="2.0")
    manager_holder: dict[str, UnifiedJobManager] = {}

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {settings.api_token}"
        if not settings.api_token or authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    async def manager() -> UnifiedJobManager:
        return manager_holder["manager"]

    @app.on_event("startup")
    async def startup() -> None:
        settings.validate()
        manager_holder["manager"] = UnifiedJobManager(settings)

    @app.get("/health")
    async def health(_: None = Depends(authorize)) -> dict[str, Any]:
        return {
            "ok": True,
            "time": utc_now(),
            "calibration_path": str(settings.calibration_path),
            "titration_run_root": str(settings.titration_run_root),
            "complex_demo_run_root": str(settings.complex_run_root),
            "worker_count": 1,
            "queue": "shared_by_titration_and_complex_demo",
        }

    @app.post("/v1/titration/jobs", status_code=202)
    async def submit_titration_job(
        rgb: UploadFile = File(...),
        depth: UploadFile = File(...),
        camera: UploadFile = File(...),
        world_ee: UploadFile = File(...),
        depth_scale: float = Form(0.001),
        gpu: int | None = Form(None),
        capture_id: str | None = Form(None),
        _: None = Depends(authorize),
        jobs: UnifiedJobManager = Depends(manager),
    ) -> dict[str, Any]:
        _validate_common_form(depth_scale, gpu)
        job = jobs.create_job("titration", {"depth_scale": depth_scale, "gpu": gpu, "capture_id": capture_id})
        job_id = job["job_id"]
        try:
            sizes = await _save_four_uploads(settings, jobs, "titration", job_id, rgb, depth, camera, world_ee)
        except Exception:
            jobs.remove_failed_upload("titration", job_id)
            raise
        jobs.update("titration", job_id, upload_sizes=sizes)
        return jobs.enqueue("titration", job_id)

    @app.get("/v1/titration/jobs/{job_id}")
    async def get_titration_job(
        job_id: str, _: None = Depends(authorize), jobs: UnifiedJobManager = Depends(manager)
    ) -> dict[str, Any]:
        try:
            return jobs.get("titration", job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.get("/v1/titration/jobs/{job_id}/result")
    async def get_titration_result(
        job_id: str, _: None = Depends(authorize), jobs: UnifiedJobManager = Depends(manager)
    ) -> dict[str, Any]:
        try:
            job = jobs.get("titration", job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        if job["state"] != "completed":
            raise HTTPException(status_code=409, detail=f"Job state is {job['state']}")
        return {"job_id": job_id, **job["result"]}

    @app.get("/v1/titration/jobs/{job_id}/bundle")
    async def download_titration_bundle(
        job_id: str,
        scope: str = "results",
        _: None = Depends(authorize),
        jobs: UnifiedJobManager = Depends(manager),
    ) -> FileResponse:
        try:
            job = jobs.get("titration", job_id)
            job_dir = jobs.job_dir("titration", job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        if job["state"] != "completed":
            raise HTTPException(status_code=409, detail=f"Job state is {job['state']}")
        bundle = build_titration_bundle(job_dir, scope)
        return FileResponse(bundle, media_type="application/zip", filename=f"titration_{job_id}_{scope}.zip")

    @app.post("/v1/complex-demo/jobs", status_code=202)
    async def submit_complex_job(
        rgb: UploadFile = File(...),
        depth: UploadFile = File(...),
        camera: UploadFile = File(...),
        world_ee: UploadFile = File(...),
        depth_scale: float = Form(0.001),
        capture_id: str = Form(...),
        requested_objects: str = Form('["wide_mouth_bottle","beaker","dropper"]'),
        requested_affordance_regions: str = Form(
            '{"wide_mouth_bottle":["stopper","body_grasp"],"beaker":["mouth"],"dropper":["bulb"]}'
        ),
        gpu: int | None = Form(None),
        _: None = Depends(authorize),
        jobs: UnifiedJobManager = Depends(manager),
    ) -> dict[str, Any]:
        _validate_common_form(depth_scale, gpu)
        objects = _decode_request_json(requested_objects, "requested_objects")
        regions = _decode_request_json(requested_affordance_regions, "requested_affordance_regions")
        try:
            validate_request_shape(objects, regions)
        except ContractError as exc:
            raise HTTPException(status_code=400, detail=exc.payload) from exc
        metadata = {
            "capture_id": capture_id,
            "depth_scale": depth_scale,
            "requested_objects": objects,
            "requested_affordance_regions": regions,
            "gpu": gpu,
        }
        job = jobs.create_job("complex_demo", metadata)
        job_id = job["job_id"]
        try:
            sizes = await _save_four_uploads(settings, jobs, "complex_demo", job_id, rgb, depth, camera, world_ee)
            input_dir = jobs.job_dir("complex_demo", job_id) / "inputs"
            validate_inputs(
                SimpleNamespace(
                    rgb=str(input_dir / "rgb.png"),
                    depth=str(input_dir / "depth.png"),
                    intrinsics=str(input_dir / "camera.json"),
                    world_ee=str(input_dir / "T_world_ee.json"),
                    depth_scale=depth_scale,
                    capture_id=capture_id,
                    requested_objects_json=requested_objects,
                    requested_affordance_regions_json=requested_affordance_regions,
                    output=None,
                )
            )
        except ContractError as exc:
            jobs.remove_failed_upload("complex_demo", job_id)
            raise HTTPException(status_code=400, detail=exc.payload) from exc
        except Exception:
            jobs.remove_failed_upload("complex_demo", job_id)
            raise
        jobs.update("complex_demo", job_id, upload_sizes=sizes)
        queued = jobs.enqueue("complex_demo", job_id)
        return {"job_id": queued["job_id"], "state": queued["state"]}

    @app.get("/v1/complex-demo/jobs/{job_id}")
    async def get_complex_job(
        job_id: str, _: None = Depends(authorize), jobs: UnifiedJobManager = Depends(manager)
    ) -> dict[str, Any]:
        try:
            return jobs.get("complex_demo", job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.get("/v1/complex-demo/jobs/{job_id}/result")
    async def get_complex_result(
        job_id: str, _: None = Depends(authorize), jobs: UnifiedJobManager = Depends(manager)
    ) -> dict[str, Any]:
        try:
            job = jobs.get("complex_demo", job_id)
            result_path = jobs.job_dir("complex_demo", job_id) / "output" / "result.json"
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        if job["state"] != "completed":
            raise HTTPException(status_code=409, detail=f"Job state is {job['state']}")
        if not result_path.is_file():
            raise HTTPException(status_code=500, detail="Completed job is missing result.json")
        return json.loads(result_path.read_text(encoding="utf-8"))

    @app.get("/v1/complex-demo/jobs/{job_id}/bundle")
    async def download_complex_bundle(
        job_id: str,
        scope: str = "results",
        _: None = Depends(authorize),
        jobs: UnifiedJobManager = Depends(manager),
    ) -> FileResponse:
        try:
            job = jobs.get("complex_demo", job_id)
            job_dir = jobs.job_dir("complex_demo", job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        if job["state"] != "completed":
            raise HTTPException(status_code=409, detail=f"Job state is {job['state']}")
        bundle = build_complex_bundle(job_dir, scope)
        return FileResponse(bundle, media_type="application/zip", filename=f"complex_demo_{job_id}_{scope}.zip")

    return app


app = create_app()
