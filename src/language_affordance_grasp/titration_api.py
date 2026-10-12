from __future__ import annotations

import hmac
import json
import os
import queue
import shutil
import subprocess
import threading
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ApiSettings:
    run_root: Path
    pipeline_script: Path
    calibration_path: Path
    api_token: str
    max_image_bytes: int = 64 * 1024 * 1024
    max_json_bytes: int = 2 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "ApiSettings":
        repo = Path(os.environ.get("LAG_REPO", "/home/dsj/code/language_affordance_grasp"))
        calibration = os.environ.get("LAG_EYE_IN_HAND_CALIBRATION", "")
        return cls(
            run_root=Path(os.environ.get("LAG_API_RUN_ROOT", str(repo / "runs" / "api_jobs"))),
            pipeline_script=Path(
                os.environ.get("LAG_TITRATION_PIPELINE", str(repo / "scripts" / "run_titration_two_object.sh"))
            ),
            calibration_path=Path(calibration) if calibration else Path(""),
            api_token=os.environ.get("LAG_API_TOKEN", ""),
            max_image_bytes=int(os.environ.get("LAG_API_MAX_IMAGE_BYTES", str(64 * 1024 * 1024))),
            max_json_bytes=int(os.environ.get("LAG_API_MAX_JSON_BYTES", str(2 * 1024 * 1024))),
        )

    def validate(self) -> None:
        if not self.api_token:
            raise RuntimeError("LAG_API_TOKEN must be set before starting the API server")
        if not self.calibration_path.is_file():
            raise RuntimeError(
                "LAG_EYE_IN_HAND_CALIBRATION must point to a valid T_ee_cam calibration JSON/NPY/TXT file"
            )
        if self.calibration_path.suffix.lower() == ".json":
            try:
                calibration_payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Unable to parse calibration JSON: {self.calibration_path}") from exc
            status = str(calibration_payload.get("status", "")).lower() if isinstance(calibration_payload, dict) else ""
            if "pending" in status and os.environ.get("LAG_ALLOW_UNCONFIRMED_CALIBRATION") != "1":
                raise RuntimeError(
                    "Calibration direction is marked pending. Confirm T_ee_cam direction and change status to 'confirmed'."
                )
        if not self.pipeline_script.is_file():
            raise RuntimeError(f"Pipeline script not found: {self.pipeline_script}")


class JobManager:
    def __init__(self, settings: ApiSettings):
        self.settings = settings
        self.settings.run_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._load_existing_jobs()
        self._worker = threading.Thread(target=self._worker_loop, name="titration-worker", daemon=True)
        self._worker.start()

    def _job_dir(self, job_id: str) -> Path:
        try:
            normalized = str(uuid.UUID(job_id))
        except ValueError as exc:
            raise KeyError(job_id) from exc
        return self.settings.run_root / normalized

    def _status_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "status.json"

    def _load_existing_jobs(self) -> None:
        for status_path in self.settings.run_root.glob("*/status.json"):
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                job_id = str(uuid.UUID(str(payload["job_id"])))
            except Exception:
                continue
            if payload.get("state") in {"queued", "running"}:
                payload.update(
                    state="interrupted",
                    updated_at=utc_now(),
                    error="API server restarted while this job was pending or running; resubmit the capture.",
                )
                status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._jobs[job_id] = payload

    def create_job(self, metadata: dict[str, Any]) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        job_dir = self._job_dir(job_id)
        (job_dir / "inputs").mkdir(parents=True)
        payload = {
            "job_id": job_id,
            "state": "uploading",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "metadata": metadata,
            "error": None,
            "result": None,
        }
        with self._lock:
            self._jobs[job_id] = payload
            self._persist(payload)
        return payload.copy()

    def enqueue(self, job_id: str) -> dict[str, Any]:
        payload = self.update(job_id, state="queued")
        self._queue.put(job_id)
        return payload

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return json.loads(json.dumps(self._jobs[job_id]))

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            self._jobs[job_id].update(changes)
            self._jobs[job_id]["updated_at"] = utc_now()
            self._persist(self._jobs[job_id])
            return json.loads(json.dumps(self._jobs[job_id]))

    def _persist(self, payload: dict[str, Any]) -> None:
        status_path = self._status_path(payload["job_id"])
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            except Exception as exc:
                self.update(job_id, state="failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        job = self.get(job_id)
        job_dir = self._job_dir(job_id)
        input_dir = job_dir / "inputs"
        output_dir = job_dir / "output"
        metadata = job["metadata"]
        command = [
            str(self.settings.pipeline_script),
            "--rgb",
            str(input_dir / "rgb.png"),
            "--depth",
            str(input_dir / "depth.png"),
            "--intrinsics",
            str(input_dir / "camera.json"),
            "--calibration",
            str(self.settings.calibration_path),
            "--world-ee",
            str(input_dir / "T_world_ee.json"),
            "--depth-scale",
            str(metadata["depth_scale"]),
            "--output-dir",
            str(output_dir),
        ]
        if metadata.get("gpu") is not None:
            command.extend(["--gpu", str(metadata["gpu"])])
        self.update(job_id, state="running", command=command, started_at=utc_now())
        worker_log = job_dir / "worker.log"
        with worker_log.open("wb") as log_handle:
            process = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        if process.returncode != 0:
            tail = tail_text(worker_log, 80)
            raise RuntimeError(f"pipeline exited with code {process.returncode}\n{tail}")

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
        self.update(job_id, state="completed", result=result, completed_at=utc_now(), error=None)


def tail_text(path: Path, line_count: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])


async def save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    written = 0
    with destination.open("wb") as output:
        while True:
            block = await upload.read(1024 * 1024)
            if not block:
                break
            written += len(block)
            if written > max_bytes:
                raise HTTPException(status_code=413, detail=f"Upload {upload.filename!r} exceeds {max_bytes} bytes")
            output.write(block)
    await upload.close()
    if written == 0:
        raise HTTPException(status_code=400, detail=f"Upload {upload.filename!r} is empty")
    return written


def build_bundle(job_dir: Path, scope: str) -> Path:
    if scope not in {"results", "all"}:
        raise HTTPException(status_code=400, detail="scope must be 'results' or 'all'")
    output_dir = job_dir / "output"
    bundle_path = job_dir / f"{scope}_bundle.zip"
    if bundle_path.is_file():
        return bundle_path
    if scope == "all":
        selected = [path for path in output_dir.rglob("*") if path.is_file()]
    else:
        selected = [path for path in (output_dir / "04_targets").rglob("*") if path.is_file()]
        selected.extend(
            path
            for path in (
                output_dir / "pipeline.log",
                output_dir / "00_segmentation" / "dropper" / "overlay.png",
                output_dir / "00_segmentation" / "beaker" / "overlay.png",
                output_dir / "01_affordance" / "dropper" / "affordance_overlay.png",
                output_dir / "01_affordance" / "beaker" / "affordance_overlay.png",
                output_dir / "01_affordance" / "dropper" / "model_input.png",
                output_dir / "01_affordance" / "beaker" / "model_input.png",
                output_dir / "01_affordance" / "dropper" / "quality_metrics.json",
                output_dir / "01_affordance" / "beaker" / "quality_metrics.json",
                output_dir / "02_sam3d" / "dropper" / "overlay.png",
                output_dir / "02_sam3d" / "beaker" / "overlay.png",
            )
            if path.is_file()
        )
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(selected)):
            archive.write(path, arcname=str(path.relative_to(output_dir)))
    return bundle_path


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings.from_env()
    app = FastAPI(title="Language Affordance Titration API", version="1.0")
    manager_holder: dict[str, JobManager] = {}

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {settings.api_token}"
        if not settings.api_token or authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    async def manager() -> JobManager:
        return manager_holder["manager"]

    @app.on_event("startup")
    async def startup() -> None:
        settings.validate()
        manager_holder["manager"] = JobManager(settings)

    @app.get("/health")
    async def health(_: None = Depends(authorize)) -> dict[str, Any]:
        return {
            "ok": True,
            "time": utc_now(),
            "calibration_path": str(settings.calibration_path),
            "run_root": str(settings.run_root),
            "worker_count": 1,
        }

    @app.post("/v1/titration/jobs", status_code=202)
    async def submit_job(
        rgb: UploadFile = File(...),
        depth: UploadFile = File(...),
        camera: UploadFile = File(...),
        world_ee: UploadFile = File(...),
        depth_scale: float = Form(0.001),
        gpu: int | None = Form(None),
        capture_id: str | None = Form(None),
        _: None = Depends(authorize),
        jobs: JobManager = Depends(manager),
    ) -> dict[str, Any]:
        if not (0.0 < depth_scale <= 1.0):
            raise HTTPException(status_code=400, detail="depth_scale must be in (0, 1]")
        if gpu is not None and not (0 <= gpu <= 63):
            raise HTTPException(status_code=400, detail="gpu must be a non-negative physical GPU index")
        job = jobs.create_job({"depth_scale": depth_scale, "gpu": gpu, "capture_id": capture_id})
        job_id = job["job_id"]
        input_dir = jobs._job_dir(job_id) / "inputs"
        try:
            sizes = {
                "rgb": await save_upload(rgb, input_dir / "rgb.png", settings.max_image_bytes),
                "depth": await save_upload(depth, input_dir / "depth.png", settings.max_image_bytes),
                "camera": await save_upload(camera, input_dir / "camera.json", settings.max_json_bytes),
                "world_ee": await save_upload(world_ee, input_dir / "T_world_ee.json", settings.max_json_bytes),
            }
        except Exception:
            shutil.rmtree(jobs._job_dir(job_id), ignore_errors=True)
            with jobs._lock:
                jobs._jobs.pop(job_id, None)
            raise
        jobs.update(job_id, upload_sizes=sizes)
        return jobs.enqueue(job_id)

    @app.get("/v1/titration/jobs/{job_id}")
    async def get_job(job_id: str, _: None = Depends(authorize), jobs: JobManager = Depends(manager)) -> dict[str, Any]:
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.get("/v1/titration/jobs/{job_id}/result")
    async def get_result(job_id: str, _: None = Depends(authorize), jobs: JobManager = Depends(manager)) -> dict[str, Any]:
        try:
            job = jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        if job["state"] != "completed":
            raise HTTPException(status_code=409, detail=f"Job state is {job['state']}")
        return {"job_id": job_id, **job["result"]}

    @app.get("/v1/titration/jobs/{job_id}/bundle")
    async def download_bundle(
        job_id: str,
        scope: str = "results",
        _: None = Depends(authorize),
        jobs: JobManager = Depends(manager),
    ) -> FileResponse:
        try:
            job = jobs.get(job_id)
            job_dir = jobs._job_dir(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        if job["state"] != "completed":
            raise HTTPException(status_code=409, detail=f"Job state is {job['state']}")
        bundle = build_bundle(job_dir, scope)
        return FileResponse(bundle, media_type="application/zip", filename=f"titration_{job_id}_{scope}.zip")

    return app


app = create_app()
