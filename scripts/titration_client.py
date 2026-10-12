#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import requests


TERMINAL_STATES = {"completed", "failed", "interrupted"}


def api_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def checked_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        payload = {"detail": response.text}
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {json.dumps(payload, ensure_ascii=False)}")
    return payload


def submit(args: argparse.Namespace) -> dict[str, Any]:
    with ExitStack() as stack:
        files = {
            "rgb": (Path(args.rgb).name, stack.enter_context(open(args.rgb, "rb")), "image/png"),
            "depth": (Path(args.depth).name, stack.enter_context(open(args.depth, "rb")), "image/png"),
            "camera": (Path(args.camera).name, stack.enter_context(open(args.camera, "rb")), "application/json"),
            "world_ee": (Path(args.world_ee).name, stack.enter_context(open(args.world_ee, "rb")), "application/json"),
        }
        data = {"depth_scale": str(args.depth_scale)}
        if args.gpu is not None:
            data["gpu"] = str(args.gpu)
        if args.capture_id:
            data["capture_id"] = args.capture_id
        response = requests.post(
            f"{args.server}/v1/titration/jobs",
            headers=api_headers(args.token),
            files=files,
            data=data,
            timeout=120,
        )
    return checked_json(response)


def get_status(server: str, token: str, job_id: str) -> dict[str, Any]:
    response = requests.get(
        f"{server}/v1/titration/jobs/{job_id}",
        headers=api_headers(token),
        timeout=30,
    )
    return checked_json(response)


def wait_for_job(server: str, token: str, job_id: str, poll_seconds: float) -> dict[str, Any]:
    previous = None
    while True:
        job = get_status(server, token, job_id)
        state = job["state"]
        if state != previous:
            print(f"job {job_id}: {state}", flush=True)
            previous = state
        if state in TERMINAL_STATES:
            return job
        time.sleep(poll_seconds)


def download(server: str, token: str, job_id: str, scope: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"titration_{job_id}_{scope}.zip"
    with requests.get(
        f"{server}/v1/titration/jobs/{job_id}/bundle",
        params={"scope": scope},
        headers=api_headers(token),
        stream=True,
        timeout=600,
    ) as response:
        if not response.ok:
            checked_json(response)
        with output_path.open("wb") as output:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    output.write(block)
    return output_path


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server", default=os.environ.get("LAG_SERVER_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--token", default=os.environ.get("LAG_API_TOKEN", ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload RGB-D titration captures and retrieve 3D targets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Submit, wait, print coordinates, and download artifacts.")
    add_connection_args(run)
    run.add_argument("--rgb", required=True)
    run.add_argument("--depth", required=True)
    run.add_argument("--camera", required=True)
    run.add_argument("--world-ee", required=True)
    run.add_argument("--depth-scale", type=float, default=0.001)
    run.add_argument("--gpu", type=int, default=None)
    run.add_argument("--capture-id", default=None)
    run.add_argument("--poll-seconds", type=float, default=5.0)
    run.add_argument("--scope", choices=["results", "all"], default="results")
    run.add_argument("--output-dir", default="./titration_server_results")

    status = subparsers.add_parser("status", help="Query one job.")
    add_connection_args(status)
    status.add_argument("job_id")

    fetch = subparsers.add_parser("download", help="Download a completed job bundle.")
    add_connection_args(fetch)
    fetch.add_argument("job_id")
    fetch.add_argument("--scope", choices=["results", "all"], default="results")
    fetch.add_argument("--output-dir", default="./titration_server_results")
    return parser


def require_token(args: argparse.Namespace) -> None:
    if not args.token:
        raise SystemExit("Set LAG_API_TOKEN or pass --token")


def main() -> None:
    args = build_parser().parse_args()
    args.server = args.server.rstrip("/")
    require_token(args)
    try:
        if args.command == "status":
            print(json.dumps(get_status(args.server, args.token, args.job_id), ensure_ascii=False, indent=2))
            return
        if args.command == "download":
            path = download(args.server, args.token, args.job_id, args.scope, Path(args.output_dir))
            print(path)
            return

        submitted = submit(args)
        job_id = submitted["job_id"]
        print(f"submitted job: {job_id}", flush=True)
        job = wait_for_job(args.server, args.token, job_id, args.poll_seconds)
        if job["state"] != "completed":
            print(json.dumps(job, ensure_ascii=False, indent=2), file=sys.stderr)
            raise SystemExit(1)
        result_response = requests.get(
            f"{args.server}/v1/titration/jobs/{job_id}/result",
            headers=api_headers(args.token),
            timeout=30,
        )
        result = checked_json(result_response)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        bundle = download(args.server, args.token, job_id, args.scope, Path(args.output_dir))
        print(f"downloaded: {bundle}")
    except (OSError, requests.RequestException, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
