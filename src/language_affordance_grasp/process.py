from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable


def build_python_command(conda_env: str | None, args: Iterable[str]) -> list[str]:
    command_args = list(args)
    if conda_env:
        return ["conda", "run", "-n", conda_env, *command_args]
    return command_args


def run_command(
    command: list[str],
    cwd: str | Path | None = None,
    env_updates: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    printable = " ".join(shlex.quote(item) for item in command)
    if cwd is not None:
        print(f"[RUN] cwd={cwd} {printable}")
    else:
        print(f"[RUN] {printable}")
    if dry_run:
        return
    environment = os.environ.copy()
    if env_updates:
        environment.update(env_updates)
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=environment, check=True)

