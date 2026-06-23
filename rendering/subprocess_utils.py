"""Helpers for running long-running inference subprocesses with live output."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run_with_live_output(
    cmd: list[str],
    *,
    cwd: Path | str,
    env: dict[str, str] | None = None,
    label: str = "subprocess",
) -> None:
    """Run a command and stream stdout/stderr to the terminal (for tqdm progress bars)."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    run_env["PYTHONUNBUFFERED"] = "1"

    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}\n", flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=run_env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {result.returncode})")
