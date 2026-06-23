"""Move intermediate artifacts to trash instead of keeping them on disk."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

TRASH_ROOT = Path("/tmp/ai_trash")


def discard_path(path: Path) -> None:
    """Move a file or directory out of the workspace."""
    if not path.exists():
        return
    TRASH_ROOT.mkdir(parents=True, exist_ok=True)
    dest = TRASH_ROOT / f"{path.name}.{int(time.time() * 1000)}"
    shutil.move(str(path), str(dest))


def deliver_output(src: Path, dst: Path) -> None:
    """Place the final video at ``dst``, moving when possible."""
    dst = dst.resolve()
    src = src.resolve()
    if src == dst:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dst))
    except OSError:
        shutil.copy2(src, dst)
        discard_path(src)


def prune_empty_dirs(path: Path, *, root: Path) -> None:
    """Remove empty directories from ``path`` up to ``root`` (exclusive)."""
    root = root.resolve()
    current = path.resolve()
    while current != root and current.is_dir():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent
