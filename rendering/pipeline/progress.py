"""Shared tqdm ETA helpers for pipeline frame loops."""

from __future__ import annotations

import time

from tqdm import tqdm


def _format_remaining_time(remaining_seconds: float) -> str:
    """Format a remaining-time estimate for a progress bar postfix."""
    remaining_seconds = max(remaining_seconds, 0.0)
    if remaining_seconds < 60:
        return f"{remaining_seconds:.1f}s"
    if remaining_seconds < 3600:
        return f"{remaining_seconds / 60:.1f}m"
    hours = int(remaining_seconds // 3600)
    minutes = int((remaining_seconds % 3600) // 60)
    return f"{hours}h{minutes}m"


def update_frame_progress(
    pbar: tqdm,
    frame_idx: int,
    start_time: float,
    frame_count: int,
) -> None:
    """Advance a frame progress bar and update ETA / FPS postfix values.

    Args:
        pbar: Active :class:`tqdm` progress bar.
        frame_idx: Zero-based index of the frame just completed.
        start_time: Start timestamp from :func:`time.time`.
        frame_count: Total number of frames in the job.
    """
    pbar.update(1)
    completed = frame_idx + 1
    if completed <= 1:
        return

    elapsed = time.time() - start_time
    if elapsed <= 0:
        return

    avg_time = elapsed / completed
    remaining = avg_time * (frame_count - completed)
    pbar.set_postfix({
        "remaining": _format_remaining_time(remaining),
        "fps": f"{completed / elapsed:.2f}",
    })
