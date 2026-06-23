"""VidRefiner post-processing: Canny edges + Wan2.2 V2V SDEdit."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rendering.cleanup import discard_path
from rendering.subprocess_utils import run_with_live_output

VIDREFINER_ROOT = Path(__file__).resolve().parents[1] / "3rd" / "VideoX-Fun"
EDGE_SCRIPT = "edge_extraction.py"
WAN_SCRIPT = "examples/wan2.2_fun/predict_v2v_control_sdedit_5b.py"

# Wan predict script matches edge filenames against these tags (rainy/snowy, not rain/snow).
EDGE_PROMPT_TAGS: dict[str, str] = {
    "rain": "rainy",
    "snow": "snowy",
}


def edge_prompt_tag(weather: str) -> str:
    """Return the edge-video prompt tag expected by VideoX-Fun Wan prompt selection."""
    return EDGE_PROMPT_TAGS.get(weather, weather)


def needs_vidrefiner(weather: str, weathers: list[str]) -> bool:
    """Return True when the weather is listed for VidRefiner."""
    return weather in weathers


def _run_subprocess(cmd: list[str], cwd: Path, env: dict[str, str], label: str) -> None:
    run_with_live_output(cmd, cwd=cwd, env=env, label=label)


def run_vidrefiner(
    rendered_video: Path,
    weather: str,
    videox_root: str | Path = VIDREFINER_ROOT,
    conda_env: str = "autoweather4d",
    strength: float = 0.4,
    seed: int = 50,
    video_length: int = 57,
    low_threshold: int = 40,
    high_threshold: int = 80,
    keep_intermediates: bool = False,
    output_path: Path | None = None,
) -> Path:
    """Refine a rendered weather video with edge-guided Wan2.2 SDEdit.

    Args:
        rendered_video: Input mp4 from the weather render stage.
        weather: Weather tag used for edge naming and Wan prompt selection.
        videox_root: VideoX-Fun project root containing edge and predict scripts.
        conda_env: Conda environment name for VideoX-Fun dependencies.
        strength: SDEdit strength passed to the Wan predict script.
        seed: Random seed for Wan inference.
        video_length: Maximum number of frames for edge extraction.
        low_threshold: Canny low threshold for edge extraction.
        high_threshold: Canny high threshold for edge extraction.
        keep_intermediates: When True, keep edge/Wan artifacts under ``edges/``.
        output_path: Destination mp4 when intermediates are discarded.

    Returns:
        Path to the refined output mp4.

    Raises:
        FileNotFoundError: If edge or refined output video is missing after subprocess runs.
        RuntimeError: If either subprocess exits with a non-zero status.
    """
    videox_root = Path(videox_root).resolve()
    rendered_video = Path(rendered_video).resolve()
    if output_path is not None:
        output_path = Path(output_path).resolve()
    prompt_tag = edge_prompt_tag(weather)
    strength_label = f"{strength:g}"
    save_name = f"{rendered_video.stem}_wan_s{strength_label}_seed{seed}.mp4"

    env = os.environ.copy()
    if not env.get("CUDA_HOME"):
        for candidate in ("/usr/local/cuda", "/usr/local/cuda-12.8", "/usr/local/cuda-12"):
            if Path(candidate).exists():
                env["CUDA_HOME"] = candidate
                break

    if keep_intermediates:
        edge_output_dir = rendered_video.parent / "edges"
        edge_output_dir.mkdir(parents=True, exist_ok=True)
        work_ctx = None
    else:
        work_ctx = tempfile.TemporaryDirectory(prefix="vidrefiner_")
        edge_output_dir = Path(work_ctx.name)

    try:
        control_video = edge_output_dir / f"{rendered_video.stem}.{prompt_tag}.edge.mp4"
        if not (keep_intermediates and control_video.exists()):
            edge_cmd = [
                "conda", "run", "--no-capture-output", "-n", conda_env, "python", "-u", EDGE_SCRIPT,
                "--input_video", str(rendered_video),
                "--output_dir", str(edge_output_dir),
                "--prompt", prompt_tag,
                "--video_length", str(video_length),
                "--low_threshold", str(low_threshold),
                "--high_threshold", str(high_threshold),
            ]
            _run_subprocess(edge_cmd, cwd=videox_root, env=env, label="VidRefiner: edge extraction")

        if not control_video.exists():
            candidates = list(edge_output_dir.glob(f"{rendered_video.stem}.*.edge.mp4"))
            if candidates:
                control_video = candidates[0]
        if not control_video.exists():
            raise FileNotFoundError(f"No edge control video under {edge_output_dir}")

        refined_video = edge_output_dir / save_name
        if keep_intermediates and refined_video.exists():
            discard_path(refined_video)

        wan_cmd = [
            "conda", "run", "--no-capture-output", "-n", conda_env, "python", "-u", WAN_SCRIPT,
            "--ori_video_path", str(rendered_video),
            "--control_video_path", str(control_video),
            "--video_save_name", save_name,
            "--strength", str(strength),
            "--seed", str(seed),
        ]
        _run_subprocess(wan_cmd, cwd=videox_root, env=env, label="VidRefiner: Wan 5B V2V SDEdit")

        if not refined_video.exists():
            raise FileNotFoundError(f"No refined mp4 under {edge_output_dir}")

        if output_path is not None:
            from rendering.cleanup import deliver_output

            deliver_output(refined_video, output_path)
            return output_path

        return refined_video
    finally:
        if work_ctx is not None:
            work_ctx.cleanup()
