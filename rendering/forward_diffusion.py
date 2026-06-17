"""DiffusionRenderer forward relighting from modified G-buffers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

DIFFUSION_ROOT = (
    Path(__file__).resolve().parents[1] / "3rd/cosmos-transfer1-diffusion-renderer"
)
GBUFFER_SUFFIXES = ("basecolor", "depth", "normal", "metallic", "roughness")
FORWARD_SCRIPT = "cosmos_predict1/diffusion/inference/inference_forward_renderer.py"
FORWARD_MODEL = "Diffusion_Renderer_Forward_Cosmos_7B"


def needs_forward_render(weather: str) -> bool:
    return weather in ("rain", "snow")


def gbuffer_scene_dir(work_dir: Path, seq_id: str, weather: str) -> Path:
    return work_dir / "gbuffer" / seq_id / weather


def _validate_gbuffer_jpgs(gbuffer_dir: Path) -> None:
    for suffix in GBUFFER_SUFFIXES:
        if not any(gbuffer_dir.glob(f"*.{suffix}.jpg")):
            raise FileNotFoundError(f"No G-buffer *.{suffix}.jpg files in {gbuffer_dir}")


def run_forward_render(
    gbuffer_dir: Path,
    output_dir: Path,
    num_frames: int,
    diffusion_root: str | Path = DIFFUSION_ROOT,
    checkpoint_dir: str | None = None,
    envlight_ind: int = 0,
    conda_env: str = "cosmos-predict1",
) -> Path:
    """Relight modified G-buffer frames; envlight_ind=0 uses asset/hdris/cloudy.hdr."""
    diffusion_root = Path(diffusion_root).resolve()
    gbuffer_dir = Path(gbuffer_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_gbuffer_jpgs(gbuffer_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(diffusion_root)
    if "CONDA_PREFIX" in env:
        env["CUDA_HOME"] = env["CONDA_PREFIX"]

    result = subprocess.run(
        [
            "conda", "run", "-n", conda_env, "python", FORWARD_SCRIPT,
            "--checkpoint_dir", checkpoint_dir or str(diffusion_root / "checkpoints"),
            "--diffusion_transformer_dir", FORWARD_MODEL,
            "--dataset_path", str(gbuffer_dir),
            "--num_video_frames", str(num_frames),
            "--use_custom_envmap", "True",
            "--envlight_ind", str(envlight_ind),
            "--video_save_folder", str(output_dir),
            "--save_image", "False",
        ],
        cwd=str(diffusion_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        log = (result.stderr or result.stdout or "")[-4000:]
        raise RuntimeError(f"DiffusionRenderer forward failed:\n{log}")

    for pattern in (f"*.relit_{envlight_ind:04d}.mp4", "*.relit_*.mp4"):
        matches = sorted(output_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No relit mp4 under {output_dir}")
