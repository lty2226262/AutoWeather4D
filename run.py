#!/usr/bin/env python3
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from rendering.cleanup import deliver_output, discard_path, prune_empty_dirs
from rendering.forward_diffusion import (
    gbuffer_scene_dir,
    needs_forward_render,
    run_forward_render,
)
from rendering.pipeline import WeatherRenderer
from rendering.vidrefiner import needs_vidrefiner, run_vidrefiner

WEATHERS = [
    "rain",
    "snow",
    "fog",
    "night",
]

REFINE_KEYS = (
    "videox_root",
    "conda_env",
    "strength",
    "seed",
    "video_length",
    "low_threshold",
    "high_threshold",
    "keep_intermediates",
)

_WEATHER_RENDER_SECTIONS = {
    "fog": "fog",
    "night": "night",
    "rain": "rain",
    "snow": "snow",
}


def _render_kwargs(cfg: OmegaConf, scene_id: str, weather: str) -> dict[str, object]:
    """Merge base render config with per-scene weather overrides."""
    render = dict(OmegaConf.to_container(cfg.render, resolve=True))
    scenes = OmegaConf.to_container(cfg.get("scenes", {}), resolve=True) or {}
    section = _WEATHER_RENDER_SECTIONS.get(weather)
    if not section:
        return render
    for scene_key in ("_default", scene_id):
        scene = scenes.get(scene_key, {})
        if section in scene:
            render.update(scene[section])
    return render


def _inside_work_dir(path: Path, work_dir: Path) -> bool:
    try:
        path.resolve().relative_to(work_dir.resolve())
        return True
    except ValueError:
        return False


def _find_video(render_dir: Path, weather: str) -> Path:
    for name in (
        f"blended_result_{weather}.mp4",
        f"blended_fog_{weather}.mp4",
        f"render_{weather}.mp4",
    ):
        path = render_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No video found in {render_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render weather video: scene .h5 -> modified G-buffer -> DiffusionRenderer forward"
    )
    parser.add_argument("--input", required=True, help="Input scene .h5")
    parser.add_argument("--output", required=True, help="Output .mp4")
    parser.add_argument("--weather", default="rain", choices=WEATHERS)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-refine", action="store_true", help="Skip VidRefiner post-processing")
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep per-scene work dirs (gbuffer, edges, relighting_forward, blended mp4)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep_intermediates = args.keep_intermediates or bool(cfg.get("keep_intermediates", False))

    renderer = WeatherRenderer(args.input)
    weather_dir = out.parent / renderer.seq_id / args.weather

    renderer.render(
        weather=args.weather,
        output_dir=str(out.parent),
        **_render_kwargs(cfg, renderer.seq_id, args.weather),
    )

    if needs_forward_render(args.weather):
        forward_cfg = dict(OmegaConf.to_container(cfg.get("forward", {}), resolve=True))
        src = run_forward_render(
            gbuffer_dir=gbuffer_scene_dir(weather_dir, renderer.seq_id, args.weather),
            output_dir=weather_dir / "relighting_forward",
            num_frames=renderer.frame_count,
            **forward_cfg,
        )
    else:
        src = _find_video(weather_dir, args.weather)

    refine_cfg = dict(OmegaConf.to_container(cfg.get("refine", {}), resolve=True))
    if (
        not args.no_refine
        and refine_cfg.get("enabled", True)
        and needs_vidrefiner(args.weather, refine_cfg.get("weathers", []))
    ):
        refine_kwargs = {k: refine_cfg[k] for k in REFINE_KEYS if k in refine_cfg}
        refine_kwargs["keep_intermediates"] = keep_intermediates
        by_weather = refine_cfg.get("strength_by_weather") or {}
        if args.weather in by_weather:
            refine_kwargs["strength"] = by_weather[args.weather]
        if not keep_intermediates:
            refine_kwargs["output_path"] = out
        src = run_vidrefiner(src, args.weather, **refine_kwargs)

    deliver_output(src, out)

    if not keep_intermediates and weather_dir.exists() and not _inside_work_dir(out, weather_dir):
        discard_path(weather_dir)
        prune_empty_dirs(weather_dir.parent, root=out.parent)

    print(out)


if __name__ == "__main__":
    main()
