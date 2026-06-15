#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

from omegaconf import OmegaConf
from rendering.forward_diffusion import (
    gbuffer_scene_dir,
    needs_forward_render,
    run_forward_render,
)
from rendering.pipeline import WeatherRenderer

WEATHERS = [
    "rain",
    "snow",
    "fog",
    "night",
]


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
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    renderer = WeatherRenderer(args.input, verbose=not bool(cfg.get("quiet", False)))
    weather_dir = out.parent / renderer.seq_id / args.weather

    renderer.render(
        weather=args.weather,
        output_dir=str(out.parent),
        **dict(OmegaConf.to_container(cfg.render, resolve=True)),
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

    if src.resolve() != out.resolve():
        shutil.copy2(src, out)
    print(out)


if __name__ == "__main__":
    main()
