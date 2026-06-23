"""Orchestrates weather rendering pipelines for rain, snow, fog, and night."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from rendering.gbuffer.material import MaterialVideo
from rendering.lighting.brdf import CookTorranceBRDF
from rendering.pipeline.brdf_video import render_brdf_video
from rendering.pipeline.forward_gbuffer import export_geometry_gbuffer
from rendering.pipeline.geometry_effects import GeometryEffectsManager
from rendering.pipeline.light_effects import LightEffectsManager
from rendering.pipeline.render_context import (
    RenderConfig,
    RenderContext,
    RenderResult,
    WeatherFlags,
    parse_weather,
)


class WeatherRenderer:
    """Orchestrates weather rendering pipelines.

    Pipeline:
      rain/snow  -> geometry effects -> G-buffer -> DiffusionRenderer -> output mp4
      fog/night  -> BRDF render + relit blend -> output mp4

    Examples:
        >>> renderer = WeatherRenderer(h5_file="scene.h5")
        >>> renderer.render(weather="rain", output_dir="output")
        >>> renderer.render(weather="night", output_dir="output")
    """

    def __init__(self, h5_file: str, device: str = "cuda", **_) -> None:
        """Load scene material data from an HDF5 file.

        Args:
            h5_file: Path to the input scene `.h5` file.
            device: Torch device name, e.g. ``"cuda"`` or ``"cpu"``.
            **_: Ignored extra keyword arguments for API compatibility.

        Raises:
            FileNotFoundError: If ``h5_file`` does not exist.
        """
        self.h5_file = Path(h5_file)
        self.device = torch.device(device)

        if not self.h5_file.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_file}")

        print(f"Loading material data from: {self.h5_file}")
        print(f"Using device: {self.device}")

        self.material_video = MaterialVideo(input_path=str(self.h5_file), device=self.device)
        self.frame_count = len(self.material_video)
        self.seq_id = self.h5_file.stem.split("_")[0]

        self.geometry_manager: GeometryEffectsManager | None = None
        self.light_manager: LightEffectsManager | None = None
        self.brdf: CookTorranceBRDF | None = None

        print(f"Loaded {self.frame_count} frames")
        print(f"Sequence ID: {self.seq_id}")

    def render(
        self,
        weather: str = "rain",
        output_dir: str = "./output",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Render weather effects for the loaded scene.

        Args:
            weather: Weather preset name (``rain``, ``snow``, ``fog``, or ``night``).
            output_dir: Root directory for rendered outputs.
            **kwargs: Extra fields mapped into :class:`RenderConfig`.

        Returns:
            Dict with ``output_dir`` and, depending on weather, ``gbuffer_dir`` or
            ``video_path``.
        """
        config = RenderConfig.from_kwargs(weather, output_dir, **kwargs)
        return self.render_with_config(config).as_dict()

    def render_with_config(self, config: RenderConfig) -> RenderResult:
        """Render weather effects using a structured configuration object.

        Args:
            config: Fully populated render configuration.

        Returns:
            Structured result containing output paths for G-buffer or video artifacts.
        """
        ctx = self._build_context(config)
        self._initialize_effects(ctx)

        if ctx.flags.use_forward_relight:
            gbuffer_dir = export_geometry_gbuffer(self, ctx)
            self._print_summary(ctx)
            return RenderResult(output_dir=ctx.final_output_dir, gbuffer_dir=gbuffer_dir)

        video_path = render_brdf_video(self, ctx)
        self._print_summary(ctx)
        return RenderResult(output_dir=ctx.final_output_dir, video_path=video_path)

    def _build_context(self, config: RenderConfig) -> RenderContext:
        """Create per-render flags, paths, and mutable runtime state.

        Args:
            config: User render configuration.

        Returns:
            Initialized :class:`RenderContext` for this render call.
        """
        geometry, light = parse_weather(config.weather)
        final_output_dir = os.path.join(config.output_dir, self.seq_id, config.weather)
        os.makedirs(final_output_dir, exist_ok=True)

        flags = WeatherFlags(
            use_snow="snow" in geometry,
            use_rain="rain" in geometry,
            use_fog="fog" in light,
            use_night="night" in light,
        )

        print(f"\n{'=' * 60}")
        print(f"Weather Rendering: {config.weather}")
        print(f"{'=' * 60}")
        print(f"Output directory: {final_output_dir}")

        return RenderContext(
            config=config,
            flags=flags,
            geometry=geometry,
            light=light,
            final_output_dir=final_output_dir,
        )

    def _initialize_effects(self, ctx: RenderContext) -> None:
        """Initialize geometry, lighting, and BRDF subsystems for the active weather.

        Args:
            ctx: Current render context whose flags determine which managers to create.
        """
        flags = ctx.flags
        cfg = ctx.config

        if flags.use_snow or flags.use_rain:
            self.geometry_manager = GeometryEffectsManager(device=str(self.device))
            if flags.use_snow:
                print("\nInitializing GEOMETRY effect: Snow")
                self.geometry_manager.initialize_snow(
                    self.material_video,
                    max_points=cfg.snow.max_points,
                    preset=cfg.snow.snow_preset,
                    dataset_id=getattr(self.material_video, "dataset_id", None),
                    accumulated_snow_amount=cfg.snow.snow_amount,
                    grid_snow_enabled=cfg.snow.grid_snow_enabled,
                    grid_snow_density=cfg.snow.grid_snow_density,
                    snowfall_num_particles=cfg.snow.snowfall_num_particles,
                    snowfall_radius_world=cfg.snow.snowfall_radius_world,
                )
            elif flags.use_rain:
                print("\nInitializing GEOMETRY effect: Rain")
                self.geometry_manager.initialize_rain(self.material_video, dt=cfg.rain.dt)

        if flags.use_fog or flags.use_night:
            self.light_manager = LightEffectsManager(device=str(self.device))
            if flags.use_fog:
                print("\nInitializing LIGHT effect: Fog (with lights for light beams)")
                self.light_manager.configure_fog(
                    h5_file=str(self.h5_file),
                    enable_lights=True,
                    enable_emissive=True,
                    density_scale=cfg.fog.fog_density,
                )
            if flags.use_night:
                print("\nInitializing LIGHT effect: Night")
                self.light_manager.configure_night(
                    h5_file=str(self.h5_file),
                    sky_darkening_factor=cfg.night.sky_darkening_factor,
                    enable_car_lights=cfg.night.enable_car_lights,
                    enable_emissive=cfg.night.enable_emissive,
                )

        if not flags.use_forward_relight:
            self.brdf = CookTorranceBRDF()

    def _print_summary(self, ctx: RenderContext) -> None:
        """Print a short completion summary to stdout.

        Args:
            ctx: Render context containing the final output directory.
        """
        print(f"\n{'=' * 60}")
        print("Rendering Summary")
        print(f"{'=' * 60}")
        print(f"Weather: {ctx.config.weather}")
        print(f"Frames: {self.frame_count}")
        print(f"Output: {ctx.final_output_dir}")
