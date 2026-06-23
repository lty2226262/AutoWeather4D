"""Fog/night path: BRDF shading with relit blending."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from rendering.color.lut import linear_blend_images, preprocess_relit_frame
from rendering.gbuffer.material import Material, MaterialVideo
from rendering.pipeline.progress import update_frame_progress
from rendering.pipeline.render_context import RenderContext
from rendering.pipeline.light_effects import load_relit_frames_from_h5
from rendering.pipeline.render_host import BrdfPipelineHost


@dataclass(frozen=True)
class _RelitMixSettings:
    """LUT and blend parameters for fog or night relit compositing."""

    h5_group: str
    output_name: str
    strength: float
    sky_darkening_factor: float
    lut_type: str
    img2_strength: float


def _relit_mix_settings(ctx: RenderContext) -> _RelitMixSettings | None:
    """Return relit blending settings for the active fog/night weather preset.

    Args:
        ctx: Current render context.

    Returns:
        Mixing settings when fog or night is active; otherwise ``None``.
    """
    weather = ctx.config.weather
    if ctx.flags.use_night:
        return _RelitMixSettings(
            h5_group="night",
            output_name=f"blended_result_{weather}.mp4",
            strength=0.5,
            sky_darkening_factor=ctx.config.night.sky_darkening_factor,
            lut_type="moonlight",
            img2_strength=0.5,
        )
    if ctx.flags.use_fog:
        return _RelitMixSettings(
            h5_group="cloudy",
            output_name=f"blended_fog_{weather}.mp4",
            strength=0.3,
            sky_darkening_factor=1.0,
            lut_type="fog",
            img2_strength=2.0,
        )
    return None


def _output_video_path(ctx: RenderContext, mix: _RelitMixSettings | None, use_mix: bool) -> str:
    """Return the output mp4 path for the current fog/night render.

    Args:
        ctx: Current render context.
        mix: Relit blending settings when mixing is enabled.
        use_mix: Whether relit frames are available for blending.

    Returns:
        Absolute or relative path to the output mp4 file.
    """
    if use_mix and mix is not None:
        return os.path.join(ctx.final_output_dir, mix.output_name)
    return os.path.join(ctx.final_output_dir, f"render_{ctx.config.weather}.mp4")


def _gather_lights(
    host: BrdfPipelineHost,
    ctx: RenderContext,
    frame_idx: int,
) -> tuple[list[dict[str, Any]] | None, torch.Tensor | None]:
    """Collect dynamic lights and emission for the current frame.

    Args:
        host: Initialized fog/night pipeline host with a light manager.
        ctx: Current render context.
        frame_idx: Zero-based frame index.

    Returns:
        Tuple of ``(lights, emission)``. Either value may be ``None``.
    """
    lights: list[dict[str, Any]] | None = None
    emission: torch.Tensor | None = None
    if ctx.flags.use_fog:
        fog_emission, fog_lights = host.light_manager.apply_fog_to_material(frame_idx=frame_idx)
        if fog_emission is not None:
            emission = fog_emission
        if fog_lights is not None:
            lights = fog_lights
    if ctx.flags.use_night:
        night_emission, night_lights = host.light_manager.apply_night_to_material(frame_idx=frame_idx)
        if night_emission is not None:
            emission = night_emission
        if night_lights is not None:
            lights = night_lights
    return lights, emission


def _shade_brdf(
    host: BrdfPipelineHost,
    ctx: RenderContext,
    frame_idx: int,
    material: Material,
    lights: list[dict[str, Any]] | None,
    emission: torch.Tensor | None,
) -> torch.Tensor:
    """Shade one frame with the Cook-Torrance BRDF shader.

    Args:
        host: Initialized fog/night pipeline host with BRDF and light managers.
        ctx: Current render context.
        frame_idx: Zero-based frame index.
        material: Per-frame material buffers.
        lights: Optional dynamic lights for the frame.
        emission: Optional per-pixel emission map.

    Returns:
        Shaded color tensor in sRGB, shape ``(3, H, W)``.
    """
    material.emission = emission
    view_dir = F.normalize(-material.position, dim=0)
    brdf_linear = host.brdf.render_many(material, view_dir, lights)

    if ctx.flags.use_fog:
        depth = torch.norm(material.position, dim=0, keepdim=True)
        brdf_linear = host.light_manager.apply_fog(
            brdf_linear,
            depth,
            material.position,
            lights=lights,
        )

    color = Material.linear_to_srgb(brdf_linear.clamp(0, 1))
    if ctx.flags.use_night:
        color = host.light_manager.apply_night(color, sky_mask=material.sky_mask)
    return color


def _load_relit_frames(
    host: BrdfPipelineHost,
    mix: _RelitMixSettings,
) -> list[np.ndarray] | None:
    """Load precomputed relit frames from the scene HDF5 file.

    Args:
        host: Initialized fog/night pipeline host.
        mix: Relit blending settings including the HDF5 group name.

    Returns:
        List of RGB uint8 frames, or ``None`` when the relit group is missing.
    """
    frames = load_relit_frames_from_h5(str(host.h5_file), mix.h5_group)
    if frames is None:
        print(f"Warning: relit group '{mix.h5_group}' not found in {host.h5_file}")
    return frames


def _tensor_to_uint8_hwc(color: torch.Tensor) -> np.ndarray:
    """Convert a CHW float color tensor to an HWC uint8 numpy frame."""
    return (color.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")


def _blend_with_relit(
    frame_np: np.ndarray,
    relit_frames: list[np.ndarray],
    frame_idx: int,
    mix: _RelitMixSettings,
    material_video: MaterialVideo,
    device: torch.device,
) -> np.ndarray:
    """Blend one BRDF frame with the matching relit frame.

    Args:
        frame_np: BRDF frame in HWC uint8 layout.
        relit_frames: Preloaded relit frames from HDF5.
        frame_idx: Zero-based frame index.
        mix: Relit blending settings.
        material_video: Scene material source for sky masks.
        device: Torch device for the blend operation.

    Returns:
        Blended frame in HWC uint8 layout.
    """
    relit_idx = min(frame_idx, len(relit_frames) - 1)
    relit_np = preprocess_relit_frame(
        relit_frames[relit_idx],
        sky_mask=material_video.get_sky_mask(frame_idx),
        strength=mix.strength,
        sky_darkening_factor=mix.sky_darkening_factor,
        lut_type=mix.lut_type,
    )
    base_t = torch.from_numpy(frame_np.astype("float32") / 255.0).permute(2, 0, 1).to(device)
    relit_t = torch.from_numpy(relit_np.astype("float32") / 255.0).permute(2, 0, 1).to(device)
    blended = linear_blend_images(relit_t, base_t, img2_strength=mix.img2_strength)
    return _tensor_to_uint8_hwc(blended)


def _render_frame(
    host: BrdfPipelineHost,
    ctx: RenderContext,
    frame_idx: int,
    mix: _RelitMixSettings | None,
    relit_frames: list[np.ndarray] | None,
    use_mix: bool,
) -> np.ndarray:
    """Shade one frame and optionally blend it with relit data.

    Args:
        host: Initialized fog/night pipeline host.
        ctx: Current render context.
        frame_idx: Zero-based frame index.
        mix: Relit blending settings when mixing is enabled.
        relit_frames: Preloaded relit frames, or ``None``.
        use_mix: Whether relit blending should be applied.

    Returns:
        Output frame in HWC uint8 layout.
    """
    material = host.material_video[frame_idx]
    lights, emission = _gather_lights(host, ctx, frame_idx)
    color = _shade_brdf(host, ctx, frame_idx, material, lights, emission)
    frame_np = _tensor_to_uint8_hwc(color)
    if use_mix and mix is not None and relit_frames is not None:
        frame_np = _blend_with_relit(
            frame_np,
            relit_frames,
            frame_idx,
            mix,
            host.material_video,
            material.device,
        )
    return frame_np


def render_brdf_video(host: BrdfPipelineHost, ctx: RenderContext) -> str:
    """Render fog/night frames to an output mp4.

    Shades each frame with BRDF, optionally blends with relit HDF5 frames, and
    writes ``blended_result_{weather}.mp4`` or ``blended_fog_{weather}.mp4``. When
    relit data is unavailable, writes ``render_{weather}.mp4`` instead. Sets
    ``ctx.video_path`` on success.

    Args:
        host: Initialized fog/night pipeline host.
        ctx: Current render context.

    Returns:
        Path to the written output mp4.

    Examples:
        >>> video_path = render_brdf_video(host, ctx)
        >>> assert video_path.endswith(".mp4")
    """
    mix = _relit_mix_settings(ctx)
    relit_frames = _load_relit_frames(host, mix) if mix is not None else None
    use_mix = relit_frames is not None and len(relit_frames) > 0
    video_path = _output_video_path(ctx, mix, use_mix)

    print(f"\nRendering output ({host.frame_count} frames)...")
    ctx.video_path = video_path
    writer = imageio.get_writer(video_path, fps=host.material_video.fps, quality=9)

    pbar = tqdm(total=host.frame_count, desc="Rendering output", unit="frame")
    start_time = time.time()

    for fi in range(host.frame_count):
        writer.append_data(_render_frame(host, ctx, fi, mix, relit_frames, use_mix))
        update_frame_progress(pbar, fi, start_time, host.frame_count)

    pbar.close()
    writer.close()
    print(f"\nOutput mp4: {video_path}")
    return video_path
