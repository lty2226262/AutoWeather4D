"""Rain/snow path: geometry effects and G-buffer export."""

from __future__ import annotations

import os
import time

from tqdm import tqdm

from rendering.gbuffer.material import Material
from rendering.gbuffer.save import save_gbuffer_frame
from rendering.pipeline.progress import update_frame_progress
from rendering.pipeline.render_context import RenderContext
from rendering.pipeline.render_host import GeometryPipelineHost


def _gbuffer_save_path(host: GeometryPipelineHost, ctx: RenderContext) -> str:
    """Return the directory path where G-buffer JPGs should be written.

    Args:
        host: Initialized rain/snow pipeline host.
        ctx: Current render context.

    Returns:
        Absolute or relative path to the G-buffer output directory.
    """
    cfg = ctx.config
    return os.path.join(
        cfg.output_dir, host.seq_id, cfg.weather, "gbuffer", host.seq_id, cfg.weather
    )


def _apply_geometry(
    host: GeometryPipelineHost,
    ctx: RenderContext,
    frame_idx: int,
    material: Material,
) -> None:
    """Apply rain or snow geometry modifications to a frame's material.

    Args:
        host: Initialized rain/snow pipeline host with a geometry manager.
        ctx: Current render context.
        frame_idx: Zero-based frame index.
        material: Mutable per-frame material buffers.
    """
    flags = ctx.flags
    cfg = ctx.config
    if flags.use_snow and host.geometry_manager is not None:
        host.geometry_manager.apply_snow(
            material,
            snow_amount=cfg.snow.snow_amount,
        )
    if flags.use_rain and host.geometry_manager is not None:
        host.geometry_manager.apply_rain(
            frame_idx=frame_idx,
            time=frame_idx * cfg.rain.dt,
        )


def _sync_material_kwargs(
    host: GeometryPipelineHost,
    ctx: RenderContext,
    frame_idx: int,
    material: Material,
) -> None:
    """Copy modified material tensors back into ``MaterialVideo`` kwargs for export.

    Args:
        host: Initialized rain/snow pipeline host.
        ctx: Current render context.
        frame_idx: Zero-based frame index.
        material: Per-frame material after geometry effects were applied.
    """
    kwargs = host.material_video.material_kwargs[frame_idx]
    kwargs["normal"] = material.normal.clone()
    kwargs["position"] = material.position.clone()
    kwargs["roughness"] = material.roughness.clone()
    kwargs["metallic"] = material.metallic.clone()
    if ctx.flags.use_snow and not ctx.flags.use_rain:
        kwargs["depth"] = material.position[2:3, :, :].clone()
    kwargs["basecolor"] = material.albedo.clone()


def _export_frame(
    host: GeometryPipelineHost,
    ctx: RenderContext,
    frame_idx: int,
    save_path: str,
) -> None:
    """Apply geometry effects and write one G-buffer frame.

    Args:
        host: Initialized rain/snow pipeline host.
        ctx: Current render context.
        frame_idx: Zero-based frame index.
        save_path: Directory where G-buffer JPGs are written.
    """
    material = host.material_video[frame_idx]
    _apply_geometry(host, ctx, frame_idx, material)
    _sync_material_kwargs(host, ctx, frame_idx, material)
    save_gbuffer_frame(
        save_path,
        frame_idx,
        material.albedo,
        host.material_video.get_depth(frame_idx),
        material.normal,
        material.metallic,
        material.roughness,
        grade_basecolor=ctx.flags.use_snow,
    )


def export_geometry_gbuffer(host: GeometryPipelineHost, ctx: RenderContext) -> str:
    """Apply geometry effects and export per-frame G-buffer JPGs.

    Sets ``ctx.gbuffer_save_path`` on success. The returned directory is passed to
    DiffusionRenderer for rain/snow output generation.

    Args:
        host: Initialized rain/snow pipeline host.
        ctx: Current render context.

    Returns:
        Directory containing exported G-buffer image files.

    Examples:
        >>> gbuffer_dir = export_geometry_gbuffer(host, ctx)
        >>> assert os.path.isdir(gbuffer_dir)
    """
    save_path = _gbuffer_save_path(host, ctx)
    os.makedirs(save_path, exist_ok=True)
    ctx.gbuffer_save_path = save_path

    print(f"\nExporting G-buffer output ({host.frame_count} frames)...")
    print(f"G-buffer directory: {save_path}")

    pbar = tqdm(total=host.frame_count, desc="G-buffer export", unit="frame")
    start_time = time.time()

    for fi in range(host.frame_count):
        _export_frame(host, ctx, fi, save_path)
        update_frame_progress(pbar, fi, start_time, host.frame_count)

    pbar.close()
    print(f"\nG-buffer output: {save_path}")
    return save_path
