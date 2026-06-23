"""Fog and night light effects for the BRDF rendering path."""

from __future__ import annotations

import os
import re
from typing import Any

import h5py
import numpy as np
import torch

from rendering.effects.light.fog_effects import FogEffects, FogRenderer, create_fog_presets
from rendering.effects.light.night_effects import NightEffects, NightRenderer, create_night_presets
from rendering.lighting.lights import load_emissive_from_h5, load_lights_from_h5

_LIGHT_INCLUDE = ("light_heads", "rear_car_heads", "clearance_lamp")


def _try_load_lights(
    h5_file: str,
    device: torch.device,
    scene_type: str,
) -> list[list[dict[str, Any]]] | None:
    """Load per-frame lights from HDF5, returning ``None`` when the group is absent."""
    try:
        return load_lights_from_h5(
            h5_file,
            device,
            include=_LIGHT_INCLUDE,
            scene_type=scene_type,
        )
    except KeyError as exc:
        print(f"Warning: optional lights not loaded from {h5_file}: {exc}")
        return None


def _try_load_emissive(h5_file: str, device: torch.device) -> list[dict[str, Any]] | None:
    """Load per-frame emissive maps from HDF5 when present."""
    emissive = load_emissive_from_h5(h5_file, device, emissive_type="tail_lights")
    return emissive or None


def _frame_key_to_int(key: str) -> int:
    digits = re.findall(r"\d+", str(key))
    return int(digits[-1]) if digits else 0


def _to_uint8_frame(frame: np.ndarray) -> np.ndarray:
    """Convert one relit frame to HWC uint8 layout."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255)
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = frame.astype(np.uint8)
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    return frame


def load_relit_frames_from_h5(h5_file: str, group_name: str) -> list[np.ndarray] | None:
    """Load precomputed relit RGB frames from an HDF5 group.

    Args:
        h5_file: Path to the scene ``.h5`` file.
        group_name: HDF5 group name, e.g. ``"cloudy"`` or ``"night"``.

    Returns:
        List of HWC uint8 frames, or ``None`` when the file or group is missing.
    """
    if not h5_file or not os.path.isfile(h5_file):
        return None

    with h5py.File(h5_file, "r") as h5f:
        if group_name not in h5f:
            return None
        group = h5f[group_name]
        frames: list[np.ndarray] = []
        if isinstance(group, h5py.Dataset):
            data = group[()]
            if data.ndim != 4:
                return None
            if data.shape[-1] == 3:
                source = data
            elif data.shape[1] == 3:
                source = data.transpose(0, 2, 3, 1)
            else:
                return None
            for frame in source:
                frames.append(_to_uint8_frame(frame))
        else:
            keys = sorted(group.keys(), key=_frame_key_to_int)
            for key in keys:
                frames.append(_to_uint8_frame(group[key][()]))
        return frames or None


class LightEffectsManager:
    """Configure and apply fog and night light effects on shaded BRDF output."""

    def __init__(self, device: str = "cuda") -> None:
        """Initialize empty fog and night effect state.

        Args:
            device: Torch device name, e.g. ``"cuda"`` or ``"cpu"``.
        """
        self.device = torch.device(device)

        self.fog_renderer: FogRenderer | None = None
        self.fog_enabled = False
        self.fog_density_scale = 1.0
        self.fog_lights: list[list[dict[str, Any]]] | None = None
        self.fog_emissive: list[dict[str, Any]] | None = None

        self.night_renderer: NightRenderer | None = None
        self.night_enabled = False
        self.night_lights: list[list[dict[str, Any]]] | None = None
        self.night_emissive: list[dict[str, Any]] | None = None

    def configure_fog(
        self,
        h5_file: str | None = None,
        preset: str = "volumetric_mist",
        fog_density: float | None = None,
        density_scale: float = 1.0,
        enable_lights: bool = True,
        enable_emissive: bool = True,
        **fog_params: Any,
    ) -> LightEffectsManager:
        """Configure volumetric fog for the BRDF path.

        Args:
            h5_file: Scene HDF5 path for optional vehicle lights and emissive maps.
            preset: Fog preset name from :func:`create_fog_presets`.
            fog_density: Optional override for base fog density.
            density_scale: Multiplier applied during rendering.
            enable_lights: Load vehicle lights for in-scattering beams.
            enable_emissive: Load emissive tail-light maps.
            **fog_params: Additional kwargs forwarded to :class:`FogRenderer`.

        Returns:
            ``self`` for chaining.
        """
        presets = create_fog_presets()
        base_params = presets.get(preset, presets["volumetric_mist"]).copy()

        if fog_density is not None:
            base_params["fog_density"] = fog_density

        base_params.update(fog_params)

        fog_effects = FogEffects(self.device)
        self.fog_renderer = FogRenderer.from_preset_dict(fog_effects, base_params)
        self.fog_density_scale = density_scale

        self.fog_lights = None
        self.fog_emissive = None
        if h5_file:
            if enable_lights:
                self.fog_lights = _try_load_lights(h5_file, self.device, scene_type="fog")
            if enable_emissive:
                self.fog_emissive = _try_load_emissive(h5_file, self.device)

        self.fog_enabled = True
        return self

    def configure_night(
        self,
        h5_file: str,
        preset: str = "night_with_lights",
        sky_darkening_factor: float | None = None,
        enable_car_lights: bool = True,
        enable_emissive: bool = False,
        **night_params: Any,
    ) -> LightEffectsManager:
        """Configure night sky darkening and optional vehicle lighting.

        Args:
            h5_file: Scene HDF5 path for optional vehicle lights and emissive maps.
            preset: Night preset name from :func:`create_night_presets`.
            sky_darkening_factor: Sky brightness scale in ``[0, 1]``.
            enable_car_lights: Load vehicle lights for BRDF shading.
            enable_emissive: Load emissive tail-light maps.
            **night_params: Additional kwargs forwarded to :class:`NightRenderer`.

        Returns:
            ``self`` for chaining.
        """
        presets = create_night_presets()
        base_params = presets.get(preset, presets["night_with_lights"]).copy()
        if sky_darkening_factor is not None:
            base_params["sky_darkening_factor"] = sky_darkening_factor
        base_params.update(night_params)
        base_params.pop("enable_car_lights", None)
        base_params.pop("enable_emissive", None)

        night_effects = NightEffects(self.device)
        self.night_renderer = NightRenderer.from_preset_dict(night_effects, base_params)

        self.night_lights = None
        self.night_emissive = None
        if enable_car_lights:
            self.night_lights = _try_load_lights(h5_file, self.device, scene_type="night")
        if enable_emissive:
            self.night_emissive = _try_load_emissive(h5_file, self.device)

        self.night_enabled = True
        return self

    def apply_fog(
        self,
        color: torch.Tensor,
        depth: torch.Tensor,
        position: torch.Tensor,
        lights: list[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        """Apply volumetric fog to one shaded frame.

        Args:
            color: Linear shaded color, shape ``(3, H, W)``.
            depth: Depth map, shape ``(1, H, W)``.
            position: World positions, shape ``(3, H, W)``.
            lights: Optional dynamic lights for in-scattering.

        Returns:
            Fogged linear color with the same shape as ``color``.
        """
        if not self.fog_enabled or self.fog_renderer is None:
            return color

        return self.fog_renderer.render_with_fog(
            color,
            depth,
            position,
            lights=lights,
            density_scale=self.fog_density_scale,
        )

    def apply_night(
        self,
        color: torch.Tensor,
        sky_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply night sky darkening to one sRGB frame.

        Args:
            color: Shaded color in sRGB, shape ``(3, H, W)``.
            sky_mask: Optional sky mask, shape ``(H, W)``.

        Returns:
            Adjusted sRGB color with the same shape as ``color``.
        """
        if not self.night_enabled or self.night_renderer is None:
            return color

        return self.night_renderer.apply_sky_darkening(color, sky_mask)

    def apply_fog_to_material(
        self,
        frame_idx: int = 0,
    ) -> tuple[torch.Tensor | None, list[dict[str, Any]] | None]:
        """Return fog emissive maps and lights for one frame index.

        Args:
            frame_idx: Zero-based frame index.

        Returns:
            Tuple of ``(emission, lights)``. Either value may be ``None``.
        """
        emission = None
        lights = None

        if self.fog_emissive and frame_idx < len(self.fog_emissive):
            emission = self.fog_emissive[frame_idx].get("emission")

        if self.fog_lights and frame_idx < len(self.fog_lights):
            lights = self.fog_lights[frame_idx]

        return emission, lights

    def apply_night_to_material(
        self,
        frame_idx: int = 0,
    ) -> tuple[torch.Tensor | None, list[dict[str, Any]] | None]:
        """Return night emissive maps and lights for one frame index.

        Args:
            frame_idx: Zero-based frame index.

        Returns:
            Tuple of ``(emission, lights)``. Either value may be ``None``.
        """
        if not self.night_enabled or self.night_renderer is None:
            return None, None

        return self.night_renderer.material_inputs_for_frame(
            frame_idx,
            self.night_emissive,
            self.night_lights,
        )
