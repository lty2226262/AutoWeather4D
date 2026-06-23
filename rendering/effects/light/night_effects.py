"""Night sky darkening and per-frame material inputs for night rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class NightEffectParams:
    """Resolved night post-processing parameters for one renderer instance."""

    sky_darkening_factor: float


class NightEffects:
    """Post-render night adjustments applied after BRDF shading."""

    def __init__(self, device: torch.device | str = "cuda") -> None:
        """Initialize night effect kernels on the given device.

        Args:
            device: Torch device for tensor ops.
        """
        self.device = torch.device(device)

    def apply_sky_darkening(
        self,
        color: torch.Tensor,
        sky_mask: torch.Tensor,
        sky_darkening_factor: float,
    ) -> torch.Tensor:
        """Darken sky pixels to simulate night ambient lighting.

        Args:
            color: Rendered color, shape ``(3, H, W)``.
            sky_mask: Boolean sky mask, shape ``(H, W)`` or ``(1, H, W)``.
            sky_darkening_factor: Multiplier for sky pixels (0.0 = black, 1.0 = unchanged).

        Returns:
            Color with darkened sky, shape ``(3, H, W)``.
        """
        sky_mask_2d = torch.as_tensor(sky_mask, device=color.device).bool()
        if sky_mask_2d.dim() == 3 and sky_mask_2d.shape[0] == 1:
            sky_mask_2d = sky_mask_2d[0]

        if not sky_mask_2d.any():
            return color

        color_out = color.clone()
        color_out[:, sky_mask_2d] = color[:, sky_mask_2d] * sky_darkening_factor
        return color_out


class NightRenderer:
    """Night renderer integrated into the BRDF pipeline."""

    def __init__(self, night_effects: NightEffects, params: NightEffectParams) -> None:
        """Store night parameters used during post-render passes.

        Args:
            night_effects: Shared night effect kernels.
            params: Resolved sky darkening settings.
        """
        self.night_effects = night_effects
        self.params = params

    @classmethod
    def from_preset_dict(
        cls,
        night_effects: NightEffects,
        params: dict[str, Any],
    ) -> NightRenderer:
        """Build a renderer from a merged preset or override dictionary.

        Args:
            night_effects: Shared night effect kernels.
            params: Flat dict with ``sky_darkening_factor``.

        Returns:
            Configured :class:`NightRenderer` instance.
        """
        return cls(night_effects, _night_params_from_dict(params))

    def apply_sky_darkening(
        self,
        color: torch.Tensor,
        sky_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply configured sky darkening when a sky mask is available.

        Args:
            color: Rendered color, shape ``(3, H, W)``.
            sky_mask: Boolean sky mask, or ``None`` to skip darkening.

        Returns:
            Color with optional sky darkening, shape ``(3, H, W)``.
        """
        if sky_mask is None:
            return color

        return self.night_effects.apply_sky_darkening(
            color,
            sky_mask,
            self.params.sky_darkening_factor,
        )

    def material_inputs_for_frame(
        self,
        frame_idx: int,
        emissive_frames: list[dict[str, Any]] | None,
        light_frames: list[list[dict[str, Any]]] | None,
    ) -> tuple[torch.Tensor | None, list[dict[str, Any]] | None]:
        """Return per-frame emission and lights for BRDF shading.

        Args:
            frame_idx: Zero-based frame index.
            emissive_frames: Per-frame emissive dicts from HDF5.
            light_frames: Per-frame light lists from HDF5.

        Returns:
            Tuple of ``(emission, lights)``. Either value may be ``None``.
        """
        emission = None
        lights = None

        if emissive_frames and frame_idx < len(emissive_frames):
            emission = emissive_frames[frame_idx].get("emission")

        if light_frames and frame_idx < len(light_frames):
            lights = light_frames[frame_idx]

        return emission, lights


def _night_params_from_dict(params: dict[str, Any]) -> NightEffectParams:
    """Build :class:`NightEffectParams` from a preset or override dictionary."""
    return NightEffectParams(sky_darkening_factor=float(params["sky_darkening_factor"]))


def create_night_presets() -> dict[str, dict[str, Any]]:
    """Return named presets for night rendering configuration.

    Returns:
        Dict mapping preset names to night parameter dicts. Each preset may
        include ``enable_car_lights`` and ``enable_emissive`` for
        :meth:`LightEffectsManager.configure_night`; only
        ``sky_darkening_factor`` is consumed by :class:`NightRenderer`.
    """
    return {
        "night_clear": {
            "sky_darkening_factor": 0.15,
            "enable_car_lights": True,
            "enable_emissive": False,
        },
        "night_with_lights": {
            "sky_darkening_factor": 0.25,
            "enable_car_lights": True,
            "enable_emissive": True,
        },
        "night_dark": {
            "sky_darkening_factor": 0.05,
            "enable_car_lights": True,
            "enable_emissive": True,
        },
    }


__all__ = [
    "NightEffectParams",
    "NightEffects",
    "NightRenderer",
    "create_night_presets",
]
