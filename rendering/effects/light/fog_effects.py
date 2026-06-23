"""Volumetric fog scattering for the fog BRDF rendering path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

_HENYEY_GREENSTEIN_G = 0.8


@dataclass(frozen=True)
class FogParams:
    """Resolved volumetric fog parameters for one renderer instance."""

    fog_density: float
    scattering: float
    absorption: float
    fog_color: tuple[float, float, float]


class FogEffects:
    """Volume-scattering fog kernels applied to linear shaded color."""

    def __init__(self, device: torch.device | str = "cuda") -> None:
        """Initialize fog kernels on the given device.

        Args:
            device: Torch device for tensor ops.
        """
        self.device = torch.device(device)

    def _process_lights_for_volumetric_fog(
        self,
        lights: list[dict[str, Any]],
        surface_position: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Convert scene lights into directions and intensities for in-scattering."""
        light_dirs: list[torch.Tensor] = []
        light_intensities: list[torch.Tensor] = []

        for light in lights:
            light_type = light.get("type")
            if light_type == "point":
                light_pos = light['desc']['pos'].squeeze()
                distance_to_light = torch.norm(light_pos - surface_position)
                radius = light["desc"].get("light_radius_of_influence", float("inf"))

                if radius < float("inf"):
                    distance_factor = torch.clamp(1.0 - distance_to_light / radius, 0.0, 1.0)
                    distance_factor = distance_factor * distance_factor
                else:
                    distance_factor = 1.0

                if distance_factor > 0.01:
                    light_dir = F.normalize(light_pos - surface_position, dim=0)
                    attenuated_intensity = light["intensity"].squeeze() * distance_factor
                    light_dirs.append(light_dir)
                    light_intensities.append(attenuated_intensity)

            elif light_type == "spot":
                light_pos = light["desc"]["pos"].squeeze()
                light_dir = F.normalize(light["desc"]["dir"].squeeze(), dim=0)
                distance_to_light = torch.norm(light_pos - surface_position)

                if distance_to_light < 1e-6:
                    to_surface_dir = F.normalize(light["desc"]["dir"].squeeze(), dim=0)
                    distance_to_light = torch.tensor(1e-3, device=light_pos.device)
                else:
                    to_surface_dir = F.normalize(surface_position - light_pos, dim=0)
                radius = light["desc"].get("light_radius_of_influence", float("inf"))

                if radius < float("inf"):
                    distance_factor = torch.clamp(1.0 - distance_to_light / radius, 0.0, 1.0)
                    distance_factor = distance_factor * distance_factor
                else:
                    distance_factor = 1.0

                cos_angle = torch.sum(light_dir * to_surface_dir)
                inner_deg = light["desc"]["inner_deg"]
                outer_deg = light["desc"]["outer_deg"]
                inner_rad = math.radians(inner_deg)
                outer_rad = math.radians(outer_deg)

                if cos_angle >= math.cos(inner_rad):
                    angle_factor = 1.0
                elif cos_angle <= math.cos(outer_rad):
                    angle_factor = 0.0
                else:
                    t = (cos_angle - math.cos(outer_rad)) / (
                        math.cos(inner_rad) - math.cos(outer_rad)
                    )
                    angle_factor = t * t * (3.0 - 2.0 * t)

                total_factor = distance_factor * angle_factor

                if total_factor > 0.01:
                    volume_light_dir = -to_surface_dir
                    attenuated_intensity = light["intensity"].squeeze() * total_factor
                    light_dirs.append(volume_light_dir)
                    light_intensities.append(attenuated_intensity)

        return light_dirs, light_intensities

    def apply_volumetric_fog(
        self,
        color: torch.Tensor,
        depth: torch.Tensor,
        position: torch.Tensor,
        fog_color: torch.Tensor,
        params: FogParams,
        light_dirs: list[torch.Tensor] | None = None,
        light_intensity: list[torch.Tensor] | None = None,
        density_scale: float = 1.0,
    ) -> torch.Tensor:
        """Apply volumetric fog with a single-scattering volume model.

        Args:
            color: Linear shaded color, shape ``(3, H, W)``.
            depth: Depth map, shape ``(1, H, W)``.
            position: World positions, shape ``(3, H, W)``.
            fog_color: Fog tint, shape ``(3, 1, 1)``.
            params: Resolved fog coefficients for this renderer.
            light_dirs: Light directions for in-scattering, or ``None``.
            light_intensity: Per-light intensities matching ``light_dirs``.
            density_scale: Runtime density multiplier.

        Returns:
            Fogged linear color, shape ``(3, H, W)``.
        """
        scaled_fog_density = params.fog_density * density_scale
        scaled_absorption = params.absorption * density_scale
        extinction_coeff = scaled_absorption + scaled_fog_density
        beam_transmittance = torch.exp(-extinction_coeff * depth)
        attenuated_color = color * beam_transmittance

        in_scattering = torch.zeros_like(color)
        if light_dirs is not None and light_intensity is not None:
            for light_dir, intensity in zip(light_dirs, light_intensity):
                if light_dir.dim() == 1:
                    light_dir = light_dir.view(3, 1, 1)

                view_dir = F.normalize(-position, dim=0)
                cos_theta = torch.sum(view_dir * light_dir, dim=0, keepdim=True)
                g = _HENYEY_GREENSTEIN_G
                phase = (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * cos_theta) ** 1.5)
                light_contrib = intensity.view(3, 1, 1) * phase * scaled_fog_density * params.scattering
                in_scattering += light_contrib

        volume_scattered_color = attenuated_color + in_scattering
        fog_factor = 1.0 - beam_transmittance
        return torch.lerp(volume_scattered_color, fog_color, fog_factor * 0.5)


class FogRenderer:
    """Fog renderer integrated into the BRDF pipeline."""

    def __init__(self, fog_effects: FogEffects, params: FogParams) -> None:
        """Store fog parameters used during post-shading fog passes.

        Args:
            fog_effects: Shared volumetric fog kernels.
            params: Resolved fog coefficients and tint.
        """
        self.fog_effects = fog_effects
        self.params = params
        self.fog_color = torch.tensor(params.fog_color, device=fog_effects.device).view(3, 1, 1)

    @classmethod
    def from_preset_dict(cls, fog_effects: FogEffects, params: dict[str, Any]) -> FogRenderer:
        """Build a renderer from a merged preset or override dictionary.

        Args:
            fog_effects: Shared volumetric fog kernels.
            params: Flat dict with ``fog_density``, ``scattering``, ``absorption``, ``fog_color``.

        Returns:
            Configured :class:`FogRenderer` instance.
        """
        return cls(fog_effects, _fog_params_from_dict(params))

    def render_with_fog(
        self,
        color: torch.Tensor,
        depth: torch.Tensor,
        position: torch.Tensor,
        lights: list[dict[str, Any]] | None = None,
        depth_max: float = 150.0,
        density_scale: float = 1.0,
    ) -> torch.Tensor:
        """Apply volumetric fog to one shaded frame.

        Args:
            color: Linear shaded color, shape ``(3, H, W)``.
            depth: Depth map, shape ``(1, H, W)``.
            position: World positions, shape ``(3, H, W)``.
            lights: Optional dynamic lights for in-scattering.
            depth_max: Maximum depth used when estimating the surface point for lights.
            density_scale: Runtime density multiplier.

        Returns:
            Fogged linear color, shape ``(3, H, W)``.
        """
        light_dirs: list[torch.Tensor] | None = None
        light_intensity: list[torch.Tensor] | None = None
        if lights is not None:
            position_valid = position[:, (depth < depth_max).squeeze(0)]
            surface_pos = position_valid.mean(dim=1)
            light_dirs, light_intensity = self.fog_effects._process_lights_for_volumetric_fog(
                lights,
                surface_pos,
            )

        return self.fog_effects.apply_volumetric_fog(
            color,
            depth,
            position,
            self.fog_color,
            self.params,
            light_dirs=light_dirs,
            light_intensity=light_intensity,
            density_scale=density_scale,
        )


def _fog_params_from_dict(params: dict[str, Any]) -> FogParams:
    """Build :class:`FogParams` from a preset or override dictionary."""
    fog_color = params["fog_color"]
    return FogParams(
        fog_density=float(params["fog_density"]),
        scattering=float(params["scattering"]),
        absorption=float(params["absorption"]),
        fog_color=(float(fog_color[0]), float(fog_color[1]), float(fog_color[2])),
    )


def create_fog_presets() -> dict[str, dict[str, Any]]:
    """Return named presets for fog rendering configuration.

    Returns:
        Dict mapping preset names to fog parameter dicts.
    """
    return {
        "volumetric_mist": {
            "fog_density": 0.01,
            "scattering": 0.1,
            "absorption": 0.05,
            "fog_color": (0.9, 0.9, 0.92),
        },
        "thick_fog": {
            "fog_density": 0.03,
            "scattering": 0.15,
            "absorption": 0.08,
            "fog_color": (0.85, 0.85, 0.88),
        },
        "glowing_fog": {
            "fog_density": 0.02,
            "scattering": 0.2,
            "absorption": 0.04,
            "fog_color": (0.95, 0.92, 0.88),
        },
    }


__all__ = [
    "FogEffects",
    "FogParams",
    "FogRenderer",
    "create_fog_presets",
]
