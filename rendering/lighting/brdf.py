"""Cook-Torrance BRDF surface shading."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from rendering.gbuffer.material import Material


def _inverse_square_attenuation(
    distances: torch.Tensor,
    light_radius_of_influence: float,
) -> torch.Tensor:
    """Compute smooth inverse-square falloff used by point and spot lights."""
    distances_squared = distances * distances
    light_inv_radius = 1.0 / light_radius_of_influence
    factor = distances_squared * light_inv_radius * light_inv_radius
    smooth_factor = (1 - factor * factor).clamp_min(0.0)
    return (smooth_factor * smooth_factor) / distances_squared.clamp_min(1e-4)


def _directional_light(
    light_dir_or_pos: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    light_dir = F.normalize(-light_dir_or_pos.to(device), dim=0).view(3, 1, 1)
    return light_dir.expand(3, height, width), 1.0


def _point_light(
    light_desc: dict[str, Any],
    position: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    light_pos = light_desc["pos"].to(device).view(3, 1, 1)
    vec = light_pos - position
    distances = torch.norm(vec, dim=0, keepdim=True)
    light_dir_map = vec / (distances + 1e-4)
    attenuation = _inverse_square_attenuation(distances, light_desc["light_radius_of_influence"])
    return light_dir_map, attenuation


def _spot_light(
    light_desc: dict[str, Any],
    position: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    light_pos = light_desc["pos"].to(device).view(3, 1, 1)
    pos_to_light = light_pos - position
    distances = torch.norm(pos_to_light, dim=0, keepdim=True)
    light_dir_map = pos_to_light / (distances + 1e-4)

    attenuation = _inverse_square_attenuation(distances, light_desc["light_radius_of_influence"])

    dir_to_light = F.normalize(-light_desc["dir"].to(device), dim=0).view(3, 1, 1)
    cos_outer = math.cos(light_desc["outer_deg"] / 180.0 * math.pi)
    cos_inner = math.cos(light_desc["inner_deg"] / 180.0 * math.pi)
    spot_scale = 1.0 / torch.clamp(
        torch.tensor([cos_inner - cos_outer], device=device),
        min=1e-4,
    )
    spot_offset = -cos_outer * spot_scale
    cos_theta = torch.sum(light_dir_map * dir_to_light, dim=0, keepdim=True)
    attenuation_spot = torch.clamp((spot_scale * cos_theta + spot_offset), min=0.0, max=1.0).pow(2)
    return light_dir_map, attenuation * attenuation_spot * 100


def _light_dir_and_attenuation(
    light_type: str,
    light_dir_or_pos: torch.Tensor | dict[str, Any],
    position: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | float]:
    if light_type == "directional":
        light_dir_map, attenuation = _directional_light(light_dir_or_pos, height, width, device)
        return light_dir_map, attenuation
    if light_type == "point":
        if not isinstance(light_dir_or_pos, dict) or not {"pos", "light_radius_of_influence"} <= light_dir_or_pos.keys():
            raise ValueError("Point light must have 'pos' and 'light_radius_of_influence' keys")
        return _point_light(light_dir_or_pos, position, device)
    if light_type == "spot":
        required = {"pos", "dir", "inner_deg", "outer_deg", "light_radius_of_influence"}
        if not isinstance(light_dir_or_pos, dict) or not required <= light_dir_or_pos.keys():
            raise ValueError(
                "Spot light must have 'pos', 'dir', 'inner_deg', 'outer_deg' "
                "and 'light_radius_of_influence' keys"
            )
        return _spot_light(light_dir_or_pos, position, device)
    raise ValueError("Invalid light type. Must be 'point', 'directional', or 'spot'")


def _d_ggx(n_dot_h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.clamp(1e-4, 1.0)
    alpha2 = alpha * alpha
    nh2 = n_dot_h * n_dot_h
    denom = nh2 * (alpha2 - 1.0) + 1.0
    return alpha2 / (math.pi * denom * denom + 1e-7)


def _f_schlick(l_dot_h: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    return f0 + (1.0 - f0) * torch.pow(1.0 - l_dot_h, 5.0)


def _v_smith_ggx_correlated(
    n_dot_v: torch.Tensor,
    n_dot_l: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    a2 = (alpha * alpha).clamp_min(1e-8)
    term_v = (-n_dot_v * a2 + n_dot_v) * n_dot_v + a2
    term_l = (-n_dot_l * a2 + n_dot_l) * n_dot_l + a2
    lambda_v = n_dot_l * torch.sqrt(term_v.clamp_min(0.0))
    lambda_l = n_dot_v * torch.sqrt(term_l.clamp_min(0.0))
    return 0.5 / (lambda_v + lambda_l + 1e-6)


def _evaluate_brdf(
    *,
    normal_map: torch.Tensor,
    view_dir: torch.Tensor,
    light_dir_map: torch.Tensor,
    perceptual_roughness: torch.Tensor,
    basecolor: torch.Tensor,
    metallic: torch.Tensor,
    f0: torch.Tensor,
    light_intensity: torch.Tensor,
    attenuation: torch.Tensor | float,
) -> torch.Tensor:
    """Evaluate linear Cook-Torrance radiance for one light."""
    normal = F.normalize(normal_map, dim=0)
    half_vector = F.normalize(light_dir_map + view_dir, dim=0)

    normal_dot_view = torch.abs(torch.sum(normal * view_dir, dim=0, keepdim=True)) + 1e-5
    normal_dot_light = torch.clamp(
        torch.sum(normal * light_dir_map, dim=0, keepdim=True) + 1e-5,
        min=0.0,
        max=1.0,
    )
    light_dot_half = torch.clamp(
        torch.sum(light_dir_map * half_vector, dim=0, keepdim=True) + 1e-5,
        min=0.0,
        max=1.0,
    )

    roughness = perceptual_roughness.pow(2).clamp(1e-4, 1.0)
    distribution = _d_ggx(
        torch.clamp(
            torch.sum(normal * half_vector, dim=0, keepdim=True) + 1e-5,
            min=0.0,
            max=1.0,
        ),
        roughness,
    )
    fresnel = _f_schlick(light_dot_half, f0)
    visibility = _v_smith_ggx_correlated(normal_dot_view, normal_dot_light, roughness)

    specular = (fresnel * distribution * visibility) * normal_dot_light * light_intensity * attenuation
    diffuse = (1 - metallic) * basecolor / math.pi * normal_dot_light * light_intensity * attenuation
    return diffuse + specular


class CookTorranceBRDF(torch.nn.Module):
    """Cook-Torrance BRDF shader for multi-light surface shading.

    Geometry effects must be applied to the material before calling into this
    class. Fog and night post-processing are applied by the pipeline after
    ``render_many``.
    """

    def __init__(self) -> None:
        """Initialize the BRDF shader."""
        super().__init__()

    def render_many(
        self,
        material: Material,
        view_dir: torch.Tensor,
        lights: list[dict[str, Any]] | None,
    ) -> torch.Tensor:
        """Accumulate linear radiance from multiple lights.

        Args:
            material: Per-frame G-buffer maps with geometry effects already applied.
            view_dir: View direction tensor, shape ``(3, H, W)``.
            lights: List of light dicts with ``type``, ``desc``, and ``intensity`` keys.

        Returns:
            Linear radiance, shape ``(3, H, W)``.
        """
        if lights is None:
            return material.linear_albedo()

        device = material.device
        final_accum = torch.zeros_like(material.linear_albedo(), device=device)
        for light in lights:
            final_accum = final_accum + self._shade_one_light(
                light["type"],
                material,
                view_dir,
                light["desc"],
                light["intensity"],
            )
        return final_accum

    def _shade_one_light(
        self,
        light_type: str,
        material: Material,
        view_dir: torch.Tensor,
        light_dir_or_pos: torch.Tensor | dict[str, Any],
        light_intensity: torch.Tensor,
    ) -> torch.Tensor:
        device = material.device
        view_dir = F.normalize(view_dir.to(device), dim=0)
        light_intensity = light_intensity.to(device).view(3, 1, 1)

        perceptual_roughness = material.roughness.to(device)
        normal_map = material.normal.to(device)
        basecolor = material.linear_albedo().to(device)
        metallic = material.metallic.to(device)
        position = material.position.to(device)
        f0 = torch.lerp(torch.full_like(basecolor, 0.04), basecolor, metallic)

        _, height, width = basecolor.shape
        assert view_dir.shape == basecolor.shape

        light_dir_map, attenuation = _light_dir_and_attenuation(
            light_type,
            light_dir_or_pos,
            position,
            height,
            width,
            device,
        )

        return _evaluate_brdf(
            normal_map=normal_map,
            view_dir=view_dir,
            light_dir_map=light_dir_map,
            perceptual_roughness=perceptual_roughness,
            basecolor=basecolor,
            metallic=metallic,
            f0=f0,
            light_intensity=light_intensity,
            attenuation=attenuation,
        )
