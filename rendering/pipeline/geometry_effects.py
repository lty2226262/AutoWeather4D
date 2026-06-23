"""Rain/snow geometry effects applied to G-buffer material before export."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from rendering.effects.geometry.rain_effects import RainPuddleSimulator
from rendering.effects.geometry.snow_config import SnowModifierConfig
from rendering.effects.geometry.snow_effects import SnowGBufferModifierSurfaceBRDF
from rendering.gbuffer.material import Material, MaterialVideo


def _prewarm_world_falling_snow(
    modifier: SnowGBufferModifierSurfaceBRDF,
    material_video: MaterialVideo,
) -> None:
    """Seed world-space falling-snow particles from the first frame pose."""
    if not (modifier.snowfall_enabled and modifier.snowfall_use_world_particles):
        return
    if len(material_video) == 0:
        return
    modifier.init_falling_snow_particles(
        num_particles=modifier.falling_snow_num,
        box_size=modifier.falling_snow_box_size,
        gravity=modifier.falling_snow_gravity,
        wind_x=modifier.falling_snow_wind_x,
        wind_z=modifier.falling_snow_wind_z,
        radius=modifier.falling_snow_radius,
        seed=modifier.falling_snow_seed,
        pose=material_video[0].pose,
    )


def _camera_inv_numpy(pose: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return world-to-camera rotation and translation as numpy arrays."""
    pose_inv = torch.linalg.inv(pose)
    return (
        pose_inv[:3, :3].detach().cpu().numpy(),
        pose_inv[:3, 3:4].detach().cpu().numpy(),
    )


def _ensure_falling_snow_particles(
    modifier: SnowGBufferModifierSurfaceBRDF,
    pose: torch.Tensor,
) -> np.ndarray:
    """Advance falling-snow simulation and return the current particle cloud."""
    if modifier.falling_snow_particles is None:
        modifier.init_falling_snow_particles(
            num_particles=modifier.falling_snow_num,
            box_size=modifier.falling_snow_box_size,
            gravity=modifier.falling_snow_gravity,
            wind_x=modifier.falling_snow_wind_x,
            wind_z=modifier.falling_snow_wind_z,
            radius=modifier.falling_snow_radius,
            seed=modifier.falling_snow_seed,
            pose=pose,
        )
    modifier.update_falling_snow_particles(pose)
    return modifier.falling_snow_particles


def _project_visible_particles(
    particles: np.ndarray,
    R_inv: np.ndarray,
    t_inv: np.ndarray,
    intrinsics: np.ndarray,
    base_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Project world particles to the image plane, keeping only in-front points."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    pc = (R_inv @ particles.T + t_inv).T
    pc = pc[pc[:, 2] > 0.0]
    if pc.shape[0] == 0:
        return None

    x_c, y_c, z_c = pc[:, 0], pc[:, 1], pc[:, 2]
    inv_z = 1.0 / np.maximum(z_c, 1e-3)
    u = fx * (x_c * inv_z) + cx
    v = fy * (y_c * inv_z) + cy
    r_px = np.clip(base_radius * fx * inv_z, 2.0, 20.0)
    return pc, u, v, r_px


def _rasterize_particle_mask(
    height: int,
    width: int,
    pc: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    r_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize projected particles into a coverage mask and depth buffer."""
    mask = np.zeros((height, width), dtype=np.float32)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)

    order = np.argsort(pc[:, 2])
    for i in order:
        ui, vi = int(round(u[i])), int(round(v[i]))
        if ui < 0 or ui >= width or vi < 0 or vi >= height:
            continue
        ri = int(max(1, round(r_px[i])))
        zi = pc[i, 2]
        cv2.circle(mask, (ui, vi), ri, 1.0, -1)
        y_min, y_max = max(0, vi - ri), min(height, vi + ri + 1)
        x_min, x_max = max(0, ui - ri), min(width, ui + ri + 1)
        for yy in range(y_min, y_max):
            for xx in range(x_min, x_max):
                if (xx - ui) ** 2 + (yy - vi) ** 2 <= ri**2 and zi < zbuf[yy, xx]:
                    zbuf[yy, xx] = zi

    return np.clip(mask, 0.0, 1.0), zbuf


def _mask_out_no_snow_regions(
    snow_mask: torch.Tensor,
    material: Material,
    height: int,
    width: int,
    snow_amount: float,
    car_mask_threshold: float,
) -> torch.Tensor:
    """Zero snowfall coverage over vehicle pixels when snow amount is low."""
    if snow_amount > car_mask_threshold or material.no_snow_mask is None:
        return snow_mask

    car_mask = material.no_snow_mask
    if isinstance(car_mask, np.ndarray):
        car_mask = torch.from_numpy(car_mask).to(snow_mask.device)
    else:
        car_mask = torch.as_tensor(car_mask, device=snow_mask.device)
    if car_mask.shape != (height, width):
        return snow_mask
    return snow_mask * (~car_mask.bool()).to(snow_mask.dtype)


def _blend_snowfall_into_material(
    material: Material,
    snow_mask: torch.Tensor,
    snow_zbuf: torch.Tensor,
    modifier: SnowGBufferModifierSurfaceBRDF,
) -> None:
    """Write falling-snow coverage into per-frame G-buffer maps."""
    if modifier.snowfall_depth_test:
        current_depth = torch.norm(material.position, dim=0)
        valid_mask = (snow_zbuf < current_depth) & (snow_zbuf < np.inf)
        snow_mask = snow_mask * valid_mask.float()

    mask3 = snow_mask.unsqueeze(0)
    alpha = modifier.snowfall_opacity
    white = torch.ones_like(material.albedo)
    material.albedo = material.albedo * (1.0 - alpha * mask3) + white * (alpha * mask3)

    target_r = torch.tensor(
        modifier.snowfall_roughness,
        device=material.roughness.device,
        dtype=material.roughness.dtype,
    )
    material.roughness = torch.clamp(
        material.roughness * (1.0 - mask3) + target_r * mask3,
        0.0,
        1.0,
    )

    target_m = torch.zeros_like(material.metallic)
    material.metallic = material.metallic * (1.0 - mask3) + target_m * mask3

    if modifier.snowfall_affect_normal:
        snow_normal = torch.zeros_like(material.normal)
        snow_normal[2] = 1.0
        material.normal = F.normalize(
            material.normal * (1.0 - mask3) + snow_normal * mask3,
            dim=0,
        )

    if not modifier.snowfall_affect_depth:
        return

    snow_depth_mask = (snow_mask > 0.5) & (snow_zbuf < np.inf)
    if not snow_depth_mask.any():
        return

    current_depth = torch.norm(material.position, dim=0)
    depth_ratio = torch.where(
        snow_depth_mask & (current_depth > 1e-6),
        snow_zbuf / current_depth.clamp(min=1e-6),
        torch.ones_like(current_depth),
    )
    depth_ratio_3 = depth_ratio.unsqueeze(0).expand_as(material.position)
    material.position = torch.where(
        snow_depth_mask.unsqueeze(0).expand_as(material.position),
        material.position * depth_ratio_3,
        material.position,
    )


class GeometryEffectsManager:
    """Manager for rain/snow geometry modifications on per-frame G-buffer data."""

    def __init__(self, device: str = "cuda") -> None:
        """Initialize geometry effect state for the given torch device.

        Args:
            device: Torch device name, e.g. ``"cuda"`` or ``"cpu"``.
        """
        self.device = torch.device(device)
        self.snow_modifier: SnowGBufferModifierSurfaceBRDF | None = None
        self.snow_initialized = False
        self.rain_simulator: RainPuddleSimulator | None = None
        self.rain_initialized = False

    def initialize_snow(
        self,
        material_video: MaterialVideo,
        max_points: int = 100000,
        preset: str = "moderate_snow",
        dataset_id: int | None = None,
        **snow_params: Any,
    ) -> GeometryEffectsManager:
        """Initialize snow accumulation and optional falling-snow particles.

        Args:
            material_video: Scene material source used to build snow coverage.
            max_points: Maximum metaball points for snow accumulation.
            preset: Snow preset name, e.g. ``moderate_snow``.
            dataset_id: Optional dataset id for ground detection.
            **snow_params: Extra preset overrides.

        Returns:
            ``self`` for chaining.
        """
        config = SnowModifierConfig.from_preset(
            preset,
            material_video,
            device=str(self.device),
            dataset_id=dataset_id,
            **snow_params,
        )
        self.snow_modifier = SnowGBufferModifierSurfaceBRDF(config)
        self.snow_modifier.initialize_from_all_frames(material_video, max_points=max_points)
        _prewarm_world_falling_snow(self.snow_modifier, material_video)

        self.snow_initialized = self.snow_modifier.is_initialized
        return self

    def initialize_rain(
        self,
        material_video: MaterialVideo,
        dt: float = 0.1,
        raindrop_count: int = 10000,
        **rain_params: Any,
    ) -> GeometryEffectsManager:
        """Initialize rain puddle and ripple simulation.

        Args:
            material_video: Scene material source used by the rain simulator.
            dt: Simulation timestep.
            raindrop_count: Number of raindrops to simulate.
            **rain_params: Extra rain simulator overrides.

        Returns:
            ``self`` for chaining.
        """
        self.rain_simulator = RainPuddleSimulator(
            material_video,
            raindrop_count=raindrop_count,
            dt=dt,
            **{k: v for k, v in rain_params.items() if k not in ("dt", "raindrop_count")},
        )
        self.rain_initialized = True
        return self

    def apply_snow(
        self,
        material: Material,
        snow_amount: float = 1.0,
        interval: float | None = None,
    ) -> None:
        """Apply snow geometry changes to one frame's material buffers.

        Args:
            material: Mutable per-frame material buffers.
            snow_amount: Snow accumulation amount.
            interval: Optional snow interval override.
        """
        if not self.snow_initialized or self.snow_modifier is None:
            return
        if snow_amount <= 0:
            return

        modifier = self.snow_modifier
        modifier.modify_gbuffer_for_snow(
            material,
            snow_amount=snow_amount,
            interval=interval or modifier.interval,
        )

        if modifier.snowfall_enabled and modifier.falling_snow_num > 0:
            self._apply_falling_snow(modifier, material, snow_amount)

    def apply_rain(
        self,
        frame_idx: int = 0,
        time: float = 0.0,
    ) -> None:
        """Apply rain geometry changes for one frame.

        Rain updates the shared ``MaterialVideo`` buffers in place through the
        simulator.

        Args:
            frame_idx: Zero-based frame index.
            time: Simulation time for ripple animation.
        """
        if not self.rain_initialized or self.rain_simulator is None:
            return
        self.rain_simulator.process_frame(frame_idx, time)

    def _apply_falling_snow(
        self,
        modifier: SnowGBufferModifierSurfaceBRDF,
        material: Material,
        snow_amount: float,
    ) -> None:
        """Rasterize falling snow particles into the current material buffers."""
        pose = material.pose
        height, width = material.albedo.shape[1], material.albedo.shape[2]

        particles = _ensure_falling_snow_particles(modifier, pose)
        R_inv, t_inv = _camera_inv_numpy(pose)
        intrinsics = material.intrinsics.detach().cpu().numpy()

        projected = _project_visible_particles(
            particles,
            R_inv,
            t_inv,
            intrinsics,
            float(modifier.falling_snow_radius),
        )
        if projected is None:
            return

        pc, u, v, r_px = projected
        mask, zbuf = _rasterize_particle_mask(height, width, pc, u, v, r_px)
        device = material.albedo.device
        dtype = material.albedo.dtype
        snow_mask = torch.from_numpy(mask).to(device=device, dtype=dtype)
        snow_zbuf = torch.from_numpy(zbuf).to(device=device, dtype=dtype)
        snow_mask = _mask_out_no_snow_regions(
            snow_mask,
            material,
            height,
            width,
            snow_amount,
            modifier.car_mask_max_snow_amount,
        )
        _blend_snowfall_into_material(material, snow_mask, snow_zbuf, modifier)
