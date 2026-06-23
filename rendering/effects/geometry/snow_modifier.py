"""Snow G-buffer modifier: metaball accumulation and ground snow blending."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from rendering.effects.geometry.rain_effects import HeightMapGenerator
from rendering.effects.geometry.snow_falling import SnowFallingParticlesMixin
from rendering.effects.geometry.snow_grid import SnowGridMixin
from rendering.effects.geometry.snow_config import SnowModifierConfig, bind_snow_config
from rendering.effects.geometry.snow_kernels import (
    dw_poly6_dr,
    ground_method_for_dataset,
    load_ground_method_config,
    w_poly6,
    weighted_sigmoid,
)
from rendering.gbuffer.material import Material, MaterialVideo


@dataclass
class _SnowFrameContext:
    """Mutable per-frame buffers and index masks used during snow blending."""

    pose: torch.Tensor
    img_height: int
    img_width: int
    snow_amt: float
    ground_mask: torch.Tensor
    cand_indices_all: torch.Tensor
    non_sky_mask_cand: torch.Tensor
    non_sky_non_car_cand: torch.Tensor
    ground_indices_flat: torch.Tensor
    non_ground_indices_flat: torch.Tensor
    sky_mask_flat: torch.Tensor
    car_mask_flat: torch.Tensor
    pos_cand: torch.Tensor
    n0_cand: torch.Tensor
    pos_cand_world_all: torch.Tensor
    albedo_flat: torch.Tensor
    rough_flat: torch.Tensor
    metal_flat: torch.Tensor
    normal_flat: torch.Tensor
    pos_flat: torch.Tensor


class SnowGBufferModifierSurfaceBRDF(SnowFallingParticlesMixin, SnowGridMixin):
    """Metaball snow accumulation with optional grid-based ground coverage."""

    def __init__(self, config: SnowModifierConfig) -> None:
        """Configure snow simulation from a nested :class:`SnowModifierConfig`."""
        bind_snow_config(self, config)

        self._selected_ground_method = 1
        self._ground_method_config = load_ground_method_config()

        self.global_metaball_centers = None
        self.global_metaball_densities = None
        self.height_map_generator = None
        self.is_initialized = False

        self.falling_snow_particles = None
        self.falling_snow_scene_center = None

        self.grid_snow_initialized = False
        self.grid_snow_bounds = None
        self.grid_snow_map = None

    def initialize_from_all_frames(
        self,
        urban_scene: MaterialVideo,
        max_points: int = 1000,
    ) -> None:
        """Build global metaball centers from horizontal non-ground points.

        Args:
            urban_scene: Source scene with per-frame material and depth data.
            max_points: Maximum number of metaball centers to retain.
        """
        self.height_map_generator = HeightMapGenerator(urban_scene, visualization=False)

        perframe_world = self.height_map_generator.perframe_world
        ground_labels1 = self.height_map_generator.ground_labels
        ground_labels2 = self.height_map_generator.ground_labels2
        avail_masks = self.height_map_generator.avail_masks
        total_frames = len(perframe_world)

        dataset_id = self.dataset_id
        if dataset_id is not None:
            method = ground_method_for_dataset(dataset_id, self._ground_method_config)
            if method == 2:
                ground_labels = ground_labels2
                self._selected_ground_method = 2
            else:
                ground_labels = ground_labels1
                self._selected_ground_method = 1
        else:
            ground_labels = ground_labels1
            self._selected_ground_method = 1

        all_horizontal_points = []

        pbar = tqdm(total=total_frames, desc="Initializing snow from frames", unit="frame")
        for frame_idx in range(total_frames):
            world_positions = perframe_world[frame_idx].to(self.device)
            ground_mask = ground_labels[frame_idx].to(self.device)
            mask2d = avail_masks[frame_idx].to(self.device)

            material = urban_scene[frame_idx]
            normal_y = material.normal[1, :, :]

            horizontal_mask = (normal_y < -0.8) & mask2d
            if not horizontal_mask.any():
                pbar.update(1)
                continue

            flat_mask2d = mask2d.flatten()
            flat_horizontal = horizontal_mask.flatten()
            valid_in_mask2d = flat_horizontal[flat_mask2d]
            if not valid_in_mask2d.any():
                pbar.update(1)
                continue

            horizontal_world_positions = world_positions[:, valid_in_mask2d]
            horizontal_ground_mask = ground_mask[valid_in_mask2d]

            non_ground = ~horizontal_ground_mask
            if non_ground.any():
                pts = horizontal_world_positions[:, non_ground].t()
                all_horizontal_points.append(pts)

            pbar.update(1)

        pbar.close()

        if not all_horizontal_points:
            self.is_initialized = False
            return

        all_points = torch.cat(all_horizontal_points, dim=0)

        torch.manual_seed(42)
        max_metaballs = min(max_points, all_points.shape[0])
        if all_points.shape[0] > max_metaballs:
            perm = torch.randperm(all_points.shape[0], device=self.device)[:max_metaballs]
            self.global_metaball_centers = all_points[perm].contiguous()
        else:
            self.global_metaball_centers = all_points.contiguous()

        base_density = torch.ones(self.global_metaball_centers.shape[0], device=self.device)
        seed = (self.global_metaball_centers * 137.0).sin().sum(dim=1)
        jitter = 0.8 + 0.4 * ((seed + 1.0) * 0.5)
        self.global_metaball_densities = base_density * jitter

        world_seed = (self.global_metaball_centers * 1000.0).long() % 2147483647
        noise_y = torch.sin(world_seed[:, 1].float() * 0.1) * 0.08
        self.global_metaball_centers[:, 1] -= noise_y

        self.is_initialized = True

        self.ensure_grid_snow_initialized()

    def _get_ground_snow_coverage_scale(self, snow_amount: float) -> float:
        """Map snow amount to ground coverage scale (0.1 -> ~0.4, 1.0 -> 1.0)."""
        snow_amt = float(snow_amount)
        if snow_amt <= 0:
            return 0.0
        if snow_amt <= 0.1:
            return 0.4
        scale = min(1.0, snow_amt**0.6)
        return max(0.4, scale)

    def _get_metaball_snow_albedo_scale(self, snow_amount: float) -> float:
        """Map snow amount to metaball albedo whitening strength."""
        snow_amt = float(snow_amount)
        if snow_amt <= 0:
            return 0.0
        scale = min(1.0, snow_amt**0.3)
        return max(0.75, scale)

    def _should_skip_snow(self, snow_amount: float | torch.Tensor) -> bool:
        if torch.is_tensor(snow_amount) and snow_amount.numel() > 1:
            return bool((snow_amount <= 0).all())
        return snow_amount <= 0

    def _resolve_snow_amount(self, snow_amount: float | torch.Tensor) -> float:
        if torch.is_tensor(snow_amount) and snow_amount.numel() > 1:
            return float(snow_amount.max())
        return float(snow_amount)

    def _build_snow_frame_context(
        self,
        material: Material,
        snow_amount: float | torch.Tensor,
    ) -> _SnowFrameContext | None:
        """Collect candidate pixels, masks, and flattened G-buffer maps."""
        pose = material.pose
        img_height, img_width = material.position.shape[1:]
        snow_amt = self._resolve_snow_amount(snow_amount)

        normal_cam = material.normal.permute(1, 2, 0).reshape(-1, 3)
        rotation = pose[:3, :3]
        normal_world = (rotation @ normal_cam.T).T
        if self.non_grid_disable_normal_filter and (not self.grid_snow_enabled):
            cand_mask_flat = torch.ones(img_height * img_width, dtype=torch.bool, device=self.device)
        else:
            cand_mask_flat = normal_world[:, 1] < -0.5
        if not cand_mask_flat.any():
            return None

        pos = material.position.permute(1, 2, 0).reshape(-1, 3)
        pos_cand = pos[cand_mask_flat]
        n0_cand = normal_cam[cand_mask_flat]

        pos_cand_world_all = (pose[:3, :3] @ pos_cand.T) + pose[:3, 3:4]
        if self._selected_ground_method == 2:
            ground_mask = HeightMapGenerator.get_ground2(pos_cand_world_all)
        else:
            ground_mask = HeightMapGenerator.get_ground(pos_cand_world_all)

        cand_indices_all = cand_mask_flat.nonzero(as_tuple=False).squeeze(1)
        use_car_mask = snow_amt <= self.car_mask_max_snow_amount

        car_mask_flat = torch.zeros(img_height * img_width, dtype=torch.bool, device=self.device)
        if use_car_mask and material.no_snow_mask is not None:
            car_mask = material.no_snow_mask
            if isinstance(car_mask, np.ndarray):
                car_mask = torch.from_numpy(car_mask).to(self.device)
            elif not isinstance(car_mask, torch.Tensor):
                car_mask = torch.as_tensor(car_mask, device=self.device, dtype=torch.bool)
            else:
                car_mask = car_mask.to(self.device)
            if car_mask.shape == (img_height, img_width):
                car_mask_flat = car_mask.flatten()

        sky_mask_flat = torch.zeros(img_height * img_width, dtype=torch.bool, device=self.device)
        if material.sky_mask is not None:
            sky_mask_2d = material.sky_mask
            if isinstance(sky_mask_2d, np.ndarray):
                sky_mask_2d = torch.from_numpy(sky_mask_2d).to(self.device)
            elif not isinstance(sky_mask_2d, torch.Tensor):
                sky_mask_2d = torch.tensor(sky_mask_2d, device=self.device)
            else:
                sky_mask_2d = sky_mask_2d.to(self.device)
            if sky_mask_2d.shape[0] == img_height and sky_mask_2d.shape[1] == img_width:
                sky_mask_flat = sky_mask_2d.flatten()

        sky_mask_cand = sky_mask_flat[cand_indices_all]
        car_mask_cand = car_mask_flat[cand_indices_all]
        non_sky_mask_cand = ~sky_mask_cand
        non_sky_non_car_cand = non_sky_mask_cand & (~car_mask_cand)

        ground_indices_flat = cand_indices_all[ground_mask & non_sky_mask_cand]
        non_ground_indices_flat = cand_indices_all[(~ground_mask) & non_sky_non_car_cand]

        ctx = _SnowFrameContext(
            pose=pose,
            img_height=img_height,
            img_width=img_width,
            snow_amt=snow_amt,
            ground_mask=ground_mask,
            cand_indices_all=cand_indices_all,
            non_sky_mask_cand=non_sky_mask_cand,
            non_sky_non_car_cand=non_sky_non_car_cand,
            ground_indices_flat=ground_indices_flat,
            non_ground_indices_flat=non_ground_indices_flat,
            sky_mask_flat=sky_mask_flat,
            car_mask_flat=car_mask_flat,
            pos_cand=pos_cand,
            n0_cand=n0_cand,
            pos_cand_world_all=pos_cand_world_all,
            albedo_flat=material.albedo.permute(1, 2, 0).reshape(-1, 3),
            rough_flat=material.roughness.permute(1, 2, 0).reshape(-1, 1),
            metal_flat=material.metallic.permute(1, 2, 0).reshape(-1, 1),
            normal_flat=material.normal.permute(1, 2, 0).reshape(-1, 3),
            pos_flat=material.position.permute(1, 2, 0).reshape(-1, 3),
        )
        self._expand_ground_indices(ctx)
        return ctx

    def _expand_ground_indices(self, ctx: _SnowFrameContext) -> None:
        """Apply non-grid ground index fallbacks when geometric detection is sparse."""
        pose = ctx.pose
        snow_amt = ctx.snow_amt
        ground_indices_flat = ctx.ground_indices_flat

        if self.non_grid_force_ground_snow and (not self.grid_snow_enabled):
            pos_all_world = (pose[:3, :3] @ ctx.pos_flat.T + pose[:3, 3:4]).T
            non_sky_non_car_all = (~ctx.sky_mask_flat) & (~ctx.car_mask_flat)
            if non_sky_non_car_all.any():
                y_vals = pos_all_world[:, 1]
                quantile = min(0.8, max(0.2, float(self.non_grid_ground_quantile)))
                y_q = torch.quantile(y_vals[non_sky_non_car_all], quantile)
                y_band = y_q + max(0.35, 0.25 * snow_amt)
                forced_ground = non_sky_non_car_all & (y_vals <= y_band)
                if forced_ground.sum() > 1024:
                    ground_indices_flat = forced_ground.nonzero(as_tuple=False).squeeze(1)

        if not self.grid_snow_enabled:
            pos_all_world = (pose[:3, :3] @ ctx.pos_flat.T + pose[:3, 3:4]).T
            non_sky_non_car_all = (~ctx.sky_mask_flat) & (~ctx.car_mask_flat)
            fallback_needed = False
            if non_sky_non_car_all.any():
                total_valid = int(non_sky_non_car_all.sum().item())
                ground_valid = int(ground_indices_flat.numel())
                fallback_needed = (ground_valid < 2048) or (ground_valid < int(0.12 * max(1, total_valid)))
            if non_sky_non_car_all.any() and fallback_needed:
                y_vals = pos_all_world[:, 1]
                y_valid = y_vals[non_sky_non_car_all]
                if y_valid.numel() > 0:
                    y_q = torch.quantile(y_valid, 0.25)
                    y_band = y_q + max(0.25, 0.25 * float(snow_amt))
                    fallback_ground = non_sky_non_car_all & (y_vals <= y_band)
                    if fallback_ground.sum() > 512:
                        fallback_indices = fallback_ground.nonzero(as_tuple=False).squeeze(1)
                        if ground_indices_flat.numel() > 0:
                            ground_indices_flat = torch.unique(
                                torch.cat([ground_indices_flat, fallback_indices], dim=0)
                            )
                        else:
                            ground_indices_flat = fallback_indices

        ctx.ground_indices_flat = ground_indices_flat

    def _apply_ground_snow(
        self,
        ctx: _SnowFrameContext,
        interval: float | None,
    ) -> torch.Tensor:
        """Blend grid or non-grid ground snow and return unsnowed ground indices."""
        ground_indices_flat = ctx.ground_indices_flat
        snow_amt = ctx.snow_amt
        pose = ctx.pose

        if self.grid_snow_enabled and self.grid_snow_initialized and ground_indices_flat.numel() > 0:
            ground_non_sky_mask = ctx.ground_mask & ctx.non_sky_mask_cand
            ground_world_pos = ctx.pos_cand_world_all[:, ground_non_sky_mask].T
            grid_snow_heights = self.sample_grid_snow(ground_world_pos)
            snow_coverage = torch.clamp(
                grid_snow_heights / max(1e-6, float(self.grid_snow_height)), 0.0, 1.0
            )
            ground_scale = self._get_ground_snow_coverage_scale(snow_amt)
            snow_coverage = snow_coverage * ground_scale

            snow_mask = snow_coverage > float(self.grid_snow_eps)
            if snow_mask.any():
                gi = ground_indices_flat[snow_mask]
                cov = snow_coverage[snow_mask].unsqueeze(1)
                snow_albedo = torch.full((gi.shape[0], 3), float(self.grid_snow_albedo), device=self.device)
                snow_rough = torch.full((gi.shape[0], 1), float(self.grid_snow_roughness), device=self.device)
                snow_metal = torch.full((gi.shape[0], 1), float(self.grid_snow_metallic), device=self.device)

                transparency_factor = 0.9
                adjusted_cov = cov * transparency_factor

                base_alb = ctx.albedo_flat[gi]
                base_rgh = ctx.rough_flat[gi]
                base_met = ctx.metal_flat[gi]
                base_norm = ctx.normal_flat[gi]

                snow_strength = torch.clamp(adjusted_cov * 2.0, 0.0, 1.0)
                ctx.albedo_flat[gi] = torch.lerp(base_alb, snow_albedo, snow_strength.expand_as(base_alb))
                ctx.rough_flat[gi] = torch.lerp(base_rgh, snow_rough, adjusted_cov)
                ctx.metal_flat[gi] = torch.lerp(base_met, snow_metal, adjusted_cov)

                if cov.numel() > 0:
                    pos_world = ctx.pos_flat[gi]
                    noise_x = torch.sin(pos_world[:, 0] * 10.0) * 0.02
                    noise_z = torch.sin(pos_world[:, 2] * 10.0) * 0.02
                    noise_y = torch.sin(pos_world[:, 1] * 10.0) * 0.01

                    noise_vec = torch.stack([noise_x, noise_y, noise_z], dim=1)
                    noise_tan = noise_vec - (noise_vec * base_norm).sum(-1, keepdim=True) * base_norm
                    noise_tan = F.normalize(noise_tan + 1e-8, dim=1)

                    normal_perturb = base_norm - adjusted_cov * noise_tan * 0.1
                    ctx.normal_flat[gi] = F.normalize(normal_perturb, dim=1)

                return ground_indices_flat[~snow_mask]
            return ground_indices_flat

        if (not self.grid_snow_enabled) and ground_indices_flat.numel() > 0:
            ground_world_pos = (pose[:3, :3] @ ctx.pos_flat[ground_indices_flat].T + pose[:3, 3:4]).T
            ground_scale = self._get_ground_snow_coverage_scale(snow_amt)

            centers_world = self.global_metaball_centers
            densities = self.global_metaball_densities
            if (
                centers_world is not None
                and densities is not None
                and centers_world.numel() > 0
                and densities.numel() > 0
            ):
                m_count = centers_world.shape[0]
                k = min(max(4, self.k_neighbors), m_count)
                chunk = 8192
                influence = torch.zeros(ground_world_pos.shape[0], device=self.device)
                base_interval = float(self.interval if interval is None else interval)
                base_interval = max(base_interval, 1e-4)
                snow_gain = min(2.5, max(0.6, snow_amt))

                for s in range(0, ground_world_pos.shape[0], chunk):
                    e = min(s + chunk, ground_world_pos.shape[0])
                    p_w = ground_world_pos[s:e]
                    dist = torch.cdist(p_w, centers_world)
                    topd, topi = torch.topk(dist, k, dim=1, largest=False)
                    a_k = torch.clamp(densities[topi], min=0.0) * snow_gain
                    local = (a_k / (1.0 + (topd / (base_interval * 2.2)) ** 2)).mean(dim=1)
                    influence[s:e] = local

                snow_coverage = torch.clamp(influence * (0.6 + 0.4 * ground_scale), 0.0, 1.0)
            else:
                snow_coverage = torch.full(
                    (ground_indices_flat.shape[0],),
                    min(1.0, 0.85 * max(0.4, ground_scale)),
                    device=self.device,
                )

            snow_mask = snow_coverage > 0.06
            if snow_mask.any():
                gi = ground_indices_flat[snow_mask]
                cov = snow_coverage[snow_mask].unsqueeze(1)
                snow_albedo = torch.full((gi.shape[0], 3), float(self.snow_albedo_value), device=self.device)
                snow_rough = torch.full((gi.shape[0], 1), float(self.snow_roughness_value), device=self.device)
                snow_metal = torch.zeros((gi.shape[0], 1), device=self.device)

                strong_cov = torch.clamp(cov * (1.45 + 0.15 * min(1.0, snow_amt / 2.0)), 0.0, 1.0)
                base_alb = ctx.albedo_flat[gi]
                base_rgh = ctx.rough_flat[gi]
                base_met = ctx.metal_flat[gi]
                ctx.albedo_flat[gi] = torch.lerp(base_alb, snow_albedo, strong_cov.expand_as(base_alb))
                ctx.rough_flat[gi] = torch.lerp(base_rgh, snow_rough, strong_cov)
                ctx.metal_flat[gi] = torch.lerp(base_met, snow_metal, strong_cov)
                return ground_indices_flat[~snow_mask]
            return ground_indices_flat

        return ground_indices_flat

    def _apply_metaball_snow(
        self,
        ctx: _SnowFrameContext,
        snow_amount: float | torch.Tensor,
        interval: float | None,
    ) -> None:
        """Apply metaball snow coverage to non-ground candidate pixels."""
        non_ground_non_sky_mask = (~ctx.ground_mask) & ctx.non_sky_non_car_cand
        pos_cand_non_ground = ctx.pos_cand[non_ground_non_sky_mask]
        n0_cand_non_ground = ctx.n0_cand[non_ground_non_sky_mask]
        cand_indices_flat = ctx.non_ground_indices_flat
        nc = pos_cand_non_ground.shape[0]

        if nc == 0:
            return

        centers_world = self.global_metaball_centers
        densities = self.global_metaball_densities

        pose = ctx.pose
        pos_cand_non_ground_world = (pose[:3, :3] @ pos_cand_non_ground.T + pose[:3, 3:4]).T
        m_count = centers_world.shape[0]
        if m_count == 0:
            return

        k = min(self.k_neighbors, m_count)
        chunk = 8192
        height_all = torch.zeros(nc, device=self.device)
        grad_all = torch.zeros(nc, 3, device=self.device)

        base_interval = self.interval if interval is None else interval
        eps = (torch.arange(m_count, device=self.device, dtype=torch.float32) / m_count) * 1e-6
        max_centers_per_chunk = 50000

        for s in range(0, nc, chunk):
            e = min(s + chunk, nc)
            p_w = pos_cand_non_ground_world[s:e]

            if m_count > max_centers_per_chunk:
                all_topd = []
                all_topi = []
                for m_start in range(0, m_count, max_centers_per_chunk):
                    m_end = min(m_start + max_centers_per_chunk, m_count)
                    centers_chunk = centers_world[m_start:m_end]
                    eps_chunk = eps[m_start:m_end]

                    dist_chunk = torch.cdist(p_w, centers_chunk) + eps_chunk.unsqueeze(0)
                    topd_chunk, topi_chunk = torch.topk(dist_chunk, min(k, m_end - m_start), dim=1, largest=False)
                    topi_chunk = topi_chunk + m_start

                    all_topd.append(topd_chunk)
                    all_topi.append(topi_chunk)

                all_topd = torch.cat(all_topd, dim=1)
                all_topi = torch.cat(all_topi, dim=1)
                topd, sort_idx = torch.topk(all_topd, k, dim=1, largest=False)
                topi = torch.gather(all_topi, 1, sort_idx)
            else:
                dist = torch.cdist(p_w, centers_world) + eps.unsqueeze(0)
                topd, topi = torch.topk(dist, k, dim=1, largest=False)

            c_k = centers_world[topi]
            a_k = densities[topi] * snow_amount

            r_vec_w = p_w.unsqueeze(1) - c_k
            r = torch.norm(r_vec_w, dim=2)
            dir_hat_w = torch.where(
                (r > 1e-8).unsqueeze(-1), r_vec_w / (r.unsqueeze(-1) + 1e-8), torch.zeros_like(r_vec_w)
            )

            r_min = r[:, 0:1].clamp(min=1e-3)
            scale = torch.clamp(r_min / base_interval, min=1.0) * self.dynamic_radius_scale

            mb_height = torch.zeros(p_w.shape[0], device=self.device)
            mb_grad_w = torch.zeros(p_w.shape[0], 3, device=self.device)
            for level in range(self.mb_cascade):
                r_l = (base_interval / (self.b**level)) * scale
                amp_l = self.amp_decay**level
                wk = w_poly6(a_k * amp_l, r_l, r)
                mb_height = mb_height + wk.sum(dim=1)
                dw = dw_poly6_dr(a_k * amp_l, r_l, r)
                mb_grad_w = mb_grad_w + (dw.unsqueeze(-1) * dir_hat_w).sum(dim=1)

            mb_height = mb_height * self.height_scale
            mb_grad_w = mb_grad_w * self.normal_slope_scale

            r_inv = pose[:3, :3].T
            grad_c = (r_inv @ mb_grad_w.T).T

            height_all[s:e] = mb_height
            grad_all[s:e] = grad_c

        cover_soft = weighted_sigmoid(height_all, self.blend_weight, self.blend_bias).clamp(0.0, 1.0)

        if abs(float(self.cover_gamma) - 1.0) > 1e-6:
            cover_soft = torch.clamp(cover_soft, 0.0, 1.0) ** float(self.cover_gamma)

        if self.cover_hard:
            cover_albedo = (cover_soft > self.cover_threshold).to(cover_soft.dtype)
        else:
            cover_albedo = cover_soft

        cover_normal = cover_soft

        grad_tan = grad_all - (grad_all * n0_cand_non_ground).sum(-1, keepdim=True) * n0_cand_non_ground
        gmag = torch.norm(grad_tan, dim=1)
        need_boost = (cover_normal > float(self.normal_min_cover)) & (gmag < float(self.normal_min_slope))

        if need_boost.any():
            p = pos_cand_non_ground_world
            f1 = torch.sin(p * 3.1).sum(dim=1)
            f2 = torch.cos(p * 7.3).sum(dim=1)
            f3 = torch.sin(p * 13.7).sum(dim=1)
            noise = (f1 + 0.5 * f2 + 0.25 * f3).unsqueeze(1).repeat(1, 3)
            rand_dir = noise - (noise * n0_cand_non_ground).sum(-1, keepdim=True) * n0_cand_non_ground
            rand_dir = F.normalize(rand_dir + 1e-8, dim=1)
            inject = rand_dir * float(self.normal_min_slope)
            grad_tan[need_boost] = inject[need_boost]

        if self.normal_slope_max_deg is not None:
            theta_max = math.radians(float(self.normal_slope_max_deg))
            s_max = math.tan(theta_max)
            gmag_after = torch.norm(grad_tan, dim=1) + 1e-8
            slope_scale = torch.clamp(s_max / gmag_after, max=1.0)
            grad_tan = grad_tan * slope_scale.unsqueeze(1)

        n_new = n0_cand_non_ground - cover_normal.unsqueeze(1) * grad_tan
        n_new = F.normalize(n_new, dim=1)

        if self.displace:
            disp = (height_all * cover_soft * self.displacement_scale).unsqueeze(1)
            pos_cand_non_ground = pos_cand_non_ground + n0_cand_non_ground * disp

        base_albedo = ctx.albedo_flat[cand_indices_flat]
        base_rough = ctx.rough_flat[cand_indices_flat]
        base_metal = ctx.metal_flat[cand_indices_flat]

        albedo_scale = self._get_metaball_snow_albedo_scale(ctx.snow_amt)
        snow_albedo_base = torch.full((nc, 3), float(self.snow_albedo_value), device=self.device)
        snow_albedo = base_albedo * (1.0 - albedo_scale) + snow_albedo_base * albedo_scale
        snow_rough = torch.full((nc, 1), float(self.snow_roughness_value), device=self.device)

        cover_3_albedo = cover_albedo.unsqueeze(1)
        cover_3_soft = cover_soft.unsqueeze(1)

        ctx.albedo_flat[cand_indices_flat] = base_albedo * (1.0 - cover_3_albedo) + snow_albedo * cover_3_albedo
        ctx.rough_flat[cand_indices_flat] = base_rough * (1.0 - cover_3_soft) + snow_rough * cover_3_soft
        ctx.metal_flat[cand_indices_flat] = base_metal * (1.0 - cover_3_soft)
        ctx.normal_flat[cand_indices_flat] = n_new
        if self.displace:
            ctx.pos_flat[cand_indices_flat] = pos_cand_non_ground

    def _apply_wet_ground(self, ctx: _SnowFrameContext, no_snow_ground_indices_flat: torch.Tensor) -> None:
        """Darken and wetten ground pixels that received no snow cover."""
        if not self.wet_ground_enabled or no_snow_ground_indices_flat.numel() == 0:
            return

        gi = no_snow_ground_indices_flat
        base_albedo = ctx.albedo_flat[gi]
        base_rough = ctx.rough_flat[gi]
        base_metal = ctx.metal_flat[gi]

        wet_intensity = torch.tensor(self.wet_ground_intensity, device=self.device)
        porosity = torch.tensor(self.wet_ground_porosity, device=self.device)
        darkening = (1.0 - porosity * wet_intensity).clamp(0.0, 1.0)
        wet_albedo = base_albedo * darkening
        water_r = float(self.wet_ground_roughness_factor or 0.03)
        wet_rough = torch.lerp(base_rough, torch.full_like(base_rough, water_r), wet_intensity)
        wet_metal = base_metal

        ctx.albedo_flat[gi] = wet_albedo
        ctx.rough_flat[gi] = wet_rough
        ctx.metal_flat[gi] = wet_metal

    def _write_back_material(self, material: Material, ctx: _SnowFrameContext) -> None:
        """Reshape flattened buffers back into ``material`` CHW tensors."""
        img_height, img_width = ctx.img_height, ctx.img_width
        material.albedo = ctx.albedo_flat.reshape(img_height, img_width, 3).permute(2, 0, 1)
        material.roughness = ctx.rough_flat.reshape(img_height, img_width, 1).permute(2, 0, 1)
        material.metallic = ctx.metal_flat.reshape(img_height, img_width, 1).permute(2, 0, 1)
        material.normal = ctx.normal_flat.reshape(img_height, img_width, 3).permute(2, 0, 1)
        if self.displace:
            material.position = ctx.pos_flat.reshape(img_height, img_width, 3).permute(2, 0, 1)

    @torch.no_grad()
    def modify_gbuffer_for_snow(
        self,
        material: Material,
        snow_amount: float | torch.Tensor = 1.0,
        interval: float | None = None,
    ) -> Material:
        """Apply snow coverage to one frame's G-buffer maps in place.

        Args:
            material: Mutable per-frame material buffers.
            snow_amount: Scalar or per-pixel snow amount.
            interval: Optional override for metaball influence interval.

        Returns:
            The same ``material`` instance with updated maps.
        """
        if self._should_skip_snow(snow_amount) or not self.is_initialized:
            return material

        ctx = self._build_snow_frame_context(material, snow_amount)
        if ctx is None:
            return material

        no_snow_ground = self._apply_ground_snow(ctx, interval)
        self._apply_metaball_snow(ctx, snow_amount, interval)
        self._apply_wet_ground(ctx, no_snow_ground)
        self._write_back_material(material, ctx)
        return material
