"""Rain puddle simulation, ripple normals, and raindrop G-buffer painting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def _mix(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear blend ``a`` toward ``b`` by weight ``t``."""
    return a * (1.0 - t) + b * t


@dataclass
class GLNFBMOpts:
    """FBM noise options for puddle mask synthesis."""

    seed: float = 1.0
    persistance: float = 0.5
    lacunarity: float = 2.0
    scale: float = 0.05
    redistribution: float = 1.0
    octaves: int = 3
    terbulance: bool = False
    ridge: bool = False


class RainPuddleSimulator:
    """Simulate rain wetness, ground ripples, and falling raindrop streaks.

    Updates shared G-buffer tensors in place through :meth:`process_frame`.
    """

    def __init__(
        self,
        urban_scene,
        raindrop_count: int = 10000,
        dt: float = 0.1,
        resolution: float = 0.02,
        angle_deg: float = 15.0,
    ) -> None:
        """Initialize rain simulation state for a scene.

        Args:
            urban_scene: Source scene with pose, depth, normal, and G-buffer accessors.
            raindrop_count: Number of simulated raindrops.
            dt: Simulation timestep in seconds.
            resolution: Ground grid resolution in meters.
            angle_deg: Maximum ground normal tilt from up, in degrees.
        """
        self.urban_scene = urban_scene
        self.dt = dt
        self.cam_pos, self.perframe_world_pts, \
            self.ground_pts, self.ground_labels, \
            self.ground_normals, self.mask_2ds = self._init_ground_and_cam_poses(urban_scene)
        self.raindrops = self._init_raindrops(raindrop_count)
        self.angle_deg = angle_deg
        self.build_global_grid_from_ground(resolution, self.angle_deg)
        self.K = urban_scene.get_intrinsics()
        self.puddle_mask = self._generate_puddle_mask()

    def build_global_grid_from_ground(
        self,
        resolution: float,
        angle_deg: float,
        target_dir: tuple[float, float, float] = (0.0, -1.0, 0.0),
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        """Build a global ground height and normal grid in reference-world coordinates.

        Args:
            resolution: Grid cell size in meters.
            angle_deg: Maximum ground normal tilt from up, in degrees.
            target_dir: Fallback normal for empty cells.

        Returns:
            Tuple of ``(height, counts, meta)`` where ``meta`` holds origin, size, and resolution.
        """
        all_ground_pts = torch.cat(self.ground_pts, dim=1).detach().clone()  # (3, N)
        all_ground_nrms = torch.cat(self.ground_normals, dim=1).detach().clone()  # (3, N) in ref-world

        cos_thr = float(np.cos(np.deg2rad(angle_deg)))
        up_dir = torch.tensor([[0.0], [-1.0], [0.0]], dtype=torch.float32, device=all_ground_pts.device)
        cosang = (all_ground_nrms * up_dir).sum(dim=0)  # (N,)
        sel = cosang >= cos_thr

        all_ground_pts = all_ground_pts[:, sel]
        all_ground_nrms = all_ground_nrms[:, sel]

        offset = 0
        for fi in range(len(self.ground_labels)):
            old_labels = self.ground_labels[fi]
            available_sum = old_labels.sum()
            # Map per-point ground selection back into each frame label mask.
            assert offset + available_sum <= sel.shape[0]
            old_labels[old_labels.clone()] = sel[offset:offset + available_sum].clone()
            self.ground_labels[fi] = old_labels
            offset += available_sum

        x, y, z = all_ground_pts[0, :], all_ground_pts[1, :], all_ground_pts[2, :]
        x_min, x_max = float(x.min().item()), float(x.max().item())
        z_min, z_max = float(z.min().item()), float(z.max().item())

        def dim_len(minv, maxv, res):
            span = max(0.0, maxv - minv)
            return int(np.floor(span / res)) + 1

        Nx = dim_len(x_min, x_max, resolution)
        Nz = dim_len(z_min, z_max, resolution)

        u = torch.floor((x - x_min) / resolution).to(dtype=torch.long)
        v = torch.floor((z - z_min) / resolution).to(dtype=torch.long)
        u.clamp_(0, Nx - 1)
        v.clamp_(0, Nz - 1)

        lin = v * Nx + u
        flat_size = Nx * Nz
        counts_flat = torch.bincount(lin, minlength=flat_size)
        counts = counts_flat.view(Nz, Nx)

        height_flat = torch.full((flat_size,), float('-inf'), dtype=torch.float32, device=all_ground_pts.device)
        vals = y.to(torch.float32)
        height_flat.scatter_reduce_(0, lin.to(device=all_ground_pts.device), vals.to(device=all_ground_pts.device), reduce='amax', include_self=True)
        height = height_flat.view(Nz, Nx)

        device = all_ground_pts.device
        # Aggregate normals at per-cell height maxima.
        nrms = F.normalize(all_ground_nrms.t().to(torch.float32), dim=1, eps=1e-12)  # (N,3)
        eps = 1e-6
        is_top = (vals >= height_flat[lin] - eps)
        top_lin = lin[is_top]
        norm_sum_top = torch.zeros((flat_size, 3), dtype=torch.float32, device=device)
        norm_sum_top.index_add_(0, top_lin, nrms[is_top])
        cnt_top = torch.bincount(top_lin, minlength=flat_size).to(torch.float32).unsqueeze(1)  # (H*W,1)
        tgt = torch.tensor(target_dir, dtype=torch.float32, device=device).view(1, 3)
        avg_norm_top = norm_sum_top / cnt_top.clamp_min(1.0)
        lens = torch.linalg.norm(avg_norm_top, dim=1, keepdim=True).clamp_min(1e-12)
        avg_norm_top = avg_norm_top / lens
        has_any = (cnt_top.squeeze(1) > 0.5)
        avg_norm_top[~has_any] = tgt
        normal_grid = avg_norm_top.view(Nz, Nx, 3).permute(2, 0, 1).contiguous()  # (3, Nz, Nx)

        self.global_height_grid = height
        self.global_height_counts = counts
        self.global_normal_grid = normal_grid
        self.global_height_meta = {
            "origin": (x_min, z_min),
            "size": (Nz, Nx),
            "resolution": resolution
        }

        return height, counts, self.global_height_meta

    def _get_fx_fy_cx_cy(self):
        K = self.K
        fx = float(K[0, 0].item())
        fy = float(K[1, 1].item())
        cx = float(K[0, 2].item())
        cy = float(K[1, 2].item())
        return fx, fy, cx, cy

    def estimate_gravity_and_ref_pose(
        self,
        ground_pts: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Estimate a gravity-aligned reference pose from ground points.

        Args:
            ground_pts: Per-frame ground point clouds, each shaped ``(3, N)``.

        Returns:
            ``4x4`` reference pose tensor.
        """
        all_ground_pts = torch.cat(ground_pts, dim=1)  # (3, N)
        mean = all_ground_pts.mean(dim=1, keepdim=True)  # (3,1)
        X = all_ground_pts - mean.view(3,1)
        denom = max(1, all_ground_pts.shape[1] - 1)
        cov = (X @ X.t()) / float(denom)  # (3,3)
        U, S, Vh = torch.linalg.svd(cov)
        R_ref = U @ Vh
        t_ref = mean.view(3, 1)
        T_ref = torch.eye(4, dtype=all_ground_pts.dtype, device=all_ground_pts.device)
        T_ref[:3, :3] = R_ref
        T_ref[:3, 3:4] = t_ref
        return T_ref

    def _init_ground_and_cam_poses(self, urban_scene, depth_min=0.1, depth_max=200.0):
        all_poses = []
        avail_masks = []
        perframe_world = []
        ground_labels = []
        ground_pts = []
        perframe_normals = []
        ground_normals = []
        for fi in range(len(urban_scene)):
            pose = urban_scene.get_pose(fi)
            all_poses.append(pose)
            position = urban_scene.get_position(fi)
            depth = urban_scene.get_depth(fi)
            normal_map = urban_scene.get_normal(fi)
            avail_mask = (depth > depth_min) & (depth < depth_max)

            mask2d = avail_mask[0]

            pos_flat = position[:,mask2d]
            nrm_flat = normal_map[:,mask2d]
            pts_w_flat = (pose[:3, :3] @ pos_flat) + pose[:3, 3:4]
            ground_label = self.get_ground(pts_w_flat)
            ground_labels.append(ground_label)
            perframe_normals.append(nrm_flat)
            ground_normals.append(nrm_flat[:,ground_label])
            ground_pts.append(pts_w_flat[:,ground_label])
            perframe_world.append(pts_w_flat)
            avail_masks.append(mask2d)

        T_ref = self.estimate_gravity_and_ref_pose(ground_pts)
        T_ref_inv = torch.linalg.inv(T_ref)
        for fi in range(len(urban_scene)):
            # how to make T_ref to be world init
            pose_old = all_poses[fi]
            pose_new = T_ref_inv @ pose_old
            all_poses[fi] = pose_new

            perframe_world_old = perframe_world[fi]
            perframe_world_new = (T_ref_inv[:3, :3] @ perframe_world_old) + T_ref_inv[:3, 3:4]
            perframe_world[fi] = perframe_world_new

            ground_pts[fi] = perframe_world_new[:,ground_labels[fi]]

        return all_poses, perframe_world, ground_pts, ground_labels, ground_normals, avail_masks

    def get_ground(self, pts: torch.Tensor) -> torch.Tensor:
        """Detect ground points with an iterative plane-fit method.

        Coordinate convention: ``x`` right, ``y`` down, ``z`` forward.

        Args:
            pts: World points shaped ``(3, N)``.

        Returns:
            Boolean mask of shape ``(N,)`` where ``True`` marks ground points.
        """
        # accept numpy or torch input
        pts_t = pts.clone().to(dtype=torch.float32)

        # normalize shape to (N,3)
        assert pts_t.dim() == 2 and pts_t.shape[0] == 3
        pts_t = pts_t.t()
        
        device = pts_t.device
        N = pts_t.shape[0]

        th_seeds_ = 1.2
        num_lpr_ = 20
        n_iter = 10
        th_dist_ = 0.3

        # Sort by y (downwards). Ground points have large y => sort descending
        idx = torch.argsort(pts_t[:, 1], descending=True)
        pts_sort = pts_t[idx]
        num_lpr = min(num_lpr_, pts_sort.shape[0])
        lpr = float(pts_sort[:num_lpr, 1].mean().item())

        # initial seed points near the representative low (large y) region
        pts_g = pts_sort[pts_sort[:, 1] > (lpr - th_seeds_)]
        if pts_g.shape[0] == 0:
            return torch.zeros((N,), dtype=torch.bool, device=device)

        normal_ = torch.zeros(3, dtype=torch.float32, device=device)
        th_dist_d_ = float(th_dist_)  # init

        for _ in range(n_iter):
            mean = pts_g.mean(dim=0)[:3]

            X = pts_g[:, :3] - mean.view(1, 3)
            # covariance (use unbiased estimator when possible)
            denom = max(1, X.shape[0] - 1)
            cov = (X.t() @ X) / float(denom)

            # SVD -> principal directions; smallest singular vector ~ plane normal
            U, S, Vh = torch.linalg.svd(cov)

            normal_ = U[:, -1]
            norm_n = normal_.norm().item()
            if norm_n < 1e-8:
                break
            normal_ = normal_ / (norm_n + 1e-12)

            if normal_[1] > 0:
                normal_ = -normal_

            d_ = - (normal_ @ mean)
            # compare n·x + d > th_dist_  -> rearranged as n·x > th_dist_ - d_
            th_dist_d_ = float(th_dist_ - d_)

            result = pts_t[:, :3] @ normal_.unsqueeze(1)  # (N,1)
            mask = (result.squeeze(-1) < th_dist_d_)
            pts_g = pts_t[mask]
            if pts_g.shape[0] == 0:
                break

        # final label (return 1D bool mask)
        result = pts_t[:, :3] @ normal_.unsqueeze(1)
        ground_label = (result.squeeze(-1) < th_dist_d_).to(torch.bool)
        return ground_label


    def _init_raindrops(self, count):
        x_pad = 10.0
        current_grid_range = [
            self.ground_pts[0].min(dim=1).values[0].item() - x_pad,
            self.ground_pts[0].max(dim=1).values[0].item() + x_pad,
            self.ground_pts[0].min(dim=1).values[2].item(),
            self.ground_pts[0].max(dim=1).values[2].item(),
        ]
        xmin, xmax, zmin, zmax = current_grid_range
        self.droplet_positions = torch.rand((count, 3), dtype=torch.float32, device=self.ground_pts[0].device)
        self.droplet_positions[:, 0] = self.droplet_positions[:, 0] * (xmax - xmin) + xmin
        self.droplet_positions[:, 2] = self.droplet_positions[:, 2] * (zmax - zmin) + zmin
        self.droplet_positions[:, 1] = self.droplet_positions[:, 1] * (-51.0)

        self.droplet_size = torch.empty((count,), dtype=torch.float32, device=self.ground_pts[0].device).uniform_(0.5, 6.0)
        # Gunn-Kinzer terminal velocity from sampled droplet diameter (mm).
        vt_y = 9.65 - 10.3 * torch.exp(-0.6 * self.droplet_size)
        vt_y = vt_y.clamp_(0.5, 9.5)
        vt_y = vt_y * torch.clamp(1.0 + 0.15 * torch.randn_like(vt_y), 0.5, 1.5)

        wind_x = 0.1
        wind_z = 0.0

        self.droplet_velocity = torch.zeros((count, 3), dtype=torch.float32, device=self.ground_pts[0].device)
        self.droplet_velocity[:, 0] = wind_x + 0.5 * torch.randn((count,), dtype=torch.float32, device=self.ground_pts[0].device)
        self.droplet_velocity[:, 2] = wind_z + 0.5 * torch.randn((count,), dtype=torch.float32, device=self.ground_pts[0].device)
        self.droplet_velocity[:, 1] = vt_y

        self.previous_grid_range = current_grid_range
        self.previous_idx = 0

    def update_raindrops(self, idx: int) -> None:
        """Advance raindrop positions for frame ``idx``.

        Args:
            idx: Zero-based frame index. Must equal ``previous_idx + 1`` after the first frame.

        Raises:
            ValueError: If ``idx`` skips frames.
        """
        x_pad = 10
        if idx == self.previous_idx:
            return
        elif idx == self.previous_idx + 1:
            self.droplet_positions += self.droplet_velocity * self.dt

            current_grid_range = [
                self.ground_pts[idx].min(dim=1).values[0].item() - x_pad,
                self.ground_pts[idx].max(dim=1).values[0].item() + x_pad,
                self.ground_pts[idx].min(dim=1).values[2].item(),
                self.ground_pts[idx].max(dim=1).values[2].item(),
            ]

            x_min, x_max, z_min, z_max = current_grid_range

            
            hit_ground_mask = self.droplet_positions[:, 1] >= 0.0
            out_of_bounds_mask = (self.droplet_positions[:, 0] < current_grid_range[0]) | \
                                 (self.droplet_positions[:, 0] > current_grid_range[1]) | \
                                 (self.droplet_positions[:, 2] < current_grid_range[2]) | \
                                 (self.droplet_positions[:, 2] > current_grid_range[3])
        
            reset_mask = hit_ground_mask | out_of_bounds_mask

            if reset_mask.any():
                self.droplet_positions[reset_mask, 0] = (torch.rand((reset_mask.sum(),), dtype=torch.float32, device=self.droplet_positions.device) * (x_max - x_min) + x_min)
                self.droplet_positions[reset_mask, 1] = (torch.rand((reset_mask.sum(),), dtype=torch.float32, device=self.droplet_positions.device) * (-51.0))
                self.droplet_positions[reset_mask, 2] = (torch.rand((reset_mask.sum(),), dtype=torch.float32, device=self.droplet_positions.device) * (z_max - z_min) + z_min)
                self.droplet_size[reset_mask] = torch.empty((reset_mask.sum(),), dtype=torch.float32, device=self.droplet_positions.device).uniform_(0.5, 6.0)
            
            self.previous_grid_range = current_grid_range
            self.previous_idx = idx

            return
        else:
            raise ValueError("Can only update raindrops to the next frame.")



    def _generate_puddle_mask(self):
        H, W = self.global_height_meta["size"]
        glnfbm_opts = GLNFBMOpts()
        puddle_mask = self._gln_sfbm(H, W, glnfbm_opts, dtype=torch.float32)

        return puddle_mask
    

    def _generate_ripple_normals(self, H: int, W: int, time: float, max_radius: int = 1,
                                cell_size: float = 64.0, intensity_min: float = 0.01,
                                intensity_max: float = 0.15, device=None, dtype=torch.float32) -> torch.Tensor:
        """Generate a ripple normal map.

        Returns:
            Normal map of shape ``(3, H, W)`` with unit-length vectors.

        Notes:
            Uses ``(x, y, z)`` with positive ``y`` pointing down.
        """
        device = device or (self.global_height_grid.device if hasattr(self, "global_height_grid") else "cpu")

        # Map pixel coordinates into coarse grid space for ripple cells.
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij"
        )
        uv = torch.stack((xx, yy), dim=0) / max(cell_size, 1e-6)  # (2,H,W) -> [x,z]
        p0 = torch.floor(uv)

        def fract(x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=dtype, device=device)
            return x - torch.floor(x)

        def smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
            t = (x - edge0) / (edge1 - edge0 + 1e-6)
            t = torch.clamp(t, 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        def hash12(pi2: torch.Tensor) -> torch.Tensor:
            x = pi2[0]; z = pi2[1]
            h = torch.sin(x * 127.1 + z * 311.7) * 43758.5453
            return fract(h)

        def hash22(pi2: torch.Tensor) -> torch.Tensor:
            x = pi2[0]; z = pi2[1]
            h1 = torch.sin(x * 127.1 + z * 311.7) * 43758.5453
            h2 = torch.sin(x * 269.5 + z * 183.3) * 43758.5453
            return torch.stack((fract(h1), fract(h2)), dim=0)

        circles = torch.zeros((2, H, W), dtype=dtype, device=device)  # [dx, dz]
        R = int(max_radius)
        t_anim = float(5.14 + time)

        for j in range(-R, R + 1):
            for i in range(-R, R + 1):
                offset = torch.tensor([float(i), float(j)], dtype=dtype, device=device).view(2, 1, 1)  # [x,z]
                pi = p0 + offset
                hsh = pi
                rnd2 = hash22(hsh)                  # (2,H,W) ∈ [0,1)
                p = pi + rnd2
                rnd1 = hash12(hsh)
                t = fract(0.3 * t_anim + rnd1)

                v = p - uv
                vlen = torch.sqrt(v[0] * v[0] + v[1] * v[1] + 1e-12)

                # Expanding ring wave; radius grows with lifecycle ``t``.
                d = vlen - (float(R) + 1.0) * t

                h = 1e-3
                d1 = d - h
                d2 = d + h

                p1 = torch.sin(31.0 * d1) * smoothstep(-0.6, -0.3, d1) * smoothstep(0.0, -0.3, d1)
                p2 = torch.sin(31.0 * d2) * smoothstep(-0.6, -0.3, d2) * smoothstep(0.0, -0.3, d2)
                deriv = (p2 - p1) / (2.0 * h) * (1.0 - t) * (1.0 - t)

                vdir = v / vlen.clamp_min(1e-6)
                circles = circles + 0.5 * vdir * deriv

        circles = circles / float((2 * R + 1) * (2 * R + 1))

        osc = fract(0.05 * t_anim + 0.5) * 2.0 - 1.0
        osc = osc.abs()
        k = smoothstep(0.1, 0.6, osc)
        intensity = intensity_min + (intensity_max - intensity_min) * k
        circles = circles * intensity

        nx = circles[0]
        nz = circles[1]
        ny = -torch.sqrt((1.0 - (nx * nx + nz * nz)).clamp_min(0.0))
        normal = torch.stack((nx, ny, nz), dim=0)
        nlen = torch.sqrt((normal * normal).sum(dim=0).clamp_min(1e-12))
        normal = normal / nlen
        return normal

    def _gln_sfbm(self, H: int, W: int, opts: GLNFBMOpts, uv_offset=(0.0, 0.0), device=None, dtype=torch.float32) -> torch.Tensor:
        """Generate a 2D FBM noise field for puddle placement.

        Returns:
            Tensor of shape ``(H, W)`` in ``[0, 1]`` after normalization and shaping.
        """
        device = device or (self.global_height_grid.device if hasattr(self, "global_height_grid") else "cpu")

        # Seed offset for reproducible value-noise octaves.
        seed_offset = float(opts.seed) * 100.0
        ou = uv_offset[0] + seed_offset
        ov = uv_offset[1] + seed_offset

        persistance = float(opts.persistance)
        lacunarity = float(opts.lacunarity)
        redistribution = float(opts.redistribution)
        octaves = int(opts.octaves)
        scale = float(opts.scale)
        terbulance = bool(opts.terbulance)
        ridge = bool(opts.terbulance and opts.ridge)

        g = torch.Generator(device=device)
        g.manual_seed(int(round(seed_offset)) & 0x7FFFFFFF)

        result = torch.zeros((1, 1, H, W), dtype=dtype, device=device)
        amplitude = 1.0
        frequency = 1.0
        maximum = amplitude

        for i in range(octaves):
            gh = max(1, int(round(H * frequency * scale)))
            gw = max(1, int(round(W * frequency * scale)))
            grid = torch.rand((1, 1, gh, gw), generator=g, device=device, dtype=dtype) * 2.0 - 1.0
            sh = (int(round(ov * gh)) % gh) if gh > 1 else 0
            sw = (int(round(ou * gw)) % gw) if gw > 1 else 0
            if (sh != 0 or sw != 0) and (gh > 1 or gw > 1):
                grid = torch.roll(grid, shifts=(sh, sw), dims=(2, 3))
            up = F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)

            noiseVal = up
            if terbulance:
                noiseVal = noiseVal.abs()
                noiseVal = noiseVal * 2.0 - 1.0
            if ridge:
                noiseVal = -1.0 * noiseVal

            result = result + noiseVal * amplitude

            frequency *= lacunarity
            amplitude *= persistance
            maximum += amplitude

        redistributed = torch.sign(result) * torch.pow(result.abs() + 1e-8, redistribution)
        out = redistributed / max(maximum, 1e-6)
        out = (out + 1) / 2.0
        out = self._smoothstep(0.0, 0.7, out)
        out = self._smoothstep(0.2, 1.0, out)
        return out[0, 0]  # (H,W)
    
    def _smoothstep(self, edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
        t = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def perturb_normal(
        self,
        base_normal: torch.Tensor,
        ripple_normal: torch.Tensor,
        strength: float = 0.25,
    ) -> torch.Tensor:
        """Blend a ripple normal into a base surface normal.

        Args:
            base_normal: Base normals shaped ``(3, N)``.
            ripple_normal: Ripple normals shaped ``(3, N)``.
            strength: Blend strength in ``[0, 1]``.

        Returns:
            Perturbed unit normals shaped ``(3, N)``.
        """
        noise_normal_orthogonal = ripple_normal - (ripple_normal * base_normal).sum(dim=0, keepdim=True) * base_normal
        noise_normal_orthogonal = F.normalize(base_normal - noise_normal_orthogonal * strength, dim=0)

        return noise_normal_orthogonal

    def _tint_overcast_sky(self,
                           basecolor_map: torch.Tensor,   # (3,H,W), float in [0,1]
                           sky_mask: torch.Tensor,        # (H,W) or (1,H,W), bool/0-1
                           tint: tuple[float, float, float] = (0.55, 0.60, 0.70),
                           strength: float = 0.6,
                           desaturate: float = 0.7) -> None:
        """Desaturate and tint sky pixels toward an overcast gray-blue look."""
        device = basecolor_map.device
        sky2d = torch.as_tensor(sky_mask, device=device).to(torch.bool)
        if sky2d.dim() == 3 and sky2d.shape[0] == 1:
            sky2d = sky2d[0]
        if not sky2d.any():
            return

        # Select sky pixels (3, M)
        pix = basecolor_map[:, sky2d]
        # Luma gray (perceptual)
        luma = torch.tensor([0.2126, 0.7152, 0.0722], dtype=pix.dtype, device=device).view(3, 1)
        gray = (pix * luma).sum(dim=0, keepdim=True).expand_as(pix)
        # Desaturate then tint
        pix_desat = _mix(pix, gray, desaturate)
        tint_col = torch.tensor(tint, dtype=pix.dtype, device=device).view(3, 1).expand_as(pix_desat)
        pix_overcast = _mix(pix_desat, tint_col, strength).clamp_(0.0, 1.0)
        basecolor_map[:, sky2d] = pix_overcast

    @staticmethod
    def _sd_uneven_capsule_batched(px: torch.Tensor, py: torch.Tensor,
                                   r0: torch.Tensor, r1: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Batched uneven capsule SDF.
        px,py: (B,H,W); r0,r1,h: (B,1,1). Return (B,H,W)
        """
        px = px.abs()
        h = h.clamp_min(1e-6)
        b = (r0 - r1) / h
        a = torch.sqrt((1.0 - b * b).clamp_min(0.0))
        k = (-b) * px + a * py
        left = k < 0.0
        right = k > (a * h)

        sd_left = torch.sqrt(px * px + py * py) - r0
        sd_right = torch.sqrt(px * px + (py - h) * (py - h)) - r1
        sd_mid = a * px + b * py - r0

        sd = torch.where(left, sd_left, torch.where(right, sd_right, sd_mid))
        return sd

    def _paint_raindrops_gbuffers(
            self, idx, basecolor_map, roughness_map, normal_map, depth_map,
            color_tint=(1.0, 1.0, 1.0),
    ):
        pose = self.cam_pos[idx]
        Pinv = torch.linalg.inv(pose)
        R_w2c = Pinv[:3, :3]
        t_w2c = Pinv[:3, 3:4]

        fx, fy, cx, cy = self._get_fx_fy_cx_cy()

        Pw = self.droplet_positions  # (N,3)
        Pc = (R_w2c @ Pw.t()) + t_w2c
        Vw = self.droplet_velocity  # (N,3)
        Vc = (R_w2c @ Vw.t())  # (3,N)

        def project(P):
            u = fx * (P[0] / P[2]) + cx
            v = fy * (P[1] / P[2]) + cy
            return u, v

        dt = self.dt
        streak_scale = 0.8

        P0 = Pc
        P1 = Pc + Vc * (streak_scale * dt)
        u0, v0 = project(P0)  # (N,)
        u1, v1 = project(P1)  # (N,)

        du = u1 - u0
        dv = v1 - v0
        L = torch.sqrt(du * du + dv * dv).clamp_min(1e-6)
        ax = du / L
        ay = dv / L
        nx = -ay
        ny = ax

        z0 = P0[2, :]
        z1 = P1[2, :]

        r_head_world = 0.5 * (self.droplet_size * 1e-3)   # (N,) in meters, radius = diameter/2
        r_tail_world = r_head_world / 0.7
        r0_px = 0.5 * (fx * (r_head_world / z0) + fy * (r_head_world / z0))
        r1_px = 0.5 * (fx * (r_tail_world / z1) + fy * (r_tail_world / z1))
        r0_px = r0_px.clamp(0.5, 5.0)
        r1_px = r1_px.clamp(0.7, 7.0)

        rain_col = torch.tensor(color_tint, dtype=basecolor_map.dtype, device=basecolor_map.device).view(3, 1)
        view_normal = torch.tensor([0.0, 0.0, -1.0], dtype=normal_map.dtype, device=normal_map.device).view(3, 1)

        H, W = basecolor_map.shape[1], basecolor_map.shape[2]
        device = basecolor_map.device
        # Filter droplets behind the camera.
        valid = (z0 > 0) & (z1 > 0)
        if not valid.any():
            return
        u0 = u0[valid]; v0 = v0[valid]; u1 = u1[valid]; v1 = v1[valid]
        Lv = L[valid]; axv = ax[valid]; ayv = ay[valid]; nxv = nx[valid]; nyv = ny[valid]
        z0v = z0[valid]; z1v = z1[valid]; r0v = r0_px[valid]; r1v = r1_px[valid]

        # Build per-droplet AABBs and clip to the image.
        pad = (torch.maximum(r0v, r1v) + 2.0)
        umin = torch.floor(torch.minimum(u0, u1) - pad).to(torch.int64).clamp(0, W - 1)
        umax = torch.ceil(torch.maximum(u0, u1) + pad).to(torch.int64).clamp(0, W - 1)
        vmin = torch.floor(torch.minimum(v0, v1) - pad).to(torch.int64).clamp(0, H - 1)
        vmax = torch.ceil(torch.maximum(v0, v1) + pad).to(torch.int64).clamp(0, H - 1)
        keep = (umax >= umin) & (vmax >= vmin)
        if not keep.any():
            return
        umin = umin[keep]; umax = umax[keep]; vmin = vmin[keep]; vmax = vmax[keep]
        u0v = u0[keep]; v0v = v0[keep]
        axv = axv[keep]; ayv = ayv[keep]; nxv = nxv[keep]; nyv = nyv[keep]
        Lv = Lv[keep]; z0v = z0v[keep]; z1v = z1v[keep]; r0v = r0v[keep]; r1v = r1v[keep]

        # Flatten G-buffers for batched writes.
        D2d = depth_map[0]
        D_flat = D2d.view(-1)
        R_flat = roughness_map[0].view(-1)
        B_flat = basecolor_map.view(3, -1)
        N_flat = normal_map.view(3, -1)

        alpha = 0.4
        depth_bias = 1e-4
        chunk = 256
        Nv = umin.numel()
        for i0 in range(0, Nv, chunk):
            i1 = min(Nv, i0 + chunk)
            Bn = i1 - i0
            umin_b = umin[i0:i1]; umax_b = umax[i0:i1]
            vmin_b = vmin[i0:i1]; vmax_b = vmax[i0:i1]
            u0_b = u0v[i0:i1]; v0_b = v0v[i0:i1]
            ax_b = axv[i0:i1]; ay_b = ayv[i0:i1]; nx_b = nxv[i0:i1]; ny_b = nyv[i0:i1]
            L_b = Lv[i0:i1]; z0_b = z0v[i0:i1]; z1_b = z1v[i0:i1]
            r0_b = r0v[i0:i1]; r1_b = r1v[i0:i1]

            widths = (umax_b - umin_b + 1)
            heights = (vmax_b - vmin_b + 1)
            maxW = int(widths.max().item())
            maxH = int(heights.max().item())
            if maxW <= 0 or maxH <= 0:
                continue
            # Batched ROI grid (B, maxH, maxW).
            Us = torch.arange(maxW, device=device, dtype=torch.float32).view(1, 1, maxW).expand(Bn, maxH, maxW)
            Vs = torch.arange(maxH, device=device, dtype=torch.float32).view(1, maxH, 1).expand(Bn, maxH, maxW)
            Uabs = Us + umin_b.view(Bn, 1, 1).to(torch.float32)
            Vabs = Vs + vmin_b.view(Bn, 1, 1).to(torch.float32)
            roi_mask = (Us < widths.view(Bn, 1, 1).to(torch.float32)) & (Vs < heights.view(Bn, 1, 1).to(torch.float32))

            du0 = Uabs - u0_b.view(Bn, 1, 1)
            dv0 = Vabs - v0_b.view(Bn, 1, 1)
            px = du0 * nx_b.view(Bn, 1, 1) + dv0 * ny_b.view(Bn, 1, 1)
            py = du0 * ax_b.view(Bn, 1, 1) + dv0 * ay_b.view(Bn, 1, 1)

            sd = self._sd_uneven_capsule_batched(
                px, py,
                r0_b.view(Bn, 1, 1),
                r1_b.view(Bn, 1, 1),
                L_b.view(Bn, 1, 1)
            )
            inside = (sd < 0.0) & roi_mask
            if not inside.any():
                continue

            t_axis = (py / L_b.view(Bn, 1, 1)).clamp(0.0, 1.0)
            zpix = z0_b.view(Bn, 1, 1) + t_axis * (z1_b.view(Bn, 1, 1) - z0_b.view(Bn, 1, 1))

            # Per-pixel nearest depth candidates along each streak.
            Uidx = Uabs.clamp_max(W - 1).to(torch.int64)
            Vidx = Vabs.clamp_max(H - 1).to(torch.int64)
            lin = (Vidx * W + Uidx).view(-1)
            D_cur = D_flat.gather(0, lin).view(Bn, maxH, maxW)
            cand = inside & (zpix < (D_cur - depth_bias))
            if not cand.any():
                continue

            lin_c = (Vidx[cand] * W + Uidx[cand])  # (K,)
            z_c = zpix[cand]                        # (K,)
            zmin = torch.full_like(D_flat, float('inf'))
            zmin.scatter_reduce_(0, lin_c, z_c, reduce='amin', include_self=True)
            upd_mask = zmin < (D_flat - depth_bias)
            if not upd_mask.any():
                continue

            idxs = torch.nonzero(upd_mask, as_tuple=False).squeeze(1)
            D_flat[idxs] = zmin[idxs]
            B_flat[:, idxs] = _mix(B_flat[:, idxs], rain_col.expand(3, idxs.numel()), alpha)
            R_flat[idxs] = _mix(R_flat[idxs], torch.full_like(R_flat[idxs], 0), alpha)
            N_flat[:, idxs] = F.normalize(
                _mix(N_flat[:, idxs], view_normal.expand(3, idxs.numel()), alpha),
                dim=0, eps=1e-12
            )

        depth_map[0].copy_(D_flat.view(H, W))
        roughness_map[0].copy_(R_flat.view(H, W))
        basecolor_map.copy_(B_flat.view(3, H, W))
        normal_map.copy_(N_flat.view(3, H, W))

    def process_frame(self, idx: int, time: float) -> None:
        """Apply rain geometry effects to one frame's G-buffer buffers.

        Args:
            idx: Zero-based frame index.
            time: Simulation time for ripple animation.
        """
        depth_map = self.urban_scene.get_depth(idx)
        normal_map = self.urban_scene.get_normal(idx)
        basecolor_map = self.urban_scene.get_basecolor(idx)
        roughness_map = self.urban_scene.get_roughness(idx)
        sky_mask = self.urban_scene.get_sky_mask(idx)
        mask_2d = self.mask_2ds[idx]
        self.update_raindrops(idx)

        self._tint_overcast_sky(
            basecolor_map, sky_mask,
            tint=(0.55, 0.60, 0.70),
            strength=0.6,
            desaturate=0.7
        )

        puddle_mask = self.puddle_mask

        normal_puddle = self._generate_ripple_normals(
            puddle_mask.shape[0], puddle_mask.shape[1], time, 
            cell_size=32.0, intensity_min=0.8, intensity_max=1.5,
            device=puddle_mask.device, dtype=torch.float32
        )

        ground_label = self.ground_labels[idx]
        pts_ground = self.perframe_world_pts[idx][:,ground_label]  # (3, M)

        origin = self.global_height_meta["origin"]
        resolution = self.global_height_meta["resolution"]

        xs = pts_ground[0, :]  # (M,)
        zs = pts_ground[2, :]  # (M,)
        us = ((xs - origin[0]) / resolution).long()
        vs = ((zs - origin[1]) / resolution).long()

        # select value from puddle_mask and normal_puddle
        puddle_at_pts = puddle_mask[vs, us]
        ripple_normals_at_pts = normal_puddle[:, vs, us]
        selected_normals = normal_map[:, mask_2d][:, ground_label]  # (3, M)

        puddle_normal = self.perturb_normal(
            selected_normals, ripple_normals_at_pts, strength=0.9
        )

        mixed_normal = F.normalize(_mix(selected_normals, 
                                            puddle_normal, 
                                            puddle_at_pts), 
                                            dim=0, 
                                            eps=1e-12)
        wet_roughness = 0.0
        global_wet_factor = 0.2

        sky2d = torch.as_tensor(sky_mask, device=roughness_map.device).to(torch.bool)
        non_sky = ~sky2d

        roughness_map[:, non_sky] = _mix(
            roughness_map[:, non_sky],
            torch.full_like(roughness_map[:, non_sky], wet_roughness),
            torch.full_like(roughness_map[:, non_sky], global_wet_factor),
        )

        w = puddle_at_pts.clamp(0.0, 1.0)

        r_sel = roughness_map[:, mask_2d][:, ground_label]
        r_sel_wet = _mix(
            r_sel,
            torch.full_like(r_sel, wet_roughness),
            torch.full_like(r_sel, 0.8)
        )
        # roughness_map[:, mask_2d][:, ground_label] = r_sel_wet

        c_sel = basecolor_map[:, mask_2d][:, ground_label]
        ny_ripple = ripple_normals_at_pts[1, :]
        rip_map = (1.0 + ny_ripple).clamp(0.0, 1.0)
        w_rip = (rip_map == 1.0).clamp(0.0, 1.0)
        ripple_white = torch.tensor(
            [0.92, 0.96, 1.00],
            dtype=c_sel.dtype,
            device=c_sel.device,
        ).view(3, 1).expand_as(c_sel)
        c_sel_wet = _mix(c_sel, ripple_white, w_rip.view(1, -1) * 0.8 * w).clamp(0.0, 1.0)

        mask_flat = mask_2d.reshape(-1)
        sel_flat_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        ground_flat_idx = sel_flat_idx[ground_label]

        basecolor_map.reshape(3, -1)[:, ground_flat_idx] = c_sel_wet
        roughness_map.reshape(1, -1)[:, ground_flat_idx] = r_sel_wet
        normal_map.reshape(3, -1)[:, ground_flat_idx] = mixed_normal



        self._paint_raindrops_gbuffers(
            idx, basecolor_map, roughness_map, normal_map, depth_map
        )
