"""Per-frame height maps and puddle masks derived from scene geometry."""

from __future__ import annotations

import math
import os

import imageio.v2 as imageio
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image


class HeightMapGenerator:
    """Build static/dynamic height maps and per-frame puddle visibility masks."""

    def __init__(
        self,
        urban_scene,
        depth_min: float = 0.1,
        depth_max: float = 200.0,
        water_level_height: float = 0.3,
        visualization: bool = True,
        resolution: float = 0.01,
    ) -> None:
        """Initialize height and puddle data for all frames in a scene.

        Args:
            urban_scene: Source scene with pose, depth, and position accessors.
            depth_min: Minimum valid depth for pixel inclusion.
            depth_max: Maximum valid depth for pixel inclusion.
            water_level_height: Target water surface height in world space.
            visualization: Write preview videos under ``height_map_vis/`` when ``True``.
            resolution: Grid resolution for global height maps, in meters.
        """
        self.resolution = resolution
        all_poses = []
        avail_masks = []
        perframe_world = []
        ground_labels = []
        ground_pts = []
        for fi in range(len(urban_scene)):
            pose = urban_scene.get_pose(fi)
            all_poses.append(pose)
            position = urban_scene.get_position(fi)
            depth = urban_scene.get_depth(fi)
            avail_mask = (depth > depth_min) & (depth < depth_max)

            mask2d = avail_mask[0]

            pos_flat = position[:,mask2d]
            pts_w_flat = (pose[:3, :3] @ pos_flat) + pose[:3, 3:4]
            ground_label = self.get_ground(pts_w_flat)
            ground_labels.append(ground_label)
            ground_pts.append(pts_w_flat[:,ground_label])
            perframe_world.append(pts_w_flat)
            avail_masks.append(mask2d)

        ground_labels2 = []
        ground_pts2 = []
        for fi in range(len(urban_scene)):
            pts_w_flat = perframe_world[fi]
            ground_label2 = self.get_ground2(pts_w_flat)
            ground_labels2.append(ground_label2)
            ground_pts2.append(pts_w_flat[:,ground_label2])

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
            ground_pts2[fi] = perframe_world_new[:,ground_labels2[fi]]


        if visualization:
            out_dir = "height_map_vis"
            os.makedirs(out_dir, exist_ok=True)
            video_path = os.path.join(out_dir, "ground_removal.mp4")
            fps = 8
            writer = imageio.get_writer(video_path, fps=fps, quality=8)

            for fi in range(len(urban_scene)):
                gl = ground_labels[fi]
                H, W = avail_masks[fi].shape
                rgb = np.zeros((H, W, 3), dtype=np.uint8)
                rgb[avail_masks[fi].cpu().numpy()] = gl.unsqueeze(-1).cpu().numpy() * np.array([[255, 255, 255]], dtype=np.uint8)  # white for ground
                writer.append_data(rgb)
            writer.close()
            print(f"Ground removal video written to: {video_path}")

        lowest_y = torch.max(torch.cat(perframe_world, dim=1)[1, ...])
        print(f"Lowest Y in world coordinates: {lowest_y.item():.2f}")
        lowest_cam = torch.max(torch.stack([p[:3,3] for p in all_poses], dim=0)[:,1])
        deepest_water_level = lowest_y - lowest_cam - 1e-2
        if water_level_height > deepest_water_level:
            print(f"Warning: specified water_level_height {water_level_height:.2f} exceeds deepest possible {deepest_water_level:.2f}. Clamping.")
            water_level_height = deepest_water_level

        self.build_global_static_height_map(
            ground_pts, all_poses, resolution=resolution, lowest_y=lowest_y
        )
        self.build_dynamic_perframe_height_maps(
            perframe_world, dt=0.1, resolution=resolution, lowest_y=lowest_y
        )

        self.perframe_world = perframe_world
        self.ground_labels = ground_labels
        self.ground_labels2 = ground_labels2
        self.ground_pts = ground_pts
        self.ground_pts2 = ground_pts2
        self.all_poses = all_poses
        self.avail_masks = avail_masks
        if visualization:
            preview_max_side = 1024
            vmin = 0.0
            denom = 10.0  # or compute global_max_dynamic_height if available
            video_writer = imageio.get_writer(os.path.join(out_dir, "dynamic_height_maps.mp4"), fps=fps, quality=8)

            for i, hm in enumerate(self.dynamic_height_maps):
                vals = hm

                Hh, Ww = vals.shape
                # normalized in [0,1]
                norm = (vals - vmin) / (denom if denom != 0 else 1.0)
                norm = np.clip(norm, 0.0, 1.0)

                # downsample normalized map for preview (operate on small array -> fast)
                scale = min(1.0, float(preview_max_side) / max(Hh, Ww))
                if scale < 1.0:
                    new_h = max(1, int(round(Hh * scale)))
                    new_w = max(1, int(round(Ww * scale)))
                    # use PIL to resize the single-channel normalized map (uint8)
                    norm_img = Image.fromarray((norm * 255).astype(np.uint8))
                    norm_img = norm_img.resize((new_w, new_h), resample=Image.BILINEAR)
                    norm_small = np.asarray(norm_img).astype(np.float32) / 255.0
                else:
                    norm_small = norm

                # apply colormap (returns RGBA float), drop alpha
                cmap = cm.get_cmap("jet")
                mapped = cmap(norm_small)[..., :3]
                rgb = (mapped * 255).astype(np.uint8)

                video_writer.append_data(rgb)

            video_writer.close()
            print(f"Saved {len(self.dynamic_height_maps)} preview frames to: {out_dir}")

        self.height_maps = []
        self.puddle_masks = []
        for fi in range(len(urban_scene)):
            self.height_maps.append(
                self.generate_height_map(
                    all_point=perframe_world[fi],  # (3,H,W)
                    avail_mask=avail_masks[fi],
                    lowest_y=lowest_y
                )
            )
            self.puddle_masks.append(
                self.generate_puddle_mask(
                    all_point=perframe_world[fi],
                    avail_mask=avail_masks[fi],
                    lowest_y=lowest_y,
                    water_level_height=water_level_height,
                    pose=all_poses[fi],
                )
            )

        if visualization:
            out_dir = "height_map_vis"
            os.makedirs(out_dir, exist_ok=True)
            video_path = os.path.join(out_dir, "height_maps.mp4")
            fps = 8
            writer = imageio.get_writer(video_path, fps=fps, quality=8)
            writer_puddle = imageio.get_writer(os.path.join(out_dir, "puddle_masks.mp4"), fps=fps, quality=8)

            for fi, height_map in enumerate(self.height_maps):
                hm = height_map.clone()
                invalid = torch.isinf(hm)   # True where no data
                valid_mask = ~invalid

                Hh, Ww = hm.shape
                rgb = np.zeros((Hh, Ww, 3), dtype=np.uint8)

                if valid_mask.any():
                    vals = hm[valid_mask].cpu().numpy().astype(np.float32)
                    # normalize using global range (lowest_y-highest_y)
                    vmin = float(lowest_y - water_level_height)
                    denom = water_level_height
                    norm = (vals - vmin) / denom

                    cmap = cm.get_cmap("jet")
                    mapped = cmap(norm)[:, :3]
                    rgb[valid_mask.cpu().numpy()] = (mapped * 255).astype(np.uint8)

                # invalid pixels remain black
                writer.append_data(rgb)
                pm = self.puddle_masks[fi].clone()
                pm_rgb = np.zeros((Hh, Ww, 3), dtype=np.uint8)
                pm_rgb[pm.cpu().numpy()] = np.array([0, 0, 255], dtype=np.uint8)  # blue for puddles
                writer_puddle.append_data(pm_rgb)

            writer.close()
            writer_puddle.close()
            print(f"Height map video written to: {video_path}")
            print(f"Puddle mask video written to: {os.path.join(out_dir, 'puddle_masks.mp4')}")
    
    def estimate_gravity_and_ref_pose(
        self,
        ground_pts: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Estimate a gravity-aligned reference pose from ground point clouds.

        Args:
            ground_pts: Per-frame ground points, each shaped ``(3, N)``.

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

    def generate_height_map(
        self,
        all_point: torch.Tensor,
        avail_mask: torch.Tensor,
        lowest_y: torch.Tensor,
    ) -> torch.Tensor:
        """Build a per-frame height map from world-space points.

        Args:
            all_point: World points shaped ``(3, N)`` for valid pixels.
            avail_mask: Boolean mask shaped ``(H, W)``.
            lowest_y: Reference ground height in world space.

        Returns:
            Height map shaped ``(H, W)``; invalid pixels are ``inf``.
        """
        H, W = avail_mask.shape
        height_map = torch.full((H, W), float('inf'), dtype=torch.float32, device=all_point.device)
        if avail_mask.any():
            # use Y channel (index 1) of all_point, only where mask is True
            y_channel = all_point[1]  # (H, W)
            height_map[avail_mask] = (lowest_y - y_channel)
        
        return height_map

    def generate_puddle_mask(
        self,
        all_point: torch.Tensor,
        avail_mask: torch.Tensor,
        lowest_y: torch.Tensor,
        water_level_height: float,
        pose: torch.Tensor,
    ) -> torch.Tensor:
        """Build a per-frame puddle visibility mask via water-plane ray casting.

        Args:
            all_point: World points shaped ``(3, N)`` for valid pixels.
            avail_mask: Boolean mask shaped ``(H, W)``.
            lowest_y: Reference ground height in world space.
            water_level_height: Water surface height in world space.
            pose: Camera pose shaped ``(4, 4)``.

        Returns:
            Boolean mask shaped ``(H, W)`` where ``True`` marks visible puddles.
        """
        cam_world = pose[:3, 3:4]
        H, W = avail_mask.shape

        dir_world = all_point - cam_world  # (3, avaliable_mask_count)
        dir_y = dir_world[1, :]

        # avoid parallel rays
        valid_dir = dir_y.abs() > 1e-8
        valid_flat_mask = torch.zeros((H, W), dtype=torch.bool, device=all_point.device)
        valid_flat_mask[avail_mask] = valid_dir

        cam_y = cam_world[1, 0]

        t_params = torch.zeros((H, W), dtype=all_point.dtype, device=all_point.device)
        t_params[avail_mask & valid_flat_mask] = (lowest_y - water_level_height - cam_y) / dir_y[valid_dir]

        in_front = t_params > 0

        p_world = cam_world + dir_world * t_params.view(1, H*W)[:, avail_mask.view(-1)]

        pose_inv = torch.linalg.inv(pose)
        R_w2c = pose_inv[:3, :3]
        t_w2c = pose_inv[:3, 3:4]
        p_cam_flat = (R_w2c @ p_world) + t_w2c

        intersect_z = torch.full((H, W), float('inf'), dtype=all_point.dtype, device=all_point.device)
        intersect_z[avail_mask] = p_cam_flat[2, :]

        orig_z = all_point[2, :]

        visible = torch.zeros((H, W), dtype=torch.bool, device=all_point.device)

        visible[avail_mask] = in_front[avail_mask] & valid_dir & (intersect_z[avail_mask] < orig_z - 1e-8) & (intersect_z[avail_mask] > 0)

        return visible

    @staticmethod
    def get_ground(pts: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def get_ground2(
        pts: np.ndarray | torch.Tensor,
        ground_thickness: float = 2.0,
        grid_size: float | None = None,
        min_points_per_cell: int = 20,
        use_gpu: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Detect ground points with a fast grid method on the x-z plane.

        Args:
            pts: World points shaped ``(3, N)`` or ``(N, 3)``.
            ground_thickness: Height band above per-cell max ``y`` treated as ground.
            grid_size: Grid cell size in meters; auto-selected when ``None``.
            min_points_per_cell: Minimum points required for a cell to be valid.
            use_gpu: Move tensors to CUDA when available.
            verbose: Print diagnostic statistics.

        Returns:
            Boolean mask of shape ``(N,)`` where ``True`` marks ground points.
        """
        if isinstance(pts, np.ndarray):
            pts_t = torch.from_numpy(pts).to(dtype=torch.float32)
        elif isinstance(pts, torch.Tensor):
            pts_t = pts.to(dtype=torch.float32)
        else:
            raise TypeError("pts must be np.ndarray or torch.Tensor")
        
        
        if pts_t.dim() != 2 or 3 not in pts_t.shape:
            raise ValueError("pts should be (3,N) or (N,3)")
        if pts_t.shape[0] == 3:
            pts_t = pts_t.t().contiguous()
        
        
        if use_gpu and torch.cuda.is_available():
            pts_t = pts_t.cuda(non_blocking=True)
        device = pts_t.device
        
        
        N = pts_t.shape[0]
        if N < 100:
            return torch.zeros((N,), dtype=torch.bool, device=device)
        
        
        x = pts_t[:, 0]
        y = pts_t[:, 1]
        z = pts_t[:, 2]
        
        
        y_min = torch.min(y)
        y_max = torch.max(y)
        y_range = (y_max - y_min).item()
        
        
        if verbose:
            print("\n=== Large Range Ground Detection (fast) ===")
            print(f"Y range: [{y_min.item():.2f}, {y_max.item():.2f}] span: {y_range:.2f}m, N={N}")

        x_min, x_max = torch.min(x), torch.max(x)
        z_min, z_max = torch.min(z), torch.max(z)
        if grid_size is None:
            extent_x = (x_max - x_min).item()
            extent_z = (z_max - z_min).item()
            gs_auto = max(2.0, min(10.0, min(extent_x, extent_z) / 20.0))
            grid_size = float(gs_auto)
        
        
        nx = max(1, int(math.floor((x_max - x_min).item() / grid_size)) + 1)
        nz = max(1, int(math.floor((z_max - z_min).item() / grid_size)) + 1)
        if verbose:
            print(f"Method3-Grid: {nx}x{nz} (cell={grid_size:.1f}m)")

        ix = torch.floor((x - x_min) / grid_size).long().clamp_(0, nx - 1)
        iz = torch.floor((z - z_min) / grid_size).long().clamp_(0, nz - 1)
        cell_id = ix * nz + iz
        num_cells = nx * nz
        
        
        cell_count = torch.bincount(cell_id, minlength=num_cells)

        neg_inf = torch.tensor(float("-inf"), device=device)
        cell_ymax = torch.full((num_cells,), neg_inf, device=device)
        cell_ymax.scatter_reduce_(0, cell_id, y, reduce="amax", include_self=True)

        cell_thr = cell_ymax - float(ground_thickness)
        sparse_cells = cell_count < int(min_points_per_cell)
        if sparse_cells.any():
            cell_thr[sparse_cells] = float("+inf")

        thr_per_point = cell_thr[cell_id]
        mask_grid = y >= thr_per_point
        
        
        if verbose:
            removed = mask_grid.sum().item()
            valid_cells = (cell_count >= min_points_per_cell).sum().item()
            valid_ymax = cell_ymax[~sparse_cells]
            if valid_ymax.numel() > 0:
                avg_ground = torch.mean(valid_ymax).item()
                std_ground = torch.std(valid_ymax).item()
                print(f"  valid_cells={valid_cells}, grid_ground={avg_ground:.2f}±{std_ground:.2f}, "
                      f"removed={removed} ({removed*100.0/N:.1f}%)")
            else:
                print(f"  valid_cells={valid_cells}, removed={removed} ({removed*100.0/N:.1f}%)")
        
        
        mask_final = mask_grid.clone()

        if verbose:
            total = int(mask_final.sum().item())
            remain_y = y[~mask_final]
            print(f"\n=== Final ===")
            print(f"Total removed: {total}/{N} ({total*100.0/N:.1f}%)")
            if remain_y.numel() > 0:
                print(f"Remaining Y range: [{remain_y.min().item():.2f}, {remain_y.max().item():.2f}]")
        
        
        return mask_final

    def build_global_static_height_map(
        self,
        ground_pts: list[torch.Tensor],
        all_poses: list[torch.Tensor],
        resolution: float = 0.01,
        lowest_y: torch.Tensor | None = None,
    ) -> None:
        """Build a scene-wide static height map from aggregated ground points."""
        all_ground_pts = torch.cat(ground_pts, dim=1).detach().clone()  # (3, N)
        all_camera_pts = torch.cat([p[:3, 3:4] for p in all_poses], dim=1).detach().clone()  # (3, num_frames)
       
        all_ground_pts[1, :] = -all_ground_pts[1, :]
        all_ground_pts[2, :] = -all_ground_pts[2, :]

        all_camera_pts[1, :] = -all_camera_pts[1, :]
        all_camera_pts[2, :] = -all_camera_pts[2, :]

        
        x_min = float(all_ground_pts[0].min().item())
        x_max = float(all_ground_pts[0].max().item())
        z_min = float(all_ground_pts[2].min().item())
        z_max = float(all_ground_pts[2].max().item())
        cam_x_min = float(all_camera_pts[0].min().item())
        cam_x_max = float(all_camera_pts[0].max().item())
        cam_z_min = float(all_camera_pts[2].min().item())
        cam_z_max = float(all_camera_pts[2].max().item())

        x_min = min(x_min, cam_x_min)
        x_max = max(x_max, cam_x_max)
        z_min = min(z_min, cam_z_min)
        z_max = max(z_max, cam_z_max)

        Ww = int(np.ceil((x_max - x_min) / resolution))
        Hh = int(np.ceil((z_max - z_min) / resolution))
        print(f"Global height map size: {Hh} x {Ww} (HxW)")
        self.x_min = x_min
        self.x_max = x_max
        self.z_min = z_min
        self.z_max = z_max


        self.global_height_map = torch.full((Hh, Ww), -lowest_y, dtype=torch.float32, device=all_ground_pts.device)

        # convert all_ground_pts to height map indices
        xs = all_ground_pts[0, :]  # (N,)
        zs = all_ground_pts[2, :]  # (N,)
        us = ((xs - x_min) / resolution).long()
        vs = ((zs - z_min) / resolution).long()

        valid_mask = (us >= 0) & (us < Ww) & (vs >= 0) & (vs < Hh)
        us = us[valid_mask]
        vs = vs[valid_mask]
        ys = all_ground_pts[1, :][valid_mask]

        device = self.global_height_map.device
        lin_idx = (vs.to(device=device) * Ww + us.to(device=device)).to(dtype=torch.long, device=device)
        new_vals = ys.to(dtype=torch.float32, device=device)
        height_flat = self.global_height_map.view(-1)
        height_flat.scatter_reduce_(0, lin_idx, new_vals, reduce='amax', include_self=True)
        self.global_height_map.sub_(-lowest_y)

    def build_dynamic_perframe_height_maps(
        self,
        perframe_world: list[torch.Tensor],
        dt: float = 0.1,
        resolution: float = 0.01,
        lowest_y: torch.Tensor | None = None,
    ) -> None:
        """Build per-frame dynamic height maps from moving scene points."""
        self.t = dt * torch.arange(len(perframe_world), dtype=torch.float32, device=perframe_world[0].device)
        self.dynamic_height_maps = []
        for fi in range(len(perframe_world)):
            pts = perframe_world[fi].detach().clone()
            pts[1, :] = -pts[1, :]
            pts[2, :] = -pts[2, :]
            static_hm = self.global_height_map.detach().clone()
            if pts.shape[1] > 0:
                xs = pts[0, :]  # (N,)
                ys = pts[1, :] + lowest_y
                zs = pts[2, :]
                device = static_hm.device
                W = static_hm.shape[1]
                H = static_hm.shape[0]

                # compute integer cell indices (floor division)
                us = ((xs - self.x_min) / resolution).to(torch.long)
                vs = ((zs - self.z_min) / resolution).to(torch.long)

                # mask valid indices inside the grid
                in_range = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
                if in_range.any():
                    us = us[in_range].to(device=device)
                    vs = vs[in_range].to(device=device)
                    vals = ys[in_range].to(dtype=torch.float32, device=device)

                    lin_idx = (vs * W + us).to(dtype=torch.long, device=device)  # (M,)
                    height_flat = static_hm.view(-1)  # (H*W,)

                    M = lin_idx.shape[0]
                    chunk = 4_000_000
                    if M <= chunk:
                        height_flat.scatter_reduce_(0, lin_idx, vals, reduce='amax', include_self=True)
                    else:
                        for i0 in range(0, M, chunk):
                            i1 = min(M, i0 + chunk)
                            height_flat.scatter_reduce_(0, lin_idx[i0:i1], vals[i0:i1], reduce='amax', include_self=True)
            self.dynamic_height_maps.append(static_hm.cpu().numpy())
