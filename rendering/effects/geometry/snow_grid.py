"""Procedural grid ground snow map generation and sampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm


class SnowGridMixin:
    """Mixin providing procedural grid snow map build and world-space sampling."""

    def ensure_grid_snow_initialized(self) -> None:
        """Generate the procedural grid snow map once scene bounds are known."""
        if (not self.grid_snow_enabled) or self.grid_snow_initialized:
            return
        if self.height_map_generator is None:
            return
        x_min = float(self.height_map_generator.x_min)
        x_max = float(self.height_map_generator.x_max)
        z_min = float(-self.height_map_generator.z_max)
        z_max = float(-self.height_map_generator.z_min)
        world_bounds = (x_min, x_max, z_min, z_max)
        print(f"Generating grid snow map (resolution={self.grid_snow_resolution})...")
        self.generate_grid_snow_map(world_bounds, resolution=self.grid_snow_resolution)

    def generate_grid_snow_map(
        self,
        world_bounds: tuple[float, float, float, float],
        resolution: int | None = None,
    ) -> torch.Tensor | None:
        """Build a procedural ground snow density map over world XZ bounds.

        Args:
            world_bounds: ``(x_min, x_max, z_min, z_max)`` in world coordinates.
            resolution: Grid resolution; defaults to ``grid_snow_resolution``.

        Returns:
            Snow density map tensor, or ``None`` when grid snow is disabled.
        """
        if not self.grid_snow_enabled:
            return None

        if resolution is None:
            resolution = self.grid_snow_resolution

        x_min, x_max, z_min, z_max = world_bounds
        device = self.device

        x_coords = torch.linspace(x_min, x_max, resolution, device=device)
        z_coords = torch.linspace(z_min, z_max, resolution, device=device)
        X, Z = torch.meshgrid(x_coords, z_coords, indexing="ij")

        with torch.random.fork_rng([device], enabled=True):
            torch.manual_seed(int(self.grid_snow_seed))

            num_centers = int(resolution * resolution * 0.0005)
            num_centers = max(20, min(num_centers, 500))

            centers_x = torch.rand(num_centers, device=device) * resolution
            centers_z = torch.rand(num_centers, device=device) * resolution
            centers_value = torch.rand(num_centers, device=device)

            ii, jj = torch.meshgrid(
                torch.arange(resolution, device=device, dtype=torch.float32),
                torch.arange(resolution, device=device, dtype=torch.float32),
                indexing="ij",
            )

            voronoi_map = torch.zeros((resolution, resolution), device=device)
            voronoi_pbar = tqdm(total=num_centers, desc="Generating Voronoi map", unit="center", leave=False)
            for i in range(num_centers):
                dx = ii - centers_x[i]
                dz = jj - centers_z[i]
                dist = torch.sqrt(dx**2 + dz**2)
                weight = torch.exp(-dist * 0.05)
                voronoi_map += weight * centers_value[i]
                voronoi_pbar.update(1)
            voronoi_pbar.close()

            voronoi_map = voronoi_map / (voronoi_map.max() + 1e-6)

            fbm = torch.zeros((resolution, resolution), device=device)
            amplitude = 1.0
            frequency = 0.5
            for _octave in range(4):
                noise = (
                    torch.sin(X * frequency + torch.rand(1, device=device) * 10)
                    * torch.cos(Z * frequency * 1.3 + torch.rand(1, device=device) * 10)
                    + torch.sin(X * frequency * 0.7 + torch.rand(1, device=device) * 10)
                    * torch.cos(Z * frequency * 0.9 + torch.rand(1, device=device) * 10)
                )
                fbm += noise * amplitude
                amplitude *= 0.5
                frequency *= 2.0

            fbm = (fbm - fbm.min()) / (fbm.max() - fbm.min() + 1e-6)

            blob_map = torch.zeros((resolution, resolution), device=device)
            num_blobs = int(resolution * 0.5)

            blob_pbar = tqdm(total=num_blobs, desc="Generating blob map", unit="blob", leave=False)
            for _blob_idx in range(num_blobs):
                cx = torch.rand(1, device=device) * resolution
                cz = torch.rand(1, device=device) * resolution
                rx = torch.rand(1, device=device) * resolution * 0.03 + resolution * 0.01
                rz = rx * (torch.rand(1, device=device) * 1.5 + 0.5)
                angle = torch.rand(1, device=device) * torch.pi

                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)
                dx = ii - cx
                dz = jj - cz
                dx_rot = dx * cos_a - dz * sin_a
                dz_rot = dx * sin_a + dz * cos_a

                dist_norm = (dx_rot / rx) ** 2 + (dz_rot / rz) ** 2
                blob = torch.exp(-dist_norm * 2.0)

                intensity = torch.rand(1, device=device) * 0.7 + 0.3
                blob_map = torch.maximum(blob_map, blob * intensity)
                blob_pbar.update(1)
            blob_pbar.close()

            combined = voronoi_map * 0.3 + fbm * 0.4 + blob_map * 0.3
            detail_noise = torch.rand((resolution, resolution), device=device)
            combined = combined * 0.9 + detail_noise * 0.1

            threshold = 0.6
            steepness = 8.0
            snow_density = torch.sigmoid((combined - threshold) * steepness)

            sparse_mask = torch.rand((resolution, resolution), device=device) > 0.6
            snow_density = snow_density * sparse_mask.float()
            snow_density = snow_density * float(self.grid_snow_density)

            if snow_density.max() > 0:
                kernel_sizes = [3, 5, 7]
                blurred_versions = []

                for ksize in kernel_sizes:
                    pad = ksize // 2
                    x_k = torch.arange(ksize, device=device, dtype=torch.float32) - pad
                    gauss_1d = torch.exp(-x_k**2 / (2 * (ksize / 4) ** 2))
                    gauss_1d = gauss_1d / gauss_1d.sum()
                    gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
                    kernel = gauss_2d.unsqueeze(0).unsqueeze(0)

                    padded = F.pad(
                        snow_density.unsqueeze(0).unsqueeze(0),
                        (pad, pad, pad, pad),
                        mode="replicate",
                    )
                    blurred = F.conv2d(padded, kernel, stride=1, padding=0)
                    blurred_versions.append(blurred.squeeze())

                snow_density = (
                    snow_density * 0.5
                    + blurred_versions[0] * 0.25
                    + blurred_versions[1] * 0.15
                    + blurred_versions[2] * 0.1
                )

            min_density = 0.1
            snow_density = torch.where(
                snow_density > min_density,
                snow_density,
                torch.zeros_like(snow_density),
            )
            snow_density = torch.clamp(snow_density, 0.0, 1.0) * float(self.grid_snow_density)

            if (snow_density > 0).sum() == 0:
                for _ in range(10):
                    cx = torch.rand(1, device=device) * resolution
                    cz = torch.rand(1, device=device) * resolution
                    r = resolution * 0.02
                    dx = ii - cx
                    dz = jj - cz
                    dist = torch.sqrt(dx**2 + dz**2)
                    fallback_blob = torch.exp(-((dist / r) ** 2))
                    snow_density = torch.maximum(
                        snow_density,
                        fallback_blob * float(self.grid_snow_density) * 0.8,
                    )

            snow_map = snow_density

        self.grid_snow_map = snow_map
        self.grid_snow_bounds = (x_min, x_max, z_min, z_max)
        self.grid_snow_resolution = resolution
        self.grid_snow_initialized = True

        return self.grid_snow_map

    def sample_grid_snow(self, world_positions: torch.Tensor) -> torch.Tensor:
        """Bilinearly sample grid snow height at world-space positions.

        Args:
            world_positions: World XYZ positions, shape ``(N, 3)``.

        Returns:
            Sampled snow heights, shape ``(N,)``.
        """
        if not self.grid_snow_enabled or not self.grid_snow_initialized:
            return torch.zeros(world_positions.shape[0], device=self.device)

        x_min, x_max, z_min, z_max = self.grid_snow_bounds
        resolution = self.grid_snow_resolution

        x_coords = (world_positions[:, 0] - x_min) / (x_max - x_min) * (resolution - 1)
        z_coords = (world_positions[:, 2] - z_min) / (z_max - z_min) * (resolution - 1)
        x_coords = torch.clamp(x_coords, 0, resolution - 1)
        z_coords = torch.clamp(z_coords, 0, resolution - 1)

        x0 = torch.floor(x_coords).long()
        x1 = torch.clamp(x0 + 1, 0, resolution - 1)
        z0 = torch.floor(z_coords).long()
        z1 = torch.clamp(z0 + 1, 0, resolution - 1)

        fx = x_coords - x0.float()
        fz = z_coords - z0.float()

        c00 = self.grid_snow_map[z0, x0]
        c01 = self.grid_snow_map[z1, x0]
        c10 = self.grid_snow_map[z0, x1]
        c11 = self.grid_snow_map[z1, x1]

        c0 = c00 * (1 - fz) + c01 * fz
        c1 = c10 * (1 - fz) + c11 * fz
        snow_heights = c0 * (1 - fx) + c1 * fx

        return snow_heights * self.grid_snow_height
