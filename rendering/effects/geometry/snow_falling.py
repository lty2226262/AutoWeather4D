"""World-space falling snow particle simulation."""

from __future__ import annotations

import numpy as np
import torch


class SnowFallingParticlesMixin:
    """Mixin providing falling-snow particle state and per-frame updates."""

    def init_falling_snow_particles(
        self,
        num_particles: int,
        box_size: float,
        gravity: float,
        wind_x: float,
        wind_z: float,
        radius: float,
        seed: int,
        pose: torch.Tensor | None = None,
    ) -> None:
        """Seed world-space falling snow particles around the scene.

        Args:
            num_particles: Number of particles to allocate.
            box_size: Simulation box extent in world units.
            gravity: Per-frame downward displacement.
            wind_x: Per-frame wind along world X.
            wind_z: Per-frame wind along world Z.
            radius: Particle radius in world units (stored for callers).
            seed: RNG seed for initial placement.
            pose: Optional camera pose used to center the spawn volume.
        """
        self.falling_snow_num = int(num_particles)
        self.falling_snow_box_size = float(box_size)
        self.falling_snow_gravity = float(gravity)
        self.falling_snow_wind_x = float(wind_x)
        self.falling_snow_wind_z = float(wind_z)
        self.falling_snow_radius = float(radius)
        self.falling_snow_seed = int(seed)

        if pose is not None:
            scene_center = pose[:3, 3].detach().cpu().numpy()
        elif self.global_metaball_centers is not None and self.global_metaball_centers.shape[0] > 0:
            centersnp = self.global_metaball_centers.detach().cpu().numpy()
            scene_center = centersnp.mean(axis=0)
        else:
            scene_center = np.array([0.0, 0.0, 0.0], dtype="float32")
        self.falling_snow_scene_center = scene_center

        rng = np.random.RandomState(seed)
        half = box_size * 0.5
        x = rng.uniform(scene_center[0] - half, scene_center[0] + half, size=num_particles)
        z = rng.uniform(scene_center[2] - half, scene_center[2] + half, size=num_particles)
        y = rng.uniform(scene_center[1], scene_center[1] + box_size, size=num_particles)
        self.falling_snow_particles = np.stack([x, y, z], axis=1).astype("float32")

    def update_falling_snow_particles(self, pose: torch.Tensor) -> None:
        """Advance falling snow particles for the current camera pose.

        Args:
            pose: Camera pose matrix, shape ``(4, 4)``.
        """
        if self.falling_snow_particles is None:
            self.init_falling_snow_particles(
                self.falling_snow_num or 6000,
                self.falling_snow_box_size,
                self.falling_snow_gravity,
                self.falling_snow_wind_x,
                self.falling_snow_wind_z,
                self.falling_snow_radius,
                self.falling_snow_seed,
                pose=pose,
            )
            return

        cam_center = pose[:3, 3].detach().cpu().numpy()
        self.falling_snow_particles[:, 1] -= self.falling_snow_gravity
        self.falling_snow_particles[:, 0] += self.falling_snow_wind_x
        self.falling_snow_particles[:, 2] += self.falling_snow_wind_z

        half = self.falling_snow_box_size * 0.5
        below = self.falling_snow_particles[:, 1] < (cam_center[1] - half)
        if below.any():
            rng = np.random.RandomState(self.falling_snow_seed + 7)
            self.falling_snow_particles[below, 1] = rng.uniform(
                cam_center[1] + half, cam_center[1] + self.falling_snow_box_size, size=below.sum()
            ).astype("float32")
            self.falling_snow_particles[below, 0] = rng.uniform(
                cam_center[0] - half, cam_center[0] + half, size=below.sum()
            ).astype("float32")
            self.falling_snow_particles[below, 2] = rng.uniform(
                cam_center[2] - half, cam_center[2] + half, size=below.sum()
            ).astype("float32")

        if self.falling_snow_scene_center is not None:
            dist = np.linalg.norm(self.falling_snow_scene_center - cam_center)
            if dist > half * 0.8:
                rng = np.random.RandomState(self.falling_snow_seed + 17)
                x = rng.uniform(cam_center[0] - half, cam_center[0] + half, size=self.falling_snow_num)
                z = rng.uniform(cam_center[2] - half, cam_center[2] + half, size=self.falling_snow_num)
                y = rng.uniform(cam_center[1], cam_center[1] + self.falling_snow_box_size, size=self.falling_snow_num)
                self.falling_snow_particles = np.stack([x, y, z], axis=1).astype("float32")
                self.falling_snow_scene_center = cam_center

    def get_falling_snow_particles(self) -> np.ndarray | None:
        """Return the current falling-snow particle array, if initialized."""
        return self.falling_snow_particles
