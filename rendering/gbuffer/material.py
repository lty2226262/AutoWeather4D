"""G-buffer material loading and per-frame material access."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  — registers HDF5 compression filters
import numpy as np
import pandas as pd
import torch

_TRACKING_CENTER_COLS = (
    "3d_front_center_x",
    "3d_front_center_y",
    "3d_front_center_z",
)


def _extract_dataset_id_from_path(file_path: str | Path) -> int | None:
    """Extract a numeric dataset id suffix from a file path."""
    path = str(file_path)
    for pattern in (r"(\d{3})", r"(\d{2})", r"(\d{1,2})"):
        matches = re.findall(pattern, path)
        if matches:
            return int(matches[-1])
    return None


def _screen_rect_from_3d_box(
    center: np.ndarray,
    half_fb: np.ndarray,
    half_rl: np.ndarray,
    half_up: np.ndarray,
    K: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Project an oriented 3D box to pixel bounds ``(u_min, u_max, v_min, v_max)``."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u_min, u_max = float("inf"), float("-inf")
    v_min, v_max = float("inf"), float("-inf")
    for sfb in (-1, 1):
        for srl in (-1, 1):
            for sdu in (-1, 1):
                pt = center + sfb * half_fb + srl * half_rl + sdu * half_up
                if pt[2] <= 1e-6:
                    continue
                u = fx * pt[0] / pt[2] + cx
                v = fy * pt[1] / pt[2] + cy
                u_min, u_max = min(u_min, u), max(u_max, u)
                v_min, v_max = min(v_min, v), max(v_max, v)
    if u_min == float("inf"):
        return None
    return u_min, u_max, v_min, v_max


def _tracking_parquet_to_no_snow_mask(
    parquet_bytes: bytes | np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray | None:
    """Build a vehicle mask from per-frame tracking parquet bytes.

    Args:
        parquet_bytes: Serialized parquet payload for one frame.
        K: ``3x3`` camera intrinsics matrix.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Bool mask of shape ``(H, W)`` where ``True`` marks vehicle pixels, or
        ``None`` when tracking data is empty.

    Raises:
        ValueError: When parquet rows exist but required tracking columns are missing.
    """
    df = pd.read_parquet(io.BytesIO(bytes(parquet_bytes)))
    if df.empty:
        return None
    if not any(col in df.columns for col in _TRACKING_CENTER_COLS):
        raise ValueError(
            f"Tracking parquet missing required columns; expected one of {_TRACKING_CENTER_COLS}"
        )

    mask = np.zeros((height, width), dtype=np.bool_)
    for _, row in df.iterrows():
        cx_3d = row.get("3d_front_center_x", np.nan)
        cy_3d = row.get("3d_front_center_y", np.nan)
        cz_3d = row.get("3d_front_center_z", np.nan)
        if np.isnan(cx_3d) or np.isnan(cy_3d) or np.isnan(cz_3d) or cz_3d <= 1e-6:
            continue

        fb = np.array(
            [row.get("front_back_x", 0), row.get("front_back_y", 0), row.get("front_back_z", 0)],
            dtype=np.float64,
        )
        rl = np.array(
            [row.get("right_left_x", 0), row.get("right_left_y", 0), row.get("right_left_z", 0)],
            dtype=np.float64,
        )
        du = row.get("down_up_scale", 0)
        center = np.array([cx_3d, cy_3d, cz_3d])
        up = np.cross(fb, rl)
        n = np.linalg.norm(up) + 1e-12
        half_up = (up / n) * (float(du) * 0.5) if du == du else np.zeros(3)

        bounds = _screen_rect_from_3d_box(center, fb * 0.5, rl * 0.5, half_up, K)
        if bounds is None:
            continue
        u_min, u_max, v_min, v_max = bounds
        x_min = max(0, int(np.floor(u_min)))
        x_max = min(width - 1, int(np.ceil(u_max)))
        y_min = max(0, int(np.floor(v_min)))
        y_max = min(height - 1, int(np.ceil(v_max)))
        if x_min <= x_max and y_min <= y_max:
            mask[y_min : y_max + 1, x_min : x_max + 1] = True
    return mask


class Material:
    """Per-frame G-buffer maps used by the BRDF renderer.

    ``MaterialVideo`` may also attach ``pose``, ``sky_mask``, ``no_snow_mask``,
    and ``intrinsics`` after construction.
    """

    def __init__(
        self,
        albedo: torch.Tensor | None = None,
        albedo_is_srgb: bool = True,
        normal: torch.Tensor | None = None,
        metallic: torch.Tensor | None = None,
        roughness: torch.Tensor | None = None,
        position: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize per-frame material buffers.

        Args:
            albedo: Base color map in CHW layout.
            albedo_is_srgb: Whether ``albedo`` is encoded in sRGB space.
            normal: Normal map in CHW layout.
            metallic: Metallic map, shape ``(1, H, W)``.
            roughness: Roughness map, shape ``(1, H, W)``.
            position: Camera-space position map in CHW layout.
            device: Torch device for tensor operations.
        """
        self.device = torch.device(device)
        self.albedo_is_srgb = albedo_is_srgb
        self.albedo = albedo
        self.normal = normal
        self.roughness = roughness
        self.metallic = metallic
        self.position = position

    @staticmethod
    def srgb_to_linear(texture: torch.Tensor) -> torch.Tensor:
        """Convert an sRGB texture to linear space.

        Args:
            texture: sRGB values in CHW layout, range ``[0, 1]``.

        Returns:
            Linear RGB in the same layout and range.
        """
        texture = texture.clamp(0, 1)
        mask = texture <= 0.04045
        linear_texture = torch.zeros_like(texture)
        linear_texture[mask] = texture[mask] / 12.92
        linear_texture[~mask] = ((texture[~mask] + 0.055) / 1.055) ** 2.4
        return linear_texture.clamp(0, 1)

    @staticmethod
    def linear_to_srgb(texture: torch.Tensor) -> torch.Tensor:
        """Convert a linear texture to sRGB space.

        Args:
            texture: Linear RGB in CHW layout, range ``[0, 1]``.

        Returns:
            sRGB values in the same layout and range.
        """
        texture = texture.clamp(0, 1)
        mask = texture <= 0.0031308
        srgb_texture = torch.zeros_like(texture)
        srgb_texture[mask] = texture[mask] * 12.92
        srgb_texture[~mask] = 1.055 * torch.pow(texture[~mask], 1 / 2.4) - 0.055
        return srgb_texture.clamp(0, 1)

    def linear_albedo(self) -> torch.Tensor:
        """Return albedo converted to linear color space.

        Returns:
            Linear base color in CHW layout.

        Raises:
            AttributeError: If ``albedo`` was not set.
        """
        if self.albedo is None:
            raise AttributeError("Albedo map not set")
        if self.albedo_is_srgb:
            return self.srgb_to_linear(self.albedo)
        return self.albedo


class MaterialVideo:
    """Load and decode per-frame G-buffer data from a scene HDF5 file."""

    def __init__(
        self,
        input_path: str | Path,
        device: torch.device | str = "cpu",
        albedo_is_srgb: bool = True,
        fps: int = 8,
        verbose: bool = True,
    ) -> None:
        """Load and decode all frames from a scene HDF5 file.

        Args:
            input_path: Path to the scene ``.h5`` file.
            device: Torch device used for decoded tensors.
            albedo_is_srgb: Whether basecolor maps are stored as sRGB.
            fps: Nominal frame rate stored for downstream video writers.
            verbose: Whether to print per-frame decode progress.

        Raises:
            FileNotFoundError: If ``input_path`` does not exist.
        """
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.input_path}")

        self.device = torch.device(device)
        self.albedo_is_srgb = albedo_is_srgb
        self.fps = fps
        self.verbose = verbose

        self.dataset_id = _extract_dataset_id_from_path(self.input_path)

        with h5py.File(self.input_path, "r") as h5file:
            self._load_metadata(h5file)
            self.material_kwargs = self._decode_all_frames()

        self.intrinsics_tensor = torch.from_numpy(np.asarray(self.K, dtype=np.float32)).to(self.device)

    def _load_metadata(self, h5file: h5py.File) -> None:
        """Read static scene attributes and dataset handles from the HDF5 file."""
        self.height = int(h5file.attrs["height"])
        self.width = int(h5file.attrs["width"])
        self.K = np.asarray(h5file.attrs["intrinsics"], dtype=np.float32)

        self.reader_depth = h5file["depths"]
        self.reader_basecolor = h5file["basecolor"]
        self.reader_normal = h5file["normal"]
        self.reader_metallic = h5file["metallic"]
        self.reader_roughness = h5file["roughness"]
        self.pose_reader = h5file["pose"]
        self.reader_sky_mask = h5file["sky_mask"]
        self.reader_tracking = h5file["tracking"] if "tracking" in h5file else None

        self.frame_count = len(self.reader_depth)
        self._uv_grid = self._build_uv_grid()

    @staticmethod
    def _frame_key(frame_idx: int) -> str:
        """Format a zero-padded HDF5 frame key."""
        return f"{frame_idx:02d}"

    def _build_uv_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Precompute normalized camera-plane coordinates shared by every frame."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        u = (np.arange(self.width, dtype=np.float32) - cx) / fx
        v = (np.arange(self.height, dtype=np.float32) - cy) / fy
        return np.meshgrid(u, v)

    def _position_from_depth(self, depth: np.ndarray) -> np.ndarray:
        """Back-project depth into camera-space positions (x right, y down, z forward)."""
        u_grid, v_grid = self._uv_grid
        position = np.empty((self.height, self.width, 3), dtype=np.float32)
        position[..., 0] = u_grid * depth
        position[..., 1] = v_grid * depth
        position[..., 2] = depth
        return position

    def _to_device_tensor(self, array: np.ndarray, channels_first: bool = False) -> torch.Tensor:
        """Copy a numpy array to a torch tensor on ``self.device``.

        Two-dimensional arrays are expanded to shape ``(1, H, W)``.
        """
        if channels_first and array.ndim == 3:
            array = np.ascontiguousarray(array.transpose(2, 0, 1))
        else:
            array = np.ascontiguousarray(array)
        tensor = torch.from_numpy(array).to(self.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _decode_frame(self, frame_idx: int) -> dict[str, Any]:
        """Decode one frame from HDF5 into tensors and numpy masks."""
        key = self._frame_key(frame_idx)

        depth = self.reader_depth[key][:]
        position = self._position_from_depth(depth)

        basecolor_frame = self.reader_basecolor[key][:].astype(np.float32) / 255.0
        normal_frame = self.reader_normal[key][:]
        normal_frame[..., 1:] *= -1

        metallic_frame = self.reader_metallic[key][:].astype(np.float32) / 255.0
        roughness_frame = self.reader_roughness[key][:].astype(np.float32) / 255.0
        pose_frame = self.pose_reader[key][:].astype(np.float32)
        sky_mask_frame = self.reader_sky_mask[key][:].astype(bool)

        no_snow_mask_frame = None
        if self.reader_tracking is not None:
            raw = self.reader_tracking[key][:]
            no_snow_mask_frame = _tracking_parquet_to_no_snow_mask(
                raw, self.K, self.height, self.width
            )

        return {
            "position": self._to_device_tensor(position, channels_first=True),
            "basecolor": self._to_device_tensor(basecolor_frame, channels_first=True),
            "normal": self._to_device_tensor(normal_frame, channels_first=True),
            "metallic": self._to_device_tensor(metallic_frame),
            "roughness": self._to_device_tensor(roughness_frame),
            "depth": self._to_device_tensor(depth),
            "pose": torch.from_numpy(pose_frame).to(self.device),
            "sky_mask": sky_mask_frame,
            "no_snow_mask": no_snow_mask_frame,
        }

    def _decode_all_frames(self) -> list[dict[str, Any]]:
        """Decode every frame while the HDF5 file is open."""
        frames: list[dict[str, Any]] = []
        for frame_idx in range(self.frame_count):
            frames.append(self._decode_frame(frame_idx))
            if self.verbose:
                print(f"Decoded frame {frame_idx + 1}/{self.frame_count}")
        return frames

    def _frame_data(self, idx: int) -> dict[str, Any]:
        """Return decoded buffers for one frame, bounds-checked."""
        if idx < 0 or idx >= self.frame_count:
            raise IndexError("Index out of range")
        return self.material_kwargs[idx]

    def _buffer(self, idx: int, field: str) -> Any:
        """Return one named entry from a decoded frame dict."""
        return self._frame_data(idx)[field]

    def __getitem__(self, idx: int) -> Material:
        """Build a :class:`Material` view for the given frame index.

        Args:
            idx: Zero-based frame index.

        Returns:
            Per-frame material with optional ``pose``, ``sky_mask``,
            ``no_snow_mask``, and ``intrinsics`` attached.
        """
        kwargs = self._frame_data(idx)
        material = Material(
            position=kwargs["position"],
            albedo=kwargs["basecolor"],
            normal=kwargs["normal"],
            metallic=kwargs["metallic"],
            roughness=kwargs["roughness"],
            device=self.device,
            albedo_is_srgb=self.albedo_is_srgb,
        )
        material.pose = kwargs["pose"]
        material.sky_mask = kwargs.get("sky_mask")
        material.no_snow_mask = kwargs.get("no_snow_mask")
        material.intrinsics = self.intrinsics_tensor
        return material

    def get_pose(self, idx: int) -> torch.Tensor:
        """Return the ``4x4`` camera pose for a frame.

        Args:
            idx: Zero-based frame index.
        """
        return self._buffer(idx, "pose")

    def get_depth(self, idx: int) -> torch.Tensor:
        """Return the depth map for a frame, shape ``(1, H, W)``.

        Args:
            idx: Zero-based frame index.
        """
        return self._buffer(idx, "depth")

    def get_position(self, idx: int) -> torch.Tensor:
        """Return the camera-space position map for a frame.

        Args:
            idx: Zero-based frame index.
        """
        return self._buffer(idx, "position")

    def get_normal(self, idx: int) -> torch.Tensor:
        """Return the normal map for a frame.

        Args:
            idx: Zero-based frame index.
        """
        return self._buffer(idx, "normal")

    def get_basecolor(self, idx: int) -> torch.Tensor:
        """Return the basecolor map for a frame.

        Args:
            idx: Zero-based frame index.
        """
        return self._buffer(idx, "basecolor")

    def get_metallic(self, idx: int) -> torch.Tensor:
        """Return the metallic map for a frame.

        Args:
            idx: Zero-based frame index.
        """
        return self._buffer(idx, "metallic")

    def get_roughness(self, idx: int) -> torch.Tensor:
        """Return the roughness map for a frame.

        Args:
            idx: Zero-based frame index.
        """
        return self._buffer(idx, "roughness")

    def get_sky_mask(self, idx: int) -> np.ndarray:
        """Return the sky mask for a frame.

        Args:
            idx: Zero-based frame index.
        """
        return self._buffer(idx, "sky_mask")

    def get_intrinsics(self) -> np.ndarray:
        """Return the ``3x3`` camera intrinsics matrix."""
        return self.K

    def __len__(self) -> int:
        """Return the number of decoded frames."""
        return self.frame_count
