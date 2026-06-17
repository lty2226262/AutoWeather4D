"""G-buffer tensor conversion and per-frame JPG export."""

from __future__ import annotations

import os

import imageio.v2 as imageio
import numpy as np
import torch

from rendering.gbuffer.grade import grade_basecolor as apply_basecolor_grade


def _gbuffer_jpg_path(save_dir: str, frame_idx: int, channel: str) -> str:
    """Return the JPG path for one G-buffer channel."""
    return os.path.join(save_dir, f"0000.{frame_idx:04d}.{channel}.jpg")


def _tensor_to_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW float tensor in ``[0, 1]`` to an HWC uint8 image."""
    return (tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)


def _gbuffer_maps_to_images(
    basecolor_map: torch.Tensor,
    depth_map: torch.Tensor,
    normal_map: torch.Tensor,
    metallic_map: torch.Tensor,
    roughness_map: torch.Tensor,
    *,
    grade_basecolor: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert G-buffer tensors to uint8 images for JPG export."""
    if grade_basecolor:
        basecolor_map = apply_basecolor_grade(basecolor_map)
    basecolor_img = _tensor_to_uint8_hwc(basecolor_map)

    depth_values = np.nan_to_num(depth_map[0].detach().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    depth_max = float(depth_values.max())
    depth_gray = (
        (depth_values / depth_max * 255).astype(np.uint8)
        if depth_max > 1e-6
        else np.zeros_like(depth_values, dtype=np.uint8)
    )
    depth_img = np.repeat(depth_gray[..., None], 3, axis=-1)

    normal_img = _tensor_to_uint8_hwc((normal_map + 1.0) * 0.5)
    metallic_img = np.repeat(
        (np.clip(metallic_map[0].detach().cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)[..., None],
        3,
        axis=-1,
    )
    roughness_img = np.repeat(
        (np.clip(roughness_map[0].detach().cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)[..., None],
        3,
        axis=-1,
    )
    return basecolor_img, depth_img, normal_img, metallic_img, roughness_img


def _write_gbuffer_frame(
    save_dir: str,
    frame_idx: int,
    images: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Write one frame of G-buffer JPGs to ``save_dir``."""
    channels = ("basecolor", "depth", "normal", "metallic", "roughness")
    for channel, image in zip(channels, images, strict=True):
        imageio.imwrite(_gbuffer_jpg_path(save_dir, frame_idx, channel), image)


def save_gbuffer_frame(
    save_dir: str,
    frame_idx: int,
    basecolor_map: torch.Tensor,
    depth_map: torch.Tensor,
    normal_map: torch.Tensor,
    metallic_map: torch.Tensor,
    roughness_map: torch.Tensor,
    *,
    grade_basecolor: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert one G-buffer frame to images and write JPG files.

    Args:
        save_dir: Output directory for ``0000.{frame:04d}.*.jpg`` files.
        frame_idx: Zero-based frame index.
        basecolor_map: Base color map, shape ``(3, H, W)``.
        depth_map: Depth map, shape ``(1, H, W)``.
        normal_map: Normal map, shape ``(3, H, W)``.
        metallic_map: Metallic map, shape ``(1, H, W)``.
        roughness_map: Roughness map, shape ``(1, H, W)``.
        grade_basecolor: Whether to apply rain/snow basecolor grading before export.

    Returns:
        Tuple of exported uint8 images:
        ``(basecolor, depth, normal, metallic, roughness)``.
    """
    images = _gbuffer_maps_to_images(
        basecolor_map,
        depth_map,
        normal_map,
        metallic_map,
        roughness_map,
        grade_basecolor=grade_basecolor,
    )
    _write_gbuffer_frame(save_dir, frame_idx, images)
    return images
