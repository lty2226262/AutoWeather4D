"""Relit LUT preprocessing and linear-space image blending for fog/night output."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from rendering.gbuffer.material import Material


def _create_moonlight_lut(strength: float = 1.0) -> np.ndarray:
    """Build a BGR moonlight color lookup table."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        brightness = 0.7 + 0.2 * (1 - strength)
        base = int(i * brightness)
        r = int(base * (0.85 - 0.15 * strength))
        g = int(base * (0.9 - 0.1 * strength))
        b = int(base * (1.05 + 0.2 * strength))
        lut[i] = [
            np.clip(b, 0, 255),
            np.clip(g, 0, 255),
            np.clip(r, 0, 255),
        ]
    return lut


def _create_fog_lut(strength: float = 1.0) -> np.ndarray:
    """Build a BGR fog color lookup table."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        brightness = 0.8 + 0.15 * (1 - strength)
        base = int(i * brightness)
        r = int(base * (0.9 - 0.1 * strength))
        g = int(base * (0.9 - 0.1 * strength))
        b = int(base * (0.95 + 0.05 * strength))
        lut[i] = [
            np.clip(b, 0, 255),
            np.clip(g, 0, 255),
            np.clip(r, 0, 255),
        ]
    return lut


def _srgb_to_linear_np(x: np.ndarray) -> np.ndarray:
    """Convert sRGB channel values to linear light."""
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def _linear_to_srgb_np(x: np.ndarray) -> np.ndarray:
    """Convert linear light channel values to sRGB."""
    a = 0.055
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1 / 2.4)) - a)


def _adjust_exposure(
    frame: np.ndarray,
    strength: float = 1.0,
    percentile: float = 70.0,
    target_luma: float = 0.22,
) -> np.ndarray:
    """Apply adaptive exposure adjustment in linear space with highlight compression."""
    img = frame.astype(np.float32) / 255.0
    rgb = img[..., ::-1]
    lin = _srgb_to_linear_np(rgb)

    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    scene_key = np.percentile(luma, percentile)
    gain_dyn = np.clip(target_luma / (scene_key + 1e-6), 0.6, 1.6)

    base_exposure = 0.98 - 0.20 * float(strength)
    gain = base_exposure * gain_dyn
    lin *= gain
    lin = lin / (1.0 + 0.25 * lin)

    srgb = _linear_to_srgb_np(lin)
    bgr = srgb[..., ::-1]

    out = (bgr * 255.0).astype(np.float32)
    contrast = 1.05 + 0.15 * float(strength)
    gamma = 1.0 - 0.03 * float(strength)
    out = (out - 127.5) * contrast + 127.5
    out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, gamma)
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_lut(
    frame: np.ndarray,
    lut: np.ndarray,
    sky_mask: np.ndarray | torch.Tensor | None = None,
    sky_darkening_factor: float = 0.6,
    use_sky_mask: bool = True,
) -> np.ndarray:
    """Apply a BGR LUT with optional soft-edge sky darkening."""
    b, g, r = cv2.split(frame)
    b = cv2.LUT(b, lut[:, 0])
    g = cv2.LUT(g, lut[:, 1])
    r = cv2.LUT(r, lut[:, 2])
    result = cv2.merge([b, g, r])

    if sky_mask is None or not use_sky_mask:
        return result

    mask = sky_mask
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = (mask.astype(np.float32) > 0.5).astype(np.uint8) * 255
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)

    dilate_px = 20
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    mask = cv2.dilate(mask, k_dilate, iterations=1)

    feather_px = 10
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), feather_px)

    scale = 1.0 - alpha[..., None] * (1.0 - float(sky_darkening_factor))
    out_scaled = (result.astype(np.float32) * scale).clip(0, 255)

    band = (alpha * (1.0 - alpha)) * 4.0
    guard = 0.5
    band3 = band[..., None] * guard
    return (out_scaled * (1.0 - band3) + frame.astype(np.float32) * band3).clip(0, 255).astype(np.uint8)


def _soften_highlights(frame_bgr: np.ndarray) -> np.ndarray:
    """Blur very bright regions to reduce harsh relit highlights."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(frame_bgr, (15, 15), 0)
    return cv2.bitwise_and(frame_bgr, frame_bgr, mask=~mask) + cv2.bitwise_and(blur, blur, mask=mask)


def preprocess_relit_frame(
    frame_rgb: np.ndarray,
    *,
    sky_mask: np.ndarray | torch.Tensor | None = None,
    strength: float = 0.8,
    sky_darkening_factor: float = 0.6,
    lut_type: str = "moonlight",
) -> np.ndarray:
    """Apply relit LUT preprocessing to one RGB uint8 frame.

    Args:
        frame_rgb: Input frame in HWC RGB uint8 layout.
        sky_mask: Optional boolean sky mask for moonlight darkening.
        strength: Effect strength in ``[0, 1]``.
        sky_darkening_factor: Sky brightness scale after LUT application.
        lut_type: ``"moonlight"`` or ``"fog"``.

    Returns:
        Preprocessed frame in HWC RGB uint8 layout.
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    lut = _create_fog_lut(strength) if lut_type == "fog" else _create_moonlight_lut(strength)
    frame_bgr = _adjust_exposure(frame_bgr, strength)
    use_sky_mask = lut_type == "moonlight"
    frame_bgr = _apply_lut(frame_bgr, lut, sky_mask, sky_darkening_factor, use_sky_mask)
    if strength > 0.5:
        frame_bgr = _soften_highlights(frame_bgr)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def linear_blend_images(
    img1: torch.Tensor,
    img2: torch.Tensor,
    img2_strength: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Blend two sRGB images in linear space.

    ``img1`` is the relit frame and ``img2`` is the BRDF base frame. Blend weight
    grows with base-frame luminance so bright regions retain more BRDF detail.

    Args:
        img1: Relit image, shape ``(3, H, W)`` or ``(H, W, 3)`` in sRGB.
        img2: Base image, shape ``(3, H, W)`` or ``(H, W, 3)`` in sRGB.
        img2_strength: Scalar multiplier on the relit-driven blend weight.
        device: Torch device for the blend. Defaults to ``img1.device``.

    Returns:
        Blended image in sRGB, shape ``(3, H, W)``.
    """
    blend_device = torch.device(device) if device is not None else img1.device
    img1 = img1.to(blend_device)
    img2 = img2.to(blend_device)

    if img1.dim() == 3 and img1.shape[0] == 3:
        img1_hwc = img1.permute(1, 2, 0)
        img2_hwc = img2.permute(1, 2, 0)
    else:
        img1_hwc = img1
        img2_hwc = img2

    img1_linear = Material.srgb_to_linear(img1_hwc.permute(2, 0, 1)).permute(1, 2, 0)
    img2_linear = Material.srgb_to_linear(img2_hwc.permute(2, 0, 1)).permute(1, 2, 0)

    light_illuminance = torch.mean(img2_linear, dim=2, keepdim=True)
    weight = torch.clamp(light_illuminance - 0.05, 0, 1) * img2_strength
    blended_linear = (1 - weight) * img1_linear + img2_linear * weight
    return Material.linear_to_srgb(blended_linear.permute(2, 0, 1))
