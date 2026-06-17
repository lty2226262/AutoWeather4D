"""G-buffer basecolor grading for rain/snow forward relighting."""

from __future__ import annotations

import torch

# Default cool/desaturated look for rain/snow forward relighting.
SATURATION = 0.50
COOL_STRENGTH = 0.20


def grade_basecolor(
    basecolor: torch.Tensor,
    *,
    saturation: float = SATURATION,
    cool_strength: float = COOL_STRENGTH,
) -> torch.Tensor:
    """Grade albedo toward a cooler, desaturated winter look.

    Args:
        basecolor: Base color map in CHW layout with values in ``[0, 1]``.
        saturation: Color saturation scale in ``[0, 1]``.
        cool_strength: Cool-temperature tint strength in ``[0, 1]``.

    Returns:
        Graded base color map with the same shape and value range as the input.
    """
    c = basecolor.clamp(0.0, 1.0)
    luma = (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]).unsqueeze(0)
    desat = torch.lerp(luma.expand_as(c), c, float(saturation))
    s = float(cool_strength)
    tint = torch.tensor(
        [1.0 - s, 1.0 - 0.35 * s, 1.0 + 0.5 * s],
        device=c.device,
        dtype=c.dtype,
    ).view(3, 1, 1)
    return (desat * tint).clamp(0.0, 1.0)
