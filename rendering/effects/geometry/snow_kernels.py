"""SPH kernels and ground-method config helpers for snow geometry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

_GROUND_METHOD_CONFIG_PATH = Path(__file__).resolve().parent.parent / "ground_method_config.json"


def load_ground_method_config() -> dict[str, Any]:
    """Load optional per-dataset ground-detection method overrides."""
    if not _GROUND_METHOD_CONFIG_PATH.is_file():
        return {"default_method": 1, "special_datasets": {}}
    with _GROUND_METHOD_CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def ground_method_for_dataset(dataset_id: int | None, config: dict[str, Any]) -> int:
    """Resolve which ground-label variant to use for a dataset id."""
    if dataset_id is None:
        return 1
    special = config.get("special_datasets", {})
    entry = special.get(str(dataset_id))
    if entry is None:
        return int(config.get("default_method", 1))
    if isinstance(entry, dict):
        return int(entry.get("method", 1))
    return int(entry)


def weighted_sigmoid(x: torch.Tensor, weight: float, bias: float) -> torch.Tensor:
    """Sigmoid coverage blend: ``1 / (1 + exp(-weight * (x - bias)))``."""
    return 1.0 / (1.0 + torch.exp(-weight * (x - bias)))


def w_poly6(a: torch.Tensor, radius: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """SPH Poly6 kernel for metaball height fields."""
    radius_safe = torch.clamp(radius, min=1e-6)
    valid = (r >= 0) & (r < radius_safe)
    coef = 315.0 / (64.0 * torch.pi * (radius_safe**9))
    val = coef * (radius_safe**2 - r**2) ** 3
    val = torch.where(valid, val, torch.zeros_like(val))
    return a * val


def dw_poly6_dr(a: torch.Tensor, radius: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Radial derivative of the Poly6 kernel."""
    radius_safe = torch.clamp(radius, min=1e-6)
    valid = (r > 0) & (r < radius_safe)
    coef = -945.0 / (32.0 * torch.pi * (radius_safe**9))
    val = coef * r * (radius_safe**2 - r**2) ** 2
    val = torch.where(valid, val, torch.zeros_like(val))
    return a * val
