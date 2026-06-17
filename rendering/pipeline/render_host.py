"""Protocol types that decouple pipeline modules from ``WeatherRenderer``."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from rendering.gbuffer.material import MaterialVideo
from rendering.lighting.brdf import CookTorranceBRDF
from rendering.pipeline.geometry_effects import GeometryEffectsManager
from rendering.pipeline.light_effects import LightEffectsManager


class BrdfPipelineHost(Protocol):
    """Host dependencies for the fog/night BRDF video path.

    ``WeatherRenderer`` provides these attributes after fog/night effects are
    initialized. Pipeline modules depend on this protocol instead of importing
    the orchestrator class.
    """

    h5_file: Path
    device: torch.device
    frame_count: int
    material_video: MaterialVideo
    brdf: CookTorranceBRDF
    light_manager: LightEffectsManager


class GeometryPipelineHost(Protocol):
    """Host dependencies for the rain/snow G-buffer export path.

    ``WeatherRenderer`` provides these attributes after rain/snow geometry
    effects are initialized. ``geometry_manager`` may be ``None`` before setup.
    """

    seq_id: str
    frame_count: int
    material_video: MaterialVideo
    geometry_manager: GeometryEffectsManager | None
