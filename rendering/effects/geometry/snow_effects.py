"""Snow geometry effects: metaball accumulation, grid ground snow, and snowfall.

Public symbols:

- :class:`SnowGBufferModifierSurfaceBRDF` — snow G-buffer modifier.
- :func:`create_snow_presets` — named preset parameter overrides.
"""

from __future__ import annotations

from rendering.effects.geometry.snow_config import create_snow_presets
from rendering.effects.geometry.snow_modifier import SnowGBufferModifierSurfaceBRDF


__all__ = [
    "SnowGBufferModifierSurfaceBRDF",
    "create_snow_presets",
]
