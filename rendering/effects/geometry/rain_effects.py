"""Rain geometry effects for height maps, puddles, ripples, and raindrops.

Public symbols:

- :class:`HeightMapGenerator` — scene height maps and puddle masks (also used by snow).
- :class:`RainPuddleSimulator` — rain wetness, ripples, and streak painting.
- :class:`GLNFBMOpts` — FBM noise settings for puddle placement.
"""

from __future__ import annotations

from rendering.effects.geometry.height_map import HeightMapGenerator
from rendering.effects.geometry.rain_puddle import GLNFBMOpts, RainPuddleSimulator

__all__ = [
    "GLNFBMOpts",
    "HeightMapGenerator",
    "RainPuddleSimulator",
]
