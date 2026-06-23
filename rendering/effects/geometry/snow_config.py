"""Snow modifier configuration dataclasses and flat-dict mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rendering.gbuffer.material import MaterialVideo


def create_snow_presets() -> dict[str, dict[str, Any]]:
    """Return named snow preset parameter overrides."""
    return {
        "light_snow": {
            "wet_ground_enabled": False,
            "wet_ground_intensity": 0.3,
            "wet_ground_porosity": 0.6,
            "wet_ground_roughness_factor": 0.2,
        },
        "moderate_snow": {
            "wet_ground_enabled": True,
            "wet_ground_intensity": 0.5,
            "wet_ground_porosity": 0.5,
            "wet_ground_roughness_factor": 0.03,
        },
        "heavy_snow": {
            "wet_ground_enabled": True,
            "wet_ground_intensity": 0.7,
            "wet_ground_porosity": 0.6,
            "wet_ground_roughness_factor": 0.02,
        },
    }

_FLAT_KEY_MAP: dict[str, tuple[str, str]] = {
    "mb_cascade": ("metaball", "mb_cascade"),
    "b": ("metaball", "b"),
    "interval": ("metaball", "interval"),
    "k_neighbors": ("metaball", "k_neighbors"),
    "height_scale": ("metaball", "height_scale"),
    "normal_slope_scale": ("metaball", "normal_slope_scale"),
    "normal_slope_max_deg": ("metaball", "normal_slope_max_deg"),
    "amp_decay": ("metaball", "amp_decay"),
    "blend_weight": ("metaball", "blend_weight"),
    "blend_bias": ("metaball", "blend_bias"),
    "cover_hard": ("metaball", "cover_hard"),
    "cover_threshold": ("metaball", "cover_threshold"),
    "cover_gamma": ("metaball", "cover_gamma"),
    "normal_min_slope": ("metaball", "normal_min_slope"),
    "normal_min_cover": ("metaball", "normal_min_cover"),
    "displace": ("metaball", "displace"),
    "displacement_scale": ("metaball", "displacement_scale"),
    "snow_albedo_value": ("metaball", "snow_albedo_value"),
    "snow_roughness_value": ("metaball", "snow_roughness_value"),
    "dynamic_radius_scale": ("metaball", "dynamic_radius_scale"),
    "grid_snow_enabled": ("grid", "enabled"),
    "grid_snow_resolution": ("grid", "resolution"),
    "grid_snow_density": ("grid", "density"),
    "grid_snow_height": ("grid", "height"),
    "grid_snow_albedo": ("grid", "albedo"),
    "grid_snow_roughness": ("grid", "roughness"),
    "grid_snow_metallic": ("grid", "metallic"),
    "grid_snow_eps": ("grid", "eps"),
    "grid_snow_seed": ("grid", "seed"),
    "wet_ground_enabled": ("wet_ground", "enabled"),
    "wet_ground_intensity": ("wet_ground", "intensity"),
    "wet_ground_porosity": ("wet_ground", "porosity"),
    "wet_ground_roughness_factor": ("wet_ground", "roughness_factor"),
    "snowfall_enabled": ("snowfall", "enabled"),
    "snowfall_use_world_particles": ("snowfall", "use_world_particles"),
    "snowfall_affect_normal": ("snowfall", "affect_normal"),
    "snowfall_affect_depth": ("snowfall", "affect_depth"),
    "snowfall_depth_test": ("snowfall", "depth_test"),
    "snowfall_opacity": ("snowfall", "opacity"),
    "snowfall_roughness": ("snowfall", "roughness"),
    "snowfall_num_particles": ("snowfall", "num_particles"),
    "snowfall_box_size": ("snowfall", "box_size"),
    "snowfall_gravity": ("snowfall", "gravity"),
    "snowfall_wind_world_x": ("snowfall", "wind_world_x"),
    "snowfall_wind_world_z": ("snowfall", "wind_world_z"),
    "snowfall_radius_world": ("snowfall", "radius_world"),
    "snowfall_seed": ("snowfall", "seed"),
    "car_mask_max_snow_amount": ("", "car_mask_max_snow_amount"),
    "non_grid_force_ground_snow": ("non_grid", "force_ground_snow"),
    "non_grid_disable_normal_filter": ("non_grid", "disable_normal_filter"),
    "non_grid_ground_quantile": ("non_grid", "ground_quantile"),
}


@dataclass
class MetaballSnowParams:
    """Metaball snow accumulation on non-ground surfaces."""

    mb_cascade: int = 3
    b: float = 1.5
    interval: float = 0.5
    k_neighbors: int = 16
    height_scale: float = 0.8
    normal_slope_scale: float = 4.0
    normal_slope_max_deg: float = 65
    amp_decay: float = 0.7
    blend_weight: float = 8.0
    blend_bias: float = 0.03
    cover_hard: bool = True
    cover_threshold: float = 0.45
    cover_gamma: float = 0.9
    normal_min_slope: float = 0.1
    normal_min_cover: float = 0.25
    displace: bool = False
    displacement_scale: float = 0.3
    snow_albedo_value: float = 1.0
    snow_roughness_value: float = 0.6
    dynamic_radius_scale: float = 4.0


@dataclass
class GridSnowParams:
    """Procedural grid-based ground snow map."""

    enabled: bool = True
    resolution: int = 8192
    density: float = 1.0
    height: float = 0.08
    albedo: float = 1.5
    roughness: float = 0.6
    metallic: float = 0.0
    eps: float = 1e-3
    seed: int = 42


@dataclass
class WetGroundParams:
    """Wet-road darkening on ground pixels without snow cover."""

    enabled: bool = False
    intensity: float = 0.5
    porosity: float = 0.8
    roughness_factor: float = 0.1


@dataclass
class SnowfallParams:
    """World-space falling snow particles."""

    enabled: bool = True
    use_world_particles: bool = True
    affect_normal: bool = True
    affect_depth: bool = True
    depth_test: bool = True
    opacity: float = 0.85
    roughness: float = 0.7
    num_particles: int = 15000
    box_size: float = 100.0
    gravity: float = 5.0
    wind_world_x: float = 0.6
    wind_world_z: float = 0.2
    radius_world: float = 0.1
    seed: int = 12345


@dataclass
class NonGridSnowParams:
    """Fallback ground-snow heuristics when grid snow is disabled."""

    force_ground_snow: bool = False
    disable_normal_filter: bool = False
    ground_quantile: float = 0.45


@dataclass
class SnowModifierConfig:
    """Full configuration for :class:`SnowGBufferModifierSurfaceBRDF`."""

    device: str = "cuda"
    dataset_id: int | None = None
    car_mask_max_snow_amount: float = 0.35
    metaball: MetaballSnowParams = field(default_factory=MetaballSnowParams)
    grid: GridSnowParams = field(default_factory=GridSnowParams)
    wet_ground: WetGroundParams = field(default_factory=WetGroundParams)
    snowfall: SnowfallParams = field(default_factory=SnowfallParams)
    non_grid: NonGridSnowParams = field(default_factory=NonGridSnowParams)

    @classmethod
    def from_preset(
        cls,
        preset: str,
        material_video: MaterialVideo,
        device: str = "cuda",
        dataset_id: int | None = None,
        **overrides: Any,
    ) -> SnowModifierConfig:
        """Build config from a named preset plus optional flat overrides."""
        presets = create_snow_presets()
        merged = {**presets.get(preset, presets["moderate_snow"]), **overrides}
        config = cls(
            device=device,
            dataset_id=dataset_id if dataset_id is not None else material_video.dataset_id,
        )
        _apply_flat_overrides(config, merged)
        return config


def _apply_flat_overrides(config: SnowModifierConfig, flat: dict[str, Any]) -> None:
    """Map legacy flat preset/render keys onto nested dataclass fields."""
    for key, value in flat.items():
        mapping = _FLAT_KEY_MAP.get(key)
        if mapping is None:
            continue
        section, attr = mapping
        if section:
            setattr(getattr(config, section), attr, value)
        else:
            setattr(config, attr, value)


def bind_snow_config(modifier: Any, config: SnowModifierConfig) -> None:
    """Copy nested config fields onto a modifier instance for mixin access."""
    modifier.device = config.device
    modifier.dataset_id = config.dataset_id
    modifier.car_mask_max_snow_amount = config.car_mask_max_snow_amount

    mb = config.metaball
    modifier.mb_cascade = mb.mb_cascade
    modifier.b = mb.b
    modifier.interval = mb.interval
    modifier.k_neighbors = mb.k_neighbors
    modifier.height_scale = mb.height_scale
    modifier.normal_slope_scale = mb.normal_slope_scale
    modifier.normal_slope_max_deg = mb.normal_slope_max_deg
    modifier.amp_decay = mb.amp_decay
    modifier.blend_weight = mb.blend_weight
    modifier.blend_bias = mb.blend_bias
    modifier.cover_hard = mb.cover_hard
    modifier.cover_threshold = mb.cover_threshold
    modifier.cover_gamma = mb.cover_gamma
    modifier.normal_min_slope = mb.normal_min_slope
    modifier.normal_min_cover = mb.normal_min_cover
    modifier.displace = mb.displace
    modifier.displacement_scale = mb.displacement_scale
    modifier.snow_albedo_value = mb.snow_albedo_value
    modifier.snow_roughness_value = mb.snow_roughness_value
    modifier.dynamic_radius_scale = mb.dynamic_radius_scale

    grid = config.grid
    modifier.grid_snow_enabled = grid.enabled
    modifier.grid_snow_resolution = grid.resolution
    modifier.grid_snow_density = grid.density
    modifier.grid_snow_height = grid.height
    modifier.grid_snow_albedo = grid.albedo
    modifier.grid_snow_roughness = grid.roughness
    modifier.grid_snow_metallic = grid.metallic
    modifier.grid_snow_eps = grid.eps
    modifier.grid_snow_seed = grid.seed

    wet = config.wet_ground
    modifier.wet_ground_enabled = wet.enabled
    modifier.wet_ground_intensity = wet.intensity
    modifier.wet_ground_porosity = wet.porosity
    modifier.wet_ground_roughness_factor = wet.roughness_factor

    snow = config.snowfall
    modifier.snowfall_enabled = snow.enabled
    modifier.snowfall_use_world_particles = snow.use_world_particles
    modifier.snowfall_affect_normal = snow.affect_normal
    modifier.snowfall_affect_depth = snow.affect_depth
    modifier.snowfall_depth_test = snow.depth_test
    modifier.snowfall_opacity = snow.opacity
    modifier.snowfall_roughness = snow.roughness
    modifier.falling_snow_num = snow.num_particles
    modifier.falling_snow_box_size = snow.box_size
    modifier.falling_snow_gravity = snow.gravity
    modifier.falling_snow_wind_x = snow.wind_world_x
    modifier.falling_snow_wind_z = snow.wind_world_z
    modifier.falling_snow_radius = snow.radius_world
    modifier.falling_snow_seed = snow.seed

    ng = config.non_grid
    modifier.non_grid_force_ground_snow = ng.force_ground_snow
    modifier.non_grid_disable_normal_filter = ng.disable_normal_filter
    modifier.non_grid_ground_quantile = ng.ground_quantile


__all__ = [
    "GridSnowParams",
    "MetaballSnowParams",
    "NonGridSnowParams",
    "SnowModifierConfig",
    "SnowfallParams",
    "WetGroundParams",
    "bind_snow_config",
    "create_snow_presets",
]
