"""Configuration and runtime context for weather rendering."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class SnowParams:
    """Snow geometry and particle parameters."""

    snow_preset: str = "moderate_snow"
    snow_amount: float = 2.0
    max_points: int = 100000
    grid_snow_enabled: bool = True
    grid_snow_density: float = 1.0
    snowfall_num_particles: int = 15000
    snowfall_radius_world: float = 0.1


@dataclass
class RainParams:
    """Rain geometry simulation parameters."""

    dt: float = 0.1


@dataclass
class FogParams:
    """Fog volumetric rendering parameters."""

    fog_density: float = 0.75


@dataclass
class NightParams:
    """Night lighting and sky treatment parameters."""

    sky_darkening_factor: float = 0.25
    enable_car_lights: bool = True
    enable_emissive: bool = False


def _params_from_kwargs(params_cls: type[Any], kwargs: dict[str, Any]) -> Any:
    """Build a nested params dataclass from flat keyword arguments."""
    valid_keys = {f.name for f in fields(params_cls)}
    return params_cls(**{k: kwargs[k] for k in valid_keys if k in kwargs})


@dataclass
class RenderConfig:
    """User-facing render settings."""

    weather: str
    output_dir: str
    snow: SnowParams = field(default_factory=SnowParams)
    rain: RainParams = field(default_factory=RainParams)
    fog: FogParams = field(default_factory=FogParams)
    night: NightParams = field(default_factory=NightParams)

    @classmethod
    def from_kwargs(cls, weather: str, output_dir: str, **kwargs: Any) -> RenderConfig:
        """Build a config object from flat keyword arguments.

        Args:
            weather: Weather preset name.
            output_dir: Root output directory.
            **kwargs: Flat snow/rain/fog/night fields routed into nested dataclasses.

        Returns:
            Populated :class:`RenderConfig` instance.
        """
        return cls(
            weather=weather,
            output_dir=output_dir,
            snow=_params_from_kwargs(SnowParams, kwargs),
            rain=_params_from_kwargs(RainParams, kwargs),
            fog=_params_from_kwargs(FogParams, kwargs),
            night=_params_from_kwargs(NightParams, kwargs),
        )


@dataclass
class WeatherFlags:
    """Boolean switches derived from the requested weather preset."""

    use_snow: bool
    use_rain: bool
    use_fog: bool
    use_night: bool

    @property
    def use_forward_relight(self) -> bool:
        """Return whether this weather uses the rain/snow G-buffer forward path.

        Returns:
            ``True`` for rain or snow; otherwise ``False``.
        """
        return self.use_rain or self.use_snow


@dataclass
class RenderContext:
    """Mutable state for one :meth:`WeatherRenderer.render` call.

    ``gbuffer_save_path`` or ``video_path`` is populated when the corresponding
    output artifacts are written.
    """

    config: RenderConfig
    flags: WeatherFlags
    geometry: list[str]
    light: list[str]
    final_output_dir: str
    gbuffer_save_path: str | None = None
    video_path: str | None = None


@dataclass
class RenderResult:
    """Structured output of a completed render call."""

    output_dir: str
    gbuffer_dir: str | None = None
    video_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Convert the result to the dict returned by :meth:`WeatherRenderer.render`.

        Returns:
            Dict with ``output_dir`` and optional ``gbuffer_dir`` / ``video_path``.
        """
        return {
            "output_dir": self.output_dir,
            "gbuffer_dir": self.gbuffer_dir,
            "video_path": self.video_path,
        }


_WEATHER_MAP: dict[str, tuple[list[str], list[str]]] = {
    "rain": (["rain"], []),
    "snow": (["snow"], []),
    "fog": ([], ["fog"]),
    "night": ([], ["night"]),
}


def parse_weather(weather: str) -> tuple[list[str], list[str]]:
    """Map a weather preset name to geometry and lighting effect lists.

    Args:
        weather: Weather preset name.

    Returns:
        Tuple of ``(geometry_effects, light_effects)``.

    Raises:
        ValueError: If ``weather`` is not a supported preset.
    """
    try:
        geometry, light = _WEATHER_MAP[weather]
    except KeyError as exc:
        valid = list(_WEATHER_MAP)
        raise ValueError(f"Unknown weather: {weather}. Valid: {valid}") from exc
    return list(geometry), list(light)
