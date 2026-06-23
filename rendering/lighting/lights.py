"""HDF5 vehicle light and emissive map loading for BRDF shading."""

from __future__ import annotations

from typing import Any, Literal

import h5py
import hdf5plugin  # noqa: F401  # registers HDF5 compression filters
import torch

SceneType = Literal["fog", "night"]

_DEFAULT_LIGHT_CFG: dict[str, dict[str, Any]] = {
    "light_heads": {
        "radius": 30.0,
        "inner_deg": 20.0,
        "outer_deg": 40.0,
        "dir": [0.0, 1.0, 0.0],
        "color": [1.0, 0.92, 0.78],
        "mult": 2.0,
    },
    "rear_car_heads": {
        "radius": 30.0,
        "inner_deg": 10.0,
        "outer_deg": 90.0,
        "color": [1.0, 0.82, 0.62],
        "mult": 5.0,
    },
    "clearance_lamp": {
        "radius": 1.0,
        "color": [1.0, 0.15, 0.15],
        "mult": 0.0,
    },
    "fog_lights": {
        "radius": 50.0,
        "inner_deg": 15.0,
        "outer_deg": 60.0,
        "color": [1.0, 0.95, 0.85],
        "mult": 3.0,
    },
}

_EMISSIVE_CONFIG: dict[str, dict[str, Any]] = {
    "tail_lights": {
        "color": [1.0, 0.2, 0.2],
        "intensity": 2.0,
    },
}

_LIGHT_INCLUDE_DEFAULT = ("light_heads", "rear_car_heads", "clearance_lamp")


def _build_point_light(
    pos_xyz: list[float],
    radius: float,
    intensity_rgb: list[float],
    device: torch.device | str,
) -> dict[str, Any]:
    """Build one point-light dict for :func:`CookTorranceBRDF.render_many`."""
    return {
        "type": "point",
        "desc": {
            "pos": torch.tensor(pos_xyz, device=device, dtype=torch.float32).view(3, 1, 1),
            "light_radius_of_influence": float(radius),
        },
        "intensity": torch.tensor(intensity_rgb, device=device, dtype=torch.float32).view(3),
    }


def _build_spot_light(
    pos_xyz: list[float],
    dir_xyz: list[float],
    inner_deg: float,
    outer_deg: float,
    radius: float,
    intensity_rgb: list[float],
    device: torch.device | str,
) -> dict[str, Any]:
    """Build one spot-light dict for :func:`CookTorranceBRDF.render_many`."""
    return {
        "type": "spot",
        "desc": {
            "pos": torch.tensor(pos_xyz, device=device, dtype=torch.float32).view(3, 1, 1),
            "dir": torch.tensor(dir_xyz, device=device, dtype=torch.float32).view(3),
            "inner_deg": float(inner_deg),
            "outer_deg": float(outer_deg),
            "light_radius_of_influence": float(radius),
        },
        "intensity": torch.tensor(intensity_rgb, device=device, dtype=torch.float32).view(3),
    }


def _read_direction(rec: h5py.Dataset | h5py.Group) -> list[float]:
    direction = rec["direction"][()]
    return [float(direction[0]), float(direction[1]), float(direction[2])]


def _intensity_from_cfg(light_cfg: dict[str, Any]) -> list[float]:
    base_color = torch.tensor(light_cfg["color"], dtype=torch.float32)
    return (base_color * float(light_cfg["mult"])).tolist()


def _append_lights_for_key(
    lights: list[dict[str, Any]],
    key: str,
    frame_group: h5py.Group,
    light_cfg: dict[str, Any],
    device: torch.device | str,
) -> None:
    """Append all lights of one HDF5 group key to ``lights``."""
    if key not in frame_group:
        return

    radius = float(light_cfg["radius"])
    intensity = _intensity_from_cfg(light_cfg)

    for _, rec in frame_group[key].items():
        pos = [float(rec["x"][()]), float(rec["y"][()]), float(rec["z"][()])]

        if key == "rear_car_heads":
            lights.append(
                _build_spot_light(
                    pos,
                    _read_direction(rec),
                    float(light_cfg["inner_deg"]),
                    float(light_cfg["outer_deg"]),
                    radius,
                    intensity,
                    device,
                )
            )
        elif key == "light_heads":
            lights.append(
                _build_spot_light(
                    pos,
                    light_cfg["dir"],
                    float(light_cfg["inner_deg"]),
                    float(light_cfg["outer_deg"]),
                    radius,
                    intensity,
                    device,
                )
            )
        elif key == "clearance_lamp":
            lights.append(_build_point_light(pos, radius, intensity, device))
        else:
            raise NotImplementedError(f"Unknown light key: {key}")


def _ego_headlight(
    scene_type: SceneType,
    light_cfg: dict[str, dict[str, Any]],
    device: torch.device | str,
) -> dict[str, Any]:
    """Return the synthetic ego-vehicle headlight for fog or night scenes."""
    ego_cfg = light_cfg["fog_lights"] if scene_type == "fog" else light_cfg["rear_car_heads"]
    return _build_spot_light(
        pos_xyz=[0.0, 0.0, 0.0],
        dir_xyz=[0.0, 0.3, 1.0],
        inner_deg=float(ego_cfg["inner_deg"]),
        outer_deg=float(ego_cfg["outer_deg"]),
        radius=float(ego_cfg["radius"]),
        intensity_rgb=[channel * float(ego_cfg["mult"]) for channel in ego_cfg["color"]],
        device=device,
    )


def load_lights_from_h5(
    h5_path: str,
    device: torch.device | str,
    include: tuple[str, ...] = _LIGHT_INCLUDE_DEFAULT,
    cfg: dict[str, dict[str, Any]] | None = None,
    scene_type: SceneType = "night",
) -> list[list[dict[str, Any]]]:
    """Load per-frame vehicle lights from an HDF5 scene file.

    Args:
        h5_path: Path to the scene ``.h5`` file.
        device: Torch device for light tensors.
        include: HDF5 light group keys to load from each frame.
        cfg: Optional override for default light color, cone, and radius settings.
        scene_type: ``"fog"`` selects high-beam ego lights; ``"night"`` uses low beam.

    Returns:
        One light list per frame, suitable for :func:`CookTorranceBRDF.render_many`.

    Raises:
        KeyError: When the HDF5 file has no ``lights`` group.
    """
    light_cfg = _DEFAULT_LIGHT_CFG if cfg is None else cfg
    all_lights: list[list[dict[str, Any]]] = []

    with h5py.File(h5_path, "r") as h5f:
        h5f_light = h5f["lights"]
        for frame_idx in range(len(h5f_light)):
            frame_group = h5f_light[f"{frame_idx:02}"]
            lights: list[dict[str, Any]] = []
            for key in include:
                _append_lights_for_key(lights, key, frame_group, light_cfg[key], device)
            lights.append(_ego_headlight(scene_type, light_cfg, device))
            all_lights.append(lights)

    return all_lights


def _decode_emissive_texture(
    emissive_texture: Any,
    device: torch.device | str,
) -> torch.Tensor | None:
    """Convert one HDF5 emissive array to ``(3, H, W)`` float tensors."""
    if emissive_texture.ndim == 3:
        emissive_tensor = torch.tensor(emissive_texture, device=device, dtype=torch.float32)
        return emissive_tensor.permute(2, 0, 1)
    if emissive_texture.ndim == 2:
        emissive_tensor = torch.tensor(emissive_texture, device=device, dtype=torch.float32)
        return emissive_tensor.unsqueeze(0).repeat(3, 1, 1)
    print(f"Warning: unknown emissive texture format {emissive_texture.shape}")
    return None


def load_emissive_from_h5(
    h5_path: str,
    device: torch.device | str,
    emissive_type: str = "tail_lights",
) -> list[dict[str, Any]]:
    """Load per-frame emissive maps from an HDF5 scene file.

    Args:
        h5_path: Path to the scene ``.h5`` file.
        device: Torch device for emissive tensors.
        emissive_type: Emissive preset name; currently only ``"tail_lights"`` is supported.

    Returns:
        One dict per frame with an ``"emission"`` tensor or ``None`` when absent.
        Returns an empty list when the emissive group or type is unavailable.
    """
    if emissive_type not in _EMISSIVE_CONFIG:
        print(f"Warning: unknown emissive type {emissive_type}")
        return []

    config = _EMISSIVE_CONFIG[emissive_type]
    base_color = torch.tensor(config["color"], device=device, dtype=torch.float32).view(3, 1, 1)
    intensity = float(config["intensity"])
    all_emissive: list[dict[str, Any]] = []

    with h5py.File(h5_path, "r") as h5f:
        if "emissive" not in h5f:
            print(f"Warning: no emissive group in {h5_path}")
            return []

        emissive_data = h5f["emissive"]
        for frame_idx in range(len(emissive_data)):
            frame_key = f"{frame_idx:02}"
            if frame_key not in emissive_data:
                all_emissive.append({"emission": None})
                continue

            emissive_tensor = _decode_emissive_texture(emissive_data[frame_key][:], device)
            if emissive_tensor is None:
                all_emissive.append({"emission": None})
                continue

            all_emissive.append({"emission": emissive_tensor * base_color * intensity})

    return all_emissive


__all__ = [
    "load_emissive_from_h5",
    "load_lights_from_h5",
]
