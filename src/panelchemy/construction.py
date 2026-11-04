"""Configurable physical construction; nozzle metadata never removes features."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isclose, isfinite, radians, tan
from pathlib import Path
from typing import Self, cast


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True)
class ConstructionProfile:
    """Heights are band thicknesses; taper angle is measured from vertical."""

    long_edge_mm: float = 250.0
    nozzle_diameter_mm: float = 0.4
    layer_height_mm: float = 0.2
    black_thickness_mm: float = 2.0
    orange_thickness_mm: float = 1.0
    white_thickness_mm: float = 1.0
    taper_angle_deg: float = 10.0
    rim_width_mm: float = 0.2
    contour_tolerance_mm: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = _number(cast("object", getattr(self, field.name)), field.name)
            if field.name == "taper_angle_deg":
                if not 0 <= value < 90:
                    raise ValueError(
                        "taper_angle_deg must be at least 0 and less than 90"
                    )
            elif field.name == "contour_tolerance_mm":
                if value < 0:
                    raise ValueError("contour_tolerance_mm must be non-negative")
            elif value <= 0:
                raise ValueError(f"{field.name} must be positive")
        for thickness in (
            self.black_thickness_mm,
            self.orange_thickness_mm,
            self.white_thickness_mm,
        ):
            layers = thickness / self.layer_height_mm
            if (
                not isfinite(layers)
                or round(layers) < 1
                or not isclose(layers, round(layers), rel_tol=0, abs_tol=1e-8)
            ):
                raise ValueError(
                    "each material band must contain a whole number of layers"
                )
        if not isfinite(self.orange_top) or not isfinite(self.white_top):
            raise ValueError("combined material heights must remain finite")

    @property
    def black_top(self) -> float:
        return self.black_thickness_mm

    @property
    def orange_top(self) -> float:
        return self.black_top + self.orange_thickness_mm

    @property
    def white_top(self) -> float:
        return self.orange_top + self.white_thickness_mm

    @property
    def taper_run_per_height(self) -> float:
        return tan(radians(self.taper_angle_deg))

    def as_dict(self) -> dict[str, float]:
        return {
            field.name: cast("float", getattr(self, field.name))
            for field in fields(self)
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Self:
        allowed = {field.name for field in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(
                f"unknown construction settings: {', '.join(sorted(unknown))}"
            )
        return cls(**{name: _number(value, name) for name, value in data.items()})


def load_profile(path: Path | None) -> ConstructionProfile:
    if path is None:
        return ConstructionProfile()
    value = cast("object", json.loads(path.read_text()))
    if not isinstance(value, dict):
        raise ValueError("construction profile must be a JSON object")
    return ConstructionProfile.from_mapping(cast("dict[str, object]", value))
