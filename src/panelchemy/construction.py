"""Configurable construction and nozzle-derived tracing guidance."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from fractions import Fraction
from math import ceil, isclose, isfinite, radians, tan
from pathlib import Path
from typing import Literal, Self, cast

type ContourMode = Literal["curves", "lines"]


def _contour_mode(value: object) -> ContourMode:
    if value == "curves":
        return "curves"
    if value == "lines":
        return "lines"
    raise ValueError("contour_mode must be curves or lines")


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
    contour_mode: ContourMode = "curves"
    corner_softening_mm: float = 0.0
    span_simplification_mm: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name == "contour_mode":
                _ = _contour_mode(self.contour_mode)
                continue
            value = _number(cast("object", getattr(self, field.name)), field.name)
            if field.name == "taper_angle_deg":
                if not 0 <= value < 90:
                    raise ValueError(
                        "taper_angle_deg must be at least 0 and less than 90"
                    )
            elif field.name in {
                "contour_tolerance_mm",
                "corner_softening_mm",
                "span_simplification_mm",
            }:
                if value < 0:
                    raise ValueError(f"{field.name} must be non-negative")
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
        if not isfinite(2 * self.assumed_line_width_mm) or self.tracing_radius_mm <= 0:
            raise ValueError(
                "nozzle diameter must produce finite positive tracing scales"
            )
        if self.corner_softening_mm > self.tracing_radius_mm:
            raise ValueError("corner_softening_mm must not exceed the tracing radius")
        if self.span_simplification_mm > self.tracing_radius_mm:
            raise ValueError(
                "span_simplification_mm must not exceed the tracing radius"
            )

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

    @property
    def assumed_line_width_mm(self) -> float:
        """Planning assumption, rather than an encoded slicer setting."""
        return 1.25 * self.nozzle_diameter_mm

    @property
    def tracing_radius_mm(self) -> float:
        """Detail-significance scale; does not prescribe circular fillets."""
        return self.assumed_line_width_mm / 2

    @property
    def minimum_source_long_edge_pixels(self) -> int:
        """Four independent source samples per assumed printed line width."""
        return ceil(
            4 * Fraction(self.long_edge_mm) / Fraction(self.assumed_line_width_mm)
        )

    @property
    def preferred_source_long_edge_pixels(self) -> int:
        return ceil(
            8 * Fraction(self.long_edge_mm) / Fraction(self.assumed_line_width_mm)
        )

    def as_dict(self) -> dict[str, float | str]:
        return {
            field.name: cast("float | str", getattr(self, field.name))
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
        return cls(
            **{
                name: _number(value, name)
                for name, value in data.items()
                if name != "contour_mode"
            },
            contour_mode=_contour_mode(data.get("contour_mode", "curves")),
        )


def load_profile(path: Path | None) -> ConstructionProfile:
    if path is None:
        return ConstructionProfile()
    value = cast("object", json.loads(path.read_text()))
    if not isinstance(value, dict):
        raise ValueError("construction profile must be a JSON object")
    return ConstructionProfile.from_mapping(cast("dict[str, object]", value))
