"""Taper requirements for shared palette boundaries, in canonical direction."""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb, isfinite
from typing import cast

from panelchemy import curvature
from panelchemy.curves import BezierSpan, Controls, CurveContours, CurveRing


@dataclass(frozen=True)
class BoundaryTaper:
    left: int
    right: int
    offset_mm: float
    inward_sign: int


class CurvatureLimitError(ValueError):
    """Selected outlines cannot be certified for the requested taper.

    Keep the exact geometry and failed bounds available for 2D diagnosis.
    Neither a missing proof nor an exhausted search authorizes CAD construction.
    """

    def __init__(
        self,
        contours: CurveContours,
        failures: dict[int, tuple[BezierSpan, BoundaryTaper]],
    ) -> None:
        self.contours: CurveContours = contours
        self.failures: dict[int, tuple[BezierSpan, BoundaryTaper]] = failures
        index, (_, limit) = next(iter(failures.items()))
        super().__init__(
            f"Minimum taper radius is not certified for {len(failures)} selected "
            + f"boundary spans; first span {index} uses a {limit.offset_mm:.6g} mm "
            + "boundary run with the required 0.9 curvature margin. "
            + "Refit the unresolved outlines before CAD construction."
        )


def _positive_area(ring: CurveRing) -> bool:
    """Exact orientation of the selected cubic geometry, not a sampled proxy."""
    area = Fraction(0)
    for span in ring.spans:
        p = [tuple(Fraction(v) for v in point) for point in span.control_points]
        if span.is_line:
            area += p[0][0] * p[3][1] - p[0][1] * p[3][0]
        else:
            for i in range(4):
                for j in range(3):
                    dx, dy = (3 * (p[j + 1][a] - p[j][a]) for a in (0, 1))
                    area += (p[i][0] * dy - p[i][1] * dx) * Fraction(
                        comb(3, i) * comb(2, j), 6 * comb(5, i + j)
                    )
    if not area:
        raise ValueError("a selected curved boundary must enclose nonzero area")
    return area > 0


def require_curvature(
    contours: CurveContours, material_runs_mm: tuple[float, float, float]
) -> None:
    """Hard pre-CAD gate on actual selected curves and shared exposed heights.

    Reconstruct exact shared-span ownership instead of trusting fitting status
    metadata, which may describe an earlier or cached selection. Winding and
    canonical direction do not change the required signed bound. The outer
    silhouette is vertical; joins remain subject to native corner handling.
    """
    # Validate the runs even when every boundary is vertical.
    _ = boundaries([[]], [range(1)], 0, material_runs_mm)
    if not any(material_runs_mm):
        return
    selected: dict[tuple[bool, Controls], int] = {}
    spans: list[BezierSpan] = []
    uses: list[list[tuple[int, bool]]] = []
    ranges: list[range] = []
    for polygon in (contours.silhouette, *contours.orange, *contours.white):
        first = len(uses)
        for ring_index, ring in enumerate((polygon.exterior, *polygon.interiors)):
            material_left = _positive_area(ring) == (ring_index == 0)
            ring_uses: list[tuple[int, bool]] = []
            for span in ring.spans:
                controls = span.control_points
                if span.is_line:
                    # Native lines use endpoints only; unused metadata handles
                    # must not prevent recognizing a shared straight boundary.
                    controls = (controls[0], controls[0], controls[3], controls[3])
                reverse = cast("Controls", tuple(reversed(controls)))
                forward = controls <= reverse
                key = span.is_line, controls if forward else reverse
                index = selected.get(key)
                if index is None:
                    index = len(spans)
                    selected[key] = index
                    spans.append(BezierSpan(key[1], key[0]))
                ring_uses.append((index, forward == material_left))
            uses.append(ring_uses)
        ranges.append(range(first, len(uses)))
    limits = boundaries(uses, ranges, len(contours.orange), material_runs_mm)
    failures = {
        index: (spans[index], limit)
        for index, limit in limits.items()
        if limit.offset_mm > 0
        and not curvature.within_curvature(
            spans[index], limit.offset_mm, limit.inward_sign
        )
    }
    if failures:
        raise CurvatureLimitError(contours, failures)


def boundaries(
    uses: Sequence[Sequence[tuple[int, bool]]],
    ranges: Sequence[range],
    orange_components: int,
    material_runs_mm: tuple[float, float, float],
) -> dict[int, BoundaryTaper]:
    """Subtract adjacent material runs; omitted colored owners are black.

    Rings have their material on the left. A reversed shared-chain use puts
    that material on the canonical right. The fixed silhouette is vertical and
    excluded. Runs are cumulative from the black plateau, not boundary widths.
    """
    if len(material_runs_mm) != 3 or any(
        isinstance(value, bool) or not isfinite(value) or value < 0
        for value in material_runs_mm
    ):
        raise ValueError("three finite nonnegative material taper runs are required")
    if not material_runs_mm[0] <= material_runs_mm[1] <= material_runs_mm[2]:
        raise ValueError("material taper runs must follow black, orange, white order")
    sides: dict[int, list[int | None]] = {}
    fixed = {chain for ring in ranges[0] for chain, _ in uses[ring]}
    for polygon, rings in enumerate(ranges[1:], 1):
        label = 1 if polygon <= orange_components else 2
        for ring in rings:
            for chain, forward in uses[ring]:
                owners = sides.setdefault(chain, [None, None])
                side = 0 if forward else 1
                if owners[side] is not None and owners[side] != label:
                    raise ValueError("shared boundary has conflicting material owners")
                owners[side] = label
    result: dict[int, BoundaryTaper] = {}
    for chain, owners in sides.items():
        if chain in fixed:
            continue
        left, right = (label if label is not None else 0 for label in owners)
        offset = abs(material_runs_mm[left] - material_runs_mm[right])
        result[chain] = BoundaryTaper(left, right, offset, -1 if left > right else 1)
    return result
