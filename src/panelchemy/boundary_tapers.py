"""Taper requirements for shared palette boundaries, in canonical direction."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class BoundaryTaper:
    left: int
    right: int
    offset_mm: float
    inward_sign: int


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
