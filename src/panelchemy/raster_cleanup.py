"""Explicit compact-detail cleanup on a coordinated three-color raster.

This changes the artwork reference before tracing. It never changes the panel
silhouette or silently increases the curve fitter's source-error allowance.
"""

from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from panelchemy.separation import Labels, Mask, SeparationResult

_FOUR = ndimage.generate_binary_structure(2, 1)
_EIGHT = ndimage.generate_binary_structure(2, 2)
_COLORS = ("black", "orange", "white")
type Components = NDArray[np.int32]
type Bounds = tuple[slice, slice]


@dataclass(frozen=True)
class RasterCleanupResult:
    original: SeparationResult
    cleaned: SeparationResult
    changed: Mask
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _Edit:
    pixels: tuple[tuple[int, int], ...]
    donor: int
    recipient: int
    kind: str


def _number(value: object, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(number) or (number < 0 if zero else number <= 0):
        raise ValueError(
            f"{name} must be {'nonnegative' if zero else 'positive'} and finite"
        )
    return number


def _label(mask: Mask) -> tuple[Components, int]:
    labels, count = ndimage.label(mask, structure=_FOUR)
    return np.asarray(labels, dtype=np.int32), int(count)


def raster_statistics(result: SeparationResult) -> dict[str, dict[str, int]]:
    """Describe color occupancy and explicitly four-connected raster topology."""
    counts: dict[str, dict[str, int]] = {}
    for color, name in enumerate(_COLORS):
        mask = cast("Mask", result.silhouette & (result.labels == color))
        _, components = _label(mask)
        holes = ndimage.binary_fill_holes(mask, structure=_FOUR) & ~mask
        _, counters = _label(holes)
        counts[name] = dict(
            pixels=int(np.count_nonzero(mask)), components=components, holes=counters
        )
    return counts


def _extent(bounds: Bounds, pitch: float) -> float:
    return (
        max(cast("int", axis.stop) - cast("int", axis.start) for axis in bounds) * pitch
    )


def _expanded(bounds: Bounds, shape: tuple[int, ...]) -> Bounds:
    return cast(
        "Bounds",
        tuple(
            slice(
                max(0, cast("int", axis.start) - 1),
                min(size, cast("int", axis.stop) + 1),
            )
            for axis, size in zip(bounds, shape, strict=True)
        ),
    )


def _compact_edits(
    source: SeparationResult,
    components: list[Components],
    bounds: list[list[Bounds | None]],
    pitch: float,
    width: float,
    protected: Mask,
    skipped: Counter[str],
) -> list[_Edit]:
    edits: list[_Edit] = []
    for color, parts in enumerate(bounds):
        for number, box in enumerate(parts, 1):
            if box is None or _extent(box, pitch) > width:
                continue
            collar = _expanded(box, source.labels.shape)
            member = cast("Mask", components[color][collar] == number)
            ring = ndimage.binary_dilation(member, structure=_EIGHT) & ~member
            if np.any(protected[collar] & member) or np.any(
                ring & ~source.silhouette[collar]
            ):
                skipped["protected_boundary"] += 1
                continue
            neighbors = cast(
                "list[int]", np.unique(source.labels[collar][ring]).tolist()
            )
            if len(neighbors) != 1 or neighbors[0] == color:
                skipped["ambiguous_or_diagonal_contact"] += 1
                continue
            positions: list[list[int]] = np.argwhere(member).tolist()
            top, left = cast("int", collar[0].start), cast("int", collar[1].start)
            pixels = tuple((y + top, x + left) for y, x in positions)
            edits.append(_Edit(pixels, color, neighbors[0], "compact_component"))
    return edits


def _contact_edits(
    source: SeparationResult,
    components: list[Components],
    bounds: list[list[Bounds | None]],
    pitch: float,
    width: float,
    protected: Mask,
) -> list[_Edit]:
    a, b = source.labels, source.silhouette
    inside = b[:-1, :-1] & b[:-1, 1:] & b[1:, :-1] & b[1:, 1:]
    saddle = cast(
        "Mask",
        (
            (a[:-1, :-1] == a[1:, 1:])
            & (a[:-1, 1:] == a[1:, :-1])
            & (a[:-1, :-1] != a[:-1, 1:])
            & inside
        ),
    )
    edits: list[_Edit] = []
    areas = [np.bincount(parts.ravel()) for parts in components]
    positions: list[list[int]] = np.argwhere(saddle).tolist()
    for y, x in positions:
        collar = _expanded((slice(y, y + 2), slice(x, x + 2)), a.shape)
        if len(np.unique(a[collar])) != 2:
            continue
        diagonals = (((y, x), (y + 1, x + 1)), ((y, x + 1), (y + 1, x)))
        for targets, pixels in (diagonals, diagonals[::-1]):
            color = int(cast("np.uint8", a[targets[0]]))
            first, second = (
                int(cast("np.int32", components[color][p])) for p in targets
            )
            if first == second:
                continue
            boxes = bounds[color][first - 1], bounds[color][second - 1]
            assert boxes[0] is not None and boxes[1] is not None
            compact = [
                _extent(boxes[0], pitch) <= width,
                _extent(boxes[1], pitch) <= width,
            ]
            if compact[0] == compact[1] or any(protected[p] for p in pixels):
                continue
            small, parent = (first, second) if compact[0] else (second, first)
            if areas[color][parent] <= areas[color][small]:
                continue
            child_box = boxes[0] if compact[0] else boxes[1]
            child_window = _expanded(child_box, a.shape)
            member = cast("Mask", components[color][child_window] == small)
            neighborhood = ndimage.binary_dilation(member, structure=_EIGHT)
            donor = int(cast("np.uint8", a[pixels[0]]))
            if (
                np.any(protected[child_window] & member)
                or np.any(neighborhood & ~b[child_window])
                or not set(
                    cast("list[int]", np.unique(a[child_window][neighborhood]).tolist())
                )
                <= {color, donor}
            ):
                continue
            contacts = set(
                cast(
                    "list[int]",
                    np.unique(
                        components[color][child_window][
                            neighborhood & ~member & (a[child_window] == color)
                        ]
                    ).tolist(),
                )
            )
            if contacts != {parent}:
                continue
            edits.append(_Edit(pixels, donor, color, "point_contact"))
    return edits


def _conflicts(edits: list[_Edit], shape: tuple[int, ...]) -> set[int]:
    """Reject both conflicting interpretations, without palette-order priority."""
    owner = np.full(shape, -1, dtype=np.int32)
    conflicts: set[int] = set()
    for index, edit in enumerate(edits):
        collar = {
            (y + dy, x + dx)
            for y, x in edit.pixels
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if 0 <= y + dy < shape[0] and 0 <= x + dx < shape[1]
        }
        for p in collar:
            previous = int(cast("np.int32", owner[p]))
            if previous >= 0:
                conflicts.update((index, previous))
            else:
                owner[p] = index
    return conflicts


def _donor_preserved(labels: Labels, silhouette: Mask, edit: _Edit) -> bool:
    """Sequentially recheck donor connectivity, including earlier remote edits."""
    mask = cast("Mask", silhouette & (labels == edit.donor))
    parts, _ = _label(mask)
    touched = {int(cast("np.int32", parts[p])) for p in edit.pixels}
    for number in touched:
        remaining = cast("Mask", parts == number)
        for p in edit.pixels:
            remaining[p] = False
        _, count = _label(remaining)
        # Contact bridges may not delete a donor or sever a second branch.
        # Standalone compact components are removed through their own rule.
        if count != 1:
            return False
    return True


def cleanup(
    result: SeparationResult,
    long_edge_mm: float = 250,
    minimum_width_mm: float = 0,
) -> RasterCleanupResult:
    """Remove bounded compact details and join eligible diagonal attachments.

    The threshold bounds a component's full pixel-cell bounding box. Connected
    tips, long narrow strokes, and ambiguous three-color junctions are retained.
    This first pass is not general minimum-width enforcement or blanket opening.
    """
    edge = _number(long_edge_mm, "long_edge_mm")
    width = _number(minimum_width_mm, "minimum feature width", zero=True)
    if (
        result.labels.ndim != 2
        or result.labels.dtype != np.uint8
        or result.silhouette.shape != result.labels.shape
        or result.silhouette.dtype != np.bool_
    ):
        raise ValueError(
            "cleanup needs same-size two-dimensional palette labels and boolean silhouette"
        )
    if np.any(result.labels > 2) or not np.any(result.silhouette):
        raise ValueError(
            "cleanup needs a nonempty silhouette and three-color palette labels"
        )
    yy, xx = np.nonzero(result.silhouette)
    pitch = edge / max(int(np.ptp(yy)) + 1, int(np.ptp(xx)) + 1)
    if not isfinite(pitch) or pitch <= 0:
        raise ValueError("physical scale must remain positive and finite")
    before = raster_statistics(result)
    labels = result.labels.copy()
    skipped: Counter[str] = Counter()
    accepted: Counter[str] = Counter()
    edits: list[_Edit] = []
    if width:
        components = [
            _label(cast("Mask", result.silhouette & (labels == color)))[0]
            for color in range(3)
        ]
        boxes = [
            cast("list[Bounds | None]", ndimage.find_objects(parts))
            for parts in components
        ]
        protected = result.silhouette & ~ndimage.binary_erosion(
            result.silhouette, structure=_EIGHT, border_value=0
        )
        edits = _compact_edits(
            result, components, boxes, pitch, width, protected, skipped
        )
        edits.extend(_contact_edits(result, components, boxes, pitch, width, protected))
        conflicts = _conflicts(edits, labels.shape)
        for index, edit in enumerate(edits):
            if index in conflicts:
                skipped["conflicting_proposals"] += 1
                continue
            assert all(
                int(cast("np.uint8", labels[p])) == edit.donor for p in edit.pixels
            )
            if edit.kind == "point_contact" and not _donor_preserved(
                labels, result.silhouette, edit
            ):
                skipped["donor_connectivity"] += 1
                continue
            for p in edit.pixels:
                labels[p] = edit.recipient
            accepted[edit.kind] += 1
    changed = cast("Mask", labels != result.labels)
    assert not np.any(changed & ~result.silhouette)
    cleaned = replace(
        result,
        labels=labels,
        silhouette=result.silhouette.copy(),
        warnings=result.warnings.copy(),
    )
    transitions = {
        f"{old}->{new}": int(
            np.count_nonzero(
                cast("Mask", changed & (result.labels == i) & (labels == j))
            )
        )
        for i, old in enumerate(_COLORS)
        for j, new in enumerate(_COLORS)
        if i != j
    }
    diagnostics: dict[str, object] = dict(
        enabled=width > 0,
        method="compact-components-and-point-contacts",
        connectivity=4,
        hole_count_reference="four-connected enclosed complement",
        minimum_width_mm=width,
        mm_per_pixel=pitch,
        threshold_reference="maximum full-cell component bounding-box extent",
        general_minimum_width_enforced=False,
        fitting_reference="cleaned-raster",
        changed_pixels=int(np.count_nonzero(changed)),
        changed_area_mm2=int(np.count_nonzero(changed)) * pitch * pitch,
        original_labels_sha256=sha256(result.labels.tobytes()).hexdigest(),
        cleaned_labels_sha256=sha256(labels.tobytes()).hexdigest(),
        silhouette_sha256=sha256(result.silhouette.tobytes()).hexdigest(),
        proposed_edits=len(edits),
        accepted_edits=dict(accepted),
        skipped=dict(skipped),
        transitions=transitions,
        before=before,
        after=raster_statistics(cleaned),
    )
    return RasterCleanupResult(result, cleaned, changed, diagnostics)
