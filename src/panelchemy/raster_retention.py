"""Physical body cleanup retaining coherent original strokes and short connections.

This is an explicit change to the artwork reference before tracing. It neither
changes the silhouette nor grants additional curve-fitting error.
"""

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from math import ceil, isfinite
from typing import cast, final

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.draw import line  # pyright: ignore[reportUnknownVariableType]

from panelchemy.raster_cleanup import RasterCleanupResult, cleanup, raster_statistics
from panelchemy.separation import Labels, Mask, SeparationResult

_FOUR = ndi.generate_binary_structure(2, 1)
_COLORS = ("black", "orange", "white")
_MAX_PIXELS = 20_000_000
_MAX_KERNEL_CELLS = 1_000_000
_MAX_MORPHOLOGY_WORK = 5_000_000_000
_MAX_WORK = 100_000_000
type Components = NDArray[np.int32]
type Bounds = tuple[slice, slice]


@final
class _Work:
    def __init__(self) -> None:
        self.used = 0

    def spend(self, amount: int) -> None:
        self.used += amount
        if self.used > _MAX_WORK:
            raise ValueError("stroke retention exceeded its bounded raster work limit")


def _number(value: object, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(number) or (number < 0 if zero else number <= 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if zero else 'positive'}"
        )
    return number


def _scale(result: SeparationResult, edge: float) -> float:
    if (
        result.labels.ndim != 2
        or result.labels.dtype != np.uint8
        or result.silhouette.shape != result.labels.shape
        or result.silhouette.dtype != np.bool_
        or not np.any(result.silhouette)
        or np.any(result.labels > 2)
    ):
        raise ValueError(
            "stroke retention requires palette labels and a nonempty boolean silhouette"
        )
    if result.labels.size > _MAX_PIXELS:
        raise ValueError("stroke retention exceeded its raster pixel limit")
    yy, xx = np.nonzero(result.silhouette)
    pitch = edge / max(int(np.ptp(yy)) + 1, int(np.ptp(xx)) + 1)
    if not isfinite(pitch) or pitch <= 0:
        raise ValueError(
            "stroke retention physical scale must remain positive and finite"
        )
    return pitch


def _label(mask: Mask) -> Components:
    return np.asarray(ndi.label(mask, _FOUR)[0], dtype=np.int32)


def _seeds(
    result: SeparationResult,
    pitch: float,
    width: float,
    length: float,
    line_width: float,
) -> tuple[list[Mask], list[Mask], Labels, dict[str, object]]:
    radius = width / (2 * pitch)
    half_length = length / (2 * pitch)
    if not isfinite(radius) or not isfinite(half_length):
        raise ValueError("stroke retention kernel scale must remain finite")
    extent, size = ceil(radius), ceil(half_length) + 1
    if max((2 * extent + 1) ** 2, (2 * size + 1) ** 2) > _MAX_KERNEL_CELLS:
        raise ValueError("stroke retention exceeded its kernel cell limit")
    axis = np.arange(-extent, extent + 1, dtype=np.int64)
    yy, xx = axis[:, None], axis[None, :]
    disk = xx * xx + yy * yy <= radius * radius
    mask = result.silhouette
    border = (
        ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * pitch <= width / 2
    )
    kernels: dict[tuple[bytes, tuple[int, int]], tuple[Mask, tuple[int, int]]] = {}
    for degrees in range(0, 180, 5):
        angle = float(cast("np.float64", np.deg2rad(degrees)))
        y, x = (
            round(half_length * float(cast("np.float64", np.sin(angle)))),
            round(half_length * float(cast("np.float64", np.cos(angle)))),
        )
        rows, cols = cast(
            "tuple[NDArray[np.intp], NDArray[np.intp]]",
            line(size - y, size - x, size + y, size + x),
        )
        kernel = np.zeros((2 * size + 1, 2 * size + 1), dtype=bool)
        kernel[rows, cols] = True
        normal = (
            round(float(cast("np.float64", np.cos(angle))) * radius),
            round(-float(cast("np.float64", np.sin(angle))) * radius),
        )
        kernels[kernel.tobytes(), normal] = kernel, normal
    morphology_work = (
        3
        * result.labels.size
        * (
            int(np.count_nonzero(disk))
            + sum(int(np.count_nonzero(kernel)) for kernel, _ in kernels.values())
        )
    )
    if morphology_work > _MAX_MORPHOLOGY_WORK:
        raise ValueError("stroke retention exceeded its morphology work limit")
    seeds = np.full(result.labels.shape, 255, dtype=np.uint8)
    body: list[Mask] = []
    support: list[Mask] = []
    counts: dict[str, object] = {}
    for color, name in enumerate(_COLORS):
        original = cast("Mask", mask & (result.labels == color))
        opened = ndi.binary_opening(original, structure=disk)
        retained = np.zeros(original.shape, dtype=bool)
        for kernel, normal in kernels.values():
            centers = ndi.binary_erosion(original, structure=kernel)
            first = ndi.shift(
                original,
                normal,
                order=0,
                mode="constant",
                cval=False,
                prefilter=False,
            )
            second = ndi.shift(
                original,
                tuple(-n for n in normal),
                order=0,
                mode="constant",
                cval=False,
                prefilter=False,
            )
            centers &= ~first & ~second
            retained |= original & ndi.binary_dilation(centers, structure=kernel)
        collar = cast(
            "Mask",
            original & (ndi.distance_transform_edt(~retained) * pitch <= line_width / 2)
            if np.any(retained)
            else np.zeros(original.shape, dtype=bool),
        )
        seeds[(opened | collar | (original & border)) & mask] = color
        body.append(opened)
        support.append(collar)
        counts[name] = dict(
            body_pixels=int(np.count_nonzero(opened)),
            stroke_pixels=int(np.count_nonzero(collar)),
            extra_seed_pixels=int(np.count_nonzero(collar & ~opened)),
        )
    if not np.any(cast("Mask", seeds != 255)):
        raise ValueError("stroke retention has no surviving original seeds")
    return (
        body,
        support,
        seeds,
        dict(
            per_color=counts,
            directional_kernels=len(kernels),
            orientation_step_degrees=5,
            morphology_work_units=morphology_work,
            morphology_work_limit=_MAX_MORPHOLOGY_WORK,
        ),
    )


def _bridges(
    original: Labels,
    current: Labels,
    silhouette: Mask,
    body: list[Mask],
    stroke: list[Mask],
    pitch: float,
    gap: float,
    work: _Work,
) -> tuple[Mask, list[dict[str, object]]]:
    maximum = min(Fraction(gap) // Fraction(pitch), original.size)
    restored = np.zeros(original.shape, dtype=bool)
    proposals: list[dict[str, object]] = []
    if not maximum:
        return restored, proposals
    height, width = cast("tuple[int, int]", original.shape)
    for color in range(3):
        source = _label(cast("Mask", silhouette & (original == color)))
        survivors = _label(
            cast("Mask", silhouette & (original == color) & (current == color))
        )
        actual = _label(cast("Mask", silhouette & (current == color)))
        boxes = cast("list[Bounds | None]", ndi.find_objects(survivors))
        body_ids = set(
            cast("list[int]", np.unique(survivors[body[color]]).tolist())
        ) - {0}
        stroke_ids = set(
            cast("list[int]", np.unique(survivors[stroke[color]]).tolist())
        ) - {0}
        actual_body = set(
            cast("list[int]", np.unique(actual[body[color]]).tolist())
        ) - {0}
        for component in sorted(stroke_ids - body_ids):
            box = boxes[component - 1]
            assert box is not None
            anchor = np.argwhere(cast("Mask", survivors[box] == component)).tolist()[0]
            y, x = (
                int(anchor[0]) + cast("int", box[0].start),
                int(anchor[1]) + cast("int", box[1].start),
            )
            if int(cast("np.int32", actual[y, x])) in actual_body:
                continue
            owner = int(cast("np.int32", source[y, x]))
            roi: Bounds = (
                slice(
                    max(0, cast("int", box[0].start) - maximum - 1),
                    min(height, cast("int", box[0].stop) + maximum + 1),
                ),
                slice(
                    max(0, cast("int", box[1].start) - maximum - 1),
                    min(width, cast("int", box[1].stop) + maximum + 1),
                ),
            )
            work.spend(survivors[roi].size)
            local = survivors[roi]
            missing = cast(
                "Mask",
                (
                    silhouette[roi]
                    & (original[roi] == color)
                    & (current[roi] != color)
                    & (source[roi] == owner)
                ),
            )
            terminal = cast(
                "Mask", np.isin(local, list(body_ids)) & (source[roi] == owner)
            )
            if not np.any(missing) or not np.any(terminal):
                continue
            near_body = ndi.binary_dilation(terminal, _FOUR)
            frontier = (
                ndi.binary_dilation(cast("Mask", local == component), _FOUR) & missing
            )
            distance = np.zeros(missing.shape, dtype=np.int64)
            minimum = 0
            hits = np.zeros(missing.shape, dtype=bool)
            for depth in range(1, maximum + 1):
                work.spend(missing.size)
                if not np.any(frontier):
                    break
                distance[frontier] = depth
                hits = frontier & near_body
                if np.any(hits):
                    minimum = depth
                    break
                frontier = cast(
                    "Mask",
                    (ndi.binary_dilation(frontier, _FOUR) & missing & (distance == 0)),
                )
            if not minimum:
                continue
            reverse = np.zeros(missing.shape, dtype=np.int64)
            frontier = hits
            for depth in range(1, minimum + 1):
                work.spend(missing.size)
                reverse[frontier] = depth
                frontier = cast(
                    "Mask",
                    (ndi.binary_dilation(frontier, _FOUR) & missing & (reverse == 0)),
                )
            selected = cast(
                "Mask",
                ((distance > 0) & (reverse > 0) & (distance + reverse == minimum + 1)),
            )
            pixels = np.argwhere(selected)
            pixels += [cast("int", roi[0].start), cast("int", roi[1].start)]
            restored[pixels[:, 0], pixels[:, 1]] = True
            proposals.append(
                dict(
                    color=color,
                    original_component=owner,
                    stroke_component=component,
                    pixels=cast("list[list[int]]", pixels[:, ::-1].tolist()),
                    removed_cell_cost=minimum,
                    gap_mm=minimum * pitch,
                )
            )
    return restored, proposals


def _saddles(labels: Labels, silhouette: Mask) -> Mask:
    encoded = np.where(silhouette, labels, 255)
    a, b, c, d = encoded[:-1, :-1], encoded[:-1, 1:], encoded[1:, :-1], encoded[1:, 1:]
    return cast("Mask", (a == d) & (b == c) & (a != b))


def _joint_guard(
    original: Labels,
    current: Labels,
    proposed: Labels,
    silhouette: Mask,
    seeds: Labels,
    changed: Mask,
    work: _Work,
) -> list[str]:
    failures: set[str] = set()
    for color in range(3):
        before = _label(cast("Mask", silhouette & (current == color)))
        after = _label(cast("Mask", silhouette & (proposed == color)))
        boxes = cast("list[Bounds | None]", ndi.find_objects(before))
        for component in cast(
            "list[int]", np.unique(before[changed & (current == color)]).tolist()
        ):
            if not component:
                continue
            box = boxes[int(component) - 1]
            assert box is not None
            work.spend(before[box].size)
            remaining = set(
                cast(
                    "list[int]",
                    np.unique(after[box][before[box] == component]).tolist(),
                )
            ) - {0}
            if len(remaining) != 1:
                failures.add("donor_empty_or_split")
        source = _label(cast("Mask", silhouette & (original == color)))
        old_memberships: dict[int, set[int]] = {}
        for component in cast(
            "list[int]", np.unique(after[changed & (proposed == color)]).tolist()
        ):
            if not component:
                continue
            work.spend(after.size)
            ancestry = cast(
                "list[int]",
                np.unique(source[(after == component) & (original == color)]).tolist(),
            )
            ancestry = [number for number in ancestry if number]
            for number in ancestry:
                if number not in old_memberships:
                    work.spend(before.size)
                    old_memberships[number] = set(
                        cast(
                            "list[int]",
                            np.unique(
                                before[(source == number) & (current == color)]
                            ).tolist(),
                        )
                    ) - {0}
            work.spend(len(ancestry) ** 2)
            for i, first in enumerate(ancestry):
                if any(
                    not (old_memberships[first] & old_memberships[second])
                    for second in ancestry[i + 1 :]
                ):
                    failures.add("foreign_original_component_relation")
    if np.any(_saddles(proposed, silhouette) & ~_saddles(current, silhouette)):
        failures.add("new_saddle")
    if np.any(cast("Mask", (seeds != 255) & (proposed != seeds))):
        failures.add("seed_changed")
    if np.any(cast("Mask", changed & ((proposed != original) | ~silhouette))):
        failures.add("non_source_restoration")
    return sorted(failures)


def _result(
    result: SeparationResult,
    labels: Labels,
    pitch: float,
    details: Mapping[str, object],
) -> RasterCleanupResult:
    changed = cast("Mask", (labels != result.labels) & result.silhouette)
    cleaned = replace(
        result,
        labels=labels,
        silhouette=result.silhouette.copy(),
        warnings=result.warnings.copy(),
    )
    diagnostics: dict[str, object] = dict(
        enabled=True,
        connectivity=4,
        fitting_reference="cleaned-raster",
        mm_per_pixel=pitch,
        changed_pixels=int(np.count_nonzero(changed)),
        changed_area_mm2=int(np.count_nonzero(changed)) * pitch * pitch,
        original_labels_sha256=sha256(result.labels.tobytes()).hexdigest(),
        cleaned_labels_sha256=sha256(labels.tobytes()).hexdigest(),
        silhouette_sha256=sha256(result.silhouette.tobytes()).hexdigest(),
        before=raster_statistics(result),
        after=raster_statistics(cleaned),
        transitions={
            f"{old}->{new}": int(
                np.count_nonzero(
                    cast("Mask", changed & (result.labels == i) & (labels == j))
                )
            )
            for i, old in enumerate(_COLORS)
            for j, new in enumerate(_COLORS)
            if i != j
        },
        **details,
    )
    return RasterCleanupResult(result, cleaned, changed, diagnostics)


def retain_strokes(
    result: SeparationResult,
    long_edge_mm: float,
    body_width_mm: float,
    stroke_length_mm: float,
    line_width_mm: float,
) -> RasterCleanupResult:
    """Clean body detail while retaining coherent original thin strokes.

    Original support needs background on both sides and directional length.
    Missing original connections may span at most half a print line width.
    Bridges are proposed after one nearest-seed fill, then jointly checked;
    unsafe unions are explicitly rejected, never applied sequentially. Resource
    exhaustion raises ValueError without returning a partial cleaned reference.
    """
    edge = _number(long_edge_mm, "long_edge_mm")
    width = _number(body_width_mm, "body_width_mm")
    length = _number(stroke_length_mm, "stroke_length_mm")
    line_width = _number(line_width_mm, "line_width_mm")
    pitch = _scale(result, edge)
    body, support, seeds, seed_report = _seeds(result, pitch, width, length, line_width)
    indices = ndi.distance_transform_edt(
        cast("Mask", seeds == 255), return_distances=False, return_indices=True
    )
    labels = seeds[tuple(indices)].copy()
    labels[~result.silhouette] = result.labels[~result.silhouette]
    work = _Work()
    changed, proposals = _bridges(
        result.labels,
        labels,
        result.silhouette,
        body,
        support,
        pitch,
        line_width / 2,
        work,
    )
    proposed = labels.copy()
    proposed[changed] = result.labels[changed]
    failures = _joint_guard(
        result.labels, labels, proposed, result.silhouette, seeds, changed, work
    )
    if not failures:
        labels = proposed
    return _result(
        result,
        labels,
        pitch,
        dict(
            method="body-opening-and-directional-stroke-retention",
            body_width_mm=width,
            stroke_length_mm=length,
            line_width_mm=line_width,
            general_minimum_width_enforced=False,
            seed_support=seed_report,
            connection_retention=dict(
                maximum_gap_mm=line_width / 2,
                maximum_removed_cells=Fraction(line_width / 2) // Fraction(pitch),
                proposed_groups=len(proposals),
                proposed_pixels=int(np.count_nonzero(changed)),
                restored_pixels=int(np.count_nonzero(changed)) if not failures else 0,
                accepted=not failures,
                rejected_reasons=failures,
                proposals=proposals,
                work_units=work.used,
                work_limit=_MAX_WORK,
                path_rule="original four-connected missing cells; all equal-shortest paths, no cascade",
            ),
        ),
    )


def prepare_raster(
    result: SeparationResult,
    *,
    long_edge_mm: float = 250,
    compact_width_mm: float = 0,
    body_width_mm: float = 0,
    stroke_length_mm: float = 3,
    line_width_mm: float = 0.5,
) -> RasterCleanupResult | None:
    """Share ordered compact cleanup and stroke retention across all adapters.

    Both disabled returns None. Compact-only preserves its existing exact result
    and diagnostics. Combined statistics, transitions and hashes describe raw
    input to final output; individual stage reports remain nested explicitly.
    """
    compact = _number(compact_width_mm, "compact_width_mm", zero=True)
    body = _number(body_width_mm, "body_width_mm", zero=True)
    if not compact and not body:
        return None
    first = cleanup(result, long_edge_mm, compact) if compact else None
    if not body:
        return first
    retained = retain_strokes(
        first.cleaned if first else result,
        long_edge_mm,
        body,
        stroke_length_mm,
        line_width_mm,
    )
    if first is None:
        return retained
    return _result(
        result,
        retained.cleaned.labels,
        cast("float", retained.diagnostics["mm_per_pixel"]),
        dict(
            method="compact-cleanup-and-stroke-retention",
            body_width_mm=body,
            stroke_length_mm=retained.diagnostics["stroke_length_mm"],
            line_width_mm=retained.diagnostics["line_width_mm"],
            stages=dict(
                compact=first.diagnostics, stroke_retention=retained.diagnostics
            ),
        ),
    )
