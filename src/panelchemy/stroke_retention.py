"""Preview bounded widening of coherent thin stroke bodies.

Centerlines are evidence about geometry, not a claim about artistic intent.
These proposals are separate from model construction and do not authorize CAD.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256
from math import ceil, floor, hypot, inf, isfinite, sqrt
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.morphology import medial_axis  # pyright: ignore[reportUnknownVariableType]

from panelchemy.raster_cleanup import raster_statistics
from panelchemy.separation import Mask, SeparationResult

type Pixel = tuple[int, int]
type Point = tuple[float, float]
type Grid = NDArray[np.int32]
type Run = tuple[int, float, float]
_FOUR = ndimage.generate_binary_structure(2, 1)
_EIGHT = ndimage.generate_binary_structure(2, 2)
_COLORS = ("black", "orange", "white")
_medial_axis = cast("Callable[..., Mask]", medial_axis)


@dataclass(frozen=True)
class StrokeGrowthPreview:
    source_reference: SeparationResult
    reference: SeparationResult
    proposed: SeparationResult
    changed: Mask
    decisions: tuple[dict[str, object], ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _Sample:
    origin: Point
    normal: Point
    center: Point
    width_mm: float
    runs: tuple[list[Run], list[Run]]
    position_mm: float


@dataclass(frozen=True)
class _Proposal:
    decision: int
    color: int
    component: int
    donor: int
    pixels: tuple[Pixel, ...]
    samples: tuple[_Sample, ...]


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


def _ray_runs(
    grid: Grid,
    origin: Point,
    direction: Point,
    limit: float,
    *,
    visits: list[Pixel] | None = None,
) -> list[Run]:
    """Traverse exact raster cell edges along a floating-point unit direction."""
    py, px = origin
    dy, dx = direction
    y, x = floor(py), floor(px)
    height, width = cast("tuple[int, int]", grid.shape)

    def label(row: int, column: int) -> int:
        return (
            int(cast("np.int32", grid[row, column]))
            if 0 <= row < height and 0 <= column < width
            else -1
        )

    sy, sx = (1 if dy > 0 else -1), (1 if dx > 0 else -1)
    ty = ((y + 1 - py) / dy if dy > 0 else (y - py) / dy) if dy else inf
    tx = ((x + 1 - px) / dx if dx > 0 else (x - px) / dx) if dx else inf
    step_y, step_x = (abs(1 / dy) if dy else inf), (abs(1 / dx) if dx else inf)
    current, start = label(y, x), 0.0
    runs: list[Run] = []
    for _ in range(ceil(2 * limit) + 4):
        if visits is not None:
            visits.append((y, x))
        stop = min(ty, tx, limit)
        if stop >= limit:
            runs.append((current, start, limit))
            return runs
        vertical, horizontal = ty == stop, tx == stop
        touch = (
            vertical
            and horizontal
            and (label(y + sy, x) != current or label(y, x + sx) != current)
        )
        if vertical:
            y += sy
            ty += step_y
        if horizontal:
            x += sx
            tx += step_x
        following = label(y, x)
        if touch or following != current:
            runs.append((current, start, stop))
            if touch and following == current:
                runs.append((-2, stop, stop))
            current, start = following, stop
    raise ValueError("ray traversal exceeded its geometric cell-crossing bound")


def _branches(
    skeleton: Mask, components: Grid, *, prune_length_px: float = 0
) -> tuple[list[tuple[Pixel, ...]], set[Pixel]]:
    positions: list[list[int]] = np.argwhere(skeleton).tolist()
    pixels = {(y, x) for y, x in positions}
    adjacency: dict[Pixel, list[Pixel]] = {}
    for y, x in sorted(pixels):
        adjacent: list[Pixel] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y + dy, x + dx)
                if (
                    not (dy or dx)
                    or q not in pixels
                    or components[q] != components[y, x]
                ):
                    continue
                # Do not add a diagonal shortcut across an existing L step.
                if dy and dx and ((y + dy, x) in pixels or (y, x + dx) in pixels):
                    continue
                adjacent.append(q)
        adjacency[y, x] = sorted(adjacent)
    terminals = {p for p, neighbors in adjacency.items() if len(neighbors) <= 1}
    removed: set[Pixel] = set()
    # One simultaneous pass on the raw graph, not recursive peeling. Raster
    # corner spurs should not split a substantial trunk into pixel-sized paths.
    # These nodes disappear only from analysis; source pixels and the original
    # donor-tip guards remain intact.
    for first in sorted(terminals):
        if len(adjacency[first]) != 1:
            continue
        spur, previous, current, length = [first], first, adjacency[first][0], 0.0
        while True:
            length += hypot(current[0] - previous[0], current[1] - previous[1])
            if length >= prune_length_px:
                break
            if len(adjacency[current]) != 2:
                if len(adjacency[current]) >= 3:
                    removed.update(spur)
                break
            spur.append(current)
            following = next(p for p in adjacency[current] if p != previous)
            previous, current = current, following
    if removed:
        adjacency = {
            p: [q for q in neighbors if q not in removed]
            for p, neighbors in adjacency.items()
            if p not in removed
        }
        pixels.difference_update(removed)
    anchors = {p for p, neighbors in adjacency.items() if len(neighbors) != 2}
    visited: set[tuple[Pixel, Pixel]] = set()
    paths: list[tuple[Pixel, ...]] = []

    def edge(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
        return (a, b) if a < b else (b, a)

    def walk(first: Pixel, second: Pixel) -> None:
        path, previous, current = [first], first, second
        while True:
            visited.add(edge(previous, current))
            path.append(current)
            if current in anchors or current == first:
                break
            following = next(p for p in adjacency[current] if p != previous)
            previous, current = current, following
        paths.append(tuple(path))

    for first in sorted(anchors):
        if not adjacency[first]:
            paths.append((first,))
        for second in adjacency[first]:
            if edge(first, second) not in visited:
                walk(first, second)
    for first in sorted(pixels):
        for second in adjacency[first]:
            if edge(first, second) not in visited:
                walk(first, second)
    return paths, terminals


def _samples(
    path: tuple[Pixel, ...], grid: Grid, pitch: float, target: float
) -> tuple[list[_Sample], float, float]:
    raw_y, raw_x = [p[0] + 0.5 for p in path], [p[1] + 0.5 for p in path]
    if len(path) < 2:
        return [], 0.0, 0.0
    # Estimate direction at print scale; do not confuse even-width skeleton
    # tie-breaking or regular pixel stairs with meaningful path tortuosity.
    ys: NDArray[np.float64] = ndimage.gaussian_filter1d(
        np.asarray(raw_y), target / (2 * pitch), mode="nearest"
    )
    xs: NDArray[np.float64] = ndimage.gaussian_filter1d(
        np.asarray(raw_x), target / (2 * pitch), mode="nearest"
    )
    ys[0], ys[-1], xs[0], xs[-1] = raw_y[0], raw_y[-1], raw_x[0], raw_x[-1]
    arc = [0.0]
    for i in range(1, len(path)):
        arc.append(
            arc[-1]
            + hypot(
                float(cast("np.float64", ys[i] - ys[i - 1])),
                float(cast("np.float64", xs[i] - xs[i - 1])),
            )
            * pitch
        )
    length = arc[-1]
    coherence = (
        hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1]) * pitch / length
        if length
        else 0.0
    )
    if length < 3 * target:
        return [], length, coherence
    samples: list[_Sample] = []
    limit = 3 * target / pitch
    for index, position in enumerate(arc):
        left, right = (
            max(0.0, position - target / 2),
            min(length, position + target / 2),
        )
        dy = float(np.interp(right, arc, ys) - np.interp(left, arc, ys))
        dx = float(np.interp(right, arc, xs) - np.interp(left, arc, xs))
        norm = hypot(dy, dx)
        if norm == 0:
            continue
        normal = (-dx / norm, dy / norm)
        origin = (raw_y[index], raw_x[index])
        positive = _ray_runs(grid, origin, normal, limit)
        negative = _ray_runs(grid, origin, (-normal[0], -normal[1]), limit)
        a, b = positive[0][2], negative[0][2]
        center = (
            origin[0] + (a - b) / 2 * normal[0],
            origin[1] + (a - b) / 2 * normal[1],
        )
        samples.append(
            _Sample(
                origin, normal, center, (a + b) * pitch, (positive, negative), position
            )
        )
    return samples, length, coherence


def _growth_pixels(
    samples: list[_Sample],
    length: float,
    grid: Grid,
    components: Grid,
    component: int,
    color: int,
    pitch: float,
    target: float,
    cap: float,
) -> tuple[tuple[Pixel, ...], float]:
    ys, xs = [s.origin[0] for s in samples], [s.origin[1] for s in samples]
    pad = ceil(2 * target / pitch) + 2
    height, width = cast("tuple[int, int]", grid.shape)
    top, bottom = max(0, floor(min(ys)) - pad), min(height, ceil(max(ys)) + pad)
    left, right = max(0, floor(min(xs)) - pad), min(width, ceil(max(xs)) + pad)
    proposed = np.zeros((bottom - top, right - left), dtype=bool)
    for sample in samples:
        if sample.width_mm >= target:
            continue
        t = min(
            1.0, sample.position_mm / target, (length - sample.position_mm) / target
        )
        blend = t * t * (3 - 2 * t)
        radius = (sample.width_mm + (target - sample.width_mm) * blend) / (2 * pitch)
        cy, cx = sample.center
        y0, y1 = max(top, floor(cy - radius)), min(bottom, ceil(cy + radius))
        x0, x1 = max(left, floor(cx - radius)), min(right, ceil(cx + radius))
        yy = np.arange(y0, y1, dtype=float)[:, None]
        xx = np.arange(x0, x1, dtype=float)[None, :]
        # Cover cells whose interiors intersect the tube. The fixed-reference
        # whole-cell displacement cap below controls raster coverage overshoot.
        disk = (
            np.maximum(np.abs(yy + 0.5 - cy) - 0.5, 0) ** 2
            + np.maximum(np.abs(xx + 0.5 - cx) - 0.5, 0) ** 2
            < radius * radius
        )
        proposed[y0 - top : y1 - top, x0 - left : x1 - left] |= disk
    added = cast("Mask", proposed & (grid[top:bottom, left:right] != color))
    if not np.any(added):
        return (), 0.0
    own = cast("Mask", components[top:bottom, left:right] == component)
    _, nearest = ndimage.distance_transform_edt(~own, return_indices=True)
    locations: list[list[int]] = np.argwhere(added).tolist()
    nearest = np.asarray(nearest, dtype=np.int32)
    squared_cap = Fraction(cap) ** 2
    squared_pitch = Fraction(pitch) ** 2
    selected: list[Pixel] = []
    largest = 0
    ends: list[tuple[Point, Point]] = []
    for end, neighbor in ((samples[0], samples[1]), (samples[-1], samples[-2])):
        tangent = (-end.normal[1], end.normal[0])
        dot = sum(
            (b - a) * t
            for a, b, t in zip(end.origin, neighbor.origin, tangent, strict=True)
        )
        if dot < 0:
            tangent = (-tangent[0], -tangent[1])
        ends.append((end.origin, tangent))
    for y, x in locations:
        dy = y - int(cast("np.int32", nearest[0, y, x]))
        dx = x - int(cast("np.int32", nearest[1, y, x]))
        distance2 = dy * dy + dx * dx
        if distance2 * squared_pitch > squared_cap:
            continue
        p = y + top, x + left
        # Clip an entire terminal slab, rather than cutting a circular hole out
        # of the tube: a hole could leave detached growth around a thin tip.
        if any(
            ((p[0] + 0.5 - ey) * ty + (p[1] + 0.5 - ex) * tx) * pitch <= target / 2
            for (ey, ex), (ty, tx) in ends
        ):
            continue
        selected.append(p)
        largest = max(largest, distance2)
    return tuple(selected), sqrt(largest) * pitch


def _gap_guard(
    grid: Grid, proposal: _Proposal, pitch: float, target: float
) -> tuple[bool, float, str]:
    """Recheck widths and both adjacent gaps on the same fixed source normals."""
    limit = 3 * target / pitch
    widths: list[float] = []
    for sample in proposal.samples:
        new_runs = (
            _ray_runs(grid, sample.origin, sample.normal, limit),
            _ray_runs(
                grid, sample.origin, (-sample.normal[0], -sample.normal[1]), limit
            ),
        )
        if any(runs[0][0] != proposal.color for runs in new_runs):
            return False, 0.0, "protected-stroke-body"
        if (
            sample.position_mm >= target
            and proposal.samples[-1].position_mm - sample.position_mm >= target
        ):
            widths.append((new_runs[0][0][2] + new_runs[1][0][2]) * pitch)
        for before, after in zip(sample.runs, new_runs, strict=True):
            if len(before) < 2:
                continue
            donor, start, stop = before[1]
            if donor < 0 or len(after) < 2 or after[1][0] != donor:
                return False, 0.0, "protected-gap"
            old_gap = (stop - start) * pitch
            new_gap = (after[1][2] - after[1][1]) * pitch
            if new_gap < min(old_gap, target):
                return False, 0.0, "protected-gap"
    achieved = float(np.median(widths)) if widths else 0.0
    return bool(widths) and achieved >= target, achieved, "insufficient-width"


def _audit_growth(
    proposed: Grid,
    reference: Grid,
    proposals: list[_Proposal],
    accepted: set[int],
    bodies: list[tuple[int, _Sample]],
    decisions: list[dict[str, object]],
    pitch: float,
    target: float,
) -> None:
    """Remove interacting edits until all final, fixed-reference checks pass."""
    owners = np.full(proposed.shape, -1, dtype=np.int32)
    for index in accepted:
        for p in proposals[index].pixels:
            owners[p] = index
    limit = 3 * target / pitch
    rejected: set[int] = set()
    while accepted:
        rejected.clear()

        def reject_along(sample: _Sample) -> None:
            visits: list[Pixel] = []
            for direction in (sample.normal, (-sample.normal[0], -sample.normal[1])):
                _ = _ray_runs(proposed, sample.origin, direction, limit, visits=visits)
            for y, x in visits:
                if 0 <= y < owners.shape[0] and 0 <= x < owners.shape[1]:
                    owner = int(cast("np.int32", owners[y, x]))
                    if owner >= 0:
                        rejected.add(owner)

        for index in sorted(accepted):
            proposal = proposals[index]
            safe, achieved, _ = _gap_guard(proposed, proposal, pitch, target)
            if not safe:
                rejected.add(index)
                for sample in proposal.samples:
                    reject_along(sample)
            else:
                decisions[proposal.decision]["median_width_after_mm"] = achieved
        nearby = ndimage.maximum_filter(
            owners >= 0, size=2 * ceil(limit) + 1, mode="constant", cval=0
        )
        # A donor stroke may be oblique to the growing stroke. Its own source
        # normals must also retain min(original width, target), not just remain
        # topologically connected. Wide and unmodified branches count too.
        for color, sample in bodies:
            if not nearby[floor(sample.origin[0]), floor(sample.origin[1])]:
                continue
            runs = (
                _ray_runs(proposed, sample.origin, sample.normal, limit),
                _ray_runs(
                    proposed,
                    sample.origin,
                    (-sample.normal[0], -sample.normal[1]),
                    limit,
                ),
            )
            width = (runs[0][0][2] + runs[1][0][2]) * pitch
            if any(side[0][0] != color for side in runs) or width < min(
                sample.width_mm, target
            ):
                reject_along(sample)
        if not rejected:
            return
        for index in sorted(rejected):
            proposal = proposals[index]
            for p in proposal.pixels:
                proposed[p], owners[p] = reference[p], -1
            decision = decisions[proposal.decision]
            decision.update(outcome="unchanged", reason="final-neighbor-protection")
            _ = decision.pop("changed_pixels", None)
            _ = decision.pop("median_width_after_mm", None)
        accepted.difference_update(rejected)


def preview_stroke_growth(
    reference: SeparationResult,
    *,
    long_edge_mm: float = 250,
    target_width_mm: float = 0.5,
    max_growth_mm: float = 0.25,
) -> StrokeGrowthPreview:
    """Propose coherent stroke widening; leave uncertain details unchanged."""
    edge = _number(long_edge_mm, "long edge")
    target = _number(target_width_mm, "target width")
    cap = _number(max_growth_mm, "maximum growth", zero=True)
    if not isfinite(3 * target):
        raise ValueError("target width exceeds finite physical sampling limits")
    if target > edge:
        raise ValueError("target width must not exceed the panel long edge")
    if (
        reference.labels.ndim != 2
        or reference.labels.dtype != np.uint8
        or reference.silhouette.shape != reference.labels.shape
        or reference.silhouette.dtype != np.bool_
        or not np.any(reference.silhouette)
        or np.any(reference.labels > 2)
    ):
        raise ValueError(
            "stroke preview requires a nonempty coordinated three-color raster"
        )
    yy, xx = np.nonzero(reference.silhouette)
    source_pitch = edge / max(int(np.ptp(yy)) + 1, int(np.ptp(xx)) + 1)
    sampling_ratio = 4 * (source_pitch / target)
    if not isfinite(sampling_ratio) or source_pitch <= 0:
        raise ValueError("physical sampling scale must remain positive and finite")
    factor = max(1, ceil(sampling_ratio))
    if reference.labels.size * factor * factor > 16_000_000:
        raise ValueError(
            "stroke preview processing grid exceeds the 16,000,000-pixel resource limit"
        )
    labels = np.repeat(np.repeat(reference.labels, factor, axis=0), factor, axis=1)
    silhouette = np.repeat(
        np.repeat(reference.silhouette, factor, axis=0), factor, axis=1
    )
    working = replace(
        reference,
        labels=labels,
        silhouette=silhouette,
        warnings=reference.warnings.copy(),
    )
    pitch = source_pitch / factor
    grid = labels.astype(np.int32)
    source_protected = reference.silhouette & ~ndimage.binary_erosion(
        reference.silhouette, structure=_EIGHT, border_value=0
    )
    protected = np.repeat(np.repeat(source_protected, factor, axis=0), factor, axis=1)
    decisions: list[dict[str, object]] = []
    proposals: list[_Proposal] = []
    tip_regions: list[Mask] = []
    atlases: list[Grid] = []
    bodies: list[tuple[int, _Sample]] = []
    for color, name in enumerate(_COLORS):
        mask = cast("Mask", silhouette & (labels == color))
        atlas = np.asarray(ndimage.label(mask, structure=_FOUR)[0], dtype=np.int32)
        atlases.append(atlas)
        skeleton = np.asarray(_medial_axis(mask, rng=0), dtype=np.bool_)
        paths, terminals = _branches(skeleton, atlas, prune_length_px=target / pitch)
        tips = np.zeros(mask.shape, dtype=bool)
        for p in terminals:
            tips[p] = True
        tip_regions.append(
            ndimage.distance_transform_edt(~tips) * pitch
            <= target / 2 + pitch / sqrt(2)
            if terminals
            else tips
        )
        for path in paths:
            samples, length, coherence = _samples(path, grid, pitch, target)
            component = int(cast("np.int32", atlas[path[0]]))
            rows, columns = [p[0] for p in path], [p[1] for p in path]
            decision: dict[str, object] = dict(
                id=len(decisions),
                color=name,
                component=component,
                bounds_px=[min(columns), min(rows), max(columns) + 1, max(rows) + 1],
                length_mm=length,
                coherence=coherence,
                outcome="unchanged",
                semantic_importance="unknown",
                classification="uncertain",
            )
            decisions.append(decision)
            body = [s for s in samples if target <= s.position_mm <= length - target]
            # Curved/closed paths cannot propose growth, but remain donors
            # whose measured widths need protection from neighboring edits.
            bodies.extend(
                (color, sample) for sample in (samples if coherence < 0.85 else body)
            )
            if not samples or coherence < 0.85:
                decision["reason"] = (
                    "short-terminal-or-noise"
                    if length < 3 * target
                    else "curvature-or-closed-path"
                )
                continue
            if not body:
                decision["reason"] = "insufficient-body"
                continue
            widths = [s.width_mm for s in body]
            decision["median_width_before_mm"] = float(np.median(widths))
            if sum(w < target for w in widths) < 0.75 * len(widths):
                decision["reason"] = "already-wide-or-mixed-neck"
                continue
            decision["classification"] = "coherent-thin-stroke"
            if not cap:
                decision["reason"] = "growth-disabled"
                continue
            pixels, displacement = _growth_pixels(
                samples, length, grid, atlas, component, color, pitch, target, cap
            )
            decision["growth_displacement_upper_bound_mm"] = displacement
            if not pixels:
                decision["reason"] = "growth-cap-or-processing-resolution"
                continue
            if any(not silhouette[p] or protected[p] for p in pixels):
                decision["reason"] = "protected-panel-boundary"
                continue
            donors = {int(cast("np.int32", grid[p])) for p in pixels}
            if len(donors) != 1:
                decision["reason"] = "multiple-donor-colors"
                continue
            donor = donors.pop()
            collar = {
                (y + dy, x + dx)
                for y, x in pixels
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if 0 <= y + dy < labels.shape[0] and 0 <= x + dx < labels.shape[1]
            }
            if any(
                int(cast("np.int32", grid[p])) not in {color, donor} for p in collar
            ):
                decision["reason"] = "third-color-contact"
                continue
            if any(
                int(cast("np.int32", atlas[p])) not in {0, component} for p in collar
            ):
                decision["reason"] = "unrelated-region-contact"
                continue
            decision["proposed_pixels"] = len(pixels)
            decision["reason"] = "pending-joint-checks"
            proposals.append(
                _Proposal(
                    len(decisions) - 1, color, component, donor, pixels, tuple(samples)
                )
            )

    # Competing edits are rejected symmetrically before any color gains pixels.
    owners: dict[Pixel, int] = {}
    conflicts: set[int] = set()
    for index, proposal in enumerate(proposals):
        for y, x in proposal.pixels:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    p = y + dy, x + dx
                    previous = owners.setdefault(p, index)
                    if previous != index:
                        conflicts.update((index, previous))
    proposed = grid.copy()
    accepted: set[int] = set()
    for index, proposal in enumerate(proposals):
        decision = decisions[proposal.decision]
        if index in conflicts:
            decision["reason"] = "competing-growth"
            continue
        if any(tip_regions[proposal.donor][p] for p in proposal.pixels):
            decision["reason"] = "protected-terminal-detail"
            continue
        trial = proposed.copy()
        for p in proposal.pixels:
            trial[p] = proposal.color
        before_count = int(
            ndimage.label(
                cast("Mask", silhouette & (proposed == proposal.color)), structure=_FOUR
            )[1]
        )
        after_count = int(
            ndimage.label(
                cast("Mask", silhouette & (trial == proposal.color)), structure=_FOUR
            )[1]
        )
        if before_count != after_count:
            decision["reason"] = "recipient-topology"
            continue
        rows, columns = [p[0] for p in proposal.pixels], [p[1] for p in proposal.pixels]
        box = (
            slice(max(0, min(rows) - 1), max(rows) + 2),
            slice(max(0, min(columns) - 1), max(columns) + 2),
        )

        def saddles(a: Grid) -> Mask:
            return cast(
                "Mask",
                (a[:-1, :-1] == a[1:, 1:])
                & (a[:-1, 1:] == a[1:, :-1])
                & (a[:-1, :-1] != a[:-1, 1:]),
            )

        if np.any(saddles(trial[box]) & ~saddles(proposed[box])):
            decision["reason"] = "new-diagonal-contact"
            continue
        # Every touched donor component must retain a connected, nonempty body.
        donor_atlas = atlases[proposal.donor]
        touched = {int(cast("np.int32", donor_atlas[p])) for p in proposal.pixels}
        safe = True
        for part in touched:
            remaining = cast("Mask", (donor_atlas == part) & (trial == proposal.donor))
            if int(ndimage.label(remaining, structure=_FOUR)[1]) != 1:
                safe = False
                break
        if not safe:
            decision["reason"] = "donor-topology"
            continue
        safe, achieved, reason = _gap_guard(trial, proposal, pitch, target)
        if not safe:
            decision["reason"] = reason
            continue
        proposed = trial
        accepted.add(index)
        decision.update(
            outcome="proposed-growth",
            reason="bounded-coherent-body",
            median_width_after_mm=achieved,
            changed_pixels=len(proposal.pixels),
        )
    _audit_growth(proposed, grid, proposals, accepted, bodies, decisions, pitch, target)
    for color, atlas in enumerate(atlases):
        count = ndimage.label(
            cast("Mask", silhouette & (proposed == color)), structure=_FOUR
        )[1]
        if count != int(atlas.max()):
            raise ValueError("stroke growth final component audit failed")
    fixed = protected | ~silhouette
    if not np.array_equal(proposed[fixed], grid[fixed]):
        raise ValueError("stroke growth changed the protected panel boundary")
    output = replace(
        working, labels=proposed.astype(np.uint8), silhouette=silhouette.copy()
    )
    changed = cast("Mask", output.labels != labels)
    diagnostics: dict[str, object] = dict(
        method="coherent-stroke-growth-preview",
        preview_only=True,
        target_width_mm=target,
        max_growth_mm=cap,
        minimum_length_mm=3 * target,
        minimum_coherence=0.85,
        minimum_thin_body_fraction=0.75,
        direction_smoothing_scale_mm=target / 2,
        analysis_spur_pruning="one raw leaf-to-junction pass; source pixels and raw donor tips retained",
        analysis_spur_length_limit_mm=target,
        protected_terminal_radius_mm=target / 2,
        final_joint_audit=True,
        donor_body_width_floor="min(original width, target), at coherent source-normal samples",
        source_mm_per_pixel=source_pitch,
        mm_per_pixel=pitch,
        working_scale_factor=factor,
        source_dimensions_px=list(reference.labels.shape),
        processing_dimensions_px=list(labels.shape),
        width_method="first pixel-cell boundary crossings along a centerline normal",
        growth_bound="whole added cells to fixed original material; not boundary Hausdorff distance",
        fitting_reference_changed=False,
        semantic_importance_inferred=False,
        source_reference_labels_sha256=sha256(reference.labels.tobytes()).hexdigest(),
        reference_labels_sha256=sha256(labels.tobytes()).hexdigest(),
        proposed_labels_sha256=sha256(output.labels.tobytes()).hexdigest(),
        silhouette_sha256=sha256(silhouette.tobytes()).hexdigest(),
        changed_pixels=int(np.count_nonzero(changed)),
        changed_area_mm2=int(np.count_nonzero(changed)) * pitch * pitch,
        before=raster_statistics(working),
        after=raster_statistics(output),
        outcomes=dict(Counter(str(d["reason"]) for d in decisions)),
    )
    return StrokeGrowthPreview(
        reference, working, output, changed, tuple(decisions), diagnostics
    )
