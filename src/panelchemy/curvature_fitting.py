"""Bounded candidate search; continuous source/curvature proofs belong to caller."""

# Optimizer callbacks run synchronously before the span-count loop advances.
# ruff: noqa: B023

from collections.abc import Callable
from itertools import pairwise
from math import atan2, cos, isfinite, pi, sin, sqrt
from time import monotonic
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

from panelchemy.fitting_feedback import Refinement
from panelchemy.nearest_polyline import NearestPolyline
from panelchemy.numeric_threads import single_threaded_blas

type FloatArray = NDArray[np.float64]
type Spline = BSpline[np.float64]
_SPAN_COUNTS = (4, 8, 16, 32)
_MAX_SPANS = 32
_MAX_POINTS = 512
_SECONDS = 12.0
_SAMPLED_FRACTION = 0.895
_SOURCE_FRACTION = 0.9
_FTOL = 1e-6
_MAX_REFINEMENTS = 3
_MAX_NEW_PARAMETERS = 16
# Keep edge differences and squared distances representable before GEOS sees an
# optimizer iterate. Such coordinates are far outside any finite source fit.
_MAX_COORDINATE = sqrt(float(np.finfo(np.float64).max)) / 4


class _BudgetExpired(Exception):
    pass


class _Accepted(Exception):
    def __init__(self, spline: Spline) -> None:
        super().__init__("continuous acceptance passed")
        self.spline: Spline = spline


class _Refined(Exception):
    def __init__(
        self, selected: FloatArray, curve: FloatArray, source: FloatArray
    ) -> None:
        super().__init__("additional constraint locations")
        self.selected: FloatArray = selected
        self.curve: FloatArray = curve
        self.source: FloatArray = source


def _additional_parameters(
    values: tuple[float, ...], existing: FloatArray
) -> FloatArray:
    requested = np.asarray(values, dtype=np.float64)
    if (
        requested.ndim != 1
        or not np.all(np.isfinite(requested))
        or np.any((requested < 0) | (requested > 1))
        or any(isinstance(value, (bool, np.bool_)) for value in values)
    ):
        raise ValueError("refinement locations must be finite parameters in0..1")
    unique = np.unique(requested)
    # This only avoids adding numerically identical search locations. It does
    # not change the geometric acceptance tolerance.
    at = np.searchsorted(existing, unique)
    left = existing[np.maximum(0, at - 1)]
    right = existing[np.minimum(len(existing) - 1, at)]
    separated = np.minimum(np.abs(unique - left), np.abs(unique - right)) > (
        16 * np.finfo(np.float64).eps
    )
    return unique[separated][:_MAX_NEW_PARAMETERS]


def _row(points: FloatArray, index: int) -> FloatArray:
    return cast("FloatArray", points[index])


@single_threaded_blas
def refit(
    points: FloatArray,
    parameters: FloatArray,
    offset_mm: float,
    inward_sign: int,
    tolerance_mm: float,
    *,
    periodic: bool,
    accept: Callable[[Spline], bool],
    refine: Callable[[Spline], Refinement] | None = None,
    endpoint_positions: FloatArray | None = None,
    tangent_directions: FloatArray | None = None,
    time_budget_seconds: float | None = None,
    knot_breaks: FloatArray | None = None,
    corner_parameters: FloatArray | None = None,
    c1_parameters: FloatArray | None = None,
    source_search_fraction: float | None = None,
) -> tuple[Spline | None, dict[str, object]]:
    """Try spline candidates without deleting details or changing source points.

    Only ``accept`` can authorize a candidate: sampled constraints are search
    guidance, not continuous distance, curvature, or topology certificates.
    Open chains pin both endpoints; periodic chains share their first poles.
    ``refine`` may suggest more sample locations after rejection; it cannot
    authorize geometry. Retries retain the basis, source data and total budget.
    Open local repairs may pin approved endpoints separately from the source
    and guide both endpoint handles along inherited forward tangent directions.
    The caller still certifies exact tangent joins after canonicalization.
    ``knot_breaks`` selects one explicit clamped open basis instead of the
    default uniform searches; its intervals also determine probe density.
    ``corner_parameters`` repeats selected interior breaks three times for C0
    joins, with both incident derivative limits included in search guidance.
    ``c1_parameters`` repeats disjoint selected breaks twice, permitting a
    curvature jump while retaining a continuous tangent. Both sides are probed.
    ``source_search_fraction`` adjusts numerical guidance within the original
    distance limit; only the unchanged ``accept`` callback authorizes geometry.
    """
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or parameters.shape != (len(points),)
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(parameters))
        or parameters[0] != 0
        or parameters[-1] != 1
        or not np.all(np.diff(parameters) > 0)
    ):
        raise ValueError(
            "curvature refit requires finite points and increasing0..1 parameters"
        )
    if (
        isinstance(offset_mm, bool)
        or isinstance(tolerance_mm, bool)
        or not isfinite(offset_mm)
        or not isfinite(tolerance_mm)
        or offset_mm < 0
        or tolerance_mm <= 0
        or type(inward_sign) is not int
        or inward_sign not in (-1, 1)
        or type(periodic) is not bool
    ):
        raise ValueError(
            "curvature refit needs nonnegative offset, positive tolerance, and signed side"
        )
    if periodic and not np.array_equal(_row(points, 0), _row(points, -1)):
        raise ValueError("periodic curvature refit requires a closed source")
    if periodic and (endpoint_positions is not None or tangent_directions is not None):
        raise ValueError("endpoint positions and tangents require an open refit")
    breaks: FloatArray | None = None
    if knot_breaks is not None:
        if (
            periodic
            or knot_breaks.ndim != 1
            or not 2 <= len(knot_breaks) <= _MAX_SPANS + 1
            or not np.all(np.isfinite(knot_breaks))
            or knot_breaks[0] != 0
            or knot_breaks[-1] != 1
            or not np.all(knot_breaks[1:] > knot_breaks[:-1])
        ):
            raise ValueError(
                "knot breaks require an open refit with finite increasing0..1 intervals and at most32 spans"
            )
        breaks = knot_breaks.copy()
    corners: FloatArray | None = None
    if corner_parameters is not None:
        if (
            periodic
            or breaks is None
            or corner_parameters.ndim != 1
            or not np.all(np.isfinite(corner_parameters))
            or not np.all((corner_parameters > 0) & (corner_parameters < 1))
            or not np.all(corner_parameters[1:] > corner_parameters[:-1])
            or not np.all(np.isin(corner_parameters, breaks))
        ):
            raise ValueError(
                "corner parameters require increasing finite interior breaks of an explicit open basis"
            )
        corners = corner_parameters.copy()
    c1: FloatArray | None = None
    if c1_parameters is not None:
        if (
            periodic
            or breaks is None
            or c1_parameters.ndim != 1
            or not np.all(np.isfinite(c1_parameters))
            or not np.all((c1_parameters > 0) & (c1_parameters < 1))
            or not np.all(c1_parameters[1:] > c1_parameters[:-1])
            or not np.all(np.isin(c1_parameters, breaks))
            or (corners is not None and np.any(np.isin(c1_parameters, corners)))
        ):
            raise ValueError(
                "C1 parameters require increasing finite interior breaks of an explicit open basis, disjoint from corners"
            )
        c1 = c1_parameters.copy()
    source_fraction = (
        _SOURCE_FRACTION if source_search_fraction is None else source_search_fraction
    )
    if (
        isinstance(source_fraction, bool)
        or not isfinite(source_fraction)
        or not 0 < source_fraction <= 1
    ):
        raise ValueError("source search fraction must be finite and in(0,1]")
    pins = points[[0, -1]].copy()
    if endpoint_positions is not None:
        if endpoint_positions.shape != (2, 2) or not np.all(
            np.isfinite(endpoint_positions)
        ):
            raise ValueError("endpoint positions require two finite planar points")
        pins = endpoint_positions.copy()
    tangents: FloatArray | None = None
    if tangent_directions is not None:
        if tangent_directions.shape != (2, 2) or not np.all(
            np.isfinite(tangent_directions)
        ):
            raise ValueError("endpoint tangents require two finite nonzero directions")
        scale = cast("FloatArray", np.max(np.abs(tangent_directions), axis=1))
        if np.min(scale) == 0:
            raise ValueError("endpoint tangents require two finite nonzero directions")
        tangents = tangent_directions / scale[:, None]
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    if time_budget_seconds is not None and (
        isinstance(time_budget_seconds, bool)
        or not isfinite(time_budget_seconds)
        or time_budget_seconds < 0
    ):
        raise ValueError("refit time budget must be finite and nonnegative")
    seconds = (
        _SECONDS if time_budget_seconds is None else min(_SECONDS, time_budget_seconds)
    )
    diagnostics: dict[str, object] = {"status": "unrefined", "attempts": 0}
    if offset_mm == 0 or len(points) > _MAX_POINTS:
        diagnostics["status"] = "zero-offset" if offset_mm == 0 else "point-budget"
        return None, diagnostics
    length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    if not isfinite(length) or length <= 0:
        raise ValueError("curvature refit requires a positive finite source length")
    if seconds <= 0:
        diagnostics["status"] = "time-budget"
        return None, diagnostics
    origin = np.mean(points, axis=0)
    source = points - origin
    nearest_source = NearestPolyline(source)
    closed = np.array_equal(source[:1], source[-1:])
    source_area = float(
        np.sum(source[:-1, 0] * source[1:, 1] - source[1:, 0] * source[:-1, 1])
    )
    orientation = (1.0 if source_area > 0 else -1.0) if closed and source_area else 0.0
    times = np.unique(
        np.concatenate(
            (parameters, np.linspace(0, 1, min(512, max(64, 2 * len(points)))))
        )
    )
    target = np.column_stack(
        [np.interp(times, parameters, source[:, i]) for i in (0, 1)]
    )
    gaps = np.diff(times)
    weights = np.concatenate((gaps[:1], gaps[:-1] + gaps[1:], gaps[-1:])) / 2
    started = monotonic()
    trials: list[dict[str, object]] = []
    diagnostics.update(
        trials=trials, source_search_fraction=source_fraction, ftol=_FTOL
    )

    base_times = times.copy()
    extra_curve: FloatArray = np.empty(0)
    extra_source: FloatArray = np.empty(0)
    adaptive_retries = 0
    diagnostics["adaptive_retries"] = 0
    span_counts = _SPAN_COUNTS if breaks is None else (len(breaks) - 1,)
    for spans in span_counts:
        if periodic:
            knots = np.arange(-3, spans + 4, dtype=np.float64) / spans
            transform = np.eye(spans)[np.arange(spans + 3) % spans]
            offsets = np.zeros((spans + 3, 2))
        else:
            knots = np.concatenate(
                (
                    np.zeros(4),
                    np.linspace(0, 1, spans + 1)[1:-1]
                    if breaks is None
                    else breaks[1:-1]
                    if corners is None and c1 is None
                    else np.repeat(
                        breaks[1:-1],
                        np.where(
                            np.isin(breaks[1:-1], () if corners is None else corners),
                            3,
                            np.where(
                                np.isin(breaks[1:-1], () if c1 is None else c1), 2, 1
                            ),
                        ),
                    ),
                    np.ones(4),
                )
            )
            count = len(knots) - 4
            transform = np.eye(count)[:, 1:-1]
            offsets = np.zeros((count, 2))
            offsets[0], offsets[-1] = pins[0] - origin, pins[-1] - origin
        count = len(knots) - 4
        basis: Spline = BSpline(knots, np.eye(count), 3)
        matrix = basis(times) @ transform
        fixed = basis(times) @ offsets
        initial = np.linalg.lstsq(
            matrix * np.sqrt(weights[:, None]),
            (target - fixed) * np.sqrt(weights[:, None]),
            rcond=None,
        )[0]
        endpoint_constraints: tuple[LinearConstraint, ...] = ()
        if tangents is not None:
            centered_pins = pins - origin
            normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
            normal_rows = np.zeros((2, initial.size))
            normal_rows[0, :2], normal_rows[1, -2:] = normals
            normal_targets = np.sum(normals * centered_pins, axis=1)
            forward_rows = np.zeros_like(normal_rows)
            forward_rows[0, :2] = tangents[0]
            forward_rows[1, -2:] = -tangents[1]
            # Match the existing sampled speed floor at clamped endpoints.
            minimum_handle = (
                length * 1e-5 / (3 * spans)
                if breaks is None
                else length * 1e-5 * np.diff(breaks)[[0, -1]] / 3
            )
            forward_targets = (
                np.array(
                    (
                        tangents[0] @ centered_pins[0],
                        -tangents[1] @ centered_pins[1],
                    )
                )
                + minimum_handle
            )
            endpoint_constraints = (
                LinearConstraint(normal_rows, normal_targets, normal_targets),
                LinearConstraint(forward_rows, forward_targets, np.inf),
            )
        seeds = [("least-squares", initial)]
        if c1 is not None and len(c1) and tangents is not None:
            centered = pins - origin
            chord = _row(centered, 1) - _row(centered, 0)
            chord_length = float(np.linalg.norm(chord))
            if chord_length > 0:
                greville = np.asarray(
                    [np.mean(knots[i + 1 : i + 4]) for i in range(count)],
                    dtype=np.float64,
                )
                straight = _row(centered, 0) + greville[:, None] * chord
                # A second physical seed avoids encoding every source ledge in
                # the starting curve. Greville distances supply its handle
                # lengths; the inherited endpoint rays remain constrained.
                straight[1] = (
                    _row(centered, 0) + _row(tangents, 0) * chord_length * greville[1]
                )
                straight[-2] = _row(centered, 1) - _row(tangents, 1) * chord_length * (
                    1 - greville[-2]
                )
                seeds.append(("tangent-chord", straight[1:-1]))
        if periodic:
            center = (np.min(source, axis=0) + np.max(source, axis=0)) / 2
            delta = _row(source, 0) - center
            area = float(
                np.sum(source[:-1, 0] * source[1:, 1] - source[1:, 0] * source[:-1, 1])
            )
            if np.linalg.norm(delta) > 0 and area != 0:
                step = 2 * pi / spans
                direction = 1 if area > 0 else -1
                phase = atan2(
                    float(cast("np.float64", delta[1])),
                    float(cast("np.float64", delta[0])),
                )
                factor = 2 * (2 + cos(step)) / (3 * (1 + cos(step)))
                radius = offset_mm / _SAMPLED_FRACTION * factor * 1.01
                pole_radius = radius * 3 / (2 + cos(step))
                rounded = np.asarray(
                    [
                        center
                        + pole_radius
                        * np.asarray(
                            (
                                cos(phase + direction * (i - 1) * step),
                                sin(phase + direction * (i - 1) * step),
                            )
                        )
                        for i in range(spans)
                    ]
                )
                seeds.append(("round-loop", rounded))
        for seed_name, seed in seeds:
            warm = seed.ravel()
            for adaptive_round in range(_MAX_REFINEMENTS + 1):
                times = np.unique(np.concatenate((base_times, extra_source)))
                target = np.column_stack(
                    [np.interp(times, parameters, source[:, i]) for i in (0, 1)]
                )
                gaps = np.diff(times)
                weights = (
                    np.concatenate((gaps[:1], gaps[:-1] + gaps[1:], gaps[-1:])) / 2
                )
                interval_probes: FloatArray = (
                    np.linspace(0, 1, spans * 8 + 1)
                    if breaks is None
                    else np.concatenate(
                        [
                            np.linspace(a, b, 9)
                            for a, b in pairwise(cast("list[float]", breaks.tolist()))
                        ]
                    )
                )
                probes = np.unique(
                    np.concatenate((interval_probes, parameters, extra_curve))
                )
                if corners is not None or c1 is not None:
                    # At repeated knots, native evaluation selects the right
                    # side. C1 curvature can jump too; guide both limits.
                    joins = np.concatenate(
                        (
                            np.empty(0) if corners is None else corners,
                            np.empty(0) if c1 is None else c1,
                        )
                    )
                    probes = np.unique(
                        np.concatenate((probes, np.nextafter(joins, 0.0)))
                    )
                probe_basis = basis(probes)
                derivative = basis(probes, nu=1)
                second = basis(probes, nu=2)
                derivative_matrix, second_matrix = (
                    derivative @ transform,
                    second @ transform,
                )
                derivative_fixed, second_fixed = derivative @ offsets, second @ offsets
                probe_matrix, probe_fixed = (
                    probe_basis @ transform,
                    probe_basis @ offsets,
                )
                probe_gaps = np.diff(probes)
                probe_weights = (
                    np.concatenate(
                        (
                            probe_gaps[:1],
                            probe_gaps[:-1] + probe_gaps[1:],
                            probe_gaps[-1:],
                        )
                    )
                    / 2
                )
                search_limit = source_fraction * tolerance_mm
                free = cast("int", initial.shape[0])
                cached_x: FloatArray | None = None
                cached_geometry: (
                    tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None
                ) = None

                def geometry(
                    x: FloatArray,
                ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
                    nonlocal cached_x, cached_geometry
                    if cached_x is not None and np.array_equal(x, cached_x):
                        assert cached_geometry is not None
                        return cached_geometry
                    proxy = probe_matrix @ x.reshape(free, 2) + probe_fixed
                    if not np.all(np.isfinite(proxy)) or np.any(
                        np.abs(proxy) > _MAX_COORDINATE
                    ):
                        # Preserve the numerical-failure path for invalid solver steps;
                        # they must not become exceptions from the geometry index.
                        cached_x = x.copy()
                        cached_geometry = (
                            np.full_like(proxy, np.nan),
                            np.full_like(target, np.nan),
                            np.zeros((len(target), free)),
                            proxy,
                        )
                        return cached_geometry
                    forward_error = proxy - nearest_source.project(proxy).points
                    reverse = NearestPolyline(proxy).project(target)
                    reverse_error = reverse.points - target
                    at = reverse.segments
                    fraction = reverse.fractions[:, None]
                    reverse_basis = (1 - fraction) * probe_matrix[
                        at
                    ] + fraction * probe_matrix[at + 1]
                    cached_x = x.copy()
                    cached_geometry = forward_error, reverse_error, reverse_basis, proxy
                    return cached_geometry

                def objective(x: FloatArray) -> float:
                    forward_error, reverse_error, _, _ = geometry(x)
                    return float(
                        (
                            np.sum(probe_weights * np.sum(forward_error**2, axis=1))
                            + np.sum(weights * np.sum(reverse_error**2, axis=1))
                        )
                        / (2 * tolerance_mm**2)
                    )

                def gradient(x: FloatArray) -> FloatArray:
                    forward_error, reverse_error, reverse_basis, _ = geometry(x)
                    return cast(
                        "FloatArray",
                        (
                            (
                                probe_matrix.T
                                @ (forward_error * probe_weights[:, None])
                                + reverse_basis.T @ (reverse_error * weights[:, None])
                            )
                            / tolerance_mm**2
                        ).ravel(),
                    )

                def constraints(x: FloatArray) -> FloatArray:
                    poles = x.reshape(free, 2)
                    v, a = (
                        derivative_matrix @ poles + derivative_fixed,
                        second_matrix @ poles + second_fixed,
                    )
                    speed2 = np.sum(v * v, axis=1)
                    cross = v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]
                    curvature = (
                        _SAMPLED_FRACTION
                        - inward_sign
                        * offset_mm
                        * cross
                        / np.maximum(speed2, length**2 * 1e-16) ** 1.5
                    )
                    forward_error, reverse_error, _, proxy = geometry(x)
                    winding = (
                        np.array(
                            [
                                orientation
                                * np.sum(
                                    proxy[:, 0] * np.roll(proxy[:, 1], -1)
                                    - proxy[:, 1] * np.roll(proxy[:, 0], -1)
                                )
                                / length**2
                                - 1e-10
                            ]
                        )
                        if orientation
                        else np.empty(0)
                    )
                    return np.concatenate(
                        (
                            curvature,
                            1 - np.sum(forward_error**2, axis=1) / search_limit**2,
                            1 - np.sum(reverse_error**2, axis=1) / search_limit**2,
                            winding,
                            speed2 / length**2 - 1e-10,
                        )
                    )

                def constraint_jacobian(x: FloatArray) -> FloatArray:
                    poles = x.reshape(free, 2)
                    v, a = (
                        derivative_matrix @ poles + derivative_fixed,
                        second_matrix @ poles + second_fixed,
                    )
                    speed2 = np.sum(v * v, axis=1)
                    floor = length**2 * 1e-16
                    safe_speed2 = np.maximum(speed2, floor)
                    cross = v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]
                    speed_jacobian = 2 * derivative_matrix[:, :, None] * v[:, None, :]
                    cross_jacobian = np.stack(
                        (
                            derivative_matrix * a[:, 1:2] - second_matrix * v[:, 1:2],
                            second_matrix * v[:, :1] - derivative_matrix * a[:, :1],
                        ),
                        axis=2,
                    )
                    # The denominator is constant where the speed floor is active.
                    curvature_jacobian = (
                        -inward_sign
                        * offset_mm
                        * (
                            cross_jacobian / safe_speed2[:, None, None] ** 1.5
                            - 1.5
                            * cross[:, None, None]
                            * speed_jacobian
                            * (speed2 > floor)[:, None, None]
                            / safe_speed2[:, None, None] ** 2.5
                        )
                    )
                    forward_error, reverse_error, reverse_basis, proxy = geometry(x)
                    # Nearest projections minimize squared distance, so their parameter
                    # derivatives cancel. Only the foot's blended endpoint basis remains.
                    forward_jacobian = (
                        -2 * probe_matrix[:, :, None] * forward_error[:, None, :]
                    ) / search_limit**2
                    reverse_jacobian = (
                        -2 * reverse_basis[:, :, None] * reverse_error[:, None, :]
                    ) / search_limit**2
                    area_derivative = np.column_stack(
                        (
                            np.roll(proxy[:, 1], -1) - np.roll(proxy[:, 1], 1),
                            np.roll(proxy[:, 0], 1) - np.roll(proxy[:, 0], -1),
                        )
                    )
                    winding_jacobian = (
                        (
                            orientation * probe_matrix.T @ area_derivative / length**2
                        ).reshape(1, 2 * free)
                        if orientation
                        else np.empty((0, 2 * free))
                    )
                    # Match constraints(): curvature, directed distances, winding, speed.
                    return np.concatenate(
                        (
                            curvature_jacobian.reshape(-1, 2 * free),
                            forward_jacobian.reshape(-1, 2 * free),
                            reverse_jacobian.reshape(-1, 2 * free),
                            winding_jacobian,
                            (speed_jacobian / length**2).reshape(-1, 2 * free),
                        )
                    )

                physical_lower = np.concatenate(
                    (
                        np.full(len(probes), _SAMPLED_FRACTION - 0.9),
                        np.full(len(probes) + len(target), 1 - 1 / source_fraction**2),
                        np.zeros(int(bool(orientation))),
                        np.zeros(len(probes)),
                    )
                )

                def feasible(x: FloatArray) -> bool:
                    physical = bool(
                        np.all(np.isfinite(x))
                        and np.all(constraints(x) >= physical_lower - 1e-8)
                    )
                    if not physical or tangents is None:
                        return physical
                    poles = x.reshape(free, 2)
                    handles = np.stack(
                        (poles[0] - centered_pins[0], centered_pins[1] - poles[-1])
                    )
                    return bool(
                        np.all(
                            np.abs(np.sum(handles * normals, axis=1))
                            <= max(1.0, length) * 1e-8
                        )
                        and np.all(np.sum(handles * tangents, axis=1) > 0)
                    )

                def candidate(x: FloatArray) -> Spline:
                    coefficients = origin + offsets + transform @ x.reshape(free, 2)
                    proposal: Spline = BSpline(
                        knots,
                        coefficients,
                        3,
                        extrapolate="periodic" if periodic else True,
                    )
                    if not periodic:
                        proposal.c[0], proposal.c[-1] = pins[0], pins[-1]
                    return proposal

                def feedback(proposal: Spline) -> tuple[FloatArray, FloatArray] | None:
                    if refine is None or adaptive_round >= _MAX_REFINEMENTS:
                        return None
                    hints = refine(proposal)
                    curve = _additional_parameters(hints.curve_parameters, probes)
                    source_queries = _additional_parameters(
                        hints.source_parameters, times
                    )
                    return (
                        (curve, source_queries)
                        if len(curve) or len(source_queries)
                        else None
                    )

                def budget(xk: FloatArray) -> None:
                    nonlocal iterations, early_checks, next_check
                    iterations += 1
                    if monotonic() - started > seconds:
                        raise _BudgetExpired
                    if early_checks < 3 and iterations >= next_check and feasible(xk):
                        early_checks += 1
                        next_check = iterations + 8
                        proposal = candidate(xk)
                        if accept(proposal):
                            raise _Accepted(proposal)
                        additions = feedback(proposal)
                        if additions is not None:
                            raise _Refined(xk.copy(), *additions)

                if monotonic() - started > seconds:
                    diagnostics["status"] = "time-budget"
                    return None, diagnostics
                diagnostics["attempts"] = len(trials) + 1
                iterations, early_checks, next_check = 0, 0, 0
                try:
                    result = minimize(
                        objective,
                        warm,
                        jac=gradient,
                        method="SLSQP",
                        constraints=(
                            [
                                NonlinearConstraint(
                                    constraints, 0, np.inf, jac=constraint_jacobian
                                ),
                                *endpoint_constraints,
                            ]
                            if endpoint_constraints
                            else NonlinearConstraint(
                                constraints, 0, np.inf, jac=constraint_jacobian
                            )
                        ),
                        callback=budget,
                        options={"maxiter": 120, "ftol": _FTOL},
                    )
                except _Accepted as success:
                    trials.append(
                        dict(
                            spans=spans,
                            seed=seed_name,
                            optimizer_success=None,
                            message="stopped at a continuously certified candidate",
                            sampled_feasible=True,
                            certified_early=True,
                            early_checks=early_checks,
                            nit=iterations,
                        )
                    )
                    diagnostics.update(
                        status="accepted",
                        spans=spans,
                        certified_early=True,
                        elapsed_seconds=monotonic() - started,
                    )
                    return success.spline, diagnostics
                except _Refined as restart:
                    trials.append(
                        dict(
                            spans=spans,
                            seed=seed_name,
                            optimizer_success=None,
                            continuous_gate_rejected=True,
                            refinement_round=adaptive_round,
                            added_curve_parameters=len(restart.curve),
                            added_source_parameters=len(restart.source),
                            nit=iterations,
                        )
                    )
                    extra_curve = np.unique(
                        np.concatenate((extra_curve, restart.curve))
                    )
                    extra_source = np.unique(
                        np.concatenate((extra_source, restart.source))
                    )
                    warm = restart.selected
                    adaptive_retries += 1
                    diagnostics["adaptive_retries"] = adaptive_retries
                    continue
                except _BudgetExpired:
                    diagnostics["status"] = "time-budget"
                    return None, diagnostics
                selected = cast("FloatArray", result.x)
                sampled_feasible = feasible(selected)
                trial: dict[str, object] = {
                    "spans": spans,
                    "seed": seed_name,
                    "optimizer_success": bool(result.success),
                    "message": str(result.message),
                    "sampled_feasible": sampled_feasible,
                    "early_checks": early_checks,
                    "nfev": int(result.nfev),
                    "njev": int(result.njev),
                    "nit": int(result.nit),
                }
                trials.append(trial)
                if not sampled_feasible:
                    break
                proposal = candidate(selected)
                if accept(proposal):
                    diagnostics.update(
                        status="accepted",
                        spans=spans,
                        elapsed_seconds=monotonic() - started,
                    )
                    return proposal, diagnostics
                trial["continuous_gate_rejected"] = True
                additions = feedback(proposal)
                if additions is None:
                    break
                curve, source_queries = additions
                trial.update(
                    refinement_round=adaptive_round,
                    added_curve_parameters=len(curve),
                    added_source_parameters=len(source_queries),
                )
                extra_curve = np.unique(np.concatenate((extra_curve, curve)))
                extra_source = np.unique(np.concatenate((extra_source, source_queries)))
                warm = selected.copy()
                adaptive_retries += 1
                diagnostics["adaptive_retries"] = adaptive_retries
    diagnostics["elapsed_seconds"] = monotonic() - started
    return None, diagnostics
