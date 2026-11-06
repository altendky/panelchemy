"""Bounded candidate search; continuous source/curvature proofs belong to caller."""

# Optimizer callbacks run synchronously before the span-count loop advances.
# ruff: noqa: B023

from collections.abc import Callable
from math import atan2, cos, isfinite, pi, sin
from time import monotonic
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.optimize import NonlinearConstraint, minimize

type FloatArray = NDArray[np.float64]
type Spline = BSpline[np.float64]
_SPAN_COUNTS = (4, 8, 16, 32)
_MAX_POINTS = 512
_SECONDS = 12.0
_SAMPLED_FRACTION = 0.895


class _BudgetExpired(Exception):
    pass


def _row(points: FloatArray, index: int) -> FloatArray:
    return cast("FloatArray", points[index])


def refit(
    points: FloatArray,
    parameters: FloatArray,
    offset_mm: float,
    inward_sign: int,
    tolerance_mm: float,
    *,
    periodic: bool,
    accept: Callable[[Spline], bool],
) -> tuple[Spline | None, dict[str, object]]:
    """Try smooth candidates without deleting details or changing source points.

    Only ``accept`` can authorize a candidate: sampled constraints are search
    guidance, not continuous distance, curvature, or topology certificates.
    Open chains pin both endpoints; periodic chains share their first poles.
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
    diagnostics: dict[str, object] = {"status": "unrefined", "attempts": 0}
    if offset_mm == 0 or len(points) > _MAX_POINTS:
        diagnostics["status"] = "zero-offset" if offset_mm == 0 else "point-budget"
        return None, diagnostics
    length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    if not isfinite(length) or length <= 0:
        raise ValueError("curvature refit requires a positive finite source length")
    origin = np.mean(points, axis=0)
    source = points - origin
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
    diagnostics["trials"] = trials

    for spans in _SPAN_COUNTS:
        if periodic:
            knots = np.arange(-3, spans + 4, dtype=np.float64) / spans
            transform = np.eye(spans)[np.arange(spans + 3) % spans]
            offsets = np.zeros((spans + 3, 2))
        else:
            knots = np.concatenate(
                (np.zeros(4), np.linspace(0, 1, spans + 1)[1:-1], np.ones(4))
            )
            count = len(knots) - 4
            transform = np.eye(count)[:, 1:-1]
            offsets = np.zeros((count, 2))
            offsets[0], offsets[-1] = source[0], source[-1]
        count = len(knots) - 4
        basis: Spline = BSpline(knots, np.eye(count), 3)
        matrix = basis(times) @ transform
        fixed = basis(times) @ offsets
        initial = np.linalg.lstsq(
            matrix * np.sqrt(weights[:, None]),
            (target - fixed) * np.sqrt(weights[:, None]),
            rcond=None,
        )[0]
        probes = np.unique(
            np.concatenate((np.linspace(0, 1, spans * 8 + 1), parameters))
        )
        probe_basis = basis(probes)
        derivative = basis(probes, nu=1)
        second = basis(probes, nu=2)
        derivative_matrix, second_matrix = derivative @ transform, second @ transform
        derivative_fixed, second_fixed = derivative @ offsets, second @ offsets
        probe_matrix, probe_fixed = probe_basis @ transform, probe_basis @ offsets
        probe_target = np.column_stack(
            [np.interp(probes, parameters, source[:, i]) for i in (0, 1)]
        )
        joins = np.linspace(0, 1, spans + 1)
        join_basis = basis(joins)
        join_matrix, join_fixed = join_basis @ transform, join_basis @ offsets
        join_target = np.column_stack(
            [np.interp(joins, parameters, source[:, i]) for i in (0, 1)]
        )
        normals = np.array([(1, 0), (2**-0.5, 2**-0.5), (0, 1), (-(2**-0.5), 2**-0.5)])
        join_limit = 0.9 * tolerance_mm * cos(pi / 8)
        free = cast("int", initial.shape[0])

        def objective(x: FloatArray) -> float:
            error = matrix @ x.reshape(free, 2) + fixed - target
            return float(
                np.sum(weights * np.sum(error * error, axis=1)) / tolerance_mm**2
            )

        def gradient(x: FloatArray) -> FloatArray:
            error = matrix @ x.reshape(free, 2) + fixed - target
            return cast(
                "FloatArray",
                (2 * matrix.T @ (error * weights[:, None]) / tolerance_mm**2).ravel(),
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
            error = probe_matrix @ poles + probe_fixed - probe_target
            distance = 1 - np.sum(error * error, axis=1) / tolerance_mm**2
            projection = (
                (join_matrix @ poles + join_fixed - join_target)
                @ normals.T
                / join_limit
            )
            return np.concatenate(
                (
                    curvature,
                    distance,
                    (1 - projection).ravel(),
                    (1 + projection).ravel(),
                    speed2 / length**2 - 1e-10,
                )
            )

        def budget(xk: FloatArray) -> None:
            _ = xk  # SciPy's callback protocol requires this argument name.
            if monotonic() - started > _SECONDS:
                raise _BudgetExpired

        seeds = [("least-squares", initial)]
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
            if monotonic() - started > _SECONDS:
                diagnostics["status"] = "time-budget"
                return None, diagnostics
            diagnostics["attempts"] = len(trials) + 1
            try:
                result = minimize(
                    objective,
                    seed.ravel(),
                    jac=gradient,
                    method="SLSQP",
                    constraints=NonlinearConstraint(constraints, 0, np.inf),
                    callback=budget,
                    options={"maxiter": 120, "ftol": 1e-11},
                )
            except _BudgetExpired:
                diagnostics["status"] = "time-budget"
                return None, diagnostics
            selected = cast("FloatArray", result.x)
            feasible = bool(
                np.all(np.isfinite(selected)) and np.min(constraints(selected)) >= -1e-8
            )
            trial: dict[str, object] = {
                "spans": spans,
                "seed": seed_name,
                "optimizer_success": bool(result.success),
                "message": str(result.message),
                "sampled_feasible": feasible,
            }
            trials.append(trial)
            if not result.success or not feasible:
                continue
            coefficients = origin + offsets + transform @ selected.reshape(free, 2)
            candidate: Spline = BSpline(
                knots, coefficients, 3, extrapolate="periodic" if periodic else True
            )
            # Pinned endpoints use original coordinates, avoiding recentering drift.
            if not periodic:
                candidate.c[0], candidate.c[-1] = points[0], points[-1]
            if accept(candidate):
                diagnostics.update(
                    status="accepted",
                    spans=spans,
                    elapsed_seconds=monotonic() - started,
                )
                return candidate, diagnostics
            trial["continuous_gate_rejected"] = True
    diagnostics["elapsed_seconds"] = monotonic() - started
    return None, diagnostics
