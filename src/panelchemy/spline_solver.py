"""Fair spline coefficients with forward and movable-join constraints.

Constraints use fixed source directions. The caller must still certify its
selected curves after conversion/rounding; these optimizer checks do not replace
the continuous distance or topology proofs.
"""

from itertools import pairwise
from math import cos, pi, sqrt
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse import coo_array, csr_array, hstack, vstack
from scipy.sparse.linalg import lsqr

from panelchemy.numeric_threads import single_threaded_blas

type FloatArray = NDArray[np.float64]
type FloatSparse = csr_array[np.float64, tuple[int, int]]


class SplineRefinementNeeded(ValueError):
    """The current basis did not produce a verified constrained candidate."""

    def __init__(
        self, message: str, interval: tuple[float, float] | None = None
    ) -> None:
        super().__init__(message)
        self.interval: tuple[float, float] | None = interval


def _scalar(values: FloatArray, index: int) -> np.float64:
    return cast("np.float64", values[index])


def _derivative_rows(knots: FloatArray) -> FloatSparse:
    count = len(knots) - 4
    denominators = knots[4 : count + 3] - knots[1:count]
    factors = 3 / denominators
    rows = np.repeat(np.arange(count - 1), 2)
    columns = np.column_stack((np.arange(count - 1), np.arange(1, count))).ravel()
    values = np.column_stack((-factors, factors)).ravel()
    return csr_array((values, (rows, columns)), shape=(count - 1, count))


@single_threaded_blas
def solve_coefficients(
    matrix: FloatSparse,
    target: FloatArray,
    transform: FloatSparse,
    offsets: FloatArray,
    knots: FloatArray,
    source_parameters: FloatArray,
    source_points: FloatArray,
    length_mm: float,
    fairing_mm: float,
    *,
    fit_target_mm: float = 0.0,
) -> FloatArray:
    """Solve the quadratic fit with forward and movable-join constraints."""
    count = len(knots) - 4
    if count > 512:
        raise ValueError("constrained spline fit exceeds 512 control points")
    knot_values = cast("list[float]", knots.tolist())
    intervals = [(a, b) for a, b in pairwise(knot_values) if b > a]
    samples = np.asarray(
        [
            t
            for a, b in intervals
            for t in (np.nextafter(a, b), (a + b) / 2, np.nextafter(b, a))
        ]
    )
    derivatives: FloatSparse = BSpline.design_matrix(
        samples, knots[1:-1], 2
    ) @ _derivative_rows(knots)
    forward_rows: list[FloatSparse] = []
    fairness_rows: list[FloatSparse] = []
    directions: list[tuple[float, float]] = []
    margins: list[float] = []
    gauss = (0.5 - 1 / (2 * sqrt(3)), 0.5 + 1 / (2 * sqrt(3)))
    for i, (a, b) in enumerate(intervals):
        h = b - a
        d0, dm, d2 = (derivatives[j : j + 1, :] for j in (3 * i, 3 * i + 1, 3 * i + 2))
        d1 = 2 * dm - (d0 + d2) / 2
        source_a, source_b = (
            np.asarray(
                [
                    np.interp(t, source_parameters, source_points[:, axis])
                    for axis in (0, 1)
                ]
            )
            for t in (a, b)
        )
        delta = source_b - source_a
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12:
            raise ValueError("spline interval has no source forward direction")
        direction = delta / distance
        # Quadratic Bernstein derivative controls, scaled to cubic handle
        # differences. Their projected positivity covers the whole interval.
        forward_rows.extend(row * (h / 3) for row in (d0, d1, d2))
        directions.extend(
            [(float(_scalar(direction, 0)), float(_scalar(direction, 1)))] * 3
        )
        margins.extend([distance * 1e-5] * 3)
        # Two-point Gauss integrates the squared affine second derivative
        # exactly on each span. Repeated corner knots are never crossed.
        second_a, second_b = 2 * (d1 - d0) / h, 2 * (d2 - d1) / h
        fairness_rows.extend(
            ((1 - u) * second_a + u * second_b)
            * sqrt(h / 2)
            * (fairing_mm / length_mm) ** 2
            for u in gauss
        )
    forward: FloatSparse = vstack(forward_rows, format="csr")
    fairness: FloatSparse = vstack(fairness_rows, format="csr")
    system: FloatSparse = vstack((matrix, fairness @ transform), format="csr")
    values = np.vstack((target, -cast("FloatArray", fairness @ offsets)))
    coefficients: FloatArray = np.column_stack(
        [
            lsqr(system, values[:, axis], atol=1e-12, btol=1e-12, iter_lim=4000)[0]
            for axis in (0, 1)
        ]
    )
    directions_array = np.asarray(directions)
    reduced = forward @ transform
    constraints: FloatSparse = hstack(
        (
            reduced.multiply(directions_array[:, :1]),
            reduced.multiply(directions_array[:, 1:]),
        ),
        format="csr",
    )
    lower = np.asarray(margins) - np.sum(
        cast("FloatArray", forward @ offsets) * directions_array, axis=1
    )
    upper = np.full(len(lower), np.inf)
    cap_parameters: FloatArray = np.array([])
    if fit_target_mm > 0:
        joins = np.unique(knots)
        basis: FloatSparse = BSpline.design_matrix(joins, knots, 3)
        moving: FloatSparse = (basis @ transform).tocsr()
        moving.eliminate_zeros()
        source: FloatArray = np.column_stack(
            [
                np.interp(joins, source_parameters, source_points[:, axis])
                for axis in (0, 1)
            ]
        )
        displacement = cast("FloatArray", basis @ offsets) - source
        movable = np.diff(moving.indptr) > 0
        if (
            np.array_equal(source[:1], source[-1:])
            and not np.any((moving[[0]] - moving[[-1]]).data)
            and np.array_equal(displacement[:1], displacement[-1:])
        ):
            movable[-1] = False  # A movable closed seam is one join, not two.
        moving, displacement = moving[movable], displacement[movable]
        cap_parameters = joins[movable]
        # Four two-sided linear bounds form an inscribed octagon: every join
        # stays within the requested Euclidean target, with no nonlinear solver.
        bound = fit_target_mm * cos(pi / 8)
        cap_rows: list[FloatSparse] = []
        cap_lower: list[FloatArray] = []
        cap_upper: list[FloatArray] = []
        for nx, ny in (
            (1, 0),
            (1 / sqrt(2), 1 / sqrt(2)),
            (0, 1),
            (-1 / sqrt(2), 1 / sqrt(2)),
        ):
            cap_rows.append(hstack((moving * nx, moving * ny), format="csr"))
            delta = displacement[:, 0] * nx + displacement[:, 1] * ny
            cap_lower.append(-bound - delta)
            cap_upper.append(bound - delta)
        constraints = vstack((constraints, *cap_rows), format="csr")
        lower = np.concatenate((lower, *cap_lower))
        upper = np.concatenate((upper, *cap_upper))
    initial = np.concatenate((coefficients[:, 0], coefficients[:, 1]))
    initial_values = cast("FloatArray", constraints @ initial)
    if np.all(initial_values >= lower) and np.all(initial_values <= upper):
        return coefficients
    # Diagonal preconditioning keeps very short knot intervals from dominating
    # the numerical scale of the quadratic optimizer.
    column_squares = cast("FloatArray", np.asarray(system.power(2).sum(axis=0)).ravel())
    scale = np.sqrt(np.maximum(column_squares, 1e-20))
    scaled_system = cast(
        "coo_array[np.float64, tuple[int, int]]", system.multiply(1 / scale[None, :])
    ).tocsr()
    scaled_constraints = constraints.multiply(1 / np.tile(scale, 2)[None, :]).toarray()
    free = cast("int", coefficients.shape[0])

    def unpack(z: FloatArray) -> FloatArray:
        return np.column_stack((z[:free] / scale, z[free:] / scale))

    def objective(z: FloatArray) -> float:
        residual = cast("FloatArray", system @ unpack(z)) - values
        return float(np.sum(residual * residual) / 2)

    def gradient(z: FloatArray) -> FloatArray:
        residual = cast("FloatArray", system @ unpack(z)) - values
        result = cast("FloatArray", scaled_system.T @ residual)
        return np.concatenate((result[:, 0], result[:, 1]))

    result = minimize(
        objective,
        initial * np.tile(scale, 2),
        jac=gradient,
        method="SLSQP",
        constraints=LinearConstraint(scaled_constraints, lower, upper),
        options={"maxiter": 300, "ftol": 1e-11},
    )
    selected = unpack(cast("FloatArray", result.x))
    selected_values = cast(
        "FloatArray", constraints @ np.concatenate((selected[:, 0], selected[:, 1]))
    )
    remaining = cast(
        "FloatArray", np.minimum(selected_values - lower, upper - selected_values)
    )
    if (
        not result.success
        or not np.all(np.isfinite(selected))
        or float(np.min(remaining)) < -1e-8
    ):
        message = f"constrained spline optimizer did not converge: {result.message}"
        if fit_target_mm > 0 and np.all(np.isfinite(selected_values)):
            worst = int(np.argmin(remaining))
            forward_count = len(forward_rows)
            if worst < forward_count:
                interval = intervals[worst // 3]
            else:
                at = float(
                    _scalar(
                        cap_parameters, (worst - forward_count) % len(cap_parameters)
                    )
                )
                closed = np.array_equal(source_points[:1], source_points[-1:])
                neighbors = [
                    (a, b)
                    for a, b in intervals
                    if at in (a, b) or (closed and at in (0, 1) and (a == 0 or b == 1))
                ]
                interval = max(neighbors, key=lambda p: p[1] - p[0])
            raise SplineRefinementNeeded(message, interval)
        raise ValueError(message)
    return selected
