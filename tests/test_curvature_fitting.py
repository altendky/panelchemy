"""Constrained candidate search preserves boundaries and delegates final proof."""

from math import radians, tan
from typing import cast

import numpy as np
import pytest

from panelchemy import curvature_fitting as fitting
from panelchemy.curvature_fitting import FloatArray, Spline, refit


def source(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    distance = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    return points, distance / distance[-1]


def circle() -> tuple[FloatArray, FloatArray]:
    angle = np.linspace(0, 2 * np.pi, 33)
    points = np.column_stack((np.cos(angle), np.sin(angle)))
    points[-1] = points[0]
    return source(points)


def allow(_candidate: Spline) -> bool:
    return True


def test_periodic_candidate_closes_with_matching_first_and_second_derivatives() -> None:
    points, parameters = circle()
    snapshot = points.copy()
    candidate, diagnostics = refit(
        points, parameters, 0.1, 1, 0.04, periodic=True, accept=allow
    )
    assert candidate is not None
    assert diagnostics["status"] == "accepted"
    for derivative in (0, 1, 2):
        assert np.allclose(
            candidate(0, nu=derivative), candidate(1, nu=derivative), atol=1e-10
        )
    assert np.array_equal(points, snapshot)


def test_open_chain_pins_nonbinary_endpoints_exactly() -> None:
    points = np.array([(1.1, 2.2), (1.7, 2.8), (2.3, 3.4)])
    points, parameters = source(points)
    candidate, _ = refit(points, parameters, 0.2, 1, 0.05, periodic=False, accept=allow)
    assert candidate is not None
    assert np.array_equal(candidate(0), cast("FloatArray", points[0]))
    assert np.array_equal(candidate(1), cast("FloatArray", points[-1]))


def test_signed_constraint_keeps_safe_opposite_bends() -> None:
    parameters = np.linspace(0, 1, 33)
    points = np.column_stack((parameters, parameters**2))
    candidate, _ = refit(
        points, parameters, 1.0, -1, 0.005, periodic=False, accept=allow
    )
    assert candidate is not None
    at = np.linspace(0, 1, 501)
    assert (
        np.max(np.linalg.norm(candidate(at) - np.column_stack((at, at**2)), axis=1))
        < 0.005
    )
    v, a = candidate(at, nu=1), candidate(at, nu=2)
    curvature = (v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]) / np.linalg.norm(v, axis=1) ** 3
    assert np.max(curvature) > 1.9  # Absolute curvature would reject this safe side.


def test_notched_loop_fits_with_taper_curvature_and_source_targets() -> None:
    points, parameters = source(
        np.array(
            [(0, 0), (0.4, 0), (0.4, 0.2), (0.6, 0.2), (0.6, 0.4), (0, 0.4), (0, 0)],
            dtype=np.float64,
        )
    )
    offset = tan(radians(10))
    candidate, _ = refit(
        points, parameters, offset, 1, 0.2, periodic=True, accept=allow
    )
    assert candidate is not None
    at = np.linspace(0, 1, 4001)
    v, a = candidate(at, nu=1), candidate(at, nu=2)
    curvature = (v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]) / np.linalg.norm(v, axis=1) ** 3
    assert np.max(curvature) * offset < 0.9
    target = np.column_stack(
        [np.interp(at, parameters, points[:, axis]) for axis in (0, 1)]
    )
    assert np.max(np.linalg.norm(candidate(at) - target, axis=1)) <= 0.2


def test_final_callback_can_reject_every_numerically_feasible_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))
    called = 0

    def reject(_candidate: Spline) -> bool:
        nonlocal called
        called += 1
        return False

    points, parameters = circle()
    candidate, diagnostics = refit(
        points, parameters, 0.1, 1, 0.04, periodic=True, accept=reject
    )
    assert candidate is None
    assert called > 0
    assert diagnostics["status"] == "unrefined"


def test_unexpected_final_checker_error_propagates() -> None:
    def fail(_candidate: Spline) -> bool:
        raise RuntimeError("continuous checker failed unexpectedly")

    points, parameters = circle()
    with pytest.raises(RuntimeError, match="unexpectedly"):
        _ = refit(points, parameters, 0.1, 1, 0.04, periodic=True, accept=fail)


@pytest.mark.parametrize("limit,status", [(0, "time-budget"), (None, "point-budget")])
def test_explicit_resource_limits_retain_source(
    monkeypatch: pytest.MonkeyPatch, limit: float | None, status: str
) -> None:
    if limit is None:
        monkeypatch.setattr(fitting, "_MAX_POINTS", 8)
    else:
        monkeypatch.setattr(fitting, "_SECONDS", limit)
    points, parameters = circle()
    candidate, diagnostics = refit(
        points, parameters, 0.1, 1, 0.04, periodic=True, accept=allow
    )
    assert candidate is None
    assert diagnostics["status"] == status


@pytest.mark.parametrize("reverse", [True, False])
def test_invalid_parameterization_is_rejected(
    reverse: bool,
) -> None:
    points, parameters = circle()
    invalid = parameters[::-1] if reverse else parameters * float("nan")
    with pytest.raises(ValueError, match="parameters"):
        _ = refit(points, invalid, 0.1, 1, 0.04, periodic=True, accept=allow)
