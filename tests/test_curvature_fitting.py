"""Constrained candidate search preserves boundaries and delegates final proof."""

# Direct probes also cover internal constraint-location resource limits.
# pyright: reportPrivateUsage=false

from collections.abc import Callable
from itertools import pairwise
from math import nextafter, radians, sqrt, tan
from types import SimpleNamespace
from typing import NoReturn, cast

import numpy as np
import pytest
from scipy.interpolate import BSpline, PPoly
from scipy.optimize import LinearConstraint, NonlinearConstraint

from panelchemy import curvature, geometric_distance
from panelchemy import curvature_fitting as fitting
from panelchemy import curves as c
from panelchemy.curvature_fitting import FloatArray, Spline, refit
from panelchemy.fitting_feedback import Refinement


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


@pytest.mark.parametrize(
    "breaks",
    [
        np.array([]),
        np.array([0.0]),
        np.array([[0.0, 1.0]]),
        np.array([0.0, 0.4, 0.4, 1.0]),
        np.array([0.0, 0.6, 0.5, 1.0]),
        np.array([-0.1, 1.0]),
        np.array([0.0, 0.9]),
        np.array([0.0, np.nan, 1.0]),
        np.array([0.0, np.inf, 1.0]),
        np.linspace(0, 1, 34),
    ],
)
def test_invalid_explicit_knot_breaks_are_rejected(breaks: FloatArray) -> None:
    points, parameters = source(np.array([(0.0, 0.0), (1.0, 0.0)]))
    with pytest.raises(ValueError, match="knot breaks"):
        _ = refit(
            points,
            parameters,
            0.1,
            1,
            0.2,
            periodic=False,
            accept=allow,
            knot_breaks=breaks,
        )


def test_explicit_knot_breaks_cannot_be_periodic() -> None:
    points, parameters = circle()
    with pytest.raises(ValueError, match="knot breaks require an open"):
        _ = refit(
            points,
            parameters,
            0.1,
            1,
            0.2,
            periodic=True,
            accept=allow,
            knot_breaks=np.array([0.0, 0.1, 1.0]),
        )


@pytest.mark.parametrize(
    "corners",
    [
        np.array([[0.5]]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([-0.1]),
        np.array([1.1]),
        np.array([0.5, 0.5]),
        np.array([0.75, 0.25]),
        np.array([0.3]),
    ],
)
def test_corner_parameters_require_unique_interior_explicit_breaks(
    corners: FloatArray,
) -> None:
    points, parameters = source(np.array([(0.0, 0.0), (1.0, 0.0)]))
    with pytest.raises(ValueError, match="corner parameters"):
        _ = refit(
            points,
            parameters,
            0.1,
            1,
            0.2,
            periodic=False,
            accept=allow,
            knot_breaks=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            corner_parameters=corners,
        )


def test_corner_parameters_require_an_explicit_basis() -> None:
    points, parameters = source(np.array([(0.0, 0.0), (1.0, 0.0)]))
    with pytest.raises(ValueError, match="explicit open basis"):
        _ = refit(
            points,
            parameters,
            0.1,
            1,
            0.2,
            periodic=False,
            accept=allow,
            corner_parameters=np.array([0.5]),
        )


def test_periodic_corner_basis_is_rejected() -> None:
    points, parameters = circle()
    with pytest.raises(ValueError, match="open"):
        _ = refit(
            points,
            parameters,
            0.1,
            1,
            0.2,
            periodic=True,
            accept=allow,
            knot_breaks=np.array([0.0, 0.5, 1.0]),
            corner_parameters=np.array([0.5]),
        )


@pytest.mark.parametrize(
    "parameters",
    [
        np.array([[0.5]]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([0.3]),
        np.array([0.5, 0.5]),
        np.array([0.75, 0.25]),
    ],
)
def test_c1_parameters_require_unique_interior_explicit_breaks(
    parameters: FloatArray,
) -> None:
    points, times = source(np.array([(0.0, 0.0), (1.0, 0.0)]))
    with pytest.raises(ValueError, match="C1 parameters"):
        _ = refit(
            points,
            times,
            0.1,
            1,
            0.2,
            periodic=False,
            accept=allow,
            knot_breaks=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            c1_parameters=parameters,
        )


@pytest.mark.parametrize("periodic", [False, True])
def test_c1_parameters_require_an_explicit_open_basis(periodic: bool) -> None:
    points, times = circle() if periodic else source(np.array([(0.0, 0.0), (1.0, 0.0)]))
    with pytest.raises(ValueError, match="C1 parameters"):
        _ = refit(
            points,
            times,
            0.1,
            1,
            0.2,
            periodic=periodic,
            accept=allow,
            c1_parameters=np.array([0.5]),
        )


def test_c1_and_corner_parameters_must_be_disjoint() -> None:
    points, times = source(np.array([(0.0, 0.0), (1.0, 0.0)]))
    with pytest.raises(ValueError, match="disjoint"):
        _ = refit(
            points,
            times,
            0.1,
            1,
            0.2,
            periodic=False,
            accept=allow,
            knot_breaks=np.array([0.0, 0.5, 1.0]),
            corner_parameters=np.array([0.5]),
            c1_parameters=np.array([0.5]),
        )


def test_mixed_c2_c1_and_c0_breaks_retain_their_distinct_continuity() -> None:
    points, times = source(np.array([(0.0, 0.0), (1.0, 0.0)]))
    breaks = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    c1, corners = np.array([0.25]), np.array([0.75])
    snapshots = [value.copy() for value in (breaks, c1, corners)]
    proposal, _ = refit(
        points,
        times,
        0.1,
        1,
        0.2,
        periodic=False,
        accept=allow,
        knot_breaks=breaks,
        c1_parameters=c1,
        corner_parameters=corners,
    )
    assert proposal is not None
    assert [proposal.t.tolist().count(t) for t in (0.25, 0.5, 0.75)] == [2, 1, 3]
    for value, before in zip((breaks, c1, corners), snapshots, strict=True):
        assert np.array_equal(value, before)


@pytest.mark.parametrize("fraction", [None, 0.9, 1.0])
def test_real_c1_curve_retains_tangent_continuity_and_allows_curvature_jump(
    fraction: float | None,
) -> None:
    known: Spline = BSpline(
        [0.0] * 4 + [0.5] * 2 + [1.0] * 4,
        np.array([(-2, 0), (-1, 0), (-0.5, 0.5), (0.5, 1.5), (1, 1.5), (2, 1.5)]),
        3,
    )
    times = np.linspace(0, 1, 65)
    points = known(times)

    def accept(proposal: Spline) -> bool:
        spans: list[c.BezierSpan] = []
        for left, right in ((0.0, 0.5), (0.5, 1.0)):
            h = right - left
            a, d = proposal(left), proposal(right)
            b, e = (
                a + proposal(left, nu=1) * h / 3,
                d - proposal(nextafter(right, left), nu=1) * h / 3,
            )
            controls = cast(
                "c.Controls",
                tuple(
                    tuple(p)
                    for p in cast("list[list[float]]", np.stack((a, b, e, d)).tolist())
                ),
            )
            spans.append(c.BezierSpan(controls, False))
        return geometric_distance.within_distance(
            [(p[0], p[1]) for p in cast("list[list[float]]", points.tolist())],
            spans,
            0.01,
        ) and all(curvature.within_curvature(span, 0.1, 1) for span in spans)

    proposal, diagnostics = refit(
        points,
        times,
        0.1,
        1,
        0.01,
        periodic=False,
        accept=accept,
        knot_breaks=np.array([0.0, 0.5, 1.0]),
        c1_parameters=np.array([0.5]),
        source_search_fraction=fraction,
    )
    assert proposal is not None
    assert diagnostics["spans"] == 2
    assert proposal.t.tolist().count(0.5) == 2
    assert np.allclose(
        proposal(nextafter(0.5, 0), nu=1), proposal(0.5, nu=1), atol=1e-10
    )
    assert not np.allclose(
        proposal(nextafter(0.5, 0), nu=2), proposal(0.5, nu=2), atol=1e-2
    )


def test_c1_search_probes_both_incident_curvature_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.array([(0.0, 0.0), (1.0, 1.0), (2.0, 1.5)])
    times = np.array([0.0, 0.5, 1.0])
    knots = [0.0] * 4 + [0.5] * 2 + [1.0] * 4
    poles = np.array([(0, 0), (0.1, 0), (0.8, 0.5), (1.2, 1.5), (1.5, 1.5), (2, 1.5)])
    known: Spline = BSpline(knots, poles, 3)
    left = nextafter(0.5, 0)
    probes = np.unique(
        np.concatenate((np.linspace(0, 0.5, 9), np.linspace(0.5, 1, 9), [left]))
    )
    at = cast("list[float]", probes.tolist())

    def kappa(t: float) -> float:
        v, a = known(t, nu=1), known(t, nu=2)
        return float(
            cast("np.float64", (v[0] * a[1] - v[1] * a[0]) / np.linalg.norm(v) ** 3)
        )

    assert kappa(left) > 0.9 and kappa(0.5) < 0

    def forbidden(_proposal: Spline) -> NoReturn:
        raise AssertionError("left C1 curvature must fail sampled guidance")

    def solve(_objective: object, _initial: FloatArray, **kwargs: object) -> object:
        selected = (poles[1:-1] - np.mean(points, axis=0)).ravel()
        constraint = cast("NonlinearConstraint", kwargs["constraints"])
        values = cast("Callable[[FloatArray], FloatArray]", constraint.fun)(selected)
        assert values[at.index(left)] == pytest.approx(
            fitting._SAMPLED_FRACTION - kappa(left)
        )
        assert values[at.index(0.5)] == pytest.approx(
            fitting._SAMPLED_FRACTION - kappa(0.5)
        )
        cast("Callable[[FloatArray], None]", kwargs["callback"])(selected)
        return SimpleNamespace(
            x=selected, success=True, message="finished", nit=1, nfev=1, njev=1
        )

    monkeypatch.setattr(fitting, "minimize", solve)
    proposal, _ = refit(
        points,
        times,
        1.0,
        1,
        2.0,
        periodic=False,
        accept=forbidden,
        knot_breaks=np.array([0.0, 0.5, 1.0]),
        c1_parameters=np.array([0.5]),
    )
    assert proposal is None


@pytest.mark.parametrize(
    "fraction", [0.0, -0.1, 1.01, float("nan"), float("inf"), True]
)
def test_invalid_source_search_fraction_is_rejected(fraction: float) -> None:
    points, times = source(np.array([(0.0, 0.0), (1.0, 0.0)]))
    with pytest.raises(ValueError, match="source search fraction"):
        _ = refit(
            points,
            times,
            0.1,
            1,
            0.2,
            periodic=False,
            accept=allow,
            source_search_fraction=fraction,
        )


def test_c1_chord_seed_can_recover_after_source_fitted_seed_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.linspace(0, 1, 33)
    pins = np.array([(1.1, 2.2), (3.1, 4.2)])
    start, end = cast("FloatArray", pins[0]), cast("FloatArray", pins[1])
    points = start + times[:, None] * (end - start)
    points[1:-1, 1] += 0.002 * (-1.0) ** np.arange(31)
    calls = 0

    def accept(proposal: Spline) -> bool:
        # The inspected seed's coefficients lie on this chord, so its full
        # curve is the line whose real source-distance certificate runs here.
        a, b = (
            cast("list[float]", proposal(0).tolist()),
            cast("list[float]", proposal(1).tolist()),
        )
        return geometric_distance.within_distance(
            [(p[0], p[1]) for p in cast("list[list[float]]", points.tolist())],
            (c._line((a[0], a[1]), (b[0], b[1])),),
            0.01,
        )

    def solve(_objective: object, initial: FloatArray, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(
                x=np.full_like(initial, np.nan),
                success=False,
                message="rejected first seed",
                nit=1,
                nfev=1,
                njev=1,
            )
        assert calls == 2
        origin = np.mean(points, axis=0)
        poles = initial.reshape(-1, 2) + origin
        expected = start + np.array([1 / 6, 1 / 3, 2 / 3, 5 / 6])[:, None] * (
            end - start
        )
        assert np.allclose(poles, expected, atol=1e-14)
        cast("Callable[[FloatArray], None]", kwargs["callback"])(initial)
        return SimpleNamespace(
            x=initial, success=True, message="done", nit=1, nfev=1, njev=1
        )

    monkeypatch.setattr(fitting, "minimize", solve)
    proposal, diagnostics = refit(
        points,
        times,
        0.1,
        1,
        0.01,
        periodic=False,
        accept=accept,
        endpoint_positions=pins,
        tangent_directions=np.array([(1.0, 1.0), (1.0, 1.0)]),
        knot_breaks=np.array([0.0, 0.5, 1.0]),
        c1_parameters=np.array([0.5]),
    )
    assert proposal is not None and calls == 2
    assert diagnostics["status"] == "accepted"
    trials = cast("list[dict[str, object]]", diagnostics["trials"])
    assert [trial["seed"] for trial in trials] == ["least-squares", "tangent-chord"]


def test_c0_basis_follows_real_piecewise_source_and_retains_exact_pins() -> None:
    points, parameters = source(np.array([(1.1, 2.2), (2.1, 2.2), (2.1, 3.2)]))
    breaks, corners = np.array([0.0, 0.5, 1.0]), np.array([0.5])
    snapshots = [value.copy() for value in (points, parameters, breaks, corners)]
    checked = 0

    def accept(proposal: Spline) -> bool:
        nonlocal checked
        checked += 1
        polynomials = [
            PPoly.from_spline((proposal.t, proposal.c[:, axis], 3)) for axis in (0, 1)
        ]
        spans: list[c.BezierSpan] = []
        for index, (left, right) in enumerate(pairwise(proposal.t.tolist())):
            if left == right:
                continue
            h = right - left
            axes: list[tuple[float, float, float, float]] = []
            for polynomial in polynomials:
                u, v, w, p = cast("list[float]", polynomial.c[:, index].tolist())
                axes.append(
                    (
                        p,
                        p + w * h / 3,
                        p + 2 * w * h / 3 + v * h**2 / 3,
                        p + w * h + v * h**2 + u * h**3,
                    )
                )
            controls = cast("c.Controls", tuple(zip(*axes, strict=True)))
            spans.append(c.BezierSpan(controls, False))
        assert len(spans) == 2
        return geometric_distance.within_distance(
            [(x, y) for x, y in cast("list[list[float]]", points.tolist())], spans, 0.05
        ) and all(curvature.within_curvature(span, 0.2, 0) for span in spans)

    candidate, diagnostics = refit(
        points,
        parameters,
        0.2,
        1,
        0.05,
        periodic=False,
        accept=accept,
        knot_breaks=breaks,
        corner_parameters=corners,
    )
    assert candidate is not None and checked > 0
    assert diagnostics["spans"] == 2
    assert candidate.t.tolist().count(0.5) == 3
    assert np.array_equal(candidate(0), cast("FloatArray", points[0]))
    assert np.array_equal(candidate(1), cast("FloatArray", points[-1]))
    assert np.allclose(
        candidate(0.5), cast("FloatArray", points[1]), atol=1e-12, rtol=0
    )
    assert np.allclose(candidate(nextafter(0.5, 0.0), nu=1), [2, 0], atol=1e-10)
    assert np.allclose(candidate(0.5, nu=1), [0, 2], atol=1e-10)
    for actual, expected in zip(
        (points, parameters, breaks, corners), snapshots, strict=True
    ):
        assert np.array_equal(actual, expected)


def test_c0_search_guides_both_incident_curvature_and_derivative_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.array([(0.0, 0.0), (1.0, 1.0), (2.0, 1.0)])
    parameters = np.array([0.0, 0.5, 1.0])
    breaks, corners = np.array([0.0, 0.5, 1.0]), np.array([0.5])
    left = nextafter(0.5, 0.0)
    probes: FloatArray = np.unique(
        np.concatenate((np.linspace(0, 0.5, 9), np.linspace(0.5, 1, 9), [left]))
    )
    probe_values = cast("list[float]", probes.tolist())
    left_index, right_index = probe_values.index(left), probe_values.index(0.5)
    calls = 0

    def forbidden(_proposal: Spline) -> NoReturn:
        raise AssertionError("left-side curvature violates sampled guidance")

    def solve(_objective: object, _initial: FloatArray, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        # The left endpoint has curvature4/3; the right piece is straight.
        poles = np.array(
            [(0, 0), (0.5, 0), (1, 0.5), (1, 1), (4 / 3, 1), (5 / 3, 1), (2, 1)]
        )
        selected = (poles[1:-1] - np.mean(points, axis=0)).ravel()
        constraint = cast("NonlinearConstraint", kwargs["constraints"])
        values = cast("Callable[[FloatArray], FloatArray]", constraint.fun)(selected)
        jacobian = cast("Callable[[FloatArray], FloatArray]", constraint.jac)(selected)
        assert values[left_index] == pytest.approx(fitting._SAMPLED_FRACTION - 4 / 3)
        assert values[right_index] == pytest.approx(fitting._SAMPLED_FRACTION)
        assert (
            values[-len(probes) + left_index] > 2 * values[-len(probes) + right_index]
        )
        assert jacobian.shape == (len(values), len(selected))
        cast("Callable[[FloatArray], None]", kwargs["callback"])(selected)
        return SimpleNamespace(
            x=selected, success=True, message="finished", nit=1, nfev=1, njev=1
        )

    monkeypatch.setattr(fitting, "minimize", solve)
    candidate, diagnostics = refit(
        points,
        parameters,
        1.0,
        1,
        2.0,
        periodic=False,
        accept=forbidden,
        knot_breaks=breaks,
        corner_parameters=corners,
    )
    assert candidate is None
    assert diagnostics["attempts"] == calls == 1


@pytest.mark.parametrize("periodic", [False, True])
def test_omitted_and_none_corner_parameters_produce_identical_default_coefficients(
    periodic: bool,
) -> None:
    points, parameters = (
        circle() if periodic else source(np.array([(1.1, 2.2), (1.7, 2.8), (2.3, 3.4)]))
    )
    default, _ = refit(points, parameters, 0.1, 1, 0.1, periodic=periodic, accept=allow)
    explicit, _ = refit(
        points,
        parameters,
        0.1,
        1,
        0.1,
        periodic=periodic,
        accept=allow,
        corner_parameters=None,
        c1_parameters=None,
        source_search_fraction=None,
    )
    assert default is not None and explicit is not None
    assert default.t.tobytes() == explicit.t.tobytes()
    assert default.c.tobytes() == explicit.c.tobytes()


@pytest.mark.parametrize("spans", [1, 32])
def test_explicit_clamped_basis_accepts_resource_boundaries_and_exact_pins(
    spans: int,
) -> None:
    points, parameters = source(np.array([(1.1, 2.2), (1.7, 2.8), (2.3, 3.4)]))
    breaks = np.linspace(0, 1, spans + 1)
    snapshots = points.copy(), parameters.copy(), breaks.copy()
    candidate, diagnostics = refit(
        points,
        parameters,
        0.1,
        1,
        0.1,
        periodic=False,
        accept=allow,
        knot_breaks=breaks,
    )
    assert candidate is not None
    assert diagnostics["spans"] == spans
    assert np.array_equal(
        candidate.t, np.concatenate((np.zeros(4), breaks[1:-1], np.ones(4)))
    )
    assert np.array_equal(candidate(0), cast("FloatArray", points[0]))
    assert np.array_equal(candidate(1), cast("FloatArray", points[-1]))
    for actual, expected in zip((points, parameters, breaks), snapshots, strict=True):
        assert np.array_equal(actual, expected)


def test_nonuniform_basis_probes_each_interval_and_scales_endpoint_speed_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points, parameters = source(np.array([(0.0, 0.0), (1.0, 0.0)]))
    breaks = np.array([0.0, 0.001, 0.002, 0.02, 0.1, 1.0])
    pins = np.array([(0.0, 0.0), (1.0, 1.0)])
    directions = np.array([(1.0, 1.0), (1.0, 1.0)])
    expected_knots = np.concatenate((np.zeros(4), breaks[1:-1], np.ones(4)))
    probes = np.unique(
        np.concatenate(
            [
                np.linspace(a, b, 9)
                for a, b in pairwise(cast("list[float]", breaks.tolist()))
            ]
        )
    )
    calls = 0
    checked = 0

    def reject(candidate: Spline) -> bool:
        nonlocal checked
        checked += 1
        assert np.array_equal(candidate.t, expected_knots)
        assert np.array_equal(candidate(0), cast("FloatArray", pins[0]))
        assert np.array_equal(candidate(1), cast("FloatArray", pins[-1]))
        return False

    def solve(_objective: object, _initial: FloatArray, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        constraints = cast("list[object]", kwargs["constraints"])
        assert isinstance(constraints[0], NonlinearConstraint)
        assert isinstance(constraints[2], LinearConstraint)
        # Greville coefficients reproduce(t,t) exactly. Its curve-to-source
        # distance is t, exposing the actual probe locations through guidance.
        greville = np.array(
            [
                np.mean(expected_knots[i + 1 : i + 4])
                for i in range(len(expected_knots) - 4)
            ]
        )
        selected = np.column_stack((greville[1:-1] - 0.5, greville[1:-1])).ravel()
        values = cast("Callable[[FloatArray], FloatArray]", constraints[0].fun)(
            selected
        )
        count = len(probes)
        assert np.allclose(
            values[count : 2 * count],
            1 - probes**2 / (2 * fitting._SOURCE_FRACTION) ** 2,
            atol=1e-14,
            rtol=0,
        )
        tangent = np.array([1.0, 1.0]) / sqrt(2)
        centered = pins - np.mean(points, axis=0)
        fixed = np.array([tangent @ centered[0], -tangent @ centered[1]])
        assert np.allclose(
            constraints[2].lb - fixed,
            1e-5 * np.diff(breaks)[[0, -1]] / 3,
            atol=1e-16,
            rtol=1e-8,
        )
        return SimpleNamespace(
            x=selected, success=True, message="finished", nit=1, nfev=1, njev=1
        )

    monkeypatch.setattr(fitting, "minimize", solve)
    candidate, diagnostics = refit(
        points,
        parameters,
        0.1,
        1,
        2.0,
        periodic=False,
        accept=reject,
        endpoint_positions=pins,
        tangent_directions=directions,
        knot_breaks=breaks,
    )
    assert candidate is None
    assert calls == checked == diagnostics["attempts"] == 1


def test_nonuniform_adaptive_retry_keeps_its_single_basis_and_warm_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points, parameters = source(np.array([(0.0, 0.0), (1.0, 0.0)]))
    breaks = np.array([0.0, 0.1, 1.0])
    calls = 0
    checked = 0
    first: FloatArray | None = None

    def accept(candidate: Spline) -> bool:
        nonlocal checked
        checked += 1
        assert np.array_equal(candidate.t, np.array([0, 0, 0, 0, 0.1, 1, 1, 1, 1]))
        return checked == 2

    def solve(_objective: object, initial: FloatArray, **kwargs: object) -> NoReturn:
        nonlocal calls, first
        calls += 1
        if first is None:
            first = initial.copy()
        else:
            assert np.array_equal(initial, first)
        cast("Callable[[FloatArray], None]", kwargs["callback"])(initial)
        raise AssertionError("adaptive refinement or strict acceptance must stop this")

    monkeypatch.setattr(fitting, "minimize", solve)
    candidate, diagnostics = refit(
        points,
        parameters,
        0.1,
        1,
        0.2,
        periodic=False,
        accept=accept,
        refine=lambda _proposal: Refinement((0.02,), (0.003,)),
        knot_breaks=breaks,
    )
    assert candidate is not None
    assert calls == checked == 2
    assert diagnostics["adaptive_retries"] == 1
    assert diagnostics["spans"] == 2


def test_local_endpoint_positions_remain_separate_from_original_source() -> None:
    points, parameters = source(np.array([(1.1, 2.2), (1.7, 2.8), (2.3, 3.4)]))
    pins: FloatArray = np.array([(1.12, 2.18), (2.32, 3.38)])
    directions: FloatArray = np.array([(2.0, 2.0), (7.0, 7.0)])
    source_before, parameters_before = points.copy(), parameters.copy()
    pins_before, directions_before = pins.copy(), directions.copy()
    candidate, _ = refit(
        points,
        parameters,
        0.1,
        1,
        0.1,
        periodic=False,
        accept=allow,
        endpoint_positions=pins,
        tangent_directions=directions,
    )
    assert candidate is not None
    assert np.array_equal(candidate(0), cast("FloatArray", pins[0]))
    assert np.array_equal(candidate(1), cast("FloatArray", pins[1]))
    for index, endpoint in enumerate((0.0, 1.0)):
        direction = cast("FloatArray", directions[index])
        velocity = candidate(endpoint, nu=1)
        determinant = cast("np.float64", np.linalg.det(np.stack((velocity, direction))))
        assert abs(determinant) < 1e-10
        assert velocity @ direction > 0
    assert np.array_equal(points, source_before)
    assert np.array_equal(parameters, parameters_before)
    assert np.array_equal(pins, pins_before)
    assert np.array_equal(directions, directions_before)


@pytest.mark.parametrize("reverse", [False, True])
def test_endpoint_tangent_feasibility_precedes_acceptance(
    monkeypatch: pytest.MonkeyPatch, reverse: bool
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))
    called = 0

    def accept(_candidate: Spline) -> bool:
        nonlocal called
        called += 1
        return True

    def solve(_objective: object, initial: FloatArray, **kwargs: object) -> object:
        constraints = cast("list[object]", kwargs["constraints"])
        assert isinstance(constraints[0], NonlinearConstraint)
        assert all(isinstance(value, LinearConstraint) for value in constraints[1:])
        selected = initial.copy()
        if reverse:
            selected[:2] = [-0.6, 0.0]  # Centered at0.5: handle lies behind start0.
        else:
            selected[1] = 0.01  # Forward, but off the inherited tangent line.
        physical = cast("Callable[[FloatArray], FloatArray]", constraints[0].fun)
        assert np.min(physical(selected)) >= -1e-8
        cast("Callable[[FloatArray], None]", kwargs["callback"])(selected)
        return SimpleNamespace(
            x=selected, success=True, message="finished", nit=1, nfev=1, njev=1
        )

    monkeypatch.setattr(fitting, "minimize", solve)
    points, parameters = source(np.array([(0.0, 0.0), (1.0, 0.0)]))
    candidate, _ = refit(
        points,
        parameters,
        0.1,
        1,
        1.0,
        periodic=False,
        accept=accept,
        tangent_directions=np.array([(1.0, 0.0), (1.0, 0.0)]),
    )
    assert candidate is None
    assert called == 0


@pytest.mark.parametrize(
    "pins,tangents", [(np.ones((2, 2)), None), (None, np.ones((2, 2)))]
)
def test_periodic_search_cannot_have_local_endpoint_pins(
    pins: FloatArray | None, tangents: FloatArray | None
) -> None:
    points, parameters = circle()
    with pytest.raises(ValueError, match="open refit"):
        _ = refit(
            points,
            parameters,
            0.1,
            1,
            0.04,
            periodic=True,
            accept=allow,
            endpoint_positions=pins,
            tangent_directions=tangents,
        )


@pytest.mark.parametrize(
    "pins,tangents",
    [
        (np.zeros((1, 2)), None),
        (np.array([(0, 0), (1, np.nan)]), None),
        (None, np.ones((2, 3))),
        (None, np.array([(0, 0), (1, 1)])),
        (None, np.array([(1, 0), (np.inf, 1)])),
    ],
)
def test_invalid_local_endpoint_data_is_reported(
    pins: FloatArray | None, tangents: FloatArray | None
) -> None:
    points, parameters = source(np.array([(0.0, 0.0), (1.0, 0.0)]))
    with pytest.raises(ValueError, match="endpoint"):
        _ = refit(
            points,
            parameters,
            0.1,
            1,
            0.1,
            periodic=False,
            accept=allow,
            endpoint_positions=pins,
            tangent_directions=tangents,
        )


@pytest.mark.parametrize("budget", [-1.0, float("nan"), float("inf"), True])
def test_invalid_remaining_time_budget_is_reported(budget: float) -> None:
    points, parameters = circle()
    with pytest.raises(ValueError, match="time budget"):
        _ = refit(
            points,
            parameters,
            0.1,
            1,
            0.1,
            periodic=True,
            accept=allow,
            time_budget_seconds=budget,
        )


def test_exhausted_remaining_time_does_not_start_another_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("exhausted local budget must not start a solver")

    monkeypatch.setattr(fitting, "minimize", unexpected)
    points, parameters = circle()
    candidate, diagnostics = refit(
        points,
        parameters,
        0.1,
        1,
        0.1,
        periodic=True,
        accept=allow,
        time_budget_seconds=0.0,
    )
    assert candidate is None
    assert diagnostics["status"] == "time-budget"


@pytest.mark.parametrize("default,remaining", [(12.0, 0.25), (0.25, 12.0)])
def test_remaining_budget_is_respected_and_cannot_extend_the_default(
    monkeypatch: pytest.MonkeyPatch, default: float, remaining: float
) -> None:
    now = 0.0
    calls = 0
    monkeypatch.setattr(fitting, "_SECONDS", default)
    monkeypatch.setattr(fitting, "monotonic", lambda: now)

    def solve(_objective: object, initial: FloatArray, **kwargs: object) -> NoReturn:
        nonlocal now, calls
        calls += 1
        now = 0.5
        cast("Callable[[FloatArray], None]", kwargs["callback"])(initial)
        raise AssertionError("the smaller default/remaining budget must stop this")

    monkeypatch.setattr(fitting, "minimize", solve)
    points, parameters = circle()
    candidate, diagnostics = refit(
        points,
        parameters,
        0.1,
        1,
        0.04,
        periodic=True,
        accept=allow,
        time_budget_seconds=remaining,
    )
    assert candidate is None and calls == 1
    assert diagnostics["status"] == "time-budget"


def test_local_endpoint_constraints_do_not_authorize_a_rejected_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))
    called = 0

    def reject(_candidate: Spline) -> bool:
        nonlocal called
        called += 1
        return False

    points, parameters = source(np.array([(0.0, 0.0), (1.0, 0.0)]))
    candidate, _ = refit(
        points,
        parameters,
        0.1,
        1,
        0.1,
        periodic=False,
        accept=reject,
        endpoint_positions=np.array([(0.0, 0.02), (1.0, 0.02)]),
        tangent_directions=np.array([(1.0, 0.0), (1.0, 0.0)]),
    )
    assert candidate is None and called > 0


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


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("inward_sign", [-1, 1])
@pytest.mark.parametrize("clamped_speed", [False, True])
def test_analytic_constraint_jacobian_matches_finite_differences(
    monkeypatch: pytest.MonkeyPatch,
    periodic: bool,
    inward_sign: int,
    clamped_speed: bool,
) -> None:
    spans = 4
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (spans,))
    points, parameters = circle()
    if clamped_speed:
        # Zero centered endpoints isolate the denominator's constant branch
        # without subtracting almost equal unit-sized control coordinates.
        points, parameters = source(
            np.array(
                [(0, 0), (1, 1), (-1, 1), (0, 0), (1, -1), (-1, -1), (0, 0)],
                dtype=np.float64,
            )
        )

    class Captured(Exception):
        pass

    def probe(_objective: object, initial: FloatArray, **kwargs: object) -> NoReturn:
        constraint = cast("NonlinearConstraint", kwargs["constraints"])
        fun = cast("Callable[[FloatArray], FloatArray]", constraint.fun)
        jac = cast("Callable[[FloatArray], FloatArray]", constraint.jac)
        if clamped_speed:
            # Equal poles produce zero derivatives, including the pinned open
            # endpoints (the source is closed). Tiny perturbations stay inside
            # the speed-denominator clamp while exercising a nonzero numerator.
            assert np.array_equal(np.mean(points, axis=0), np.zeros(2))
            selected = 1e-10 * np.sin(np.arange(len(initial)))
            step = 1e-13
        else:
            selected = initial + 0.01 * np.sin(np.arange(len(initial)))
            step = 1e-6
        expected = jac(selected)
        numerical = np.column_stack(
            [
                (fun(selected + delta * step) - fun(selected - delta * step))
                / (2 * step)
                for delta in np.eye(len(selected))
            ]
        )
        assert expected.shape == (len(fun(selected)), len(selected))
        assert np.all(np.isfinite(expected))
        if clamped_speed:
            # The collapsed proxy has near-tied closest edges; normal finite
            # differences switch branches, while tiny differences lose digits
            # in unit-sized distance constants. Isolate the speed clamp here;
            # ordinary cases above test both geometric-distance blocks.
            count = len(
                np.unique(
                    np.concatenate((np.linspace(0, 1, spans * 8 + 1), parameters))
                )
            )
            rows = np.concatenate(
                (np.arange(count), np.arange(len(expected) - count, len(expected)))
            )
            expected, numerical = expected[rows], numerical[rows]
        else:
            objective = cast("Callable[[FloatArray], float]", _objective)
            gradient = cast("Callable[[FloatArray], FloatArray]", kwargs["jac"])
            numeric_gradient = np.array(
                [
                    (
                        objective(selected + delta * step)
                        - objective(selected - delta * step)
                    )
                    / (2 * step)
                    for delta in np.eye(len(selected))
                ]
            )
            assert np.allclose(
                gradient(selected), numeric_gradient, rtol=3e-5, atol=3e-5
            )
        # Check every row independently: the clamped curvature block can be
        # enormous and must not hide an incorrect distance or speed row.
        error = np.linalg.norm(expected - numerical, axis=1)
        scale = np.maximum(1.0, np.linalg.norm(numerical, axis=1))
        assert np.all(error / scale < 3e-5)
        raise Captured

    monkeypatch.setattr(fitting, "minimize", probe)
    with pytest.raises(Captured):
        _ = refit(
            points, parameters, 0.1, inward_sign, 0.04, periodic=periodic, accept=allow
        )


def test_geometric_search_allows_different_progress_along_the_same_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))

    class Captured(Exception):
        pass

    def probe(objective: object, initial: FloatArray, **kwargs: object) -> NoReturn:
        selected = np.zeros_like(initial).reshape(-1, 2)
        selected[:, 0] = np.array([0.01, 0.04, 0.16, 0.64, 2.56]) - 2
        flattened = selected.ravel()
        fun = cast("Callable[[FloatArray], float]", objective)
        constraint = cast("NonlinearConstraint", kwargs["constraints"])
        bounds = cast("Callable[[FloatArray], FloatArray]", constraint.fun)(flattened)
        # The full segment is unchanged although parameter progress is strongly
        # nonlinear. Fixed parameter displacement caps would reject this curve.
        assert fun(flattened) < 1e-25
        assert np.all(bounds >= 0)
        raise Captured

    monkeypatch.setattr(fitting, "minimize", probe)
    points, parameters = source(np.array([(0, 0), (4, 0)], dtype=np.float64))
    with pytest.raises(Captured):
        _ = refit(points, parameters, 0.1, 1, 0.04, periodic=False, accept=allow)


def test_certified_early_candidate_stops_without_claiming_optimizer_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stop_at_callback(
        _objective: object, initial: FloatArray, **kwargs: object
    ) -> NoReturn:
        callback = cast("Callable[[FloatArray], None]", kwargs["callback"])
        callback(initial)
        raise AssertionError("certified candidate should have interrupted the solve")

    monkeypatch.setattr(fitting, "minimize", stop_at_callback)
    points, parameters = circle()
    candidate, diagnostics = refit(
        points, parameters, 0.1, 1, 0.04, periodic=True, accept=allow
    )
    assert candidate is not None
    assert diagnostics["certified_early"] is True
    trials = cast("list[dict[str, object]]", diagnostics["trials"])
    assert trials[0]["optimizer_success"] is None
    assert trials[0]["early_checks"] == 1


def test_geometric_search_preserves_closed_boundary_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))
    points, parameters = circle()
    probes = np.unique(np.concatenate((np.linspace(0, 1, 33), parameters)))

    class Captured(Exception):
        pass

    def probe(_objective: object, initial: FloatArray, **kwargs: object) -> NoReturn:
        constraint = cast("NonlinearConstraint", kwargs["constraints"])
        values = cast("Callable[[FloatArray], FloatArray]", constraint.fun)
        reverse = initial.reshape(-1, 2)[::-1].ravel()
        normal, reversed_values = values(initial), values(reverse)
        winding_row = -len(probes) - 1
        assert normal[winding_row] > 0
        assert reversed_values[winding_row] < 0
        raise Captured

    monkeypatch.setattr(fitting, "minimize", probe)
    with pytest.raises(Captured):
        _ = refit(points, parameters, 0.1, 1, 0.04, periodic=True, accept=allow)


def test_nonconverged_output_can_only_be_accepted_by_continuous_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))

    def incomplete(
        _objective: object, initial: FloatArray, **_kwargs: object
    ) -> object:
        return SimpleNamespace(
            x=initial, success=False, message="iteration limit", nit=1, nfev=1, njev=1
        )

    monkeypatch.setattr(fitting, "minimize", incomplete)
    points, parameters = circle()
    candidate, diagnostics = refit(
        points, parameters, 0.1, 1, 0.04, periodic=True, accept=allow
    )
    assert candidate is not None
    trials = cast("list[dict[str, object]]", diagnostics["trials"])
    assert trials[0]["optimizer_success"] is False
    candidate, diagnostics = refit(
        points, parameters, 0.1, 1, 0.04, periodic=True, accept=lambda _spline: False
    )
    assert candidate is None
    assert diagnostics["status"] == "unrefined"


def test_early_checks_are_spaced_and_limited_when_continuous_gate_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))
    iteration = 0
    observed: list[int] = []

    def reject(_candidate: Spline) -> bool:
        observed.append(iteration)
        return False

    def incomplete(_objective: object, initial: FloatArray, **kwargs: object) -> object:
        nonlocal iteration
        callback = cast("Callable[[FloatArray], None]", kwargs["callback"])
        for value in range(1, 31):
            iteration = value
            callback(initial)
        return SimpleNamespace(
            x=initial, success=False, message="iteration limit", nit=30, nfev=1, njev=1
        )

    monkeypatch.setattr(fitting, "minimize", incomplete)
    points, parameters = circle()
    candidate, _ = refit(points, parameters, 0.1, 1, 0.04, periodic=True, accept=reject)
    assert candidate is None
    # The least-squares seed is feasible; the tiny round-loop seed is not.
    assert observed == [1, 9, 17, 30]  # Three early checks plus the final result.


@pytest.mark.parametrize("value", [np.nan, 1e200])
def test_invalid_optimizer_geometry_retains_source_without_index_exception(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))

    def invalid(objective: object, initial: FloatArray, **kwargs: object) -> object:
        selected = np.full_like(initial, value)
        fun = cast("Callable[[FloatArray], float]", objective)
        constraint = cast("NonlinearConstraint", kwargs["constraints"])
        bounds = cast("Callable[[FloatArray], FloatArray]", constraint.fun)
        callback = cast("Callable[[FloatArray], None]", kwargs["callback"])
        with np.errstate(over="ignore", invalid="ignore"):
            assert np.isnan(fun(selected))
            assert np.any(np.isnan(bounds(selected)))
            callback(selected)
        return SimpleNamespace(
            x=selected, success=False, message="invalid step", nit=1, nfev=1, njev=1
        )

    def unexpected(_candidate: Spline) -> NoReturn:
        raise AssertionError("invalid geometry cannot reach acceptance")

    monkeypatch.setattr(fitting, "minimize", invalid)
    points, parameters = circle()
    with np.errstate(over="ignore", invalid="ignore"):
        candidate, diagnostics = refit(
            points, parameters, 0.1, 1, 0.04, periodic=True, accept=unexpected
        )
    assert candidate is None
    assert diagnostics["status"] == "unrefined"


def test_adaptive_restart_keeps_basis_warm_start_and_rebuilds_both_sample_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))
    calls = 0
    checked = 0
    first: FloatArray | None = None
    rows: list[int] = []

    def accept(_proposal: Spline) -> bool:
        nonlocal checked
        checked += 1
        return checked == 2

    def solve(_objective: object, initial: FloatArray, **kwargs: object) -> NoReturn:
        nonlocal calls, first
        calls += 1
        constraint = cast("NonlinearConstraint", kwargs["constraints"])
        values = cast("Callable[[FloatArray], FloatArray]", constraint.fun)
        jacobian = cast("Callable[[FloatArray], FloatArray]", constraint.jac)
        callback = cast("Callable[[FloatArray], None]", kwargs["callback"])
        if first is None:
            first = initial + 0.001
            selected = first
        else:
            assert np.array_equal(initial, first)
            selected = initial
        rows.append(len(values(selected)))
        assert jacobian(selected).shape == (rows[-1], len(selected))
        callback(selected)
        raise AssertionError("strict callback or new constraints should stop the solve")

    monkeypatch.setattr(fitting, "minimize", solve)
    points, parameters = circle()
    candidate, diagnostics = refit(
        points,
        parameters,
        0.1,
        1,
        0.04,
        periodic=True,
        accept=accept,
        refine=lambda _proposal: Refinement((0.123,), (0.234,)),
    )
    assert candidate is not None
    assert calls == checked == 2
    assert rows[1] == rows[0] + 4  # Curvature/forward/speed plus reverse query.
    assert diagnostics["adaptive_retries"] == 1
    assert diagnostics["spans"] == 4


def test_duplicate_feedback_does_not_restart_a_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))
    calls = 0

    def solve(_objective: object, initial: FloatArray, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        callback = cast("Callable[[FloatArray], None]", kwargs["callback"])
        callback(initial)
        return SimpleNamespace(
            x=initial, success=False, message="finished", nit=1, nfev=1, njev=1
        )

    monkeypatch.setattr(fitting, "minimize", solve)
    points, parameters = source(np.array([(0, 0), (1, 0)], dtype=np.float64))
    candidate, diagnostics = refit(
        points,
        parameters,
        0.1,
        1,
        0.04,
        periodic=False,
        accept=lambda _proposal: False,
        refine=lambda _proposal: Refinement((0.0, 0.25, 1.0), (0.0, 1.0)),
    )
    assert candidate is None
    assert calls == 1
    assert diagnostics["adaptive_retries"] == 0


def test_adaptive_retry_limit_retains_every_added_location_without_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))
    calls = 0
    feedbacks = 0
    row_counts: list[int] = []

    def solve(_objective: object, initial: FloatArray, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        constraint = cast("NonlinearConstraint", kwargs["constraints"])
        values = cast("Callable[[FloatArray], FloatArray]", constraint.fun)
        row_counts.append(len(values(initial)))
        cast("Callable[[FloatArray], None]", kwargs["callback"])(initial)
        return SimpleNamespace(
            x=initial, success=False, message="finished", nit=1, nfev=1, njev=1
        )

    def feedback(_proposal: Spline) -> Refinement:
        nonlocal feedbacks
        feedbacks += 1
        return Refinement((0.111 + 0.01 * feedbacks,))

    monkeypatch.setattr(fitting, "minimize", solve)
    points, parameters = source(np.array([(0, 0), (1, 0)], dtype=np.float64))
    candidate, diagnostics = refit(
        points,
        parameters,
        0.1,
        1,
        0.04,
        periodic=False,
        accept=lambda _proposal: False,
        refine=feedback,
    )
    assert candidate is None  # Feedback cannot authorize a candidate.
    assert calls == 4 and feedbacks == 3
    assert row_counts == [row_counts[0] + 3 * i for i in range(4)]
    assert diagnostics["adaptive_retries"] == 3


def test_adaptive_retries_share_the_original_time_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))
    monkeypatch.setattr(fitting, "_SECONDS", 1.0)
    now = 0.0
    monkeypatch.setattr(fitting, "monotonic", lambda: now)
    calls = 0

    def solve(_objective: object, initial: FloatArray, **kwargs: object) -> NoReturn:
        nonlocal calls
        calls += 1
        cast("Callable[[FloatArray], None]", kwargs["callback"])(initial)
        raise AssertionError("new constraint should interrupt")

    def feedback(_proposal: Spline) -> Refinement:
        nonlocal now
        now = 2.0
        return Refinement((0.123,))

    monkeypatch.setattr(fitting, "minimize", solve)
    points, parameters = source(np.array([(0, 0), (1, 0)], dtype=np.float64))
    candidate, diagnostics = refit(
        points,
        parameters,
        0.1,
        1,
        0.04,
        periodic=False,
        accept=lambda _proposal: False,
        refine=feedback,
    )
    assert candidate is None
    assert calls == 1
    assert diagnostics["status"] == "time-budget"


@pytest.mark.parametrize("parameter", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_invalid_feedback_parameters_are_reported(
    monkeypatch: pytest.MonkeyPatch, parameter: float
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))

    def solve(_objective: object, initial: FloatArray, **kwargs: object) -> NoReturn:
        cast("Callable[[FloatArray], None]", kwargs["callback"])(initial)
        raise AssertionError("invalid feedback must fail")

    monkeypatch.setattr(fitting, "minimize", solve)
    points, parameters = source(np.array([(0, 0), (1, 0)], dtype=np.float64))
    with pytest.raises(ValueError, match="finite parameters"):
        _ = refit(
            points,
            parameters,
            0.1,
            1,
            0.04,
            periodic=False,
            accept=lambda _proposal: False,
            refine=lambda _proposal: Refinement((parameter,)),
        )


def test_rejected_final_output_can_refine_without_an_intermediate_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))
    calls = checks = 0
    warm: FloatArray | None = None

    def solve(_objective: object, initial: FloatArray, **_kwargs: object) -> object:
        nonlocal calls, warm
        calls += 1
        if warm is None:
            warm = initial + 0.001
        else:
            assert np.array_equal(initial, warm)
        return SimpleNamespace(
            x=warm, success=False, message="iteration limit", nit=1, nfev=1, njev=1
        )

    def accept(_proposal: Spline) -> bool:
        nonlocal checks
        checks += 1
        return checks == 2

    monkeypatch.setattr(fitting, "minimize", solve)
    points, parameters = source(np.array([(0, 0), (1, 0)], dtype=np.float64))
    candidate, diagnostics = refit(
        points,
        parameters,
        0.1,
        1,
        0.04,
        periodic=False,
        accept=accept,
        refine=lambda _proposal: Refinement((0.123,)),
    )
    assert candidate is not None
    assert calls == checks == 2
    assert diagnostics["adaptive_retries"] == 1


def test_feedback_errors_are_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fitting, "_SPAN_COUNTS", (4,))

    def solve(_objective: object, initial: FloatArray, **kwargs: object) -> NoReturn:
        cast("Callable[[FloatArray], None]", kwargs["callback"])(initial)
        raise AssertionError("feedback exception should propagate")

    def feedback(_proposal: Spline) -> Refinement:
        raise RuntimeError("feedback failed unexpectedly")

    monkeypatch.setattr(fitting, "minimize", solve)
    points, parameters = source(np.array([(0, 0), (1, 0)], dtype=np.float64))
    with pytest.raises(RuntimeError, match="failed unexpectedly"):
        _ = refit(
            points,
            parameters,
            0.1,
            1,
            0.04,
            periodic=False,
            accept=lambda _proposal: False,
            refine=feedback,
        )


def test_new_constraint_locations_are_deduplicated_and_bounded() -> None:
    existing = np.array([0.0, 0.5, 1.0])
    requested = (0.0, 0.5 + 1e-16, *np.linspace(0.01, 0.49, 100))
    added = fitting._additional_parameters(requested, existing)
    assert len(added) == fitting._MAX_NEW_PARAMETERS
    assert len(np.unique(added)) == len(added)
    assert np.all((added > 0) & (added < 0.5))
