"""Physical fairing and hard forward constraints on small synthetic outlines."""

from itertools import pairwise
from math import atan2, degrees, hypot
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.sparse import csr_array
from shapely.affinity import scale, translate
from shapely.geometry import Point, Polygon, box
from shapely.geometry.polygon import orient

from panelchemy.approximation import boundary_within_tolerance
from panelchemy.contours import PanelContours
from panelchemy.curves import BezierSpan, CurveRing
from panelchemy.outline_fitting import fit_outlines
from panelchemy.spline_solver import FloatSparse, solve_coefficients

type FloatArray = NDArray[np.float64]


def _source(polygon: Polygon) -> PanelContours:
    silhouette = box(0, 0, 20, 20)
    polygon = orient(polygon, sign=1)
    return PanelContours(
        silhouette,
        [],
        [polygon],
        0.1,
        (0, 0, 200, 200),
        4 + len(polygon.exterior.coords) - 1,
    )


def _solve(
    source_parameters: FloatArray,
    source_points: FloatArray,
    knots: FloatArray,
    *,
    fairing_mm: float,
    length_mm: float,
    pin_ends: bool = False,
) -> BSpline[np.float64]:
    """Supply densely sampled data; test the solver's public geometric output."""
    count = len(knots) - 4
    times = np.linspace(0, 1, 401)
    points = np.column_stack(
        [np.interp(times, source_parameters, source_points[:, axis]) for axis in (0, 1)]
    )
    offsets = np.zeros((count, 2))
    if pin_ends:
        free = count - 2
        transform: FloatSparse = csr_array(
            (np.ones(free), (np.arange(1, count - 1), np.arange(free))),
            shape=(count, free),
        )
        offsets[0], offsets[-1] = source_points[0], source_points[-1]
    else:
        transform = csr_array(np.eye(count))
    weights = np.full(len(times), cast("float", (1 / (len(times) - 1)) ** 0.5))
    weights[0] /= 2**0.5
    weights[-1] /= 2**0.5
    basis: FloatSparse = BSpline.design_matrix(times, knots, 3)
    matrix = (basis @ transform).multiply(weights[:, None]).tocsr()
    target = (points - basis @ offsets) * weights[:, None]
    coefficients = solve_coefficients(
        matrix,
        target,
        transform,
        offsets,
        knots,
        source_parameters,
        source_points,
        length_mm,
        fairing_mm,
    )
    return BSpline(knots, offsets + transform @ coefficients, 3)


def _bend_energy(spline: BSpline[np.float64], length_mm: float) -> float:
    times = np.linspace(0, 1, 1001)
    acceleration = spline(times, nu=2) / length_mm**2
    return float(np.trapezoid(np.sum(acceleration * acceleration, axis=1), times))


def test_physical_fairing_removes_unnecessary_bends_from_a_stepped_edge() -> None:
    parameters = np.linspace(0, 1, 41)
    points = np.column_stack(
        (8 * parameters, np.where(np.arange(len(parameters)) % 2, -0.06, 0.06))
    )
    length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    knots = np.concatenate((np.zeros(4), np.linspace(0, 1, 81)[1:-1], np.ones(4)))
    raw = _solve(parameters, points, knots, fairing_mm=0, length_mm=length)
    fair = _solve(parameters, points, knots, fairing_mm=0.25, length_mm=length)
    assert _bend_energy(raw, length) > 0
    assert _bend_energy(fair, length) < _bend_energy(raw, length) / 2
    samples = fair(np.linspace(0, 1, 1001))
    sampled_y = samples[:, 1]
    assert float(cast("np.float64", np.max(np.abs(sampled_y)))) < 0.06
    assert float(np.min(fair(np.linspace(0, 1, 1001), nu=1)[:, 0])) > 0
    expected_y = np.interp(np.linspace(0, 1, 1001), parameters, points[:, 1])
    assert float(cast("np.float64", np.max(np.abs(sampled_y - expected_y)))) < 0.2


def test_zero_fairing_still_prevents_backward_motion_on_a_synthetic_notch() -> None:
    knots = np.array([0.0] * 4 + [1.0] * 4)
    reference = BSpline(
        knots,
        np.array([(0, 0), (1.2, 0.05 / 3), (-0.2, 0.1 / 3), (1, 0.05)]),
        3,
    )
    raw_parameters = np.linspace(0, 1, 101)
    points = reference(raw_parameters)
    distances = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    length = float(cast("np.float64", distances[-1]))
    parameters = distances / length
    direction = cast("FloatArray", points[-1]) - cast("FloatArray", points[0])
    direction /= np.linalg.norm(direction)
    derivative = reference(raw_parameters, nu=1)
    assert float(np.min(derivative @ direction)) < -0.1
    selected = _solve(
        parameters, points, knots, fairing_mm=0, length_mm=length, pin_ends=True
    )
    assert selected(0) == pytest.approx(cast("FloatArray", points[0]), abs=1e-10)
    assert selected(1) == pytest.approx(cast("FloatArray", points[-1]), abs=1e-10)
    selected_derivative = selected(np.linspace(0, 1, 1001), nu=1)
    assert float(np.min(selected_derivative @ direction)) >= -1e-8
    assert np.all(np.isfinite(selected.c))


def _join_angles(ring: CurveRing) -> list[tuple[tuple[float, float], float]]:
    def direction(span: BezierSpan, end: bool) -> tuple[float, float]:
        a, b, c, d = span.control_points
        if span.is_line:
            return d[0] - a[0], d[1] - a[1]
        return (d[0] - c[0], d[1] - c[1]) if end else (b[0] - a[0], b[1] - a[1])

    joins: list[tuple[tuple[float, float], float]] = []
    for previous, following in pairwise((*ring.spans, ring.spans[0])):
        assert previous.control_points[-1] == following.control_points[0]
        a, b = direction(previous, True), direction(following, False)
        assert hypot(*a) > 0 and hypot(*b) > 0
        angle = degrees(
            atan2(abs(a[0] * b[1] - a[1] * b[0]), a[0] * b[0] + a[1] * b[1])
        )
        joins.append((previous.control_points[-1], angle))
    return joins


def test_fairing_keeps_a_deliberate_sharp_point() -> None:
    tip = (17.0, 10.0)
    original = _source(Polygon([(3, 3), tip, (3, 17)]))
    result = fit_outlines(original, 0.2, corner_scale_mm=1, fairing_mm=0.5)
    point, angle = min(
        _join_angles(result.contours.white[0].exterior),
        key=lambda join: hypot(join[0][0] - tip[0], join[0][1] - tip[1]),
    )
    assert hypot(point[0] - tip[0], point[1] - tip[1]) <= 0.2
    assert angle > 60
    assert boundary_within_tolerance(
        original.white[0].boundary,
        result.proxy.white[0].boundary,
        0.2 + result.proxy_deviation_mm,
    )


@pytest.mark.parametrize("factor,shift", [(1.0, (123.375, -45.625)), (2.0, (0.0, 0.0))])
def test_fairing_is_consistent_under_translation_and_physical_scaling(
    factor: float, shift: tuple[float, float]
) -> None:
    original = _source(Point(10, 10).buffer(4, quad_segs=12))
    first = fit_outlines(original, 0.2, corner_scale_mm=1, fairing_mm=0.25)

    def moved(polygon: Polygon) -> Polygon:
        return translate(scale(polygon, factor, factor, origin=(0, 0)), *shift)

    transformed = PanelContours(
        moved(original.silhouette),
        [],
        [moved(original.white[0])],
        original.mm_per_pixel * factor,
        original.source_bounds_px,
        original.vertex_count,
    )
    second = fit_outlines(
        transformed, 0.2 * factor, corner_scale_mm=factor, fairing_mm=0.25 * factor
    )
    restored = scale(
        translate(second.proxy.white[0], -shift[0], -shift[1]),
        1 / factor,
        1 / factor,
        origin=(0, 0),
    )
    assert boundary_within_tolerance(
        first.proxy.white[0].boundary, restored.boundary, 1e-5
    )
    assert all(
        angle < 1e-4 for _, angle in _join_angles(second.contours.white[0].exterior)
    )


@pytest.mark.parametrize(
    "value", [-1.0, float("inf"), float("nan"), True, None, "0.25"]
)
def test_invalid_fairing_scale_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="fairing scale"):
        _ = fit_outlines(
            _source(box(3, 3, 17, 17)), 0.2, fairing_mm=cast("float", value)
        )
