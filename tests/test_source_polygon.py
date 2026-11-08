"""Exact low-complexity source loops are distinct from raster stair contours."""

# pyright: reportPrivateUsage=false

from itertools import pairwise
from math import radians, tan

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from panelchemy import boundary_tapers, curved_occt, occt
from panelchemy import curves as c
from panelchemy import outline_fitting as fitting
from panelchemy.export import _step

LIMIT = boundary_tapers.BoundaryTaper(2, 0, 0.35, -1)


def fit_for(points: tuple[c.Point2, ...], *, free_loop: bool = False) -> fitting._Fit:
    parameters = np.asarray(c._parameters(points), dtype=np.float64)
    return fitting._Fit(
        c._Chain(points, False, free_loop),
        parameters,
        [],
        [],
        [
            c._Piece(0, i, i + 1, c._line(a, b))
            for i, (a, b) in enumerate(pairwise(points))
        ],
        list(
            pairwise(float(fitting._scalar(parameters, i)) for i in range(len(points)))
        ),
        source_corners=(0.123,),
    )


@pytest.mark.parametrize(
    "vertices",
    [
        ((0.0, 0.0), (1.0, 0.0), (0.0, 2.0)),
        ((0.0, 0.0), (0.18, 0.0), (0.18, 0.88), (0.0, 0.88)),
        ((0.0, 0.0), (2.0, 0.5), (1.5, 2.0), (-0.5, 1.0)),
    ],
)
@pytest.mark.parametrize("free_loop", [False, True])
def test_exact_convex_source_primitive_retains_corners_and_source_provenance(
    vertices: tuple[c.Point2, ...],
    free_loop: bool,
) -> None:
    fit = fit_for((*vertices, vertices[0]), free_loop=free_loop)
    result, report = fitting._polygon_curvature_candidate(fit, 0, LIMIT, 0.2)
    assert result is not None and report["source_displacement_mm"] == 0
    assert all(p.span.is_line for p in result.pieces)
    assert len(result.pieces) == len(vertices)
    assert LineString(
        [p.span.control_points[0] for p in result.pieces]
        + [result.pieces[-1].span.control_points[-1]]
    ).equals(LineString(fit.chain.points))
    assert len(result.corners) == len(vertices) and 0 in result.corners
    assert result.source_corners is fit.source_corners
    assert not fitting._source_failures(result, 0.2)


def test_collinear_subdivisions_collapse_without_losing_a_mid_edge_graph_pin() -> None:
    points = (
        (0.0, 0.5),
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (2.0, 0.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.25),
        (0.0, 0.5),
    )
    fit = fit_for(points, free_loop=True)
    result, _ = fitting._polygon_curvature_candidate(fit, 0, LIMIT, 0.2)
    assert result is not None and len(result.pieces) == 5
    assert len(result.corners) == 4 and 0 not in result.corners
    assert result.pieces[0].span.control_points[0] == points[0]
    assert result.pieces[-1].span.control_points[-1] == points[0]
    before, after = (
        c._controls(result.pieces[-1].span),
        c._controls(result.pieces[0].span),
    )
    u = (before[-1][0] - before[-2][0], before[-1][1] - before[-2][1])
    v = (after[1][0] - after[0][0], after[1][1] - after[0][1])
    assert u[0] * v[1] == u[1] * v[0] and u[0] * v[0] + u[1] * v[1] > 0


@pytest.mark.parametrize(
    "points",
    [
        ((0.0, 0.0), (1.0, 0.0), (2.0, 1.0)),  # Open chain.
        (
            (0.0, 0.0),
            (2.0, 0.0),
            (0.5, 0.5),
            (0.0, 2.0),
            (0.0, 0.0),
        ),  # Concave quadrilateral.
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 1.0),
            (1.0, 2.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ),  # Convex pentagon.
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.2),
            (1.2, 0.2),
            (1.2, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ),  # Stairs.
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ),  # Backtracking.
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0)),  # Crossing.
    ],
)
def test_complex_noisy_or_nonconvex_source_is_not_replaced_with_raw_edges(
    points: tuple[c.Point2, ...],
) -> None:
    result, report = fitting._polygon_curvature_candidate(
        fit_for(points), 0, LIMIT, 0.2
    )
    assert result is None and report["status"] == "unsupported-source"


def test_exact_primitive_still_requires_its_source_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = fit_for(((0.0, 0.0), (0.18, 0.0), (0.18, 0.88), (0.0, 0.88), (0.0, 0.0)))

    def reject(*_args: object) -> set[int]:
        return {0}

    monkeypatch.setattr(fitting, "_source_failures", reject)
    result, report = fitting._polygon_curvature_candidate(fit, 0, LIMIT, 0.2)
    assert result is None and report["status"] == "uncertified"


def test_point_contact_slot_keeps_its_top_and_closes_lower_down() -> None:
    exterior = box(0, 0, 6, 6).difference(box(2.82, 3.88, 3.0, 6.1))
    assert isinstance(exterior, Polygon)
    polygon = Polygon(exterior.exterior.coords, [box(3, 3, 3.18, 3.88).exterior.coords])
    assert polygon.is_valid
    shape, _ = curved_occt.roof(c.polygon_lines(polygon), 2, 4, tan(radians(10)))
    assert occt.valid(shape)
    assert occt.area(occt.cross_section(shape, 4)) == pytest.approx(
        polygon.area, abs=1e-8
    )
    patch = box(3.08, 3.4, 3.10, 3.42)
    upper = occt.common(occt.cross_section(shape, 3.9), occt.polygon_face(patch, 3.9))
    lower = occt.common(occt.cross_section(shape, 3.2), occt.polygon_face(patch, 3.2))
    assert occt.area(upper) == pytest.approx(0, abs=1e-9)
    assert occt.area(lower) == pytest.approx(patch.area, abs=1e-9)
    _, roundtrip = _step(shape)
    assert roundtrip["valid"] is True
