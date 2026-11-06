"""Boundary-aware refitting keeps counters and passes unchanged native checks."""

# pyright: reportPrivateUsage=false

from dataclasses import replace
from fractions import Fraction
from itertools import pairwise
from math import radians, tan
from typing import cast

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.polygon import orient

from panelchemy import curvature, curvature_fitting, curved_occt, occt
from panelchemy import curves as c
from panelchemy import outline_fitting as fitting
from panelchemy.contours import PanelContours
from panelchemy.outline_fitting import fit_outlines


def source_with_notched_counter() -> PanelContours:
    counter = Polygon(
        [(4, 4), (4.53, 4), (4.53, 4.17), (4.35, 4.17), (4.35, 4.35), (4, 4.35)]
    )
    orange = orient(Polygon(box(1, 1, 9, 9).exterior, [counter.exterior]))
    silhouette = box(0, 0, 10, 10)
    return PanelContours(silhouette, [orange], [], 0.1, (0, 0, 100, 100), 14)


def test_final_refit_preserves_counter_and_produces_valid_native_taper() -> None:
    source = source_with_notched_counter()
    original = source.orange[0].wkb
    slope = tan(radians(10))
    fitted = fit_outlines(
        source,
        0.2,
        corner_softening_mm=0.25,
        span_simplification_mm=0.1,
        material_taper_runs_mm=(0, slope, 2 * slope),
    )
    diagnostics = cast("dict[str, object]", fitted.diagnostics["boundary_curvature"])
    assert cast("int", diagnostics["refined_chains"]) >= 1
    assert not diagnostics["unrefined_chains"]
    assert len(fitted.contours.orange[0].interiors) == 1
    assert all(
        curvature.within_curvature(s, slope, -1)
        for s in fitted.contours.orange[0].interiors[0].spans
    )
    spans = fitted.contours.orange[0].interiors[0].spans
    for before, after in pairwise((*spans, spans[0])):
        assert before.control_points[-1] == after.control_points[0]
        u = [
            Fraction(before.control_points[3][i])
            - Fraction(before.control_points[2][i])
            for i in (0, 1)
        ]
        v = [
            Fraction(after.control_points[1][i]) - Fraction(after.control_points[0][i])
            for i in (0, 1)
        ]
        assert u[0] * v[1] == u[1] * v[0]
        assert u[0] * v[0] + u[1] * v[1] > 0
    assert source.orange[0].wkb == original
    roof, _ = curved_occt.roof(fitted.contours.orange[0], 2, 3, slope)
    assert occt.valid(roof)
    assert occt.volume(roof) > 0


def test_zero_taper_keeps_ordinary_tracing_behavior() -> None:
    source = source_with_notched_counter()
    ordinary = fit_outlines(source, 0.2)
    flat = fit_outlines(source, 0.2, material_taper_runs_mm=(0, 0, 0))
    assert flat.contours == ordinary.contours
    assert "boundary_curvature" not in flat.diagnostics


def test_unavailable_refit_preserves_the_certified_source_and_reports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(
        *_args: object, **_kwargs: object
    ) -> tuple[None, dict[str, object]]:
        return None, {"status": "time-budget"}

    monkeypatch.setattr(curvature_fitting, "refit", unavailable)
    source = source_with_notched_counter()
    ordinary = fit_outlines(source, 0.2, corner_softening_mm=0.25)
    fitted = fit_outlines(
        source,
        0.2,
        corner_softening_mm=0.25,
        material_taper_runs_mm=(0, 0.176, 0.353),
    )
    assert fitted.contours == ordinary.contours
    diagnostics = cast("dict[str, object]", fitted.diagnostics["boundary_curvature"])
    assert diagnostics["refined_chains"] == 0
    assert diagnostics["unrefined_chains"]
    assert len(fitted.contours.orange[0].interiors) == 1


def test_public_curvature_fitting_normalizes_source_winding() -> None:
    source = source_with_notched_counter()
    reversed_source = replace(
        source,
        orange=[orient(source.orange[0], sign=-1)],
    )
    runs = (0.0, 0.176, 0.353)
    normal = fit_outlines(source, 0.2, material_taper_runs_mm=runs)
    reversed_fit = fit_outlines(reversed_source, 0.2, material_taper_runs_mm=runs)
    assert normal.contours == reversed_fit.contours


def test_shared_material_ownership_accepts_reversed_input_ring() -> None:
    source = PanelContours(
        box(0, 0, 10, 10),
        [box(2, 2, 5, 8)],
        [box(5, 2, 8, 8)],
        0.1,
        (0, 0, 100, 100),
        12,
    )
    reversed_source = replace(source, white=[orient(source.white[0], sign=-1)])
    runs = (0.0, 0.1, 0.2)
    normal = fit_outlines(source, 0.05, material_taper_runs_mm=runs)
    reversed_fit = fit_outlines(reversed_source, 0.05, material_taper_runs_mm=runs)
    assert normal.contours == reversed_fit.contours


def test_topology_rollback_keeps_an_independent_safe_refit() -> None:
    source = PanelContours(
        box(0, 0, 10, 10),
        [box(1, 2, 2, 3), box(5, 2, 6, 3), box(6.2, 2, 7.2, 3)],
        [],
        0.1,
        (0, 0, 100, 100),
        16,
    )
    rings = [LineString(p.exterior.coords) for p in [source.silhouette, *source.orange]]
    chains, uses = c._graph(rings, split_loops=False)
    ranges = [range(i, i + 1) for i in range(len(rings))]
    fits: list[fitting._Fit] = []
    for index, chain in enumerate(chains):
        pieces = [
            c._Piece(index, i, i + 1, c._line(a, b))
            for i, (a, b) in enumerate(pairwise(chain.points))
        ]
        times = np.linspace(0, 1, len(chain.points))
        fits.append(
            fitting._Fit(
                chain,
                times,
                [],
                [],
                pieces,
                list(pairwise(times.tolist())),
            )
        )
    original_pieces = [p for f in fits for p in f.pieces]
    unsafe_pairs, cuts = c._unsafe_pair(original_pieces)
    assert not unsafe_pairs
    original_curves, original_proxy, _, _ = c._assemble(
        original_pieces, uses, ranges, source, 0.01, cuts
    )
    candidates = list(fits)
    safe, unsafe = uses[1][0][0], uses[2][0][0]
    for index, shift in ((safe, 0.1), (unsafe, 0.3)):
        pieces = [
            replace(
                p,
                span=c._line(
                    (p.span.control_points[0][0] + shift, p.span.control_points[0][1]),
                    (p.span.control_points[3][0] + shift, p.span.control_points[3][1]),
                ),
            )
            for p in fits[index].pieces
        ]
        candidates[index] = replace(fits[index], pieces=pieces)
    verified = fitting._verify_refits(
        fits,
        candidates,
        {safe, unsafe},
        uses,
        ranges,
        source,
        0.2,
        original_curves,
        original_proxy,
    )
    assert verified.accepted == {safe}
    assert verified.blocked == {unsafe}
    assert verified.proxy.orange[0].equals(
        affinity.translate(source.orange[0], xoff=0.1)
    )
    assert verified.proxy.orange[1].equals(source.orange[1])
