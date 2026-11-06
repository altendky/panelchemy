"""Curve fitting reaches native CAD while source islands and counters survive."""

from typing import cast

import numpy as np
import pytest
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GeomAbs import GeomAbs_BezierCurve, GeomAbs_BSplineCurve
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from PIL import Image

from panelchemy import occt
from panelchemy.construction import ConstructionProfile, ContourMode
from panelchemy.relief import build_relief
from panelchemy.separation import SeparationConfig, separate


@pytest.mark.parametrize("mode", ["curves", "lines"])
def test_fitted_region_keeps_native_outlines_counters_and_islands(
    mode: ContourMode,
) -> None:
    rows, columns = np.indices((64, 64))
    distance_squared = (rows - 32) ** 2 + (columns - 32) ** 2
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[(distance_squared <= 20**2) & (distance_squared >= 5**2)] = 2
    labels[7, 7] = 2
    silhouette = np.ones(labels.shape, dtype=np.bool_)
    result = separate(
        Image.fromarray(np.asarray(SeparationConfig().palette, dtype=np.uint8)[labels]),
        silhouette=silhouette,
    )
    model = build_relief(
        result,
        ConstructionProfile(
            long_edge_mm=32,
            contour_tolerance_mm=0.2,
            contour_mode=mode,
            taper_angle_deg=10,
        ),
    )
    assert model.curves is not None
    assert len(model.curves.white) == 2
    assert sum(len(p.interiors) for p in model.curves.white) == 1
    assert (model.footprint_proxy_tolerance_mm > 0) == (mode == "curves")
    assert occt.valid(model.solid)
    assert model.diagnostics["white_top_missing_mm2"] == pytest.approx(0, abs=1e-8)
    assert model.diagnostics["white_top_extra_mm2"] == pytest.approx(0, abs=1e-8)
    assert occt.topology_counts(model.solid)["solids"] == 1
    edge = TopExp_Explorer(model.solid, TopAbs_EDGE)
    curved_edges = 0
    while edge.More():
        native = BRepAdaptor_Curve(TopoDS.Edge(edge.Current()))
        curved_edges += native.GetType() in (GeomAbs_BezierCurve, GeomAbs_BSplineCurve)
        edge.Next()
    if mode == "curves":
        assert curved_edges > 0
    else:
        assert all(
            span.is_line
            for polygon in (*model.curves.orange, *model.curves.white)
            for ring in (polygon.exterior, *polygon.interiors)
            for span in ring.spans
        )
    diagnostics = cast("dict[str, object]", model.diagnostics["contour_approximation"])
    assert diagnostics["tolerance_mm"] == 0.2
    assert diagnostics["mode"] == mode
    if mode == "curves":
        assert diagnostics["method"] == "joint cubic B-spline outline fitting"
        assert diagnostics["corner_scale_mm"] == 1
        assert diagnostics["fairing_mm"] == 0.25
    assert (" C " in (model.contour_comparison_svg or "")) == (mode == "curves")


def test_exact_mode_does_not_claim_curve_approximation() -> None:
    labels = np.zeros((10, 10), dtype=np.uint8)
    labels[3:6, 3:6] = 2
    result = separate(
        Image.fromarray(np.asarray(SeparationConfig().palette, dtype=np.uint8)[labels]),
        silhouette=np.ones(labels.shape, dtype=np.bool_),
    )
    model = build_relief(result, ConstructionProfile(long_edge_mm=10))
    assert model.curves is None
    assert model.footprint_proxy_tolerance_mm == 0
    assert model.diagnostics["contour_approximation"] is None
