"""Native curved source tops retain certified export budgets and topology."""

from dataclasses import replace
from io import BytesIO
from typing import cast

import numpy as np
import pytest
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_BezierCurve, GeomAbs_BSplineCurve
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from PIL import Image
from shapely import constrained_delaunay_triangles
from shapely.geometry import MultiPolygon, Polygon

from panelchemy.construction import ConstructionProfile
from panelchemy.export import (
    ExportBundle,
    ExportConfig,
    ExportError,
    _footprints,  # pyright: ignore[reportPrivateUsage]
    _Mesh,  # pyright: ignore[reportPrivateUsage]
    _read_stl,  # pyright: ignore[reportPrivateUsage]
    _triangle_intersections,  # pyright: ignore[reportPrivateUsage]
    export_model,
)
from panelchemy.relief import ReliefModel, build_relief
from panelchemy.separation import SeparationConfig, separate


def _model(tolerance: float) -> ReliefModel:
    rows, columns = np.indices((80, 80))
    distance = (columns - 34) ** 2 + (rows - 40) ** 2
    labels = np.zeros((80, 80), dtype=np.uint8)
    labels[(distance <= 25**2) & (distance >= 5**2)] = 2
    labels[9:13, 66:70] = 2
    palette = np.asarray(SeparationConfig().palette, dtype=np.uint8)
    silhouette = np.zeros(labels.shape, dtype=np.bool_)
    silhouette[1:-1, 1:-1] = True
    return build_relief(
        separate(Image.fromarray(palette[labels]), silhouette=silhouette),
        ConstructionProfile(
            # Half-millimeter pixels resolve the fitter's physical corner
            # neighborhood; millimeter pixels preserve the raster staircase.
            long_edge_mm=39,
            taper_angle_deg=0,
            contour_tolerance_mm=tolerance,
        ),
    )


@pytest.fixture(scope="module")
def curved_export() -> tuple[ReliefModel, ExportConfig, ExportBundle]:
    model = _model(1.0)
    config = ExportConfig(linear_deflection_mm=0.03)
    before = _snapshot(model)
    bundle = export_model(model, config)
    assert _snapshot(model) == before
    return model, config, bundle


def _snapshot(model: ReliefModel) -> list[bytes]:
    snapshots: list[bytes] = []
    for shape in [model.solid, *model.materials.values()]:
        stream = BytesIO()
        BRepTools.Write_s(shape, stream)
        snapshots.append(stream.getvalue())
    return snapshots


def _meshes(bundle: ExportBundle) -> dict[str, _Mesh]:
    return {
        name: _read_stl(bundle.files[f"{name}.stl"])
        for name in ("black", "orange", "white")
    }


def _report(bundle: ExportBundle, name: str) -> dict[str, float]:
    reports = cast("dict[str, dict[str, float]]", bundle.validation[name])
    return reports["white_top"]


def test_native_curved_export_reports_proxy_mesh_and_rounding_budget(
    curved_export: tuple[ReliefModel, ExportConfig, ExportBundle],
) -> None:
    model, config, bundle = curved_export
    assert model.curves is not None
    assert model.footprint_proxy_tolerance_mm > 0
    explorer = TopExp_Explorer(model.materials["white"], TopAbs_EDGE)
    has_bezier = False
    while explorer.More():
        curve = BRepAdaptor_Curve(TopoDS.Edge(explorer.Current()))
        has_bezier |= curve.GetType() in (GeomAbs_BezierCurve, GeomAbs_BSplineCurve)
        explorer.Next()
    assert has_bezier
    assert len(model.contours.white) == 2
    assert sum(len(p.interiors) for p in model.contours.white) == 1
    assert set(bundle.files) == {
        "black.stl",
        "orange.stl",
        "white.stl",
        "relief.stl",
        "relief.step",
    }
    assert cast("dict[str, object]", bundle.validation["step"])["valid"] is True
    for name in ("footprints", "serialized_stl_footprints"):
        report = _report(bundle, name)
        assert report["curve_proxy_tolerance_mm"] == model.footprint_proxy_tolerance_mm
        assert report["mesh_linear_deflection_mm"] == config.linear_deflection_mm
        assert report["boundary_tolerance_mm"] == pytest.approx(
            report["coordinate_tolerance_mm"]
            + model.footprint_proxy_tolerance_mm
            + config.linear_deflection_mm
        )
        silhouette = cast("dict[str, dict[str, float]]", bundle.validation[name])[
            "silhouette_bottom"
        ]
        assert silhouette["curve_proxy_tolerance_mm"] == 0
        assert silhouette["mesh_linear_deflection_mm"] == 0
        assert (
            silhouette["boundary_tolerance_mm"] == silhouette["coordinate_tolerance_mm"]
        )
    assert (
        _report(bundle, "serialized_stl_footprints")["coordinate_tolerance_mm"]
        > _report(bundle, "footprints")["coordinate_tolerance_mm"]
    )


def test_nonzero_tapered_native_curves_export_with_conformal_support() -> None:
    rows, columns = np.indices((64, 64))
    distance = (columns - 32) ** 2 + (rows - 32) ** 2
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[(distance <= 20**2) & (distance >= 5**2)] = 2
    labels[7, 7] = 2
    palette = np.asarray(SeparationConfig().palette, dtype=np.uint8)
    model = build_relief(
        separate(
            Image.fromarray(palette[labels]),
            silhouette=np.ones(labels.shape, dtype=np.bool_),
        ),
        ConstructionProfile(
            long_edge_mm=32, contour_tolerance_mm=0.2, taper_angle_deg=10
        ),
    )
    bundle = export_model(model)
    assert model.curves is not None
    assert any(
        not span.is_line
        for region in model.curves.white
        for ring in (region.exterior, *region.interiors)
        for span in ring.spans
    )
    assert cast("dict[str, object]", bundle.validation["step"])["valid"] is True
    reports = cast("dict[str, dict[str, object]]", bundle.validation["stl"])
    assert reports["relief"]["closed_shells"] == 1
    interfaces = cast(
        "dict[str, dict[str, int]]", bundle.validation["serialized_stl_interfaces"]
    )
    assert interfaces["orange/white"]["shared_triangles"] > 0
    assert interfaces["orange/white"]["missing_shared_triangles"] == 0
    footprints = cast(
        "dict[str, object]", bundle.validation["serialized_stl_footprints"]
    )
    assert footprints["orange_source_top_coverage"] is True
    assert footprints["vertical_outer_wall"] is True
    assert _report(bundle, "serialized_stl_footprints")["curve_proxy_tolerance_mm"] > 0


def test_indexed_shared_vertex_is_exactly_on_both_triangle_planes() -> None:
    # Native tapered annulus pair: a floating normal gives the indexed shared
    # vertex a signed distance of -1.53e-16, inventing a crossing 3.66e-12 away.
    mesh = _Mesh(
        [
            (6.415182185995, 15.211903310523, 3.0),
            (6.292893236116, 15.084877085248, 2.0),
            (6.297445583325, 15.080068923669, 2.0),
            (6.397738361171, 15.227236510668, 3.0),
            (6.287088768978, 15.089960204989, 2.0),
        ],
        [(0, 1, 2), (3, 4, 1)],
    )
    checked, contacts, tolerance = _triangle_intersections(mesh, 7e-13)
    assert checked == 1
    assert contacts == 0
    assert tolerance < 3e-12


def test_nearby_unshared_vertex_does_not_gain_exact_plane_incidence() -> None:
    mesh = _Mesh(
        [
            (0, 0, 0),
            (2, 0, 0),
            (0, 2, 0),
            (1e-11, 1e-11, 1e-11),
            (1, -0.1, -1),
            (1, 1, -1),
        ],
        [(0, 1, 2), (3, 4, 5)],
    )
    with pytest.raises(ExportError, match="intersecting"):
        _ = _triangle_intersections(mesh, 7e-13)


def test_shared_vertex_cannot_hide_a_crossing_elsewhere() -> None:
    mesh = _Mesh(
        [(0, 0, 0), (2, 0, 0), (0, 2, 0), (1, 1, 1), (1, 1, -1)],
        [(0, 1, 2), (0, 3, 4)],
    )
    with pytest.raises(ExportError, match="intersecting"):
        _ = _triangle_intersections(mesh, 7e-13)


def test_curved_footprint_corruption_beyond_budget_is_rejected(
    curved_export: tuple[ReliefModel, ExportConfig, ExportBundle],
) -> None:
    model, config, bundle = curved_export
    meshes = _meshes(bundle)
    white = meshes["white"]
    shift = (
        2 + 4 * _report(bundle, "serialized_stl_footprints")["boundary_tolerance_mm"]
    )
    meshes["white"] = _Mesh(
        [(x + shift, y, z) for x, y, z in white.points], white.triangles
    )
    with pytest.raises(ExportError, match="white_top footprint"):
        _ = _footprints(meshes, model, 1e-6, config.linear_deflection_mm)


@pytest.mark.parametrize("wall_only", [False, True])
def test_curve_allowance_does_not_relax_exact_silhouette(
    curved_export: tuple[ReliefModel, ExportConfig, ExportBundle], wall_only: bool
) -> None:
    model, config, bundle = curved_export
    meshes = _meshes(bundle)
    black = meshes["black"]
    meshes["black"] = _Mesh(
        [
            (x + 0.001 if not wall_only or z > 0 else x, y, z)
            for x, y, z in black.points
        ],
        black.triangles,
    )
    message = (
        "outer wall is not vertical" if wall_only else "silhouette_bottom footprint"
    )
    with pytest.raises(ExportError, match=message):
        _ = _footprints(meshes, model, 1e-6, config.linear_deflection_mm)


@pytest.mark.parametrize(
    "height_spread,uncertainty", [(1e-12, 0.0), (1e-10, 0.0), (1e-6, 1e-6)]
)
def test_horizontal_cap_coordinate_noise_is_not_an_interior_wall(
    curved_export: tuple[ReliefModel, ExportConfig, ExportBundle],
    height_spread: float,
    uncertainty: float,
) -> None:
    model, config, bundle = curved_export
    meshes = _meshes(bundle)
    black = meshes["black"]
    start = len(black.points)
    meshes["black"] = _Mesh(
        [
            *black.points,
            (10.0, 10.0, 2.0),
            (11.0, 10.0, 2.0),
            (10.0, 11.0, 2.0 + height_spread),
        ],
        [*black.triangles, (start, start + 1, start + 2)],
    )
    result = _footprints(meshes, model, uncertainty, config.linear_deflection_mm)
    assert result["vertical_outer_wall"] is True
    assert cast("float", result["black_cap_height_tolerance_mm"]) >= height_spread


@pytest.mark.parametrize("height,spread", [(2.0, 1e-6), (1.5, 1e-12), (2.0, 0.001)])
def test_interior_triangle_away_from_native_caps_is_rejected(
    curved_export: tuple[ReliefModel, ExportConfig, ExportBundle],
    height: float,
    spread: float,
) -> None:
    model, config, bundle = curved_export
    meshes = _meshes(bundle)
    black = meshes["black"]
    start = len(black.points)
    meshes["black"] = _Mesh(
        [
            *black.points,
            (10.0, 10.0, height),
            (11.0, 10.0, height),
            (10.0, 11.0, height + spread),
        ],
        [*black.triangles, (start, start + 1, start + 2)],
    )
    with pytest.raises(ExportError, match="outer wall is not vertical"):
        _ = _footprints(meshes, model, 0.0, config.linear_deflection_mm)


@pytest.mark.parametrize("remove", ["island", "counter"])
def test_curve_budget_cannot_hide_lost_islands_or_counters(
    curved_export: tuple[ReliefModel, ExportConfig, ExportBundle], remove: str
) -> None:
    model, config, bundle = curved_export
    parts = sorted(model.contours.white, key=lambda polygon: polygon.area, reverse=True)
    geometry = (
        parts[0]
        if remove == "island"
        else MultiPolygon([Polygon(parts[0].exterior), *parts[1:]])
    )
    points: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for triangle in constrained_delaunay_triangles(geometry).geoms:
        assert isinstance(triangle, Polygon)
        start = len(points)
        points.extend(
            (float(x), float(y), model.profile.white_top)
            for x, y in triangle.exterior.coords[:-1]
        )
        triangles.append((start, start + 1, start + 2))
    meshes = _meshes(bundle)
    meshes["white"] = _Mesh(points, triangles)
    with pytest.raises(ExportError, match="white_top footprint"):
        _ = _footprints(
            meshes,
            replace(model, footprint_proxy_tolerance_mm=100),
            1e-6,
            config.linear_deflection_mm,
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.01])
def test_invalid_proxy_budget_is_rejected_before_export(
    curved_export: tuple[ReliefModel, ExportConfig, ExportBundle], invalid: float
) -> None:
    model, config, _ = curved_export
    with pytest.raises(ExportError, match="footprint_proxy_tolerance_mm"):
        _ = export_model(replace(model, footprint_proxy_tolerance_mm=invalid), config)


def test_exact_export_retains_coordinate_only_footprint_budget() -> None:
    model = _model(0)
    assert model.curves is None
    assert model.footprint_proxy_tolerance_mm == 0
    bundle = export_model(model, ExportConfig(linear_deflection_mm=0.5))
    report = _report(bundle, "serialized_stl_footprints")
    assert report["curve_proxy_tolerance_mm"] == 0
    assert report["mesh_linear_deflection_mm"] == 0
    assert report["boundary_tolerance_mm"] == report["coordinate_tolerance_mm"]
    meshes = _meshes(bundle)
    white = meshes["white"]
    meshes["white"] = _Mesh(
        [(x + 0.001, y, z) for x, y, z in white.points], white.triangles
    )
    with pytest.raises(ExportError, match="white_top footprint"):
        _ = _footprints(meshes, model, 1e-8, 0.5)
