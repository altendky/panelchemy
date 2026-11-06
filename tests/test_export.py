"""Serialized registration, material contacts and explicit invalid-mesh failures."""

from fractions import Fraction
from io import BytesIO
from typing import cast, override

import numpy as np
import pytest
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepTools import BRepTools
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from PIL import Image

from panelchemy import export as export_module
from panelchemy import occt
from panelchemy.construction import ConstructionProfile

# Corruption probes exercise the internal serializer/validator directly so the
# production API does not expose mutable mesh internals merely for tests.
from panelchemy.export import (
    ExportConfig,
    ExportError,
    _check_float32,  # pyright: ignore[reportPrivateUsage]
    _combined,  # pyright: ignore[reportPrivateUsage]
    _conform_meshes,  # pyright: ignore[reportPrivateUsage]
    _Mesh,  # pyright: ignore[reportPrivateUsage]
    _normal,  # pyright: ignore[reportPrivateUsage]
    _normalize_material_boundaries,  # pyright: ignore[reportPrivateUsage]
    _quantize_meshes,  # pyright: ignore[reportPrivateUsage]
    _read_stl,  # pyright: ignore[reportPrivateUsage]
    _Relief,  # pyright: ignore[reportPrivateUsage]
    _step,  # pyright: ignore[reportPrivateUsage]
    _stl,  # pyright: ignore[reportPrivateUsage]
    _triangle_intersections,  # pyright: ignore[reportPrivateUsage]
    _validate_mesh,  # pyright: ignore[reportPrivateUsage]
    export_model,
)
from panelchemy.relief import ReliefModel, build_relief
from panelchemy.separation import SeparationConfig, separate


def _model(colors: int = 3, *, counter: bool = False, taper: float = 10) -> ReliefModel:
    labels = np.full((14, 16), 2, dtype=np.uint8)
    labels[1:-1, 1:-1] = 0
    if colors >= 2:
        labels[3:5, 3:6] = 1
    if colors == 3:
        labels[6:9, 7:10] = 2
        labels[3:5, 10] = 2  # Fine stroke independent of nozzle diameter.
        if counter:
            labels[7, 8] = 0
    palette = np.asarray(SeparationConfig().palette, dtype=np.uint8)
    silhouette = np.zeros(labels.shape, dtype=np.bool_)
    silhouette[1:-1, 1:-1] = True
    return build_relief(
        separate(Image.fromarray(palette[labels]), silhouette=silhouette),
        ConstructionProfile(long_edge_mm=14, taper_angle_deg=taper),
    )


def _snapshot(model: ReliefModel) -> list[bytes]:
    snapshots: list[bytes] = []
    for shape in [model.solid, *model.materials.values()]:
        stream = BytesIO()
        BRepTools.Write_s(shape, stream)
        snapshots.append(stream.getvalue())
    return snapshots


@pytest.mark.parametrize("zero_bridge", [False, True])
def test_export_cancels_opposite_sheets_after_preserving_zero_bridge_hints(
    monkeypatch: pytest.MonkeyPatch,
    zero_bridge: bool,
) -> None:
    model = _model(1, taper=0)
    clean = export_model(model)
    original_partition = export_module._partition  # pyright: ignore[reportPrivateUsage]
    input_triangles = 0

    def with_double_sheet(
        selected: _Relief, config: ExportConfig
    ) -> tuple[dict[str, _Mesh], dict[str, object]]:
        nonlocal input_triangles
        meshes, diagnostics = original_partition(selected, config)
        black = meshes["black"]
        a, b, c = black.triangles[0]
        left, right = black.points[a], black.points[b]
        axis = min(range(3), key=lambda i: abs(left[i] - right[i]))
        middle = [(left[i] + right[i]) / 2 for i in range(3)]
        if not zero_bridge:
            middle[axis] += 2.0**-40
        point = (middle[0], middle[1], middle[2])
        if not zero_bridge:
            _ = _normal([left, right, point])  # Positive, tiny sheet area.
        extra = len(black.points)
        positive = (
            [(a, extra, c), (extra, b, c), *black.triangles[1:]]
            if zero_bridge
            else black.triangles
        )
        meshes["black"] = _Mesh(
            [*black.points, point],
            [*positive, (a, b, extra), (a, extra, b)],
        )
        input_triangles = len(meshes["black"].triangles)
        return meshes, diagnostics

    monkeypatch.setattr(export_module, "_partition", with_double_sheet)
    repaired = export_model(model)
    for name in ("black.stl", "relief.stl"):
        assert _exact_volume(_read_stl(repaired.files[name])) == _exact_volume(
            _read_stl(clean.files[name])
        )
        if not zero_bridge:
            assert repaired.files[name] == clean.files[name]
    numerical = cast("dict[str, object]", repaired.validation["numerical_processing"])
    conformance = cast("dict[str, object]", numerical["native_conformance"])
    normalization = cast(
        "dict[str, object]", numerical["native_boundary_normalization"]
    )
    materials = cast("dict[str, dict[str, object]]", normalization["materials"])
    assert conformance["input_triangles"] == input_triangles
    assert materials["black"]["input_triangles"] == conformance["output_triangles"]
    assert materials["black"]["output_triangles"] == input_triangles - (
        1 if zero_bridge else 2
    )
    assert materials["black"]["exact_opposite_pairs_removed"] == int(not zero_bridge)
    assert materials["black"]["coordinate_movement_mm"] == 0
    if zero_bridge:
        assert conformance["removed_exact_zero_triangles"] == 2
        assert conformance["added_positive_triangles"] == 1


@pytest.mark.parametrize("colors,highest", [(1, 2), (2, 3), (3, 4)])
def test_serialized_exports_keep_registration_and_optional_bands(
    colors: int, highest: int
) -> None:
    model = _model(colors, taper=0)
    before = _snapshot(model)
    bundle = export_model(model)
    assert _snapshot(model) == before
    expected = (
        {"black", "orange", "white"}
        if colors == 3
        else {"black", "orange"}
        if colors == 2
        else {"black"}
    )
    assert set(bundle.files) == {f"{name}.stl" for name in expected} | {
        "relief.stl",
        "relief.step",
    }
    for name in expected:
        stl = _read_stl(bundle.files[f"{name}.stl"])
        report = _validate_mesh(stl)
        assert report["volume_mm3"] == pytest.approx(
            occt.volume(model.materials[name]), rel=1e-6
        )
        assert (
            min(p[2] for p in stl.points) == {"black": 0, "orange": 2, "white": 3}[name]
        )
        assert (
            max(p[2] for p in stl.points) == {"black": 2, "orange": 3, "white": 4}[name]
        )
    combined = _read_stl(bundle.files["relief.stl"])
    assert max(p[2] for p in combined.points) == highest
    assert _validate_mesh(combined)["volume_mm3"] == pytest.approx(
        occt.volume(model.solid), rel=1e-6
    )
    assert b"SI_UNIT(.MILLI.,.METRE.)" in bundle.files["relief.step"]


def test_taper_counter_and_shared_interfaces_are_serialized_conformally() -> None:
    model = _model(counter=True, taper=35)
    bundle = export_model(model)
    interfaces = cast(
        "dict[str, dict[str, int]]", bundle.validation["serialized_stl_interfaces"]
    )
    assert set(interfaces) == {"black/orange", "orange/white"}
    assert all(
        report["shared_triangles"] > 0 and report["missing_shared_triangles"] == 0
        for report in interfaces.values()
    )
    reports = cast("dict[str, dict[str, object]]", bundle.validation["stl"])
    assert reports["relief"]["closed_shells"] == 1
    assert cast("int", reports["orange"]["numerical_boundary_contact_pairs"]) > 0
    processing = cast("dict[str, object]", bundle.validation["numerical_processing"])
    assert cast("float", processing["maximum_coordinate_rounding_mm"]) < 1e-11
    assert "collapsed_native_triangles_discarded" in processing


def test_shape_tolerance_envelope_does_not_cross_geometric_material_planes() -> None:
    model = _model(1)
    builder = BRep_Builder()
    for kind in (TopAbs_EDGE, TopAbs_VERTEX):
        explorer = TopExp_Explorer(model.materials["black"], kind)
        while explorer.More():
            if kind == TopAbs_EDGE:
                builder.UpdateEdge(TopoDS.Edge(explorer.Current()), 2.5e-6)
            else:
                builder.UpdateVertex(TopoDS.Vertex(explorer.Current()), 2.5e-6)
            explorer.Next()
    assert occt.valid(model.materials["black"])
    conservative = Bnd_Box()
    BRepBndLib.Add_s(model.materials["black"], conservative, False)
    assert conservative.CornerMin().Z() < -1e-6
    assert conservative.CornerMax().Z() > model.profile.black_top + 1e-6
    before = _snapshot(model)
    bundle = export_model(model)
    assert _snapshot(model) == before
    points = _read_stl(bundle.files["black.stl"]).points
    assert min(point[2] for point in points) == 0
    assert max(point[2] for point in points) == model.profile.black_top


def test_actual_material_geometry_crossing_still_rejects() -> None:
    model = _model(1)
    model.materials["black"] = occt.prism(
        model.contours.silhouette, -0.01, model.profile.black_top
    )
    before = _snapshot(model)
    with pytest.raises(ExportError, match="partition cell crosses a material boundary"):
        _ = export_model(model)
    assert _snapshot(model) == before


def _tetra(offset: tuple[float, float, float] = (0, 0, 0)) -> _Mesh:
    points = [
        (float(x + offset[0]), float(y + offset[1]), float(z + offset[2]))
        for x, y, z in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    ]
    return _Mesh(points, [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)])


def _bridged_tetra(middle_y: float = 0.0) -> _Mesh:
    return _Mesh(
        [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, middle_y, 0)],
        [(0, 2, 1), (0, 4, 3), (4, 1, 3), (0, 3, 2), (1, 2, 3), (0, 1, 4)],
    )


def _exact_volume(mesh: _Mesh) -> Fraction:
    terms: list[Fraction] = []
    for triangle in mesh.triangles:
        a, b, c = [
            [Fraction(value) for value in mesh.points[index]] for index in triangle
        ]
        cross = (
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        )
        terms.append(
            sum((a[index] * cross[index] for index in range(3)), Fraction()) / 6
        )
    return sum(terms, Fraction())


def test_boundary_cancellation_preserves_closed_surface_and_exact_volume() -> None:
    clean = _tetra()
    sheet = _Mesh(
        [*clean.points, (0.5, -(2.0**-40), 0.0)],
        [*clean.triangles, (0, 1, 4), (0, 4, 1)],
    )
    _ = _normal([sheet.points[i] for i in (0, 1, 4)])
    with pytest.raises(ExportError, match="nonmanifold edge"):
        _ = _validate_mesh(sheet)
    normalized, report = _normalize_material_boundaries({"part": sheet})
    result = normalized["part"]
    assert result.points == clean.points
    assert result.triangles == clean.triangles
    assert _exact_volume(result) == _exact_volume(sheet) == Fraction(1, 6)
    assert _validate_mesh(result)["closed_shells"] == 1
    assert report["exact_opposite_pairs_removed"] == 1
    assert report["coordinate_movement_mm"] == 0


def test_boundary_normalization_keeps_close_distinct_closed_facets() -> None:
    upper = _tetra()
    lower = _tetra()
    lower.points = [(x, y, -z - 2.0**-40) for x, y, z in lower.points]
    lower.triangles = [(a, c, b) for a, b, c in lower.triangles]
    mesh = _combined({"upper": upper, "lower": lower})
    normalized, report = _normalize_material_boundaries({"part": mesh})
    result = normalized["part"]
    assert result.points == mesh.points
    assert result.triangles == mesh.triangles
    assert _exact_volume(result) == _exact_volume(mesh)
    assert _validate_mesh(result)["closed_shells"] == 2
    assert report["exact_opposite_pairs_removed"] == 0


@pytest.mark.parametrize("multiplicity", [2, 3])
def test_boundary_normalization_still_rejects_duplicate_corruption(
    multiplicity: int,
) -> None:
    mesh = _tetra()
    mesh.triangles.extend([mesh.triangles[0]] * (multiplicity - 1))
    message = "inconsistent winding" if multiplicity == 2 else "multiplicity"
    with pytest.raises(ExportError, match=message):
        _ = _normalize_material_boundaries({"part": mesh})


def _triangle_keys(mesh: _Mesh) -> set[tuple[tuple[float, float, float], ...]]:
    return {
        tuple(sorted(mesh.points[index] for index in triangle))
        for triangle in mesh.triangles
    }


@pytest.mark.parametrize("vertical_edge", [False, True])
def test_exact_zero_bridge_requires_conforming_subdivision_and_keeps_coordinates(
    vertical_edge: bool,
) -> None:
    mesh = _bridged_tetra()
    if vertical_edge:
        mesh.points = [(y, z, x) for x, y, z in mesh.points]
    with pytest.raises(ExportError, match="zero-area"):
        _ = _validate_mesh(mesh)
    with pytest.raises(ExportError, match="open"):
        _ = _validate_mesh(_Mesh(mesh.points, mesh.triangles[:-1]))
    conformed, report = _conform_meshes({"part": mesh}, 100)
    fixed = conformed["part"]
    assert fixed.points == mesh.points
    assert _exact_volume(fixed) == _exact_volume(mesh) == Fraction(4, 3)
    assert _validate_mesh(fixed)["closed_shells"] == 1
    assert report["removed_exact_zero_triangles"] == 1
    assert report["added_positive_triangles"] == 1
    assert report["coordinate_movement_mm"] == 0


def test_three_constrained_cap_edges_match_opposite_winding() -> None:
    points = [
        (0.0, 0.0, 2.0),
        (2.0, 0.0, 2.0),
        (0.0, 2.0, 2.0),
        (1.0, 0.0, 2.0),
        (1.0, 1.0, 2.0),
        (0.0, 1.0, 2.0),
    ]
    lower = _Mesh(points, [(0, 1, 2), (0, 1, 3), (1, 2, 4), (2, 0, 5)])
    upper = _Mesh(points[:3], [(2, 1, 0)])
    result, report = _conform_meshes({"lower": lower, "upper": upper}, 100)
    assert _triangle_keys(result["lower"]) == _triangle_keys(result["upper"])
    assert len(result["lower"].triangles) == len(result["upper"].triangles) == 4
    assert _exact_volume(result["lower"]) == _exact_volume(lower)
    assert _exact_volume(result["upper"]) == _exact_volume(upper)
    assert result["lower"].points == points
    assert set(result["upper"].points) == set(points)
    for name, expected_sign in (("lower", 1), ("upper", -1)):
        assert all(
            _normal([result[name].points[index] for index in triangle])[2]
            == expected_sign
            for triangle in result[name].triangles
        )
    assert report["removed_exact_zero_triangles"] == 3
    assert report["added_positive_triangles"] == 6
    with pytest.raises(ExportError, match="subdivision exceeds max_triangles"):
        _ = _conform_meshes({"lower": lower, "upper": upper}, 5)


def test_nested_middle_vertices_split_containing_edges_across_parts() -> None:
    points = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.5, 0.0, 0.0),
    ]
    first = _Mesh(points, [(0, 1, 2), (0, 1, 3), (0, 1, 4)])
    second = _Mesh(points[:3], [(1, 0, 2)])
    result, _ = _conform_meshes({"first": first, "second": second}, 100)
    assert _triangle_keys(result["first"]) == _triangle_keys(result["second"])
    assert len(result["first"].triangles) == 3
    assert set(result["second"].points) == set(points)


def test_conformance_rejects_losing_a_zero_only_material() -> None:
    line = _Mesh([(0, 0, 0), (1, 0, 0), (0.5, 0, 0)], [(0, 1, 2)])
    with pytest.raises(ExportError, match="discard a material vertex"):
        _ = _conform_meshes({"solid": _tetra(), "line": line}, 100)


def test_positive_exact_cross_survives_floating_product_cancellation() -> None:
    epsilon = 2.0**-27
    mesh = _Mesh([(0, 0, 0), (1 + epsilon, 1, 0), (1, 1 - epsilon, 0)], [(0, 1, 2)])
    # Naive multiplication rounds both products to one; the exact determinant
    # is negative and this face must remain present with an outward normal.
    assert (1 + epsilon) * (1 - epsilon) - 1 == 0
    result, report = _conform_meshes({"part": mesh}, 10)
    assert result["part"].triangles == mesh.triangles
    assert result["part"].points == mesh.points
    assert report["removed_exact_zero_triangles"] == 0
    assert _normal(mesh.points) == (0, 0, -1)


def test_float32_created_exact_zero_has_separate_conformance_phase() -> None:
    mesh = _bridged_tetra(middle_y=1e-46)
    native, native_report = _conform_meshes({"part": mesh}, 100)
    assert native_report["removed_exact_zero_triangles"] == 0
    quantized, quantization = _quantize_meshes(native, 0.02)
    assert len(set(quantized["part"].points)) == len(mesh.points)
    assert quantization["maximum_coordinate_error_mm"] == 1e-46
    final, report = _conform_meshes(quantized, 100)
    assert report["removed_exact_zero_triangles"] == 1
    assert report["added_positive_triangles"] == 1
    assert final["part"].points == quantized["part"].points
    assert _validate_mesh(final["part"])["closed_shells"] == 1
    serialized = _read_stl(_stl(final["part"]))
    assert _check_float32(final["part"], serialized, 0.02) == 0


def test_global_quantization_rejects_coordinate_merges_between_materials() -> None:
    first, second = _tetra((250, 0, 0)), _tetra((250.000001, 0, 0))
    with pytest.raises(ExportError, match="collapses distinct feature vertices"):
        _ = _quantize_meshes({"first": first, "second": second}, 0.02)


@pytest.mark.parametrize("value", [1e40, float("inf"), float("nan")])
def test_quantization_unrepresentable_coordinates_fail_actionably(value: float) -> None:
    mesh = _tetra()
    mesh.points[0] = (value, 0, 0)
    with pytest.raises(ExportError, match="float32"):
        _ = _quantize_meshes({"part": mesh}, 0.02)


def test_float32_rejects_collapsing_distinct_feature_vertices() -> None:
    mesh = _tetra((250, 0, 0))
    mesh.points[1] = (250.000001, 0, 0)
    assert cast("float", _validate_mesh(mesh)["volume_mm3"]) > 0
    serialized = _read_stl(_stl(mesh))
    with pytest.raises(ExportError, match="float32 collapses distinct"):
        _ = _check_float32(mesh, serialized, 0.02)


def test_vertex_link_rejects_point_touching_closed_components() -> None:
    first = _tetra()
    second = _tetra((-1, -1, -1))
    # Put its far corner at the first tetrahedron's origin, sharing one vertex.
    second.points = [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    second.triangles = [(a, c, b) for a, b, c in second.triangles]
    combined = _combined({"first": first, "second": second})
    with pytest.raises(ExportError, match="pinched nonmanifold vertex"):
        _ = _validate_mesh(combined)


def test_self_intersection_rejects_overlapping_closed_shells() -> None:
    combined = _combined({"a": _tetra(), "b": _tetra((0.2, 0.2, 0.2))})
    with pytest.raises(ExportError, match="intersecting"):
        _ = _validate_mesh(combined)


def test_numerical_seam_contact_does_not_allow_interior_overlap() -> None:
    mesh = _Mesh(
        [(0, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1e-13, 0), (2, -1, 0)],
        [(0, 1, 2), (3, 4, 1), (0, 3, 1)],
    )
    checked, contacts, tolerance = _triangle_intersections(mesh, 1e-12)
    assert checked > 0 and contacts > 0
    assert tolerance < 1e-10
    # The same topological connection cannot excuse a real overlap inside A.
    mesh.points[4] = (0.5, 0.5, 0)
    with pytest.raises(ExportError, match="overlapping"):
        _ = _triangle_intersections(mesh, 1e-12)


def test_near_collinear_triangle_inside_another_face_is_rejected() -> None:
    mesh = _Mesh(
        [
            (0, 0, 0),
            (2, 0, 0),
            (0, 2, 0),
            (0.2, 0.2, 0),
            (0.8, 0.2, 0),
            (0.5, 0.2 + 1e-13, 0),
        ],
        [(0, 1, 2), (3, 4, 5)],
    )
    with pytest.raises(ExportError, match="overlapping"):
        _ = _triangle_intersections(mesh, 1e-12)


def test_skinny_seam_chord_cannot_cross_an_opposing_triangle_interior() -> None:
    mesh = _Mesh(
        [
            (0, 0, 0),
            (2, 0, 0),
            (0, 2, 0),
            (0.5, 0, 0),
            (0, 0.5, 0),
            (0.5 + 1e-13, 1e-13, 0),
        ],
        [(0, 1, 2), (3, 4, 5), (3, 5, 0)],
    )
    with pytest.raises(ExportError, match="overlapping"):
        _ = _triangle_intersections(mesh, 1e-12)


def test_float32_check_rejects_reordered_triangle_coordinates() -> None:
    mesh = _tetra()
    parsed = _read_stl(_stl(mesh))
    a, b, c = parsed.triangles[0]
    parsed.triangles[0] = (a, c, b)
    with pytest.raises(ExportError, match="oriented triangle coordinates"):
        _ = _check_float32(mesh, parsed, 0.02)


def test_step_selects_millimeter_context_and_restores_external_unit() -> None:
    _ = STEPControl_Writer()  # Initialize OCCT's static exchange parameters.
    previous = Interface_Static.CVal_s("write.step.unit")
    try:
        assert Interface_Static.SetCVal_s("write.step.unit", "INCH")
        bundle = export_model(_model(1))
        assert Interface_Static.CVal_s("write.step.unit") == "INCH"
        report = cast("dict[str, object]", bundle.validation["step"])
        assert report["serialized_length_units"] == ["millimetre"]
    finally:
        assert Interface_Static.SetCVal_s("write.step.unit", previous)


def test_step_restores_external_unit_after_serialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(1)
    _ = STEPControl_Writer()
    previous = Interface_Static.CVal_s("write.step.unit")

    class FailingWriter(STEPControl_Writer):
        @override
        def WriteStream(self, theOStream: BytesIO) -> IFSelect_ReturnStatus:
            raise RuntimeError("injected STEP serialization failure")

    monkeypatch.setattr(export_module, "STEPControl_Writer", FailingWriter)
    try:
        assert Interface_Static.SetCVal_s("write.step.unit", "INCH")
        with pytest.raises(RuntimeError, match="injected STEP"):
            _ = _step(model.solid)
        assert Interface_Static.CVal_s("write.step.unit") == "INCH"
    finally:
        assert Interface_Static.SetCVal_s("write.step.unit", previous)


def test_mesh_validation_rejects_hole_and_inverted_shell() -> None:
    mesh = _tetra()
    with pytest.raises(ExportError, match="open"):
        _ = _validate_mesh(_Mesh(mesh.points, mesh.triangles[:-1]))
    with pytest.raises(ExportError, match="nonpositive signed volume"):
        _ = _validate_mesh(
            _Mesh(mesh.points, [(a, c, b) for a, b, c in mesh.triangles])
        )


def test_mesh_budget_and_config_fail_explicitly() -> None:
    with pytest.raises(ExportError, match="max_triangles"):
        _ = export_model(_model(1), ExportConfig(max_triangles=1))
    for mapping in (
        {"linear_deflection_mm": float("nan")},
        {"angular_deflection_rad": 0},
        {"max_triangles": True},
        {"unknown": 1},
    ):
        with pytest.raises(ValueError):
            _ = ExportConfig.from_mapping(mapping)


def test_unrepresentable_integer_deflections_raise_value_error() -> None:
    for field in ("linear_deflection_mm", "angular_deflection_rad"):
        with pytest.raises(ValueError, match="finite and positive"):
            _ = ExportConfig.from_mapping({field: 10**1000})
    with pytest.raises(ValueError, match="finite and positive"):
        _ = ExportConfig(linear_deflection_mm=10**1000)
    with pytest.raises(ValueError, match="finite and positive"):
        _ = ExportConfig(angular_deflection_rad=10**1000)


def test_truncated_serialized_stl_is_rejected() -> None:
    with pytest.raises(ExportError, match="triangle count"):
        _ = _read_stl(_stl(_tetra())[:-1])
