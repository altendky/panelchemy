"""Serialized registration, material contacts and explicit invalid-mesh failures."""

from io import BytesIO
from typing import cast, override

import numpy as np
import pytest
from OCP.BRepTools import BRepTools
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_Writer
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
    _Mesh,  # pyright: ignore[reportPrivateUsage]
    _read_stl,  # pyright: ignore[reportPrivateUsage]
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


def _tetra(offset: tuple[float, float, float] = (0, 0, 0)) -> _Mesh:
    points = [
        (float(x + offset[0]), float(y + offset[1]), float(z + offset[2]))
        for x, y, z in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    ]
    return _Mesh(points, [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)])


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
