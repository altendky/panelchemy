"""Native STEP assemblies preserve material grouping, colors and coordinates."""

from dataclasses import replace
from io import BytesIO
from typing import cast, override

import numpy as np
import pytest
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Trsf, gp_Vec
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.Interface import Interface_Static
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Shape
from PIL import Image
from shapely.geometry import Polygon, box

from panelchemy import export as mesh_export
from panelchemy import occt, step_export, step_topology
from panelchemy.construction import ConstructionProfile
from panelchemy.export import ExportError
from panelchemy.relief import ReliefModel, build_relief
from panelchemy.separation import Palette, SeparationConfig, separate
from panelchemy.step_export import export_step


def _model(colors: int = 3) -> ReliefModel:
    labels = np.zeros((16, 22), dtype=np.uint8)
    if colors >= 2:
        labels[3:6, 3:6] = 1
    if colors == 3:
        labels[8:11, 7:10] = 2
        labels[8:11, 14:17] = 2  # Two disconnected islands in one band.
    palette = np.asarray(SeparationConfig().palette, dtype=np.uint8)
    return build_relief(
        separate(
            Image.fromarray(palette[labels]),
            silhouette=np.ones(labels.shape, dtype=np.bool_),
        ),
        ConstructionProfile(long_edge_mm=22, taper_angle_deg=10),
    )


def _snapshots(model: ReliefModel) -> list[bytes]:
    result: list[bytes] = []
    for shape in [model.solid, *model.materials.values()]:
        stream = BytesIO()
        BRepTools.Write_s(shape, stream)
        result.append(stream.getvalue())
    return result


@pytest.mark.parametrize("colors", [1, 2, 3])
def test_named_colored_assembly_roundtrip_without_meshing(
    colors: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_mesh(*_args: object) -> None:
        pytest.fail("STEP-only export invoked mesh construction")

    monkeypatch.setattr(mesh_export, "_partition", no_mesh)
    model = _model(colors)
    before = _snapshots(model)
    palette: Palette = ((12, 23, 34), (210, 98, 45), (240, 249, 255))
    bundle = export_step(model, palette=palette)
    assert set(bundle.files) == {"relief.step", "relief-multipart.step"}
    report = cast("dict[str, object]", bundle.validation["multipart_step"])
    parts = cast("dict[str, dict[str, object]]", report["materials"])
    names = ("black", "orange", "white")[:colors]
    assert set(parts) == set(names)
    assert report["component_count"] == colors
    assert report["units"] == "millimeter"
    for i, name in enumerate(names):
        assert parts[name]["color_rgb"] == list(palette[i])
        assert parts[name]["volume_mm3"] == pytest.approx(
            occt.volume(model.materials[name]), rel=1e-8
        )
        bounds = cast("list[list[float]]", parts[name]["bounds_mm"])
        assert bounds[0][2] == pytest.approx((0, 2, 3)[i], abs=1e-6)
        assert bounds[1][2] == pytest.approx((2, 3, 4)[i], abs=1e-6)
    if colors == 3:
        assert parts["white"]["solid_count"] == 2
    assert bundle.validation["mesh_validated"] is False
    assert _snapshots(model) == before


def test_custom_materials_are_checked_against_source() -> None:
    model = replace(_model(), material_partition=None)
    assert export_step(model).validation["material_partition_method"] == (
        "checked-supplied-materials"
    )
    model.materials["white"] = model.materials["black"]
    with pytest.raises(ExportError, match="material boundary"):
        _ = export_step(model)


def test_valid_step_contacts_do_not_trigger_unneeded_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.zeros((16, 22), dtype=np.uint8)
    labels[3:13, 3:19] = 2
    labels[6:8, 7:9] = 0
    labels[8:10, 9:11] = 0  # Two counter boundaries meet at one native vertex.
    palette = np.asarray(SeparationConfig().palette, dtype=np.uint8)
    model = build_relief(
        separate(
            Image.fromarray(palette[labels]),
            silhouette=np.ones(labels.shape, dtype=np.bool_),
        ),
        ConstructionProfile(long_edge_mm=22, taper_angle_deg=10),
    )

    def unnecessary(*_args: object, **_kwargs: object) -> None:
        pytest.fail("A successful ordinary STEP must retain its original topology")

    monkeypatch.setattr(step_topology, "normalize_step_topology", unnecessary)
    bundle = export_step(model)
    fused = cast("dict[str, object]", bundle.validation["step"])
    multipart = cast("dict[str, object]", bundle.validation["multipart_step"])
    assert fused["native_topology_preparation"] == []
    assert multipart["native_topology_preparation"] == []


def test_changed_retained_materials_are_rejected() -> None:
    model = _model()
    model.materials["orange"] = model.materials["black"]
    with pytest.raises(ExportError, match="material boundary"):
        _ = export_step(model)


def test_custom_materials_cannot_move_a_counter_below_volume_budget() -> None:
    model = _model(2)
    panel = box(0, 0, 20, 20)
    black = occt.prism(panel, 0, 2)
    original_outline = panel.difference(box(5, 5, 5.001, 5.001))
    shifted_outline = panel.difference(box(10, 5, 10.001, 5.001))
    assert isinstance(original_outline, Polygon)
    assert isinstance(shifted_outline, Polygon)
    original = occt.prism(original_outline, 2, 3)
    shifted = occt.prism(shifted_outline, 2, 3)
    model = replace(
        model,
        solid=occt.fuse([black, original]),
        materials={"black": black, "orange": shifted},
        material_partition=None,
    )
    assert occt.volume(original) == pytest.approx(occt.volume(shifted), abs=1e-7)
    with pytest.raises(ExportError, match="do not cover"):
        _ = export_step(model)


def test_multipart_writer_restores_external_unit_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Initialize OCCT's STEP statics, then exercise the assembly writer itself.
    _ = STEPCAFControl_Writer()
    previous = Interface_Static.CVal_s("write.step.unit")
    assert Interface_Static.SetCVal_s("write.step.unit", "INCH")

    class BrokenWriter(STEPCAFControl_Writer):
        @override
        def WriteStream(self, theStream: BytesIO) -> IFSelect_ReturnStatus:
            raise RuntimeError("synthetic write failure")

    model = _model(1)
    monkeypatch.setattr(step_export, "STEPCAFControl_Writer", BrokenWriter)
    try:
        with pytest.raises(RuntimeError, match="synthetic write failure"):
            _ = step_export._multipart(  # pyright: ignore[reportPrivateUsage]
                model.materials, SeparationConfig().palette
            )
        assert Interface_Static.CVal_s("write.step.unit") == "INCH"
    finally:
        assert Interface_Static.SetCVal_s("write.step.unit", previous)


def test_roundtrip_validator_rejects_changed_colors_and_registration() -> None:
    model = _model()
    bundle = export_step(model)
    changed_palette: Palette = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    with pytest.raises(ExportError, match="colors"):
        _ = step_export._read_assembly(  # pyright: ignore[reportPrivateUsage]
            bundle.files["relief-multipart.step"], model.materials, changed_palette
        )
    swapped: dict[str, TopoDS_Shape] = {
        **model.materials,
        "white": model.materials["orange"],
    }
    with pytest.raises(ExportError, match=r"solids|volume|registration"):
        _ = step_export._read_assembly(  # pyright: ignore[reportPrivateUsage]
            bundle.files["relief-multipart.step"], swapped, SeparationConfig().palette
        )


def test_serialized_component_translation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(1)
    translated = gp_Trsf()
    translated.SetTranslation(gp_Vec(0, 0, 0.5))
    monkeypatch.setattr(
        step_export, "TopLoc_Location", lambda: TopLoc_Location(translated)
    )
    with pytest.raises(ExportError, match="placement"):
        _ = export_step(model)
