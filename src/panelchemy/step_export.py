"""Named, colored native material assemblies, independent of mesh export."""

from collections.abc import Callable
from io import BytesIO
from math import isclose, isfinite
from typing import Protocol, cast

from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.collections import Sequence_TCollection_AsciiString, Sequence_TDF_Label
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.Quantity import Quantity_Color, Quantity_TOC_sRGB
from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_SOLID
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Compound, TopoDS_Iterator, TopoDS_Shape
from OCP.XCAFDoc import (
    XCAFDoc_ColorGen,
    XCAFDoc_ColorSurf,
    XCAFDoc_DocumentTool,
    XCAFDoc_ShapeTool,
)

from panelchemy import material_partition, occt, step_topology
from panelchemy.construction import ConstructionProfile
from panelchemy.export import (
    _STEP_LOCK,  # pyright: ignore[reportPrivateUsage]
    ExportBundle,
    ExportError,
    StepTopologyError,
    _bounds,  # pyright: ignore[reportPrivateUsage]
    _step,  # pyright: ignore[reportPrivateUsage]
)
from panelchemy.separation import Palette, SeparationConfig

_NAMES = ("black", "orange", "white")


class StepModel(Protocol):
    """Native fields required by STEP export; no tessellation is required."""

    profile: ConstructionProfile
    solid: TopoDS_Shape
    materials: dict[str, TopoDS_Shape]


def _solids(shape: TopoDS_Shape) -> list[TopoDS_Shape]:
    if shape.ShapeType() == TopAbs_SOLID:
        return [shape]
    if shape.ShapeType() not in (TopAbs_COMPOUND, TopAbs_COMPSOLID):
        raise ExportError("STEP material contains loose non-solid geometry")
    result: list[TopoDS_Shape] = []
    children = TopoDS_Iterator(shape)
    while children.More():
        result.extend(_solids(children.Value()))
        children.Next()
    return result


def _compound(shapes: list[TopoDS_Shape]) -> TopoDS_Compound:
    result = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(result)
    for shape in shapes:
        builder.Add(result, shape)
    return result


def _copy_materials(model: StepModel) -> tuple[dict[str, TopoDS_Shape], str]:
    proof: object = getattr(model, "material_partition", None)
    if proof is not None:
        if not isinstance(proof, material_partition.MaterialPartition):
            raise ExportError("invalid retained material partition proof")
        _, materials = material_partition.copy_verified_partition(
            model.solid, model.materials, model.profile, proof
        )
        return materials, "retained-native-partition"
    # Custom models must prove their supplied bands cover exactly the source.
    # Equal total volume alone would allow a displaced or overlapping band.
    materials = {
        name: BRepBuilderAPI_Copy(shape, True, False).Shape()
        for name, shape in model.materials.items()
    }
    material_partition.validate_materials(materials, model.profile)
    source = BRepBuilderAPI_Copy(model.solid, True, False).Shape()
    supplied = _compound(list(materials.values()))
    for difference in (occt.cut(source, supplied), occt.cut(supplied, source)):
        if _solids(difference):
            raise ExportError("STEP material bands do not cover the source relief")
    return materials, "checked-supplied-materials"


def _name(label: TDF_Label) -> str:
    attribute = TDataStd_Name()
    if not label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
        raise ExportError("STEP roundtrip lost a component name")
    return attribute.Get().ToExtString()


def _check_shape(original: TopoDS_Shape, restored: TopoDS_Shape) -> dict[str, object]:
    cells = _solids(restored)
    if not cells or len(cells) != len(_solids(original)):
        raise ExportError("STEP roundtrip changed the number of material solids")
    checker = BRepCheck_Analyzer(restored)
    if not checker.IsValid():
        points = step_topology.open_wire_contact_points(restored, checker)
        if points:
            raise StepTopologyError(
                "STEP roundtrip has open wires at coincident endpoints", points
            )
        raise ExportError("STEP roundtrip changed or invalidated material solids")
    if any(not checker.IsValid(cell) or occt.volume(cell) <= 0 for cell in cells):
        raise ExportError("STEP roundtrip invalidated a material solid")
    before, after = occt.volume(original), occt.volume(restored)
    if not isfinite(after) or not isclose(before, after, rel_tol=1e-8, abs_tol=1e-7):
        raise ExportError("STEP roundtrip changed material volume")
    low, high = _bounds(restored)
    if any(
        abs(a - b) > 1e-6
        for first, second in zip(_bounds(original), (low, high), strict=True)
        for a, b in zip(first, second, strict=True)
    ):
        raise ExportError("STEP roundtrip changed material registration or dimensions")
    return {
        "valid": True,
        "solid_count": len(cells),
        "bounds_mm": [list(low), list(high)],
        "volume_mm3": after,
        "volume_difference_mm3": abs(before - after),
    }


def _read_assembly(
    data: bytes, materials: dict[str, TopoDS_Shape], palette: Palette
) -> dict[str, object]:
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    if reader.ReadStream("relief-multipart.step", BytesIO(data)) != IFSelect_RetDone:
        raise ExportError("serialized multipart STEP cannot be read back")
    lengths = Sequence_TCollection_AsciiString()
    reader.Reader().FileUnits(
        lengths,
        Sequence_TCollection_AsciiString(),
        Sequence_TCollection_AsciiString(),
    )
    units = [lengths.Value(i).ToCString() for i in range(1, lengths.Length() + 1)]
    if not units or any(
        unit.lower() not in {"millimetre", "millimeter", "mm"} for unit in units
    ):
        raise ExportError("multipart STEP length contexts are not millimeters")
    reader.ChangeReader().SetSystemLengthUnit(1.0)
    document = TDocStd_Document(TCollection_ExtendedString("BinXCAF"))
    if not reader.Transfer(document):
        raise ExportError("multipart STEP cannot transfer its CAD assembly")
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    colors = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    roots = Sequence_TDF_Label()
    shapes.GetFreeShapes(roots)
    components = Sequence_TDF_Label()
    if (
        roots.Length() != 1
        or _name(roots.Value(1)) != "Panelchemy relief"
        or not XCAFDoc_ShapeTool.GetComponents_s(roots.Value(1), components)
        or components.Length() != len(materials)
    ):
        raise ExportError("STEP roundtrip lost the material assembly structure")
    reports: dict[str, object] = {}
    for index in range(1, components.Length() + 1):
        instance = components.Value(index)
        definition = TDF_Label()
        if not XCAFDoc_ShapeTool.GetReferredShape_s(instance, definition):
            raise ExportError("STEP material component has no shape definition")
        if XCAFDoc_ShapeTool.IsAssembly_s(definition):
            raise ExportError(
                "STEP roundtrip split a material into assembly components"
            )
        name = _name(instance)
        if name not in materials or name in reports or _name(definition) != name:
            raise ExportError("STEP roundtrip changed material names")
        transform = XCAFDoc_ShapeTool.GetLocation_s(instance).Transformation()
        if any(
            abs(transform.Value(row, column) - float(row == column)) > 1e-12
            for row in range(1, 4)
            for column in range(1, 5)
        ):
            raise ExportError("STEP roundtrip changed component placement")
        restored = XCAFDoc_ShapeTool.GetShape_s(instance)
        rgb = palette[_NAMES.index(name)]
        # STEPCAF may distribute a compound's style onto its solid sublabels.
        # Check every island rather than requiring a compound-root color label.
        for cell in _solids(XCAFDoc_ShapeTool.GetShape_s(definition)):
            color = Quantity_Color()
            if not colors.GetColor(cell, XCAFDoc_ColorSurf, color) or any(
                abs(actual - expected / 255) > 1e-6
                for actual, expected in zip(
                    color.Values(Quantity_TOC_sRGB), rgb, strict=True
                )
            ):
                raise ExportError("STEP roundtrip changed material colors")
        try:
            report = _check_shape(materials[name], restored)
        except StepTopologyError as error:
            raise StepTopologyError(
                f"{name}: {error}", error.contact_points, name
            ) from error
        except ExportError as error:
            raise ExportError(f"{name}: {error}") from error
        report["color_rgb"] = list(rgb)
        report["placement"] = "identity; original model coordinates"
        reports[name] = report
    return {
        "valid": True,
        "assembly_name": "Panelchemy relief",
        "component_count": len(reports),
        "materials": reports,
        "units": "millimeter",
        "serialized_length_units": units,
        "timestamp_may_vary": True,
        "bambu_import_tested": False,
    }


def _multipart_once(
    materials: dict[str, TopoDS_Shape], palette: Palette
) -> tuple[bytes, dict[str, object]]:
    document = TDocStd_Document(TCollection_ExtendedString("BinXCAF"))
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    colors = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    root = shapes.NewShape()
    _ = TDataStd_Name.Set_s(root, TCollection_ExtendedString("Panelchemy relief"))
    for name in _NAMES:
        if name not in materials:
            continue
        # A single part definition can contain disconnected islands. They remain
        # grouped by band rather than becoming one top-level part per island.
        prepared = BRepBuilderAPI_Copy(materials[name], True, False).Shape()
        label = shapes.AddShape(prepared, False)
        _ = TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
        rgb = palette[_NAMES.index(name)]
        color = Quantity_Color(
            rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, Quantity_TOC_sRGB
        )
        for kind in (XCAFDoc_ColorGen, XCAFDoc_ColorSurf):
            colors.SetColor(label, color, kind)
        instance = shapes.AddComponent(root, label, TopLoc_Location())
        _ = TDataStd_Name.Set_s(instance, TCollection_ExtendedString(name))
    shapes.UpdateAssemblies()
    stream = BytesIO()
    # Coordinate with the fused writer: both consult OCCT's global STEP units.
    with _STEP_LOCK:
        writer = STEPCAFControl_Writer()
        writer.SetColorMode(True)
        writer.SetNameMode(True)
        previous = Interface_Static.CVal_s("write.step.unit")
        if not Interface_Static.SetCVal_s("write.step.unit", "MM"):
            raise ExportError("OCCT could not select millimeters for multipart STEP")
        try:
            writer.ChangeWriter().Model().SetLocalLengthUnit(1.0)
            transfer = cast("Callable[..., bool]", writer.Transfer)
            if not transfer(document, STEPControl_AsIs):
                raise ExportError("multipart STEP transfer failed")
            if writer.WriteStream(stream) != IFSelect_RetDone:
                raise ExportError("multipart STEP serialization failed")
        finally:
            if not Interface_Static.SetCVal_s("write.step.unit", previous):
                raise ExportError("OCCT could not restore the STEP output unit")
    data = stream.getvalue()
    return data, _read_assembly(data, materials, palette)


def _multipart(
    materials: dict[str, TopoDS_Shape], palette: Palette
) -> tuple[bytes, dict[str, object]]:
    """Correct only a material/contact implicated by an actual STEP failure."""
    working = dict(materials)
    preparations: list[dict[str, object]] = []
    for attempt in range(9):
        try:
            data, report = _multipart_once(working, palette)
        except StepTopologyError as error:
            if attempt == 8:
                raise ExportError(
                    "multipart STEP contact correction exceeded its retry bound"
                ) from error
            name = error.material
            if name is None or name not in working:
                raise ExportError(
                    "STEP failure could not be bound to a material"
                ) from error
            corrected, preparation = step_topology.normalize_step_topology(
                working[name], contact_points=error.contact_points
            )
            if not preparation["splits"]:
                raise ExportError(
                    f"{name}: STEP open-wire failure has no certified inner-hole contact correction"
                ) from error
            working[name] = corrected
            preparations.append({"material": name, **preparation})
        else:
            report["native_topology_preparation"] = preparations
            return data, report
    raise AssertionError("unreachable multipart STEP retry state")


def export_step(model: StepModel, *, palette: Palette | None = None) -> ExportBundle:
    """Return fused and colored multipart STEP after native/serialized checks.

    No STL construction or mesh validation is performed. Material solids retain
    their shared origin, scale and band heights, including disconnected islands.
    """
    palette = (
        SeparationConfig(palette=palette).palette
        if palette is not None
        else SeparationConfig().palette
    )
    try:
        source = BRepBuilderAPI_Copy(model.solid, True, False).Shape()
        if not occt.valid(source) or len(_solids(source)) != 1:
            raise ExportError("source relief CAD must be one valid positive solid")
        volume = occt.volume(source)
        if not isfinite(volume) or volume <= 0:
            raise ExportError("source relief CAD must be one valid positive solid")
        materials, method = _copy_materials(model)
        total = sum(occt.volume(shape) for shape in materials.values())
        if not isclose(total, volume, rel_tol=1e-8, abs_tol=1e-7):
            raise ExportError("material volumes do not account for the combined solid")
        fused, fused_report = _step(source)
        multipart, multipart_report = _multipart(materials, palette)
    except (RuntimeError, occt.KernelError) as error:
        raise ExportError(f"OCCT STEP export failed: {error}") from error
    return ExportBundle(
        {"relief.step": fused, "relief-multipart.step": multipart},
        {
            "step": fused_report,
            "multipart_step": multipart_report,
            "material_partition_method": method,
            "source_volume_mm3": volume,
            "material_volume_difference_mm3": abs(total - volume),
            "mesh_validated": False,
        },
    )
