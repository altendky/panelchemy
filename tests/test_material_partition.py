"""Shared cell identity, sealed CAD provenance and custom-model compatibility."""

from dataclasses import dataclass, replace
from io import BytesIO
from itertools import pairwise

import numpy as np
import pytest
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape
from PIL import Image
from shapely.affinity import translate
from shapely.geometry import Polygon, box

from panelchemy import material_partition, occt
from panelchemy.construction import ConstructionProfile
from panelchemy.contours import PanelContours
from panelchemy.export import ExportError, export_model
from panelchemy.material_partition import (
    build_partition,
    copy_verified_partition,
    materials_for,
    shape_digest,
)
from panelchemy.relief import ReliefModel, build_relief
from panelchemy.separation import SeparationConfig, separate


def model(*, orange: bool = True, white: bool = True) -> ReliefModel:
    labels = np.zeros((12, 12), dtype=np.uint8)
    if orange:
        labels[3:9, 3:9] = 1
    if white:
        labels[4:8, 4:8] = 2
        labels[5, 5] = 0
    return build_relief(
        separate(
            Image.fromarray(
                np.asarray(SeparationConfig().palette, dtype=np.uint8)[labels]
            ),
            silhouette=np.ones(labels.shape, dtype=np.bool_),
        ),
        ConstructionProfile(long_edge_mm=12, taper_angle_deg=0),
    )


def snapshots(value: ReliefModel) -> list[bytes]:
    shapes = [value.solid, *value.materials.values()]
    if value.material_partition is not None:
        shapes.append(value.material_partition.shape)
    result: list[bytes] = []
    for shape in shapes:
        stream = BytesIO()
        BRepTools.Write_s(shape, stream)
        result.append(stream.getvalue())
    return result


@pytest.mark.parametrize("colors", [1, 2, 3])
def test_native_depth_partition_and_copy_share_opposite_interface_faces(
    colors: int,
) -> None:
    shapes = [occt.prism(box(0, 0, 8, 8), 0, 2)]
    if colors >= 2:
        shapes.append(occt.prism(box(1, 1, 7, 7), 2, 3))
    if colors == 3:
        shapes.append(occt.prism(box(2, 2, 3, 3), 3, 4))
    source = occt.fuse(shapes)
    profile = ConstructionProfile()
    names = ("black", "orange", "white")[:colors]
    proof = build_partition(source, profile, names)
    materials = materials_for(proof)
    source_before = shape_digest(source)
    partition_before = shape_digest(proof.shape)
    copied, copied_materials = copy_verified_partition(
        source, materials, profile, proof
    )
    assert occt.valid(copied)
    assert set(copied_materials) == set(names)
    assert sum(
        occt.volume(part) for part in copied_materials.values()
    ) == pytest.approx(occt.volume(source), rel=1e-8, abs=1e-7)
    assert proof.diagnostics["internal_plane_count"] == colors - 1
    for lower, upper in pairwise(names):
        faces: dict[str, list[TopoDS_Shape]] = {}
        for name in (lower, upper):
            explorer = TopExp_Explorer(copied_materials[name], TopAbs_FACE)
            faces[name] = []
            while explorer.More():
                faces[name].append(explorer.Current())
                explorer.Next()
        pairs = [(a, b) for a in faces[lower] for b in faces[upper] if a.IsSame(b)]
        assert pairs
        assert all(
            {a.Orientation(), b.Orientation()} == {TopAbs_FORWARD, TopAbs_REVERSED}
            for a, b in pairs
        )
    assert shape_digest(source) == source_before
    assert shape_digest(proof.shape) == partition_before


def test_white_only_artwork_retains_orange_support_and_both_interfaces() -> None:
    value = model(orange=False)
    assert not value.contours.orange
    assert set(value.materials) == {"black", "orange", "white"}
    assert value.material_partition is not None
    assert value.material_partition.diagnostics["internal_plane_count"] == 2


def test_export_preserves_retained_partition_and_tessellation_cache_snapshots() -> None:
    value = model()
    proof = value.material_partition
    assert proof is not None
    for shape in (value.solid, proof.shape):
        mesher = BRepMesh_IncrementalMesh(shape, 0.05, False, 0.3, False)
        assert mesher.IsDone()
    assert shape_digest(value.solid) == proof.source_sha256
    assert shape_digest(proof.shape) == proof.partition_sha256
    before = snapshots(value)
    bundle = export_model(value)
    assert snapshots(value) == before
    processing = bundle.validation["numerical_processing"]
    assert isinstance(processing, dict)
    assert processing["partition_method"] == "retained-native-partition"


def test_partition_binds_band_heights_without_binding_nozzle_metadata() -> None:
    value = model()
    proof = value.material_partition
    assert proof is not None
    copied, _ = copy_verified_partition(
        value.solid,
        value.materials,
        replace(value.profile, nozzle_diameter_mm=1.2),
        proof,
    )
    assert occt.valid(copied)
    with pytest.raises(occt.KernelError, match="bands"):
        _ = copy_verified_partition(
            value.solid,
            value.materials,
            replace(value.profile, orange_thickness_mm=2),
            proof,
        )


def test_equal_volume_moved_counter_cannot_replace_a_retained_material() -> None:
    value = model()
    white = value.contours.white[0]
    assert len(white.interiors) == 1
    moved = translate(Polygon(white.interiors[0]), xoff=1)
    replacement = Polygon(white.exterior.coords, [moved.exterior.coords])
    changed = occt.prism(replacement, 3, 4)
    assert occt.volume(changed) == pytest.approx(occt.volume(value.materials["white"]))
    value.materials["white"] = changed
    before = snapshots(value)
    with pytest.raises(ExportError, match="binding"):
        _ = export_model(value)
    assert snapshots(value) == before


@pytest.mark.parametrize("corruption", ["duplicate-cell", "loose-face"])
def test_extra_recorded_geometry_is_not_hidden_by_unique_identity_maps(
    corruption: str,
) -> None:
    value = model()
    proof = value.material_partition
    assert proof is not None
    cell = next(cell.shape for cell in proof.cells if cell.material == "black")
    builder = BRep_Builder()
    changed = TopoDS_Compound()
    builder.MakeCompound(changed)
    builder.Add(changed, cell)
    if corruption == "duplicate-cell":
        builder.Add(changed, cell)
    else:
        explorer = TopExp_Explorer(cell, TopAbs_FACE)
        builder.Add(changed, explorer.Current())
    value.materials["black"] = changed
    with pytest.raises(ExportError, match=r"binding|loose"):
        _ = export_model(value)


def test_empty_supplied_material_cannot_reuse_a_retained_partition() -> None:
    value = model()
    empty = TopoDS_Compound()
    BRep_Builder().MakeCompound(empty)
    value.materials["white"] = empty
    with pytest.raises(ExportError, match="positive"):
        _ = export_model(value)


@pytest.mark.parametrize(
    "corruption", ["source-geometry", "partition-precision", "proof-digest"]
)
def test_mutated_canonical_cad_or_proof_rejects_before_export(corruption: str) -> None:
    value = model()
    proof = value.material_partition
    assert proof is not None
    if corruption == "proof-digest":
        value.material_partition = replace(proof, partition_sha256="0" * 64)
    else:
        shape = value.solid if corruption == "source-geometry" else proof.shape
        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        vertex = TopoDS.Vertex(explorer.Current())
        builder = BRep_Builder()
        if corruption == "source-geometry":
            point = BRep_Tool.Pnt_s(vertex)
            builder.UpdateVertex(
                vertex,
                gp_Pnt(point.X() + 0.01, point.Y(), point.Z()),
                BRep_Tool.Tolerance_s(vertex),
            )
        else:
            builder.UpdateVertex(vertex, 0.001)
    before = snapshots(value)
    with pytest.raises(ExportError, match="proof"):
        _ = export_model(value)
    assert snapshots(value) == before


@dataclass
class CustomModel:
    profile: ConstructionProfile
    contours: PanelContours
    solid: TopoDS_Shape
    materials: dict[str, TopoDS_Shape]
    footprint_proxy_tolerance_mm: float


def test_structural_custom_model_without_proof_keeps_legacy_material_inputs() -> None:
    value = model(white=False)
    custom = CustomModel(
        value.profile,
        value.contours,
        value.solid,
        value.materials,
        value.footprint_proxy_tolerance_mm,
    )
    before = snapshots(value)
    bundle = export_model(custom)
    assert snapshots(value) == before
    processing = bundle.validation["numerical_processing"]
    assert isinstance(processing, dict)
    assert processing["partition_method"] == "legacy-general-fuse"


def test_invalid_actual_copy_fails_before_splitter_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = occt.prism(box(0, 0, 4, 4), 0, 2)
    empty = TopoDS_Compound()
    BRep_Builder().MakeCompound(empty)

    class InvalidCopy:
        def __init__(self, *_args: object) -> None:
            pass

        def Shape(self) -> TopoDS_Shape:
            return empty

    monkeypatch.setattr(material_partition, "BRepBuilderAPI_Copy", InvalidCopy)
    with pytest.raises(occt.KernelError, match="copied source"):
        _ = build_partition(source, ConstructionProfile(), ("black",))
