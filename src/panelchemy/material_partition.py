"""One shared native cell graph for horizontal material bands.

Materials retain the actual split cells. A sealed proof binds that graph and
the combined source CAD, so export can copy it once without reconstructing
coincident interfaces or silently ignoring changed material geometry.
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
from math import isclose, isfinite
from time import monotonic

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy, BRepBuilderAPI_MakeFace
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepTools import BRepTools
from OCP.collections import (
    IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
    List_TopoDS_Shape,
)
from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt
from OCP.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_COMPSOLID,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_REVERSED,
    TopAbs_SOLID,
)
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS_Compound, TopoDS_Iterator, TopoDS_Shape
from OCP.TopTools import TopTools_FormatVersion_VERSION_3

from panelchemy import occt
from panelchemy.construction import ConstructionProfile

_NAMES = ("black", "orange", "white")
type Band = tuple[str, float, float]


@dataclass(frozen=True)
class MaterialCell:
    material: str
    shape: TopoDS_Shape


@dataclass(frozen=True)
class MaterialPartition:
    shape: TopoDS_Shape
    cells: tuple[MaterialCell, ...]
    bands: tuple[Band, ...]
    source_volume_mm3: float
    source_sha256: str
    partition_sha256: str
    diagnostics: dict[str, object]


def shape_digest(shape: TopoDS_Shape) -> str:
    """Hash physical BRep data, including precision but excluding mesh caches."""
    stream = BytesIO()
    BRepTools.Write_s(shape, stream, False, False, TopTools_FormatVersion_VERSION_3)
    return sha256(stream.getvalue()).hexdigest()


def _bounds(
    shape: TopoDS_Shape,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    bounds = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, bounds, False, False)
    if bounds.IsVoid() or bounds.IsWhole():
        raise occt.KernelError("material CAD has no finite geometric bounds")
    low, high = bounds.CornerMin(), bounds.CornerMax()
    return (low.X(), low.Y(), low.Z()), (high.X(), high.Y(), high.Z())


def _bands(profile: ConstructionProfile, names: Sequence[str]) -> tuple[Band, ...]:
    if tuple(names) != _NAMES[: len(names)] or not names:
        raise occt.KernelError("expected supported black/orange/white material bands")
    return (
        ("black", 0.0, profile.black_top),
        ("orange", profile.black_top, profile.orange_top),
        ("white", profile.orange_top, profile.white_top),
    )[: len(names)]


def _solids(shape: TopoDS_Shape) -> tuple[TopoDS_Shape, ...]:
    """Keep occurrences and reject loose children, even duplicate loose faces."""
    kind = shape.ShapeType()
    if kind == TopAbs_SOLID:
        return (shape,)
    if kind not in (TopAbs_COMPOUND, TopAbs_COMPSOLID):
        raise occt.KernelError("material CAD contains loose non-solid geometry")
    result: list[TopoDS_Shape] = []
    children = TopoDS_Iterator(shape)
    while children.More():
        result.extend(_solids(children.Value()))
        children.Next()
    return tuple(result)


def _same_cells(
    actual: Sequence[TopoDS_Shape], expected: Sequence[TopoDS_Shape]
) -> bool:
    remaining = list(expected)
    for shape in actual:
        for index, candidate in enumerate(remaining):
            if shape.IsEqual(candidate):
                _ = remaining.pop(index)
                break
        else:
            return False
    return not remaining


def materials_for(partition: MaterialPartition) -> dict[str, TopoDS_Shape]:
    result: dict[str, TopoDS_Shape] = {}
    builder = BRep_Builder()
    for name, _, _ in partition.bands:
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for cell in partition.cells:
            if cell.material == name:
                builder.Add(compound, cell.shape)
        result[name] = compound
    return result


def validate_materials(
    materials: dict[str, TopoDS_Shape], profile: ConstructionProfile
) -> None:
    """Validate private copies before construction; use geometric band bounds."""
    bands = _bands(profile, [name for name in _NAMES if name in materials])
    if len(bands) != len(materials):
        raise occt.KernelError("unexpected material name")
    for name, bottom, top in bands:
        shape = materials[name]
        checker = BRepCheck_Analyzer(shape)
        cells = _solids(shape)
        if not cells or not checker.IsValid():
            raise occt.KernelError("material CAD must be valid with positive volume")
        for cell in cells:
            volume = occt.volume(cell)
            if not checker.IsValid(cell) or not isfinite(volume) or volume <= 0:
                raise occt.KernelError(
                    "material CAD must be valid with positive volume"
                )
            low, high = _bounds(cell)
            if low[2] < bottom - 1e-6 or high[2] > top + 1e-6:
                raise occt.KernelError("partition cell crosses a material boundary")


def _audit(
    shape: TopoDS_Shape, bands: tuple[Band, ...], source_volume: float
) -> tuple[tuple[MaterialCell, ...], dict[str, object]]:
    checker = BRepCheck_Analyzer(shape)
    if not checker.IsValid():
        raise occt.KernelError("shared material partition is invalid")
    cells: list[MaterialCell] = []
    volumes: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    faces = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    owners: dict[int, list[tuple[str, object]]] = defaultdict(list)
    seen = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    for solid in _solids(shape):
        if seen.FindIndex(solid):
            raise occt.KernelError("shared partition contains duplicate cells")
        _ = seen.Add(solid)
        volume = occt.volume(solid)
        if not checker.IsValid(solid) or not isfinite(volume) or volume <= 0:
            raise occt.KernelError("shared partition cell must be valid and positive")
        low, high = _bounds(solid)
        middle = (low[2] + high[2]) / 2
        matches = [
            (name, bottom, top) for name, bottom, top in bands if bottom < middle < top
        ]
        if len(matches) != 1:
            raise occt.KernelError(
                "partition cell cannot be assigned to a material band"
            )
        name, bottom, top = matches[0]
        if low[2] < bottom - 1e-6 or high[2] > top + 1e-6:
            raise occt.KernelError("partition cell crosses a material boundary")
        cells.append(MaterialCell(name, solid))
        counts[name] += 1
        volumes[name] += volume
        local = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
        TopExp.MapShapes_s(solid, TopAbs_FACE, local)
        for index in range(1, local.Extent() + 1):
            face = local.FindKey(index)
            owners[faces.Add(face)].append((name, face.Orientation()))
    if set(counts) != {name for name, _, _ in bands} or not isclose(
        sum(volumes.values()), source_volume, rel_tol=1e-8, abs_tol=1e-7
    ):
        raise occt.KernelError("material volumes do not account for the combined solid")
    shared: Counter[str] = Counter()
    boundaries = {
        frozenset((bands[i][0], bands[i + 1][0])): bands[i][2]
        for i in range(len(bands) - 1)
    }
    for index, adjacent in owners.items():
        if len(adjacent) == 1:
            continue
        if len(adjacent) != 2:
            raise occt.KernelError(
                "shared material face has unexpected owner multiplicity"
            )
        (first, a), (second, b) = adjacent
        pair = frozenset((first, second))
        if pair not in boundaries or {a, b} != {TopAbs_FORWARD, TopAbs_REVERSED}:
            raise occt.KernelError(
                "shared material face lacks opposite adjacent owners"
            )
        low, high = _bounds(faces.FindKey(index))
        if (
            abs(low[2] - boundaries[pair]) > 1e-6
            or abs(high[2] - boundaries[pair]) > 1e-6
        ):
            raise occt.KernelError("shared material face is outside its band plane")
        if occt.area(faces.FindKey(index)) <= 0:
            raise occt.KernelError("shared material face has no positive area")
        shared["/".join(sorted((first, second), key=_NAMES.index))] += 1
    if any(
        shared[f"{bands[i][0]}/{bands[i + 1][0]}"] == 0 for i in range(len(bands) - 1)
    ):
        raise occt.KernelError("material partition lacks native shared interfaces")
    return tuple(cells), {
        "cell_counts": dict(counts),
        "material_volumes_mm3": dict(volumes),
        "volume_accounting_difference_mm3": abs(sum(volumes.values()) - source_volume),
        "shared_face_counts": dict(shared),
    }


def seal_partition(
    source: TopoDS_Shape, partition: MaterialPartition
) -> MaterialPartition:
    """Seal after construction checks; later native mutations reject export."""
    return replace(
        partition,
        source_sha256=shape_digest(source),
        partition_sha256=shape_digest(partition.shape),
    )


def build_partition(
    source: TopoDS_Shape, profile: ConstructionProfile, names: Sequence[str]
) -> MaterialPartition:
    started = monotonic()
    bands = _bands(profile, names)
    copied = BRepBuilderAPI_Copy(source, True, False).Shape()
    if not occt.valid(copied) or len(_solids(copied)) != 1:
        raise occt.KernelError("copied source relief must be one valid positive solid")
    source_volume = occt.volume(copied)
    if not isfinite(source_volume) or source_volume <= 0:
        raise occt.KernelError("copied source relief must be one valid positive solid")
    shape = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(shape)
    builder.Add(shape, copied)
    if len(bands) > 1:
        low, high = _bounds(copied)
        arguments, tools = List_TopoDS_Shape(), List_TopoDS_Shape()
        _ = arguments.Append(copied)
        for _, _, height in bands[:-1]:
            plane = gp_Pln(
                gp_Ax3(gp_Pnt(0, 0, height), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))
            )
            _ = tools.Append(
                BRepBuilderAPI_MakeFace(
                    plane, low[0] - 2, high[0] + 2, low[1] - 2, high[1] + 2
                ).Face()
            )
        operation = BRepAlgoAPI_Splitter()
        operation.SetArguments(arguments)
        operation.SetTools(tools)
        operation.SetNonDestructive(True)
        operation.SetRunParallel(False)
        operation.Build()
        if not operation.IsDone():
            raise occt.KernelError(
                "OCCT could not split the source into material bands"
            )
        shape = operation.Shape()
    cells, diagnostics = _audit(shape, bands, source_volume)
    diagnostics.update(
        method="single-source-depth-split" if len(bands) > 1 else "single-band-copy",
        elapsed_seconds=monotonic() - started,
        internal_plane_count=len(bands) - 1,
        xy_tool_padding_mm=2.0,
    )
    return seal_partition(
        source,
        MaterialPartition(shape, cells, bands, source_volume, "", "", diagnostics),
    )


def copy_verified_partition(
    source: TopoDS_Shape,
    materials: dict[str, TopoDS_Shape],
    profile: ConstructionProfile,
    proof: MaterialPartition,
) -> tuple[TopoDS_Shape, dict[str, TopoDS_Shape]]:
    """Verify canonical bindings, then copy one shared graph for export work."""
    names = [name for name in _NAMES if name in materials]
    if len(names) != len(materials) or proof.bands != _bands(profile, names):
        raise occt.KernelError(
            "retained material partition bands do not match the model"
        )
    if not _same_cells(_solids(proof.shape), [cell.shape for cell in proof.cells]):
        raise occt.KernelError("retained material partition cell proof is invalid")
    matches = all(
        _same_cells(
            _solids(materials[name]),
            [cell.shape for cell in proof.cells if cell.material == name],
        )
        for name in names
    )
    if not matches:
        # Preserve the established crossing diagnosis, without querying or
        # modifying the caller's unbound material geometry.
        validate_materials(
            {
                name: BRepBuilderAPI_Copy(shape, True, False).Shape()
                for name, shape in materials.items()
            },
            profile,
        )
        raise occt.KernelError(
            "supplied material cells no longer match the retained partition binding"
        )
    if shape_digest(source) != proof.source_sha256:
        raise occt.KernelError(
            "source relief no longer matches its retained partition proof"
        )
    if shape_digest(proof.shape) != proof.partition_sha256:
        raise occt.KernelError(
            "material geometry no longer matches its retained partition proof"
        )
    copier = BRepBuilderAPI_Copy(proof.shape, True, False)
    copied = copier.Shape()
    cells, diagnostics = _audit(copied, proof.bands, proof.source_volume_mm3)
    for cell in proof.cells:
        mapped = copier.ModifiedShape(cell.shape)
        if not any(
            candidate.material == cell.material and candidate.shape.IsEqual(mapped)
            for candidate in cells
        ):
            raise occt.KernelError(
                "copied material cell lost its native partition binding"
            )
    copied_proof = MaterialPartition(
        copied, cells, proof.bands, proof.source_volume_mm3, "", "", diagnostics
    )
    return copied, materials_for(copied_proof)
