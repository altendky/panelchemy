"""Private precision conditioning preserves geometry and strict partition gates."""

# pyright: reportPrivateUsage=false

from math import radians

import pytest
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.ShapeFix import ShapeFix_ShapeTolerance
from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape
from shapely.geometry import box

from panelchemy import material_partition as mp
from panelchemy import occt
from panelchemy.construction import ConstructionProfile


def inflated_source() -> TopoDS_Shape:
    source = occt.prism(box(0, 0, 4, 4), 0, 4)
    edges = TopExp_Explorer(source, TopAbs_EDGE)
    ShapeFix_ShapeTolerance().SetTolerance(edges.Current(), 0.013, TopAbs_EDGE)
    return source


def test_conditioning_is_private_and_changes_only_edge_precision() -> None:
    source = inflated_source()
    before = mp.shape_digest(source)
    precision = mp._precisions(source)
    result = mp._condition_edge_precisions(source)
    assert result is not None
    copied, diagnostics = result
    assert not copied.IsSame(source)
    assert BRepCheck_Analyzer(copied, True, False, True).IsValid()
    assert mp._precisions(copied)[:2] == precision[:2]
    assert max(mp._precisions(copied)[2]) == 1e-5
    assert mp._geometry_digest(copied) == mp._geometry_digest(source)
    assert occt.volume(copied) == occt.volume(source)
    assert diagnostics["changed_edges"] == 1
    assert mp.shape_digest(source) == before


def test_native_narrow_cone_split_retries_without_merging_nearby_rulings() -> None:
    cone = BRepPrimAPI_MakeCone(
        gp_Ax2(gp_Pnt(0, 0, 2), gp_Dir(0, 0, 1)), 0.35, 0, 2, radians(3)
    ).Shape()
    source = occt.fuse([occt.prism(box(-1, -1, 1, 1), 0, 2), cone])
    edges = TopExp_Explorer(source, TopAbs_EDGE)
    while edges.More():
        edge = TopoDS.Edge(edges.Current())
        edges.Next()
        if BRep_Tool.Degenerated_s(edge):
            continue
        curve = BRepAdaptor_Curve(edge)
        heights = (
            curve.Value(curve.FirstParameter()).Z(),
            curve.Value(curve.LastParameter()).Z(),
        )
        if min(heights) > 1.99 and abs(heights[0] - heights[1]) > 1.9:
            break
    else:
        pytest.fail("cone ruling not found")
    precision = ShapeFix_ShapeTolerance()
    precision.SetTolerance(edge, 0.013, TopAbs_EDGE)
    for vertex in (
        TopExp.FirstVertex_s(edge, False),
        TopExp.LastVertex_s(edge, False),
    ):
        precision.SetTolerance(vertex, 0.013, TopAbs_VERTEX)
    before = mp.shape_digest(source)
    profile = ConstructionProfile(
        black_thickness_mm=2, orange_thickness_mm=1, white_thickness_mm=1
    )
    names = ("black", "orange", "white")
    bands = mp._bands(profile, names)
    with pytest.raises(mp._BandBoundaryFailure):
        _ = mp._audit(
            mp._split_bands(BRepBuilderAPI_Copy(source, True, False).Shape(), bands),
            bands,
            occt.volume(source),
        )
    proof = mp.build_partition(source, profile, names)
    assert "edge_precision_retry" in proof.diagnostics
    assert proof.diagnostics["volume_accounting_difference_mm3"] == pytest.approx(
        0, abs=1e-7
    )
    assert proof.diagnostics["shared_face_counts"] == {
        "black/orange": 1,
        "orange/white": 1,
    }
    mp.validate_materials(mp.materials_for(proof), profile)
    assert mp.shape_digest(source) == before


def test_retry_retains_original_source_binding_and_shared_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = inflated_source()
    before = mp.shape_digest(source)
    original_audit = mp._audit
    calls = 0

    def first_boundary_failure(
        shape: TopoDS_Shape, bands: tuple[mp.Band, ...], volume: float
    ) -> tuple[tuple[mp.MaterialCell, ...], dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise mp._BandBoundaryFailure("partition cell crosses a material boundary")
        return original_audit(shape, bands, volume)

    monkeypatch.setattr(mp, "_audit", first_boundary_failure)
    profile = ConstructionProfile(
        black_thickness_mm=2, orange_thickness_mm=1, white_thickness_mm=1
    )
    proof = mp.build_partition(source, profile, ("black", "orange", "white"))
    assert calls == 2
    assert proof.source_sha256 == before == mp.shape_digest(source)
    assert "edge_precision_retry" in proof.diagnostics
    assert proof.diagnostics["shared_face_counts"] == {
        "black/orange": 1,
        "orange/white": 1,
    }
    mp.validate_materials(mp.materials_for(proof), profile)
    copied, _ = mp.copy_verified_partition(
        source, mp.materials_for(proof), profile, proof
    )
    assert occt.valid(copied)


@pytest.mark.parametrize("single_band", [False, True])
def test_unrelated_or_single_band_failure_does_not_condition(
    monkeypatch: pytest.MonkeyPatch, single_band: bool
) -> None:
    error = (
        mp._BandBoundaryFailure("crossing")
        if single_band
        else occt.KernelError("unrelated Boolean failure")
    )

    def fail(*_: object) -> None:
        raise error

    def forbidden(*_: object) -> None:
        pytest.fail("precision conditioning must not run")

    monkeypatch.setattr(mp, "_audit", fail)
    monkeypatch.setattr(mp, "_condition_edge_precisions", forbidden)
    names = ("black",) if single_band else ("black", "orange", "white")
    with pytest.raises(occt.KernelError) as caught:
        _ = mp.build_partition(inflated_source(), ConstructionProfile(), names)
    assert caught.value is error


def test_failed_exact_conditioning_preserves_original_boundary_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = inflated_source()
    before = mp.shape_digest(source)
    native_checker = BRepCheck_Analyzer
    checks = 0

    class RejectConditioned:
        def __init__(self, *_: object) -> None:
            pass

        def IsValid(self) -> bool:
            return False

    def check(shape: TopoDS_Shape, *args: bool) -> object:
        nonlocal checks
        checks += 1
        if checks == 2:
            return RejectConditioned()
        return native_checker(shape, *args)

    monkeypatch.setattr(mp, "BRepCheck_Analyzer", check)
    assert mp._condition_edge_precisions(source) is None
    assert checks == 2
    assert mp.shape_digest(source) == before


def test_strict_bounds_still_reject_even_after_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = mp._BandBoundaryFailure("partition cell crosses a material boundary")
    calls = 0

    def fail(*_: object) -> None:
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(mp, "_audit", fail)
    with pytest.raises(mp._BandBoundaryFailure) as caught:
        _ = mp.build_partition(
            inflated_source(), ConstructionProfile(), ("black", "orange", "white")
        )
    assert calls == 2
    assert caught.value is error


def test_normal_source_has_no_conditioning_or_retry() -> None:
    source = occt.prism(box(0, 0, 4, 4), 0, 4)
    assert mp._condition_edge_precisions(source) is None
    proof = mp.build_partition(
        source,
        ConstructionProfile(
            black_thickness_mm=2, orange_thickness_mm=1, white_thickness_mm=1
        ),
        ("black", "orange", "white"),
    )
    assert "edge_precision_retry" not in proof.diagnostics
