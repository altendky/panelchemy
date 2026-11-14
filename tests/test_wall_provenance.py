"""Wall ownership follows actual native generators without changing geometry."""

# pyright: reportPrivateUsage=false

import pytest
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Line
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopoDS import TopoDS, TopoDS_Edge, TopoDS_Face, TopoDS_Shape, TopoDS_Wire
from shapely.geometry import box

from panelchemy import curved_occt as n
from panelchemy import material_partition, native_sweep, occt
from panelchemy.curves import BezierSpan, CurveRing, polygon_lines


def circle() -> CurveRing:
    k = 0.5522847498307936
    return CurveRing(
        tuple(
            BezierSpan(points, False)
            for points in (
                ((1, 0), (1, k), (k, 1), (0, 1)),
                ((0, 1), (-k, 1), (-1, k), (-1, 0)),
                ((-1, 0), (-1, -k), (-k, -1), (0, -1)),
                ((0, -1), (k, -1), (1, -k), (1, 0)),
            )
        )
    )


def digests(shapes: list[TopoDS_Shape]) -> list[str]:
    return [material_partition.shape_digest(shape) for shape in shapes]


def check_ownership(
    walls: list[TopoDS_Shape],
    generators: list[tuple[TopoDS_Edge, TopoDS_Face]],
) -> None:
    assert generators
    for edge, face in generators:
        owner = next(wall for wall in walls if face.IsSame(wall))
        assert face.Orientation() == owner.Orientation()
        assert face.Location().IsEqual(owner.Location())
        curve = BRepAdaptor_Curve(edge)
        assert curve.Value(curve.FirstParameter()).Z() == 4
        assert curve.Value(curve.LastParameter()).Z() == 4


@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("precise", [False, True])
def test_whole_wire_metadata_retains_real_walls_and_default_geometry(
    internal: bool, precise: bool
) -> None:
    ring = circle()
    original = tuple(span.control_points for span in ring.spans)
    baseline = n._unjoined_sheets(ring, 2, 4, 0.1, internal, precise_draft=precise)
    generators: list[tuple[TopoDS_Edge, TopoDS_Face]] = []
    walls = n._unjoined_sheets(
        ring, 2, 4, 0.1, internal, precise_draft=precise, generators=generators
    )
    assert digests(walls) == digests(baseline)
    assert len(generators) == len(ring.spans)
    check_ownership(walls, generators)
    assert tuple(span.control_points for span in ring.spans) == original


def test_independent_curves_and_analytic_line_keep_actual_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    line = BezierSpan(((-1, 0), (-1, 0), (1, 0), (1, 0)), True)
    ring = CurveRing((*circle().spans[:2], line))
    original_verified = n._verified_draft
    successful: list[tuple[TopoDS_Edge, n._Draft]] = []

    def independent(
        wire: TopoDS_Wire,
        z0: float,
        z1: float,
        slope: float,
        internal: bool,
        diagnostics: dict[str, object] | None,
        *,
        precise_draft: bool = False,
    ) -> n._Draft:
        if occt.topology_counts(wire)["edges"] > 1:
            raise n._DraftArrayFailure("force independent native generators")
        draft = original_verified(
            wire, z0, z1, slope, internal, diagnostics, precise_draft=precise_draft
        )
        successful.append((BRepTools_WireExplorer(wire).Current(), draft))
        return draft

    monkeypatch.setattr(n, "_verified_draft", independent)
    baseline = n._unjoined_sheets(ring, 2, 4, 0.1, False, precise_draft=True)
    successful.clear()
    generators: list[tuple[TopoDS_Edge, TopoDS_Face]] = []
    walls = n._unjoined_sheets(
        ring, 2, 4, 0.1, False, precise_draft=True, generators=generators
    )
    assert digests(walls) == digests(baseline)
    check_ownership(walls, generators)
    assert len(generators) == 3
    for source, draft in successful:
        assert isinstance(draft, native_sweep.Draft)
        expected = [
            face
            for shape in draft.Generated(source)
            for face in n._entities(shape, TopAbs_FACE)
        ]
        assert all(
            any(edge.IsSame(source) and face.IsSame(owner) for edge, face in generators)
            for owner in expected
        )
    line_records = [
        (edge, face)
        for edge, face in generators
        if BRepAdaptor_Curve(edge).GetType() == GeomAbs_Line
    ]
    assert len(line_records) == 1
    edge, face = line_records[0]
    assert any(edge.IsSame(item) for item in n._entities(face, TopAbs_EDGE))


def test_planar_walls_record_their_own_exact_top_edges() -> None:
    ring = polygon_lines(box(0, 0, 2, 1)).exterior
    baseline = n._unjoined_sheets(ring, 2, 4, 0.1, False)
    prefix = (TopoDS_Edge(), TopoDS_Face())
    generators: list[tuple[TopoDS_Edge, TopoDS_Face]] = [prefix]
    walls = n._unjoined_sheets(ring, 2, 4, 0.1, False, generators=generators)
    assert digests(walls) == digests(baseline)
    assert generators[0] is prefix
    assert len(generators) == 5
    check_ownership(walls, generators[1:])
    for edge, face in generators[1:]:
        line = BRepAdaptor_Curve(edge).Line()
        assert line.Direction().Z() == 0 and line.Location().Z() == 4
        assert any(edge.IsSame(item) for item in n._entities(face, TopAbs_EDGE))


def test_late_convex_cone_failure_leaves_external_collector_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring = polygon_lines(box(0, 0, 2, 1)).exterior
    original_valid = occt.valid

    def invalid_cone(shape: TopoDS_Shape) -> bool:
        if BRepAdaptor_Surface(TopoDS.Face(shape)).GetType() == GeomAbs_Cone:
            return False
        return original_valid(shape)

    monkeypatch.setattr(occt, "valid", invalid_cone)
    prefix = (TopoDS_Edge(), TopoDS_Face())
    generators: list[tuple[TopoDS_Edge, TopoDS_Face]] = [prefix]
    with pytest.raises(occt.KernelError, match="convex taper corner is invalid"):
        _ = n._unjoined_sheets(ring, 2, 4, 0.1, False, generators=generators)
    assert len(generators) == 1 and generators[0] is prefix


@pytest.mark.parametrize("missing", [False, True])
def test_unrelated_or_missing_generated_faces_are_rejected_atomically(
    monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    original_verified = n._verified_draft
    unrelated = BRepBuilderAPI_MakeFace(gp_Pln(), 0, 1, 0, 1).Face()

    def wrong_ownership(
        wire: TopoDS_Wire,
        z0: float,
        z1: float,
        slope: float,
        internal: bool,
        diagnostics: dict[str, object] | None,
        *,
        precise_draft: bool = False,
    ) -> native_sweep.Draft:
        draft = original_verified(
            wire, z0, z1, slope, internal, diagnostics, precise_draft=precise_draft
        )
        generated: list[tuple[TopoDS_Edge, TopoDS_Shape]] = []
        edges = BRepTools_WireExplorer(wire)
        while edges.More():
            generated.append((edges.Current(), unrelated))
            edges.Next()
        return native_sweep.Draft(draft.Shell(), () if missing else tuple(generated))

    monkeypatch.setattr(n, "_verified_draft", wrong_ownership)
    assert n._unjoined_sheets(circle(), 2, 4, 0.1, False)
    prefix = (TopoDS_Edge(), TopoDS_Face())
    generators: list[tuple[TopoDS_Edge, TopoDS_Face]] = [prefix]
    with pytest.raises(occt.KernelError, match="ownership disagrees"):
        _ = n._unjoined_sheets(circle(), 2, 4, 0.1, False, generators=generators)
    assert len(generators) == 1 and generators[0] is prefix


def test_analytic_wall_without_unique_top_edge_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flat = BRepBuilderAPI_MakeFace(
        gp_Pln(gp_Pnt(0, 0, 4), gp_Dir(0, 0, 1)),
        0,
        1,
        0,
        1,
    ).Face()

    def flat_walls(*_args: object) -> list[TopoDS_Shape]:
        return [flat]

    monkeypatch.setattr(n, "_line_wall_sheets", flat_walls)
    generators: list[tuple[TopoDS_Edge, TopoDS_Face]] = []
    with pytest.raises(occt.KernelError, match="unique exact top generator"):
        _ = n._unjoined_sheets(
            polygon_lines(box(0, 0, 2, 1)).exterior,
            2,
            4,
            0.1,
            False,
            generators=generators,
        )
    assert generators == []
