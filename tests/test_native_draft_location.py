"""Native precision stays strict when near-linear wall generation is retried."""

# pyright: reportPrivateUsage=false

from math import radians, tan
from typing import cast

import pytest
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepFill import BRepFill_Draft
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.TopoDS import TopoDS_Edge, TopoDS_Shape
from shapely.geometry import box

from panelchemy import curved_occt as n
from panelchemy import occt
from panelchemy.curves import BezierSpan, CurveRing


def test_source_location_rejection_keeps_native_precision() -> None:
    wire = n._wire(
        CurveRing(
            (BezierSpan(((0, 0), (0.03, 0.001), (0.06, 0.001), (0.09, 0)), False),)
        ),
        4,
        {},
        draft_compatible=True,
    )
    wrong_wall = occt.polygon_face(box(-1, -1, 1, 1), 0)

    class WrongWall:
        @staticmethod
        def Generated(_edge: TopoDS_Edge) -> list[TopoDS_Shape]:
            return [wrong_wall]

    with pytest.raises(n._DraftLocationFailure, match="cannot locate"):
        _ = n._draft_direction(
            cast("BRepFill_Draft", cast("object", WrongWall())), wire, internal=False
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_quasilinear_native_cubic_retains_its_exact_source_coefficients(
    reverse: bool,
) -> None:
    ring = CurveRing(
        (BezierSpan(((0, 0), (0.03, 0.0000004), (0.06, 0.0000004), (0.09, 0)), False),)
    )
    edge = BRepTools_WireExplorer(n._wire(ring, 4, {}, draft_compatible=True)).Current()
    if reverse:
        edge.Reverse()
    before = n._geometry_bytes(BRepAdaptor_Curve(edge).BSpline())
    slope = tan(radians(10))
    drafts = n._independent_drafts(edge, 2 - 0.02 * slope, 4, slope, False, {})
    assert drafts and all(occt.valid(draft.Shell()) for draft in drafts)
    assert n._geometry_bytes(BRepAdaptor_Curve(edge).BSpline()) == before
