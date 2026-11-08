"""Shared boundaries use exposed height differences and reverse curvature signs."""

from dataclasses import replace
from math import sqrt
from typing import cast

import pytest
from shapely.geometry import box

from panelchemy import curvature
from panelchemy.boundary_tapers import (
    CurvatureLimitError,
    boundaries,
    require_curvature,
)
from panelchemy.curves import (
    BezierSpan,
    Controls,
    CurveContours,
    CurvePolygon,
    CurveRing,
    polygon_lines,
)


def test_white_orange_boundary_uses_one_band_not_full_white_support_depth() -> None:
    # Ring0 silhouette; ring1 orange; ring2 white. Chain2 is shared in reverse.
    uses = [[(0, True)], [(1, True), (2, True)], [(2, False), (3, False)]]
    result = boundaries(uses, [range(1), range(1, 2), range(2, 3)], 1, (0, 0.2, 0.4))
    assert 0 not in result
    assert (result[1].left, result[1].right, result[1].offset_mm) == (1, 0, 0.2)
    assert (result[2].left, result[2].right, result[2].offset_mm) == (1, 2, 0.2)
    assert result[3].offset_mm == 0.4
    assert result[1].inward_sign == -1
    assert result[2].inward_sign == 1
    assert result[3].inward_sign == 1


def test_reversing_canonical_direction_reverses_required_curvature_sign() -> None:
    ranges = [range(1), range(1, 2)]
    a = boundaries([[(0, True)], [(1, True)]], ranges, 1, (0, 0.176, 0.353))[1]
    b = boundaries([[(0, True)], [(1, False)]], ranges, 1, (0, 0.176, 0.353))[1]
    assert a.offset_mm == b.offset_mm
    assert a.inward_sign == -b.inward_sign
    assert (a.left, a.right) == (b.right, b.left)


def test_boundary_ownership_conflicts_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting"):
        _ = boundaries(
            [[(0, True)], [(1, True)], [(1, True)]],
            [range(1), range(1, 2), range(2, 3)],
            1,
            (0, 1, 2),
        )


@pytest.mark.parametrize("runs", [(0, -1, 2), (0, 2, 1), (0, float("nan"), 2)])
def test_invalid_material_runs_are_rejected(runs: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        _ = boundaries([[]], [range(1)], 0, runs)


def circle(radius: float) -> CurveRing:
    k = 4 * (sqrt(2) - 1) / 3
    quadrant = ((1.0, 0.0), (1.0, k), (k, 1.0), (0.0, 1.0))
    spans: list[BezierSpan] = []
    for _ in range(4):
        spans.append(
            BezierSpan(
                cast(
                    "Controls",
                    tuple((5 + radius * x, 5 + radius * y) for x, y in quadrant),
                ),
                False,
            )
        )
        quadrant = tuple((-y, x) for x, y in quadrant)
    return CurveRing(tuple(spans))


def reversed_ring(ring: CurveRing) -> CurveRing:
    return CurveRing(
        tuple(
            replace(
                span,
                control_points=cast("Controls", tuple(reversed(span.control_points))),
            )
            for span in reversed(ring.spans)
        )
    )


def selected_curves(*, orange_pocket: bool = True) -> CurveContours:
    pocket = circle(0.3)
    return CurveContours(
        polygon_lines(box(0, 0, 10, 10)),
        (CurvePolygon(pocket),) if orange_pocket else (),
        (replace(polygon_lines(box(1, 1, 9, 9)), interiors=(reversed_ring(pocket),)),),
        0.1,
        (0, 0, 100, 100),
    )


@pytest.mark.parametrize("reverse", ["none", "white", "orange", "all"])
def test_final_curvature_gate_uses_shared_exposure_and_selected_winding(
    reverse: str,
) -> None:
    shared = selected_curves()
    black = selected_curves(orange_pocket=False)
    if reverse != "none":

        def backwards(polygon: CurvePolygon) -> CurvePolygon:
            return CurvePolygon(
                reversed_ring(polygon.exterior),
                tuple(map(reversed_ring, polygon.interiors)),
            )

        shared = replace(
            shared,
            silhouette=backwards(shared.silhouette)
            if reverse == "all"
            else shared.silhouette,
            orange=tuple(map(backwards, shared.orange))
            if reverse in ("orange", "all")
            else shared.orange,
            white=tuple(map(backwards, shared.white))
            if reverse in ("white", "all")
            else shared.white,
        )
        black = replace(
            black,
            silhouette=backwards(black.silhouette)
            if reverse == "all"
            else black.silhouette,
            white=tuple(map(backwards, black.white))
            if reverse in ("white", "all")
            else black.white,
        )
    require_curvature(shared, (0, 0.2, 0.4))
    with pytest.raises(
        CurvatureLimitError, match="4 selected boundary spans"
    ) as caught:
        require_curvature(black, (0, 0.2, 0.4))
    assert caught.value.contours is black
    assert all(limit.offset_mm == 0.4 for _, limit in caught.value.failures.values())
    # Failure leaves the selected source geometry available and unchanged.
    assert black.white[0].interiors


def test_curvature_gate_does_not_forbid_safe_diverging_curvature_or_vertical_walls() -> (
    None
):
    selected = selected_curves(orange_pocket=False)
    convex = replace(selected, white=(CurvePolygon(circle(0.01)),))
    require_curvature(convex, (0, 50, 100))
    require_curvature(selected, (0, 0, 0))


def test_curvature_gate_requires_a_continuous_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def inconclusive(*_args: object) -> bool:
        return False

    monkeypatch.setattr(curvature, "within_curvature", inconclusive)
    with pytest.raises(CurvatureLimitError):
        require_curvature(selected_curves(), (0, 0.2, 0.4))


def test_curvature_gate_ignores_unused_line_handles_for_shared_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = selected_curves()
    # An orange/white straight interface remains one shared boundary even when
    # line metadata contains different unused controls in each direction.
    orange = polygon_lines(box(1, 1, 5, 5))
    white = polygon_lines(box(5, 1, 9, 5))
    orange = replace(
        orange,
        exterior=CurveRing(
            tuple(
                replace(
                    s,
                    control_points=(
                        s.control_points[0],
                        (123, 456),
                        (789, 123),
                        s.control_points[3],
                    ),
                )
                for s in orange.exterior.spans
            )
        ),
    )
    calls: list[float] = []
    original = curvature.within_curvature

    def record(span: BezierSpan, offset: float, sign: int) -> bool:
        if {span.control_points[0], span.control_points[-1]} == {(5, 1), (5, 5)}:
            calls.append(offset)
        return original(span, offset, sign)

    monkeypatch.setattr(curvature, "within_curvature", record)
    require_curvature(
        replace(selected, orange=(orange,), white=(white,)), (0, 0.2, 0.4)
    )
    assert calls == [0.2]
