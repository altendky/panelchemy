"""Roundoff-sized line gaps remain unresolved rather than authorizing CAD."""

# pyright: reportPrivateUsage=false

from dataclasses import replace
from math import inf, nextafter
from typing import Never, cast

import pytest
from shapely.geometry import Polygon, box

from panelchemy import boundary_tapers, contact_refinement, geometric_distance
from panelchemy import curves as c
from panelchemy.contours import PanelContours


def piece(index: int, start: c.Point2, end: c.Point2) -> c._Piece:
    return c._Piece(index, 0, 1, c._line(start, end))


def test_exact_source_fit_does_not_authorize_a_roundoff_near_self_contact() -> None:
    near = (0.4, nextafter(0.4, inf))
    fitted = ((0, 0), (1, 1), (0, 2), (0, 1), near, (-1, 0), (0, 0))
    original = (fitted[0], (0.5, 0.25), *fitted[1:])
    assert Polygon(fitted).is_valid and Polygon(original).is_valid
    selected = c.polygon_lines(Polygon(fitted)).exterior.spans
    assert geometric_distance.within_paths([original], selected, 0.2)
    pieces = [c._Piece(i, 0, 1, span) for i, span in enumerate(selected)]
    unsafe, _ = c._unsafe_pair(pieces)
    assert (0, 3) in unsafe and (0, 4) in unsafe
    source_spans = c.polygon_lines(Polygon(original)).exterior.spans
    assert (
        c._unsafe_pair(
            [c._Piece(i, 0, 1, span) for i, span in enumerate(source_spans)]
        )[0]
        == []
    )
    # The real contact-assembly gate rejects this source-close candidate, then
    # accepts a source-corner alternative with the same palette and radius checks.
    reference = PanelContours(
        box(-2, -2, 3, 3),
        [],
        [Polygon(original)],
        1,
        (-2, -2, 3, 3),
        4 + len(original) - 1,
    )
    silhouette = c.polygon_lines(reference.silhouette)
    cache = c.ProofCache()
    assert (
        contact_refinement._assemble(
            [silhouette, c.polygon_lines(Polygon(fitted))],
            reference,
            0.001,
            proofs=cache,
        )
        is None
    )
    accepted = contact_refinement._assemble(
        [silhouette, c.polygon_lines(Polygon(original))], reference, 0.001, proofs=cache
    )
    assert accepted is not None
    assert accepted[1].white[0].equals(Polygon(original))
    boundary_tapers.require_curvature(accepted[0], (0, 0.2, 0.4))


@pytest.mark.parametrize("shift", [(0, 0), (1024, -2048), (-8192, 4096)])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_guard_handles_world_scale_both_sides_and_reversed_lines(
    shift: c.Point2, side: int, reverse: bool
) -> None:
    x, y = shift
    near = x + 0.5, nextafter(y + 0.5, inf if side > 0 else -inf)
    remote = (x, y + 1) if side > 0 else (x + 1, y)
    pieces = [piece(0, (x, y), (x + 1, y + 1)), piece(1, near, remote)]
    if reverse:
        pieces = [
            replace(
                p,
                span=replace(
                    p.span,
                    control_points=cast(
                        "c.Controls", tuple(reversed(p.span.control_points))
                    ),
                ),
            )
            for p in pieces
        ]
    a, b = (c._controls(p.span) for p in pieces)
    assert c._disjoint(a, b, set())  # positive exact gap alone is insufficient
    assert c._unsafe_pair(pieces)[0] == [(0, 1)]
    assert c._unsafe_pair(list(reversed(pieces)))[0] == [(0, 1)]


@pytest.mark.parametrize("vertical", [False, True])
def test_expanded_index_finds_parallel_lines_with_disjoint_original_boxes(
    vertical: bool,
) -> None:
    points = ((0, 1), (2, 1), (0.5, nextafter(1, inf)), (1.5, nextafter(1, inf)))
    if vertical:
        points = tuple((y, x) for x, y in points)
    a, b, q, r = points
    assert c._unsafe_pair([piece(0, a, b), piece(1, q, r)])[0] == [(0, 1)]


@pytest.mark.parametrize("gap", [0.01, 1e-6, 2**-38])
def test_resolved_gaps_are_accepted_without_a_new_fixed_clearance(gap: float) -> None:
    assert (
        c._unsafe_pair(
            [piece(0, (0, 1), (2, 1)), piece(1, (0.5, 1 + gap), (1.5, 1 + gap))]
        )[0]
        == []
    )


def test_explicit_shared_endpoint_remains_allowed() -> None:
    pieces = [piece(0, (0, 0), (1, 1)), piece(1, (1, 1), (2, 1))]
    assert c._unsafe_pair(pieces)[0] == []
    cache = c.ProofCache()
    assert c._unsafe_pair(pieces, cache) == c._unsafe_pair(pieces)
    assert c._unsafe_pair(pieces, cache) == c._unsafe_pair(pieces)


def test_unused_line_handles_do_not_enlarge_the_uncertainty_scale() -> None:
    first = piece(0, (0, 1), (2, 1))
    controls = first.span.control_points
    first = replace(
        first,
        span=replace(
            first.span,
            control_points=(
                controls[0],
                (1e100, 1e100),
                (-1e100, -1e100),
                controls[-1],
            ),
        ),
    )
    assert c._unsafe_pair([first, piece(1, (0.5, 1.000001), (1.5, 1.000001))])[0] == []


def unexpected(*_args: object, **_kwargs: object) -> Never:
    raise AssertionError("the exact separation proof should already be cached")


def test_warm_exact_pair_cache_cannot_bypass_the_current_roundoff_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pieces = [piece(0, (0, 0), (1, 1)), piece(1, (0.5, nextafter(0.5, inf)), (0, 1))]
    a, b = (c._controls(p.span) for p in pieces)
    cache = c.ProofCache()
    assert cache.disjoint(a, b, set(), set(), set())
    expected = c._unsafe_pair(pieces)
    assert expected[0] == [(0, 1)]
    monkeypatch.setattr(c, "_disjoint", unexpected)
    assert c._unsafe_pair(pieces, cache) == expected
    assert c._unsafe_pair(pieces, cache) == expected
