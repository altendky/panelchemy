"""Assembly reuse preserves actual geometry and reruns current safety gates."""

# pyright: reportPrivateUsage=false

import gc
import weakref
from dataclasses import replace
from fractions import Fraction
from math import inf, nextafter
from typing import Never

import pytest
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from panelchemy import contact_refinement as cr
from panelchemy import curves as c
from panelchemy.contours import PanelContours


def unexpected(*_args: object, **_kwargs: object) -> Never:
    raise AssertionError("an identical artifact should have been reused")


def parts(polygon: Polygon) -> tuple[c._RingPart, ...]:
    return tuple(
        (span, True, c._flatten(span, 0.01, set())[0])
        for span in c.polygon_lines(polygon).exterior.spans
    )


def test_ring_key_binds_metadata_direction_and_actual_rounded_vertices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, original = c.ProofCache(), parts(box(0, 0, 2, 2))
    ring = cache.ring(original)
    backwards = tuple((span, False, points) for span, _, points in reversed(original))
    reversed_ring = cache.ring(backwards)
    assert tuple(reversed_ring.proxy.coords) == tuple(reversed(ring.proxy.coords))
    first, forward, vertices = original[0]
    metadata = (
        (replace(first, certified_error_mm=0.25), forward, vertices),
        *original[1:],
    )
    tagged = cache.ring(metadata)
    assert tagged is not ring
    assert tagged.curve.spans[0].certified_error_mm == 0.25
    assert tagged.proxy.wkb == ring.proxy.wkb
    moved_span = replace(
        first,
        control_points=(first.control_points[0], (1.5, 1), *first.control_points[2:]),
    )
    moved = ((moved_span, forward, vertices), *original[1:])
    moved_ring = cache.ring(moved)
    assert moved_ring.curve != ring.curve and moved_ring.proxy.wkb == ring.proxy.wkb
    curved_span = replace(first, is_line=False)
    curved = ((curved_span, forward, vertices), *original[1:])
    curved_ring = cache.ring(curved)
    assert curved_ring is not ring and curved_ring.curve != ring.curve
    assert curved_ring.proxy.wkb == ring.proxy.wkb
    rounded = ((first, forward, (vertices[0], (2.125, 1), vertices[-1])), *original[1:])
    changed = cache.ring(rounded)
    assert changed is not ring and changed.proxy.wkb != ring.proxy.wkb
    assert changed.curve == ring.curve
    monkeypatch.setattr(c, "_assemble_ring", unexpected)
    assert cache.ring(original) is ring
    assert cache.ring(backwards) is reversed_ring
    assert cache.ring(metadata) is tagged
    assert cache.ring(moved) is moved_ring
    assert cache.ring(curved) is curved_ring
    assert cache.ring(rounded) is changed


def test_polygon_key_preserves_exterior_and_hole_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = c.ProofCache()
    outer, a, b = (
        cache.ring(parts(p))
        for p in (box(0, 0, 10, 10), box(1, 1, 2, 2), box(6, 6, 8, 8))
    )
    versions = ((outer, a, b), (outer, b, a), (a, outer, b), (outer, a))
    assembled = [cache.polygon(rings) for rings in versions]
    for rings, actual in zip(versions, assembled, strict=True):
        expected = c._assemble_polygon(rings)
        assert actual[0] == expected[0]
        assert actual[1].wkb == expected[1].wkb
    assert assembled[0][1].wkb != assembled[1][1].wkb
    assert not assembled[2][1].is_valid
    monkeypatch.setattr(c, "_assemble_polygon", unexpected)
    for rings, actual in zip(versions, assembled, strict=True):
        assert cache.polygon(rings) is actual


def test_validity_binds_both_actual_polygon_and_source_silhouette() -> None:
    cache = c.ProofCache()
    polygon, source = box(1, 1, 3, 3), box(0, 0, 4, 4)
    assert cache.valid_polygon(polygon, source)
    bowtie = Polygon(((0, 0), (3, 3), (0, 3), (3, 0), (0, 0)))
    cases = (
        (polygon, box(0, 0, 2, 2)),
        (box(5, 5, 6, 6), source),
        (bowtie, source),
        (Polygon(((1, 1), (2, 2), (3, 3), (1, 1))), source),
        (Polygon(), source),
        (polygon, bowtie),
        (polygon, Polygon()),
    )
    for actual, silhouette in cases:
        expected = actual.is_valid and actual.area > 0 and silhouette.covers(actual)
        assert cache.valid_polygon(actual, silhouette) == expected
        assert cache.valid_polygon(actual, silhouette) == expected
    assert cache.valid_polygon(polygon, source)


def test_identity_keys_retain_operands_and_do_not_coalesce_equal_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = c.ProofCache()
    source, polygon = box(0, 0, 5, 5), box(1, 1, 2, 2)
    source_ref, polygon_ref = weakref.ref(source), weakref.ref(polygon)
    assert cache.valid_polygon(polygon, source)
    expected = cache.relation(source, polygon)
    calls = 0
    original = BaseGeometry.relate

    def record(left: BaseGeometry, right: BaseGeometry) -> str:
        nonlocal calls
        calls += 1
        return original(left, right)

    monkeypatch.setattr(BaseGeometry, "relate", record)
    assert cache.relation(source, polygon) == expected
    equal_copy = box(1, 1, 2, 2)
    assert equal_copy is not polygon
    assert cache.relation(source, equal_copy) == expected
    assert calls == 1
    del source, polygon
    _ = gc.collect()
    assert source_ref() is not None and polygon_ref() is not None
    del cache
    _ = gc.collect()
    assert source_ref() is None and polygon_ref() is None


def test_relations_are_ordered_and_relationship_indices_are_current() -> None:
    cache = c.ProofCache()
    outer, inner, touching = box(0, 0, 5, 5), box(1, 1, 2, 2), box(5, 1, 6, 2)
    assert cache.relation(outer, inner) == outer.relate(inner)
    assert cache.relation(inner, outer) == inner.relate(outer)
    assert cache.relation(inner, outer) != cache.relation(outer, inner)
    for geometries in (
        [outer, inner, touching],
        [touching, outer, inner],
        [inner, touching],
        [inner, touching, box(1.5, 1, 6, 2)],
    ):
        assert c._relationships(geometries, cache) == c._relationships(geometries)
    # Bounding boxes overlap, but the inner polygon is wholly within a hole.
    donut = Polygon(outer.exterior.coords, [box(0.5, 0.5, 4.5, 4.5).exterior.coords])
    assert not donut.intersects(inner)
    assert c._relationships([donut, inner], cache) == {}


def assembly_input() -> tuple[
    list[c._Piece], list[list[tuple[int, bool]]], list[range], PanelContours
]:
    source = PanelContours(
        box(0, 0, 10, 10), [], [box(2, 2, 6, 6)], 1, (0, 0, 10, 10), 8
    )
    inner = c.CurveRing(
        (
            c._line((2, 2), (6, 2)),
            c._line((6, 2), (6, 6)),
            c.BezierSpan(((6, 6), (5, 7), (3, 7), (2, 6)), False),
            c._line((2, 6), (2, 2)),
        )
    )
    pieces: list[c._Piece] = []
    uses: list[list[tuple[int, bool]]] = []
    for ring in (c.polygon_lines(source.silhouette).exterior, inner):
        selected: list[tuple[int, bool]] = []
        for span in ring.spans:
            index = len(pieces)
            pieces.append(c._Piece(index, 0, 1, span))
            selected.append((index, True))
        uses.append(selected)
    return pieces, uses, [range(1), range(1, 2)], source


def test_full_assembly_artifacts_follow_current_cuts_and_tolerance() -> None:
    cache = c.ProofCache()
    pieces, uses, ranges, source = assembly_input()
    cuts = [set[Fraction]() for _ in pieces]
    versions: list[bytes] = []
    for tolerance, extra in ((0.02, False), (0.02, True), (0.001, True), (0.02, False)):
        cuts[6] = {Fraction(1, 3)} if extra else set()
        expected = c._assemble(pieces, uses, ranges, source, tolerance, cuts)
        actual = c._assemble(pieces, uses, ranges, source, tolerance, cuts, cache)
        assert actual[0] == expected[0]
        assert actual[1].vertex_count == expected[1].vertex_count
        assert actual[1].silhouette is source.silhouette
        assert [p.wkb for p in actual[1].white] == [p.wkb for p in expected[1].white]
        assert [r.wkb for r in actual[2]] == [r.wkb for r in expected[2]]
        assert actual[3] == expected[3]
        versions.append(actual[1].white[0].wkb)
    assert versions[0] != versions[1] != versions[2]
    assert versions[3] == versions[0]


@pytest.mark.parametrize("stage", ["native", "proxy", "silhouette", "hole"])
def test_warm_artifacts_still_obey_current_native_and_proxy_safety(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    source = PanelContours(
        box(0, 0, 10, 10), [], [box(2, 2, 6, 6)], 1, (0, 0, 10, 10), 8
    )
    polygons = [c.polygon_lines(source.silhouette), c.polygon_lines(source.white[0])]
    cache = c.ProofCache()
    expected = cr._assemble(polygons, source, 0.01)
    assert expected is not None
    actual = cr._assemble(polygons, source, 0.01, proofs=cache)
    assert actual is not None and actual[0] == expected[0]
    assert actual[1].white[0].wkb == expected[1].white[0].wkb
    assert cache._rings and cache._polygons and cache._validity
    if stage == "native":

        def unsafe_native(*_args: object) -> tuple[set[int], list[set[Fraction]]]:
            return {0}, []

        monkeypatch.setattr(c, "_unsafe_pair", unsafe_native)
    elif stage == "proxy":

        def unsafe_proxy(*_args: object) -> list[tuple[int, int]]:
            return [(0, 1)]

        monkeypatch.setattr(c, "_proxy_unsafe", unsafe_proxy)
    elif stage == "silhouette":
        source = replace(source, silhouette=box(0, 0, 3, 3))
    else:
        source = replace(
            source,
            white=[
                Polygon(
                    source.white[0].exterior.coords, [box(3, 3, 4, 4).exterior.coords]
                )
            ],
        )
    assert cr._assemble(polygons, source, 0.01, proofs=cache) is None


def test_artifact_maps_evict_without_losing_geometry_or_retaining_old_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c, "_PROOF_CACHE_LIMIT", 1)
    cache = c.ProofCache()
    first_parts, second_parts = parts(box(0, 0, 1, 1)), parts(box(0, 0, 2, 2))
    first_ring = cache.ring(first_parts)
    first_ref = weakref.ref(first_ring)
    _ = cache.polygon((first_ring,))
    _ = cache.ring(second_parts)
    del first_ring
    _ = gc.collect()
    assert first_ref() is not None  # polygon key retains its ring identity
    _ = cache.polygon((cache.ring(second_parts),))
    _ = gc.collect()
    assert first_ref() is None
    rebuilt = cache.ring(first_parts)
    assert rebuilt.curve == c._assemble_ring(first_parts).curve
    source, old = box(0, 0, 5, 5), box(1, 1, 2, 2)
    old_ref = weakref.ref(old)
    assert cache.valid_polygon(old, source)
    _ = cache.relation(old, source)
    new = box(3, 3, 4, 4)
    assert cache.valid_polygon(new, source)
    _ = cache.relation(new, source)
    del old
    _ = gc.collect()
    assert old_ref() is None
    assert all(
        len(mapping) <= 1
        for mapping in (
            cache._rings,
            cache._polygons,
            cache._validity,
            cache._relations,
        )
    )


@pytest.mark.parametrize("scale", [Fraction(1), Fraction(1, 10**400)])
def test_proxy_box_preserves_outward_nextafter_bounds(scale: Fraction) -> None:
    hull = tuple(
        (x * scale, y * scale)
        for x, y in (
            (Fraction(-2, 3), Fraction(1, 3)),
            (Fraction(7, 3), Fraction(11, 3)),
        )
    )
    actual = c._proxy_box(hull)
    expected = (
        nextafter(float(Fraction(-2, 3) * scale), -inf),
        nextafter(float(Fraction(1, 3) * scale), -inf),
        nextafter(float(Fraction(7, 3) * scale), inf),
        nextafter(float(Fraction(11, 3) * scale), inf),
    )
    assert actual.bounds == expected
    left, bottom, right, top = map(Fraction, actual.bounds)
    assert all(left <= x <= right and bottom <= y <= top for x, y in hull)
    assert c.ProofCache().proxy_box(hull).wkb == actual.wkb


def test_proxy_box_reuses_identity_but_not_equal_or_changed_hulls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hull = c._controls(c._line((0, 0), (2, 2)))
    cache, original, calls = c.ProofCache(), c._proxy_box, 0

    def record(value: tuple[c.RationalPoint, ...]) -> Polygon:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(c, "_proxy_box", record)
    first = cache.proxy_box(hull)
    assert cache.proxy_box(hull) is first and calls == 1
    equal = tuple(list(hull))
    assert equal == hull and equal is not hull
    second = cache.proxy_box(equal)
    assert second is not first and second.wkb == first.wkb and calls == 2
    changed = (*hull[:-1], (Fraction(3), Fraction(3)))
    assert cache.proxy_box(changed).bounds != first.bounds and calls == 3


def test_warm_proxy_boxes_still_query_current_hulls_cuts_and_piece_indices() -> None:
    cache = c.ProofCache()
    arch = c._Piece(0, 0, 1, c.BezierSpan(((0, 0), (0, 1), (1, 1), (1, 0)), False))
    above = c._Piece(1, 0, 1, c._line((0, 1), (1, 1)))
    crossed = replace(above, span=c._line((0, 0.6), (1, 0.6)))
    far = c._Piece(2, 0, 1, c._line((4, 4), (5, 5)))
    for pieces, cuts in (
        ([arch, above], set[Fraction]()),
        ([arch, crossed], {Fraction(1, 3)}),
        ([far, crossed, arch], set[Fraction]()),
    ):
        hulls = [cache.flatten(piece.span, 0.01, cuts)[1] for piece in pieces]
        expected = c._proxy_unsafe(pieces, hulls)
        assert c._proxy_unsafe(pieces, hulls, cache) == expected
        assert c._proxy_unsafe(pieces, hulls, cache) == expected
        if crossed in pieces:
            left, right = sorted((pieces.index(arch), pieces.index(crossed)))
            assert expected == [(left, right)]
        else:
            assert not expected
    assert cache._proxy_boxes


def test_proxy_box_identity_entry_retains_operand_and_evicts_its_polygon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c, "_PROOF_CACHE_LIMIT", 1)
    cache = c.ProofCache()
    hull = tuple(list(c._controls(c._line((0, 0), (1, 1)))))
    identity = id(hull)
    polygon = cache.proxy_box(hull)
    reference = weakref.ref(polygon)
    assert cache._proxy_boxes[identity][0] is hull
    del hull, polygon
    _ = gc.collect()
    assert reference() is not None
    _ = cache.proxy_box(c._controls(c._line((2, 2), (3, 3))))
    _ = gc.collect()
    assert reference() is None and identity not in cache._proxy_boxes
    assert len(cache._proxy_boxes) == 1
