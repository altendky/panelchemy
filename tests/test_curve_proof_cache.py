"""Exact topology proof reuse retains subdivisions and every acceptance input."""

# pyright: reportPrivateUsage=false

import gc
import weakref
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from typing import Never

import pytest
from shapely.geometry import LineString, box

from panelchemy import curves as c
from panelchemy.contours import PanelContours

type Assembly = tuple[
    c.CurveContours, PanelContours, list[LineString], list[tuple[int, int]]
]


def arch() -> c.BezierSpan:
    return c.BezierSpan(((0, 0), (0, 1), (1, 1), (1, 0)), False, 0.2)


def unexpected(*_args: object, **_kwargs: object) -> Never:
    raise AssertionError("a complete cached proof should have been reused")


def test_pair_cache_replays_both_cut_contributions_without_losing_existing_cuts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, b = c._controls(arch()), c._controls(c._line((0, 1), (1, 1)))
    expected_a, expected_b = set[Fraction](), set[Fraction]()
    assert c._disjoint(a, b, set(), cuts_a=expected_a, cuts_b=expected_b)
    assert expected_a == {Fraction(1, 2)}
    cache = c.ProofCache()
    left, right = set[Fraction](), set[Fraction]()
    assert cache.disjoint(a, b, set(), left, right)
    assert (left, right) == (expected_a, expected_b)
    monkeypatch.setattr(c, "_disjoint", unexpected)
    left, right = {Fraction(1, 4)}, {Fraction(3, 4)}
    assert cache.disjoint(a, b, set(), left, right)
    assert left == expected_a | {Fraction(1, 4)}
    assert right == expected_b | {Fraction(3, 4)}


def test_rejected_pair_cache_replays_partial_subdivision_cuts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = c._controls(c._line((0, 0), (1, 0)))
    b = c._controls(c._line((1, 0), (1, 1)))
    expected_a, expected_b = set[Fraction](), set[Fraction]()
    assert not c._disjoint(a, b, set(), cuts_a=expected_a, cuts_b=expected_b)
    assert expected_a or expected_b
    cache = c.ProofCache()
    left, right = set[Fraction](), set[Fraction]()
    assert not cache.disjoint(a, b, set(), left, right)
    monkeypatch.setattr(c, "_disjoint", unexpected)
    left, right = set[Fraction](), set[Fraction]()
    assert not cache.disjoint(a, b, set(), left, right)
    assert (left, right) == (expected_a, expected_b)


def test_pair_cache_is_a_unit_interval_proof_not_a_recursive_range_cache() -> None:
    a, b = c._controls(arch()), c._controls(c._line((0, 1), (1, 1)))
    other_a, other_b = set[Fraction](), set[Fraction]()
    assert c._disjoint(
        a,
        b,
        set(),
        cuts_a=other_a,
        cuts_b=other_b,
        range_a=(Fraction(1), Fraction(2)),
    )
    assert other_a == {Fraction(3, 2)}
    cache = c.ProofCache()
    for _ in range(2):
        left, right = set[Fraction](), set[Fraction]()
        assert cache.disjoint(a, b, set(), left, right)
        assert left == {Fraction(1, 2)} and not right


def test_pair_cache_invalidates_changed_controls_and_allowed_endpoints() -> None:
    cache = c.ProofCache()
    a, b = c._controls(arch()), c._controls(c._line((0, 0.6), (1, 0.6)))
    assert not cache.disjoint(a, b, set(), set(), set())
    lower = replace(arch(), control_points=((0, 0), (0, 0.5), (1, 0.5), (1, 0)))
    assert cache.disjoint(c._controls(lower), b, set(), set(), set())
    a = c._controls(c._line((0, 0), (1, 0)))
    b = c._controls(c._line((1, 0), (1, 1)))
    assert not cache.disjoint(a, b, set(), set(), set())
    assert cache.disjoint(a, b, {c._rational((1, 0))}, set(), set())


def test_flatten_cache_binds_full_cuts_tolerance_and_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span, cache = arch(), c.ProofCache()
    plain = cache.flatten(span, 0.01, set())
    original = c._flatten
    monkeypatch.setattr(c, "_flatten", unexpected)
    assert cache.flatten(span, 0.01, set()) == plain
    monkeypatch.setattr(c, "_flatten", original)
    cuts = {Fraction(1, 3)}
    subdivided = cache.flatten(span, 0.01, cuts)
    assert subdivided == original(span, 0.01, cuts)
    assert subdivided != plain
    assert cache.flatten(span, 0.02, cuts) == original(span, 0.02, cuts)
    changed = replace(span, control_points=((0, 0), (0, 0.5), (1, 0.5), (1, 0)))
    assert cache.flatten(changed, 0.01, cuts) == original(changed, 0.01, cuts)
    assert cache.flatten(changed, 0.01, cuts) != subdivided
    line = replace(span, is_line=True)
    assert cache.flatten(line, 0.01, cuts) == original(line, 0.01, cuts)
    assert cache.flatten(line, 0.01, cuts) != subdivided


def test_flatten_cache_returns_caller_safe_hull_lists() -> None:
    span, cache = arch(), c.ProofCache()
    expected = c._flatten(span, 0.01, set())
    _, hulls = cache.flatten(span, 0.01, set())
    hulls.clear()
    assert cache.flatten(span, 0.01, set()) == expected


def test_proxy_hull_cache_binds_full_hulls_and_allowed_endpoints() -> None:
    a = (c._rational((0, 0)), c._rational((1, 0)))
    b = (c._rational((1, 0)), c._rational((1, 1)))
    allowed = {c._rational((1, 0))}
    cache = c.ProofCache()
    assert cache.separated(a, b, allowed)
    assert not cache.separated(a, b, set())
    assert not cache.separated((*a, c._rational((1, 0.5))), b, allowed)


def test_bounded_cache_eviction_recomputes_without_changing_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c, "_PROOF_CACHE_LIMIT", 2)
    original, calls = c._separated, 0

    def checked(
        a: Sequence[c.RationalPoint],
        b: Sequence[c.RationalPoint],
        allowed: set[c.RationalPoint],
    ) -> bool:
        nonlocal calls
        calls += 1
        return original(a, b, allowed)

    monkeypatch.setattr(c, "_separated", checked)
    a = (c._rational((0, 0)), c._rational((1, 0)))
    others = [(c._rational((2, y)), c._rational((3, y))) for y in (1.0, 2.0, 3.0)]
    cache = c.ProofCache()
    for i in (0, 1, 0, 2, 0):
        assert cache.separated(a, others[i], set())
    assert calls == 3
    assert cache.separated(a, others[1], set())
    assert calls == 4


def test_cache_has_no_global_reference_retaining_its_owner() -> None:
    cache = c.ProofCache()
    _ = cache.flatten(arch(), 0.01, set())
    reference = weakref.ref(cache)
    del cache
    _ = gc.collect()
    assert reference() is None


def test_pair_checks_follow_new_piece_order_and_geometry() -> None:
    arch_piece = c._Piece(10, 0, 1, arch())
    upper = c._Piece(20, 0, 1, c._line((0, 1), (1, 1)))
    crossing = c._Piece(20, 0, 1, c._line((0, 0.6), (1, 0.6)))
    far = c._Piece(30, 0, 1, c._line((3, 3), (4, 4)))
    cache = c.ProofCache()
    for pieces in (
        [arch_piece, upper],
        [far, upper, arch_piece],
        [arch_piece, crossing],
    ):
        assert c._unsafe_pair(pieces, cache) == c._unsafe_pair(pieces)


def assembly_input() -> tuple[
    list[c._Piece], list[list[tuple[int, bool]]], list[range], PanelContours
]:
    source = PanelContours(
        box(0, 0, 10, 10), [], [box(2, 2, 6, 6)], 1, (0, 0, 10, 10), 8
    )
    exterior = c.polygon_lines(source.silhouette).exterior
    interior = c.CurveRing(
        (
            c._line((2, 2), (6, 2)),
            c._line((6, 2), (6, 6)),
            c.BezierSpan(((6, 6), (5, 7), (3, 7), (2, 6)), False, 0.2),
            c._line((2, 6), (2, 2)),
        )
    )
    pieces: list[c._Piece] = []
    uses: list[list[tuple[int, bool]]] = []
    for ring in (exterior, interior):
        selected: list[tuple[int, bool]] = []
        for span in ring.spans:
            index = len(pieces)
            pieces.append(c._Piece(index, 0, 1, span))
            selected.append((index, True))
        uses.append(selected)
    return pieces, uses, [range(0, 1), range(1, 2)], source


def assert_same_assembly(actual: Assembly, expected: Assembly) -> None:
    assert actual[0] == expected[0]
    assert actual[1].vertex_count == expected[1].vertex_count
    assert actual[1].silhouette is expected[1].silhouette
    assert [p.wkb for p in actual[1].white] == [p.wkb for p in expected[1].white]
    assert [p.wkb for p in actual[2]] == [p.wkb for p in expected[2]]
    assert actual[3] == expected[3]


def test_whole_assembly_matches_uncached_after_local_change() -> None:
    pieces, uses, ranges, source = assembly_input()
    cache = c.ProofCache()
    for changed in (False, False, True):
        if changed:
            span = pieces[6].span
            controls = list(span.control_points)
            controls[1] = (controls[1][0] + 2**-10, controls[1][1])
            pieces[6] = replace(
                pieces[6], span=replace(span, control_points=tuple(controls))
            )
        unsafe, cuts = c._unsafe_pair(pieces)
        assert not unsafe
        assert c._unsafe_pair(pieces, cache) == (unsafe, cuts)
        expected = c._assemble(pieces, uses, ranges, source, 0.01, cuts)
        actual = c._assemble(pieces, uses, ranges, source, 0.01, cuts, cache)
        assert_same_assembly(actual, expected)


def test_native_pair_proof_does_not_fill_proxy_hull_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = c._controls(c._line((0, 0), (1, 0)))
    b = c._controls(c._line((1, 0), (1, 1)))
    allowed = {c._rational((1, 0))}
    cache = c.ProofCache()
    assert cache.disjoint(a, b, allowed, set(), set())
    monkeypatch.setattr(c, "_separated", unexpected)
    with pytest.raises(AssertionError, match="cached proof"):
        _ = cache.separated(a, b, allowed)
