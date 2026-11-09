"""Exact reverse receipts survive only while every actual witness curve does."""

# Exercise the proof core's exact synthetic lines and bounded cache state.
# pyright: reportPrivateUsage=false

from fractions import Fraction
from math import cos, pi, sin

import pytest

from panelchemy import geometric_distance as g
from panelchemy.curves import BezierSpan, RationalControls, _controls, _line


def _near_limit() -> tuple[tuple[tuple[float, float], ...], BezierSpan]:
    source = tuple(
        (1.195 * cos(i * pi / 200), 1.195 * sin(i * pi / 200)) for i in range(101)
    )
    k = 0.5522847498307936
    return source, BezierSpan(((1, 0), (1, k), (k, 1), (0, 1)), False)


def test_parallel_union_receipt_requires_all_owners() -> None:
    source = ((0.0, 0.0), (2.0, 0.0))
    curves = (_line((0, 0), (0.7, 0)), _line((0.7, 0), (2, 0)))
    cache = g.ForwardProofCache((source,), 0)
    assert cache.within_extra_paths((), curves)
    assert list(cache._reverse.values()) == [frozenset(map(_controls, curves))]
    assert not cache.within_extra_paths((), curves[:1])
    assert not cache._reverse


def test_subdivision_receipt_unions_every_accepted_leaf_owner() -> None:
    source = ((0.0, 0.0), (2.0, 0.0))
    curves = (_line((0, 0), (1, 0.1)), _line((1, 0.1), (2, 0)))
    cache = g.ForwardProofCache((source,), 0.15)
    assert cache.within_extra_paths((), curves)
    assert list(cache._reverse.values()) == [frozenset(map(_controls, curves))]
    assert not cache.within_extra_paths((), curves[1:])


def test_partial_subdivision_success_never_becomes_a_receipt() -> None:
    source = ((0.0, 0.0), (4.0, 0.0))
    curves = (_line((0, 0), (2.7, 0)), _line((3.3, 0), (4, 0)))
    cache = g.ForwardProofCache((source,), 0.1)
    assert not cache.within_extra_paths((), curves)
    assert not cache._reverse
    assert cache.within_extra_paths((), (*curves, _line((2.7, 0), (3.3, 0))))


def test_new_paths_are_checked_before_cached_base_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = (((0.0, 0.0), (1.0, 0.0)),)
    extra = (((4.0, 0.0), (5.0, 0.0)),)
    curve = _line((0, 0), (1, 0))
    cache = g.ForwardProofCache(base, 0.2)
    assert cache.within_extra_paths((), (curve,))
    checked: list[g._Segment] = []
    original = g._source_to_proxy

    def reverse(
        segment: g._Segment,
        target: g._Capsules,
        budget: g._Budget,
        receipt: set[RationalControls] | None = None,
    ) -> bool:
        checked.append(segment)
        return original(segment, target, budget, receipt)

    monkeypatch.setattr(g, "_source_to_proxy", reverse)
    assert not cache.within_extra_paths(extra, (curve,))
    assert checked == [((Fraction(4), Fraction(0)), (Fraction(5), Fraction(0)))]
    assert cache.within_extra_paths(extra, (curve, _line((4, 0), (5, 0))))
    assert not cache.within_extra_paths(
        (((4.0, 0.0), (6.0, 0.0)),), (curve, _line((4, 0), (5, 0)))
    )


def test_cache_hits_do_not_waive_new_forward_excursions() -> None:
    cache = g.ForwardProofCache((((0.0, 0.0), (1.0, 0.0)),), 0.2)
    curve = _line((0, 0), (1, 0))
    assert cache.within_extra_paths((), (curve,))
    excursion = BezierSpan(((0, 0), (0, 10), (1, 10), (1, 0)), False)
    assert not cache.within_extra_paths((), (curve, excursion))


def test_rejected_extra_candidate_does_not_erase_restored_curve_forward_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = g.ForwardProofCache((((0.0, 0.0), (1.0, 0.0)),), 0.2)
    original_curve = _line((0, 0), (1, 0))
    alternative = _line((0, 0.1), (1, 0.1))
    extra = (((4.0, 0.0), (5.0, 0.0)),)
    computed: list[RationalControls] = []
    original = g._curve_to_source

    def forward(
        controls: RationalControls, target: g._Capsules, budget: g._Budget
    ) -> bool:
        computed.append(controls)
        return original(controls, target, budget)

    monkeypatch.setattr(g, "_curve_to_source", forward)
    assert cache.within_extra_paths((), (original_curve,))
    assert not cache.within_extra_paths(extra, (alternative,))
    assert not cache.within_extra_paths(extra, (original_curve,))
    assert computed == [_controls(original_curve), _controls(alternative)]


def test_exact_proxy_cache_is_frozen_and_reverse_hits_are_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, curve = _near_limit()
    cache = g.ForwardProofCache((source,), 0.2)
    proxy_results: list[list[g._Segment]] = []
    original = g._proxy

    def proxy(
        controls: RationalControls, epsilon: Fraction, budget: g._Budget
    ) -> list[g._Segment] | None:
        selected = original(controls, epsilon, budget)
        if selected is not None:
            proxy_results.append(selected)
        return selected

    monkeypatch.setattr(g, "_proxy", proxy)
    assert cache.within_extra_paths((), (curve,))
    assert cache._proxies
    assert all(isinstance(value, tuple) for value in cache._proxies.values())
    computations = len(proxy_results)
    assert computations > 1  # The near-limit source needs finer local proxies.
    for selected in proxy_results:
        selected.clear()  # Caller-owned lists cannot mutate retained values.
    assert cache.within_extra_paths((), (curve,))
    assert len(proxy_results) == computations
    cache._reverse.clear()  # Force reverse work while keeping exact proxy facts.
    assert cache.within_extra_paths((), (curve,))
    assert len(proxy_results) == computations


@pytest.mark.parametrize("setting", ["_MAX_NODES", "_MAX_DEPTH", "_PROXY_REFINEMENTS"])
def test_resource_policy_changes_evict_cached_certificates(
    monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    source, curve = _near_limit()
    cache = g.ForwardProofCache((source,), 0.2)
    assert cache.within_extra_paths((), (curve,))
    assert cache._reverse and cache._proxies
    monkeypatch.setattr(g, setting, 0)
    assert not cache.within_extra_paths((), (curve,))
    assert not cache._reverse


def test_cache_capacity_bounds_witness_references_and_policy_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0))
    curve = _line((0, 0), (3, 0))
    cache = g.ForwardProofCache((source,), 0.2)
    assert cache.within_extra_paths((), (curve,))
    assert len(cache._reverse) == 3
    monkeypatch.setattr(g, "_MAX_CACHE_UNITS", 3)
    assert cache.within_extra_paths((), (curve,))
    assert cache._cache_size <= 3
    assert len(cache._reverse) == 1
    monkeypatch.setattr(g, "_MAX_CACHE_UNITS", 0)
    assert cache.within_extra_paths((), (curve,))
    assert not cache._positive and not cache._proxies and not cache._reverse
    assert cache._cache_size == 0


def test_removed_geometry_keeps_only_base_forward_facts() -> None:
    source, curve = _near_limit()
    cache = g.ForwardProofCache((source,), 0.2)
    assert cache.within_extra_paths((), (curve,))
    assert cache._proxies and cache._reverse
    assert not cache.within_extra_paths((), (_line((10, 10), (11, 10)),))
    assert cache._positive == {_controls(curve)}
    assert not cache._proxies and not cache._reverse
    assert cache._cache_size == 1
