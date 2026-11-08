"""The native proof retains the Python reference's exact acceptance decisions."""

# This suite intentionally compares the independent reference backend.
# pyright: reportPrivateUsage=false

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from math import cos, inf, nextafter, pi, sin
from random import Random
from typing import cast

import pytest

from panelchemy import geometric_distance as distance
from panelchemy.curves import BezierSpan, Controls, Point2, _line


def quarter(radius: float = 1) -> tuple[tuple[Point2, ...], BezierSpan]:
    path = tuple(
        (radius * cos(i * pi / 80), radius * sin(i * pi / 80)) for i in range(41)
    )
    k = 0.5522847498307936
    return path, BezierSpan(((1, 0), (1, k), (k, 1), (0, 1)), False)


@pytest.mark.parametrize("nodes", [0, 1, 20, 100_000])
@pytest.mark.parametrize("depth", [0, 1, 24])
@pytest.mark.parametrize("refinements", [0, 3])
def test_native_matches_reference_with_shared_resource_limits(
    monkeypatch: pytest.MonkeyPatch, nodes: int, depth: int, refinements: int
) -> None:
    monkeypatch.setattr(distance, "_MAX_NODES", nodes)
    monkeypatch.setattr(distance, "_MAX_DEPTH", depth)
    monkeypatch.setattr(distance, "_PROXY_REFINEMENTS", refinements)
    path, curve = quarter(1.195)
    assert distance._within_paths_native(
        (path,), (curve,), 0.2
    ) == distance._within_paths_python((path,), (curve,), 0.2)


@pytest.mark.parametrize("threshold", [0, 32, 10_000])
def test_native_retains_proxy_refinement_and_neighboring_tolerance_limits(
    monkeypatch: pytest.MonkeyPatch, threshold: int
) -> None:
    monkeypatch.setattr(distance, "_INDEX_MIN_SEGMENTS", threshold)
    path, curve = quarter(1.195)
    for tolerance in (0.194, 0.2, nextafter(0.2, 0), nextafter(0.2, inf)):
        assert distance._within_paths_native(
            (path,), (curve,), tolerance
        ) == distance._within_paths_python((path,), (curve,), tolerance)


def test_line_metadata_ignores_finite_handles_but_still_validates_them() -> None:
    path = ((0.1, 0.2), (0.27, 0.2), (0.9, 0.2))
    curve = BezierSpan((path[0], (1000, -500), (-2000, 800), path[-1]), True)
    assert distance._within_paths_native((path,), (curve,), 0)
    assert distance._within_paths_python((path,), (curve,), 0)
    invalid = BezierSpan((path[0], (inf, 0), (0, 0), path[-1]), True)
    with pytest.raises(ValueError, match="finite planar"):
        _ = distance._within_paths_native((path,), (invalid,), 0)


def test_non_binary64_coordinates_and_tolerance_keep_exact_reference_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_native(*_args: object) -> bool:
        raise AssertionError("nonrepresentable values must not be rounded")

    monkeypatch.setattr(distance, "_native_within_paths", unexpected_native)
    large = 2**53
    assert not distance._within_paths_native(
        (((large, 0), (large + 2, 0)),),
        (_line((large + 1, 0), (large + 2, 0)),),
        0,
    )
    third = cast("float", cast("object", Fraction(1, 3)))
    assert distance._within_paths_native(
        (((0, 0), (third, 0)),), (_line((0, 0), (third, 0)),), 0
    )
    tenth = cast("float", cast("object", Fraction(1, 10)))
    assert distance._within_paths_native(
        (((0, 0), (1, 0)),), (_line((0, 0), (1, 0)),), tenth
    )


def test_binary64_inputs_really_use_native_with_current_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def observed(*args: object) -> bool:
        calls.append(args)
        return False

    monkeypatch.setattr(distance, "_native_within_paths", observed)
    assert not distance._within_paths_native(
        (((0.0, 0.0), (1.0, 0.0)),), (_line((0, 0), (1, 0)),), 0.2
    )
    assert len(calls) == 1
    assert calls[0][2:] == (0.2, 100_000, 24, 3, 32)


def test_native_handles_finite_extremes_with_conservative_index_fallback() -> None:
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    tiny = float.fromhex("0x0.0000000000001p-1022")
    for x in (largest, -largest, tiny, -tiny):
        path = ((x, 0.0), (x, 1.0))
        spans = (BezierSpan((path[0], path[0], path[-1], path[-1]), True),)
        assert distance._within_paths_native((path,), spans, 0)
        assert distance._within_paths_python((path,), spans, 0)


def test_random_cubics_and_reverse_source_excursions_match_reference() -> None:
    random = Random(734)
    for _ in range(12):
        controls = cast(
            "Controls", tuple((i / 3, random.uniform(-1, 1)) for i in range(4))
        )
        curve = BezierSpan(controls, False)
        path: list[Point2] = []
        for i in range(33):
            t, u = i / 32, 1 - i / 32
            path.append(
                tuple(  # pyright: ignore[reportArgumentType]
                    u**3 * controls[0][j]
                    + 3 * u * u * t * controls[1][j]
                    + 3 * u * t * t * controls[2][j]
                    + t**3 * controls[3][j]
                    for j in (0, 1)
                )
            )
        for excursion in (False, True):
            selected = path.copy()
            if excursion:
                x, y = selected[16]
                selected[16] = (x, y + 0.3)
            assert distance._within_paths_native(
                (selected,), (curve,), 0.02
            ) == distance._within_paths_python((selected,), (curve,), 0.02)


def test_native_proofs_are_independent_across_concurrent_calls() -> None:
    cases = [
        (*quarter(radius), tolerance)
        for radius, tolerance in ((1.0, 0.002), (1.195, 0.2), (1.205, 0.2))
    ]
    expected = [
        distance._within_paths_python((path,), (curve,), tolerance)
        for path, curve, tolerance in cases
    ]

    def run(index: int) -> bool:
        path, curve, tolerance = cases[index % len(cases)]
        return distance._within_paths_native((path,), (curve,), tolerance)

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(run, range(24))) == expected * 8


def test_batch_and_serial_public_paths_preserve_order_and_decisions() -> None:
    cases = [quarter(radius) for radius in (1.0, 1.195, 1.205)]
    jobs: list[distance.DistanceJob] = [
        ((path,), (curve,), 0.2) for path, curve in cases
    ]
    expected = [distance.within_paths(*job) for job in jobs]
    assert distance.within_paths_many(jobs, workers=4) == expected
    assert distance.within_paths_many(jobs, workers=1) == expected
    assert distance.within_paths_many([]) == []


def test_singleton_batch_and_serial_api_do_not_pay_native_overhead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_native(*_args: object) -> bool:
        raise AssertionError("serial checks retain the faster reference")

    monkeypatch.setattr(distance, "_native_within_paths", unexpected_native)
    job: distance.DistanceJob = (
        (((0.0, 0.0), (1.0, 0.0)),),
        (_line((0, 0), (1, 0)),),
        0,
    )
    assert distance.within_paths(*job)
    assert distance.within_paths_many([job]) == [True]
    assert distance.within_paths_many([job, job], workers=1) == [True, True]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_batch_worker_count_rejects(workers: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _ = distance.within_paths_many([], workers=workers)
