"""Physical stroke seeds and jointly validated source-only short connections."""

# pyright: reportPrivateUsage=false

from itertools import permutations
from math import nextafter
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from panelchemy import raster_retention as rr
from panelchemy.raster_cleanup import cleanup, raster_statistics
from panelchemy.separation import Mask, SeparationConfig, SeparationResult


def source(labels: NDArray[np.uint8]) -> SeparationResult:
    return SeparationResult(
        labels,
        np.ones(labels.shape, dtype=bool),
        SeparationConfig(),
        "explicit-mask",
        [],
        1,
    )


def masks(shape: tuple[int, ...]) -> list[Mask]:
    return [np.zeros(shape, dtype=bool) for _ in range(3)]


def corridor() -> tuple[NDArray[np.uint8], NDArray[np.uint8], list[Mask], list[Mask]]:
    original = np.ones((9, 12), dtype=np.uint8)
    original[4, 1:11] = 2
    current = original.copy()
    current[4, 5:7] = 1
    body, stroke = masks(original.shape), masks(original.shape)
    body[2][4, 1] = True
    stroke[2][4, 10] = True
    return original, current, body, stroke


def bridge(
    original: NDArray[np.uint8],
    current: NDArray[np.uint8],
    body: list[Mask],
    stroke: list[Mask],
    pitch: float = 0.125,
    gap: float = 0.25,
) -> Mask:
    return rr._bridges(
        original,
        current,
        np.ones(original.shape, dtype=bool),
        body,
        stroke,
        pitch,
        gap,
        rr._Work(),
    )[0]


def test_thin_coherent_stroke_survives_while_isolated_detail_is_cleaned() -> None:
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[20:45, 30:55] = 2
    labels[30, 10:31] = 2
    labels[7, 7] = 2
    result = rr.retain_strokes(source(labels), 64, 6, 9, 1)
    assert result.cleaned.labels[30, 15] == 2
    assert result.cleaned.labels[7, 7] == 0
    assert result.cleaned.labels[30, 40] == 2
    assert (
        result.diagnostics["method"] == "body-opening-and-directional-stroke-retention"
    )


def test_input_arrays_outside_silhouette_and_metadata_are_preserved() -> None:
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[8:24, 8:24] = 2
    original = source(labels)
    snapshot = labels.copy()
    result = rr.retain_strokes(original, 32, 4, 6, 1)
    assert np.array_equal(original.labels, snapshot)
    assert np.all(original.silhouette)
    assert not np.shares_memory(result.cleaned.labels, original.labels)
    assert not np.shares_memory(result.cleaned.silhouette, original.silhouette)
    padded = SeparationResult(
        np.pad(labels, 5, constant_values=1),
        np.pad(original.silhouette, 5),
        original.config,
        "explicit-mask",
        [],
        1,
    )
    selected = rr.retain_strokes(padded, 32, 4, 6, 1)
    assert np.array_equal(selected.cleaned.labels[5:-5, 5:-5], result.cleaned.labels)
    assert np.array_equal(
        selected.cleaned.labels[~padded.silhouette], padded.labels[~padded.silhouette]
    )
    assert np.array_equal(selected.cleaned.silhouette, padded.silhouette)


@pytest.mark.parametrize("corner", [(0, 0), (0, -1), (-1, 0), (-1, -1)])
def test_border_retention_is_independent_of_exterior_padding(
    corner: tuple[int, int],
) -> None:
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[corner] = 2
    original = source(labels)
    padded = SeparationResult(
        np.pad(labels, 5, constant_values=1),
        np.pad(original.silhouette, 5),
        original.config,
        "explicit-mask",
        [],
        1,
    )
    direct = rr.retain_strokes(original, 32, 4, 6, 1)
    surrounded = rr.retain_strokes(padded, 32, 4, 6, 1)
    assert direct.cleaned.labels[corner] == 2
    assert np.array_equal(direct.cleaned.labels, surrounded.cleaned.labels[5:-5, 5:-5])


def test_multicell_physical_gap_threshold_is_exact() -> None:
    original, current, body, stroke = corridor()
    selected = bridge(original, current, body, stroke)
    assert selected.sum() == 2 and np.all(selected[4, 5:7])
    assert not np.any(bridge(original, current, body, stroke, gap=nextafter(0.25, 0)))
    assert not np.any(bridge(original, current, body, stroke, pitch=0.25))


def test_foreign_original_pixels_and_unseeded_survivors_are_path_barriers() -> None:
    original, current, body, stroke = corridor()
    original[4, 5] = 0
    assert not np.any(bridge(original, current, body, stroke))
    original, current, body, stroke = corridor()
    current[4, 5] = 2
    current[4, 4] = 1
    assert not np.any(bridge(original, current, body, stroke))


def ties() -> tuple[NDArray[np.uint8], NDArray[np.uint8], list[Mask], list[Mask]]:
    original = np.ones((9, 12), dtype=np.uint8)
    original[2:7, 1:3] = original[2:7, 9:11] = 2
    original[2, 1:11] = original[6, 1:11] = 2
    current = original.copy()
    current[2, 5:7] = current[6, 5:7] = 1
    body, stroke = masks(original.shape), masks(original.shape)
    body[2][4, 10] = stroke[2][4, 1] = True
    return original, current, body, stroke


def test_all_equal_shortest_routes_are_symmetric_and_never_cascade() -> None:
    original, current, body, stroke = ties()
    selected = bridge(original, current, body, stroke)
    assert selected.sum() == 4
    for turn in range(4):
        assert np.array_equal(
            bridge(
                np.rot90(original, turn),
                np.rot90(current, turn),
                [np.rot90(m, turn) for m in body],
                [np.rot90(m, turn) for m in stroke],
            ),
            np.rot90(selected, turn),
        )
    assert np.array_equal(
        bridge(original.T, current.T, [m.T for m in body], [m.T for m in stroke]),
        selected.T,
    )
    current[2, 6] = 2
    shortest = bridge(original, current, body, stroke)
    assert shortest.sum() == 1 and shortest[2, 5]


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_bridge_selection_is_palette_order_independent(order: tuple[int, ...]) -> None:
    original, current, body, stroke = ties()
    permutation = np.asarray(order, dtype=np.uint8)
    expected = bridge(original, current, body, stroke)
    new_body, new_stroke = masks(original.shape), masks(original.shape)
    for color, mapped in enumerate(order):
        new_body[mapped], new_stroke[mapped] = body[color], stroke[color]
    assert np.array_equal(
        bridge(permutation[original], permutation[current], new_body, new_stroke),
        expected,
    )


def guard(
    original: NDArray[np.uint8], before: NDArray[np.uint8], after: NDArray[np.uint8]
) -> list[str]:
    return rr._joint_guard(
        original,
        before,
        after,
        np.ones(original.shape, dtype=bool),
        np.full(original.shape, 255, dtype=np.uint8),
        cast("Mask", before != after),
        rr._Work(),
    )


def test_collective_union_is_checked_for_donor_splitting() -> None:
    before = np.ones((9, 9), dtype=np.uint8)
    before[2, 2:7] = before[6, 2:7] = 0
    before[2:7, 2] = before[2:7, 6] = 0
    original = before.copy()
    original[2, 4] = original[6, 4] = 2
    for row in (2, 6):
        one = before.copy()
        one[row, 4] = 2
        assert not guard(original, before, one)
    assert "donor_empty_or_split" in guard(original, before, original)


def test_new_saddles_and_deleted_donors_are_rejected() -> None:
    before = np.zeros((5, 5), dtype=np.uint8)
    before[2, 2] = 2
    after = before.copy()
    after[1, 1] = 2
    assert "new_saddle" in guard(after, before, after)
    after = before.copy()
    after[2, 2] = 0
    assert "donor_empty_or_split" in guard(after, before, after)


def test_indirect_foreign_original_component_relationship_is_rejected() -> None:
    original = np.ones((7, 13), dtype=np.uint8)
    original[3, 1:12] = 2
    original[1, 2] = original[1, 10] = 2
    before = original.copy()
    before[2, 2] = before[2, 10] = 2
    before[3, 6] = 1
    after = before.copy()
    after[3, 6] = 2
    assert "foreign_original_component_relation" in guard(original, before, after)


def test_preparation_disabled_and_compact_only_are_backward_compatible() -> None:
    labels = np.zeros((16, 16), dtype=np.uint8)
    labels[6, 6] = 2
    original = source(labels)
    assert rr.prepare_raster(original) is None
    expected = cleanup(original, 16, 1)
    actual = rr.prepare_raster(original, long_edge_mm=16, compact_width_mm=1)
    assert actual is not None
    assert np.array_equal(actual.cleaned.labels, expected.cleaned.labels)
    assert actual.diagnostics == expected.diagnostics


def test_composed_report_uses_raw_input_to_final_output() -> None:
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[20:45, 30:55] = 2
    labels[30, 10:31] = labels[7, 7] = 2
    original = source(labels)
    result = rr.prepare_raster(
        original,
        long_edge_mm=64,
        compact_width_mm=1,
        body_width_mm=6,
        stroke_length_mm=9,
        line_width_mm=1,
    )
    assert result is not None and result.original is original
    assert np.array_equal(
        result.changed, cast("Mask", result.cleaned.labels != original.labels)
    )
    assert result.diagnostics["changed_pixels"] == np.count_nonzero(result.changed)
    stages = cast("dict[str, object]", result.diagnostics["stages"])
    assert set(stages) == {"compact", "stroke_retention"}
    assert result.diagnostics["before"] == raster_statistics(original)
    assert result.diagnostics["after"] == raster_statistics(result.cleaned)


@pytest.mark.parametrize("value", [-1, 0, float("nan"), float("inf"), True])
def test_invalid_physical_parameters_are_rejected(value: float) -> None:
    original = source(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        _ = rr.retain_strokes(original, 8, value, 3, 0.5)


def test_resource_exhaustion_is_explicit_and_preserves_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, current, body, stroke = corridor()
    snapshot = current.copy()
    monkeypatch.setattr(rr, "_MAX_WORK", 1)
    with pytest.raises(ValueError, match="work limit"):
        _ = bridge(original, current, body, stroke)
    assert np.array_equal(current, snapshot)
    with pytest.raises(ValueError, match="kernel cell limit"):
        _ = rr.retain_strokes(source(original), 12, 10_000, 3, 0.5)
    monkeypatch.setattr(rr, "_MAX_MORPHOLOGY_WORK", 1)
    with pytest.raises(ValueError, match="morphology work limit"):
        _ = rr.retain_strokes(source(original), 12, 1, 3, 0.5)
