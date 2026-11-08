"""Physical, coordinated cleanup changes explicit details and keeps useful tips."""

from itertools import permutations
from math import nextafter
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from panelchemy.raster_cleanup import cleanup
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


def empty() -> NDArray[np.uint8]:
    return np.zeros((16, 16), dtype=np.uint8)


def test_compact_island_and_enclosed_counter_relabel_to_their_surrounding_color() -> (
    None
):
    labels = empty()
    labels[3, 3] = 2
    labels[8:14, 8:14] = 2
    labels[10, 10] = 0
    original = source(labels)
    result = cleanup(original, 16, 1)
    assert result.cleaned.labels[3, 3] == 0
    assert result.cleaned.labels[10, 10] == 2
    assert np.count_nonzero(result.changed) == 2
    assert result.diagnostics["accepted_edits"] == {"compact_component": 2}
    assert np.count_nonzero(cast("Mask", result.cleaned.labels == 2)) == 36


def test_zero_cutoff_preserves_all_pixels_and_inputs_are_never_mutated() -> None:
    labels = empty()
    labels[3, 3] = 2
    original = source(labels)
    snapshot = original.labels.copy(), original.silhouette.copy()
    disabled = cleanup(original, 16, 0)
    assert not np.any(disabled.changed)
    assert np.array_equal(disabled.cleaned.labels, original.labels)
    first, second = cleanup(original, 16, 1), cleanup(original, 16, 1)
    assert np.array_equal(first.cleaned.labels, second.cleaned.labels)
    assert first.diagnostics == second.diagnostics
    assert np.array_equal(original.labels, snapshot[0])
    assert np.array_equal(original.silhouette, snapshot[1])
    assert not np.shares_memory(first.cleaned.labels, original.labels)
    assert not np.shares_memory(first.cleaned.silhouette, original.silhouette)


def test_attached_points_and_long_thin_strokes_stay_pixel_exact() -> None:
    labels = empty()
    labels[3:8, 3:8] = 2
    labels[8, 5] = labels[9, 5] = 2
    labels[12, 3:13] = 1
    result = cleanup(source(labels), 16, 2)
    assert not np.any(result.changed)
    assert np.array_equal(result.cleaned.labels, labels)


def test_cell_extent_uses_physical_scale_and_includes_the_whole_pixel() -> None:
    labels = empty()
    labels[4, 4] = 2
    assert cleanup(source(labels), 16, 1).changed[4, 4]
    assert not cleanup(source(labels), 16, nextafter(1.0, 0.0)).changed[4, 4]
    assert not cleanup(source(labels), 32, 1).changed[4, 4]


def test_exterior_padding_never_changes_physical_scale_or_cleanup() -> None:
    labels = empty()
    labels[4, 4] = 2
    first = cleanup(source(labels), 16, 1)
    padded = np.pad(labels, 7, constant_values=2)
    silhouette = np.pad(np.ones(labels.shape, dtype=bool), 7)
    original = SeparationResult(
        padded, silhouette, SeparationConfig(), "explicit-mask", [], 1
    )
    second = cleanup(original, 16, 1)
    assert second.diagnostics["mm_per_pixel"] == first.diagnostics["mm_per_pixel"]
    assert np.array_equal(second.cleaned.labels[7:-7, 7:-7], first.cleaned.labels)
    assert np.array_equal(second.cleaned.labels[~silhouette], padded[~silhouette])
    assert np.array_equal(second.cleaned.silhouette, silhouette)


def test_panel_edge_pixels_and_three_color_junctions_are_preserved() -> None:
    labels = empty()
    labels[0, 3] = 2
    labels[5, 5] = 2
    labels[5, 6:11] = 1
    result = cleanup(source(labels), 16, 1)
    assert not np.any(result.changed)
    assert np.array_equal(result.cleaned.labels, labels)


def contact() -> NDArray[np.uint8]:
    labels = empty()
    labels[3:7, 3:7] = 2
    labels[7, 7] = 2
    return labels


def test_unambiguous_diagonal_attachment_joins_with_both_bridge_pixels() -> None:
    labels = contact()
    result = cleanup(source(labels), 16, 1)
    assert result.cleaned.labels[7, 7] == 2
    assert result.cleaned.labels[6, 7] == result.cleaned.labels[7, 6] == 2
    assert np.count_nonzero(result.changed) == 2
    assert result.diagnostics["accepted_edits"] == {"point_contact": 1}
    assert np.all(
        result.cleaned.masks["support-white"] <= result.cleaned.masks["support-orange"]
    )
    assert np.array_equal(
        sum(
            mask.astype(int)
            for key, mask in result.cleaned.masks.items()
            if key in {"black", "orange", "white"}
        ),
        result.cleaned.silhouette.astype(int),
    )


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_eligible_contact_is_independent_of_palette_order(
    order: tuple[int, ...],
) -> None:
    labels = contact()
    mapping = np.asarray(order, dtype=np.uint8)
    expected = cleanup(source(labels), 16, 1).cleaned.labels
    selected = cleanup(source(mapping[labels]), 16, 1)
    assert np.array_equal(selected.cleaned.labels, mapping[expected])


def test_third_color_near_a_contact_requires_an_explicit_interpretation() -> None:
    labels = contact()
    labels[8, 6] = 1
    result = cleanup(source(labels), 16, 1)
    assert not np.any(result.changed)


def test_far_end_of_compact_child_cannot_create_a_third_color_relationship() -> None:
    labels = contact()
    labels[7, 7:10] = 2
    labels[8, 10] = 1
    result = cleanup(source(labels), 16, 3)
    assert not np.any(result.changed)


def test_a_compact_child_cannot_bridge_two_substantial_parents() -> None:
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[3:10, 3:10] = 2
    labels[10, 10:16] = 2
    labels[11:18, 16:23] = 2
    result = cleanup(source(labels), 32, 6)
    assert not np.any(result.changed)


def test_a_contact_may_not_sever_the_surrounding_color() -> None:
    labels = np.full((16, 16), 2, dtype=np.uint8)
    labels[5:8, 5:8] = 0
    labels[6, 6] = labels[5, 5] = 2
    labels[2:6, 6] = 0
    labels[6, 2:6] = 0
    result = cleanup(source(labels), 16, 1)
    assert not np.any(result.changed)
    assert (
        cast("dict[str, int]", result.diagnostics["skipped"])["donor_connectivity"] == 1
    )


def test_conflicting_nearby_proposals_are_both_rejected() -> None:
    labels = empty()
    labels[5, 5] = labels[5, 7] = 2
    result = cleanup(source(labels), 16, 1)
    assert not np.any(result.changed)
    assert (
        cast("dict[str, int]", result.diagnostics["skipped"])["conflicting_proposals"]
        == 2
    )


@pytest.mark.parametrize(
    "edge,width",
    [
        (0, 1),
        (-1, 1),
        (True, 1),
        (float("inf"), 1),
        (16, -1),
        (16, True),
        (16, float("nan")),
        (16, float("inf")),
        (10**1000, 1),
    ],
)
def test_invalid_physical_parameters_reject(edge: float, width: float) -> None:
    with pytest.raises(ValueError):
        _ = cleanup(source(empty()), edge, width)


def test_invalid_label_arrays_and_empty_silhouettes_reject() -> None:
    wrong = source(empty())
    wrong.labels[4, 4] = 3
    with pytest.raises(ValueError, match="palette"):
        _ = cleanup(wrong, 16, 1)
    wrong = source(empty())
    wrong.silhouette[:] = False
    with pytest.raises(ValueError, match="nonempty"):
        _ = cleanup(wrong, 16, 1)
