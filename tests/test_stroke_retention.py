"""Retain coherent stroke bodies with bounded cell growth and preserved gaps."""

# pyright: reportPrivateUsage=false

from dataclasses import replace
from hashlib import sha256
from math import sqrt
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import ndimage

from panelchemy import stroke_retention as retention
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


def line(width: int = 2) -> SeparationResult:
    labels = np.zeros((80, 120), dtype=np.uint8)
    labels[39 : 39 + width, 25:95] = 2
    return source(labels)


def test_normal_crossings_measure_full_cells_and_even_width_center_offset() -> None:
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[3:5, :] = 2
    positive = retention._ray_runs(grid, (3.5, 5.5), (1, 0), 4)
    negative = retention._ray_runs(grid, (3.5, 5.5), (-1, 0), 4)
    assert positive[0] == (2, 0, 1.5)
    assert negative[0] == (2, 0, 0.5)
    assert positive[0][2] + negative[0][2] == 2
    assert 3.5 + (positive[0][2] - negative[0][2]) / 2 == 4


def test_normal_ray_does_not_measure_through_a_diagonal_pinch() -> None:
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[1, 1] = grid[2, 2] = 2
    selected = retention._ray_runs(grid, (1.5, 1.5), (1 / sqrt(2), 1 / sqrt(2)), 3)
    assert selected[0][2] == pytest.approx(1 / sqrt(2))
    assert selected[1][0] == -2


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_odd_and_even_strokes_gain_a_print_scale_body_and_keep_terminal_pixels(
    width: int,
) -> None:
    original = line(width)
    result = retention.preview_stroke_growth(original, long_edge_mm=12)
    assert result.source_reference is original
    assert np.any(result.changed)
    # At .1mm/pixel, 5 or6 whole cells cover the .5mm tube, depending on parity.
    assert 5 <= np.count_nonzero(cast("Mask", result.proposed.labels[:, 60] == 2)) <= 6
    assert np.array_equal(result.proposed.labels[:, 25], original.labels[:, 25])
    assert np.array_equal(result.proposed.labels[:, 94], original.labels[:, 94])
    assert np.array_equal(result.proposed.silhouette, original.silhouette)
    assert any(d["outcome"] == "proposed-growth" for d in result.decisions)
    distances = ndimage.distance_transform_edt(
        cast("Mask", result.reference.labels != 2)
    )
    assert np.max(distances[result.changed]) * 0.1 <= 0.25


@pytest.mark.parametrize("flip", [False, True])
def test_physical_result_survives_mirroring_and_transposition(flip: bool) -> None:
    original = line(2)
    labels = original.labels.T.copy() if flip else np.flipud(original.labels).copy()
    result = retention.preview_stroke_growth(source(labels), long_edge_mm=12)
    assert np.any(result.changed)
    assert any(d["outcome"] == "proposed-growth" for d in result.decisions)
    for decision in result.decisions:
        if decision["outcome"] == "proposed-growth":
            assert cast("float", decision["growth_displacement_upper_bound_mm"]) <= 0.25


def test_coarse_processing_grid_is_exact_nearest_cell_refinement() -> None:
    original = line(1)
    result = retention.preview_stroke_growth(original, long_edge_mm=24)
    assert result.diagnostics["working_scale_factor"] == 2
    expected = np.repeat(np.repeat(original.labels, 2, axis=0), 2, axis=1)
    assert np.array_equal(result.reference.labels, expected)
    assert result.diagnostics["source_mm_per_pixel"] == 0.2
    assert result.diagnostics["mm_per_pixel"] == 0.1
    assert (
        result.diagnostics["source_reference_labels_sha256"]
        == sha256(original.labels.tobytes()).hexdigest()
    )
    assert np.any(result.changed)


def test_a_short_terminal_tip_on_a_substantial_body_stays_unchanged() -> None:
    labels = np.zeros((80, 120), dtype=np.uint8)
    labels[30:50, 20:50] = 2
    labels[39:41, 50:54] = 2
    result = retention.preview_stroke_growth(source(labels), long_edge_mm=12)
    assert np.array_equal(result.proposed.labels, labels)
    assert any(d.get("reason") == "short-terminal-or-noise" for d in result.decisions)


def test_growth_cannot_make_a_printable_neighboring_gap_too_narrow() -> None:
    original = line(2)
    original.labels[47:60, 20:100] = 2  # .6mm gap below the thin line.
    result = retention.preview_stroke_growth(original, long_edge_mm=12)
    assert any(d.get("reason") == "protected-gap" for d in result.decisions)
    assert np.array_equal(
        result.proposed.labels[39:47, 35:85], original.labels[39:47, 35:85]
    )


def test_approaching_an_unrelated_region_or_third_color_is_vetoed() -> None:
    original = line(2)
    original.labels[38, 60] = 1
    result = retention.preview_stroke_growth(original, long_edge_mm=12)
    assert not np.any(result.changed)
    assert any(
        d.get("reason") in {"multiple-donor-colors", "third-color-contact"}
        for d in result.decisions
    )


@pytest.mark.parametrize("long_edge_mm", [12, 24])
def test_panel_boundary_and_exterior_are_fixed(long_edge_mm: float) -> None:
    labels = np.zeros((80, 120), dtype=np.uint8)
    labels[1:3, 25:95] = 2
    result = retention.preview_stroke_growth(source(labels), long_edge_mm=long_edge_mm)
    assert not np.any(result.changed)
    assert any(d.get("reason") == "protected-panel-boundary" for d in result.decisions)
    original = line()
    silhouette = original.silhouette.copy()
    silhouette[:, :10] = False
    original = replace(original, silhouette=silhouette)
    result = retention.preview_stroke_growth(original, long_edge_mm=11)
    assert np.array_equal(
        result.proposed.labels[~silhouette], original.labels[~silhouette]
    )


def test_analysis_prunes_only_short_raw_spurs_and_keeps_original_tip_guards() -> None:
    skeleton = np.zeros((20, 60), dtype=bool)
    skeleton[10, 3:57] = True
    skeleton[8:10, 15] = skeleton[8:10, 30] = skeleton[8:10, 45] = True
    atlas = np.ones(skeleton.shape, dtype=np.int32)
    snapshot = skeleton.copy()
    raw_paths, raw_tips = retention._branches(skeleton, atlas)
    paths, tips = retention._branches(skeleton, atlas, prune_length_px=3)
    assert len(raw_paths) > 1
    assert len(paths) == 1
    assert len(paths[0]) == 54
    assert tips == raw_tips
    assert np.array_equal(skeleton, snapshot)
    # Equality with the cutoff retains a genuine fork; no recursive peeling.
    at_limit, _ = retention._branches(skeleton, atlas, prune_length_px=2)
    assert len(at_limit) == len(raw_paths)


def test_final_audit_protects_a_donor_on_its_own_normal_not_only_connectivity() -> None:
    grid = np.ones((12, 32), dtype=np.int32)
    grid[5:7, 2:30] = 2

    def sample(
        origin: retention.Point, normal: retention.Point, position: float
    ) -> retention._Sample:
        rays = (
            retention._ray_runs(grid, origin, normal, 15),
            retention._ray_runs(grid, origin, (-normal[0], -normal[1]), 15),
        )
        return retention._Sample(
            origin,
            normal,
            origin,
            sum(side[0][2] for side in rays) * 0.1,
            rays,
            position,
        )

    # An adversarial orange edit narrows the middle of a white bar without
    # disconnecting either color or affecting the orange stroke's own normals.
    donor = sample((6.5, 16.5), (1, 0), 1)
    proposal = retention._Proposal(
        0,
        1,
        1,
        2,
        ((5, 16),),
        (
            sample((2.5, 15.5), (0, 1), 0.5),
            sample((2.5, 16.5), (0, 1), 1.5),
        ),
    )
    trial = grid.copy()
    trial[5, 16] = 1
    assert ndimage.label(cast("Mask", trial == 2))[1] == 1
    assert retention._gap_guard(trial, proposal, 0.1, 0.5)[0]
    accepted = {0}
    decisions: list[dict[str, object]] = [dict(outcome="proposed-growth")]
    retention._audit_growth(
        trial, grid, [proposal], accepted, [(2, donor)], decisions, 0.1, 0.5
    )
    assert not accepted
    assert np.array_equal(trial, grid)
    assert decisions[0]["reason"] == "final-neighbor-protection"


def test_growth_is_deterministic_and_never_mutates_its_reference() -> None:
    original = line()
    snapshot = original.labels.copy()
    first = retention.preview_stroke_growth(original, long_edge_mm=12)
    second = retention.preview_stroke_growth(original, long_edge_mm=12)
    assert np.array_equal(first.proposed.labels, second.proposed.labels)
    assert first.diagnostics == second.diagnostics
    assert first.decisions == second.decisions
    assert np.array_equal(original.labels, snapshot)
    assert np.array_equal(first.reference.labels, snapshot)
    assert first.diagnostics["preview_only"] is True
    assert first.diagnostics["semantic_importance_inferred"] is False


def test_zero_growth_is_analysis_only_and_a_smaller_cap_is_not_ignored() -> None:
    original = line()
    disabled = retention.preview_stroke_growth(
        original, long_edge_mm=12, max_growth_mm=0
    )
    assert not np.any(disabled.changed)
    assert any(d.get("reason") == "growth-disabled" for d in disabled.decisions)
    bounded = retention.preview_stroke_growth(
        original, long_edge_mm=12, max_growth_mm=0.05
    )
    assert not np.any(bounded.changed)


@pytest.mark.parametrize(
    "settings",
    [
        {"long_edge_mm": 0},
        {"long_edge_mm": True},
        {"long_edge_mm": float("inf")},
        {"target_width_mm": 0},
        {"target_width_mm": True},
        {"target_width_mm": float("nan")},
        {"max_growth_mm": -0.1},
        {"max_growth_mm": True},
        {"max_growth_mm": float("inf")},
        {"target_width_mm": 1e-300},
        {"long_edge_mm": 1e308, "target_width_mm": 1e308},
    ],
)
def test_invalid_or_unrepresentable_parameters_reject(
    settings: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        _ = retention.preview_stroke_growth(line(), **settings)
