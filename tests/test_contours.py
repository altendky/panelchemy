"""Exact cell areas, holes, registration, and vectorization resource budgets."""

from typing import cast

import numpy as np
import pytest
from PIL import Image
from shapely import union_all
from shapely.geometry import Polygon, box

from panelchemy.contours import vectorize
from panelchemy.separation import (
    Labels,
    Mask,
    SeparationConfig,
    SeparationResult,
    separate,
)


def separated(labels: Labels, silhouette: Mask | None = None) -> SeparationResult:
    palette = np.asarray(SeparationConfig().palette, dtype=np.uint8)
    return separate(
        Image.fromarray(palette[labels]),
        silhouette=np.ones(labels.shape, dtype=bool)
        if silhouette is None
        else silhouette,
    )


def test_selected_panel_scale_origin_and_image_y_inversion() -> None:
    labels = np.zeros((10, 14), dtype=np.uint8)
    silhouette = np.zeros_like(labels, dtype=bool)
    silhouette[2:8, 3:11] = True
    labels[2, 3] = 1
    labels[7, 10] = 2
    contours = vectorize(separated(labels, silhouette), long_edge_mm=80)
    assert contours.mm_per_pixel == 10
    assert contours.source_bounds_px == (3, 2, 11, 8)
    assert contours.silhouette.bounds == (0, 0, 80, 60)
    assert contours.orange[0].equals(box(0, 50, 10, 60))
    assert contours.white[0].equals(box(70, 0, 80, 10))
    assert contours.vertex_count == 12
    assert contours.silhouette.exterior.is_ccw


def test_color_counters_holes_and_single_pixel_strokes_survive() -> None:
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[1:4, 1:4] = 1
    labels[2, 2] = 0
    labels[5:9, 5:9] = 2
    labels[6:8, 6:8] = 0
    labels[1, 5:9] = 2
    contours = vectorize(separated(labels), long_edge_mm=12)
    assert contours.silhouette.area == 144
    assert sum(polygon.area for polygon in contours.orange) == 8
    assert sum(polygon.area for polygon in contours.white) == 16
    assert len(contours.orange) == 1
    assert len(contours.orange[0].interiors) == 1
    assert len(contours.white) == 2
    assert sum(len(polygon.interiors) for polygon in contours.white) == 1
    assert contours.vertex_count == 24
    assert all(
        contours.silhouette.covers(polygon)
        for polygon in contours.orange + contours.white
    )


def test_collinear_seams_are_removed_without_changing_geometry() -> None:
    labels = np.zeros((40, 100), dtype=np.uint8)
    contours = vectorize(separated(labels), long_edge_mm=100, max_vertices=4)
    assert contours.silhouette.equals(box(0, 0, 100, 40))
    assert len(contours.silhouette.exterior.coords) == 5
    assert contours.vertex_count == 4
    assert contours.orange == []
    assert contours.white == []


def test_diagonal_color_islands_are_kept_separately() -> None:
    labels = np.zeros((4, 4), dtype=np.uint8)
    labels[1, 1] = labels[2, 2] = 2
    contours = vectorize(separated(labels), long_edge_mm=4)
    assert len(contours.white) == 2
    assert sum(polygon.area for polygon in contours.white) == 2
    assert contours.vertex_count == 12


def test_clipped_outline_matches_exact_pixel_cell_union() -> None:
    silhouette = np.array(
        [
            [False, True, True, False],
            [True, True, True, True],
            [True, True, True, True],
            [False, True, True, True],
        ],
        dtype=bool,
    )
    labels = np.zeros(silhouette.shape, dtype=np.uint8)
    contours = vectorize(separated(labels, silhouette), long_edge_mm=4)
    selected_cells: list[list[int]] = np.argwhere(silhouette).tolist()
    expected = union_all(
        [box(column, 3 - row, column + 1, 4 - row) for row, column in selected_cells]
    )
    assert contours.silhouette.equals(expected)
    assert contours.silhouette.area == np.count_nonzero(silhouette)


def test_disconnected_or_holed_silhouettes_are_rejected_without_filtering() -> None:
    labels = np.zeros((5, 5), dtype=np.uint8)
    disconnected = np.zeros_like(labels, dtype=bool)
    disconnected[0, 0] = disconnected[1, 1] = True
    with pytest.raises(ValueError, match="one connected panel"):
        _ = vectorize(separated(labels, disconnected))
    holed = np.ones_like(labels, dtype=bool)
    holed[2, 2] = False
    with pytest.raises(ValueError, match="hole-free"):
        _ = vectorize(separated(labels, holed))


def test_vertex_budget_rejects_before_constructing_polygons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[1:-1:2, 1:-1:2] = 2

    def unexpected_union(_geometries: object) -> None:
        raise AssertionError(
            "polygon union must not start when the corner budget fails"
        )

    monkeypatch.setattr("panelchemy.contours.union_all", unexpected_union)
    with pytest.raises(
        ValueError, match="exceeding max_vertices=100; no features were removed"
    ):
        _ = vectorize(separated(labels), max_vertices=100)


def test_long_stripes_do_not_allocate_one_rectangle_per_scan_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.zeros((300, 64), dtype=np.uint8)
    labels[:, 10] = labels[:, 20] = 2
    calls = 0

    def counted_box(minx: float, miny: float, maxx: float, maxy: float) -> Polygon:
        nonlocal calls
        calls += 1
        return box(minx, miny, maxx, maxy)

    monkeypatch.setattr("panelchemy.contours.box", counted_box)
    contours = vectorize(separated(labels), long_edge_mm=300, max_vertices=12)
    assert calls == 3
    assert contours.silhouette.equals(box(0, 0, 64, 300))
    assert sum(polygon.area for polygon in contours.white) == 600


@pytest.mark.parametrize("edge", [0, -1, float("nan"), float("inf"), True, 10**1000])
def test_invalid_physical_edge_is_rejected(edge: object) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        _ = vectorize(separated(np.zeros((2, 2), dtype=np.uint8)), cast("float", edge))


@pytest.mark.parametrize("budget", [0, -1, True, 4.5])
def test_invalid_vertex_budget_is_rejected(budget: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _ = vectorize(
            separated(np.zeros((2, 2), dtype=np.uint8)),
            max_vertices=cast("int", budget),
        )


def test_actual_minimum_corner_budget_is_enforced() -> None:
    result = separated(np.zeros((1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="require 4 vertices"):
        _ = vectorize(result, max_vertices=3)
    assert vectorize(result, max_vertices=4).vertex_count == 4
