"""Indexed nearest projections and envelope derivatives guide geometric fitting."""

from typing import cast

import numpy as np
import pytest

from panelchemy.nearest_polyline import FloatArray, NearestPolyline


def test_projection_uses_original_segment_addresses_and_copies_coordinates() -> None:
    vertices = np.array([(0.1, 0.2), (0.1, 0.2), (1.1, 0.2), (1.1, 0.2), (2.1, 0.2)])
    nearest = NearestPolyline(vertices)
    vertices[:] = 9
    result = nearest.project(np.array([(0.6, 1.2), (1.6, 1.2), (3, 0.2)]))
    assert np.array_equal(result.segments, [1, 3, 3])
    assert np.allclose(result.fractions, [0.5, 0.5, 1])
    assert np.allclose(result.points, [(0.6, 0.2), (1.6, 0.2), (2.1, 0.2)])
    assert np.array_equal(result.points[-1:], [[2.1, 0.2]])


def test_indexed_distances_match_exhaustive_segment_projection() -> None:
    random = np.random.default_rng(634)
    vertices = np.cumsum(random.normal(size=(81, 2)), axis=0)
    queries = random.normal(size=(120, 2)) * 7
    result = NearestPolyline(vertices).project(queries)
    edges = np.diff(vertices, axis=0)
    fractions = np.clip(
        np.sum((queries[:, None] - vertices[None, :-1]) * edges[None], axis=2)
        / np.sum(edges * edges, axis=1),
        0,
        1,
    )
    feet = vertices[None, :-1] + fractions[:, :, None] * edges[None]
    expected = np.min(np.sum((queries[:, None] - feet) ** 2, axis=2), axis=1)
    actual = np.sum((queries - result.points) ** 2, axis=1)
    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert np.all((result.fractions >= 0) & (result.fractions <= 1))


@pytest.mark.parametrize("copies", [1, 2, 5])
def test_degenerate_polyline_is_a_point(copies: int) -> None:
    result = NearestPolyline(np.tile([1.1, 2.2], (copies, 1))).project(
        np.array([(3.1, 4.2), (1.1, 2.2)])
    )
    assert np.array_equal(result.points, [(1.1, 2.2), (1.1, 2.2)])
    assert np.array_equal(result.segments, [0, 0])
    assert np.array_equal(result.fractions, [0, 0])


def test_corner_and_disconnected_ties_have_finite_nearest_feet() -> None:
    vertices = np.array([(0, 0), (0, 2), (2, 2), (2, 0)], dtype=np.float64)
    queries = np.array([(1, 0), (0, -1), (-1, 3)], dtype=np.float64)
    result = NearestPolyline(vertices).project(queries)
    assert np.all(np.isfinite(result.points))
    assert np.allclose(np.sum((queries - result.points) ** 2, axis=1), [1, 1, 2])
    assert np.array_equal(result.points[1:2], [[0, 0]])
    assert np.array_equal(result.points[2:3], [[0, 2]])


@pytest.mark.parametrize("query", [(0.7, 0.8), (-0.4, 0.7), (2.7, -0.3)])
def test_squared_distance_derivative_matches_finite_differences(
    query: tuple[float, float],
) -> None:
    nearest = NearestPolyline(np.array([(0, 0), (2, 0)], dtype=np.float64))
    point = np.array(query)
    foot = cast("FloatArray", nearest.project(point[None]).points[0])

    def distance(at: FloatArray) -> float:
        result = nearest.project(at[None])
        return float(np.sum((at[None] - result.points) ** 2))

    step = 1e-6
    numerical = np.array(
        [
            (distance(point + axis * step) - distance(point - axis * step)) / (2 * step)
            for axis in np.eye(2)
        ]
    )
    assert np.allclose(2 * (point - foot), numerical, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("query", [(0.7, 0.8), (-0.4, 0.7), (2.7, -0.3)])
def test_moving_segment_endpoint_derivative_matches_finite_differences(
    query: tuple[float, float],
) -> None:
    vertices = np.array([(0, 0), (2, 0)], dtype=np.float64)
    point = np.array(query)
    result = NearestPolyline(vertices).project(point[None])
    fraction = cast("float", result.fractions[0])
    error = cast("FloatArray", result.points[0]) - point
    expected = np.array([2 * (1 - fraction) * error, 2 * fraction * error])

    def distance(at: FloatArray) -> float:
        foot = cast(
            "FloatArray",
            NearestPolyline(at.reshape(2, 2)).project(point[None]).points[0],
        )
        return float(np.sum((point - foot) ** 2))

    flat = vertices.ravel()
    step = 1e-6
    numerical = np.array(
        [
            (distance(flat + axis * step) - distance(flat - axis * step)) / (2 * step)
            for axis in np.eye(4)
        ]
    )
    assert np.allclose(expected.ravel(), numerical, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("vertices", [[], [0, 1], [[0, 0, 0]], [[0, np.nan]]])
def test_invalid_vertices_are_rejected(vertices: list[object]) -> None:
    with pytest.raises(ValueError, match="finite planar vertices"):
        _ = NearestPolyline(np.asarray(vertices, dtype=np.float64))


def test_invalid_queries_and_empty_batch() -> None:
    nearest = NearestPolyline(np.array([(0, 0), (1, 0)], dtype=np.float64))
    with pytest.raises(ValueError, match="finite planar points"):
        _ = nearest.project(np.array([[np.nan, 0]]))
    result = nearest.project(np.empty((0, 2)))
    assert result.points.shape == (0, 2)
    assert result.segments.shape == result.fractions.shape == (0,)
