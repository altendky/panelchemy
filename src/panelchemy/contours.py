"""Exact raster-cell contours, registered in millimeters with positive Y up."""

from dataclasses import dataclass
from math import isfinite
from typing import cast

import numpy as np
from scipy import ndimage
from shapely import union_all
from shapely.coords import CoordinateSequence
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from panelchemy.separation import Mask, SeparationResult


@dataclass(frozen=True)
class PanelContours:
    """Shared origin, scale, and exact cell footprints for relief construction.

    ``source_bounds_px`` is (left, top, right, bottom) in source image-cell
    coordinates. The silhouette's lower-left world bound is (0, 0), so image
    rows are inverted. ``vertex_count`` excludes repeated ring-closing points.
    """

    silhouette: Polygon
    orange: list[Polygon]
    white: list[Polygon]
    mm_per_pixel: float
    source_bounds_px: tuple[float, float, float, float]
    vertex_count: int


def _boundary_vertices(mask: Mask) -> int:
    """Count exact cell-union corners before allocating any polygon objects.

    At a grid vertex, one or three occupied neighbors produce one corner.
    Two diagonal occupied neighbors produce two corners, one for each region.
    Adjacent occupied pairs produce a straight edge and therefore no corner.
    """
    padded = np.pad(mask, 1)
    northwest = padded[:-1, :-1]
    northeast = padded[:-1, 1:]
    southwest = padded[1:, :-1]
    southeast = padded[1:, 1:]
    occupied = northwest.astype(np.uint8) + northeast + southwest + southeast
    turning_corners = cast("Mask", (occupied == 1) | (occupied == 3))
    corners = np.count_nonzero(turning_corners)
    diagonal_pairs = cast(
        "Mask",
        (occupied == 2) & ((northwest & southeast) | (northeast & southwest)),
    )
    return int(corners + 2 * np.count_nonzero(diagonal_pairs))


def _clean_ring(coordinates: CoordinateSequence) -> list[tuple[float, float]]:
    """Remove only exact duplicates and points on unchanged straight segments."""
    points: list[tuple[float, float]] = []
    for coordinate in coordinates:
        point = (float(coordinate[0]), float(coordinate[1]))
        if not points or point != points[-1]:
            points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        _ = points.pop()
    cleaned: list[tuple[float, float]] = []
    for index, current in enumerate(points):
        previous, following = points[index - 1], points[(index + 1) % len(points)]
        before = (current[0] - previous[0], current[1] - previous[1])
        after = (following[0] - current[0], following[1] - current[1])
        cross = before[0] * after[1] - before[1] * after[0]
        dot = before[0] * after[0] + before[1] * after[1]
        if cross != 0 or dot < 0:
            cleaned.append(current)
    if len(cleaned) < 3:
        raise ValueError("raster boundary did not produce a closed polygon ring")
    return cleaned


def _polygons(mask: Mask) -> list[Polygon]:
    """Union rectangles for contiguous true runs, preserving every raster cell."""
    rectangles: list[Polygon] = []
    # Scan the shorter axis and coalesce identical runs across neighboring rows.
    # Long stripes therefore require one rectangle each, not one per scan line.
    shape = cast("tuple[int, ...]", mask.shape)
    transposed = shape[0] > shape[1]
    scan_rows = mask.T if transposed else mask
    active: dict[tuple[int, int], int] = {}

    def finish(interval: tuple[int, int], start_row: int, stop_row: int) -> None:
        left, right = interval
        if transposed:
            rectangles.append(box(start_row, left, stop_row, right))
        else:
            rectangles.append(box(left, start_row, right, stop_row))

    for row_index in range(shape[1] if transposed else shape[0]):
        row = cast("Mask", scan_rows[row_index])
        transitions: list[int] = np.flatnonzero(np.diff(np.pad(row, 1))).tolist()
        runs = list(zip(transitions[::2], transitions[1::2], strict=True))
        present = set(runs)
        for interval in list(active):
            if interval not in present:
                finish(interval, active.pop(interval), row_index)
        for interval in runs:
            _ = active.setdefault(interval, row_index)
    for interval, start_row in active.items():
        finish(interval, start_row, len(scan_rows))
    if not rectangles:
        return []
    geometry: BaseGeometry = union_all(rectangles)
    if isinstance(geometry, Polygon):
        parts = [geometry]
    elif isinstance(geometry, MultiPolygon):
        parts = list(geometry.geoms)
    else:
        raise ValueError("raster mask did not produce polygonal contours")
    cleaned: list[Polygon] = []
    for part in parts:
        polygon = Polygon(
            _clean_ring(part.exterior.coords),
            [_clean_ring(ring.coords) for ring in part.interiors],
        )
        # Both geometries are still in integer cell coordinates: equality is
        # checked before millimeter scaling introduces any rounding.
        if not polygon.is_valid or not polygon.equals(part):
            raise ValueError("exact collinear removal changed the raster geometry")
        cleaned.append(polygon)
    return sorted(
        cleaned, key=lambda polygon: (polygon.bounds, polygon.area, polygon.wkb)
    )


def _physical(polygon: Polygon, scale: float, height: int) -> Polygon:
    def transform(coordinates: CoordinateSequence) -> list[tuple[float, float]]:
        return [(float(x) * scale, (height - float(y)) * scale) for x, y in coordinates]

    transformed = orient(
        Polygon(
            transform(polygon.exterior.coords),
            [transform(ring.coords) for ring in polygon.interiors],
        ),
        sign=1.0,
    )
    if (
        not transformed.is_valid
        or not isfinite(transformed.area)
        or transformed.area <= 0
    ):
        raise ValueError("physical scale cannot represent these contours finitely")
    return transformed


def _vertex_count(polygon: Polygon) -> int:
    return (
        len(polygon.exterior.coords)
        - 1
        + sum(len(ring.coords) - 1 for ring in polygon.interiors)
    )


def vectorize(
    result: SeparationResult,
    long_edge_mm: float = 250,
    max_vertices: int = 25_000,
) -> PanelContours:
    """Preserve exact pixel-cell regions and refuse unsupported topology or size.

    The selected silhouette's bounding-box long edge defines physical scale;
    external gutter pixels never affect the resulting panel dimensions. There
    is no simplification, island selection, gap closing, or feature filtering.
    """
    edge_value = cast("object", long_edge_mm)
    if isinstance(edge_value, bool) or not isinstance(edge_value, (int, float)):
        raise ValueError("long_edge_mm must be a positive finite number")
    try:
        physical_edge = float(edge_value)
    except OverflowError as error:
        raise ValueError("long_edge_mm must be a positive finite number") from error
    if not isfinite(physical_edge) or physical_edge <= 0:
        raise ValueError("long_edge_mm must be a positive finite number")
    if type(max_vertices) is not int or max_vertices < 1:
        raise ValueError("max_vertices must be a positive integer")
    silhouette = result.silhouette
    if (
        silhouette.ndim != 2
        or result.labels.shape != silhouette.shape
        or silhouette.dtype != np.bool_
    ):
        raise ValueError(
            "labels and silhouette must be same-size two-dimensional masks"
        )
    if not np.any(silhouette):
        raise ValueError("silhouette must contain at least one pixel")
    if np.any((result.labels < 0) | (result.labels > 2)):
        raise ValueError("palette labels must be black, orange, or white")
    structure = ndimage.generate_binary_structure(2, 1)
    if int(ndimage.label(silhouette, structure=structure)[1]) != 1:
        raise ValueError(
            "silhouette must be one connected panel; no components were removed"
        )
    holes = cast(
        "Mask", ndimage.binary_fill_holes(silhouette, structure=structure) != silhouette
    )
    if np.any(holes):
        raise ValueError(
            "silhouette must be hole-free; black counters belong to artwork, not boundary holes"
        )
    rows: list[int] = np.flatnonzero(np.any(silhouette, axis=1)).tolist()
    columns: list[int] = np.flatnonzero(np.any(silhouette, axis=0)).tolist()
    left, right = columns[0], columns[-1] + 1
    top, bottom = rows[0], rows[-1] + 1
    height, width = bottom - top, right - left
    selected = silhouette[top:bottom, left:right]
    labels = result.labels[top:bottom, left:right]
    masks = [
        selected,
        cast("Mask", selected & (labels == 1)),
        cast("Mask", selected & (labels == 2)),
    ]
    required_vertices = sum(_boundary_vertices(mask) for mask in masks)
    if required_vertices > max_vertices:
        raise ValueError(
            f"contours require {required_vertices:,} vertices, exceeding max_vertices={max_vertices:,}; no features were removed"
        )
    pixel_polygons = [_polygons(mask) for mask in masks]
    if len(pixel_polygons[0]) != 1 or pixel_polygons[0][0].interiors:
        raise ValueError("silhouette must be a single hole-free polygon")
    scale = physical_edge / max(width, height)
    if not isfinite(scale) or scale <= 0:
        raise ValueError("physical scale cannot represent these contours finitely")
    physical = [
        [_physical(polygon, scale, height) for polygon in polygons]
        for polygons in pixel_polygons
    ]
    vertex_count = sum(
        _vertex_count(polygon) for polygons in physical for polygon in polygons
    )
    if vertex_count != required_vertices:
        raise ValueError(
            "raster corner accounting disagrees with exact polygon contours"
        )
    return PanelContours(
        silhouette=physical[0][0],
        orange=physical[1],
        white=physical[2],
        mm_per_pixel=scale,
        source_bounds_px=(float(left), float(top), float(right), float(bottom)),
        vertex_count=vertex_count,
    )
