"""Exact regional division for native expanding-distance roofs.

At every height, a distance expansion distributes over a union of source sets.
Coarse overlapping source pieces can therefore be built independently and fused
without changing the selected top or the intended taper.
"""

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from math import isfinite, ulp
from typing import cast

from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Trsf, gp_Vec
from OCP.Standard import Standard_Failure, Standard_OutOfMemory
from OCP.TopoDS import TopoDS_Shape
from shapely import union_all
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry

from panelchemy import occt

type PieceBuilder = Callable[[Polygon, dict[str, object]], tuple[TopoDS_Shape, str]]
type PieceValidator = Callable[[TopoDS_Shape, Polygon], None]


def _edges(polygon: Polygon) -> int:
    return (
        len(polygon.exterior.coords)
        - 1
        + sum(len(ring.coords) - 1 for ring in polygon.interiors)
    )


def _parts(geometry: BaseGeometry) -> Iterator[Polygon]:
    if geometry.is_empty:
        return
    if not geometry.is_valid:
        raise occt.KernelError("spatial clipping produced an invalid source piece")
    if geometry.area == 0:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, (MultiPolygon, GeometryCollection)):
        for child in geometry.geoms:
            yield from _parts(cast(BaseGeometry, child))
    else:
        raise occt.KernelError(
            "spatial clipping produced an unsupported source geometry"
        )


def _coordinate_roundoff(polygon: Polygon) -> float:
    return 128 * ulp(max(1.0, *(abs(value) for value in polygon.bounds)))


def _check_cover(source: Polygon, pieces: Sequence[Polygon]) -> dict[str, object]:
    """Prove the cover and its boundary agree, without cleaning either set."""
    if not pieces or any(not piece.is_valid or piece.area <= 0 for piece in pieces):
        raise occt.KernelError("spatial roof needs valid positive source pieces")
    cover = union_all(pieces)
    linear = _coordinate_roundoff(source)
    area_allowance = max(1e-12, linear * source.length * 8)
    difference = source.symmetric_difference(cover).area
    boundary_distance = source.boundary.hausdorff_distance(cover.boundary)
    if (
        not isfinite(difference)
        or not isfinite(boundary_distance)
        or difference > area_allowance
        or boundary_distance > linear
        or any(piece.difference(source).area > area_allowance for piece in pieces)
    ):
        raise occt.KernelError(
            "spatial pieces do not preserve the original source cover"
        )
    return {
        "spatial_cover_difference_mm2": difference,
        "spatial_cover_area_allowance_mm2": area_allowance,
        "spatial_cover_boundary_distance_mm": boundary_distance,
        "spatial_cover_boundary_allowance_mm": linear,
    }


def _initial_pieces(source: Polygon, grid: int, overlap: float) -> Iterator[Polygon]:
    lx, ly, hx, hy = source.bounds
    half = overlap / 2
    for x in range(grid):
        for y in range(grid):
            yield from _parts(
                source.intersection(
                    box(
                        lx + (hx - lx) * x / grid - half,
                        ly + (hy - ly) * y / grid - half,
                        lx + (hx - lx) * (x + 1) / grid + half,
                        ly + (hy - ly) * (y + 1) / grid + half,
                    )
                )
            )


def _bisect(source: Polygon, overlap: float) -> list[Polygon]:
    lx, ly, hx, hy = source.bounds
    axis = 0 if hx - lx >= hy - ly else 1
    extent = (hx - lx, hy - ly)[axis]
    # An overlapping bisection cannot shrink a piece this small. This is a
    # construction resource bound, never permission to discard its material.
    if extent <= overlap * (1 + 1e-12):
        raise occt.KernelError("spatial piece reached its minimum subdivision extent")
    middle = ((lx + hx), (ly + hy))[axis] / 2
    half = overlap / 2
    clips = (
        (box(lx, ly, middle + half, hy), box(middle - half, ly, hx, hy))
        if axis == 0
        else (box(lx, ly, hx, middle + half), box(lx, middle - half, hx, hy))
    )
    pieces = [part for clip in clips for part in _parts(source.intersection(clip))]
    if not pieces or any(
        piece.bounds[axis + 2] - piece.bounds[axis] >= extent for piece in pieces
    ):
        raise occt.KernelError("spatial subdivision did not strictly shrink its pieces")
    _ = _check_cover(source, pieces)
    return pieces


def _positive(shape: TopoDS_Shape) -> None:
    if not occt.valid(shape) or occt.volume(shape) <= 0:
        raise occt.KernelError(
            "spatial construction did not return a valid positive solid"
        )


def _piece_roof(
    source: Polygon,
    builder: PieceBuilder,
    validate: PieceValidator,
    record: dict[str, object],
) -> TopoDS_Shape:
    diagnostics: dict[str, object] = {}
    record["global_diagnostics"] = diagnostics
    try:
        shape, method = builder(source, diagnostics)
        _positive(shape)
        record.update(frame="global", method=method)
        return shape
    except (MemoryError, Standard_OutOfMemory):
        raise
    except (occt.KernelError, Standard_Failure) as error:
        record["global_failure"] = str(error)

    lx, ly, hx, hy = source.bounds
    cx, cy = lx + (hx - lx) / 2, ly + (hy - ly) / 2
    roundtrip = max(
        max(abs((x - cx) + cx - x), abs((y - cy) + cy - y))
        for ring in (source.exterior, *source.interiors)
        for x, y in ring.coords
    )
    allowance = min(1e-7, _coordinate_roundoff(source))
    record.update(
        local_origin_mm=(cx, cy),
        local_roundtrip_error_mm=roundtrip,
        local_roundtrip_allowance_mm=allowance,
    )
    if roundtrip > allowance:
        raise occt.KernelError("local spatial frame exceeded coordinate roundoff")
    local_diagnostics: dict[str, object] = {}
    record["local_diagnostics"] = local_diagnostics
    try:
        shape, method = builder(
            translate(source, xoff=-cx, yoff=-cy), local_diagnostics
        )
        transform = gp_Trsf()
        transform.SetTranslation(gp_Vec(cx, cy, 0))
        shape = BRepBuilderAPI_Transform(shape, transform, True).Shape()
        _positive(shape)
        validate(shape, source)
        record.update(frame="local", method=method)
        return shape
    except (MemoryError, Standard_OutOfMemory):
        raise
    except (occt.KernelError, Standard_Failure) as error:
        record["local_failure"] = str(error)
        raise occt.KernelError("both native spatial piece frames failed") from error


def build_roof(
    source: Polygon,
    run: float,
    builder: PieceBuilder,
    validate: PieceValidator,
    *,
    diagnostics: dict[str, object] | None = None,
    initial_grid: int = 4,
    max_piece_edges: int = 512,
    max_pieces: int = 128,
    max_depth: int = 12,
) -> TopoDS_Shape:
    """Fuse verified regional roofs; the caller checks the final original top.

    ``builder`` must apply the existing top and section checks to its input.
    ``validate`` applies those same checks after a local result is translated
    back into the original global frame. Timeouts remain the caller's concern.
    """
    if (
        not source.is_valid
        or source.is_empty
        or source.area <= 0
        or not all(isfinite(value) for value in source.bounds)
        or not isfinite(run)
        or run < 0
        or initial_grid < 1
        or max_piece_edges < 3
        or max_pieces < 1
        or max_depth < 0
    ):
        raise occt.KernelError("invalid spatial roof source or construction limits")
    state = diagnostics if diagnostics is not None else {}
    overlap = max(0.5, 2 * run)
    pending: list[tuple[Polygon, int]] = []
    for piece in _initial_pieces(source, initial_grid, overlap):
        if len(pending) >= max_pieces:
            raise occt.KernelError("spatial source cover exceeded its piece limit")
        pending.append((piece, 0))
    selected: list[Polygon] = []
    shapes: list[TopoDS_Shape] = []
    records: list[dict[str, object]] = []
    state.update(
        spatial_initial_grid=initial_grid,
        spatial_overlap_mm=overlap,
        spatial_edge_limit=max_piece_edges,
        spatial_piece_limit=max_pieces,
        spatial_depth_limit=max_depth,
        spatial_minimum_split_extent_mm=overlap * (1 + 1e-12),
        spatial_attempts=records,
    )
    state.update(_check_cover(source, [piece for piece, _ in pending]))
    splits = 0
    while pending:
        if len(selected) + len(pending) > max_pieces:
            raise occt.KernelError("spatial source cover exceeded its piece limit")
        piece, depth = pending.pop(0)
        error: Exception | None = None
        if _edges(piece) <= max_piece_edges:
            record: dict[str, object] = {"edges": _edges(piece), "depth": depth}
            records.append(record)
            try:
                shape = _piece_roof(piece, builder, validate, record)
            except (MemoryError, Standard_OutOfMemory):
                raise
            except (occt.KernelError, Standard_Failure) as failure:
                error = failure
                record["failure"] = str(failure)
            else:
                shapes.append(shape)
                selected.append(piece)
                continue
        if depth >= max_depth:
            raise occt.KernelError(
                "spatial piece reached its subdivision depth limit"
            ) from error
        children = _bisect(piece, overlap)
        pending[:0] = [(child, depth + 1) for child in children]
        if len(selected) + len(pending) > max_pieces:
            raise occt.KernelError("spatial source cover exceeded its piece limit")
        splits += 1
        state.update(_check_cover(source, [*selected, *(part for part, _ in pending)]))
    state.update(_check_cover(source, selected))
    state.update(
        spatial_piece_count=len(selected),
        spatial_split_count=splits,
        spatial_piece_methods=dict(
            Counter(str(row["method"]) for row in records if "method" in row)
        ),
        spatial_piece_frames=dict(
            Counter(str(row["frame"]) for row in records if "frame" in row)
        ),
    )
    shape = occt.fuse(shapes)
    _positive(shape)
    return shape
