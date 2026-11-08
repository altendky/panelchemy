"""Indexed floating-point projections for search guidance, not certification."""

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from shapely import STRtree, linestrings, points

type FloatArray = NDArray[np.float64]
type IndexArray = NDArray[np.intp]


@dataclass(frozen=True)
class Projection:
    """Nearest feet and their original consecutive-vertex segment addresses."""

    points: FloatArray
    segments: IndexArray
    fractions: FloatArray


class NearestPolyline:
    """Prepare a source or moving proxy once for a batch of nearest queries.

    Input coordinates are copied without snapping. Zero-length edges are omitted
    from the index, while returned segment indices retain their original values.
    A wholly degenerate polyline projects onto its first point with segment 0.
    Ties may select either closest edge; squared distance has derivative
    ``2 * (query - foot)`` wherever the closest projection is unique.
    """

    def __init__(self, vertices: FloatArray) -> None:
        selected = np.array(vertices, dtype=np.float64, copy=True)
        if (
            selected.ndim != 2
            or selected.shape[1] != 2
            or not len(selected)
            or not np.all(np.isfinite(selected))
        ):
            raise ValueError("nearest polyline requires finite planar vertices")
        self._starts: FloatArray = selected[:-1] if len(selected) > 1 else selected
        self._ends: FloatArray = selected[1:] if len(selected) > 1 else selected
        with np.errstate(over="ignore", invalid="ignore"):
            self._vectors: FloatArray = self._ends - self._starts
            self._lengths_squared: FloatArray = np.sum(
                self._vectors * self._vectors, axis=1
            )
        if not np.all(np.isfinite(self._lengths_squared)):
            raise ValueError("nearest polyline edge lengths must be representable")
        valid = np.flatnonzero(self._lengths_squared > 0)
        if len(valid):
            self._original_segments: IndexArray = valid
            self._tree: STRtree = STRtree(
                cast(
                    "NDArray[np.object_]",
                    linestrings(
                        np.stack((self._starts[valid], self._ends[valid]), axis=1)
                    ),
                )
            )
        else:
            self._original_segments = np.zeros(1, dtype=np.intp)
            self._tree = STRtree(cast("NDArray[np.object_]", points(selected[:1])))

    def project(self, queries: FloatArray) -> Projection:
        """Project query rows without constructing a query-by-edge distance array."""
        if (
            queries.ndim != 2
            or queries.shape[1] != 2
            or not np.all(np.isfinite(queries))
        ):
            raise ValueError("nearest queries require finite planar points")
        if not len(queries):
            return Projection(np.empty((0, 2)), np.empty(0, dtype=np.intp), np.empty(0))
        matches = cast(
            "IndexArray", self._tree.query_nearest(points(queries), all_matches=False)
        )
        segments = np.empty(len(queries), dtype=np.intp)
        segments[matches[0]] = self._original_segments[matches[1]]
        starts, ends = self._starts[segments], self._ends[segments]
        vectors, lengths = self._vectors[segments], self._lengths_squared[segments]
        fractions: FloatArray = np.divide(
            np.sum((queries - starts) * vectors, axis=1),
            lengths,
            out=np.zeros(len(queries)),
            where=lengths > 0,
        )
        fractions = np.clip(fractions, 0, 1)
        feet = starts + fractions[:, None] * vectors
        # Retain exact input endpoints rather than round-trip them through a
        # subtraction and addition, including nonbinary and translated inputs.
        at_start = cast("NDArray[np.bool_]", fractions == 0)
        at_end = cast("NDArray[np.bool_]", fractions == 1)
        feet = np.where(at_start[:, None], starts, feet)
        feet = np.where(at_end[:, None], ends, feet)
        return Projection(feet, segments, fractions)
