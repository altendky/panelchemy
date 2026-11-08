"""Owned-data bindings for exact, interpreter-independent geometry proofs."""

from collections.abc import Sequence

from panelchemy.curves import Controls, Point2

def within_paths(
    source_paths: Sequence[Sequence[Point2]],
    spans: Sequence[tuple[Controls, bool]],
    tolerance: float,
    max_nodes: int,
    max_depth: int,
    proxy_refinements: int,
    index_min_segments: int,
) -> bool: ...
