"""Actual polynomial reduction retains original artwork and topology checks."""

from collections.abc import Sequence
from fractions import Fraction
from typing import cast

import numpy as np
import pytest
from PIL import Image

from panelchemy import curves, outline_simplification
from panelchemy.approximation import boundary_within_tolerance
from panelchemy.contours import PanelContours, vectorize
from panelchemy.outline_fitting import fit_outlines
from panelchemy.separation import SeparationConfig, separate


@pytest.fixture
def raster_annulus() -> PanelContours:
    rows, columns = np.indices((64, 64))
    distance = (rows - 32) ** 2 + (columns - 32) ** 2
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[(distance <= 20**2) & (distance >= 5**2)] = 2
    labels[7, 7] = 2
    image = Image.fromarray(
        np.asarray(SeparationConfig().palette, dtype=np.uint8)[labels]
    )
    return vectorize(
        separate(image, silhouette=np.ones(labels.shape, dtype=np.bool_)), 32, 100_000
    )


def test_reduction_removes_polynomials_and_keeps_islands_counters_and_source_error(
    raster_annulus: PanelContours,
) -> None:
    before = fit_outlines(raster_annulus, 0.2, corner_softening_mm=0.25)
    after = fit_outlines(
        raster_annulus, 0.2, corner_softening_mm=0.25, span_simplification_mm=0.1
    )
    assert after.contours.edge_count < before.contours.edge_count
    assert cast("int", after.diagnostics["removed_polynomial_spans"]) > 0
    assert len(after.proxy.white) == len(raster_annulus.white) == 2
    assert after.proxy.silhouette.equals(raster_annulus.silhouette)
    for original, reduced in zip(raster_annulus.white, after.proxy.white, strict=True):
        assert reduced.is_valid
        assert len(original.interiors) == len(reduced.interiors)
        assert boundary_within_tolerance(
            original.boundary, reduced.boundary, 0.2 + after.proxy_deviation_mm
        )


@pytest.mark.parametrize("resource_cap", [False, True])
def test_rejected_reduction_restores_accepted_curves(
    raster_annulus: PanelContours, monkeypatch: pytest.MonkeyPatch, resource_cap: bool
) -> None:
    baseline = fit_outlines(raster_annulus, 0.2, corner_softening_mm=0.25)
    count = cast("int", baseline.diagnostics["selected_unique_spans"])
    assemble = curves._assemble  # pyright: ignore[reportPrivateUsage]
    unsafe_pair = curves._unsafe_pair  # pyright: ignore[reportPrivateUsage]
    simplify = outline_simplification.simplify
    reduction_started = False

    def start_reduction(*args: object, **kwargs: object):
        nonlocal reduction_started
        reduction_started = True
        return simplify(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    def reject_reduced(
        pieces: Sequence[curves._Piece],  # pyright: ignore[reportPrivateUsage]
        uses: Sequence[Sequence[tuple[int, bool]]],
        ranges: Sequence[range],
        source: PanelContours,
        tolerance: float,
        cuts: Sequence[set[Fraction]],
    ):
        if reduction_started and len(pieces) < count:
            raise ValueError("curve proxy cannot meet its continuous error budget")
        return assemble(pieces, uses, ranges, source, tolerance, cuts)

    def reject_reduction_pairs(pieces: Sequence[curves._Piece]):  # pyright: ignore[reportPrivateUsage]
        if reduction_started:
            raise ValueError("curve topology exceeds 1,000,000 candidate pairs")
        return unsafe_pair(pieces)

    if resource_cap:
        monkeypatch.setattr(curves, "_unsafe_pair", reject_reduction_pairs)
    else:
        monkeypatch.setattr(curves, "_assemble", reject_reduced)
    monkeypatch.setattr(outline_simplification, "simplify", start_reduction)
    result = fit_outlines(
        raster_annulus, 0.2, corner_softening_mm=0.25, span_simplification_mm=0.1
    )
    assert result.contours == baseline.contours
    assert result.diagnostics["removed_polynomial_spans"] == 0
    assert cast("int", result.diagnostics["simplification_blocked_chains"]) > 0


@pytest.mark.parametrize("budget", [-0.1, True, float("nan"), float("inf")])
def test_invalid_simplification_budget_is_rejected(
    raster_annulus: PanelContours, budget: float
) -> None:
    with pytest.raises(ValueError, match="span simplification"):
        _ = fit_outlines(raster_annulus, 0.2, span_simplification_mm=budget)
