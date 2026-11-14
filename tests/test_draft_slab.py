"""Wall extensions cannot create material outside the requested finite slab."""

# pyright: reportPrivateUsage=false

from typing import override

import pytest
from OCP.BOPAlgo import BOPAlgo_MakerVolume
from OCP.BRep import BRep_Builder
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape
from shapely.geometry import box

from panelchemy import curved_occt as n
from panelchemy import material_partition, occt
from panelchemy.curves import polygon_lines


@pytest.mark.parametrize("heights", [(1.0, 2.0), (4.0, 5.0)])
def test_outside_cells_are_excluded_by_exact_intersection_not_size(
    heights: tuple[float, float],
) -> None:
    slab = occt.prism(box(-2, -2, 2, 2), 2, 4)
    cell = occt.prism(box(-1, -1, 1, 1), *heights)
    before = material_partition.shape_digest(cell)
    diagnostics: dict[str, object] = {}
    assert n._cells_in_slab([cell], slab, diagnostics) == []
    assert diagnostics == {"slab_excluded_cells": 1}
    assert material_partition.shape_digest(cell) == before


@pytest.mark.parametrize("heights", [(1.5, 2.5), (3.5, 4.5)])
def test_crossing_cell_retains_its_positive_volume_inside_the_slab(
    heights: tuple[float, float],
) -> None:
    slab = occt.prism(box(-2, -2, 2, 2), 2, 4)
    cell = occt.prism(box(0, 0, 1, 1), *heights)
    before = material_partition.shape_digest(cell)
    diagnostics: dict[str, object] = {}
    parts = n._cells_in_slab([cell], slab, diagnostics)
    assert len(parts) == 1 and occt.valid(parts[0])
    assert occt.volume(parts[0]) == pytest.approx(0.5)
    low, high = n._bounds(parts[0])
    assert low[2] == pytest.approx(max(2, heights[0]), abs=1e-6)
    assert high[2] == pytest.approx(min(4, heights[1]), abs=1e-6)
    assert diagnostics == {"slab_clipped_cells": 1}
    assert material_partition.shape_digest(cell) == before


def test_tiny_inside_material_is_retained_without_a_volume_cutoff() -> None:
    slab = occt.prism(box(-2, -2, 2, 2), 2, 4)
    tiny = occt.prism(box(0, 0, 0.001, 0.001), 2.1, 2.101)
    assert 0 < occt.volume(tiny) < 2e-9
    diagnostics: dict[str, object] = {}
    result = n._cells_in_slab([tiny], slab, diagnostics)
    assert len(result) == 1 and result[0].IsSame(tiny)
    assert diagnostics == {}


def test_failed_domain_intersection_is_not_treated_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed(*_args: object) -> TopoDS_Shape:
        raise occt.KernelError("domain intersection failed")

    monkeypatch.setattr(occt, "common", failed)
    with pytest.raises(occt.KernelError, match="domain intersection failed"):
        _ = n._cells_in_slab(
            [occt.prism(box(0, 0, 1, 1), 1, 2)],
            occt.prism(box(-2, -2, 2, 2), 2, 4),
        )


def test_closed_shell_result_is_not_silently_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solid = occt.prism(box(0, 0, 1, 1), 2, 3)
    shell = n._entities(solid, TopAbs_SHELL)[0]
    assert occt.valid(shell) and occt.volume(shell) > 0

    def shell_result(*_args: TopoDS_Shape) -> TopoDS_Shape:
        return shell

    monkeypatch.setattr(occt, "common", shell_result)
    with pytest.raises(occt.KernelError, match="lost its solid topology"):
        _ = n._cells_in_slab(
            [occt.prism(box(0, 0, 1, 1), 1, 2)],
            occt.prism(box(-2, -2, 2, 2), 2, 4),
        )


def test_roof_does_not_select_an_enclosed_wall_margin_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    below_floor = occt.prism(box(1, 1, 2, 2), 1.9, 2)

    class WithWallMargin(BOPAlgo_MakerVolume):
        @override
        def Shape(self) -> TopoDS_Shape:
            compound = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(compound)
            builder.Add(compound, super().Shape())
            builder.Add(compound, below_floor)
            return compound

    monkeypatch.setattr(n, "BOPAlgo_MakerVolume", WithWallMargin)
    region = polygon_lines(box(0, 0, 4, 4))
    diagnostics: dict[str, object] = {}
    shape, method = n.roof(region, 2, 4, 0.1, diagnostics=diagnostics)
    assert method == "surface"
    assert diagnostics["slab_excluded_cells"] == 1
    n._check_roof(shape, region, 2, 4)
    n._check_sections(shape, region, 2, 4, 0.1)
