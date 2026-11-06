"""Coarse source covers preserve native distance roofs and resource failures."""

from collections.abc import Iterator
from math import pi

import pytest
from OCP.Standard import Standard_OutOfMemory
from OCP.TopoDS import TopoDS_Shape
from shapely import union_all
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry

from panelchemy import curved_occt, occt, spatial_roof
from panelchemy.curves import CurvePolygon, polygon_lines
from panelchemy.spatial_roof import (
    _check_cover,  # pyright: ignore[reportPrivateUsage]
    _initial_pieces,  # pyright: ignore[reportPrivateUsage]
)


def build_piece(
    source: Polygon, diagnostics: dict[str, object]
) -> tuple[TopoDS_Shape, str]:
    return curved_occt.roof(polygon_lines(source), 2, 4, 0.2, diagnostics=diagnostics)


def validate(shape: TopoDS_Shape, source: Polygon) -> None:
    region = polygon_lines(source)
    curved_occt._check_roof(shape, region, 2, 4)  # pyright: ignore[reportPrivateUsage]
    curved_occt._check_sections(shape, region, 2, 4, 0.2)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("counter", [False, True])
def test_source_union_preserves_native_top_volume_and_closing_counter(
    counter: bool,
) -> None:
    if counter:
        source = Polygon(
            box(0, 0, 4, 4).exterior.coords,
            [box(1.9, 1.9, 2.1, 2.1).exterior.coords],
        )
        expected_volume = 32 + 6.4 + pi * 0.2**2 * 2**3 / 3 - 0.04 * 0.5 / 3
    else:
        source = Polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)])
        expected_volume = 14 + 6.4 + (5 * pi / 4 - 1) * 0.2**2 * 2**3 / 3
    diagnostics: dict[str, object] = {}
    shape = spatial_roof.build_roof(
        source,
        0.4,
        build_piece,
        validate,
        initial_grid=2,
        max_pieces=4 if counter else 3,
        diagnostics=diagnostics,
    )
    validate(shape, source)
    assert occt.valid(shape)
    assert occt.topology_counts(shape)["solids"] == 1
    assert occt.volume(shape) == pytest.approx(expected_volume, rel=1e-8)
    assert occt.area(occt.cross_section(shape, 4)) == pytest.approx(
        source.area, abs=1e-8
    )
    direct, _ = build_piece(source, {})
    for z in (2, 3, 3.75, 4):
        partitioned_section = occt.cross_section(shape, z)
        direct_section = occt.cross_section(direct, z)
        assert occt.area(occt.cut(partitioned_section, direct_section)) < 1e-8
        assert occt.area(occt.cut(direct_section, partitioned_section)) < 1e-8
    if counter:
        expected_bottom = 16 + 16 * 0.4 + pi * 0.4**2
        assert occt.area(occt.cross_section(shape, 2)) == pytest.approx(
            expected_bottom, abs=1e-8
        )
    piece_count = diagnostics["spatial_piece_count"]
    assert isinstance(piece_count, int) and 1 < piece_count <= 4
    assert diagnostics["spatial_cover_difference_mm2"] == pytest.approx(0, abs=1e-12)


def test_cover_rejects_missing_material_and_filled_counter() -> None:
    source = Polygon(box(0, 0, 4, 4).exterior.coords, [box(1, 1, 2, 2).exterior.coords])
    with pytest.raises(occt.KernelError, match="preserve"):
        _ = _check_cover(source, [box(0, 0, 4, 4)])
    missing = source.intersection(box(0, 0, 3, 4))
    assert isinstance(missing, Polygon)
    with pytest.raises(occt.KernelError, match="preserve"):
        _ = _check_cover(source, [missing])


def test_edge_budget_subdivides_the_source_before_native_construction() -> None:
    source = Polygon([(1, 0), (3, 0), (4, 1), (4, 3), (3, 4), (1, 4), (0, 3), (0, 1)])
    edge_counts: list[int] = []

    def bounded_piece(
        selected: Polygon, diagnostics: dict[str, object]
    ) -> tuple[TopoDS_Shape, str]:
        edge_counts.append(len(selected.exterior.coords) - 1)
        return build_piece(selected, diagnostics)

    shape = spatial_roof.build_roof(
        source, 0.4, bounded_piece, validate, initial_grid=1, max_piece_edges=6
    )
    assert len(edge_counts) >= 2 and max(edge_counts) <= 6
    validate(shape, source)


def test_piece_limit_stops_initial_generation_before_native_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated: list[Polygon] = []
    original_parts = spatial_roof._parts  # pyright: ignore[reportPrivateUsage]

    def observed_parts(geometry: BaseGeometry) -> Iterator[Polygon]:
        for piece in original_parts(geometry):
            generated.append(piece)
            yield piece

    def unexpected_builder(
        _source: Polygon, _diagnostics: dict[str, object]
    ) -> tuple[TopoDS_Shape, str]:
        pytest.fail("source piece limit must be checked before construction")

    monkeypatch.setattr(spatial_roof, "_parts", observed_parts)
    with pytest.raises(occt.KernelError, match="piece limit"):
        _ = spatial_roof.build_roof(
            box(0, 0, 4, 4), 0.4, unexpected_builder, validate, max_pieces=1
        )
    # Read only the first excess positive piece, rather than all sixteen tiles.
    assert len(generated) == 2


def test_clipping_retains_tiny_disconnected_positive_components() -> None:
    source = union_all([box(0, 0, 10, 1), box(9, 0, 10, 10), box(5.39, 5.39, 10, 5.41)])
    assert isinstance(source, Polygon)
    first_tile = source.intersection(box(-0.4, -0.4, 5.4, 5.4))
    assert isinstance(first_tile, MultiPolygon) and len(first_tile.geoms) == 2
    tiny = min(first_tile.geoms, key=lambda part: part.area)
    assert tiny.area == pytest.approx(0.0001, abs=1e-12)
    pieces = list(_initial_pieces(source, 2, 0.8))
    assert len(pieces) == 5
    assert any(piece.equals(tiny) for piece in pieces)
    cover = _check_cover(source, pieces)
    assert cover["spatial_cover_difference_mm2"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("failure", [MemoryError, Standard_OutOfMemory])
def test_resource_failure_never_retries_another_frame_or_subdivision(
    failure: type[Exception],
) -> None:
    calls = 0

    def out_of_memory(
        _source: Polygon, _diagnostics: dict[str, object]
    ) -> tuple[TopoDS_Shape, str]:
        nonlocal calls
        calls += 1
        raise failure("test resource failure")

    with pytest.raises(failure):
        _ = spatial_roof.build_roof(box(0, 0, 4, 4), 0.4, out_of_memory, validate)
    assert calls == 1


def test_geometry_failure_respects_subdivision_depth_without_dropping_source() -> None:
    calls = 0

    def unsupported(
        _source: Polygon, _diagnostics: dict[str, object]
    ) -> tuple[TopoDS_Shape, str]:
        nonlocal calls
        calls += 1
        raise occt.KernelError("unsupported native geometry")

    with pytest.raises(occt.KernelError, match="depth"):
        _ = spatial_roof.build_roof(
            box(0, 0, 4, 4), 0.4, unsupported, validate, initial_grid=1, max_depth=0
        )
    assert calls == 2


def test_subdivision_recovers_after_both_native_frames_fail() -> None:
    source = Polygon([(1, 0), (3, 0), (4, 1), (4, 3), (3, 4), (1, 4), (0, 3), (0, 1)])
    failed_frames: list[Polygon] = []
    diagnostics: dict[str, object] = {}

    def regional_builder(
        selected: Polygon, piece_diagnostics: dict[str, object]
    ) -> tuple[TopoDS_Shape, str]:
        if selected.bounds[2] - selected.bounds[0] > 2.5:
            failed_frames.append(selected)
            raise occt.KernelError("test native arrangement needs smaller regions")
        return build_piece(selected, piece_diagnostics)

    shape = spatial_roof.build_roof(
        source,
        0.4,
        regional_builder,
        validate,
        initial_grid=1,
        diagnostics=diagnostics,
    )
    assert len(failed_frames) == 2
    assert failed_frames[0].bounds == (0, 0, 4, 4)
    assert failed_frames[1].bounds == (-2, -2, 2, 2)
    assert diagnostics["spatial_split_count"] == 1
    validate(shape, source)
    direct, _ = build_piece(source, {})
    assert occt.volume(shape) == pytest.approx(occt.volume(direct), rel=1e-8)
    for z in (2, 4):
        actual = occt.cross_section(shape, z)
        expected = occt.cross_section(direct, z)
        assert occt.area(occt.cut(actual, expected)) < 1e-8
        assert occt.area(occt.cut(expected, actual)) < 1e-8


def test_local_frame_requires_original_global_top_and_sections() -> None:
    source = translate(box(0, 0, 4, 4), xoff=100, yoff=150)
    validations: list[Polygon] = []
    diagnostics: dict[str, object] = {}

    def locally_conditioned(
        selected: Polygon, piece_diagnostics: dict[str, object]
    ) -> tuple[TopoDS_Shape, str]:
        if selected.bounds[0] > 50:
            raise occt.KernelError("test global numerical construction failure")
        return build_piece(selected, piece_diagnostics)

    def checked_global(shape: TopoDS_Shape, selected: Polygon) -> None:
        validations.append(selected)
        validate(shape, selected)

    shape = spatial_roof.build_roof(
        source,
        0.4,
        locally_conditioned,
        checked_global,
        initial_grid=1,
        diagnostics=diagnostics,
    )
    assert len(validations) == 1 and validations[0].equals(source)
    assert diagnostics["spatial_piece_frames"] == {"local": 1}
    validate(shape, source)


def test_large_line_routing_checks_the_whole_original_source_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)])
    expected, _ = build_piece(source, {})
    checked: list[str] = []
    original_roof_check = curved_occt._check_roof  # pyright: ignore[reportPrivateUsage]
    original_sections_check = curved_occt._check_sections  # pyright: ignore[reportPrivateUsage]

    def supplied_roof(*_args: object, **_kwargs: object) -> TopoDS_Shape:
        return expected

    def roof_check(
        shape: TopoDS_Shape, selected: CurvePolygon, z0: float, z1: float
    ) -> None:
        checked.append("top")
        original_roof_check(shape, selected, z0, z1)

    def sections_check(
        shape: TopoDS_Shape,
        selected: CurvePolygon,
        z0: float,
        z1: float,
        slope: float,
    ) -> None:
        checked.append("sections")
        original_sections_check(shape, selected, z0, z1, slope)

    monkeypatch.setattr(curved_occt, "_SPATIAL_EDGE_THRESHOLD", 5)
    monkeypatch.setattr(spatial_roof, "build_roof", supplied_roof)
    monkeypatch.setattr(curved_occt, "_check_roof", roof_check)
    monkeypatch.setattr(curved_occt, "_check_sections", sections_check)
    actual, method = curved_occt.roof(polygon_lines(source), 2, 4, 0.2)
    assert actual.IsSame(expected)
    assert method == "spatial-decomposition"
    assert checked == ["top", "sections"]
