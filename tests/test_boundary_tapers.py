"""Shared boundaries use exposed height differences and reverse curvature signs."""

import pytest

from panelchemy.boundary_tapers import boundaries


def test_white_orange_boundary_uses_one_band_not_full_white_support_depth() -> None:
    # Ring0 silhouette; ring1 orange; ring2 white. Chain2 is shared in reverse.
    uses = [[(0, True)], [(1, True), (2, True)], [(2, False), (3, False)]]
    result = boundaries(uses, [range(1), range(1, 2), range(2, 3)], 1, (0, 0.2, 0.4))
    assert 0 not in result
    assert (result[1].left, result[1].right, result[1].offset_mm) == (1, 0, 0.2)
    assert (result[2].left, result[2].right, result[2].offset_mm) == (1, 2, 0.2)
    assert result[3].offset_mm == 0.4
    assert result[1].inward_sign == -1
    assert result[2].inward_sign == 1
    assert result[3].inward_sign == 1


def test_reversing_canonical_direction_reverses_required_curvature_sign() -> None:
    ranges = [range(1), range(1, 2)]
    a = boundaries([[(0, True)], [(1, True)]], ranges, 1, (0, 0.176, 0.353))[1]
    b = boundaries([[(0, True)], [(1, False)]], ranges, 1, (0, 0.176, 0.353))[1]
    assert a.offset_mm == b.offset_mm
    assert a.inward_sign == -b.inward_sign
    assert (a.left, a.right) == (b.right, b.left)


def test_boundary_ownership_conflicts_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting"):
        _ = boundaries(
            [[(0, True)], [(1, True)], [(1, True)]],
            [range(1), range(1, 2), range(2, 3)],
            1,
            (0, 1, 2),
        )


@pytest.mark.parametrize("runs", [(0, -1, 2), (0, 2, 1), (0, float("nan"), 2)])
def test_invalid_material_runs_are_rejected(runs: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        _ = boundaries([[]], [range(1)], 0, runs)
