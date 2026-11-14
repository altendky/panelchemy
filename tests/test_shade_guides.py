"""RGB evidence guides search without changing categorical source geometry."""

# pyright: reportPrivateUsage=false

from typing import cast

import numpy as np
import pytest
from PIL import Image

from panelchemy.contours import PanelContours, vectorize
from panelchemy.separation import SeparationConfig, SeparationResult
from panelchemy.shade_guides import ShadeGuides


def fixture() -> tuple[
    np.ndarray[tuple[int, ...], np.dtype[np.uint8]], SeparationResult, PanelContours
]:
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[:, 22:43] = 1
    labels[:, 43:] = 2
    palette = np.array([[0, 0, 0], [240, 80, 0], [240, 240, 240]], dtype=np.uint8)
    rgb = palette[labels]
    rgb[:, 21] = (24, 8, 0)
    rgb[:, 22] = (168, 56, 0)
    result = SeparationResult(
        labels,
        np.ones(labels.shape, dtype=bool),
        SeparationConfig(),
        "explicit-mask",
        [],
        1,
    )
    contours = vectorize(result, long_edge_mm=64)
    return rgb, result, contours


def test_registered_edge_densification_and_reversal_preserve_source_parameters():
    rgb, result, contours = fixture()
    guide = ShadeGuides(
        Image.fromarray(rgb), original=result, retained=result, source=contours
    )
    points = ((22.0, 10.0), (22.0, 30.0))
    samples = guide.for_chain(points, np.array([0.0, 1.0]), free_loop=False)
    assert len(samples.points) == 22
    assert samples.points[0] == points[0] and samples.points[-1] == points[-1]
    assert samples.parameters[1] == 0.025
    assert samples.points[1][0] == pytest.approx(22 + 1 / 6)
    reverse = guide.for_chain(
        tuple(reversed(points)), np.array([0.0, 1.0]), free_loop=False
    )
    assert reverse.points == tuple(reversed(samples.points))
    assert guide.diagnostics["calibrated_material_pairs"] == [(0, 1), (0, 2), (1, 2)]


def test_guide_owns_snapshots_without_mutating_inputs():
    rgb, result, contours = fixture()
    before = result.labels.copy(), rgb.copy()
    guide = ShadeGuides(
        Image.fromarray(rgb), original=result, retained=result, source=contours
    )
    points = ((22.0, 10.0), (22.0, 30.0))
    first = guide.for_chain(points, np.array([0.0, 1.0]), free_loop=False)
    assert np.array_equal(before[0], result.labels) and np.array_equal(before[1], rgb)
    result.labels[:] = 0
    rgb[:] = 0
    assert guide.for_chain(points, np.array([0.0, 1.0]), free_loop=False) == first


def test_cleanup_edit_third_color_and_off_mixture_fall_back():
    for mode in ("cleanup", "third", "mixture", "unbracketed", "contrast"):
        rgb, original, contours = fixture()
        retained = SeparationResult(
            original.labels.copy(),
            original.silhouette.copy(),
            original.config,
            "explicit-mask",
            [],
            1,
        )
        # Test a single crossing at pixel y=20, beside x=22.
        if mode == "cleanup":
            retained.labels[19, 21] = 1
        elif mode == "third":
            original.labels[19, 21] = retained.labels[19, 21] = 2
        elif mode == "mixture":
            rgb[20, 21] = (24, 8, 200)
        elif mode == "unbracketed":
            rgb[20, 21:23] = (24, 8, 0)
        else:
            rgb[20, 21] = (119, 40, 0)
            rgb[20, 22] = (121, 40, 0)
        guide = ShadeGuides(
            Image.fromarray(rgb), original=original, retained=retained, source=contours
        )
        assert np.array_equal(guide._crossing((22, 20), (22, 21)), (22.0, 20.5)), mode


def test_missing_calibration_and_diagonal_contact_fall_back():
    rgb, result, contours = fixture()
    result.labels[20, 22], result.labels[20, 21] = 0, 1
    guide = ShadeGuides(
        Image.fromarray(rgb), original=result, retained=result, source=contours
    )
    assert np.array_equal(guide._crossing((22, 19), (22, 20)), (22.0, 19.5))
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[:, 4:] = 2
    small = SeparationResult(
        labels,
        np.ones(labels.shape, dtype=bool),
        SeparationConfig(),
        "explicit-mask",
        [],
        1,
    )
    guide = ShadeGuides(
        Image.new("RGB", (8, 8)),
        original=small,
        retained=small,
        source=vectorize(small, long_edge_mm=8),
    )
    assert guide.diagnostics["calibrated_material_pairs"] == []
    assert np.array_equal(guide._crossing((4, 3), (4, 4)), (4, 3.5))


def test_free_loop_seam_is_interpolated_search_value_not_a_source_pin():
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[8:24, 8:24] = 2
    result = SeparationResult(
        labels,
        np.ones(labels.shape, dtype=bool),
        SeparationConfig(),
        "explicit-mask",
        [],
        1,
    )
    guide = ShadeGuides(
        Image.new("RGB", (32, 32)),
        original=result,
        retained=result,
        source=vectorize(result, long_edge_mm=32),
    )
    points = ((8.0, 8.0), (24.0, 8.0), (24.0, 24.0), (8.0, 24.0), (8.0, 8.0))
    samples = guide.for_chain(points, np.linspace(0, 1, 5), free_loop=True)
    assert samples.points[0] == samples.points[-1] == (8.25, 8.25)
    assert np.array_equal(
        cast(
            "np.ndarray[tuple[int, ...], np.dtype[np.float64]]",
            samples.sample(np.array([0.0, 1.0]))[0],
        ),
        cast(
            "np.ndarray[tuple[int, ...], np.dtype[np.float64]]",
            samples.sample(np.array([0.0, 1.0]))[1],
        ),
    )
    assert points[0] == (8, 8)


def test_registration_rejects_nonlattice_diagonal_and_interior_edges():
    rgb, result, contours = fixture()
    guide = ShadeGuides(
        Image.fromarray(rgb), original=result, retained=result, source=contours
    )
    for points in (
        ((22.01, 10.0), (22.0, 30.0)),
        ((22.0, 10.0), (23.0, 30.0)),
        ((20.0, 10.0), (20.0, 30.0)),
    ):
        with pytest.raises(ValueError):
            _ = guide.for_chain(points, np.array([0.0, 1.0]), free_loop=False)
    with pytest.raises(ValueError, match="registered"):
        _ = ShadeGuides(
            Image.new("RGB", (63, 64)),
            original=result,
            retained=result,
            source=contours,
        )


@pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
def test_all_material_pair_models_use_the_same_guarded_transition(
    pair: tuple[int, int],
):
    rgb, result, contours = fixture()
    a, b = pair
    palette = np.array([[0, 0, 0], [240, 80, 0], [240, 240, 240]], dtype=float)
    # A separate patch retains enough untouched interiors for all calibrations.
    result.labels[25:35, 20:22] = a
    result.labels[25:35, 22:24] = b
    first = cast("np.ndarray[tuple[int, ...], np.dtype[np.float64]]", palette[a])
    last = cast("np.ndarray[tuple[int, ...], np.dtype[np.float64]]", palette[b])
    rgb[25:35, 21] = np.rint(0.9 * first + 0.1 * last).astype(np.uint8)
    rgb[25:35, 22] = np.rint(0.3 * first + 0.7 * last).astype(np.uint8)
    guide = ShadeGuides(
        Image.fromarray(rgb), original=result, retained=result, source=contours
    )
    assert guide._crossing((22, 30), (22, 31))[0] == pytest.approx(22 + 1 / 6)


def test_rgba_matches_separations_explicit_palette_background_composite():
    rgb, result, contours = fixture()
    rgba = Image.fromarray(rgb).convert("RGBA")
    rgba.putalpha(Image.fromarray(np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))))
    explicit = Image.alpha_composite(
        Image.new("RGBA", rgba.size, (*result.config.palette[2], 255)), rgba
    ).convert("RGB")
    first = ShadeGuides(rgba, original=result, retained=result, source=contours)
    second = ShadeGuides(explicit, original=result, retained=result, source=contours)
    points = ((22.0, 10.0), (22.0, 30.0))
    assert first.for_chain(
        points, np.array([0.0, 1.0]), free_loop=False
    ) == second.for_chain(points, np.array([0.0, 1.0]), free_loop=False)


def test_nonunit_scale_and_cropped_origin_use_forward_exact_registration():
    rgb, result, _ = fixture()
    result.silhouette[:8] = False
    result.silhouette[56:] = False
    result.silhouette[:, :10] = False
    result.silhouette[:, 54:] = False
    contours = vectorize(result, long_edge_mm=14.4)
    pitch = contours.mm_per_pixel
    points = (
        ((22 - 10) * pitch, (56 - 20) * pitch),
        ((22 - 10) * pitch, (56 - 40) * pitch),
    )
    guide = ShadeGuides(
        Image.fromarray(rgb), original=result, retained=result, source=contours
    )
    samples = guide.for_chain(points, np.array([0.0, 1.0]), free_loop=False)
    assert samples.points[0] == points[0] and samples.points[-1] == points[-1]
    assert samples.points[1][0] == pytest.approx((22 + 1 / 6 - 10) * pitch)
