"""Fine serration is distinct from raster stairs and protected small features."""

from math import cos, hypot, pi, sin

import pytest

from panelchemy.curves import Point2
from panelchemy.edge_detail import Sample, classify_chain


def digital(points: list[Point2], pitch: float) -> tuple[Point2, ...]:
    result: list[Point2] = []
    for x, y in points:
        q = round(x / pitch) * pitch, round(y / pitch) * pitch
        if not result:
            result.append(q)
        else:
            for p in ((q[0], result[-1][1]), q):
                if p != result[-1]:
                    result.append(p)
    return tuple(result)


def rotate(
    points: tuple[Point2, ...], angle: float, scale: float = 1.0
) -> tuple[Point2, ...]:
    a = angle * pi / 180
    return tuple(
        (scale * (x * cos(a) - y * sin(a)) + 17, scale * (x * sin(a) + y * cos(a)) - 21)
        for x, y in points
    )


def center(
    points: tuple[Point2, ...], target: Point2, pitch: float, angle: float = 0
) -> Sample:
    index = min(
        range(len(points)),
        key=lambda i: hypot(points[i][0] - target[0], points[i][1] - target[1]),
    )
    return classify_chain(
        rotate(points, angle) if angle else points,
        pixel_pitch_mm=pitch,
        line_width_mm=0.5,
    )[index]


def normal_texture(pitch: float) -> tuple[Point2, ...]:
    return digital(
        [
            (x, 0.15 * (1 - 4 * abs((x / 0.4 + 0.5) % 1 - 0.5)))
            for x in [i / 200 for i in range(-600, 601)]
        ],
        pitch,
    )


def backward_texture() -> tuple[Point2, ...]:
    result: list[Point2] = []
    for k in range(-20, 20, 2):
        for p in (
            (k, -k),
            (k + 1, -k),
            (k + 1, -k - 1),
            (k, -k - 1),
            (k, -k - 2),
            (k + 2, -k - 2),
        ):
            q = 0.1 * p[0], 0.1 * p[1]
            if not result or q != result[-1]:
                result.append(q)
    return tuple(result)


@pytest.mark.parametrize("pitch", [0.025, 0.05, 0.1])
@pytest.mark.parametrize("angle", [0, 15, 45, 75, 90, 135])
@pytest.mark.parametrize("upsample", [1, 2])
def test_monotone_raster_stairs_are_not_fine_texture(
    pitch: float, angle: float, upsample: int
) -> None:
    a = angle * pi / 180
    points = digital(
        [(i / 100 * cos(a), i / 100 * sin(a)) for i in range(-300, 301)],
        pitch * upsample,
    )
    result = classify_chain(points, pixel_pitch_mm=pitch, line_width_mm=0.5)
    assert not any(s.kind == "fine_texture" for s in result)
    assert any(s.kind == "raster_steps" for s in result)


@pytest.mark.parametrize("pitch", [0.025, 0.05, 0.1])
@pytest.mark.parametrize("radius", [0.4, 1.0, 2.0, 5.0])
def test_smooth_curved_raster_boundary_is_not_fine_texture(
    pitch: float, radius: float
) -> None:
    points = digital(
        [
            (radius * cos(i * pi / 800), radius * sin(i * pi / 800))
            for i in range(-600, 601)
        ],
        pitch,
    )
    assert not any(
        s.kind == "fine_texture"
        for s in classify_chain(points, pixel_pitch_mm=pitch, line_width_mm=0.5)
    )


@pytest.mark.parametrize("angle", [0, 31, 90, 137])
@pytest.mark.parametrize("scale", [0.1, 1.0, 7.0])
def test_repeated_backward_texture_is_rotation_and_scale_invariant(
    angle: float, scale: float
) -> None:
    raw = backward_texture()
    moved = rotate(raw, angle, scale)
    index = min(range(len(raw)), key=lambda i: hypot(*raw[i]))
    result = classify_chain(
        moved, pixel_pitch_mm=0.05 * scale, line_width_mm=0.5 * scale
    )
    sample = result[index]
    assert sample.kind == "fine_texture"
    assert min(sample.backtracks, sample.coarse_backtracks) >= 2
    assert sample.max_event_mm <= 0.5 * scale


@pytest.mark.parametrize("pitch", [0.025, 0.05, 0.1])
@pytest.mark.parametrize("angle", [0, 31, 90, 137])
def test_monotone_progress_normal_texture_does_not_need_backtracking(
    pitch: float,
    angle: float,
) -> None:
    points = normal_texture(pitch)
    sample = center(points, (0, 0.15), pitch, angle)
    assert sample.kind == "fine_texture"
    assert sample.backtracks == 0
    assert min(sample.normal_reversals, sample.coarse_normal_reversals) >= 2


@pytest.mark.parametrize("pitch", [0.025, 0.05, 0.1])
@pytest.mark.parametrize("angle", [0, 31, 90, 137])
def test_broad_v_corner_is_structural(pitch: float, angle: float) -> None:
    points = digital([(i / 200, 3 - abs(i / 200)) for i in range(-600, 601)], pitch)
    assert center(points, (0, 3), pitch, angle).kind == "structural"


@pytest.mark.parametrize("pitch", [0.025, 0.05, 0.1])
@pytest.mark.parametrize("angle", [0, 31, 90, 137])
def test_isolated_sub_width_tip_is_protected(pitch: float, angle: float) -> None:
    points = digital(
        [
            (i / 200, max(0.0, 0.2 * (1 - abs(i / 200) / 0.15)))
            for i in range(-600, 601)
        ],
        pitch,
    )
    assert center(points, (0, 0.2), pitch, angle).kind in ("structural", "uncertain")


@pytest.mark.parametrize("pitch", [0.025, 0.05, 0.1])
@pytest.mark.parametrize("angle", [0, 31, 90, 137])
def test_long_narrow_whisker_tip_is_not_texture(pitch: float, angle: float) -> None:
    points = digital([(-3, 0.1), (0, 0.1), (0.3, 0), (0, -0.1), (-3, -0.1)], pitch)
    assert center(points, (0.3, 0), pitch, angle).kind in ("structural", "uncertain")


def test_short_closed_feature_and_closing_sample_are_structural() -> None:
    points = ((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2), (0.0, 0.0))
    samples = classify_chain(points, pixel_pitch_mm=0.05, line_width_mm=0.5)
    assert len(samples) == len(points)
    assert all(s.kind == "structural" for s in samples)
    assert samples[0].kind == samples[-1].kind


def test_explicit_feature_barrier_has_priority_without_mutation() -> None:
    points = backward_texture()
    snapshot = tuple(points)
    index = min(range(len(points)), key=lambda i: hypot(*points[i]))
    samples = classify_chain(
        points,
        pixel_pitch_mm=0.05,
        line_width_mm=0.5,
        protected_indices=frozenset({0, index}),
    )
    assert samples[index].kind == "structural" and samples[0].kind == "structural"
    assert points == snapshot


def test_insufficient_resolution_and_enormous_support_are_uncertain() -> None:
    points = ((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0))
    for pitch, width in ((0.3, 0.5), (1.0, 1e308)):
        assert all(
            s.kind == "uncertain"
            for s in classify_chain(points, pixel_pitch_mm=pitch, line_width_mm=width)
        )


@pytest.mark.parametrize(
    "pitch,width", [(0.0, 0.5), (0.1, -1), (float("nan"), 0.5), (True, 0.5)]
)
def test_invalid_scales_are_rejected(pitch: float, width: float) -> None:
    with pytest.raises(ValueError):
        _ = classify_chain(
            ((0.0, 0.0), (1.0, 0.0)), pixel_pitch_mm=pitch, line_width_mm=width
        )
