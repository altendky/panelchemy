"""Detail classes steer fit families without spending the ordinary search budget."""

# Synthetic line certificates isolate classification and repair scheduling.
# pyright: reportPrivateUsage=false

from dataclasses import replace
from itertools import pairwise
from typing import cast

import numpy as np
import pytest
from shapely.geometry import LineString, box

from panelchemy import curvature, edge_detail
from panelchemy import curves as c
from panelchemy import outline_fitting as fitting
from panelchemy.contours import PanelContours


def fixture(
    *, neighbor: bool = False
) -> tuple[
    list[fitting._Fit],
    list[list[tuple[int, bool]]],
    list[range],
    PanelContours,
    c.CurveContours,
]:
    white = [box(2, 2, 4, 4), box(6, 6, 8, 8)] if neighbor else [box(2, 2, 8, 8)]
    source = PanelContours(box(0, 0, 10, 10), [], white, 0.1, (0, 0, 100, 100), 8)
    chains, uses = c._graph(
        [LineString(p.exterior) for p in [source.silhouette, *source.white]],
        split_loops=False,
    )
    fits: list[fitting._Fit] = []
    for index, chain in enumerate(chains):
        parameters = np.linspace(0, 1, len(chain.points))
        pieces = [
            c._Piece(
                index,
                i,
                i + 1,
                replace(
                    c._line(a, b), certified_error_mm=float(not chain.fixed and i == 1)
                ),
            )
            for i, (a, b) in enumerate(pairwise(chain.points))
        ]
        fits.append(
            fitting._Fit(
                chain, parameters, [], [], pieces, list(pairwise(parameters.tolist()))
            )
        )
    original = c.CurveContours(
        c.polygon_lines(source.silhouette),
        (),
        tuple(c.polygon_lines(p) for p in source.white),
        0.1,
        source.source_bounds_px,
    )
    return (
        fits,
        uses,
        [range(i, i + 1) for i in range(1 + len(white))],
        source,
        original,
    )


def test_source_detail_regions_stop_at_structural_vertices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = tuple((float(i), 0.0) for i in range(8))
    fit = fitting._Fit(c._Chain(points, False), np.linspace(0, 1, 8), [], [], [], [])
    kinds: tuple[edge_detail.Kind, ...] = (
        "structural",
        "raster_steps",
        "fine_texture",
        "fine_texture",
        "raster_steps",
        "structural",
        "raster_steps",
        "structural",
    )

    def classify(*_args: object, **kwargs: object) -> tuple[edge_detail.Sample, ...]:
        assert kwargs["pixel_pitch_mm"] == 0.1
        assert kwargs["line_width_mm"] == 0.5
        assert kwargs["protected_indices"] == frozenset({0, 7})
        return tuple(
            edge_detail.Sample(i, kind, "synthetic") for i, kind in enumerate(kinds)
        )

    monkeypatch.setattr(edge_detail, "classify_chain", classify)
    source = fixture()[3]
    result = fitting._detail_guidance([fit], source, 0.5)
    assert fit.texture_intervals == ((1 / 7, 4 / 7),)
    assert fit.texture_barriers == tuple(
        float(fitting._scalar(fit.parameters, i)) for i in (0, 5, 7)
    )
    summary = cast("dict[str, object]", result["edge_detail"])
    assert summary["grants_error_allowance"] is False
    assert source.silhouette.equals(box(0, 0, 10, 10))


@pytest.mark.parametrize(
    "textured,preferred_passes,ordinary_passes",
    [
        (True, True, True),
        (True, False, True),
        (False, False, True),
        (True, False, False),
    ],
)
def test_texture_prefers_smooth_fit_and_preserves_the_ordinary_fallback_budget(
    monkeypatch: pytest.MonkeyPatch,
    textured: bool,
    preferred_passes: bool,
    ordinary_passes: bool,
) -> None:
    fits, uses, ranges, source, original = fixture()
    target = next(i for i, fit in enumerate(fits) if not fit.chain.fixed)
    fits[target].texture_intervals = ((0.0, 1.0),) if textured else ()
    clock = [0.0]
    calls: list[str] = []

    def repaired(fit: fitting._Fit) -> fitting._Fit:
        return replace(
            fit,
            pieces=[
                replace(p, span=replace(p.span, certified_error_mm=0))
                for p in fit.pieces
            ],
        )

    def compact(
        fit: fitting._Fit, *_args: object, **kwargs: object
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        restricted = kwargs.get("texture_only", False)
        calls.append("texture" if restricted else "late")
        clock[0] += 3
        accepted = preferred_passes or not restricted
        return (repaired(fit) if accepted else None), {
            "status": "accepted" if accepted else "unrefined"
        }

    def ordinary(
        fit: fitting._Fit, *_args: object, **kwargs: object
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        calls.append("ordinary")
        assert kwargs["time_budget_seconds"] == 12
        return (repaired(fit) if ordinary_passes else None), {
            "status": "accepted" if ordinary_passes else "unrefined",
            "strategy": "ordinary",
        }

    def radius(span: c.BezierSpan, _run: float, _sign: int) -> bool:
        return span.certified_error_mm == 0

    def no_corners(*_args: object) -> list[tuple[fitting._Fit, str]]:
        return []

    monkeypatch.setattr(curvature, "within_curvature", radius)
    monkeypatch.setattr(fitting, "monotonic", lambda: clock[0])
    monkeypatch.setattr(fitting, "_corner_curvature_candidates", no_corners)
    monkeypatch.setattr(fitting, "_compact_curvature_candidate", compact)
    monkeypatch.setattr(fitting, "_curvature_candidate", ordinary)
    selected, proxy, report = fitting._curvature_verified(
        fits, uses, ranges, source, 0.2, (0.0, 0.1, 0.2), original, source
    )
    assert proxy.white[0].equals(source.white[0])
    assert all(s.certified_error_mm == 0 for s in selected.white[0].exterior.spans)
    assert calls == (
        ["texture"]
        if preferred_passes
        else ["texture", "ordinary", "late"]
        if not ordinary_passes
        else ["texture", "ordinary"]
        if textured
        else ["ordinary"]
    )
    diagnostics = cast("dict[str, object]", report["boundary_curvature"])
    assert diagnostics["margin_status"] == "certified"


def test_palette_rejected_texture_candidate_is_rechecked_after_neighbor_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fits, uses, ranges, source, original = fixture(neighbor=True)
    target, neighbor = (i for i, fit in enumerate(fits) if not fit.chain.fixed)
    fits[target].texture_intervals = ((0, 1),)

    def repair(fit: fitting._Fit) -> fitting._Fit:
        return replace(
            fit,
            pieces=[
                replace(p, span=replace(p.span, certified_error_mm=0))
                for p in fit.pieces
            ],
        )

    preferred = repair(fits[target])
    neighbor_first = repair(fits[neighbor])
    neighbor_second = replace(neighbor_first, fairing_mm=0.3)
    compact_calls = 0
    verification_calls = 0

    def radius(span: c.BezierSpan, _run: float, _sign: int) -> bool:
        return span.certified_error_mm == 0

    def corners(
        _fit: fitting._Fit, index: int, *_args: object
    ) -> list[tuple[fitting._Fit, str]]:
        return (
            [(neighbor_first, "first"), (neighbor_second, "second")]
            if index == neighbor
            else []
        )

    def compact(
        _fit: fitting._Fit, index: int, *_args: object, **_kwargs: object
    ) -> tuple[fitting._Fit, dict[str, object]]:
        nonlocal compact_calls
        compact_calls += 1
        assert index == target
        return preferred, {"status": "accepted"}

    def ordinary(*_args: object, **kwargs: object) -> tuple[None, dict[str, object]]:
        assert kwargs["time_budget_seconds"] == 12
        return None, {"status": "unrefined"}

    def verify(
        before: list[fitting._Fit],
        candidates: list[fitting._Fit],
        changed: set[int],
        *_args: object,
    ) -> fitting._VerifiedRefits:
        nonlocal verification_calls
        verification_calls += 1
        if verification_calls == 1:
            assert changed == {target, neighbor}
            return fitting._VerifiedRefits(
                original, source, list(before), set(), set(changed), None
            )
        if verification_calls == 2:
            assert changed == {neighbor}
            assert candidates[neighbor] is neighbor_second
        else:
            assert verification_calls == 3 and changed == {target}
            assert before[neighbor] is neighbor_second
            assert candidates[target] is preferred
        return fitting._VerifiedRefits(
            original, source, list(candidates), set(changed), set(), None
        )

    monkeypatch.setattr(curvature, "within_curvature", radius)
    monkeypatch.setattr(fitting, "monotonic", lambda: 0.0)
    monkeypatch.setattr(fitting, "_corner_curvature_candidates", corners)
    monkeypatch.setattr(fitting, "_compact_curvature_candidate", compact)
    monkeypatch.setattr(fitting, "_curvature_candidate", ordinary)
    monkeypatch.setattr(fitting, "_verify_refits", verify)
    _, _, report = fitting._curvature_verified(
        fits, uses, ranges, source, 0.2, (0, 0.1, 0.2), original, source
    )
    assert compact_calls == 1 and verification_calls == 3
    assert (
        cast("dict[str, object]", report["boundary_curvature"])["margin_status"]
        == "certified"
    )


@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize(
    "family,partial",
    [("apex", False), ("polygon", False), ("corner_run", False), ("corner_run", True)],
)
def test_late_family_dispatch_preserves_other_repairs_and_requires_palette_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    reject: bool,
    family: str,
    partial: bool,
) -> None:
    fits, uses, ranges, source, original = fixture(neighbor=True)
    target, neighbor = (i for i, fit in enumerate(fits) if not fit.chain.fixed)
    if partial:
        piece = fits[target].pieces[3]
        fits[target].pieces[3] = replace(
            piece, span=replace(piece.span, certified_error_mm=1)
        )
    calls: list[tuple[str, int]] = []
    late_candidate: list[fitting._Fit] = []

    def repaired(fit: fitting._Fit) -> fitting._Fit:
        if partial and fit is fits[target]:
            first = fit.pieces[0]
            a, b = first.span.control_points[0], first.span.control_points[-1]
            middle = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            spans = [
                c._line(a, middle),
                c._line(middle, b),
                replace(fit.pieces[1].span, certified_error_mm=0),
                fit.pieces[2].span,
                fit.pieces[3].span,
            ]
            start, stop = fit.intervals[0]
            seam = (start + stop) / 2
            return replace(
                fit,
                pieces=[
                    replace(first, start=i, stop=i + 1, span=s)
                    for i, s in enumerate(spans)
                ],
                intervals=[(start, seam), (seam, stop), *fit.intervals[1:]],
            )
        return replace(
            fit,
            pieces=[
                replace(p, span=replace(p.span, certified_error_mm=0))
                for p in fit.pieces
            ],
        )

    def radius(span: c.BezierSpan, _run: float, _sign: int) -> bool:
        return span.certified_error_mm == 0

    def corners(*_args: object) -> list[tuple[fitting._Fit, str]]:
        return []

    def ordinary(
        fit: fitting._Fit, index: int, *_args: object, **_kwargs: object
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        calls.append(("ordinary", index))
        return (repaired(fit) if index == neighbor else None), {}

    def compact(
        _fit: fitting._Fit, index: int, *_args: object
    ) -> tuple[None, dict[str, object]]:
        assert index == target
        calls.append(("compact", index))
        return None, {}

    def apex(
        fit: fitting._Fit, index: int, *_args: object
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        assert index == target
        calls.append(("apex", index))
        if family != "apex":
            return None, {}
        proposal = repaired(fit)
        late_candidate.append(proposal)
        return proposal, {"status": "accepted"}

    def polygon(
        fit: fitting._Fit, index: int, *_args: object
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        if family != "polygon":
            return None, {}  # This regression isolates the selected late family.
        assert index == target
        calls.append(("polygon", index))
        proposal = repaired(fit)
        late_candidate.append(proposal)
        return proposal, {"status": "accepted"}

    def corner_run(
        fit: fitting._Fit, index: int, *_args: object
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        if family != "corner_run":
            return None, {}
        assert index == target
        calls.append(("corner_run", index))
        proposal = repaired(fit)
        late_candidate.append(proposal)
        return proposal, {"status": "partial" if partial else "accepted"}

    verify = fitting._verify_refits

    def palette(
        before: list[fitting._Fit],
        candidates: list[fitting._Fit],
        changed: set[int],
        selected_uses: list[list[tuple[int, bool]]],
        selected_ranges: list[range],
        selected_source: PanelContours,
        tolerance: float,
        curves: c.CurveContours,
        proxy: PanelContours,
    ) -> fitting._VerifiedRefits:
        if late_candidate and reject:
            assert changed == {target} and candidates[target] is late_candidate[0]
            assert before[neighbor].pieces[1].span.certified_error_mm == 0
            return fitting._VerifiedRefits(
                curves,
                proxy,
                list(before),
                set(),
                {target},
                "synthetic palette rejection",
            )
        return verify(
            before,
            candidates,
            changed,
            selected_uses,
            selected_ranges,
            selected_source,
            tolerance,
            curves,
            proxy,
        )

    monkeypatch.setattr(curvature, "within_curvature", radius)
    monkeypatch.setattr(fitting, "_corner_curvature_candidates", corners)
    monkeypatch.setattr(fitting, "_curvature_candidate", ordinary)
    monkeypatch.setattr(fitting, "_compact_curvature_candidate", compact)
    monkeypatch.setattr(fitting, "_apex_curvature_candidate", apex)
    monkeypatch.setattr(fitting, "_polygon_curvature_candidate", polygon)
    monkeypatch.setattr(fitting, "_corner_run_curvature_candidate", corner_run)
    monkeypatch.setattr(fitting, "_verify_refits", palette)
    selected, _, report = fitting._curvature_verified(
        fits, uses, ranges, source, 0.2, (0, 0.1, 0.2), original, source
    )
    expected = [
        ("ordinary", target),
        ("ordinary", neighbor),
        ("compact", target),
        ("apex", target),
    ]
    if family == "polygon":
        expected.append(("polygon", target))
    elif family == "corner_run":
        expected.append(("corner_run", target))
    assert calls == expected
    assert all(s.certified_error_mm == 0 for s in selected.white[1].exterior.spans)
    detail = cast("dict[str,object]", report["boundary_curvature"])
    assert detail["unrefined_chains"] == ([target] if reject or partial else [])
    assert detail["refined_chains"] == (1 if reject or partial else 2)
    assert detail["partially_refined_chains"] == (
        [target] if partial and not reject else []
    )
    if partial:
        assert detail["margin_status"] == "unresolved"
        expected_fit = fits[target] if reject else late_candidate[0]
        remaining = cast("list[dict[str, object]]", detail["unresolved_boundaries"])
        assert len(remaining) == 1 and remaining[0]["chain"] == target
        assert remaining[0]["spans"] == [
            dict(piece=i, interval=expected_fit.intervals[i])
            for i, p in enumerate(expected_fit.pieces)
            if p.span.certified_error_mm
        ]
        assert len(selected.white[0].exterior.spans) == (4 if reject else 5)
