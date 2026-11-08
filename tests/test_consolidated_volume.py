"""Exact multispan curves retain accurate native mass and area measurements."""

from math import hypot
from typing import cast

import pytest
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.Geom import Geom_SurfaceOfLinearExtrusion
from OCP.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shell
from scipy.integrate import quad
from shapely.geometry import box

from panelchemy import curved_occt, occt
from panelchemy.curves import BezierSpan, Controls, CurvePolygon, CurveRing
from panelchemy.material_partition import shape_digest

_K = 0.5522847498307936
# Green's theorem applied to the four polynomial quarter-circle approximations.
# This is the exact cubic-curve area, rather than pi or sampled polygon area.
_UNIT_AREA = 2 + 12 * _K / 5 - 3 * _K**2 / 5


def _quarter_speed(t: float) -> float:
    # Derivative of one original cubic, independent of the consolidated native
    # B-spline and of OCCT's surface/curve quadrature.
    x = 6 * (1 - t) * t * (_K - 1) - 3 * t**2 * _K
    y = 3 * (1 - t) ** 2 * _K + 6 * (1 - t) * t * (1 - _K)
    return hypot(x, y)


_UNIT_PERIMETER = 4 * quad(_quarter_speed, 0, 1, epsabs=1e-12)[0]


def _circle(radius: float, factor: float, shift: tuple[float, float]) -> CurveRing:
    coordinates = (
        ((1, 0), (1, _K), (_K, 1), (0, 1)),
        ((0, 1), (-_K, 1), (-1, _K), (-1, 0)),
        ((-1, 0), (-1, -_K), (-_K, -1), (0, -1)),
        ((0, -1), (_K, -1), (1, -_K), (1, 0)),
    )
    return CurveRing(
        tuple(
            BezierSpan(
                cast(
                    "Controls",
                    tuple(
                        (x * radius * factor + shift[0], y * radius * factor + shift[1])
                        for x, y in points
                    ),
                ),
                False,
            )
            for points in coordinates
        )
    )


@pytest.mark.parametrize(
    "factor,shift,z0,z1",
    [
        (1.0, (0.0, 0.0), 2.0, 4.0),
        (2.0, (123.375, 45.625), 6.0, 8.0),
        (0.25, (-23.5, 4.125), -2.0, 1.0),
    ],
)
def test_consolidated_prism_volume_matches_analytic_polynomial_area(
    factor: float, shift: tuple[float, float], z0: float, z1: float
) -> None:
    region = CurvePolygon(_circle(5, factor, shift), (_circle(1, factor, shift),))
    face = curved_occt.polygon_face(region, z0)
    # Every ring is a single CAD edge containing four unchanged cubic pieces.
    assert occt.topology_counts(face)["edges"] == 2
    expected_area = _UNIT_AREA * (25 - 1) * factor**2
    assert occt.area(face) == pytest.approx(expected_area, rel=1e-10)
    shape, method = curved_occt.roof(region, z0, z1, 0)
    assert method == "prism"
    assert occt.valid(shape)
    digest = shape_digest(shape)
    expected_surface = 2 * expected_area + (z1 - z0) * 6 * factor * _UNIT_PERIMETER
    assert occt.area(shape) == pytest.approx(expected_surface, rel=1e-10)
    assert occt.area(shape.Reversed()) == pytest.approx(expected_surface, rel=1e-10)
    assert shape_digest(shape) == digest
    assert occt.volume(shape) == pytest.approx(expected_area * (z1 - z0), rel=1e-10)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        section = occt.cross_section(shape, z0 + fraction * (z1 - z0))
        assert occt.area(section) == pytest.approx(expected_area, rel=1e-8)
    assert occt.volume(shape.Reversed()) == pytest.approx(
        -expected_area * (z1 - z0), rel=1e-10
    )


def test_trimmed_extrusion_uses_unchanged_trim_curves_and_source_brep() -> None:
    region = CurvePolygon(_circle(5, 1, (13.5, -3.25)), ())
    shape, _ = curved_occt.roof(region, 2, 4, 0)
    faces = TopExp_Explorer(shape, TopAbs_FACE)
    while faces.More():
        surface = BRep_Tool.Surface_s(TopoDS.Face(faces.Current()))
        if isinstance(surface, Geom_SurfaceOfLinearExtrusion):
            break
        faces.Next()
    else:
        pytest.fail("procedural prism has no extrusion face")
    # Cut the V domain independently of the original prism's cap boundaries.
    trimmed = BRepBuilderAPI_MakeFace(surface, 0, 4, -1.75, -0.25, 1e-9).Face()
    assert occt.valid(trimmed)
    digest = shape_digest(trimmed)
    length = 5 * _UNIT_PERIMETER
    assert occt.area(trimmed) == pytest.approx(1.5 * length, rel=1e-10)
    assert shape_digest(trimmed) == digest
    # A trim through the middle of the first and fourth cubics retains the
    # exact full support basis and the original (partial-U) trim domain.
    trimmed_u = BRepBuilderAPI_MakeFace(surface, 0.25, 3.25, -1.75, -0.25, 1e-9).Face()
    digest_u = shape_digest(trimmed_u)
    length_u = 5 * (
        quad(_quarter_speed, 0.25, 1, epsabs=1e-12)[0]
        + 2 * quad(_quarter_speed, 0, 1, epsabs=1e-12)[0]
        + quad(_quarter_speed, 0, 0.25, epsabs=1e-12)[0]
    )
    assert occt.area(trimmed_u) == pytest.approx(1.5 * length_u, rel=1e-10)
    assert shape_digest(trimmed_u) == digest_u


def test_area_preserves_repeated_face_use_multiplicity() -> None:
    region = CurvePolygon(_circle(5, 1, (0, 0)), ())
    shape, _ = curved_occt.roof(region, 2, 4, 0)
    builder = BRep_Builder()
    repeated = TopoDS_Compound()
    builder.MakeCompound(repeated)
    builder.Add(repeated, shape)
    builder.Add(repeated, shape.Reversed())
    assert occt.area(repeated) == pytest.approx(2 * occt.area(shape), rel=1e-12)


def test_boolean_trimmed_hollow_prism_area_matches_independent_half_prism() -> None:
    region = CurvePolygon(_circle(5, 1, (0, 0)), (_circle(1, 1, (0, 0)),))
    shape, _ = curved_occt.roof(region, 2, 4, 0)
    cutter = BRepPrimAPI_MakeBox(gp_Pnt(0, -10, 1), 10, 20, 4).Shape()
    half = BRepAlgoAPI_Common(shape, cutter).Shape()
    assert occt.valid(half)
    digest = shape_digest(half)
    # Two half-caps, half of both curved walls, and two new straight cut walls.
    expected = 24 * _UNIT_AREA + 6 * _UNIT_PERIMETER + 16
    assert occt.area(half) == pytest.approx(expected, rel=1e-10)
    assert shape_digest(half) == digest


def test_located_extrusion_area_does_not_bake_or_mutate_placement() -> None:
    region = CurvePolygon(_circle(5, 1, (0, 0)), ())
    shape, _ = curved_occt.roof(region, 2, 4, 0)
    transform = gp_Trsf()
    transform.SetTranslation(gp_Vec(123.375, -45.625, 17.25))
    moved = shape.Moved(TopLoc_Location(transform))
    digest = shape_digest(moved)
    expected = 2 * 25 * _UNIT_AREA + 2 * 5 * _UNIT_PERIMETER
    assert occt.area(moved) == pytest.approx(expected, rel=1e-10)
    assert shape_digest(moved) == digest


def test_face_and_open_shell_have_no_closed_volume_domain() -> None:
    face = occt.polygon_face(box(0, 0, 2, 3))
    assert occt.volume(face) == 0
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    builder.Add(compound, face)
    assert occt.volume(compound) == 0
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    builder.Add(shell, face)
    assert occt.volume(shell) == 0


def test_closed_shell_without_solid_still_has_signed_volume() -> None:
    solid = BRepPrimAPI_MakeBox(1, 2, 3).Shape()
    shells = TopExp_Explorer(solid, TopAbs_SHELL)
    assert shells.More()
    shell = shells.Current()
    assert shell.ShapeType() == TopAbs_SHELL
    assert occt.volume(shell) == pytest.approx(6)
    assert occt.volume(shell.Reversed()) == pytest.approx(-6)


def test_closed_domain_is_measured_when_free_faces_are_also_present() -> None:
    builder = BRep_Builder()
    mixed = TopoDS_Compound()
    builder.MakeCompound(mixed)
    builder.Add(mixed, BRepPrimAPI_MakeBox(1, 2, 3).Shape())
    builder.Add(mixed, occt.polygon_face(box(20, 20, 21, 21), 10))
    assert occt.volume(mixed) == pytest.approx(6)


def test_negative_integration_error_on_a_closed_domain_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = BRepPrimAPI_MakeBox(1, 2, 3).Shape()

    def failed_integral(*_args: object, **_kwargs: object) -> float:
        return -1.0

    monkeypatch.setattr(BRepGProp, "VolumePropertiesGK_s", failed_integral)
    with pytest.raises(occt.KernelError, match="could not integrate"):
        _ = occt.volume(closed)
