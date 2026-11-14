"""Wire-only native draft sweeps with explicit surface approximation accuracy."""

from dataclasses import dataclass
from math import isfinite, pi

from OCP.Bnd import Bnd_Box
from OCP.BndLib import BndLib_Add3dCurve
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepFill import (
    BRepFill_DraftLaw,
    BRepFill_Modified,
    BRepFill_ShapeLaw,
    BRepFill_Sweep,
)
from OCP.BRepLib import BRepLib_MakeEdge, BRepLib_MakeWire
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.collections import (
    DataMap_TopoDS_Shape_NCollection_HArray2_TopTools_ShapeMapHasher,
    Map_TopoDS_Shape_TopTools_ShapeMapHasher,
)
from OCP.Geom import Geom_Line, Geom_TrimmedCurve
from OCP.GeomAbs import GeomAbs_C1
from OCP.GeomFill import GeomFill_Location, GeomFill_LocationDraft
from OCP.gp import gp_Ax3, gp_Dir, gp_Mat, gp_Pnt, gp_Trsf, gp_Vec, gp_XYZ
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import (
    TopoDS,
    TopoDS_Edge,
    TopoDS_Shape,
    TopoDS_Shell,
    TopoDS_Vertex,
    TopoDS_Wire,
)

from panelchemy import occt

ACCURACY_MM = 1e-6
_ANGLE_MIN = 1e-8


@dataclass(frozen=True)
class Draft:
    """Only the result/ownership interface used by the regional wall builder."""

    shell: TopoDS_Shell | None
    generated: tuple[tuple[TopoDS_Edge, TopoDS_Shape], ...] = ()
    error_mm: float | None = None

    def IsDone(self) -> bool:
        return self.shell is not None

    def Shell(self) -> TopoDS_Shell:
        if self.shell is None:
            raise occt.KernelError("native precise draft did not complete")
        return self.shell

    def Generated(self, original: TopoDS_Shape) -> list[TopoDS_Shape]:
        return [face for edge, face in self.generated if edge.IsSame(original)]


def _box(wire: TopoDS_Wire, direction: gp_Dir) -> Bnd_Box:
    """Retain Draft's local-frame bounds used by its orientation calculation."""
    barycenter, count = gp_XYZ(0, 0, 0), 0
    edges = BRepTools_WireExplorer(wire)
    while edges.More():
        barycenter.Add(BRep_Tool.Pnt_s(edges.CurrentVertex()).XYZ())
        count += 1
        edges.Next()
    if not count:
        raise occt.KernelError("native precise draft needs a nonempty wire")
    barycenter.Divide(count)
    transform = gp_Trsf()
    transform.SetTransformation(gp_Ax3(gp_Pnt(barycenter), direction))
    located = TopoDS.Wire(wire.Located(TopLoc_Location(transform)))
    bounds = Bnd_Box()
    edges = BRepTools_WireExplorer(located)
    while edges.More():
        BndLib_Add3dCurve.Add_s(BRepAdaptor_Curve(edges.Current()), 0.1, bounds)
        edges.Next()
    return bounds


def _good_orientation(
    bounds: Bnd_Box, location: BRepFill_DraftLaw, direction: gp_Dir
) -> bool:
    """Mirror native Draft's winding test, including open single-edge wires."""
    _, length = location.CurvilinearBounds(location.NbLaw())
    ratio = gp_Vec(bounds.CornerMin(), bounds.CornerMax()).Magnitude() / length
    count = int(4 + 10 * ratio)
    step = length / count
    points: list[gp_Pnt] = []
    barycenter = gp_XYZ(0, 0, 0)
    for i in range(count + 1):
        index, parameter = location.Parameter(i * step)
        point = gp_Pnt()
        location.Law(index).GetCurve().D0(parameter, point)
        points.append(point)
        barycenter.Add(point.XYZ())
    barycenter.Divide(count + 1)
    center, normal = gp_Pnt(barycenter), gp_Vec(direction.XYZ())
    previous = gp_Vec(center, points[0])
    angle = 0.0
    for point in points[1:]:
        current = gp_Vec(center, point)
        angle += previous.AngleWithRef(current, normal)
        previous = current
    return angle >= 0


def draft(
    wire: TopoDS_Wire,
    direction: gp_Dir,
    angle: float,
    length: float,
    internal: bool,
    *,
    accuracy_mm: float = ACCURACY_MM,
) -> Draft:
    """Use Draft's public native building blocks with a tighter surface budget.

    BRepFill_Draft fixes its surface tolerance at 1e-4 mm without a setter.
    Keep its wire-only location, section, orientation and generator ownership
    rules while controlling BRepFill_Sweep's accuracy directly. This does not
    change the source wire or any regional acceptance tolerance.
    """
    copied = BRepBuilderAPI_Copy(wire, True, False)
    selected = TopoDS.Wire(copied.Shape())
    if not selected.Closed():
        first, last = TopoDS_Vertex(), TopoDS_Vertex()
        TopExp.Vertices_s(selected, first, last)
        if not first.IsNull() and not last.IsNull() and first.IsSame(last):
            selected.Closed(True)
    law = GeomFill_LocationDraft(direction, abs(angle))
    location = BRepFill_DraftLaw(selected, law)
    winding = _good_orientation(_box(selected, direction), location, direction)
    if internal ^ (not winding):
        law.SetAngle(-abs(angle))
        location = BRepFill_DraftLaw(selected, law)
    location.CleanLaw(_ANGLE_MIN)
    first, last = location.Law(1).GetDomain()
    matrix, translation = gp_Mat(), gp_Vec()
    _ = location.Law(1).D0((first + last) / 2, matrix, translation)
    section_direction = gp_Vec(0, 1, 0)
    if direction.Angle(gp_Dir(matrix.Column(2))) > pi / 2:
        section_direction.Reverse()
    section_curve = Geom_TrimmedCurve(
        Geom_Line(gp_Pnt(0, 0, 0), gp_Dir(section_direction)), 0, length
    )
    section = BRepFill_ShapeLaw(
        BRepLib_MakeWire(BRepLib_MakeEdge(section_curve).Edge()).Wire(), True
    )
    sweep = BRepFill_Sweep(section, location, True)
    sweep.SetTolerance(accuracy_mm)
    sweep.SetAngularControl(_ANGLE_MIN, 3.0)
    sweep.Build(
        Map_TopoDS_Shape_TopTools_ShapeMapHasher(),
        DataMap_TopoDS_Shape_NCollection_HArray2_TopTools_ShapeMapHasher(),
        DataMap_TopoDS_Shape_NCollection_HArray2_TopTools_ShapeMapHasher(),
        BRepFill_Modified,
        GeomAbs_C1,
        GeomFill_Location,
        11,
        30,
    )
    if not sweep.IsDone():
        return Draft(None)
    error = sweep.ErrorOnSurface()
    if not isfinite(error) or not 0 <= error <= accuracy_mm:
        return Draft(None, error_mm=error)
    shell = TopoDS.Shell(sweep.Shape())
    faces = sweep.SubShape()
    explorer = TopExp_Explorer(shell, TopAbs_FACE)
    if not explorer.More():
        raise occt.KernelError("native precise draft returned no wall faces")
    face = TopoDS.Face(explorer.Current())
    surface = BRepAdaptor_Surface(face)
    point, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    surface.D1(surface.FirstUParameter(), surface.FirstVParameter(), point, du, dv)
    normal = du.Crossed(dv)
    if face.Orientation() == TopAbs_REVERSED:
        normal.Reverse()
    outward = True
    if normal.Magnitude() > 1e-10:
        outward = direction.Angle(gp_Dir(normal)) > pi / 2
    if outward == internal:
        shell.Reverse()
    generated: list[tuple[TopoDS_Edge, TopoDS_Shape]] = []
    edges = BRepTools_WireExplorer(wire)
    while edges.More():
        original = edges.Current()
        equivalent = copied.ModifiedShape(original)
        for index in range(1, location.NbLaw() + 1):
            if equivalent.IsSame(location.Edge(index)):
                generated.append((original, faces.Value(1, index)))
                break
        edges.Next()
    return Draft(shell, tuple(generated), error)
