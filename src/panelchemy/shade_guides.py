"""Guarded RGB transition estimates for search, never material boundaries.

Labels and original pixel footprints remain the fitting constraints and proof
reference. These samples affect only the least-squares objective.
"""

from dataclasses import dataclass
from itertools import combinations, pairwise
from math import isfinite, sqrt
from typing import cast, final

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage

from panelchemy.contours import PanelContours
from panelchemy.curves import Point2
from panelchemy.separation import Mask, SeparationResult

type FloatArray = NDArray[np.float64]
type Labels = NDArray[np.uint8]


@dataclass(frozen=True)
class GuideSamples:
    """Search samples registered to the unchanged source parameter frame."""

    points: tuple[Point2, ...]
    parameters: tuple[float, ...]

    def sample(self, at: FloatArray) -> FloatArray:
        return np.column_stack(
            [
                np.interp(at, self.parameters, [p[axis] for p in self.points])
                for axis in (0, 1)
            ]
        )


@dataclass(frozen=True)
class _Model:
    first: FloatArray
    delta: FloatArray
    length_squared: float
    residual_limit: float
    contrast_floor: float


@final
class ShadeGuides:
    """Bind original RGB evidence to exact, retained raster-cell boundaries.

    Missing interior calibration, edited cleanup neighborhoods, third colors,
    diagonal contacts, unsupported mixtures and weak transitions all retain the
    exact hard-edge midpoint. No material label or endpoint policy is changed.
    """

    def __init__(
        self,
        image: Image.Image,
        *,
        original: SeparationResult,
        retained: SeparationResult,
        source: PanelContours,
    ) -> None:
        shape = retained.labels.shape
        if (
            len(shape) != 2
            or original.labels.shape != shape
            or original.silhouette.shape != shape
            or retained.silhouette.shape != shape
            or original.silhouette.dtype != np.bool_
            or retained.silhouette.dtype != np.bool_
            or image.size != (shape[1], shape[0])
            or not np.array_equal(original.silhouette, retained.silhouette)
            or original.config.palette != retained.config.palette
            or not np.any(retained.silhouette)
            or not np.isin(original.labels, (0, 1, 2)).all()
            or not np.isin(retained.labels, (0, 1, 2)).all()
        ):
            raise ValueError(
                "shade guidance requires registered RGB, labels and silhouette"
            )
        rows, columns = np.nonzero(retained.silhouette)
        bounds = (
            float(columns.min()),
            float(rows.min()),
            float(columns.max() + 1),
            float(rows.max() + 1),
        )
        if (
            source.source_bounds_px != bounds
            or not isfinite(source.mm_per_pixel)
            or source.mm_per_pixel <= 0
        ):
            raise ValueError(
                "shade guidance requires the retained contour origin and scale"
            )
        self._pitch = source.mm_per_pixel
        self._left, _, _, self._bottom = bounds
        background = Image.new("RGBA", image.size, (*original.config.palette[2], 255))
        composited = Image.alpha_composite(background, image.convert("RGBA")).convert(
            "RGB"
        )
        self._rgb: FloatArray = np.asarray(composited, dtype=np.float64).copy()
        self._original: Labels = np.where(
            original.silhouette, original.labels, 255
        ).astype(np.uint8)
        self._retained: Labels = np.where(
            retained.silhouette, retained.labels, 255
        ).astype(np.uint8)
        for array in (self._rgb, self._original, self._retained):
            array.flags.writeable = False
        interiors = [
            ndimage.binary_erosion(cast("Mask", self._original == color), iterations=2)
            for color in range(3)
        ]
        representatives = {
            color: np.median(self._rgb[mask], axis=0)
            for color, mask in enumerate(interiors)
            if np.count_nonzero(mask) >= 100
        }
        self._models: dict[tuple[int, int], _Model] = {}
        for a, b in combinations(representatives, 2):
            first, last = representatives[a], representatives[b]
            delta = last - first
            length_squared = float(delta @ delta)
            if length_squared == 0:
                continue
            pixels = self._rgb[interiors[a] | interiors[b]]
            alpha = (pixels - first) @ delta / length_squared
            residual = np.linalg.norm(pixels - first - alpha[:, None] * delta, axis=1)
            sigma: list[float] = []
            for color in (a, b):
                values = (self._rgb[interiors[color]] - first) @ delta / length_squared
                sigma.append(
                    max(
                        float(
                            cast(
                                "np.float64",
                                1.4826 * np.median(np.abs(values - np.median(values))),
                            )
                        ),
                        float(
                            cast(
                                "np.float64",
                                0.5 * np.sum(np.abs(delta)) / length_squared,
                            )
                        ),
                    )
                )
            self._models[a, b] = _Model(
                first,
                delta,
                length_squared,
                max(
                    sqrt(3) / 2,
                    float(np.quantile(residual, 0.99)),
                ),
                3 * sum(sigma),
            )

    @property
    def diagnostics(self) -> dict[str, object]:
        return dict(
            method="guarded RGB midpoint least-squares guidance",
            calibrated_material_pairs=sorted(self._models),
            labels_changed=False,
            source_constraints_changed=False,
            caveat="RGB mixtures guide search; retained pixel boundaries remain the accuracy reference",
        )

    def _pixel(self, point: Point2) -> tuple[int, int]:
        x, y = point
        if not isfinite(x) or not isfinite(y):
            raise ValueError("shade source vertices must be finite raster corners")
        ix, iy = (
            round(x / self._pitch + self._left),
            round(self._bottom - y / self._pitch),
        )
        if (
            float((ix - self._left) * self._pitch),
            float((self._bottom - iy) * self._pitch),
        ) != point:
            raise ValueError(
                "shade source vertices must be exact registered raster corners"
            )
        return ix, iy

    def _world(self, point: FloatArray) -> Point2:
        return float((cast("np.float64", point[0]) - self._left) * self._pitch), float(
            (self._bottom - cast("np.float64", point[1])) * self._pitch
        )

    def _crossing(self, a: tuple[int, int], b: tuple[int, int]) -> FloatArray:
        first, last = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        midpoint = (first + last) / 2
        dx, dy = b[0] - a[0], b[1] - a[1]
        normal = np.asarray((dy, -dx), dtype=np.float64)
        left, right = midpoint + normal / 2, midpoint - normal / 2
        xa, ya = map(int, np.floor(left))
        xb, yb = map(int, np.floor(right))
        height, width = cast("tuple[int, int]", self._retained.shape)
        if (
            min(xa, xb) < 0
            or min(ya, yb) < 0
            or max(xa, xb) >= width
            or max(ya, yb) >= height
        ):
            return midpoint
        ca, cb = (
            int(cast("np.uint8", self._retained[ya, xa])),
            int(cast("np.uint8", self._retained[yb, xb])),
        )
        if ca == cb:
            raise ValueError("shade source edge is not a retained material boundary")
        pair = (min(ca, cb), max(ca, cb))
        model = self._models.get(pair)
        if model is None:
            return midpoint
        pa, pb = (left, right) if ca == pair[0] else (right, left)
        xa, ya = map(int, np.floor(pa))
        xb, yb = map(int, np.floor(pb))
        x0, x1, y0, y1 = min(xa, xb), max(xa, xb) + 1, min(ya, yb), max(ya, yb) + 1
        if dx:
            x0 -= 1
            x1 += 1
        else:
            y0 -= 1
            y1 += 1
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            return midpoint
        patch = self._retained[y0:y1, x0:x1]
        if (
            not np.array_equal(patch, self._original[y0:y1, x0:x1])
            or not np.isin(patch, pair).all()
        ):
            return midpoint
        if np.any(
            cast(
                "Mask",
                (patch[:-1, :-1] == patch[1:, 1:])
                & (patch[:-1, 1:] == patch[1:, :-1])
                & (patch[:-1, :-1] != patch[:-1, 1:]),
            )
        ):
            return midpoint
        pixels = self._rgb[[ya, yb], [xa, xb]]
        alpha = (pixels - model.first) @ model.delta / model.length_squared
        residual = np.linalg.norm(
            pixels - model.first - alpha[:, None] * model.delta, axis=1
        )
        va, vb = map(float, alpha)
        if (
            np.max(residual) > model.residual_limit
            or not va <= 0.5 <= vb
            or vb - va <= model.contrast_floor
        ):
            return midpoint
        return pa + ((0.5 - va) / (vb - va)) * (pb - pa)

    def for_chain(
        self, points: tuple[Point2, ...], parameters: FloatArray, *, free_loop: bool
    ) -> GuideSamples:
        """Densify collinear raster runs with exact known source parameters."""
        if (
            len(points) < 2
            or parameters.shape != (len(points),)
            or not np.isfinite(parameters).all()
            or parameters[0] != 0
            or parameters[-1] != 1
            or not np.all(np.diff(parameters) > 0)
            or (free_loop and points[0] != points[-1])
        ):
            raise ValueError(
                "shade guidance requires an ordered source parameter frame"
            )
        pixels = [self._pixel(point) for point in points]
        guide: list[Point2] = []
        times: list[float] = []
        for index, (a, b) in enumerate(pairwise(pixels)):
            dx, dy = b[0] - a[0], b[1] - a[1]
            if (dx == 0) == (dy == 0):
                raise ValueError(
                    "shade guidance requires nonzero axis-aligned raster edges"
                )
            count = abs(dx) + abs(dy)
            step = (dx // count, dy // count)
            start, stop = (
                float(cast("np.float64", parameters[index])),
                float(cast("np.float64", parameters[index + 1])),
            )
            for j in range(count):
                p = (a[0] + j * step[0], a[1] + j * step[1])
                q = (p[0] + step[0], p[1] + step[1])
                guide.append(self._world(self._crossing(p, q)))
                times.append(start + ((j + 0.5) / count) * (stop - start))
        if free_loop:
            # The cyclic midpoint guide crosses t=0 between its last and first
            # samples. This search-only seam value is not a geometric pin.
            fraction = (1 - times[-1]) / (1 + times[0] - times[-1])
            seam = cast(
                "Point2",
                tuple(
                    (1 - fraction) * a + fraction * b
                    for a, b in zip(guide[-1], guide[0], strict=True)
                ),
            )
            guide = [seam, *guide, seam]
        else:
            guide = [points[0], *guide, points[-1]]
        return GuideSamples(tuple(guide), (0.0, *times, 1.0))
