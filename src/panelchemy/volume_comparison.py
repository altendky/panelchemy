"""Engineering volume comparisons, separate from geometric preservation checks."""

from dataclasses import dataclass
from math import isfinite

# 100 ppm tolerates negligible mass-integration/serialization drift. It is not
# a bound on local surface displacement or a proof of geometric equivalence.
RELATIVE_VOLUME_ALLOWANCE = 1e-4


@dataclass(frozen=True)
class VolumeComparison:
    difference_mm3: float
    numerical_allowance_mm3: float
    allowance_mm3: float
    acceptable: bool

    @property
    def warning(self) -> str | None:
        if self.acceptable and self.difference_mm3 > self.numerical_allowance_mm3:
            return "Volume drift exceeds the strict numerical limit but remains within the engineering allowance."
        return None

    def diagnostics(self) -> dict[str, object]:
        return {
            "volume_difference_mm3": self.difference_mm3,
            "numerical_volume_allowance_mm3": self.numerical_allowance_mm3,
            "volume_allowance_mm3": self.allowance_mm3,
            "relative_volume_allowance": RELATIVE_VOLUME_ALLOWANCE,
            "volume_warning": self.warning,
        }


def compare_volumes(
    source_mm3: float,
    measured_mm3: float,
    *,
    numerical_allowance_mm3: float = 1e-7,
    allow_zero: bool = False,
) -> VolumeComparison:
    numerical = max(numerical_allowance_mm3, abs(source_mm3) * 1e-8)
    allowance = max(numerical, abs(source_mm3) * RELATIVE_VOLUME_ALLOWANCE)
    difference = abs(source_mm3 - measured_mm3)
    acceptable = (
        all(isfinite(value) for value in (source_mm3, measured_mm3, numerical))
        and (
            (source_mm3 > 0 and measured_mm3 > 0)
            or (allow_zero and source_mm3 == measured_mm3 == 0)
        )
        and numerical > 0
        and difference <= allowance
    )
    return VolumeComparison(difference, numerical, allowance, acceptable)
