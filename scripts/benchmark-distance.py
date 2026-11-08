"""Compare reference, native, and concurrent source proofs on identical jobs.

Run with ``uv run python scripts/benchmark-distance.py``. Optional --cases JSON
is a list of {name, source_paths, spans:[{control_points,is_line}],tolerance_mm}.
Artwork-derived case packets should remain local. No model or STEP is built.
"""

# Benchmarks deliberately exercise the retained reference implementation.
# pyright: reportPrivateUsage=false

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path
from time import perf_counter
from typing import cast

from panelchemy import geometric_distance as distance
from panelchemy.curves import BezierSpan, Controls, Point2


@dataclass(frozen=True)
class Case:
    name: str
    paths: tuple[tuple[Point2, ...], ...]
    spans: tuple[BezierSpan, ...]
    tolerance: float


def load_cases(path: Path | None) -> list[Case]:
    if path is None:
        k = 0.5522847498307936
        return [
            Case(
                f"arc-{i}",
                (
                    tuple(
                        (radius * cos(j * pi / 160), radius * sin(j * pi / 160))
                        for j in range(81)
                    ),
                ),
                (BezierSpan(((1, 0), (1, k), (k, 1), (0, 1)), False),),
                0.2,
            )
            for i, radius in enumerate((1.0, 1.195, 1.205) * 8)
        ]
    packets = cast("list[dict[str, object]]", json.loads(path.read_text()))
    cases: list[Case] = []
    for packet in packets:
        paths = cast("list[list[list[float]]]", packet["source_paths"])
        spans = cast("list[dict[str, object]]", packet["spans"])
        cases.append(
            Case(
                str(packet["name"]),
                tuple(tuple((x, y) for x, y in path) for path in paths),
                tuple(
                    BezierSpan(
                        cast(
                            "Controls",
                            tuple(
                                tuple(point)
                                for point in cast(
                                    "list[list[float]]", span["control_points"]
                                )
                            ),
                        ),
                        cast("bool", span["is_line"]),
                    )
                    for span in spans
                ),
                cast("float", packet["tolerance_mm"]),
            )
        )
    if not cases:
        raise ValueError("benchmark requires at least one case")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--cases", type=Path)
    _ = parser.add_argument("--repeat", type=int, default=1)
    _ = parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    repeat, workers = cast("int", args.repeat), cast("int", args.workers)
    if repeat < 1 or workers < 1:
        parser.error("repeat and workers must be positive")
    cases = load_cases(cast("Path | None", args.cases)) * repeat

    def reference(case: Case) -> bool:
        return distance._within_paths_python(case.paths, case.spans, case.tolerance)

    def native(case: Case) -> bool:
        return distance._within_paths_native(case.paths, case.spans, case.tolerance)

    tick = perf_counter()
    expected = [reference(case) for case in cases]
    python_seconds = perf_counter() - tick
    tick = perf_counter()
    serial = [native(case) for case in cases]
    rust_seconds = perf_counter() - tick
    tick = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parallel = list(executor.map(native, cases))
    threaded_seconds = perf_counter() - tick
    if expected != serial or expected != parallel:
        raise AssertionError("native and reference certificate decisions differ")
    print(
        json.dumps(
            dict(
                scope="source-distance certificates only; includes packing and executor startup/shutdown, excludes CAD",
                calls=len(cases),
                workers=workers,
                accepted=sum(expected),
                all_decisions_match=True,
                python_seconds=python_seconds,
                rust_seconds=rust_seconds,
                threaded_rust_seconds=threaded_seconds,
                serial_speedup=python_seconds / rust_seconds,
                total_speedup=python_seconds / threaded_seconds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
