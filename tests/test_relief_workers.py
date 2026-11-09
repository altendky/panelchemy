"""Execution worker settings reach the fitter without becoming print settings."""

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

from panelchemy import relief
from panelchemy.cli import main
from panelchemy.construction import ConstructionProfile
from panelchemy.curves import CurveFitResult
from panelchemy.separation import SeparationResult, separate


class ReachedFitter(Exception):
    pass


@pytest.mark.parametrize("workers", [None, 4])
def test_build_relief_forwards_default_and_explicit_worker_counts(
    monkeypatch: pytest.MonkeyPatch, workers: int | None
) -> None:
    seen: list[int] = []

    def fit(*_args: object, workers: int = 1, **_kwargs: object) -> CurveFitResult:
        seen.append(workers)
        raise ReachedFitter

    monkeypatch.setattr(relief, "fit_outlines", fit)
    image = Image.new("RGB", (20, 20), "black")
    image.paste("white", (5, 5, 15, 15))
    result = separate(image, silhouette=np.ones((20, 20), dtype=np.bool_))
    profile = ConstructionProfile(long_edge_mm=20, contour_tolerance_mm=0.2)
    kwargs = {} if workers is None else {"workers": workers}
    with pytest.raises(ReachedFitter):
        _ = relief.build_relief(result, profile, **kwargs)
    assert seen == [1 if workers is None else workers]
    assert "workers" not in profile.as_dict()


@pytest.mark.parametrize("workers", [None, 3])
def test_file_adapter_forwards_worker_count_before_native_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int | None
) -> None:
    source = tmp_path / "input.png"
    Image.new("RGB", (3, 3), "black").save(source)
    seen: list[int] = []

    def build(
        _result: SeparationResult,
        _profile: ConstructionProfile | None = None,
        *,
        max_vertices: int = 25_000,
        workers: int = 1,
    ) -> relief.ReliefModel:
        assert max_vertices == 123
        seen.append(workers)
        raise ReachedFitter

    monkeypatch.setattr(relief, "build_relief", build)
    with pytest.raises(ReachedFitter):
        if workers is None:
            _ = relief.process_relief_file(source, tmp_path / "out", max_vertices=123)
        else:
            _ = relief.process_relief_file(
                source, tmp_path / "out", max_vertices=123, workers=workers
            )
    assert seen == [1 if workers is None else workers]
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("workers", [0, -1, True, False, 1.5, "2"])
def test_public_relief_adapters_reject_nonpositive_or_noninteger_workers(
    tmp_path: Path, workers: object
) -> None:
    count = cast("int", workers)
    result = separate(Image.new("RGB", (3, 3), "black"))
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        _ = relief.build_relief(result, workers=count)
    # Invalid execution settings fail before trying to read a missing source.
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        _ = relief.process_relief_file(
            tmp_path / "missing.png", tmp_path / "out", workers=count
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("command", ["relief", "export"])
@pytest.mark.parametrize("workers", [None, 4])
def test_cli_forwards_worker_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    workers: int | None,
) -> None:
    seen: list[int] = []

    def process(
        *_args: object, workers: int = 1, **_kwargs: object
    ) -> dict[str, object]:
        seen.append(workers)
        return {"workers": workers}

    monkeypatch.setattr(relief, "process_relief_file", process)
    arguments = [
        "panelchemy",
        command,
        str(tmp_path / "input.png"),
        "--output",
        str(tmp_path / "out"),
    ]
    if workers is not None:
        arguments.extend(("--workers", str(workers)))
    if command == "export":
        arguments.extend(("--format", "step"))
    monkeypatch.setattr("sys.argv", arguments)
    main()
    expected = 1 if workers is None else workers
    assert seen == [expected]
    assert json.loads(capsys.readouterr().out) == {"workers": expected}


@pytest.mark.parametrize("command", ["relief", "export"])
@pytest.mark.parametrize("workers", ["0", "-1", "1.5"])
def test_invalid_cli_workers_are_actionable_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    workers: str,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "panelchemy",
            command,
            str(tmp_path / "missing.png"),
            "--output",
            str(tmp_path / "out"),
            "--workers",
            workers,
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert "workers" in error
    assert "Traceback" not in error
    assert not (tmp_path / "out").exists()
