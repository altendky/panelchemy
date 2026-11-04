"""Actionable command-line errors for rejected image inputs."""

from pathlib import Path

import pytest
from PIL import Image

from panelchemy.cli import main


@pytest.mark.parametrize("command", ["separate", "relief"])
def test_oversize_image_is_reported_without_traceback(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "oversize.png"
    Image.new("RGB", (3, 3)).save(source)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    monkeypatch.setattr(
        "sys.argv",
        ["panelchemy", command, str(source), "--output", str(tmp_path / "out")],
    )
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert "exceeds limit" in error
    assert "Traceback" not in error
    assert not (tmp_path / "out").exists()
