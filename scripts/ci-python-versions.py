#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Expose the mise interpreter pins as GitHub Actions outputs."""

import json
import tomllib
from pathlib import Path
from typing import cast


def main() -> None:
    """Use the same interpreter versions for local tooling and CI."""
    config = cast(
        "dict[str, dict[str, object]]",
        tomllib.loads((Path(__file__).resolve().parents[1] / "mise.toml").read_text()),
    )
    versions: object = config["tools"]["python"]
    if not isinstance(versions, list) or not versions:
        raise ValueError("mise.toml must declare a non-empty Python version list")
    if not all(isinstance(value, str) for value in cast("list[object]", versions)):
        raise ValueError("Python version pins must be strings")
    print(f"versions={json.dumps(versions)}")
    print(f"primary={versions[0]}")


if __name__ == "__main__":
    main()
