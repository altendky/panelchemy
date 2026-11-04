"""Exercise the actual local HTTP adapter and shared conversion artifacts."""

import base64
import json
from collections.abc import Iterator
from contextlib import closing
from http.client import HTTPConnection
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import cast
from zipfile import ZipFile

import pytest
from PIL import Image, ImageDraw

from panelchemy.separation import process_file
from panelchemy.server import Limits, PreviewServer


def artwork() -> bytes:
    image = Image.new("RGB", (12, 16), "white")
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((1, 1, 10, 14), fill="black")
    drawing.rectangle((3, 3, 6, 9), fill="white")
    drawing.rectangle((3, 11, 8, 13), fill=(245, 166, 35))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def server() -> Iterator[PreviewServer]:
    app = PreviewServer(0)
    thread = Thread(target=app.serve_forever, daemon=True)
    thread.start()
    try:
        yield app
    finally:
        app.shutdown()
        app.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def request(
    server: PreviewServer,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    with closing(HTTPConnection("127.0.0.1", server.server_port, timeout=5)) as client:
        client.request(method, path, body=body, headers=headers or {})
        response = client.getresponse()
        return response.status, response.read()


def payload() -> dict[str, object]:
    return {
        "source": {"name": "input.png", "data": base64.b64encode(artwork()).decode()}
    }


def post(server: PreviewServer, data: object) -> tuple[int, dict[str, object]]:
    status, body = request(
        server,
        "POST",
        "/api/separate",
        json.dumps(data).encode(),
        {"Content-Type": "application/json"},
    )
    return status, cast("dict[str, object]", json.loads(body))


def test_browser_response_and_zip_match_cli(
    server: PreviewServer, tmp_path: Path
) -> None:
    source = tmp_path / "input.png"
    _ = source.write_bytes(artwork())
    output = tmp_path / "cli"
    manifest = process_file(source, output)
    status, response = post(server, payload())
    assert status == 200
    assert response["manifest"] == manifest
    archive_url = cast("str", response["archive"])
    assert archive_url.startswith("data:application/zip;base64,")
    with ZipFile(BytesIO(base64.b64decode(archive_url.split(",", 1)[1]))) as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == {
            path.name: path.read_bytes() for path in output.iterdir()
        }
    previews = cast("dict[str, str]", response["previews"])
    assert previews["support-orange"].startswith("data:image/png;base64,")
    assert cast("str", response["source_preview"]).startswith("data:image/png;base64,")


def test_static_assets_and_traversal(server: PreviewServer) -> None:
    for path, marker in [
        ("/", b"Panelchemy"),
        ("/app.js", b"/api/separate"),
        ("/style.css", b"{"),
    ]:
        status, body = request(server, "GET", path)
        assert status == 200
        assert marker in body
    for path in ["/../pyproject.toml", "/%2e%2e/pyproject.toml", "/separation.py"]:
        assert request(server, "GET", path)[0] == 404


def test_host_and_origin_checks(server: PreviewServer) -> None:
    assert request(server, "GET", "/", headers={"Host": "attacker.example"})[0] == 403
    body = json.dumps(payload()).encode()
    assert (
        request(
            server,
            "POST",
            "/api/separate",
            body,
            {
                "Origin": "https://attacker.example",
                "Content-Type": "application/json",
            },
        )[0]
        == 403
    )
    assert (
        request(
            server,
            "POST",
            "/api/separate",
            body,
            {
                "Origin": f"http://127.0.0.1:{server.server_port}",
                "Content-Type": "application/json",
            },
        )[0]
        == 200
    )


def test_invalid_json_and_media_type(server: PreviewServer) -> None:
    assert (
        request(server, "POST", "/api/separate", b"{}", {"Content-Type": "text/plain"})[
            0
        ]
        == 415
    )
    assert (
        request(
            server, "POST", "/api/separate", b"{", {"Content-Type": "application/json"}
        )[0]
        == 400
    )
    assert (
        request(server, "POST", "/other", b"{}", {"Content-Type": "application/json"})[
            0
        ]
        == 404
    )


@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {"source": {"name": "a.png", "data": "!"}},
        {
            "source": {
                "name": "a.png",
                "data": base64.b64encode(b"not an image").decode(),
            }
        },
        {**payload(), "config": {"neutral_tolerance": -1}},
        {**payload(), "config": []},
        {**payload(), "silhouette": {"name": "mask.png", "data": "!"}},
    ],
)
def test_invalid_requests_are_actionable(
    server: PreviewServer, invalid: object
) -> None:
    status, response = post(server, invalid)
    assert status == 422
    assert isinstance(response["error"], str)
    assert "Traceback" not in response["error"]


def test_limits_and_busy_conversion_are_explicit(server: PreviewServer) -> None:
    server.limits = Limits(max_pixels=1)
    assert post(server, payload())[0] == 413
    server.limits = Limits(max_request_bytes=2)
    assert post(server, payload())[0] == 413
    server.limits = Limits()
    assert server.conversion_slot.acquire(blocking=False)
    try:
        assert post(server, payload())[0] == 429
    finally:
        server.conversion_slot.release()
    assert post(server, payload())[0] == 200


def test_mask_override_and_dimension_error(server: PreviewServer) -> None:
    image = Image.new("L", (12, 16), 255)
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    data = payload()
    data["silhouette"] = {
        "name": "mask.png",
        "data": base64.b64encode(encoded.getvalue()).decode(),
    }
    status, response = post(server, data)
    assert status == 200
    assert (
        cast("dict[str, object]", response["manifest"])["silhouette_method"]
        == "explicit-mask"
    )
    data["silhouette"] = {
        "name": "small.png",
        "data": base64.b64encode(artwork()).decode(),
    }
    source = Image.new("RGB", (13, 16), "black")
    encoded = BytesIO()
    source.save(encoded, format="PNG")
    data["source"] = {
        "name": "larger.png",
        "data": base64.b64encode(encoded.getvalue()).decode(),
    }
    status, response = post(server, data)
    assert status == 422
    assert "same-size" in cast("str", response["error"])
