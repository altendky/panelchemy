"""Loopback-only browser adapter; conversion and artifacts stay in memory."""

import base64
import json
from contextlib import suppress
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from io import BytesIO
from pathlib import PurePosixPath
from socket import socket
from threading import BoundedSemaphore
from typing import cast, override

from PIL import Image

from panelchemy.separation import SeparationConfig, prepare_artifacts


@dataclass(frozen=True)
class Limits:
    max_request_bytes: int = 32 * 1024 * 1024
    max_pixels: int = 4_000_000

    def __post_init__(self) -> None:
        if self.max_request_bytes < 1 or self.max_pixels < 1:
            raise ValueError("request and pixel limits must be positive")


class ResourceLimitError(ValueError):
    """An upload exceeds an explicit local resource budget."""


def _data_url(content: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(content).decode('ascii')}"


def _upload(value: object, limits: Limits) -> tuple[str, bytes]:
    if not isinstance(value, dict):
        raise ValueError("upload must contain a name and base64 data")
    data = cast("dict[str, object]", value)
    name, encoded = data.get("name"), data.get("data")
    if not isinstance(name, str) or not name or len(name) > 255:
        raise ValueError("upload needs a filename of 1-255 characters")
    name = PurePosixPath(name.replace("\\", "/")).name
    if not name:
        raise ValueError("upload filename is empty")
    if not isinstance(encoded, str):
        raise ValueError("upload data must be a base64 string")
    try:
        content = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("upload is not valid base64") from error
    try:
        with Image.open(BytesIO(content)) as image:
            if image.width * image.height > limits.max_pixels:
                raise ResourceLimitError(
                    f"image exceeds the {limits.max_pixels:,}-pixel limit"
                )
    except OSError as error:
        raise ValueError(
            "Choose a readable image file for the artwork or mask"
        ) from error
    return name, content


def conversion_response(payload: object, limits: Limits) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    data = cast("dict[str, object]", payload)
    source_name, source = _upload(data.get("source"), limits)
    raw_config = data.get("config", {})
    if not isinstance(raw_config, dict):
        raise ValueError("config must be a JSON object")
    config = SeparationConfig.from_mapping(cast("dict[str, object]", raw_config))
    mask_name, mask_bytes = "silhouette.png", None
    if data.get("silhouette") is not None:
        mask_name, mask_bytes = _upload(data["silhouette"], limits)
    bundle = prepare_artifacts(
        source,
        source_name,
        config,
        silhouette_bytes=mask_bytes,
        silhouette_name=mask_name,
    )
    return {
        "manifest": bundle.manifest,
        "source_preview": _data_url(bundle.source_preview, "image/png"),
        "previews": {
            name.removesuffix(".png"): _data_url(content, "image/png")
            for name, content in bundle.files.items()
            if name.endswith(".png")
        },
        "archive": _data_url(bundle.archive(), "application/zip"),
    }


class PreviewServer(ThreadingHTTPServer):
    daemon_threads: bool = True

    def __init__(self, port: int = 8765, limits: Limits | None = None) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        self.limits: Limits = limits or Limits()
        self.conversion_slot: BoundedSemaphore = BoundedSemaphore(1)
        super().__init__(("127.0.0.1", port), PreviewHandler)


class PreviewHandler(BaseHTTPRequestHandler):
    """Fixed routes; uploads and filenames never become filesystem paths."""

    @override
    def setup(self) -> None:
        super().setup()
        cast("socket", self.connection).settimeout(15)

    @property
    def app(self) -> PreviewServer:
        return cast("PreviewServer", self.server)

    @override
    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, status: int, content: bytes, media_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        _ = self.wfile.write(content)

    def _json(self, status: int, data: dict[str, object]) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def _allowed(self) -> bool:
        hosts = {
            f"127.0.0.1:{self.app.server_port}",
            f"localhost:{self.app.server_port}",
        }
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in hosts or (
            origin is not None and origin not in {f"http://{host}" for host in hosts}
        ):
            self._json(
                HTTPStatus.FORBIDDEN, {"error": "Use this server's local browser URL"}
            )
            return False
        return True

    def do_GET(self) -> None:
        if not self._allowed():
            return
        routes = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        if self.path == "/api/health":
            self._json(HTTPStatus.OK, {"status": "ready"})
        elif self.path in routes:
            name, media_type = routes[self.path]
            self._send(
                HTTPStatus.OK,
                files("panelchemy").joinpath("web", name).read_bytes(),
                media_type,
            )
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown route"})

    def do_POST(self) -> None:
        if not self._allowed():
            return
        if self.path != "/api/separate":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown route"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Send application/json"}
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length"})
            return
        if length < 1:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Request body is empty"})
            return
        if length > self.app.limits.max_request_bytes:
            self._json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "Upload exceeds the request size limit"},
            )
            return
        if not self.app.conversion_slot.acquire(blocking=False):
            self._json(
                HTTPStatus.TOO_MANY_REQUESTS,
                {"error": "A conversion is already running; try again shortly"},
            )
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete request body")
            try:
                payload = cast("object", json.loads(raw))
            except (ValueError, UnicodeError, RecursionError):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
                return
            try:
                response = conversion_response(payload, self.app.limits)
            except (ResourceLimitError, Image.DecompressionBombError) as error:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": str(error)})
            except (ValueError, OSError) as error:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})
            else:
                self._json(HTTPStatus.OK, response)
        except (TimeoutError, ValueError):
            self._json(
                HTTPStatus.BAD_REQUEST, {"error": "Incomplete or timed-out upload"}
            )
        finally:
            self.app.conversion_slot.release()


def serve(port: int = 8765, limits: Limits | None = None) -> None:
    with PreviewServer(port, limits) as server:
        print(
            f"Panelchemy is ready at http://127.0.0.1:{server.server_port}", flush=True
        )
        with suppress(KeyboardInterrupt):
            server.serve_forever()
