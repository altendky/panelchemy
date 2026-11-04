# Local Browser Workflow

Start the application with:

```sh
uv run --locked panelchemy serve
```

Open the printed local URL, normally `http://127.0.0.1:8765`. Select artwork,
adjust the palette and neutral tolerance, and generate previews. Inspect the
normalized artwork, panel cutout, visible-color masks, and material supports.
Boundary warnings and the reproducibility record accompany each result.

An optional same-size silhouette image overrides the boundary decision. White
selects the panel; black excludes the gutter. The conversion does not remove
small features or repair gaps automatically.

Download outputs to save the preview PNGs, resolved configuration, and manifest
as a ZIP. Keep the original source image so its recorded hash can be checked
and the result regenerated through the CLI. Changing an input disables the
previous download until a new result is generated.

## Runtime

The Python server and browser adapter call the same in-memory artifact builder
as the CLI. Uploads are processed on this computer, retained only for the request,
and never interpreted as filesystem paths. Source previews use the same decoded
pixel orientation as processing. The server binds only to the loopback address
and checks request Host/Origin; it is a local development adapter.

The default limits are one concurrent conversion, 32 MiB per JSON upload request,
and 4 million pixels per source/mask image. Oversize requests fail explicitly;
images are never silently reduced. The JSON request includes base64 image data,
which increases request size relative to the original file. Adjust limits or
the port when needed:

```sh
uv run --locked panelchemy serve --port 9000 --max-pixels 8000000 --max-request-mib 64
```

Stop the server with Ctrl+C. Hosted deployment and solid generation are separate
stages.

## Browser Checks

The front end is packaged plain HTML/CSS/JavaScript. Its test-only npm manifest
and lockfile live under `browser-tests/`. To run actual Chromium checks:

```sh
uv sync --locked
cd browser-tests
npm ci
npx --no-install playwright install chromium
npm test
```

The test runner starts and stops its own local server on port 8766. Override
`PANELCHEMY_TEST_PORT` if that port is occupied. Python tests separately verify
request limits, error handling, local-origin restrictions, and byte-for-byte
artifact equivalence with CLI output. CI also checks the browser flow.
