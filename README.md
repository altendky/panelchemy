# Panelchemy

Panelchemy is a local tool in development for converting comic-panel artwork
into configurable, three-color layered relief.

The repository foundation, image-space separation, and local browser workflow
are in place. Geometry and physical printing remain tracked stages; no printable
model exists yet.

## Development

Install [mise](https://mise.jdx.dev/), then run:

```sh
mise trust
mise install --locked
mise exec -- uv sync --locked
mise exec -- pre-commit install
mise exec -- uv run --locked panelchemy --help
```

See the [project documentation](docs/src/project/index.md) for the current
direction, [development commands](docs/src/project/development.md), and
[work stages](docs/src/project/stages.md).

Launch the [local browser workflow](docs/src/project/local-browser.md) with:

```sh
mise exec -- uv run --locked panelchemy serve
```

Create [separation previews](docs/src/project/separation.md) with:

```sh
mise exec -- uv run --locked panelchemy separate /path/to/artwork.png --output local-inputs/preview
```

## Layout

- `src/panelchemy/`: reusable Python core and command-line adapter.
- `tests/`: behavioral tests as conversion features are implemented.
- `docs/src/project/`: scope, decisions, architecture, and development guidance.
- `experiments/`: bounded investigations and future browser experiments.
- `examples/`: reproducible, deliberately retained examples.
- `local-inputs/`: ignored local evaluation inputs.
- `scripts/`: repository tooling.

Scansor is a reference for repository structure and tooling. The projects have
different purposes and processing logic.

## License

Code is available under either the MIT or Apache 2.0 license, at your option.
See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
User-supplied artwork is not included in this repository or covered by that
statement.
