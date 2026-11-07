# Development and Automation

## Tool Ownership

`mise.toml` pins Python, uv, Node, pre-commit, and Lychee, following Scansor's
current tool configuration. `mise.lock` records platform resolutions, with
referenced pipx dependency locks under `.mise/locks/`. Python
dependencies live in `pyproject.toml` and `uv.lock`. Node is available for
repository checks and the isolated browser-test package. Front-end assets are
plain HTML/CSS/JavaScript packaged with Python.

The initial interpreter matrix is Python 3.12 and 3.13. CI reads exact pins from
`mise.toml`, so local tooling and CI share one source.

## Commands

```sh
mise exec -- uv sync --locked
mise exec -- uv run --locked ruff format --check src scripts tests
mise exec -- uv run --locked ruff check src scripts tests
mise exec -- uv run --locked basedpyright
mise exec -- uv build
mise exec -- uv run --locked panelchemy --version
mise exec -- pre-commit run --all-files --show-diff-on-failure
```

Run behavioral tests with `uv run --locked pytest`. CI tests conversion behavior,
builds, and installed entry points on both interpreter versions.

## CI and Repository Settings

CI uses independent jobs for mise-lock validation, repository checks, and Python
checks/builds. The aggregate result is named `all`.
Actions are pinned by digest, workflow permissions default to read-only, and
checkout does not retain credentials. Internal Markdown links are blocking;
external link checks are advisory.

The repository follows Scansor's public visibility, merge-commit policy,
automatic branch deletion, and default-branch rules for signed commits,
pull requests, required CI, and protection from deletion or force pushes.

Renovate owns dependency-update proposals for active manifests. Mergify uses the
`enqueue` label and a single merge queue; app access and credentials are external
configuration prerequisites. Release publishing and browser deployment are
deferred until those artifacts exist.
