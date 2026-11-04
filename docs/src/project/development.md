# Development and Automation

## Tool Ownership

`mise.toml` pins Python, uv, Node, pre-commit, and Lychee, following Scansor's
current tool configuration. `mise.lock` records platform resolutions, with
referenced pipx dependency locks under `.mise/locks/`. Python
dependencies live in `pyproject.toml` and `uv.lock`. Node is available for
repository checks and a future isolated front end; no browser package exists yet.

The initial interpreter matrix is Python 3.12 and 3.13. CI reads exact pins from
`mise.toml`, so local tooling and CI share one source.

## Commands

```sh
mise exec -- uv sync --locked
mise exec -- uv run --locked ruff format --check src scripts
mise exec -- uv run --locked ruff check src scripts
mise exec -- uv run --locked basedpyright
mise exec -- uv build
mise exec -- uv run --locked panelchemy --version
mise exec -- pre-commit run --all-files --show-diff-on-failure
```

When feature tests are added, run them with `uv run --locked pytest` and add them
to the Python CI gate in the same change. The foundation currently validates
builds and the installed CLI on both interpreter versions.

## CI and Repository Settings

CI uses independent jobs for mise-lock validation, repository checks, and Python
checks/builds. The aggregate result is named `all`.
Actions are pinned by digest, workflow permissions default to read-only, and
checkout does not retain credentials. Internal Markdown links are blocking;
external link checks are advisory.

The repository follows Scansor's public visibility, merge-commit policy,
automatic branch deletion, and default-branch rules for signed commits, reviewed
pull requests, required CI, and protection from deletion or force pushes.

Renovate owns dependency-update proposals for active manifests. Mergify uses the
`enqueue` label and a single merge queue; app access and credentials are external
configuration prerequisites. Release publishing and browser deployment are
deferred until those artifacts exist.
