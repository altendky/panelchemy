# Development and Automation

## Tool Ownership

`mise.toml` pins Python, Rust, uv, Node, pre-commit, and Lychee, following Scansor's
current tool configuration. `mise.lock` records platform resolutions, with
referenced pipx dependency locks under `.mise/locks/`. Python
dependencies live in `pyproject.toml` and `uv.lock`. Node is available for
repository checks and the isolated browser-test package. Front-end assets are
plain HTML/CSS/JavaScript packaged with Python. The Rust toolchain includes
rustfmt and Clippy. Native crate dependencies are recorded in `rust/Cargo.lock`.

The initial interpreter matrix is Python 3.12 and 3.13. CI reads exact pins from
`mise.toml`, so local tooling and CI share one source.

## Commands

```sh
mise exec -- uv sync --locked
mise exec -- cargo fmt --manifest-path rust/Cargo.toml --all -- --check
mise exec -- cargo clippy --manifest-path rust/Cargo.toml \
  --all-targets --locked --release -- -D warnings
mise exec -- cargo test --manifest-path rust/Cargo.toml --locked --release
mise exec -- uv run --locked ruff format --check src scripts tests
mise exec -- uv run --locked ruff check src scripts tests
mise exec -- uv run --locked basedpyright
mise exec -- uv build
mise exec -- uv run --locked panelchemy --version
mise exec -- pre-commit run --all-files --show-diff-on-failure
```

Run behavioral tests with `uv run --locked pytest`. CI tests conversion behavior,
builds, and installed entry points on both interpreter versions. It also checks
the native extension and packaged browser assets in the installed wheel, and
the Rust sources and Cargo lockfile in the source distribution.

## Native Distance Certificate

Maturin builds the mixed Python/Rust package, including the required
`panelchemy._native` extension. Both editable installs and distribution builds
use Cargo's optimized release profile and require the committed Cargo lockfile.
Use `mise exec -- uv sync --locked` before running from a source checkout;
building from source requires the pinned Rust toolchain. The Python CLI and
browser assets remain in the same package.

uv tracks Python code, type stubs, browser assets, the Cargo manifest, lockfile,
and Rust sources as build-cache inputs. Package changes invalidate cached
builds; changing native sources also triggers a new editable build.
The extension uses PyO3's Python 3.12 stable ABI and is tested with the pinned
Python 3.12 and 3.13 interpreters.

The geometric-distance certificate supports a partial native batch path.
Individual `within_paths` calls retain the Python implementation because the
measured serial Rust path did not improve their runtime. `within_paths_many`
can evaluate independent jobs through a bounded thread pool using the Rust
kernel, with four workers by default; a single job keeps the Python path.
Contact refinement uses this batch API for uncached, independent baseline ring
proofs during preparation, before the candidate-search deadline starts.

Input validation and exact acceptance limits remain unchanged. Inputs outside
the lossless native numeric interface retain their explicit Python path, and
the reference implementation remains a differential-test oracle. The native
extension is still a required package component: import or execution errors do
not silently choose another implementation. Other fitting, topology, and CAD
stages remain separate.

See [tracing performance](performance.md) for measured bottlenecks, implemented
solver improvements, and the limits of the timing comparisons.

## CI and Repository Settings

CI uses independent jobs for mise-lock validation, repository checks, Rust
format/lint/tests, and Python checks/builds. The aggregate result is named `all`.
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
