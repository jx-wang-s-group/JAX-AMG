# Development Guide

Notes for contributors working on JAX-AMG: setting up a development install,
running the test suite, and the code-quality workflow.

## Development Install

Follow the [Installation Guide](install.md) to set up CUDA, AmgX, and the build
environment variables (`CUDA_HOME`, `AMGX_ROOT`, `AMGX_BUILD`). Then install in
editable mode with the development and MPI extras:

```bash
pip install -e ".[all]"     # editable install + dev + mpi extras
```

This pulls in `pytest`, `pytest-mpi`, `ruff`, `black`, `mypy`, `pre-commit`, and
the MPI dependencies. The equivalent pinned list lives in
`requirements-dev.txt`.

!!! note
    JAX-AMG links against a CUDA build of AmgX, so AmgX and CUDA must be on
    `LD_LIBRARY_PATH` at runtime (see
    [Post-Installation Setup](install.md#post-installation-setup)). Running with
    a bare path to the Python interpreter, instead of an activated environment,
    drops these variables and yields `AMGX Error: Error initializing amgx core`.

## Running Tests

The suite uses `pytest`. Pin to a specific device with `CUDA_VISIBLE_DEVICES`.

### Non-MPI tests

Tests that require MPI are marked `@pytest.mark.mpi` and are **skipped by
default**, so a plain run exercises everything else on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/
```

### MPI tests

The MPI tests (`tests/test_mpi.py`) must be launched under `mpirun` and enabled
with the `--with-mpi` flag provided by `pytest-mpi`. Most need at least two
ranks; assign a distinct GPU to each via `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMPI_MCA_opal_cuda_support=true MPI4JAX_USE_CUDA_MPI=1 \
mpirun -n 2 python -m pytest --only-mpi tests/
```

- `--only-mpi` runs *only* the MPI-marked tests.
- Omit the GPU-aware MPI variables (or set `MPI4JAX_USE_CUDA_MPI=0`) to stage
  communication through host memory when GPU-aware MPI is unavailable. See the
  [MPI Guide](mpi.md#gpu-aware-mpi) and
  [Environment Variables Reference](environ.md) for details.

!!! tip
    The MPI demos under `demo/` (e.g. `mpirun -n 4 python demo/mpi_autodiff.py`)
    are a quick way to sanity-check distributed behavior end to end.

## Code Quality

Formatting and linting are enforced with `pre-commit` (black, ruff with
`--fix`, and mypy).

```bash
pre-commit install            # run automatically on each commit
pre-commit run --all-files    # run manually across the repo
```

## Rebuilding the Native Extension

The solver is a C++ FFI extension. After editing any C++ source or header in
`jaxamg/`, reinstall to rebuild it before testing; Python-only changes do not
need a rebuild:

```bash
pip install -e . -v
```

MPI support in the extension is auto-enabled when AmgX was built with MPI; set
`JAXAMG_ENABLE_MPI=1` to force it on.

## Building the Docs

The documentation is built with MkDocs from the sources in `docs/`
(configuration in `mkdocs.yml`). Preview it locally with:

```bash
mkdocs serve
```
