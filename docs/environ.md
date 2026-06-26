
# Environment Variables Reference

| Variable | When Needed | Description |
|----------|------------|------------|
| `AMGX_ROOT` | Installation | Path to AmgX source directory (auto-detected if not set) |
| `AMGX_BUILD` | Installation | Path to AmgX build dir (defaults to `$AMGX_ROOT/build`) |
| `CUDA_HOME` | Installation | Path to CUDA toolkit (auto-detected if not set) |
| `MPI_HOME` | Installation | Path to custom MPI installation (optional) |
| `JAXAMG_ENABLE_MPI` | Installation | Native MPI build mode (auto-detected from AmgX by default): `1` forces MPI on (build aborts if MPI is unavailable — no MPI compiler, or AmgX built without MPI), `0` forces a clean non-MPI build (no MPI toolchain needed), unset auto-enables MPI only if AmgX requires it |
| `LD_LIBRARY_PATH` | Runtime | Need to include AmgX and CUDA library paths |
| `JAXAMG_CACHE_SIZE` | Runtime | Native AmgX resource cache size: `0` disables resource caching (isolated mode), positive values (default is `1`) enables caching for performance improvement |
| `OMPI_MCA_opal_cuda_support` | Runtime | Set to `true` for GPU-aware MPI (when using OpenMPI) |
| `MPI4JAX_USE_CUDA_MPI` | Runtime | Set to `1` for GPU-aware MPI (for mpi4jax) |