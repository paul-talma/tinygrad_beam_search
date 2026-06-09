# Beam Search: Metal Multiprocessing Crash

## Issue

Running beam search (`BEAM=N`) on Metal crashes with:

```
IndexError: pop from an empty deque
```

This is a secondary exception from Python's `multiprocessing.pool` — the real cause is that worker processes die silently. Setting `PARALLEL=0` works around it but makes beam search slow (sequential compilation and benchmarking).

## Root Cause (suspected)

`beam_search` in `tinygrad/codegen/opt/search.py` spawns a worker pool using `multiprocessing.get_context("spawn")`. Workers compile candidate kernels via `to_program`, which on Metal likely calls into the Metal framework API to compile shaders. That framework state is not available in `spawn`ed subprocesses on macOS, causing workers to crash.

Workers are initialized with `ALLOW_DEVICE_USAGE=0` (`_init_worker`), but shader compilation appears to still require Metal framework access.

## Workaround

```sh
BEAM=4 PARALLEL=0 uv run python3 your_script.py
```

## Potential Fix

Separate shader compilation into two phases:
1. Generate the Metal source string (CPU-side, safe in workers)
2. Compile the Metal source via the framework (main process only)

Only phase 1 needs to run in the worker pool. Phase 2 could be done in the main process before benchmarking.

Alternatively, investigate whether `fork` context instead of `spawn` avoids the issue (risky if any Metal state is initialized before the fork).

## Relevant Files

- `tinygrad/codegen/opt/search.py` — `beam_search`, `_try_compile`, `_init_worker`
