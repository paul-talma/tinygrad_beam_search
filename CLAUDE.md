# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
# Install base package (creates .venv and uv.lock)
uv sync

# Install with test dependencies
uv sync --extra testing

# Install with linting tools only
uv sync --extra linting

# Run the minimal pre-commit test suite
uv run python3 -m pytest test/test_tiny.py

# Run a specific test file
uv run python3 test/backend/test_ops.py

# Run a single test
uv run python3 -m pytest test/backend/test_ops.py::TestOps::test_add -x

# Run the full suite (slow)
uv run python3 -m pytest test/

# Lint
uv run python3 -m ruff check .
uv run python3 -m pylint --disable=all -e W0311 -e C0303 --jobs=0 --indent-string='  ' --recursive=y .

# Type check
uv run python3 -m mypy

# Install pre-commit hooks (runs ruff, mypy, test_tiny, test_ops on every commit)
uv run pre-commit install
```

## Debugging env vars

| Variable | Values | Effect |
|---|---|---|
| `DEBUG` | 1–7 | Increasing verbosity: devices → timings → opts → generated code → UOps → linearized UOps → assembly |
| `DEV` | `CPU`, `METAL`, `AMD`, `NV`, `CUDA`, `CL`, … | Select backend (e.g. `DEV=CPU:LLVM`) |
| `BEAM` | integer | Kernel beam-search width (0 = disabled) |
| `NOOPT` | 1 | Disable all kernel optimizations |
| `VIZ` | 1 | Launch the graph-rewrite visualizer |

Use `Context(DEBUG=4)` as a decorator or context manager to scope debugging to one function.

## External references

- Architecture notes: https://mesozoic-egg.github.io/tinygrad-notes
- Official docs: https://docs.tinygrad.org (speed bottlenecks: https://docs.tinygrad.org/developer/speed)

## Architecture

tinygrad is a **lazy tensor library with an end-to-end compiler**. A tensor op is never executed immediately; instead it builds a UOp graph that is compiled and run only when `.realize()` or `.numpy()` is called.

### Data flow

```
Tensor API  →  UOp graph  →  Schedule  →  Codegen  →  Renderer  →  Runtime
```

### Key modules

**`tinygrad/tensor.py`** — The user-facing `Tensor` class. Every tensor operation records a UOp node rather than executing; no computation happens here.

**`tinygrad/uop/`** — The universal IR.
- `ops.py` — `UOp` (the IR node), `Ops` (the op enum), `PatternMatcher` / `graph_rewrite` (the rewrite engine), `Variable` (symbolic ints), and helpers like `smax`/`smin`/`resolve`.
- `symbolic.py` — Symbolic simplification patterns applied throughout the pipeline.
- `decompositions.py` — Decomposition patterns that lower complex ops into primitives.
- `spec.py` — Type-checking specifications (`type_verify`).

**`tinygrad/schedule/`** — Turns a UOp graph into a linear sequence of kernel calls.
- `__init__.py` (`create_schedule`) — Topological sort of kernel dependencies.
- `rangeify.py` — Adds RANGE loops and buffer assignments.
- `indexing.py` — Derives index math from shapes/strides.
- `memory.py` — Buffer allocation and aliasing.

**`tinygrad/codegen/`** — Lowers a scheduled kernel UOp into target code.
- `__init__.py` (`full_rewrite_to_sink`) — Orchestrates all lowering passes: symbolic simplification → range splitting → GPU dim assignment → optimization → late lowering → linearization.
- `gpudims.py` — Maps axes to GLOBAL/LOCAL/THREAD dimensions.
- `opt/` — Kernel optimizations searched by beam search (`postrange.py`, `tc.py` for tensor cores).
- `late/` — Final lowering: expander (vectorization), devectorizer, linearizer (CFG generation).
- `simplify.py` — Range merging/flattening passes.

**`tinygrad/renderer/`** — Converts a linearized UOp sequence into source text for a target.
- `cstyle.py` — Base C-style renderer (shared by CUDA, HIP, Metal, OpenCL).
- `llvmir.py`, `ptx.py`, `nir.py`, `wgsl.py` — Target-specific renderers.
- `amd/` — AMD-specific renderers.

**`tinygrad/engine/`** — Execution.
- `realize.py` — `run_linear`: takes the linearized schedule, compiles programs, and dispatches them to devices.
- `jit.py` — `TinyJit`: captures a kernel sequence on first run, replays it on subsequent runs.

**`tinygrad/runtime/`** — Device backends. Each `ops_*.py` file implements the low-level interface (~25 ops) for a hardware target (Metal, CUDA, AMD, NV, CPU/LLVM, OpenCL, …).

**`tinygrad/device.py`** — `Buffer` and `Device` abstractions shared across backends.

**`tinygrad/dtype.py`** — `DType`, `dtypes`, `PtrDType`, dtype promotion rules.

**`tinygrad/helpers.py`** — Utilities: `getenv`, `ContextVar`, `Context`, `DEBUG`, `BEAM`, `prod`, `flatten`, `partition`, etc.

**`tinygrad/gradient.py`** — Autograd: computes gradients symbolically over the UOp graph.

**`tinygrad/nn/`** — Layers, optimizers (`Adam`, `SGD`, …), and data loaders.

### ShapeTracker

Zero-copy shape manipulation via `(shape, strides, offset, mask)` Views. A multi-view `ShapeTracker` handles cases not expressible as a single affine index map (e.g. after a non-contiguous `reshape`).

### Test layout

- `test/test_tiny.py` — Minimal fast suite; runs on every commit.
- `test/backend/` — Full backend tests: ops, schedule, tensor, JIT, multitensor, etc.
- `test/unit/` — Focused unit tests for scheduler, codegen, optimizer, etc.
- `test/null/` — Pure-Python / device-free tests (pattern matcher, UOp symbolic, etc.).

## Speed bottlenecks

- **Compile speed** — UOp graph rewrite passes (Python); on par with `torch.compile`.
- **Execution speed** — not a bottleneck; `TinyJit` replay is fast.
- **Model speed (scheduler)** — biggest training bottleneck; recompute-vs-materialize decision is unsolved.
- **Kernel speed (codegen)** — no explicit SRAM tiling; limited tensor core support; no TMA support.

## Active project: learned beam search cost model

The current focus is replacing exhaustive benchmarking in `beam_search` (`tinygrad/codegen/opt/search.py`)
with a learned cost model that predicts kernel runtime from features, inspired by TVM/Ansor.

### Problem

`beam_search` finds the best `Opt` sequence for a kernel by compiling and timing every candidate.
This is correct but slow: for beam width `amt` and `N` candidate actions per step, each beam step
times O(amt × N) kernels. The cost dominates any workload that hasn't been seen before (cold cache).

### Approach

Predict `(kernel_features, applied_opts) → runtime` without running the kernel. The model prunes
or ranks candidates so only the top-K are actually benchmarked.

**Unit of prediction:** one `Scheduler` instance after a specific `applied_opts` sequence.
The Scheduler is the object beam search already operates on; see `benchmarks/scheduler_features.md`
for a full attribute reference.

**Key feature groups** (all extractable from a `Scheduler` before benchmarking):
- Shape: per-`AxisType` size products (`global_size`, `reduce_size`, `upcast_size`, …)
- Reduction structure: presence, op type (`ADD`/`MAX`), output dtype
- Buffer stride matrix: which loop axes appear in each buffer's index expression, and with what stride — extracted from `INDEX` node index expressions in `ast`
- Op mix: ALU node counts from `ast.toposort()`, especially transcendental count
- Applied opts: bag-of-opts vector or sequence encoding of `applied_opts`
- Device constants: `shared_max`, `global_max`, device embedding

**Key files:**
- `tinygrad/codegen/opt/search.py` — beam search loop; `get_kernel_actions`, `_time_program`
- `tinygrad/codegen/opt/postrange.py` — `Scheduler` class; all kernel features live here
- `tinygrad/codegen/opt/heuristic.py` — hand-coded optimizer; good reference for which features actually predict good opts
- `benchmarks/beam_search/` — benchmarking harness and collected results
- `benchmarks/scheduler_features.md` — annotated reference for all `Scheduler` attributes

### Design notes

- Beam search operates **per-kernel**, not over the full graph. Each kernel has its own `Scheduler`.
- tinygrad does **not** do horizontal fusion of independent computations — two independent matmuls
  are always separate kernels, each with its own beam search run.
- The stride matrix (from `bufs`) is the richest non-obvious feature: it encodes broadcast vs.
  streaming vs. strided access per buffer, which is what determines whether LOCAL/GROUP opts help.
- Predict **relative ranking** (pairwise/listwise) rather than absolute runtime to avoid
  cross-device calibration issues. This is what Ansor's TreeGRU cost model does.

## Code style

- **2-space indentation**, 150-character line limit (enforced by ruff).
- **No code golf.** The goal is lower *complexity*, not lower line count. Deleting newlines is not a win.
- New features should match the `torch` / `numpy` API where applicable.
- Bug fixes require a regression test.
- Speedups must be benchmarked; a marginal speedup that hurts readability won't be accepted.
- Keep PRs small. Large diffs are not reviewed.
- Code outside `tinygrad/` (the `extra/` directory) is not well tested and is lower priority.
