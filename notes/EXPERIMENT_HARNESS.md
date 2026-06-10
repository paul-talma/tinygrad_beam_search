# Experiment Harness

Two harnesses in `experiment/` for studying tinygrad beam search behavior — without touching any production code.

## Directory layout

```
experiment/
  explore/
    ops.py            # registry of ~12 representative workloads
    instrumented.py   # monkeypatched beam_search with per-step metric capture
    run.py            # CLI runner; prints step tables and summary
  validate/
    harness.py        # compare baseline beam vs cost-model-guided beam
    display.py        # table formatting + CSV output
  results/            # CSV output from validation runs
```

---

## How instrumentation works without modifying tinygrad

The hook point is `apply_opts()` in `tinygrad/codegen/opt/postrange.py`. It does a **deferred import** inside the function body rather than at module level:

```python
def apply_opts(ast, ren, beam):
    ...
    from tinygrad.codegen.opt.search import beam_search   # re-imported on each call
    return beam_search(s, rawbufs, beam)
```

Because Python module imports are cached in `sys.modules`, this resolves to whatever `tinygrad.codegen.opt.search.beam_search` currently points to. `install()` replaces that attribute:

```python
import tinygrad.codegen.opt.search as sm
sm.beam_search = _instrumented_beam_search
```

After that, every call to `apply_opts()` picks up the instrumented version. `uninstall()` restores the original. If `apply_opts` had captured `beam_search` at module load time, this approach would not work.

---

## `experiment/explore/instrumented.py`

The core module. Contains the instrumented beam search function and module-level state.

### Data structures

```python
@dataclass
class StepRecord:
    step: int
    candidates_generated: int   # len(get_kernel_actions(...).values())
    candidates_compiled: int    # non-None returns from _try_compile
    candidates_unique: int      # after seen_libs dedup
    candidates_timed: int       # survived compute-budget filter
    runtimes_us: list[float]    # best time per timed candidate (µs)
    compile_times_s: list[float]
    best_time_us: float
    best_opts: list              # list[Opt], Scheduler.applied_opts of current beam leader

@dataclass
class SearchTrace:
    op_name: str
    steps: list[StepRecord]
    total_beam_time_s: float
    baseline_time_us: float     # heuristic kernel time (hand_coded_optimizations)
    best_time_us: float
    best_opts: list
    n_kernels_compiled: int
    n_kernels_timed: int
    error: str | None
```

### Module-level state

Three globals drive behavior at runtime. All are plain Python — no tinygrad objects at module level (required to avoid circular import chains):

| Variable | Type | Purpose |
|---|---|---|
| `_pending_traces` | `list[SearchTrace]` | Accumulates traces until `pop_traces()` is called |
| `_current_op_name` | `str` | Tag written into each new `SearchTrace.op_name` |
| `_candidate_filter` | `Callable \| None` | Optional per-step pruner; receives `list[Scheduler]`, returns filtered list |

### Public API

```python
install()                        # monkeypatch; safe to call multiple times
uninstall()                      # restore original beam_search
set_op_name(name: str)           # set tag for the next trace
set_candidate_filter(fn | None)  # wire in a cost model filter (or clear it)
pop_traces() -> list[SearchTrace] # retrieve and clear pending traces
```

### `_instrumented_beam_search`

Mirrors `search.py::beam_search` exactly, adding metric capture. Key differences:

1. **Baseline timing** — before the beam loop, runs `hand_coded_optimizations(s.copy())` once to get the heuristic kernel time. Stored in `trace.baseline_time_us`.

2. **`_candidate_filter` hook** — applied to the candidate list immediately after `get_kernel_actions`, before any compilation:
   ```python
   candidates = flatten([get_kernel_actions(si, ...).values() for si, _ in beam])
   if _candidate_filter is not None:
       candidates = _candidate_filter(candidates)
   ```

3. **`StepRecord` capture** — counters are incremented inside the compile/time loops at the same points where the original increments its local tracking variables.

4. **Pool initialization** — replicates the `_sm.beam_pool` lazy-init and atexit cleanup from the original. Required because the pool is module-level state on `search`.

All tinygrad imports are inside the function body to avoid the circular import chain `codegen.__init__` → `uop.spec` → `schedule.__init__` → `engine.realize` → `codegen.__init__`.

---

## `experiment/explore/ops.py`

Registry of workloads. Each entry is a zero-argument callable returning a `Tensor` with pending computation; `.realize()` triggers beam search.

Inputs are **pre-realized** so the lambda always produces the same kernel AST (enabling beam cache hits across repeated calls in the same process):

```python
@_reg("matmul_1024")
def _():
    a = Tensor.randn(1024, 1024, dtype=_f16).realize()
    b = Tensor.randn(1024, 1024, dtype=_f16).realize()
    return lambda: a @ b
```

Current workloads: `matmul_256`, `matmul_1024`, `matmul_2048`, `matmul_4096`, `matmul_8192`, `matmul_rect` (512×1024 × 1024×2048), `conv_3x3`, `conv_5x5`, `elem_relu`, `elem_fused` (add+relu), `reduce_sum`, `attention` (QKᵀV, 2-head 128-seq).

**Import note:** `from tinygrad import Tensor` fails at module level due to a namespace package conflict — the tinygrad submodule root wins `sys.meta_path` order over the editable install. Use `from tinygrad.tensor import Tensor` and `from tinygrad.dtype import dtypes` instead. This applies everywhere in the `experiment/` tree.

---

## `experiment/explore/run.py`

CLI runner.

```sh
uv run python -m experiment.explore.run [--beam N] [--ops op1,op2,...]
```

For each op: calls `install()`, tags the op name, runs `.realize()` under `Context(BEAM=N, IGNORE_BEAM_CACHE=1)`, calls `pop_traces()`, and prints a per-step table plus a summary. `IGNORE_BEAM_CACHE=1` forces fresh beam search on every run.

The `attention` op decomposes into 4 sub-kernels, so `pop_traces()` returns 4 `SearchTrace` objects for it. The step table is printed for each.

---

## `experiment/validate/harness.py`

Validation harness for comparing baseline beam search against cost-model-guided beam search.

### `RunResult` dataclass

```python
@dataclass
class RunResult:
    op_name: str
    mode: str               # "baseline" or "model:<name>"
    total_wall_s: float     # wall time including compilation
    beam_wall_s: float      # wall time inside beam search
    n_compiled: int
    n_timed: int
    kernel_time_us: float   # best kernel execution time found
    best_opts: list
    error: str | None
```

### `run_baseline(ops, beam_width) -> list[RunResult]`

Calls `install()` and `set_candidate_filter(None)`, then runs each op under `Context(BEAM=..., IGNORE_BEAM_CACHE=1)`. Collects timing from the first trace returned by `pop_traces()`.

### `run_with_model(ops, cost_model, model_name, beam_width, prune_factor) -> list[RunResult]`

Same as `run_baseline` but installs a `_candidate_filter` that:
1. Scores each candidate `Scheduler` with `cost_model(scheduler)` (no compilation).
2. Keeps top `ceil(beam_width * prune_factor)` by score.
3. Passes only those to the beam loop for compilation + timing.

**Cost model interface:** `Callable[[Scheduler], float]` — lower score = predicted faster.

### CLI

```sh
uv run python -m experiment.validate.harness [--beam N] [--prune K] [--stub] [--ops ...]
```

`--stub` uses a random-score model to verify harness plumbing (random pruning should degrade quality).
Without `--stub`, the CLI auto-discovers the latest `.lgb` checkpoint from `model/` via `model.predict.load_model`.

---

## `experiment/validate/display.py`

`print_comparison(baseline, model_results, ...)` — side-by-side table with columns:

- B-Beam s / M-Beam s / Search Δ — beam search wall time and relative change
- B-Total s / M-Total s — total wall time including compilation
- B-µs / M-µs / Quality Δ — best kernel execution time and relative change
- B-compiled / M-compiled — total kernels compiled

`save_csv(results, path)` — writes all `RunResult` fields to CSV. Output goes to `experiment/results/validation_YYYYMMDD_HHMMSS.csv`.

---

## Caveats

- **`allow_test_size` scaling** — for large kernels (matmul_8192), beam search scales down `global_size` to stay under 65536 for timing. This inflates per-step `Max µs` dramatically (step 1 of matmul_8192 shows ~69s max). The `best_time_us` from the final chosen kernel is still accurate.
- **Multi-kernel ops** — `attention` produces 4 traces. `harness.py` uses only `traces[0]` for `RunResult`, so it captures the first sub-kernel only.
- **Baseline in explore vs validate** — in `explore`, the baseline is re-timed via `hand_coded_optimizations` inside `_instrumented_beam_search`. In `validate`, both baseline and model runs go through the same instrumented path, so comparison is fair.
