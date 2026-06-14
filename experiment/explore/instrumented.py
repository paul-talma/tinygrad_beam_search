"""Instrumented beam search that captures per-step metrics.

Usage:
  from experiment.explore.instrumented import install, uninstall, pop_traces, SearchTrace

  install()                 # monkeypatch once at process start
  ...run ops with BEAM=N... # each beam_search call appends to trace list
  traces = pop_traces()     # retrieve and clear collected traces
  uninstall()               # optional: restore original
"""
import math, time, multiprocessing, atexit
from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data structures (no tinygrad imports at module level to avoid circular deps)
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
  step: int
  candidates_generated: int
  candidates_compiled: int     # non-None returns from _try_compile
  candidates_unique: int       # after seen_libs dedup
  candidates_timed: int        # passed compute-budget filter and were timed
  runtimes_us: list[float] = field(default_factory=list)
  compile_times_s: list[float] = field(default_factory=list)
  best_time_us: float = math.inf
  best_opts: list = field(default_factory=list)  # list[Opt]

@dataclass
class SearchTrace:
  op_name: str
  steps: list[StepRecord] = field(default_factory=list)
  total_beam_time_s: float = 0.0
  baseline_time_us: float = math.inf   # heuristic-only kernel time
  best_time_us: float = math.inf       # best time found during search (may be estimated)
  true_time_us: float = math.inf       # winner re-timed at allow_test_size=False
  kernel_id: str = ""                  # AST key of the unoptimized kernel
  best_opts: list = field(default_factory=list)  # list[Opt]
  n_kernels_compiled: int = 0          # total across all steps
  n_kernels_timed: int = 0             # total across all steps
  error: str | None = None

# ---------------------------------------------------------------------------
# Module-level state (no tinygrad objects)
# ---------------------------------------------------------------------------

_pending_traces: list[SearchTrace] = []
_current_op_name: str = "unknown"
_candidate_filter: Callable | None = None  # (list[Scheduler]) -> list[Scheduler]
_override_allow_test_size: bool | None = None  # None = use caller's value

# Deferred references to tinygrad module objects — filled in by install()
_search_mod = None
_original_beam_search = None


def set_op_name(name: str) -> None:
  global _current_op_name
  _current_op_name = name

def set_candidate_filter(fn: "Callable | None") -> None:
  """Install a per-step candidate filter (or None to clear)."""
  global _candidate_filter
  _candidate_filter = fn

def set_timing_mode(allow_test_size: "bool | None") -> None:
  """Override allow_test_size for all subsequent beam searches.

  Pass False to force real-size timing (model-driven runs).
  Pass None to restore caller-controlled behaviour (default).
  """
  global _override_allow_test_size
  _override_allow_test_size = allow_test_size

def pop_traces() -> list[SearchTrace]:
  """Return and clear all traces collected since last call."""
  global _pending_traces
  out, _pending_traces = _pending_traces, []
  return out

# ---------------------------------------------------------------------------
# Instrumented beam search — mirrors search.py::beam_search exactly,
# adding one StepRecord per iteration and a baseline timing call.
# ---------------------------------------------------------------------------

def _instrumented_beam_search(s, rawbufs, amt, allow_test_size=True, disable_cache=None):
  import tinygrad.codegen.opt.search as _sm
  from tinygrad.codegen.opt.search import (
    _try_compile, _time_program, get_kernel_actions,
    _ensure_buffer_alloc, _init_worker,
  )
  from tinygrad.codegen import to_program
  from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
  from tinygrad.device import Device
  from tinygrad.helpers import flatten, getenv, CACHELEVEL, diskcache_put, IGNORE_BEAM_CACHE
  from tinygrad.uop.ops import sym_infer

  if disable_cache is None:
    disable_cache = IGNORE_BEAM_CACHE.value

  # Apply module-level timing override (set by harness for model-driven runs).
  if _override_allow_test_size is not None:
    allow_test_size = _override_allow_test_size

  global _pending_traces, _current_op_name, _candidate_filter

  trace = SearchTrace(op_name=_current_op_name)
  # Capture kernel_id from the unoptimized AST key (same formula as collect/hook.py).
  trace.kernel_id = s.ast.key.hex() if isinstance(s.ast.key, (bytes, bytearray)) else str(s.ast.key)
  _pending_traces.append(trace)

  # --- baseline: heuristic-only kernel timing ---
  try:
    hk = hand_coded_optimizations(s.copy())
    hk_ast = hk.get_optimized_ast(name_override="test")
    hk_prg = to_program(hk_ast, s.ren)
    var_vals_b: dict[str, int] = {k.expr: int(k.vmax + k.vmin) // 2 for k in s.ast.variables()}
    hk_rawbufs = [b.ensure_allocated() for b in rawbufs]
    tms_b = _time_program(hk_prg, var_vals_b, hk_rawbufs, allow_test_size=allow_test_size)
    trace.baseline_time_us = min(tms_b) * 1e6
  except Exception:
    trace.baseline_time_us = math.inf

  # --- replicated beam search with metric capture ---
  beam: list[tuple] = [(s, float("inf"))]
  seen_libs: set = set()

  # Spawned workers can't import tinygrad.codegen in this submodule setup due to a
  # circular import chain (codegen → spec → schedule → realize → codegen).  Default
  # to single-threaded compilation; set PARALLEL=N explicitly to override.
  default_parallel = 0
  if _sm.beam_pool is None and (workers := getenv("PARALLEL", default_parallel)):
    _sm.beam_pool = multiprocessing.get_context("spawn").Pool(
      workers, _init_worker, (), getenv("BEAM_MAX_TASKS_PER_CHILD", 16)
    )
    @atexit.register
    def _close_pool(): _sm.beam_pool.close()

  min_progress = getenv("BEAM_MIN_PROGRESS", 0.01) / 1e6

  try:
    rawbufs = _ensure_buffer_alloc(rawbufs)
    var_vals: dict[str, int] = {k.expr: int(k.vmax + k.vmin) // 2 for k in s.ast.variables()}
    dev = Device[s.ren.target.device]
    exiting, wall_st = False, time.perf_counter()
    step_idx = 0

    while not exiting:
      candidates = flatten([get_kernel_actions(si, include_0=False).values() for si, _ in beam])
      if _candidate_filter is not None:
        candidates = _candidate_filter(candidates)

      rec = StepRecord(
        step=step_idx,
        candidates_generated=len(candidates),
        candidates_compiled=0,
        candidates_unique=0,
        candidates_timed=0,
      )
      step_idx += 1

      timed: list[tuple] = []
      least_compute_ops = math.inf

      pool_map = _sm.beam_pool.imap_unordered if _sm.beam_pool is not None else map
      for i, proc in pool_map(_try_compile, enumerate(candidates)):
        if proc is None:
          continue
        prg, compile_et = proc
        rec.candidates_compiled += 1
        rec.compile_times_s.append(compile_et)

        lib = prg.src[4].arg
        if lib in seen_libs:
          continue
        rec.candidates_unique += 1

        estimates = prg.src[0].arg.estimates
        this_compute_ops = sym_infer(estimates.ops if estimates is not None else 0, var_vals)
        least_compute_ops = min(this_compute_ops, least_compute_ops)
        if least_compute_ops * 1000 < this_compute_ops:
          continue

        seen_libs.add(lib)
        try:
          tms = _time_program(
            prg, var_vals, rawbufs,
            early_stop=beam[0][1] * 3 if beam else 1.0,
            allow_test_size=allow_test_size,
            clear_l2=hasattr(dev, "invalidate_caches"),
            dev_timeout=getenv("BEAM_DEV_TIMEOUT", 1),
          )
        except Exception:
          continue

        best_tm = min(tms)
        rec.candidates_timed += 1
        rec.runtimes_us.append(best_tm * 1e6)
        timed.append((candidates[i], best_tm))

      # beam update
      opts = sorted(timed, key=lambda x: x[1])
      exiting = len(opts) == 0 or (opts[0][1] < min_progress) or (
        len(beam) > 0 and (beam[0][1] - opts[0][1]) < min_progress
      )
      if not exiting:
        beam = opts[:amt]
      elif len(opts) > 0 and opts[0][1] < beam[0][1]:
        beam = opts[:1]

      rec.best_time_us = beam[0][1] * 1e6 if beam[0][1] != float("inf") else math.inf
      rec.best_opts = list(beam[0][0].applied_opts)
      trace.steps.append(rec)

  except KeyboardInterrupt:
    if _sm.beam_pool is not None:
      _sm.beam_pool.terminate()
    raise

  trace.total_beam_time_s = time.perf_counter() - wall_st
  trace.best_time_us = beam[0][1] * 1e6 if beam[0][1] != float("inf") else math.inf
  trace.best_opts = list(beam[0][0].applied_opts)
  trace.n_kernels_compiled = sum(r.candidates_compiled for r in trace.steps)
  trace.n_kernels_timed = sum(r.candidates_timed for r in trace.steps)

  # Re-time the winning kernel at allow_test_size=False for a fair quality comparison.
  # This is a single extra compile+time call after the search completes.
  if beam[0][1] != float("inf"):
    try:
      best_ast = beam[0][0].get_optimized_ast(name_override="test")
      best_prg = to_program(best_ast, s.ren)
      tms_true = _time_program(
        best_prg, var_vals, rawbufs,
        allow_test_size=False,
        clear_l2=hasattr(dev, "invalidate_caches"),
        dev_timeout=getenv("BEAM_DEV_TIMEOUT", 1),
      )
      trace.true_time_us = min(tms_true) * 1e6
    except Exception:
      trace.true_time_us = trace.best_time_us  # fallback: use search-time estimate

  if CACHELEVEL >= 1 and not disable_cache:
    key = {"ast": s.ast.key, "amt": amt, "allow_test_size": allow_test_size,
           "device": s.ren.target.device, "suffix": s.ren.suffix}
    diskcache_put("beam_search", key, beam[0][0].applied_opts)

  return beam[0][0]


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------

def install() -> None:
  """Patch tinygrad's beam_search with the instrumented version."""
  global _search_mod, _original_beam_search
  import tinygrad.codegen.opt.search as sm
  import tinygrad.codegen.opt.postrange as pr
  _search_mod = sm
  if _original_beam_search is None:
    _original_beam_search = sm.beam_search
  sm.beam_search = _instrumented_beam_search
  pr.beam_search = _instrumented_beam_search  # patch postrange's reference too

def uninstall() -> None:
  """Restore the original beam_search."""
  global _search_mod, _original_beam_search
  if _search_mod is not None and _original_beam_search is not None:
    _search_mod.beam_search = _original_beam_search
    import tinygrad.codegen.opt.postrange as pr
    pr.beam_search = _original_beam_search
