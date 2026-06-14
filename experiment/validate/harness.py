"""Validation harness: compare baseline beam search vs cost-model-guided beam search.

A cost model is any callable: Scheduler -> float (lower = predicted faster).
The model is used to rank candidates before compilation; only the top-K are
actually compiled and timed (where K = ceil(beam_width * prune_factor)).

Timing modes
------------
Baseline  : allow_test_size=True  (vanilla tinygrad default; scales down global_size
            to stay under 65536, then extrapolates — same as production behaviour)
Model run : allow_test_size=False (full-size timing; affordable because the model
            already prunes the candidate set before compilation)
True time : after each search (both modes), the winning kernel is re-timed at
            allow_test_size=False so the quality comparison is apples-to-apples

Seen / unseen
-------------
Pass --training-data to specify a JSONL file.  kernel_ids found there are tagged
is_trained=True.  Quality metrics in the summary table are split by seen/unseen so
model performance on held-out kernels can be assessed without contamination.

Usage:
  uv run python -m experiment.validate.harness [--beam N] [--stub] [--ops op1,op2]
  uv run python -m experiment.validate.harness --training-data data/v2/combined.jsonl

  --stub  runs with a random-score "model" to verify the harness plumbing
          (random pruning should show worse quality than baseline).

Plugging in a real model:
  from experiment.validate.harness import run_with_model
  results = run_with_model(ops_dict, cost_model=my_model, beam_width=5)
"""
import argparse, json, math, time, sys
from collections.abc import Callable
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
  op_name: str
  mode: str                          # "baseline" or "model:<name>"
  total_wall_s: float = 0.0          # wall time including all compilation
  beam_wall_s: float = 0.0           # wall time spent inside beam search only
  n_compiled: int = 0
  n_timed: int = 0
  kernel_time_us: float = math.inf       # best kernel time found during search
                                         # (may be estimated if allow_test_size=True)
  kernel_time_true_us: float = math.inf  # winner re-timed at allow_test_size=False
  heuristic_time_us: float = math.inf    # hand_coded_optimizations only, no beam search
  kernel_id: str = ""                    # AST key of the unoptimized kernel
  is_trained: bool = False               # True if kernel_id was in training data
  best_opts: list = field(default_factory=list)
  error: str | None = None


# ---------------------------------------------------------------------------
# Training-data helpers
# ---------------------------------------------------------------------------

def load_trained_kernel_ids(data_path: str) -> set[str]:
  """Return the set of kernel_ids present in a training JSONL file."""
  ids: set[str] = set()
  try:
    with open(data_path) as f:
      for line in f:
        if line.strip():
          ids.add(json.loads(line)["kernel_id"])
  except FileNotFoundError:
    print(f"[warn] training-data file not found: {data_path}", file=sys.stderr)
  return ids


def tag_seen_unseen(results: list[RunResult], trained_ids: set[str]) -> None:
  """Set is_trained on each result based on whether its kernel_id is in trained_ids."""
  for r in results:
    r.is_trained = r.kernel_id in trained_ids


# ---------------------------------------------------------------------------
# Shared inner runner
# ---------------------------------------------------------------------------

def _run_op(name: str, op_fn: Callable, mode: str, beam_width: int) -> RunResult:
  """Run a single op inside a BEAM context and return a RunResult.
  Assumes instrumented beam_search is already installed and
  _candidate_filter / _override_allow_test_size are already configured.
  """
  from tinygrad.helpers import Context, IGNORE_BEAM_CACHE
  from experiment.explore.instrumented import pop_traces, set_op_name

  set_op_name(name)
  result = RunResult(op_name=name, mode=mode)

  # Clear the in-process program cache so beam search runs fresh each time.
  import tinygrad.codegen as _codegen_mod
  _codegen_mod.to_program_cache.clear()

  t_total = time.perf_counter()
  try:
    with Context(BEAM=beam_width, IGNORE_BEAM_CACHE=1):
      op_fn().realize()
    traces = pop_traces()
    result.total_wall_s = time.perf_counter() - t_total
    if traces:
      tr = traces[0]
      result.beam_wall_s         = tr.total_beam_time_s
      result.n_compiled          = tr.n_kernels_compiled
      result.n_timed             = tr.n_kernels_timed
      result.kernel_time_us      = tr.best_time_us
      result.kernel_time_true_us = tr.true_time_us
      result.heuristic_time_us   = tr.baseline_time_us
      result.kernel_id           = tr.kernel_id
      result.best_opts           = tr.best_opts
  except Exception as e:
    pop_traces()
    result.error = str(e)
    result.total_wall_s = time.perf_counter() - t_total
  return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_baseline(
    ops: dict[str, Callable],
    beam_width: int = 5,
) -> list[RunResult]:
  """Run standard beam search (allow_test_size=True) on each op."""
  from experiment.explore.instrumented import install, set_candidate_filter

  install()
  set_candidate_filter(None)
  results = []
  for name, op_fn in ops.items():
    print(f"  [baseline] {name} ...", end=" ", flush=True)
    r = _run_op(name, op_fn, "baseline", beam_width)
    if r.error:
      print(f"ERROR: {r.error}")
    else:
      print(f"{r.kernel_time_us:.1f} µs (est)  {r.kernel_time_true_us:.1f} µs (true)"
            f"  ({r.beam_wall_s:.2f}s beam)")
    results.append(r)
  return results


def run_with_model(
    ops: dict[str, Callable],
    cost_model: "Callable",
    model_name: str = "model",
    beam_width: int = 5,
    prune_factor: int = 4,
    allow_test_size: bool = False,
) -> list[RunResult]:
  """Run beam search with candidate pruning via cost_model.

  allow_test_size=False (default): real-size timing; affordable because the model
    prunes candidates before compilation.
  allow_test_size=True: same estimated timing as baseline; useful for comparison.

  cost_model(scheduler) -> float: predicted runtime (lower is better).
  prune_factor: keep top ceil(beam_width * prune_factor) candidates before compilation.
  """
  from experiment.explore.instrumented import install, set_candidate_filter, set_timing_mode

  install()
  keep_k = math.ceil(beam_width * prune_factor)
  mode_str = f"model:{model_name}"

  def _filter(candidates: list) -> list:
    scored = [(cost_model(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:keep_k]]

  timing_suffix = "est" if allow_test_size else "full"
  mode_str = f"{mode_str}:{timing_suffix}"
  set_candidate_filter(_filter)
  set_timing_mode(None if allow_test_size else False)
  try:
    results = []
    for name, op_fn in ops.items():
      print(f"  [{mode_str}] {name} ...", end=" ", flush=True)
      r = _run_op(name, op_fn, mode_str, beam_width)
      if r.error:
        print(f"ERROR: {r.error}")
      else:
        print(f"{r.kernel_time_us:.1f} µs ({timing_suffix})  ({r.beam_wall_s:.2f}s beam)")
      results.append(r)
  finally:
    set_candidate_filter(None)
    set_timing_mode(None)
  return results


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _row_to_result(row: dict) -> RunResult:
  r = RunResult(op_name=row["op_name"], mode=row.get("mode", ""))
  r.total_wall_s        = float(row.get("total_wall_s") or 0)
  r.beam_wall_s         = float(row.get("beam_wall_s") or 0)
  r.n_compiled          = int(row.get("n_compiled") or 0)
  r.n_timed             = int(row.get("n_timed") or 0)
  r.kernel_time_us      = float(row.get("kernel_time_us") or math.inf)
  r.kernel_time_true_us = float(row.get("kernel_time_true_us") or math.inf)
  r.heuristic_time_us   = float(row.get("heuristic_time_us") or math.inf)
  r.kernel_id           = row.get("kernel_id", "")
  r.is_trained          = row.get("is_trained", "False") == "True"
  r.error               = row.get("error") or None
  return r


def load_baseline_from_csv(path: str) -> list[RunResult]:
  """Reconstruct baseline RunResult objects from a previously saved CSV."""
  import csv as _csv
  results = []
  with open(path, newline="") as f:
    for row in _csv.DictReader(f):
      if row.get("mode") == "baseline":
        results.append(_row_to_result(row))
  return results


def load_all_from_csv(path: str) -> list[RunResult]:
  """Reconstruct all RunResult objects from a previously saved CSV."""
  import csv as _csv
  results = []
  with open(path, newline="") as f:
    for row in _csv.DictReader(f):
      results.append(_row_to_result(row))
  return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
  from experiment.explore.ops import OPS
  from experiment.validate.display import print_comparison, save_csv

  parser = argparse.ArgumentParser(description="Beam search validation harness")
  parser.add_argument("--beam",            type=int, default=5,    help="Beam width (default: 5)")
  parser.add_argument("--prune",           type=int, default=4,    help="Prune factor for model run (default: 4)")
  parser.add_argument("--stub",            action="store_true",    help="Use random cost model stub")
  parser.add_argument("--model",           type=str, default=None, help="Path to .lgb checkpoint (default: auto-discover latest)")
  parser.add_argument("--ops",             type=str, default="",   help="Comma-separated op names (default: all)")
  parser.add_argument("--training-data",   type=str, default="data/v2/combined.jsonl",
                      help="JSONL file used for training; kernel_ids in it are tagged is_trained=True")
  parser.add_argument("--baseline-csv",    type=str, default=None,
                      help="Load baseline results from a previously saved CSV instead of re-running")
  parser.add_argument("--model-test-size", action="store_true",
                      help="Use allow_test_size=True for model-guided search (estimated timing, matches baseline mode)")
  args = parser.parse_args()

  ops_to_run: dict[str, Callable] = {}
  names = [s.strip() for s in args.ops.split(",") if s.strip()] if args.ops else list(OPS.keys())
  for n in names:
    if n in OPS:
      ops_to_run[n] = OPS[n]
    else:
      print(f"[warn] unknown op '{n}', skipping", file=sys.stderr)

  if not ops_to_run:
    print("No ops to run.", file=sys.stderr)
    sys.exit(1)

  if args.baseline_csv:
    print(f"\n{'='*60}")
    print(f"Baseline results loaded from {args.baseline_csv}")
    print(f"{'='*60}")
    baseline = load_baseline_from_csv(args.baseline_csv)
    # Filter to only the ops we're running.
    op_set = set(ops_to_run)
    baseline = [r for r in baseline if r.op_name in op_set]
    for r in baseline:
      tag = "est" if math.isinf(r.kernel_time_true_us) else "true"
      print(f"  [baseline] {r.op_name} ... {r.kernel_time_true_us:.1f} µs ({tag})  ({r.beam_wall_s:.2f}s beam)")
  else:
    print(f"\n{'='*60}")
    print(f"Baseline beam search (beam_width={args.beam}, allow_test_size=True)")
    print(f"{'='*60}")
    baseline = run_baseline(ops_to_run, beam_width=args.beam)

  import random
  cost_model: Callable = lambda _s: random.random()
  model_name = "random_stub"
  if args.stub:
    pass  # use random stub as-is
  else:
    try:
      from model.predict import load_model
      _cm = load_model(args.model)
      cost_model = _cm
      model_name = f"lgbm:{_cm._path.split('/')[-1]}"
    except FileNotFoundError:
      print("\n[warn] No cost model checkpoint found; using random stub.", file=sys.stderr)

  allow_test_size = args.model_test_size
  print(f"\n{'='*60}")
  print(f"Model beam search (model={model_name}, beam_width={args.beam}, "
        f"prune_factor={args.prune}, allow_test_size={allow_test_size})")
  print(f"{'='*60}")
  model_results = run_with_model(ops_to_run, cost_model, model_name=model_name,
                                  beam_width=args.beam, prune_factor=args.prune,
                                  allow_test_size=allow_test_size)

  # Tag seen / unseen using training data kernel_ids.
  trained_ids = load_trained_kernel_ids(args.training_data)
  tag_seen_unseen(baseline + model_results, trained_ids)

  print_comparison(baseline, model_results, beam_width=args.beam,
                   model_name=model_name, prune_factor=args.prune)

  import datetime, pathlib
  ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = pathlib.Path(__file__).parents[2] / "experiment" / "results"
  out_dir.mkdir(parents=True, exist_ok=True)
  csv_path = out_dir / f"validation_{ts}.csv"
  # When --baseline-csv is set, carry ALL rows from the source CSV forward so
  # the output is self-contained and report.py can read everything from one file.
  rows_to_save = baseline + model_results
  if args.baseline_csv:
    existing = load_all_from_csv(args.baseline_csv)
    op_set = set(ops_to_run)
    existing = [r for r in existing if r.op_name in op_set]
    rows_to_save = existing + model_results

  save_csv(rows_to_save, str(csv_path))
  print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
  main()
