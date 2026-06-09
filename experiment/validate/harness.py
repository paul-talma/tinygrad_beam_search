"""Validation harness: compare baseline beam search vs cost-model-guided beam search.

A cost model is any callable: Scheduler -> float (lower = predicted faster).
The model is used to rank candidates before compilation; only the top-K are
actually compiled and timed (where K = ceil(beam_width * prune_factor)).

Usage:
  uv run python -m experiment.validate.harness [--beam N] [--stub] [--ops op1,op2]

  --stub  runs with a random-score "model" to verify the harness plumbing
          (random pruning should show worse quality than baseline).

Plugging in a real model:
  from experiment.validate.harness import run_with_model
  results = run_with_model(ops_dict, cost_model=my_model, beam_width=5)
"""
import argparse, math, time, sys
from collections.abc import Callable
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result dataclass (no tinygrad imports at module level — circular deps)
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
  op_name: str
  mode: str                        # "baseline" or "model:<name>"
  total_wall_s: float = 0.0        # wall time including all compilation
  beam_wall_s: float = 0.0         # wall time spent inside beam search only
  n_compiled: int = 0
  n_timed: int = 0
  kernel_time_us: float = math.inf # best kernel execution time found by beam
  best_opts: list = field(default_factory=list)  # list[Opt]
  error: str | None = None


# ---------------------------------------------------------------------------
# Shared inner runner
# ---------------------------------------------------------------------------

def _run_op(name: str, op_fn: Callable, mode: str, beam_width: int) -> RunResult:
  """Run a single op inside a BEAM context and return a RunResult.
  Assumes instrumented beam_search is already installed and
  _candidate_filter is already set.
  """
  from tinygrad.helpers import Context, IGNORE_BEAM_CACHE
  from experiment.explore.instrumented import pop_traces, set_op_name

  set_op_name(name)
  result = RunResult(op_name=name, mode=mode)
  t_total = time.perf_counter()
  try:
    with Context(BEAM=beam_width, IGNORE_BEAM_CACHE=1):
      op_fn().realize()
    traces = pop_traces()
    result.total_wall_s = time.perf_counter() - t_total
    if traces:
      tr = traces[0]
      result.beam_wall_s    = tr.total_beam_time_s
      result.n_compiled     = tr.n_kernels_compiled
      result.n_timed        = tr.n_kernels_timed
      result.kernel_time_us = tr.best_time_us
      result.best_opts      = tr.best_opts
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
  """Run standard beam search on each op and collect timing metrics."""
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
      print(f"{r.kernel_time_us:.1f} µs  ({r.beam_wall_s:.2f}s beam)")
    results.append(r)
  return results


def run_with_model(
    ops: dict[str, Callable],
    cost_model: "Callable",
    model_name: str = "model",
    beam_width: int = 5,
    prune_factor: int = 4,
) -> list[RunResult]:
  """Run beam search with candidate pruning via cost_model.

  cost_model(scheduler) -> float: predicted runtime (lower is better).
  prune_factor: keep top ceil(beam_width * prune_factor) candidates
  before compilation.  E.g. beam_width=5, prune_factor=4 → keep top 20.
  """
  from experiment.explore.instrumented import install, set_candidate_filter

  install()
  keep_k = math.ceil(beam_width * prune_factor)
  mode_str = f"model:{model_name}"

  def _filter(candidates: list) -> list:
    scored = [(cost_model(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:keep_k]]

  set_candidate_filter(_filter)
  results = []
  for name, op_fn in ops.items():
    print(f"  [{mode_str}] {name} ...", end=" ", flush=True)
    r = _run_op(name, op_fn, mode_str, beam_width)
    if r.error:
      print(f"ERROR: {r.error}")
    else:
      print(f"{r.kernel_time_us:.1f} µs  ({r.beam_wall_s:.2f}s beam)")
    results.append(r)
  set_candidate_filter(None)
  return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
  from experiment.explore.ops import OPS
  from experiment.validate.display import print_comparison, save_csv

  parser = argparse.ArgumentParser(description="Beam search validation harness")
  parser.add_argument("--beam",   type=int, default=5,   help="Beam width (default: 5)")
  parser.add_argument("--prune",  type=int, default=4,   help="Prune factor for model run (default: 4)")
  parser.add_argument("--stub",   action="store_true",   help="Use random cost model stub")
  parser.add_argument("--model",  type=str, default=None, help="Path to .lgb checkpoint (default: auto-discover latest)")
  parser.add_argument("--ops",    type=str, default="",  help="Comma-separated op names (default: all)")
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

  print(f"\n{'='*60}")
  print(f"Baseline beam search (beam_width={args.beam})")
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

  print(f"\n{'='*60}")
  print(f"Model beam search (model={model_name}, beam_width={args.beam}, prune_factor={args.prune})")
  print(f"{'='*60}")
  model_results = run_with_model(ops_to_run, cost_model, model_name=model_name,
                                  beam_width=args.beam, prune_factor=args.prune)

  print_comparison(baseline, model_results, beam_width=args.beam, model_name=model_name, prune_factor=args.prune)

  import datetime, pathlib
  ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = pathlib.Path(__file__).parents[2] / "experiment" / "results"
  out_dir.mkdir(parents=True, exist_ok=True)
  csv_path = out_dir / f"validation_{ts}.csv"
  save_csv(baseline + model_results, str(csv_path))
  print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
  main()
