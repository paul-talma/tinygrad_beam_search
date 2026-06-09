#!/usr/bin/env python3
"""Beam search benchmark orchestrator.

For each (op, beam_width) pair this runs two subprocesses:
  1. compile  -- with IGNORE_BEAM_CACHE=1 so beam search always re-runs.
                 Times beam search + kernel compilation + one execution.
                 Also warms the disk cache for the exec phase.
  2. exec     -- without IGNORE_BEAM_CACHE so the disk-cached opts are used
                 and no search runs.  Times pure kernel execution via
                 wall-clock bracketed by device synchronize.

Results are saved to results/beam_results.json; four plots are saved to plots/.

Usage:
  # Full run:
  uv run python benchmarks/beam_search/benchmark.py

  # Run one named suite:
  uv run python benchmarks/beam_search/benchmark.py --suite vanilla_attentions

  # Subset:
  uv run python benchmarks/beam_search/benchmark.py \\
      --ops matmul_512 conv_medium attn_256 --beams 0 1 2 4 8

  # Re-plot from saved results:
  uv run python benchmarks/beam_search/benchmark.py --load

Plots:
  compile_time.png  -- beam width vs search+compile wall time per op
  exec_time.png     -- beam width vs kernel execution time (ms) per op
  speedup.png       -- beam width vs execution speedup relative to beam=0
  cost_benefit.png  -- scatter: search cost vs kernel quality, annotated by beam
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

from ops import OP_NAMES, get_suite_names, get_suite_ops

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
PLOTS_DIR   = SCRIPT_DIR / "plots"

DEFAULT_SUITE    = "default"
DEFAULT_BEAMS    = [0, 1, 2, 4, 8]
DEFAULT_N_EXEC   = 10
DEFAULT_N_WARMUP = 2
SUBPROCESS_TIMEOUT = 600  # seconds per subprocess


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _require_tqdm():
  try:
    from tqdm import tqdm
  except ImportError:
    print("tqdm not found. Install with: uv sync --extra benchmarking", file=sys.stderr)
    sys.exit(1)
  return tqdm


def _require_matplotlib():
  global plt, _CMAP
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("matplotlib not found. Install with: uv sync --extra benchmarking", file=sys.stderr)
    sys.exit(1)
  _CMAP = plt.cm.tab10

def _select_ops(suite: str, ops: list[str] | None) -> list[str]:
  selected = ops if ops is not None else get_suite_ops(suite)
  unknown = [op for op in selected if op not in OP_NAMES]
  if unknown:
    raise ValueError(f"unknown op(s): {unknown}. Available ops: {list(OP_NAMES)}")
  return list(selected)


def _filter_results(results: dict, ops: list[str]) -> dict:
  filtered = {op: results[op] for op in ops if op in results}
  missing = [op for op in ops if op not in results]
  if missing:
    print(f"Skipping ops not present in loaded results: {missing}", file=sys.stderr)
  return filtered

def _base_env() -> dict[str, str]:
  """Environment without IGNORE_BEAM_CACHE (for exec phase)."""
  return {k: v for k, v in os.environ.items() if k != "IGNORE_BEAM_CACHE"}


def _run(mode: str, op: str, beam: int, n_exec: int, n_warmup: int,
         extra_env: dict[str, str] | None = None) -> dict | None:
  env = {**_base_env(), **(extra_env or {})}
  cmd = [sys.executable, str(SCRIPT_DIR / "run_single.py"),
         "--op", op, "--beam", str(beam), "--mode", mode,
         "--n-exec", str(n_exec), "--n-warmup", str(n_warmup)]
  try:
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=SUBPROCESS_TIMEOUT)
  except subprocess.TimeoutExpired:
    return None
  if proc.returncode != 0:
    return None
  try:
    return json.loads(proc.stdout)
  except json.JSONDecodeError:
    return None


def collect_results(ops: list[str], beams: list[int],
                    n_exec: int, n_warmup: int) -> dict:
  tqdm = _require_tqdm()
  RESULTS_DIR.mkdir(exist_ok=True)
  results: dict[str, dict[int, dict]] = {}

  total = len(ops) * len(beams)
  bar = tqdm(total=total, unit="run", ncols=80)

  for op in ops:
    results[op] = {}
    for beam in beams:
      bar.set_description(f"{op}  beam={beam}")

      # Phase 1: compile + beam search (cache-busted)
      cr = _run("compile", op, beam, n_exec, n_warmup,
                extra_env={"IGNORE_BEAM_CACHE": "1"})
      if cr is None:
        bar.write(f"  ✗ {op} beam={beam}: compile FAILED")
        bar.update(1)
        continue
      compile_time = cr["compile_time_s"]

      # Phase 2: pure execution (uses disk cache from phase 1)
      er = _run("exec", op, beam, n_exec, n_warmup)
      if er is None:
        bar.write(f"  ✗ {op} beam={beam}: exec FAILED  (compile={compile_time:.2f}s)")
        results[op][beam] = {**cr, "exec_time_s": None, "exec_time_min_s": None}
        bar.update(1)
        continue

      results[op][beam] = {
        "op":              op,
        "beam":            beam,
        "compile_time_s":  compile_time,
        "exec_time_s":     er["exec_time_s"],
        "exec_time_min_s": er["exec_time_min_s"],
        "n_exec":          er["n_exec"],
      }
      et_ms = er["exec_time_s"] * 1e3
      bar.write(f"  ✓ {op:15s} beam={beam}  compile={compile_time:6.2f}s  exec={et_ms:.3f}ms")
      bar.update(1)

  bar.close()

  out = RESULTS_DIR / "beam_results.json"
  with open(out, "w") as f:
    json.dump(results, f, indent=2)
  print(f"\nResults saved to {out}")
  return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+"]
_CMAP = None
plt = None


def _iter_ops(results: dict, beams: list[int], need_beam: int | None = None):
  for i, op in enumerate(results):
    op_data = results[op]
    valid   = [b for b in beams if b in op_data]
    if need_beam is not None and need_beam not in op_data:
      continue
    if valid:
      yield op, valid, op_data, i


def _save(fig, path: Path) -> None:
  PLOTS_DIR.mkdir(exist_ok=True)
  fig.savefig(path, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved {path}")


def plot_compile_time(results: dict, beams: list[int]) -> None:
  fig, ax = plt.subplots(figsize=(10, 6))
  for op, valid_beams, op_data, i in _iter_ops(results, beams):
    times = [op_data[b]["compile_time_s"] for b in valid_beams]
    ax.plot(valid_beams, times, marker=_MARKERS[i % len(_MARKERS)],
            color=_CMAP(i / 10), label=op)
  ax.set_xlabel("Beam Width")
  ax.set_ylabel("Search + Compile Time (s)")
  ax.set_title("Beam Width vs Optimization Time")
  ax.set_xticks(beams)
  ax.legend(loc="upper left", fontsize=8)
  ax.grid(True, alpha=0.3)
  _save(fig, PLOTS_DIR / "compile_time.png")


def plot_exec_time(results: dict, beams: list[int]) -> None:
  fig, ax = plt.subplots(figsize=(10, 6))
  for op, valid_beams, op_data, i in _iter_ops(results, beams):
    valid_b = [b for b in valid_beams if op_data[b].get("exec_time_s") is not None]
    times   = [op_data[b]["exec_time_s"] * 1e3 for b in valid_b]
    if times:
      ax.plot(valid_b, times, marker=_MARKERS[i % len(_MARKERS)],
              color=_CMAP(i / 10), label=op)
  ax.set_xlabel("Beam Width")
  ax.set_ylabel("Kernel Execution Time (ms, lower is better)")
  ax.set_title("Beam Width vs Kernel Quality")
  ax.set_xticks(beams)
  ax.legend(loc="upper right", fontsize=8)
  ax.grid(True, alpha=0.3)
  _save(fig, PLOTS_DIR / "exec_time.png")


def plot_speedup(results: dict, beams: list[int]) -> None:
  non_zero = [b for b in beams if b > 0]
  fig, ax  = plt.subplots(figsize=(10, 6))
  for op, valid_beams, op_data, i in _iter_ops(results, beams, need_beam=0):
    baseline = op_data[0].get("exec_time_s")
    if not baseline:
      continue
    valid_b  = [b for b in non_zero if b in op_data and op_data[b].get("exec_time_s")]
    speedups = [baseline / op_data[b]["exec_time_s"] for b in valid_b]
    if speedups:
      ax.plot(valid_b, speedups, marker=_MARKERS[i % len(_MARKERS)],
              color=_CMAP(i / 10), label=op)
  ax.axhline(1.0, linestyle="--", color="gray", alpha=0.6, label="beam=0 baseline")
  ax.set_xlabel("Beam Width")
  ax.set_ylabel("Speedup vs beam=0 (higher is better)")
  ax.set_title("Beam Width vs Optimization Gain")
  ax.set_xticks(non_zero)
  ax.legend(loc="lower right", fontsize=8)
  ax.grid(True, alpha=0.3)
  _save(fig, PLOTS_DIR / "speedup.png")


def plot_cost_benefit(results: dict, beams: list[int]) -> None:
  fig, ax = plt.subplots(figsize=(10, 6))
  for op, valid_beams, op_data, i in _iter_ops(results, beams):
    first = True
    for b in valid_beams:
      d = op_data[b]
      if d.get("exec_time_s") is None:
        continue
      ax.scatter(d["compile_time_s"], d["exec_time_s"] * 1e3,
                 color=_CMAP(i / 10), marker=_MARKERS[i % len(_MARKERS)],
                 s=60 + b * 15, label=op if first else None, zorder=3)
      ax.annotate(str(b), (d["compile_time_s"], d["exec_time_s"] * 1e3),
                  textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.8)
      first = False
  ax.set_xlabel("Search + Compile Time (s)")
  ax.set_ylabel("Kernel Execution Time (ms)")
  ax.set_title("Cost (search time) vs Benefit (kernel quality)\nannotations show beam width, bubble size ∝ beam width")
  ax.legend(loc="upper right", fontsize=7, ncol=2)
  ax.grid(True, alpha=0.3)
  _save(fig, PLOTS_DIR / "cost_benefit.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--suite", choices=get_suite_names(), default=DEFAULT_SUITE,
                      help="Named benchmark suite to run or plot. Ignored when --ops is passed.")
  parser.add_argument("--ops",      nargs="+", default=None,
                      help="Explicit op list. Overrides --suite.")
  parser.add_argument("--beams",    nargs="+", type=int, default=DEFAULT_BEAMS)
  parser.add_argument("--n-exec",   type=int,  default=DEFAULT_N_EXEC)
  parser.add_argument("--n-warmup", type=int,  default=DEFAULT_N_WARMUP)
  parser.add_argument("--load", action="store_true",
                      help="Skip collection; re-plot from results/beam_results.json")
  args = parser.parse_args()
  try:
    selected_ops = _select_ops(args.suite, args.ops)
  except ValueError as e:
    parser.error(str(e))

  if args.load:
    results_file = RESULTS_DIR / "beam_results.json"
    if not results_file.exists():
      print(f"No results at {results_file}; run without --load first.", file=sys.stderr)
      sys.exit(1)
    with open(results_file) as f:
      raw = json.load(f)
    results = _filter_results({op: {int(b): v for b, v in bmap.items()} for op, bmap in raw.items()}, selected_ops)
    beams   = sorted({b for bmap in results.values() for b in bmap})
    print(f"Loaded {results_file}")
  else:
    results = collect_results(selected_ops, args.beams, args.n_exec, args.n_warmup)
    beams   = args.beams

  if not results:
    print("No benchmark results to plot.", file=sys.stderr)
    sys.exit(1)

  print("\nGenerating plots ...")
  _require_matplotlib()
  plot_compile_time(results, beams)
  plot_exec_time(results, beams)
  plot_speedup(results, beams)
  plot_cost_benefit(results, beams)
  print("Done.")


if __name__ == "__main__":
  main()
