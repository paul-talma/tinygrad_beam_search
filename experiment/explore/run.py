"""Exploration harness: collect beam search metrics for ~10 representative ops.

Usage:
  uv run python -m experiment.explore.run [--beam N] [--ops op1,op2,...]

Prints per-op step tables and a final summary, then exits.
"""
import argparse, math, statistics, sys
from tinygrad.helpers import Context

from experiment.explore.instrumented import install, pop_traces, set_op_name, SearchTrace, StepRecord
from experiment.explore.ops import OPS

# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

_SEP  = "│"
_RULE = "─"
_DIV  = "┼"

def _fmt(v: float, decimals: int = 1) -> str:
  if math.isinf(v) or math.isnan(v): return "  n/a"
  return f"{v:.{decimals}f}"

def _col(width: int, s: str, align: str = ">") -> str:
  return f"{s:{align}{width}}"

def _header(*cols: tuple[str, int, str]) -> str:
  return " " + f" {_SEP} ".join(_col(w, h, a) for h, w, a in cols) + " "

def _rule(*cols: tuple[str, int, str]) -> str:
  # Each part is (w+2) dashes — matches the " content " cell in _header/_row
  return _DIV.join(_RULE * (w + 2) for _, w, _ in cols)

def _row(*cells: tuple[str, int, str]) -> str:
  return " " + f" {_SEP} ".join(_col(w, v, a) for v, w, a in cells) + " "


# ---------------------------------------------------------------------------
# Per-op display
# ---------------------------------------------------------------------------

def print_step_table(trace: SearchTrace, beam_width: int) -> None:
  cols = [
    ("Step",      4, ">"),
    ("Generated", 9, ">"),
    ("Compiled",  8, ">"),
    ("Unique",    6, ">"),
    ("Timed",     5, ">"),
    ("Min µs",    7, ">"),
    ("Max µs",    7, ">"),
    ("Mean µs",   8, ">"),
    ("Std µs",    7, ">"),
    ("Best opts so far", 40, "<"),
  ]
  print()
  print(f"Op: {trace.op_name}  (beam_width={beam_width})")
  print(_rule(*cols))
  print(_header(*cols))
  print(_rule(*cols))
  for rec in trace.steps:
    rts = rec.runtimes_us
    if rts:
      mn, mx, mu = min(rts), max(rts), statistics.mean(rts)
      sd = statistics.stdev(rts) if len(rts) > 1 else 0.0
    else:
      mn = mx = mu = sd = math.nan
    opts_str = str(rec.best_opts)[:40] if rec.best_opts else "[]"
    print(_row(
      (str(rec.step + 1),                    4, ">"),
      (str(rec.candidates_generated),         9, ">"),
      (str(rec.candidates_compiled),          8, ">"),
      (str(rec.candidates_unique),            6, ">"),
      (str(rec.candidates_timed),             5, ">"),
      (_fmt(mn),                              7, ">"),
      (_fmt(mx),                              7, ">"),
      (_fmt(mu),                              8, ">"),
      (_fmt(sd),                              7, ">"),
      (opts_str,                             40, "<"),
    ))
  print(_rule(*cols))
  print()

  base  = trace.baseline_time_us
  best  = trace.best_time_us
  spd   = base / best if best > 0 and not math.isinf(base) and not math.isinf(best) else float("nan")
  print(f"  Baseline (heuristic):  {_fmt(base, 2):>10} µs")
  print(f"  Best beam:             {_fmt(best, 2):>10} µs")
  if not math.isnan(spd):
    print(f"  Speedup:               {spd:>9.1f}×")
  print(f"  Beam search time:      {trace.total_beam_time_s:>9.2f} s")
  print(f"  Kernels compiled:      {trace.n_kernels_compiled:>9d}")
  print(f"  Kernels timed:         {trace.n_kernels_timed:>9d}")
  if trace.best_opts:
    print(f"  Best opts:             {trace.best_opts}")
  else:
    print(f"  Best opts:             (none — baseline was optimal)")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(traces: list[SearchTrace], beam_width: int) -> None:
  cols = [
    ("Op",              22, "<"),
    ("Steps",            5, ">"),
    ("Kernels",          7, ">"),
    ("Baseline µs",     11, ">"),
    ("Best µs",          8, ">"),
    ("Speedup",          7, ">"),
    ("Beam s",           6, ">"),
    ("Depth=1?",         8, "<"),
  ]
  print()
  print(f"{'─'*80}")
  print(f"Summary  (beam_width={beam_width})")
  print(f"{'─'*80}")
  print(_rule(*cols))
  print(_header(*cols))
  print(_rule(*cols))
  for t in traces:
    base = t.baseline_time_us
    best = t.best_time_us
    spd  = base / best if best > 0 and not math.isinf(base) and not math.isinf(best) else float("nan")
    depth1 = "yes" if len(t.steps) == 1 else ""
    print(_row(
      (t.op_name[:22],                    22, "<"),
      (str(len(t.steps)),                  5, ">"),
      (str(t.n_kernels_timed),             7, ">"),
      (_fmt(base, 1),                     11, ">"),
      (_fmt(best, 1),                      8, ">"),
      (f"{spd:.1f}×" if not math.isnan(spd) else "n/a", 7, ">"),
      (_fmt(t.total_beam_time_s, 2),       6, ">"),
      (depth1,                             8, "<"),
    ))
  print(_rule(*cols))
  print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(beam_width: int = 5, op_names: list[str] | None = None) -> list[SearchTrace]:
  install()  # monkeypatch once

  names = op_names if op_names else list(OPS.keys())
  all_traces: list[SearchTrace] = []

  for name in names:
    if name not in OPS:
      print(f"[warn] unknown op '{name}', skipping", file=sys.stderr)
      continue
    op_fn = OPS[name]
    print(f"\n{'='*60}\nRunning {name} ...", flush=True)
    set_op_name(name)
    try:
      with Context(BEAM=beam_width, IGNORE_BEAM_CACHE=1):
        op_fn().realize()
    except Exception as e:
      print(f"[error] {name}: {e}", file=sys.stderr)
      traces = pop_traces()
      if traces:
        traces[-1].error = str(e)
        all_traces.extend(traces)
      continue
    traces = pop_traces()
    all_traces.extend(traces)
    for t in traces:
      print_step_table(t, beam_width)

  print_summary(all_traces, beam_width)
  return all_traces


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Beam search exploration harness")
  parser.add_argument("--beam", type=int, default=5, help="Beam width (default: 5)")
  parser.add_argument("--ops", type=str, default="", help="Comma-separated op names (default: all)")
  args = parser.parse_args()
  op_names = [s.strip() for s in args.ops.split(",") if s.strip()] or None
  run(beam_width=args.beam, op_names=op_names)
