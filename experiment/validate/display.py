"""Table formatting and CSV output for validation results."""
import csv, math
from experiment.validate.harness import RunResult

_SEP = "│"
_RULE = "─"
_DIV  = "┼"

def _col(w: int, s: str, a: str = ">") -> str:
  return f"{s:{a}{w}}"

def _header(*cols: tuple[str, int, str]) -> str:
  return " " + f" {_SEP} ".join(_col(w, h, a) for h, w, a in cols) + " "

def _rule(*cols: tuple[str, int, str]) -> str:
  return _DIV.join(_RULE * (w + 2) for _, w, _ in cols)

def _row(*cells: tuple[str, int, str]) -> str:
  return " " + f" {_SEP} ".join(_col(w, v, a) for v, w, a in cells) + " "

def _f(v: float, d: int = 2) -> str:
  if math.isinf(v) or math.isnan(v): return "n/a"
  return f"{v:.{d}f}"

def _pct(a: float, b: float) -> str:
  if math.isinf(a) or math.isinf(b) or b == 0: return "n/a"
  return f"{(a - b) / b * 100:+.1f}%"


def print_comparison(
    baseline: list[RunResult],
    model_results: list[RunResult],
    beam_width: int = 5,
    model_name: str = "model",
    prune_factor: int = 4,
) -> None:
  """Print a side-by-side comparison table to stdout."""
  base_by_name = {r.op_name: r for r in baseline}
  model_by_name = {r.op_name: r for r in model_results}

  cols = [
    ("Op",             22, "<"),
    ("B-Beam s",        8, ">"),
    ("M-Beam s",        8, ">"),
    ("Search Δ",        9, ">"),
    ("B-Total s",       9, ">"),
    ("M-Total s",       9, ">"),
    ("B-µs",            8, ">"),
    ("M-µs",            8, ">"),
    ("Quality Δ",       9, ">"),
    ("B-compiled",     10, ">"),
    ("M-compiled",     10, ">"),
  ]

  print()
  print(f"Validation: Baseline vs {model_name}  "
        f"(beam_width={beam_width}, prune_factor={prune_factor})")
  print()
  print(_rule(*cols))
  print(_header(*cols))
  print(_rule(*cols))

  op_names = list(dict.fromkeys([r.op_name for r in baseline + model_results]))
  for name in op_names:
    b = base_by_name.get(name)
    m = model_by_name.get(name)
    if b is None or m is None:
      continue
    print(_row(
      (name[:22],                                  22, "<"),
      (_f(b.beam_wall_s),                           8, ">"),
      (_f(m.beam_wall_s),                           8, ">"),
      (_pct(m.beam_wall_s, b.beam_wall_s),          9, ">"),
      (_f(b.total_wall_s),                          9, ">"),
      (_f(m.total_wall_s),                          9, ">"),
      (_f(b.kernel_time_us, 1),                     8, ">"),
      (_f(m.kernel_time_us, 1),                     8, ">"),
      (_pct(m.kernel_time_us, b.kernel_time_us),    9, ">"),
      (str(b.n_compiled),                          10, ">"),
      (str(m.n_compiled),                          10, ">"),
    ))

  print(_rule(*cols))
  print()
  print("Columns: B=baseline, M=model, Δ=(M-B)/B×100%")
  print("Search Δ: negative = model is faster at search (good)")
  print("Quality Δ: negative = model chose a better kernel (good)")


def save_csv(results: list[RunResult], path: str) -> None:
  """Write all RunResult objects to a CSV file."""
  fields = [
    "op_name", "mode", "total_wall_s", "beam_wall_s",
    "n_compiled", "n_timed", "kernel_time_us", "best_opts", "error",
  ]
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in results:
      w.writerow({
        "op_name":        r.op_name,
        "mode":           r.mode,
        "total_wall_s":   r.total_wall_s,
        "beam_wall_s":    r.beam_wall_s,
        "n_compiled":     r.n_compiled,
        "n_timed":        r.n_timed,
        "kernel_time_us": r.kernel_time_us,
        "best_opts":      str(r.best_opts),
        "error":          r.error or "",
      })
