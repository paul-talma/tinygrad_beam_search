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
    ("Op",           24, "<"),
    ("B-Beam s",      8, ">"),
    ("M-Beam s",      8, ">"),
    ("Search Δ",      9, ">"),
    ("B-µs(est)",     9, ">"),
    ("M-µs(real)",    9, ">"),
    ("B-µs(true)",    9, ">"),
    ("M-µs(true)",    9, ">"),
    ("True-Δ",        9, ">"),
    ("B-cpld",        6, ">"),
    ("M-cpld",        6, ">"),
  ]

  print()
  print(f"Validation: Baseline vs {model_name}  "
        f"(beam_width={beam_width}, prune_factor={prune_factor})")
  print("★ = unseen kernel (not in training data)")
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
    unseen = not b.is_trained or not m.is_trained
    label = ("★ " + name)[:24] if unseen else name[:24]
    print(_row(
      (label,                                              24, "<"),
      (_f(b.beam_wall_s),                                  8, ">"),
      (_f(m.beam_wall_s),                                  8, ">"),
      (_pct(m.beam_wall_s, b.beam_wall_s),                 9, ">"),
      (_f(b.kernel_time_us, 1),                            9, ">"),
      (_f(m.kernel_time_us, 1),                            9, ">"),
      (_f(b.kernel_time_true_us, 1),                       9, ">"),
      (_f(m.kernel_time_true_us, 1),                       9, ">"),
      (_pct(m.kernel_time_true_us, b.kernel_time_true_us), 9, ">"),
      (str(b.n_compiled),                                  6, ">"),
      (str(m.n_compiled),                                  6, ">"),
    ))

  print(_rule(*cols))
  print()
  print("Columns: B=baseline, M=model, Δ=(M−B)/B×100%")
  print("Search Δ: negative = model searches faster (good)")
  print("True-Δ:   negative = model found a better kernel (good)")
  print("B-µs(est): baseline timing uses allow_test_size=True (scaled estimate)")
  print("M-µs(real): model timing uses allow_test_size=False (true real-size timing)")
  print("B/M-µs(true): winner re-timed at allow_test_size=False for apples-to-apples comparison")

  # --- Search time summary (all ops) ---
  paired = [(base_by_name[n], model_by_name[n])
            for n in op_names if n in base_by_name and n in model_by_name
            and not base_by_name[n].error and not model_by_name[n].error]
  if paired:
    n_faster = sum(1 for b, m in paired if m.beam_wall_s < b.beam_wall_s)
    mean_delta = sum((m.beam_wall_s - b.beam_wall_s) / b.beam_wall_s
                     for b, m in paired if b.beam_wall_s > 0) / len(paired) * 100
    print()
    print(f"Search time (all {len(paired)} ops):  "
          f"model faster on {n_faster}/{len(paired)},  mean Δ {mean_delta:+.1f}%")

  # --- Quality summary for unseen kernels only ---
  unseen_pairs = [(b, m) for b, m in paired
                  if not b.is_trained or not m.is_trained]
  if unseen_pairs:
    valid = [(b, m) for b, m in unseen_pairs
             if not math.isinf(b.kernel_time_true_us) and not math.isinf(m.kernel_time_true_us)
             and b.kernel_time_true_us > 0]
    if valid:
      n_wins = sum(1 for b, m in valid if m.kernel_time_true_us < b.kernel_time_true_us)
      mean_quality = sum((m.kernel_time_true_us - b.kernel_time_true_us) / b.kernel_time_true_us
                         for b, m in valid) / len(valid) * 100
      print()
      print(f"Quality on unseen kernels ★ ({len(valid)} ops):  "
            f"model wins {n_wins}/{len(valid)},  mean true-time Δ {mean_quality:+.1f}%")


def save_csv(results: list[RunResult], path: str) -> None:
  """Write all RunResult objects to a CSV file."""
  fields = [
    "op_name", "mode", "total_wall_s", "beam_wall_s",
    "n_compiled", "n_timed", "kernel_time_us", "kernel_time_true_us",
    "heuristic_time_us", "kernel_id", "is_trained", "best_opts", "error",
  ]
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in results:
      w.writerow({
        "op_name":              r.op_name,
        "mode":                 r.mode,
        "total_wall_s":         r.total_wall_s,
        "beam_wall_s":          r.beam_wall_s,
        "n_compiled":           r.n_compiled,
        "n_timed":              r.n_timed,
        "kernel_time_us":       r.kernel_time_us,
        "kernel_time_true_us":  r.kernel_time_true_us,
        "heuristic_time_us":    r.heuristic_time_us,
        "kernel_id":            r.kernel_id,
        "is_trained":           r.is_trained,
        "best_opts":            str(r.best_opts),
        "error":                r.error or "",
      })
