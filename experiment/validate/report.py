"""Produce a human-readable summary from validation CSV(s).

Usage:
  uv run python -m experiment.validate.report                    # latest CSV
  uv run python -m experiment.validate.report results/foo.csv   # specific CSV
  uv run python -m experiment.validate.report a.csv b.csv       # merge two CSVs

Columns reported per op:
  Heur µs      — hand_coded_optimizations only, no beam search
  B-srch s     — wall time of baseline beam search
  B-kern µs    — baseline winner re-timed at full size
  F-srch s     — model search time, allow_test_size=False (full timing)
  F-kern µs    — model winner, full timing, re-timed at full size
  E-srch s     — model search time, allow_test_size=True (estimated timing)
  E-kern µs    — model winner, estimated timing, re-timed at full size
  B→F Δ        — quality change: full-timing model vs baseline
  B→E Δ        — quality change: est-timing model vs baseline
  F vs E       — kernel quality: full-timing model vs est-timing model
"""
import argparse, csv, math, pathlib, sys


def _f(v: float, d: int = 1) -> str:
  if math.isinf(v) or math.isnan(v) or v == 0.0:
    return "n/a"
  return f"{v:.{d}f}"


def _pct(a: float, b: float) -> str:
  if math.isinf(a) or math.isinf(b) or b == 0:
    return "n/a"
  return f"{(a - b) / b * 100:+.1f}%"


def _latest_csv() -> str:
  results_dir = pathlib.Path(__file__).parents[2] / "experiment" / "results"
  # Prefer combined CSVs (contain all modes) over individual run CSVs.
  combined = sorted(results_dir.glob("validation_combined_*.csv"), reverse=True)
  if combined:
    return str(combined[0])
  csvs = sorted(results_dir.glob("validation_*.csv"), reverse=True)
  if not csvs:
    print("No validation CSVs found in experiment/results/", file=sys.stderr)
    sys.exit(1)
  return str(csvs[0])


def _val(row: dict, key: str) -> float:
  try:
    return float(row.get(key) or "inf")
  except (ValueError, TypeError):
    return math.inf


def load_csv(path: str) -> list[dict]:
  rows = []
  with open(path, newline="") as f:
    for row in csv.DictReader(f):
      rows.append(row)
  return rows


def classify_mode(mode: str) -> str:
  """Return 'baseline', 'model-full', or 'model-est' for a mode string."""
  if mode == "baseline":
    return "baseline"
  if mode.endswith(":full"):
    return "model-full"
  if mode.endswith(":est"):
    return "model-est"
  # Legacy mode strings without suffix — treat as full (previous behaviour).
  return "model-full"


def _lp(a: float, b: float) -> str:
  """LaTeX coloured percentage, or -- if undefined."""
  if math.isinf(a) or math.isinf(b) or b == 0:
    return r"--"
  v = (a - b) / b * 100
  cmd = r"\better" if v < -0.5 else (r"\worse" if v > 0.5 else r"\same")
  return rf"{cmd}{{{v:+.1f}\%}}"


def _lf_us(v: float) -> str:
  """Format microseconds as integer."""
  if math.isinf(v) or math.isnan(v) or v == 0.0:
    return "--"
  return f"{v:.0f}"


def _lf_s(v: float) -> str:
  """Format seconds with one decimal."""
  if math.isinf(v) or math.isnan(v) or v == 0.0:
    return "--"
  return f"{v:.1f}"


def _lmean_pct(lst: list, invert: bool = False) -> str:
  if not lst:
    return "--"
  v = sum(lst) / len(lst)
  cmd = r"\better" if v < -0.5 else (r"\worse" if v > 0.5 else r"\same")
  return rf"{cmd}{{{v:+.1f}\%}}"


def print_latex(by_op: dict, ops: list,
                search_deltas_f: list, search_deltas_e: list,
                qual_deltas_f: list, qual_deltas_e: list) -> None:
  lines = []
  a = lines.append

  a(r"% Requires: \usepackage{booktabs,xcolor,siunitx}")
  a(r"% In preamble:")
  a(r"%   \newcommand{\better}[1]{\textcolor{teal}{#1}}")
  a(r"%   \newcommand{\worse}[1]{\textcolor{red!70!black}{#1}}")
  a(r"%   \newcommand{\same}[1]{#1}")
  a(r"")
  a(r"\begin{table*}[t]")
  a(r"\centering\footnotesize\setlength{\tabcolsep}{5pt}")
  # Cols: kernel | heur | B-srch | B-kern | F-srch | F-kern | E-srch | E-kern | B→F | B→E | F↔E
  # Use S columns for numeric data so decimal points align; wrap non-numeric in {}
  a(r"\begin{tabular}{l"
    r" S[table-format=6.0]"           # Heur µs
    r" S[table-format=3.1]"           # B-srch s
    r" S[table-format=6.0]"           # B-kern µs
    r" S[table-format=3.1]"           # F-srch s
    r" S[table-format=6.0]"           # F-kern µs
    r" S[table-format=3.1]"           # E-srch s
    r" S[table-format=6.0]"           # E-kern µs
    r" r r r}")
  a(r"\toprule")
  a(r"& {\textbf{Heur}} & \multicolumn{2}{c}{\textbf{Baseline (B)}}"
    r" & \multicolumn{2}{c}{\textbf{Model-Full (F)}}"
    r" & \multicolumn{2}{c}{\textbf{Model-Est (E)}}"
    r" & \multicolumn{3}{c}{\textbf{Quality $\Delta$}} \\")
  a(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}")
  a(r"{\textbf{Kernel}} & {(\textmu s)} & {srch (s)} & {kern (\textmu s)}"
    r" & {srch (s)} & {kern (\textmu s)}"
    r" & {srch (s)} & {kern (\textmu s)}"
    r" & {B$\to$F} & {B$\to$E} & {F$\leftrightarrow$E} \\")
  a(r"\midrule")

  for op in ops:
    modes = by_op.get(op, {})
    b  = modes.get("baseline")
    mf = modes.get("model-full")
    me = modes.get("model-est")
    if b is None:
      continue

    label = op.replace("_", r"\_")
    heur   = _val(b,  "heuristic_time_us")
    b_srch = _val(b,  "beam_wall_s");    b_kern = _val(b,  "kernel_time_true_us")
    f_srch = _val(mf, "beam_wall_s") if mf else math.inf
    f_kern = _val(mf, "kernel_time_true_us") if mf else math.inf
    e_srch = _val(me, "beam_wall_s") if me else math.inf
    e_kern = _val(me, "kernel_time_true_us") if me else math.inf

    # S columns need non-numeric content in braces
    def sc(s): return "{" + s + "}" if not s.lstrip("-").replace(".","").isdigit() else s

    a(f"  {label}"
      f" & {sc(_lf_us(heur))}"
      f" & {_lf_s(b_srch)} & {sc(_lf_us(b_kern))}"
      f" & {_lf_s(f_srch)} & {sc(_lf_us(f_kern))}"
      f" & {_lf_s(e_srch)} & {sc(_lf_us(e_kern))}"
      f" & {_lp(f_kern, b_kern)}"
      f" & {_lp(e_kern, b_kern)}"
      f" & {_lp(f_kern, e_kern)}"
      r" \\")

  # Summary rows — each average sits in the column it describes:
  # cols: 1=kernel 2=heur 3=B-srch 4=B-kern 5=F-srch 6=F-kern 7=E-srch 8=E-kern 9=B→F 10=B→E 11=F↔E
  a(r"\midrule")
  sf = _lmean_pct(search_deltas_f)
  se = _lmean_pct(search_deltas_e)
  qf = _lmean_pct(qual_deltas_f)
  qe = _lmean_pct(qual_deltas_e)
  a(rf"  \textit{{Search $\Delta$}} & & & & {sf} & & {se} & & & & \\")
  a(rf"  \textit{{Quality $\Delta$}} & & & & & & & & {qf} & {qe} & \\")

  a(r"\bottomrule")
  a(r"\end{tabular}")
  a(r"\caption{%")
  a(r"  Beam search validation on 15 kernels (beam width $= 3$, prune factor $= 4$).")
  a(r"  \textbf{Heur}: hand-coded heuristic, no beam search.")
  a(r"  \textbf{B}: vanilla beam search.")
  a(r"  \textbf{F}: model-guided, full timing (\texttt{allow\_test\_size=False}).")
  a(r"  \textbf{E}: model-guided, estimated timing (\texttt{allow\_test\_size=True}).")
  a(r"  Search time in seconds; kernel time in \textmu s, re-timed at full size.")
  a(r"  $\Delta = (\text{model} - \text{B})/\text{B}$;"
    r"  \textcolor{teal}{teal} = improvement over B, \textcolor{red!70!black}{red} = regression.")
  a(r"}")
  a(r"\label{tab:beam_results}")
  a(r"\end{table*}")

  print("\n".join(lines))


def main() -> None:
  parser = argparse.ArgumentParser(description="Validation result summary")
  parser.add_argument("csvs", nargs="*", help="Validation CSV path(s) (default: latest)")
  parser.add_argument("--latex", action="store_true", help="Output LaTeX table instead of ASCII")
  args = parser.parse_args()

  paths = args.csvs or [_latest_csv()]

  # Load and merge all rows; last write wins per (op, mode-class).
  by_op: dict[str, dict[str, dict]] = {}  # op -> mode_class -> row
  for path in paths:
    for row in load_csv(path):
      op   = row["op_name"]
      cls  = classify_mode(row.get("mode", ""))
      by_op.setdefault(op, {})[cls] = row

  # Ordered op list (preserving first-seen order).
  ops = list(dict.fromkeys(op for path in paths for row in load_csv(path) for op in [row["op_name"]]))

  # ── column layout ──────────────────────────────────────────────────────────
  W = [20, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8]
  H = ["Op", "Heur µs", "B-srch s", "B-kern µs",
       "F-srch s", "F-kern µs", "E-srch s", "E-kern µs",
       "B→F Δ", "B→E Δ", "F vs E"]
  AL = ["<", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">"]

  sep, div = "─", "┼"

  def _row(*cells):
    return " " + f" {div} ".join(f"{v:{a}{w}}" for v, a, w in zip(cells, AL, W)) + " "

  def _rule():
    return div.join(sep * (w + 2) for w in W)

  # ── collect per-op data ────────────────────────────────────────────────────
  search_deltas_f, search_deltas_e = [], []
  qual_deltas_f, qual_deltas_e = [], []

  for op in ops:
    modes = by_op.get(op, {})
    b  = modes.get("baseline")
    mf = modes.get("model-full")
    me = modes.get("model-est")
    if b is None:
      continue

    is_unseen = b.get("is_trained", "False") == "False"
    b_srch = _val(b, "beam_wall_s");  b_kern = _val(b, "kernel_time_true_us")
    f_srch = _val(mf, "beam_wall_s") if mf else math.inf
    f_kern = _val(mf, "kernel_time_true_us") if mf else math.inf
    e_srch = _val(me, "beam_wall_s") if me else math.inf
    e_kern = _val(me, "kernel_time_true_us") if me else math.inf

    if mf and not math.isinf(f_srch) and b_srch > 0:
      search_deltas_f.append((f_srch - b_srch) / b_srch * 100)
    if me and not math.isinf(e_srch) and b_srch > 0:
      search_deltas_e.append((e_srch - b_srch) / b_srch * 100)
    if mf and is_unseen and not math.isinf(b_kern) and not math.isinf(f_kern) and b_kern > 0:
      qual_deltas_f.append((f_kern - b_kern) / b_kern * 100)
    if me and is_unseen and not math.isinf(b_kern) and not math.isinf(e_kern) and b_kern > 0:
      qual_deltas_e.append((e_kern - b_kern) / b_kern * 100)

  if args.latex:
    print_latex(by_op, ops, search_deltas_f, search_deltas_e, qual_deltas_f, qual_deltas_e)
    return

  # ── ASCII header ───────────────────────────────────────────────────────────
  print()
  print("Validation Summary — all four approaches")
  print(f"  Sources: {', '.join(paths)}")
  print("  ★ = unseen kernel | B = baseline beam | F = model+full | E = model+est")
  print()
  print("  Heur: hand_coded_optimizations only, no beam search")
  print("  B:    vanilla beam search (allow_test_size=True, estimated timing)")
  print("  F:    model-guided beam search (allow_test_size=False, real-size timing)")
  print("  E:    model-guided beam search (allow_test_size=True, estimated timing)")
  print()
  print(_rule())
  print(_row(*H))
  print(_rule())

  # ── ASCII per-op rows ──────────────────────────────────────────────────────
  for op in ops:
    modes = by_op.get(op, {})
    b  = modes.get("baseline")
    mf = modes.get("model-full")
    me = modes.get("model-est")
    if b is None:
      continue

    is_unseen = b.get("is_trained", "False") == "False"
    label = ("★ " + op)[:W[0]] if is_unseen else op[:W[0]]

    heur   = _val(b,  "heuristic_time_us")
    b_srch = _val(b,  "beam_wall_s");    b_kern = _val(b,  "kernel_time_true_us")
    f_srch = _val(mf, "beam_wall_s") if mf else math.inf
    f_kern = _val(mf, "kernel_time_true_us") if mf else math.inf
    e_srch = _val(me, "beam_wall_s") if me else math.inf
    e_kern = _val(me, "kernel_time_true_us") if me else math.inf

    print(_row(
      label,
      _f(heur),
      _f(b_srch, 2), _f(b_kern),
      _f(f_srch, 2), _f(f_kern),
      _f(e_srch, 2), _f(e_kern),
      _pct(f_kern, b_kern),
      _pct(e_kern, b_kern),
      _pct(f_kern, e_kern),
    ))

  print(_rule())
  print()
  print("Δ = (model − baseline) / baseline × 100%  (negative = model is better)")
  print()

  if search_deltas_f:
    nf = sum(1 for d in search_deltas_f if d < 0)
    print(f"Search time — F (full): model faster {nf}/{len(search_deltas_f)},  mean {sum(search_deltas_f)/len(search_deltas_f):+.1f}%")
  if search_deltas_e:
    ne = sum(1 for d in search_deltas_e if d < 0)
    print(f"Search time — E (est):  model faster {ne}/{len(search_deltas_e)},  mean {sum(search_deltas_e)/len(search_deltas_e):+.1f}%")
  if qual_deltas_f:
    nf = sum(1 for d in qual_deltas_f if d < 0)
    print(f"Kernel quality — F (★ unseen):  model wins {nf}/{len(qual_deltas_f)},  mean {sum(qual_deltas_f)/len(qual_deltas_f):+.1f}%")
  if qual_deltas_e:
    ne = sum(1 for d in qual_deltas_e if d < 0)
    print(f"Kernel quality — E (★ unseen):  model wins {ne}/{len(qual_deltas_e)},  mean {sum(qual_deltas_e)/len(qual_deltas_e):+.1f}%")


if __name__ == "__main__":
  main()
