#!/usr/bin/env python3
"""Print beam-search layer metrics from a consolidated suite JSON file."""
import argparse, json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


def _fmt_list(xs) -> str:
  return "[" + ", ".join(str(x) for x in xs) + "]"


def _load_suite_file(path: Path) -> dict:
  if not path.exists(): raise SystemExit(f"results file does not exist: {path}")
  with open(path) as f: payload = json.load(f)
  if "results" not in payload:
    raise SystemExit(f"{path} does not look like a consolidated suite metrics file")
  return payload


def _collect_rows(payload: dict, op_filter: set[str] | None, beam_filter: set[int] | None):
  rows, layer_rows = [], []
  results = payload["results"]
  for op in payload.get("ops", results.keys()):
    if op_filter is not None and op not in op_filter: continue
    if op not in results: continue
    for beam_key, data in sorted(results[op].items(), key=lambda x: int(x[0])):
      beam = int(beam_key)
      if beam_filter is not None and beam not in beam_filter: continue
      for kernel_idx, record in enumerate(data.get("beam_kernels", []), start=1):
        executed_per_layer = record.get("executed_candidates_per_layer", [])
        total_per_layer = record.get("total_candidates_per_layer", [])
        rows.append({
          "op": op,
          "beam": beam,
          "kernel_index": kernel_idx,
          "kernel_name": record.get("kernel_name") or "<unknown>",
          "depth": record.get("depth", len(executed_per_layer)),
          "executed_total": record.get("executed_candidates_total", sum(executed_per_layer)),
          "candidate_compile_wall_time_s": record.get("candidate_compile_wall_time_s"),
          "executed_per_layer": executed_per_layer,
        })
        for layer_idx, executed in enumerate(executed_per_layer, start=1):
          total = total_per_layer[layer_idx-1] if layer_idx-1 < len(total_per_layer) else None
          layer_rows.append((op, beam, kernel_idx, record.get("kernel_name") or "<unknown>", layer_idx, executed, total))
  return rows, layer_rows


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--suite", default="default",
                      help="Suite result name to read from results/<suite>.json. Default: default")
  parser.add_argument("--file", type=Path, default=None,
                      help="Explicit consolidated JSON file. Overrides --suite and --results-dir.")
  parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                      help=f"Directory containing suite JSON files. Default: {DEFAULT_RESULTS_DIR}")
  parser.add_argument("--op", nargs="+", default=None,
                      help="Only print these test cases, for example: --op matmul_128 matmul_512")
  parser.add_argument("--beam", nargs="+", type=int, default=None,
                      help="Only print these beam widths, for example: --beam 1 2 4")
  parser.add_argument("--layers", action="store_true",
                      help="Also print one row per beam-search layer.")
  args = parser.parse_args()

  path = args.file if args.file is not None else args.results_dir / f"{args.suite}.json"
  payload = _load_suite_file(path)
  op_filter = set(args.op) if args.op else None
  beam_filter = set(args.beam) if args.beam else None
  rows, layer_rows = _collect_rows(payload, op_filter, beam_filter)

  if not rows:
    print(f"No beam-search kernel metrics found in {path}.")
    print("This is expected for BEAM=0 entries, because beam_search is not called.")
    return

  print(f"Beam Search Kernel Summary: {path}")
  print("-" * 142)
  print(f"{'test_case':24s} {'beam':>4s} {'kernel':>6s} {'kernel_name':28s} {'depth':>5s} {'executed':>9s} {'compile_wall_s':>14s}  executed_per_layer")
  print("-" * 142)
  for row in rows:
    compile_time = "" if row["candidate_compile_wall_time_s"] is None else f"{row['candidate_compile_wall_time_s']:.6f}"
    print(f"{row['op']:24s} {row['beam']:4d} {row['kernel_index']:6d} {row['kernel_name'][:28]:28s} "
          f"{row['depth']:5d} {row['executed_total']:9d} {compile_time:>14s}  {_fmt_list(row['executed_per_layer'])}")

  if args.layers:
    print("\nBeam Search Layer Details")
    print("-" * 100)
    print(f"{'test_case':24s} {'beam':>4s} {'kernel':>6s} {'layer':>5s} {'executed':>9s} {'total_candidates':>16s}  kernel_name")
    print("-" * 100)
    for op, beam, kernel_idx, kernel_name, layer_idx, executed, total in layer_rows:
      total_str = "" if total is None else str(total)
      print(f"{op:24s} {beam:4d} {kernel_idx:6d} {layer_idx:5d} {executed:9d} {total_str:>16s}  {kernel_name}")


if __name__ == "__main__":
  main()
