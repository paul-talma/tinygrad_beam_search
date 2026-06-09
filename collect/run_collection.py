"""Collect (features, runtime) training data by running beam search over a suite of ops.

Usage:
  # All ops, beam width 5, output to data/train.jsonl
  uv run python collect/run_collection.py

  # Specific ops
  uv run python collect/run_collection.py --ops matmul_4096 conv_medium

  # Specific suite
  uv run python collect/run_collection.py --suite matmuls

  # Custom beam width and output
  uv run python collect/run_collection.py --beam 3 --out data/test.jsonl
"""
import argparse
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_here, ".."))
_submodule_root = os.path.join(_repo_root, "tinygrad")

# Ensure the repo root is on sys.path so `collect` and `benchmarks` are importable
# when this script is run directly (python collect/run_collection.py).
if _repo_root not in sys.path:
  sys.path.insert(0, _repo_root)

# The tinygrad/ submodule directory at the repo root shadows the editable install because
# Python finds it as a namespace package via '' in sys.path. Insert the submodule root
# so Python finds tinygrad/tinygrad/__init__.py as a real package first.
if _submodule_root not in sys.path:
  sys.path.insert(0, _submodule_root)

# Run with IGNORE_BEAM_CACHE so every op actually runs beam search
os.environ.setdefault("IGNORE_BEAM_CACHE", "1")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ops",   nargs="*", help="Specific op names to collect (default: all)")
  parser.add_argument("--suite", default=None, help="Named suite from ops.py (e.g. matmuls)")
  parser.add_argument("--beam",  type=int, default=5, help="Beam width (default: 5)")
  parser.add_argument("--out",   default="data/train.jsonl", help="Output JSONL path")
  args = parser.parse_args()

  # tinygrad must be fully initialized before importing codegen submodules
  import tinygrad  # noqa: F401
  from tinygrad.helpers import Context
  from collect.hook import DataCollector, patch_beam_search, set_op_name
  import benchmarks.beam_search.ops as op_registry
  from tqdm import tqdm

  # Resolve op list
  if args.ops:
    op_names = args.ops
  elif args.suite:
    op_names = op_registry.get_suite_ops(args.suite)
  else:
    op_names = list(op_registry.BENCHMARK_SUITES["all"].keys())

  unknown = [n for n in op_names if n not in op_registry.BENCHMARK_SUITES["all"]]
  if unknown:
    print(f"Unknown ops: {unknown}", file=sys.stderr)
    print(f"Available: {sorted(op_registry.BENCHMARK_SUITES['all'].keys())}", file=sys.stderr)
    sys.exit(1)

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  collector = DataCollector(args.out)
  patch_beam_search(collector)

  print(f"Collecting {len(op_names)} ops, beam={args.beam}, output={args.out}")

  with Context(BEAM=args.beam):
    for op_name in tqdm(op_names, desc="ops", unit="op"):
      set_op_name(op_name)
      try:
        op = op_registry.get_op(op_name)
        op().realize()
      except Exception as e:
        tqdm.write(f"FAILED {op_name}: {e}", file=sys.stderr)

  collector.close()
  print("Done.")

if __name__ == "__main__":
  main()
