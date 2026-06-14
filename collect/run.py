"""Collect (features, runtime) training data by running beam search over ops.

Usage:
  # All ops, beam width 5, output to data/v2/train.jsonl
  uv run python -m collect.run

  # Specific ops
  uv run python -m collect.run --ops matmul_4096 conv_medium

  # Custom beam width and output
  uv run python -m collect.run --beam 3 --out data/v2/test.jsonl
"""
import argparse
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_here, ".."))
_submodule_root = os.path.join(_repo_root, "tinygrad")

if _repo_root not in sys.path:
  sys.path.insert(0, _repo_root)
if _submodule_root not in sys.path:
  sys.path.insert(0, _submodule_root)

os.environ.setdefault("IGNORE_BEAM_CACHE", "1")


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--ops",  nargs="*", help="Specific op names (default: all)")
  parser.add_argument("--beam", type=int, default=5, help="Beam width (default: 5)")
  parser.add_argument("--out",  default="data/v2/train.jsonl", help="Output JSONL path")
  parser.add_argument("--resume-after", metavar="OP", help="Skip all ops up to and including this op name")
  args = parser.parse_args()

  import tinygrad  # noqa: F401
  from tinygrad.helpers import Context
  from collect.hook import DataCollector, patch_beam_search, set_op_name
  from collect.ops import ALL_OPS, get_op
  from tqdm import tqdm

  op_names = args.ops if args.ops else list(ALL_OPS)
  unknown = [n for n in op_names if n not in ALL_OPS]
  if unknown:
    print(f"Unknown ops: {unknown}", file=sys.stderr)
    print(f"Available: {sorted(ALL_OPS)}", file=sys.stderr)
    sys.exit(1)

  if args.resume_after:
    if args.resume_after not in op_names:
      print(f"--resume-after op {args.resume_after!r} not in op list", file=sys.stderr)
      sys.exit(1)
    cut = op_names.index(args.resume_after) + 1
    skipped = op_names[:cut]
    op_names = op_names[cut:]
    print(f"Skipping {len(skipped)} ops (through {args.resume_after!r}), {len(op_names)} remaining.")

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  collector = DataCollector(args.out)
  patch_beam_search(collector)

  print(f"Collecting {len(op_names)} ops, beam={args.beam}, TC=0, allow_test_size=False, output={args.out}")

  with Context(BEAM=args.beam, TC=0):
    for op_name in tqdm(op_names, desc="ops", unit="op"):
      set_op_name(op_name)
      try:
        op = get_op(op_name)
        op().realize()
      except Exception as e:
        tqdm.write(f"FAILED {op_name}: {e}", file=sys.stderr)

  collector.close()
  print("Done.")


if __name__ == "__main__":
  main()
