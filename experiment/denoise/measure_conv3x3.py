"""Measure execution time for tinygrad convolution benchmark kernels.

Examples:
  python -m experiment.denoise.measure_conv3x3
  python -m experiment.denoise.measure_conv3x3 --op conv_5x5 --beam 3 --warmup 3 --runs 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import statistics
from pathlib import Path

_repo_root = Path(__file__).parents[2]
_tinygrad_root = _repo_root / "tinygrad"
if str(_tinygrad_root) not in sys.path:
  sys.path.insert(0, str(_tinygrad_root))
if "DEBUG" in os.environ:
  try:
    int(os.environ["DEBUG"])
  except ValueError:
    os.environ.pop("DEBUG")

from tinygrad import GlobalCounters, Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import compile_linear, run_linear
from tinygrad.helpers import Context
from tinygrad.uop import Ops


def make_conv_op(op: str, cachelevel: int):
  with Context(CACHELEVEL=cachelevel):
    if op == "conv_3x3":
      x = Tensor.randn(4, 64, 32, 32, dtype=dtypes.float16).realize()
      w = Tensor.randn(128, 64, 3, 3, dtype=dtypes.float16).realize()
      return lambda: x.conv2d(w, padding=1)
    if op == "conv_5x5":
      x = Tensor.randn(4, 32, 32, 32, dtype=dtypes.float16).realize()
      w = Tensor.randn(64, 32, 5, 5, dtype=dtypes.float16).realize()
      return lambda: x.conv2d(w, padding=2)
  raise ValueError(f"unknown op {op!r}")


def kernel_signature(compiled_linear) -> tuple[str, list[dict], list[dict]]:
  """Fingerprint generated program source, launch metadata, and applied opts.

  Auto-generated function names can differ across runs even when the kernel body
  and optimization choices are identical, so names are normalized out.
  """
  programs = []
  sources = []
  for program in [u for u in compiled_linear.toposort() if u.op is Ops.PROGRAM]:
    kernel_info = program.src[0].arg if program.src and program.src[0].op is Ops.SINK else None
    applied_opts = tuple(getattr(kernel_info, "applied_opts", ()))
    normalized_opts = [
      (getattr(opt.op, "name", str(opt.op)), opt.axis, opt.arg)
      for opt in applied_opts
    ]
    source = next((u.arg for u in program.src if u.op is Ops.SOURCE), "")
    sources.append({
      "program_name": getattr(program.arg, "name", ""),
      "global_size": tuple(getattr(program.arg, "global_size", ())),
      "local_size": tuple(getattr(program.arg, "local_size", ())),
      "applied_opts": normalized_opts,
      "source": source,
    })
    program_name = getattr(program.arg, "name", "")
    kernel_name = getattr(kernel_info, "name", "")
    for name in {program_name, kernel_name} - {""}:
      source = source.replace(name, "<kernel>")
    programs.append({
      "global_size": tuple(getattr(program.arg, "global_size", ())),
      "local_size": tuple(getattr(program.arg, "local_size", ())),
      "applied_opts": normalized_opts,
      "source": source,
    })
  payload = json.dumps(programs, sort_keys=True).encode()
  return hashlib.sha256(payload).hexdigest()[:16], programs, sources


def run_once(op_fn, beam: int, cachelevel: int) -> tuple[float, int, str, list[dict]]:
  """Return timing, kernel count, signature, and generated program sources for one realize."""
  import tinygrad.codegen as codegen

  codegen.to_program_cache.clear()
  GlobalCounters.reset()
  with Context(BEAM=beam, CACHELEVEL=cachelevel):
    linear, var_vals = Tensor.linear_with_vars(op_fn())
    compiled_linear = compile_linear(linear)
    sig, _programs, sources = kernel_signature(compiled_linear)
    run_linear(compiled_linear, var_vals, wait=True, jit=True)
  Device[Device.DEFAULT].synchronize()
  return GlobalCounters.time_sum_s * 1e6, GlobalCounters.kernel_count, sig, sources


def _run_with_optional_regeneration(op_name: str, op_fn, regenerate: bool, beam: int, cachelevel: int) -> tuple[float, int, str, list[dict]]:
  if regenerate:
    op_fn = make_conv_op(op_name, cachelevel)
  return run_once(op_fn, beam, cachelevel)


def measure(op: str, beam: int, warmup: int, runs: int, cachelevel: int, regenerate: bool, include_sources: bool) -> dict:
  op_fn = None if regenerate else make_conv_op(op, cachelevel)
  for _ in range(warmup):
    _run_with_optional_regeneration(op, op_fn, regenerate, beam, cachelevel)

  times_us: list[float] = []
  kernel_counts: list[int] = []
  kernel_signatures: list[str] = []
  applied_opts_per_run: list[list[list[tuple[str, int | None, int | None]]]] = []
  generated_sources: list[list[dict]] = []
  for _ in range(runs):
    t_us, kernel_count, sig, sources = _run_with_optional_regeneration(op, op_fn, regenerate, beam, cachelevel)
    times_us.append(t_us)
    kernel_counts.append(kernel_count)
    kernel_signatures.append(sig)
    applied_opts_per_run.append([src["applied_opts"] for src in sources])
    if include_sources:
      generated_sources.append(sources)

  signature_counts = {sig: kernel_signatures.count(sig) for sig in sorted(set(kernel_signatures))}
  opt_sequence_keys = [json.dumps(opts) for opts in applied_opts_per_run]
  opt_sequence_counts = {key: opt_sequence_keys.count(key) for key in sorted(set(opt_sequence_keys))}

  result = {
    "op": op,
    "beam": beam,
    "warmup": warmup,
    "runs": runs,
    "cachelevel": cachelevel,
    "regenerate": regenerate,
    "kernel_count_median": statistics.median(kernel_counts),
    "unique_generated_kernels": len(signature_counts),
    "kernel_signature_counts": signature_counts,
    "kernel_signatures": kernel_signatures,
    "unique_applied_opt_sequences": len(opt_sequence_counts),
    "applied_opt_sequence_counts": opt_sequence_counts,
    "applied_opts_per_run": applied_opts_per_run,
    "median_us": statistics.median(times_us),
    "min_us": min(times_us),
    "max_us": max(times_us),
    "mean_us": statistics.mean(times_us),
    "times_us": times_us,
  }
  if include_sources:
    result["generated_sources"] = generated_sources
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description="Measure convolution kernel execution time.")
  parser.add_argument("--op", choices=("conv_3x3", "conv_5x5"), default="conv_3x3", help="Convolution benchmark to measure.")
  parser.add_argument("--beam", type=int, default=0, help="tinygrad BEAM value. 0 means no beam search.")
  parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs before measuring.")
  parser.add_argument("--runs", type=int, default=10, help="Number of measured runs.")
  parser.add_argument("--cachelevel", type=int, default=0, help="tinygrad CACHELEVEL value.")
  parser.add_argument("--regenerate", action="store_true", help="Regenerate and realize fresh input tensors for every run.")
  parser.add_argument("--print-source", action="store_true", help="Print generated source for each measured run.")
  parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
  args = parser.parse_args()

  result = measure(
    op=args.op, beam=args.beam, warmup=args.warmup, runs=args.runs,
    cachelevel=args.cachelevel, regenerate=args.regenerate, include_sources=args.print_source,
  )
  if args.json:
    print(json.dumps(result))
    return

  print(f"{args.op} kernel execution time (BEAM={args.beam})")
  print(f"runs: {args.runs}, warmup: {args.warmup}, regenerate: {args.regenerate}, median kernels/run: {result['kernel_count_median']}")
  print(f"unique generated kernels: {result['unique_generated_kernels']}")
  print(f"kernel signatures: {result['kernel_signature_counts']}")
  print(f"unique applied opt sequences: {result['unique_applied_opt_sequences']}")
  print("applied opts per run:")
  for run_idx, opts in enumerate(result["applied_opts_per_run"], start=1):
    print(f"  run {run_idx}: {opts}")
  print(f"median: {result['median_us']:.3f} us")
  print(f"min:    {result['min_us']:.3f} us")
  print(f"max:    {result['max_us']:.3f} us")
  print(f"mean:   {result['mean_us']:.3f} us")
  print("all:    " + ", ".join(f"{t:.3f}" for t in result["times_us"]))
  if args.print_source:
    for run_idx, sources in enumerate(result["generated_sources"], start=1):
      print(f"\n=== run {run_idx} generated source ===")
      for program_idx, src in enumerate(sources, start=1):
        print(f"--- program {program_idx}: {src['program_name']} ---")
        print(f"global_size={src['global_size']} local_size={src['local_size']}")
        print(f"applied_opts={src['applied_opts']}")
        print(src["source"])


if __name__ == "__main__":
  main()
