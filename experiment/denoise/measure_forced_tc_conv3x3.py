"""Measure conv_3x3 with a fixed tinygrad optimization sequence.

Forced opts:
  [('TC', 2, (-1, 2, 1)), ('UPCAST', 2, 4)]

Example:
  DEV=METAL python -m experiment.denoise.measure_forced_tc_conv3x3 --runs 20 --regenerate
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import replace
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
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import compile_linear, run_linear
from tinygrad.helpers import Context
from tinygrad.uop import Ops
from tinygrad.uop.ops import KernelInfo

FORCED_OPTS = (
  Opt(OptOps.TC, 2, (-1, 2, 1)),
  Opt(OptOps.UPCAST, 2, 4),
)


def make_conv_3x3(cachelevel: int):
  with Context(CACHELEVEL=cachelevel):
    x = Tensor.randn(4, 64, 32, 32, dtype=dtypes.float16).realize()
    w = Tensor.randn(128, 64, 3, 3, dtype=dtypes.float16).realize()
  return lambda: x.conv2d(w, padding=1)


def force_opts(linear):
  """Attach opts_to_apply to every kernel sink in the linear graph."""
  def visit(u):
    new_src = tuple(visit(s) for s in u.src)
    if u.op is Ops.SINK:
      info = u.arg if isinstance(u.arg, KernelInfo) else KernelInfo()
      return u.replace(src=new_src, arg=replace(info, opts_to_apply=FORCED_OPTS))
    return u.replace(src=new_src)
  return visit(linear)


def normalized_opts(applied_opts) -> list[tuple[str, int | None, object]]:
  return [(getattr(opt.op, "name", str(opt.op)), opt.axis, opt.arg) for opt in applied_opts]


def program_info(compiled_linear) -> list[dict]:
  out = []
  for program in [u for u in compiled_linear.toposort() if u.op is Ops.PROGRAM]:
    kernel_info = program.src[0].arg if program.src and program.src[0].op is Ops.SINK else None
    source = next((u.arg for u in program.src if u.op is Ops.SOURCE), "")
    out.append({
      "program_name": getattr(program.arg, "name", ""),
      "global_size": tuple(getattr(program.arg, "global_size", ())),
      "local_size": tuple(getattr(program.arg, "local_size", ())),
      "applied_opts": normalized_opts(getattr(kernel_info, "applied_opts", ())),
      "source": source,
    })
  return out


def rawbufs_from_program(program) -> list[Buffer]:
  device = next((u.arg for u in program.src if u.op is Ops.DEVICE), Device.DEFAULT)
  params = sorted(
    [u for u in program.toposort() if u.op is Ops.PARAM],
    key=lambda u: u.arg.slot,
  )
  return [Buffer(device, p.max_numel(), p.dtype.base).ensure_allocated() for p in params]


def run_once(op_fn, cachelevel: int) -> tuple[float, list[float], int, list[dict]]:
  import tinygrad.codegen as codegen
  from tinygrad.codegen.opt.search import _time_program

  codegen.to_program_cache.clear()
  GlobalCounters.reset()
  with Context(BEAM=0, CACHELEVEL=cachelevel):
    linear, var_vals = Tensor.linear_with_vars(op_fn())
    compiled_linear = compile_linear(force_opts(linear), beam=0)
    info = program_info(compiled_linear)
    programs = [u for u in compiled_linear.toposort() if u.op is Ops.PROGRAM]
    beam_style_times = []
    for program in programs:
      beam_style_times.extend(_time_program(program, var_vals, rawbufs_from_program(program)))
    run_linear(compiled_linear, var_vals, wait=True, jit=True)
  Device[Device.DEFAULT].synchronize()
  return GlobalCounters.time_sum_s * 1e6, [x * 1e6 for x in beam_style_times], GlobalCounters.kernel_count, info


def measure(runs: int, warmup: int, regenerate: bool, cachelevel: int) -> dict:
  op_fn = None if regenerate else make_conv_3x3(cachelevel)
  for _ in range(warmup):
    run_once(make_conv_3x3(cachelevel) if regenerate else op_fn, cachelevel)

  times_us: list[float] = []
  beam_style_times_us: list[list[float]] = []
  kernel_counts: list[int] = []
  infos: list[list[dict]] = []
  for _ in range(runs):
    t_us, beam_t_us, kernel_count, info = run_once(make_conv_3x3(cachelevel) if regenerate else op_fn, cachelevel)
    times_us.append(t_us)
    beam_style_times_us.append(beam_t_us)
    kernel_counts.append(kernel_count)
    infos.append(info)

  beam_style_mins_us = [min(x) for x in beam_style_times_us]

  return {
    "op": "conv_3x3",
    "forced_opts": normalized_opts(FORCED_OPTS),
    "runs": runs,
    "warmup": warmup,
    "regenerate": regenerate,
    "cachelevel": cachelevel,
    "kernel_count_median": statistics.median(kernel_counts),
    "median_us": statistics.median(times_us),
    "min_us": min(times_us),
    "max_us": max(times_us),
    "mean_us": statistics.mean(times_us),
    "times_us": times_us,
    "beam_style_median_us": statistics.median(beam_style_mins_us),
    "beam_style_min_us": min(beam_style_mins_us),
    "beam_style_max_us": max(beam_style_mins_us),
    "beam_style_mean_us": statistics.mean(beam_style_mins_us),
    "beam_style_min_times_us": beam_style_mins_us,
    "beam_style_all_times_us": beam_style_times_us,
    "programs_per_run": infos,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Measure conv_3x3 with forced TC + UPCAST opts.")
  parser.add_argument("--runs", type=int, default=10, help="Number of measured runs.")
  parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs.")
  parser.add_argument("--regenerate", action="store_true", help="Regenerate and realize fresh inputs for every run.")
  parser.add_argument("--cachelevel", type=int, default=0, help="tinygrad CACHELEVEL value.")
  parser.add_argument("--print-source", action="store_true", help="Print generated source for each measured run.")
  parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
  args = parser.parse_args()

  result = measure(args.runs, args.warmup, args.regenerate, args.cachelevel)
  if args.json:
    print(json.dumps(result))
    return

  print("conv_3x3 forced opts")
  print(f"forced_opts: {result['forced_opts']}")
  print(f"runs: {args.runs}, warmup: {args.warmup}, regenerate: {args.regenerate}")
  print(f"median kernels/run: {result['kernel_count_median']}")
  print(f"median: {result['median_us']:.3f} us")
  print(f"min:    {result['min_us']:.3f} us")
  print(f"max:    {result['max_us']:.3f} us")
  print(f"mean:   {result['mean_us']:.3f} us")
  print("all:    " + ", ".join(f"{t:.3f}" for t in result["times_us"]))
  print()
  print("beam-search-style _time_program min timings")
  print(f"median: {result['beam_style_median_us']:.3f} us")
  print(f"min:    {result['beam_style_min_us']:.3f} us")
  print(f"max:    {result['beam_style_max_us']:.3f} us")
  print(f"mean:   {result['beam_style_mean_us']:.3f} us")
  print("mins:   " + ", ".join(f"{t:.3f}" for t in result["beam_style_min_times_us"]))
  print("all _time_program calls per run:")
  for idx, times in enumerate(result["beam_style_all_times_us"], start=1):
    print(f"  run {idx}: " + ", ".join(f"{t:.3f}" for t in times))
  print("applied opts per run:")
  for idx, programs in enumerate(result["programs_per_run"], start=1):
    print(f"  run {idx}: {[p['applied_opts'] for p in programs]}")

  if args.print_source:
    for run_idx, programs in enumerate(result["programs_per_run"], start=1):
      print(f"\n=== run {run_idx} generated source ===")
      for program_idx, program in enumerate(programs, start=1):
        print(f"--- program {program_idx}: {program['program_name']} ---")
        print(f"global_size={program['global_size']} local_size={program['local_size']}")
        print(f"applied_opts={program['applied_opts']}")
        print(program["source"])


if __name__ == "__main__":
  main()
