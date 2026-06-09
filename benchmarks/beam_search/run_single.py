#!/usr/bin/env python3
"""Run a single (op, beam_width) benchmark and print JSON to stdout.

Two modes, always called by benchmark.py as separate subprocesses:

  --mode compile  (env must contain IGNORE_BEAM_CACHE=1)
      Times the first realize(), which includes beam search, kernel
      compilation, and one kernel execution.  The beam search result and
      compiled binary are stored in tinygrad's disk cache.

  --mode exec  (env must NOT contain IGNORE_BEAM_CACHE=1)
      Uses the disk cache populated by a prior compile run.  Warm-up
      realizes are used to ensure the compiled binary is loaded, then
      N wall-clock measurements (each followed by a device synchronize)
      give pure kernel execution time.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--op",     required=True)
  parser.add_argument("--beam",   type=int, required=True)
  parser.add_argument("--mode",   choices=["compile", "exec"], required=True)
  parser.add_argument("--n-exec", type=int, default=10)
  parser.add_argument("--n-warmup", type=int, default=2)
  parser.add_argument("--collect-metrics", action="store_true")
  args = parser.parse_args()

  from tinygrad import Context, GlobalCounters, Tensor
  from tinygrad.device import Device
  from tinygrad.engine.realize import run_linear
  from ops import OP_NAMES, get_op

  if args.op not in OP_NAMES:
    print(f"Unknown op '{args.op}'. Available: {list(OP_NAMES)}", file=sys.stderr)
    sys.exit(1)

  op_fn = get_op(args.op)

  def realize_with_metrics():
    GlobalCounters.reset()
    with Context(BEAM=args.beam):
      linear, var_vals = Tensor.linear_with_vars(op_fn())
      run_linear(linear, var_vals, wait=True)
    return {
      "kernel_execution_time_s": GlobalCounters.time_sum_s,
      "total_kernels": GlobalCounters.kernel_count,
    }

  if args.mode == "compile":
    # Time the full beam search + compile + first execution.
    t0 = time.perf_counter()
    extra = realize_with_metrics() if args.collect_metrics else {}
    if not args.collect_metrics:
      with Context(BEAM=args.beam):
        op_fn().realize()
    compile_time = time.perf_counter() - t0
    print(json.dumps({"op": args.op, "beam": args.beam, "mode": "compile",
                      "compile_time_s": compile_time, **extra}))

  else:  # exec
    # Warm up: make sure the compiled program is in memory.
    dev = Device[Device.DEFAULT]
    for _ in range(args.n_warmup):
      if args.collect_metrics: realize_with_metrics()
      else:
        with Context(BEAM=args.beam):
          op_fn().realize()
      dev.synchronize()

    # Timed runs: wall-clock bracketed by device synchronize.
    exec_times: list[float] = []
    kernel_times: list[float] = []
    kernel_counts: list[int] = []
    for _ in range(args.n_exec):
      t0 = time.perf_counter()
      if args.collect_metrics:
        extra = realize_with_metrics()
        kernel_times.append(extra["kernel_execution_time_s"])
        kernel_counts.append(extra["total_kernels"])
      else:
        with Context(BEAM=args.beam):
          op_fn().realize()
      dev.synchronize()
      exec_times.append(time.perf_counter() - t0)

    ret = {
      "op":              args.op,
      "beam":            args.beam,
      "mode":            "exec",
      "exec_time_s":     statistics.median(exec_times),
      "exec_time_min_s": min(exec_times),
      "n_exec":          len(exec_times),
    }
    if args.collect_metrics:
      ret.update({
        "kernel_execution_time_s": statistics.median(kernel_times),
        "kernel_execution_time_min_s": min(kernel_times),
        "total_kernels": int(statistics.median(kernel_counts)),
      })
    print(json.dumps(ret))


if __name__ == "__main__":
  main()
