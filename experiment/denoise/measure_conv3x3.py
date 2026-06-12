"""Measure execution time for tinygrad benchmark kernels.

Examples:
  python -m experiment.denoise.measure_conv3x3
  python -m experiment.denoise.measure_conv3x3 --op conv_5x5 --beam 3 --warmup 3 --runs 20
  python -m experiment.denoise.measure_conv3x3 --op matmul --beam 3
  python -m experiment.denoise.measure_conv3x3 --op elementwise --runs 20
  python -m experiment.denoise.measure_conv3x3 --op softmax --beam 5
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import multiprocessing
import os
import time
import sys
import statistics
from pathlib import Path

_repo_root = Path(__file__).parents[2]
_tinygrad_root = _repo_root / 'tinygrad'
if str(_tinygrad_root) not in sys.path:
    sys.path.insert(0, str(_tinygrad_root))
if 'DEBUG' in os.environ:
    try:
        int(os.environ['DEBUG'])
    except ValueError:
        os.environ.pop('DEBUG')

from tinygrad import GlobalCounters, Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import compile_linear, run_linear
from tinygrad.helpers import Context, flatten
from tinygrad.uop import Ops

_beam_trace_enabled = False
_beam_traces: list[dict] = []
_original_beam_search = None


def normalize_opts(applied_opts) -> list[tuple[str, int | None, object]]:
    return [
        (getattr(opt.op, 'name', str(opt.op)), opt.axis, opt.arg)
        for opt in applied_opts
    ]


ALL_OPS = (
    'conv_3x3',
    'conv_5x5',
    'matmul',
    'matmul_batched',
    'elementwise',
    'reduce',
    'softmax',
)


def make_op(op: str, cachelevel: int):
    with Context(CACHELEVEL=cachelevel):
        if op == 'conv_3x3':
            x = Tensor.randn(4, 64, 32, 32, dtype=dtypes.float16).realize()
            w = Tensor.randn(128, 64, 3, 3, dtype=dtypes.float16).realize()
            return lambda: x.conv2d(w, padding=1)
        if op == 'conv_5x5':
            x = Tensor.randn(4, 32, 32, 32, dtype=dtypes.float16).realize()
            w = Tensor.randn(64, 32, 5, 5, dtype=dtypes.float16).realize()
            return lambda: x.conv2d(w, padding=2)
        if op == 'matmul':
            # (M=256, K=1024) @ (K=1024, N=1024) — single compute-heavy matmul
            a = Tensor.randn(256, 1024, dtype=dtypes.float16).realize()
            b = Tensor.randn(1024, 1024, dtype=dtypes.float16).realize()
            return lambda: a @ b
        if op == 'matmul_batched':
            # (B=8, M=64, K=1024) @ (B=8, K=1024, N=1024) — batched, attention-like
            a = Tensor.randn(8, 64, 1024, dtype=dtypes.float16).realize()
            b = Tensor.randn(8, 1024, 1024, dtype=dtypes.float16).realize()
            return lambda: a @ b
        if op == 'elementwise':
            # fused multiply-add + relu: exercises vectorization and loop fusion
            a = Tensor.randn(4, 128, 256, 256, dtype=dtypes.float16).realize()
            b = Tensor.randn(4, 128, 256, 256, dtype=dtypes.float16).realize()
            return lambda: (a * b + a).relu()
        if op == 'reduce':
            # sum reduction over the last axis
            a = Tensor.randn(4, 128, 1024, dtype=dtypes.float16).realize()
            return lambda: a.sum(axis=-1)
        if op == 'softmax':
            # softmax: reduce max → subtract → exp → reduce sum → divide
            a = Tensor.randn(4, 32, 1024, dtype=dtypes.float16).realize()
            return lambda: a.softmax(axis=-1)
    raise ValueError(f'unknown op {op!r}')


def install_beam_trace() -> None:
    """Patch beam_search in this process to record per-step candidate timings."""
    global _original_beam_search
    if _original_beam_search is not None:
        return
    import tinygrad.codegen.opt.search as search_mod

    _original_beam_search = search_mod.beam_search
    search_mod.beam_search = traced_beam_search


def pop_beam_traces() -> list[dict]:
    global _beam_traces
    out, _beam_traces = _beam_traces, []
    return out


def traced_beam_search(s, rawbufs, amt: int, allow_test_size=True, disable_cache=None):
    import tinygrad.codegen.opt.search as search_mod
    from tinygrad.device import Device
    from tinygrad.helpers import (
        CACHELEVEL,
        IGNORE_BEAM_CACHE,
        diskcache_get,
        diskcache_put,
        getenv,
    )
    from tinygrad.uop.ops import sym_infer

    if disable_cache is None:
        disable_cache = IGNORE_BEAM_CACHE.value

    key = {
        'ast': s.ast.key,
        'amt': amt,
        'allow_test_size': allow_test_size,
        'device': s.ren.target.device,
        'suffix': s.ren.suffix,
    }
    if (
        not disable_cache
        and CACHELEVEL >= 1
        and (val := diskcache_get('beam_search', key)) is not None
    ):
        ret = s.copy()
        for o in val[len(s.applied_opts) :]:
            ret.apply_opt(o)
        return ret

    trace = {
        'kernel_id': s.ast.key.hex()
        if isinstance(s.ast.key, (bytes, bytearray))
        else str(s.ast.key),
        'device': s.ren.target.device,
        'steps': [],
        'final_runtime_s': math.inf,
        'final_applied_opts': [],
    }
    _beam_traces.append(trace)

    beam = [(s, float('inf'))]
    seen_libs = set()

    default_parallel = (
        multiprocessing.cpu_count()
        if s.ren.target.device in {'CUDA', 'AMD', 'NV', 'METAL', 'HIP'}
        else 0
    )
    if search_mod.beam_pool is None and (
        workers := getenv('PARALLEL', default_parallel)
    ):
        search_mod.beam_pool = multiprocessing.get_context('spawn').Pool(
            workers, search_mod._init_worker, (), getenv('BEAM_MAX_TASKS_PER_CHILD', 16)
        )

        @atexit.register
        def close_pool():
            search_mod.beam_pool.close()

    min_progress = getenv('BEAM_MIN_PROGRESS', 0.01) / 1e6

    try:
        rawbufs = search_mod._ensure_buffer_alloc(rawbufs)
        var_vals: dict[str, int] = {
            k.expr: int(k.vmax + k.vmin) // 2 for k in s.ast.variables()
        }
        dev = Device[s.ren.target.device]
        exiting = False
        step_idx = 0

        while not exiting:
            candidates = flatten(
                [
                    search_mod.get_kernel_actions(si, include_0=False).values()
                    for si, _ in beam
                ]
            )
            timed = []
            step_records = []
            least_compute_ops = math.inf

            pool_map = (
                search_mod.beam_pool.imap_unordered
                if search_mod.beam_pool is not None
                else map
            )
            for i, proc in pool_map(search_mod._try_compile, enumerate(candidates)):
                if proc is None:
                    continue
                prg, compile_et = proc
                if (lib := prg.src[4].arg) in seen_libs:
                    continue
                estimates = prg.src[0].arg.estimates
                this_compute_ops = sym_infer(
                    estimates.ops if estimates is not None else 0, var_vals
                )
                least_compute_ops = min(this_compute_ops, least_compute_ops)
                if least_compute_ops * 1000 < this_compute_ops:
                    continue
                seen_libs.add(lib)
                try:
                    tms = search_mod._time_program(
                        prg,
                        var_vals,
                        rawbufs,
                        early_stop=beam[0][1] * 3 if beam else 1.0,
                        allow_test_size=allow_test_size,
                        clear_l2=hasattr(dev, 'invalidate_caches'),
                        dev_timeout=getenv('BEAM_DEV_TIMEOUT', 1),
                    )
                except Exception as e:
                    if isinstance(e, RuntimeError):
                        continue
                    raise

                runtime_s = min(tms)
                timed.append((candidates[i], runtime_s))
                step_records.append(
                    {
                        'runtime_s': runtime_s,
                        'compile_time_s': compile_et,
                        'applied_opts': normalize_opts(candidates[i].applied_opts),
                        'colored_shape': candidates[i].colored_shape(),
                        'compiled_uops': len(prg.src[2].src),
                    }
                )

            opts = sorted(timed, key=lambda x: x[1])
            top_records = sorted(step_records, key=lambda x: x['runtime_s'])
            exiting = (
                len(opts) == 0
                or (opts[0][1] < min_progress)
                or (len(beam) > 0 and ((beam[0][1] - opts[0][1]) < min_progress))
            )
            if not exiting:
                beam = opts[:amt]
            elif len(opts) > 0 and opts[0][1] < beam[0][1]:
                beam = opts[:1]

            trace['steps'].append(
                {
                    'step': step_idx,
                    'candidates_generated': len(candidates),
                    'candidates_timed': len(step_records),
                    'top_candidates': top_records,
                    'selected_beam': [
                        {
                            'runtime_s': runtime_s,
                            'applied_opts': normalize_opts(scheduler.applied_opts),
                        }
                        for scheduler, runtime_s in beam[:amt]
                    ],
                }
            )
            step_idx += 1

    except KeyboardInterrupt as e:
        if search_mod.beam_pool is not None:
            search_mod.beam_pool.terminate()
        raise e

    if CACHELEVEL >= 1:
        diskcache_put('beam_search', key, beam[0][0].applied_opts)
    trace['final_runtime_s'] = beam[0][1]
    trace['final_applied_opts'] = normalize_opts(beam[0][0].applied_opts)
    return beam[0][0]


def kernel_signature(compiled_linear) -> tuple[str, list[dict], list[dict]]:
    """Fingerprint generated program source, launch metadata, and applied opts.

    Auto-generated function names can differ across runs even when the kernel body
    and optimization choices are identical, so names are normalized out.
    """
    programs = []
    sources = []
    for program in [u for u in compiled_linear.toposort() if u.op is Ops.PROGRAM]:
        kernel_info = (
            program.src[0].arg
            if program.src and program.src[0].op is Ops.SINK
            else None
        )
        applied_opts = tuple(getattr(kernel_info, 'applied_opts', ()))
        normalized_opts = [
            (getattr(opt.op, 'name', str(opt.op)), opt.axis, opt.arg)
            for opt in applied_opts
        ]
        source = next((u.arg for u in program.src if u.op is Ops.SOURCE), '')
        sources.append(
            {
                'program_name': getattr(program.arg, 'name', ''),
                'global_size': tuple(getattr(program.arg, 'global_size', ())),
                'local_size': tuple(getattr(program.arg, 'local_size', ())),
                'applied_opts': normalized_opts,
                'source': source,
            }
        )
        program_name = getattr(program.arg, 'name', '')
        kernel_name = getattr(kernel_info, 'name', '')
        for name in {program_name, kernel_name} - {''}:
            source = source.replace(name, '<kernel>')
        programs.append(
            {
                'global_size': tuple(getattr(program.arg, 'global_size', ())),
                'local_size': tuple(getattr(program.arg, 'local_size', ())),
                'applied_opts': normalized_opts,
                'source': source,
            }
        )
    payload = json.dumps(programs, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16], programs, sources


def run_once(
    op_fn, beam: int, cachelevel: int
) -> tuple[float, int, str, list[dict], list[dict]]:
    """Return timing, kernel count, signature, generated sources, and beam traces for one realize."""
    import tinygrad.codegen as codegen

    pop_beam_traces()
    codegen.to_program_cache.clear()
    GlobalCounters.reset()
    with Context(BEAM=beam, CACHELEVEL=cachelevel):
        linear, var_vals = Tensor.linear_with_vars(op_fn())
        compiled_linear = compile_linear(linear)
        sig, _programs, sources = kernel_signature(compiled_linear)
        run_linear(compiled_linear, var_vals, wait=True, jit=True)
    Device[Device.DEFAULT].synchronize()
    return (
        GlobalCounters.time_sum_s * 1e6,
        GlobalCounters.kernel_count,
        sig,
        sources,
        pop_beam_traces(),
    )


def _run_with_optional_regeneration(
    op_name: str, op_fn, regenerate: bool, beam: int, cachelevel: int
) -> tuple[float, int, str, list[dict], list[dict]]:
    if regenerate:
        op_fn = make_op(op_name, cachelevel)
    return run_once(op_fn, beam, cachelevel)


def measure(
    op: str,
    beam: int,
    warmup: int,
    runs: int,
    cachelevel: int,
    regenerate: bool,
    include_sources: bool,
    include_beam_exec_time: bool,
) -> dict:
    if include_beam_exec_time:
        install_beam_trace()
    op_fn = None if regenerate else make_op(op, cachelevel)
    for _ in range(warmup):
        _run_with_optional_regeneration(op, op_fn, regenerate, beam, cachelevel)

    times_us: list[float] = []
    kernel_counts: list[int] = []
    kernel_signatures: list[str] = []
    applied_opts_per_run: list[list[list[tuple[str, int | None, int | None]]]] = []
    generated_sources: list[list[dict]] = []
    beam_traces_per_run: list[list[dict]] = []
    for _ in range(runs):
        t_us, kernel_count, sig, sources, beam_traces = _run_with_optional_regeneration(
            op, op_fn, regenerate, beam, cachelevel
        )
        times_us.append(t_us)
        kernel_counts.append(kernel_count)
        kernel_signatures.append(sig)
        applied_opts_per_run.append([src['applied_opts'] for src in sources])
        if include_sources:
            generated_sources.append(sources)
        if include_beam_exec_time:
            beam_traces_per_run.append(beam_traces)

    signature_counts = {
        sig: kernel_signatures.count(sig) for sig in sorted(set(kernel_signatures))
    }
    opt_sequence_keys = [json.dumps(opts) for opts in applied_opts_per_run]
    opt_sequence_counts = {
        key: opt_sequence_keys.count(key) for key in sorted(set(opt_sequence_keys))
    }

    result = {
        'op': op,
        'beam': beam,
        'warmup': warmup,
        'runs': runs,
        'cachelevel': cachelevel,
        'regenerate': regenerate,
        'kernel_count_median': statistics.median(kernel_counts),
        'unique_generated_kernels': len(signature_counts),
        'kernel_signature_counts': signature_counts,
        'kernel_signatures': kernel_signatures,
        'unique_applied_opt_sequences': len(opt_sequence_counts),
        'applied_opt_sequence_counts': opt_sequence_counts,
        'applied_opts_per_run': applied_opts_per_run,
        'median_us': statistics.median(times_us),
        'min_us': min(times_us),
        'max_us': max(times_us),
        'mean_us': statistics.mean(times_us),
        'times_us': times_us,
    }
    if include_sources:
        result['generated_sources'] = generated_sources
    if include_beam_exec_time:
        result['beam_traces_per_run'] = beam_traces_per_run
    return result


def print_beam_exec_time_traces(result: dict, top_k: int) -> None:
    for run_idx, traces in enumerate(result.get('beam_traces_per_run', []), start=1):
        print(f'\n=== run {run_idx} beam-search candidate timings ===')
        if not traces:
            print('  no beam_search trace captured')
            continue
        for kernel_idx, trace in enumerate(traces, start=1):
            print(
                f'\n  kernel {kernel_idx}: id={trace["kernel_id"]} device={trace["device"]}'
            )
            print(
                f'  final: {trace["final_runtime_s"] * 1e6:.3f} us opts={trace["final_applied_opts"]}'
            )
            for step in trace['steps']:
                print(
                    f'    step {step["step"]}: generated={step["candidates_generated"]} '
                    f'timed={step["candidates_timed"]} top {min(top_k, len(step["top_candidates"]))}'
                )
                for rank, cand in enumerate(step['top_candidates'][:top_k], start=1):
                    print(
                        f'      {rank:2d}. exec={cand["runtime_s"] * 1e6:9.3f} us '
                        f'compile={cand["compile_time_s"] * 1e3:8.3f} ms '
                        f'uops={cand["compiled_uops"]:4d} opts={cand["applied_opts"]}'
                    )
                if step['selected_beam']:
                    print('      selected beam:')
                    for rank, cand in enumerate(step['selected_beam'], start=1):
                        print(
                            f'        {rank:2d}. exec={cand["runtime_s"] * 1e6:9.3f} us opts={cand["applied_opts"]}'
                        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure convolution kernel execution time.'
    )
    parser.add_argument(
        '--op', choices=ALL_OPS, default='conv_3x3', help='Kernel benchmark to measure.'
    )
    parser.add_argument(
        '--beam',
        type=int,
        default=0,
        help='tinygrad BEAM value. 0 means no beam search.',
    )
    parser.add_argument(
        '--warmup', type=int, default=3, help='Number of warmup runs before measuring.'
    )
    parser.add_argument('--runs', type=int, default=10, help='Number of measured runs.')
    parser.add_argument(
        '--cachelevel', type=int, default=0, help='tinygrad CACHELEVEL value.'
    )
    parser.add_argument(
        '--regenerate',
        action='store_true',
        help='Regenerate and realize fresh input tensors for every run.',
    )
    parser.add_argument(
        '--print-source',
        action='store_true',
        help='Print generated source for each measured run.',
    )
    parser.add_argument(
        '--beam_exec_time',
        action='store_true',
        help='Print per-step beam-search candidate execution times.',
    )
    parser.add_argument(
        '--json', action='store_true', help='Print machine-readable JSON only.'
    )
    args = parser.parse_args()

    result = measure(
        op=args.op,
        beam=args.beam,
        warmup=args.warmup,
        runs=args.runs,
        cachelevel=args.cachelevel,
        regenerate=args.regenerate,
        include_sources=args.print_source,
        include_beam_exec_time=args.beam_exec_time,
    )
    if args.json:
        print(json.dumps(result))
        return

    print(f'{args.op} kernel execution time (BEAM={args.beam})')
    print(
        f'runs: {args.runs}, warmup: {args.warmup}, regenerate: {args.regenerate}, median kernels/run: {result["kernel_count_median"]}'
    )
    print(f'unique generated kernels: {result["unique_generated_kernels"]}')
    print(f'kernel signatures: {result["kernel_signature_counts"]}')
    print(f'unique applied opt sequences: {result["unique_applied_opt_sequences"]}')
    print('applied opts per run:')
    for run_idx, opts in enumerate(result['applied_opts_per_run'], start=1):
        print(f'  run {run_idx}: {opts}')
    print(f'median: {result["median_us"]:.3f} us')
    print(f'min:    {result["min_us"]:.3f} us')
    print(f'max:    {result["max_us"]:.3f} us')
    print(f'mean:   {result["mean_us"]:.3f} us')
    print('all:    ' + ', '.join(f'{t:.3f}' for t in result['times_us']))
    if args.beam_exec_time:
        print_beam_exec_time_traces(result, top_k=max(1, args.beam))
    if args.print_source:
        for run_idx, sources in enumerate(result['generated_sources'], start=1):
            print(f'\n=== run {run_idx} generated source ===')
            for program_idx, src in enumerate(sources, start=1):
                print(f'--- program {program_idx}: {src["program_name"]} ---')
                print(
                    f'global_size={src["global_size"]} local_size={src["local_size"]}'
                )
                print(f'applied_opts={src["applied_opts"]}')
                print(src['source'])


if __name__ == '__main__':
    main()
