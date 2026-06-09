"""Monkey-patch beam_search to write (features, runtime) records to a JSONL file.

Usage:
  from collect.hook import DataCollector, patch_beam_search, set_op_name

  collector = DataCollector("data/train.jsonl")
  patch_beam_search(collector)

  set_op_name("matmul_4096")
  op().realize()   # triggers beam_search internally; records are written automatically

  collector.close()
"""

import json
import math
import time
import multiprocessing
import atexit

from tqdm import tqdm

from collect.features import extract_features

# ---------------------------------------------------------------------------
# Current op name — set by the driver before each beam_search call
# ---------------------------------------------------------------------------

_current_op_name: str = '<unknown>'


def set_op_name(name: str) -> None:
    global _current_op_name
    _current_op_name = name


# ---------------------------------------------------------------------------
# DataCollector
# ---------------------------------------------------------------------------


class DataCollector:
    def __init__(self, path: str) -> None:
        self._f = open(path, 'a')  # noqa: SIM115 — intentionally kept open

    def record(
        self,
        op_name: str,
        kernel_id: str,
        beam_step: int,
        scheduler,
        prg,
        compile_time_s: float,
        runtime_s: float,
        var_vals: dict,
    ) -> None:
        from tinygrad.uop.ops import sym_infer

        estimates = prg.src[0].arg.estimates
        rec = {
            'op_name': op_name,
            'kernel_id': kernel_id,
            'beam_step': beam_step,
            'features': extract_features(scheduler),
            'compiled_uops': len(prg.src[2].src),
            'flop_estimate': int(
                sym_infer(estimates.ops if estimates is not None else 0, var_vals)
            ),
            'compile_time_s': compile_time_s,
            'runtime_s': runtime_s,
        }
        self._f.write(json.dumps(rec) + '\n')
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ---------------------------------------------------------------------------
# Collecting beam search — mirrors tinygrad's beam_search with a record() call
# inserted after each candidate is timed.
# ---------------------------------------------------------------------------


def _make_collecting_beam_search(collector: DataCollector):
    def collecting_beam_search(
        s, rawbufs, amt: int, allow_test_size=True, disable_cache=None
    ):
        import tinygrad.codegen.opt.search as _search
        from tinygrad.uop.ops import sym_infer
        from tinygrad.device import Device
        from tinygrad.helpers import (
            flatten,
            DEBUG,
            CACHELEVEL,
            diskcache_get,
            diskcache_put,
            getenv,
            colored,
            time_to_str,
        )
        from tinygrad.helpers import IGNORE_BEAM_CACHE

        if disable_cache is None:
            disable_cache = IGNORE_BEAM_CACHE.value
        op_name = _current_op_name
        kernel_id = (
            s.ast.key.hex()
            if isinstance(s.ast.key, (bytes, bytearray))
            else str(s.ast.key)
        )

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

        beam = [(s, float('inf'))]
        seen_libs: set = set()

        default_parallel = (
            multiprocessing.cpu_count()
            if s.ren.target.device in {'CUDA', 'AMD', 'NV', 'METAL', 'HIP'}
            else 0
        )
        if _search.beam_pool is None and (
            workers := getenv('PARALLEL', default_parallel)
        ):
            _search.beam_pool = multiprocessing.get_context('spawn').Pool(
                workers,
                _search._init_worker,
                (),
                getenv('BEAM_MAX_TASKS_PER_CHILD', 16),
            )

            @atexit.register
            def close_pool():
                _search.beam_pool.close()

        min_progress = getenv('BEAM_MIN_PROGRESS', 0.01) / 1e6
        if _search.BEAM_DEBUG:
            from tinygrad.uop.render import pyrender

            print('BEAM_SEARCH:')
            print(pyrender(s.ast.replace(arg=None)))
        if DEBUG >= 2:
            print(
                f'   0.00s:                from   1 ->   1 actions {s.colored_shape()}'
            )

        try:
            rawbufs = _search._ensure_buffer_alloc(rawbufs)
            var_vals: dict[str, int] = {
                k.expr: int(k.vmax + k.vmin) // 2 for k in s.ast.variables()
            }
            exiting, st = False, time.perf_counter()
            dev = Device[s.ren.target.device]
            beam_step = 0

            while not exiting:
                candidates = flatten(
                    [
                        _search.get_kernel_actions(si, include_0=False).values()
                        for si, _ in beam
                    ]
                )
                timed = []
                least_compute_ops = math.inf

                pbar = tqdm(
                    total=len(candidates),
                    desc=f'{op_name} step {beam_step}',
                    unit='cand',
                    leave=False,
                )
                for i, proc in (
                    map
                    if _search.beam_pool is None
                    else _search.beam_pool.imap_unordered
                )(_search._try_compile, enumerate(candidates)):
                    pbar.update(1)
                    if proc is None:
                        continue
                    prg, compile_et = proc
                    if (lib := prg.src[4].arg) in seen_libs:
                        continue
                    estimates = prg.src[0].arg.estimates
                    least_compute_ops = min(
                        this_compute_ops := sym_infer(
                            estimates.ops if estimates is not None else 0, var_vals
                        ),
                        least_compute_ops,
                    )
                    if least_compute_ops * 1000 < this_compute_ops:
                        if getenv('BEAM_LOG_SURPASS_MAX'):
                            print(
                                f'too much compute. {this_compute_ops} when least is {least_compute_ops}'
                            )
                        continue
                    seen_libs.add(lib)
                    try:
                        tms = _search._time_program(
                            prg,
                            var_vals,
                            rawbufs,
                            early_stop=beam[0][1] * 3 if len(beam) else 1.0,
                            allow_test_size=allow_test_size,
                            clear_l2=hasattr(dev, 'invalidate_caches'),
                            dev_timeout=getenv('BEAM_DEV_TIMEOUT', 1),
                        )
                    except Exception as e:
                        if _search.BEAM_DEBUG:
                            print(
                                f'BEAM failed for opts: {candidates[i].applied_opts}\n{e}'
                            )
                        if isinstance(e, RuntimeError):
                            continue
                        raise
                    timed.append((candidates[i], min(tms)))
                    # --- data collection hook ---
                    collector.record(
                        op_name,
                        kernel_id,
                        beam_step,
                        candidates[i],
                        prg,
                        compile_et,
                        min(tms),
                        var_vals,
                    )
                    # ----------------------------
                    if _search.BEAM_DEBUG > 1:
                        print(
                            f'{time.perf_counter() - st:7.2f}s: {i:5d} {len(prg.src[2].src):5d} uops',
                            f'{time_to_str(compile_et, w=12)} compile/{time_to_str(timed[-1][1], w=12)} run',
                            f'      {len(timed):4d}/{len(candidates):4d}         {timed[-1][0].colored_shape()}',
                        )
                    elif DEBUG >= 2:
                        print(
                            f'\r{time.perf_counter() - st:7.2f}s: {time_to_str(timed[-1][1], w=12)}',
                            f'      {len(timed):4d}/{len(candidates):4d}         {timed[-1][0].colored_shape()}\033[K',
                            end='',
                        )

                pbar.close()
                opts = sorted(timed, key=lambda x: x[1])
                exiting = (
                    len(opts) == 0
                    or (opts[0][1] < min_progress)
                    or (len(beam) > 0 and ((beam[0][1] - opts[0][1]) < min_progress))
                )
                if not exiting:
                    beam = opts[:amt]
                elif len(opts) > 0 and opts[0][1] < beam[0][1]:
                    beam = opts[:1]
                if DEBUG >= 2:
                    print(
                        f'\r{time.perf_counter() - st:7.2f}s:',
                        colored(
                            time_to_str(beam[0][1], w=12), 'green' if exiting else None
                        ),
                        f'from {len(candidates):3d} -> {len(opts):3d} actions\033[K',
                        beam[0][0].colored_shape(),
                    )
                beam_step += 1

        except KeyboardInterrupt as e:
            if _search.beam_pool is not None:
                _search.beam_pool.terminate()
            raise e

        if CACHELEVEL >= 1:
            diskcache_put('beam_search', key, beam[0][0].applied_opts)
        if _search.BEAM_DEBUG:
            print(
                f'BEAM_SEARCH: final tm={time_to_str(beam[0][1], w=0)}, applied_opts={beam[0][0].applied_opts}'
            )
        return beam[0][0]

    return collecting_beam_search


# ---------------------------------------------------------------------------
# Patch / unpatch
# ---------------------------------------------------------------------------

_original_beam_search = None


def patch_beam_search(collector: DataCollector) -> None:
    global _original_beam_search
    if _original_beam_search is not None:
        return  # already patched
    import tinygrad.codegen.opt.search as _search
    import tinygrad.codegen.opt.postrange as _postrange

    _original_beam_search = _search.beam_search
    patched = _make_collecting_beam_search(collector)
    _search.beam_search = patched
    _postrange.beam_search = patched


def unpatch_beam_search() -> None:
    global _original_beam_search
    if _original_beam_search is None:
        return
    import tinygrad.codegen.opt.search as _search
    import tinygrad.codegen.opt.postrange as _postrange

    _search.beam_search = _original_beam_search
    _postrange.beam_search = _original_beam_search
    _original_beam_search = None
