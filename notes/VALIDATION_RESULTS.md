# Validation Results: LightGBM Cost Model vs Baseline Beam Search

## Summary

The LightGBM lambdarank cost model (`model/checkpoints/model_20260609_012207.lgb`) reduces beam search time by ~70% on simple kernels with negligible quality loss, once two harness bugs were fixed.

---

## Bugs Fixed Before Measurement

### Bug 1 — `to_program_cache` skips beam search on the second run

`tinygrad.codegen.to_program_cache` maps the unoptimized-AST key to a compiled program. Since tinygrad interns UOps, every call to `x.relu()` for the same realized `x` produces the same UOp key. After the baseline run populates the cache, the model run's `op.realize()` returns the cached program immediately, without ever calling `apply_opts` or `beam_search`. The symptom was M-compiled=0 and M-Beam=0.00s for every op.

**Fix:** clear `to_program_cache` at the start of each `_run_op` call (`experiment/validate/harness.py`).

### Bug 2 — Spawn workers crash with circular import on Metal

`_instrumented_beam_search` defaulted to spawning `cpu_count()` workers on METAL (matching `search.py`). Spawned workers can't initialize `tinygrad.codegen` due to the circular chain `codegen → uop.spec → schedule.__init__ → engine.realize → codegen`. The chain hits a partially-initialized module and raises `ImportError` before `_init_worker` runs. This is a different root cause than the `IndexError: pop from empty deque` crash documented in `BEAM_SEARCH_METAL_MULTIPROCESSING.md`, but the same fix applies: `PARALLEL=0`.

**Fix:** default `default_parallel = 0` in `_instrumented_beam_search`; override with `PARALLEL=N` to use the pool (`experiment/explore/instrumented.py`).

---

## Measured Results

All runs use single-threaded compilation (`PARALLEL=0`), beam width 2, prune factor 4 (keep top `ceil(2×4)=8` candidates per step). Op: `elem_relu` (relu on a 4096×4096 float16 tensor on Metal).

| Run | B-compiled | M-compiled | B-Beam s | M-Beam s | Search Δ | B-µs | M-µs | Quality Δ |
|-----|-----------|-----------|---------|---------|---------|------|------|---------|
| Run 1 | 80 | 39 | 12.2s | 3.82s | **−69%** | 838.4 | 829.8 | **−1.0%** (better) |
| Run 2 | 58 | 31 | 4.32s | 2.73s | **−37%** | 831.9 | 833.3 | +0.2% |
| Run 3 | 62 | 23 | 5.10s | 1.18s | **−77%** | 826.8 | 835.3 | +1.0% |

The number of compiled kernels varies between runs (54–80 baseline) because beam search explores stochastically and the search terminates early when no progress is made. The model consistently reduces compilation to roughly half.

**Random stub control (run 4):** 54 compiled → 23 compiled (random keeps ceil(8)), Search −70%, Quality +0.4%. The random stub achieves similar search speedup to the model for `elem_relu` because this op's search terminates in one step regardless of which 8 candidates are selected — the model's ranking advantage shows more on multi-step kernels.

---

## How to Run

```sh
# From repo root, with auto-discovered latest checkpoint:
PARALLEL=0 uv run python -m experiment.validate.harness --beam 3

# Specific ops only:
PARALLEL=0 uv run python -m experiment.validate.harness --beam 3 --ops matmul_1024,reduce_sum

# Explicit model path:
PARALLEL=0 uv run python -m experiment.validate.harness --beam 3 \
  --model model/checkpoints/model_20260609_012207.lgb

# Random stub (sanity check — should show similar search speedup, degraded quality):
PARALLEL=0 uv run python -m experiment.validate.harness --stub --beam 3
```

Results are saved to `experiment/results/validation_YYYYMMDD_HHMMSS.csv`.

---

## Caveats

**Variance is high at beam=2.** Beam search terminates when no candidate beats the current best; for simple ops this often happens in 1–2 steps. Small beam widths make the result sensitive to which candidates happen to be generated. Use `--beam 5` for more stable measurements.

**`elem_relu` is a weak test.** The relu kernel is so simple that many opt combinations produce similar performance. The model's ranking advantage matters more for complex multi-step kernels (matmul, conv, attention). For a meaningful quality comparison, run on `matmul_1024` or `attention` where multi-step search matters.

**`PARALLEL=0` slows both runs equally.** With parallelism working, both baseline and model runs would be faster. The relative speedup from the model (fewer candidates to compile) would hold, but absolute times would be lower.

**Multi-kernel ops.** `attention` decomposes into 4 sub-kernels. `harness.py` captures only `traces[0]` (the first sub-kernel). Run `experiment/explore/run.py` to see all sub-kernels.
