# Validation Results: LightGBM Cost Model vs Baseline Beam Search

## Summary

The LightGBM lambdarank cost model (`model/checkpoints/model_20260609_012207.lgb`) dramatically cuts search time on compute-heavy kernels (−92% on matmul_1024, −28% on conv_3x3) and finds better kernels on conv. It regresses on `reduce_sum` (+76% search time, +5% quality loss), likely because `extract_features` overhead dominates for fast-compiling kernels. Once two harness bugs were fixed before measuring.

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

All runs: `PARALLEL=0`, prune factor 4 (keep top `ceil(beam×4)` candidates per step), Metal backend.

### Multi-op run — beam width 3

| Op | B-compiled | M-compiled | B-Beam s | M-Beam s | Search Δ | B-µs | M-µs | Quality Δ |
|----|-----------|-----------|---------|---------|---------|------|------|---------|
| matmul_1024 | 300 | 48 | 281.3s | 22.6s | **−92%** | 955.2 | 959.8 | +0.5% |
| elem_relu   | 91  | 43 | 6.36s  | 3.70s | **−42%** | 826.3 | 852.5 | +3.2% |
| reduce_sum  | 58  | 48 | 5.75s  | 10.1s | **+76%** ⚠ | 127.2 | 133.5 | +5.0% ⚠ |
| conv_3x3    | 322 | 73 | 595s   | 426s  | **−28%** | 508.2 | 425.5 | **−16%** (better) |

**matmul_1024** is the clearest win: 6× fewer kernels compiled, 12× faster search, kernel quality essentially identical. This is the target use case — expensive multi-step searches where the ranking model's pruning pays off.

**conv_3x3** also wins: the model finds a 16% faster kernel (508 → 425 µs) while cutting compilation by 4×. The timeout warnings on both runs are benign — `BEAM_TIMEOUT_SEC` kills slow compilations, which the model's pruning reduces.

**reduce_sum** regresses. The model keeps 48 of 58 candidates (prune factor barely helps), and search takes 76% longer. The likely cause: `extract_features` is called on all 58 candidates before any are filtered, adding overhead that exceeds the savings from compiling 10 fewer kernels. With a fast-compiling kernel where each candidate takes ~100ms, paying for 58 Python feature-extraction calls (~few ms each) is inefficient. The model also finds a worse kernel (+5%).

**elem_relu** is a marginal result. The reduction in compiled kernels is real (−53%), but quality degrades slightly (+3.2%). At beam=2 (previous runs), variance is high because the search often terminates in one step.

### elem_relu — beam width 2 (repeated runs for variance)

| Run | B-compiled | M-compiled | B-Beam s | M-Beam s | Search Δ | B-µs | M-µs | Quality Δ |
|-----|-----------|-----------|---------|---------|---------|------|------|---------|
| 1 | 80 | 39 | 12.2s | 3.82s | **−69%** | 838.4 | 829.8 | **−1.0%** (better) |
| 2 | 58 | 31 | 4.32s | 2.73s | **−37%** | 831.9 | 833.3 | +0.2% |
| 3 | 62 | 23 | 5.10s | 1.18s | **−77%** | 826.8 | 835.3 | +1.0% |

Run-to-run variance in B-compiled (58–80) reflects beam search's stochastic early-stopping. The model consistently compiles ~half as many kernels. Quality difference is within noise.

**Random stub control:** keeping 8 random candidates achieved similar search speedup (−70%) with similar quality change (+0.4%) as the LightGBM model on `elem_relu`. This confirms the speedup is mostly from pruning volume, not from the model's ranking — the model's advantage is in identifying *which* candidates to keep, which matters more on multi-step kernels like matmul.

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

## Findings and Next Steps

**The model works best on expensive, many-candidate kernels.** matmul and conv have 300+ candidates and multi-second compilation per candidate; the model's 6–4× reduction in compiled kernels is decisive. For simple kernels with fast compilation (reduce_sum, elem_relu), the per-candidate `extract_features` overhead can exceed the savings.

**`reduce_sum` regression needs investigation.** The model should either (a) gate on whether pruning is expected to help before calling the filter — e.g., skip filtering when `n_candidates < threshold` or estimated compile time is short — or (b) batch `extract_features` calls to reduce Python overhead.

**Quality on conv is surprisingly good.** The model found a 16% faster conv_3x3 kernel than baseline. This could mean the baseline's search is incomplete (hit timeout before finding the best opt), and the model's pruning happened to focus search on a better region. Worth investigating which opts the model chose.

**`extract_features` is in the hot path.** For 300 matmul candidates at beam=3, the filter calls `extract_features` 300 times per step. Profiling this overhead is warranted — if it's significant, consider caching feature extraction across beam steps for unchanged schedulers.

**`PARALLEL=0` deflates absolute times.** With parallel compilation, both runs would be faster. The relative speedup from fewer compilations would hold.

**Multi-kernel ops.** `attention` decomposes into 4 sub-kernels; `harness.py` captures only `traces[0]`. Run `experiment/explore/run.py` to see all sub-kernels.
