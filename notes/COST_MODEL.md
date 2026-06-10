# Learned Cost Model for Beam Search

A LightGBM lambdarank model that predicts kernel execution time rankings without compiling or running kernels. Used to prune beam search candidates before compilation, trading a small quality loss for large speedups in beam search wall time.

## Directory layout

```
collect/
  features.py         # extract_features(scheduler) -> dict
  hook.py             # DataCollector + monkey-patched beam_search for JSONL writing
  run_collection.py   # CLI driver: collect training data across all ops

model/
  encode.py           # encode_features(feat_dict) -> np.ndarray  (stateless)
  dataset.py          # load JSONL + build LightGBM Dataset with groups/labels
  train.py            # CLI: train lambdarank model, save checkpoint
  predict.py          # load_model() -> CostModel; inference interface
  evaluate.py         # CLI: NDCG@K evaluation on a JSONL dataset
  checkpoints/        # saved .lgb model files

data/
  train.jsonl         # 36k+ collected (features, runtime) records
```

---

## Problem statement

`beam_search` finds the best `Opt` sequence by compiling and timing every candidate at each step. For beam width 5 and ~100 actions per step, each step requires ~100 compile+time calls. The cost model prunes candidates **before** compilation so only the top K are compiled and timed.

The harness interface is `cost_model(scheduler) -> float` — lower score = predicted faster. The model is plugged into `experiment/validate/harness.py`'s `run_with_model()`, which installs a `_candidate_filter` that scores all candidates and keeps the top `ceil(beam_width * prune_factor)`.

---

## Data collection (`collect/`)

### `collect/features.py` — `extract_features(s: Scheduler) -> dict`

Extracts a flat dict from a `Scheduler` object before any compilation. The Scheduler's state **already encodes all applied opts** — `full_shape`, `axis_types`, `stride_matrix` all change as opts are applied, so there is no need to encode the opt sequence as a separate feature. The dict is what gets serialized into JSONL and later fed to the encoder.

Key feature groups:

| Group | Fields | Notes |
|---|---|---|
| Shape | `shape_len`, `full_shape`, `axis_types`, `global_size`, `local_size`, `reduce_size`, `warp_size`, `upcast_size` | From Scheduler properties directly |
| Reduction | `has_reduce`, `reduce_op`, `reduce_dtype` | `reduce_op` is `reduceop.arg[0].name` e.g. "ADD"/"MAX" |
| Buffer strides | `stride_matrix`, `buf_bytes` | See stride extraction below |
| Opt state | `upcasted`, `group_for_reduces`, `dont_use_locals` | Applied-opt-derived scalars |
| Op mix | `n_mulacc`, `n_transcendental`, `n_where`, `n_cast` | From `s.ast.toposort()` |
| Device | `device`, `shared_max`, `global_max` | From `s.ren` |
| dtype | `dtype` | First buffer's base dtype name |

**Stride matrix extraction:** For each buffer in `s.bufs`, walks the index expression of `buf.src[1]` to find which loop range (`RANGE` UOp) contributes to the index and with what multiplier. Uses `id(r)` to identify ranges, mirroring `tinygrad/codegen/opt/heuristic.py`'s use of `is` identity checks on UOp objects.

```python
rng_to_idx = {id(r): i for i, r in enumerate(s.rngs)}
# walks ADD/MUL terms in the index expression
```

This produces a `(n_bufs × shape_len)` matrix where each cell is the stride of loop axis `j` in buffer `i`'s index. A zero means that axis doesn't appear in that buffer's address (broadcast). This is the richest structural feature: it distinguishes streaming, broadcast, and strided access patterns per buffer.

**Post-compilation features:** Two additional features are only available after `_try_compile` and are stored at the top level of the JSONL record (not inside `features`):
- `compiled_uops` — `len(prg.src[2].src)` — linearized UOp count, proxy for code complexity
- `flop_estimate` — `sym_infer(prg.src[0].arg.estimates.ops, var_vals)` — estimated FLOPs

These are set to 0 at inference time (no compilation has happened yet).

### `collect/hook.py` — `DataCollector` + `patch_beam_search`

`DataCollector` opens a JSONL file in append mode. `patch_beam_search(collector)` monkey-patches both `tinygrad.codegen.opt.search.beam_search` and `tinygrad.codegen.opt.postrange.beam_search` with a copy of the beam search loop that calls `collector.record(...)` immediately after each `timed.append(...)`:

```python
timed.append((candidates[i], min(tms)))
# --- collection hook ---
collector.record(op_name, kernel_id, beam_step, candidates[i], prg, compile_et, min(tms), var_vals)
```

All tinygrad imports in this file are **inside function bodies** to avoid a circular import:
`codegen.__init__` → `uop.spec` → `schedule.__init__` → `engine.realize` → `codegen.__init__`.

`kernel_id` is `s.ast.key.hex()` — the hex-encoded AST cache key, stable across beam steps for the same kernel.

### `collect/run_collection.py` — CLI driver

```sh
uv run python collect/run_collection.py                     # all ops, beam=5, out=data/train.jsonl
uv run python collect/run_collection.py --ops matmul_1024   # specific ops
uv run python collect/run_collection.py --suite matmuls     # named suite
```

Two `sys.path` fixes at the top are required:

1. `sys.path.insert(0, _repo_root)` — makes `collect` and `benchmarks` importable when run directly.
2. `sys.path.insert(0, _submodule_root)` where `_submodule_root = repo_root/tinygrad/` — prevents the `tinygrad/` directory at the repo root from shadowing the editable install as a namespace package. Python's PathFinder finds `tinygrad/tinygrad/__init__.py` as a regular package first.

Sets `IGNORE_BEAM_CACHE=1` so beam search always runs (never hits the disk cache).

---

## JSONL record schema

One record per timed candidate per beam step:

```json
{
  "op_name": "matmul_4096",
  "kernel_id": "b222b56895c6bfce...",
  "beam_step": 1,
  "features": { ... },
  "compiled_uops": 73,
  "flop_estimate": 137438953472,
  "compile_time_s": 0.042,
  "runtime_s": 0.000318
}
```

`op_name` is for analysis/debugging only — not a model feature. `kernel_id` is not a feature either; it's used to form groups for LambdaRank training. `runtime_s` is the target.

---

## Training objective — LambdaRank

The model does **not** predict absolute runtime. It predicts relative ranking within groups.

**Groups** — one `(kernel_id, beam_step)` tuple = all candidates timed at one beam step for one kernel. Within a group, every candidate has the same kernel structure and differs only in which opts were applied.

**Relevance labels** — within a group of `n` candidates sorted by ascending `runtime_s`:
```
fastest:  label = n - 1
2nd:      label = n - 2
...
slowest:  label = 0
```

**LambdaMART** — LightGBM's `lambdarank` objective computes per-pair gradients weighted by the NDCG change from swapping that pair's positions. The model learns to push faster candidates to the top of the ranking.

**`label_gain`** — set to `[0, 1, 2, ..., max_label]` (linear gains) rather than the default exponential `2^l - 1`. With group sizes up to 80+ (large transformer sub-kernels), exponential gains would make label 80 worth `2^80` — numerically unstable and giving the single fastest candidate overwhelming weight. Linear gains distribute attention more evenly across the top of the ranking.

**Eval metric** — NDCG@K where K = `prune_factor × beam_width` (default 20). If the fastest kernel is in the top-20 after model pruning, beam search will find it.

**Train/val split** — `kernel_split()` in `dataset.py` splits by `kernel_id`, not by record. This means all records for a given kernel stay in the same split. Splitting at a finer granularity (per-group or per-record) would introduce leakage: the model could see some members of a ranking group during training and rank the val members of that group more easily. Per-kernel splitting answers the harder question: *how well does the model generalize to kernel ASTs it has never seen at all?* — which is the actual deployment scenario.

**Note:** per-record splitting would also break NDCG computation, since NDCG requires complete groups.

---

## Feature encoding (`model/encode.py`)

Converts one feature dict to a fixed-length `float32` numpy array. Stateless: all vocabulary lists are fixed constants — no fitted state, no separate encoder file to save.

### Variable-length → fixed-size

| Feature | Fixed size | Transform |
|---|---|---|
| `full_shape` | `MAX_AXES = 8` slots | `log2(x + 1)`; pad with 0 |
| `axis_types` | `MAX_AXES = 8` slots | ordinal (GLOBAL=0 … UPCAST=4); pad with -1 |
| `stride_matrix` | `MAX_BUFS=5 × MAX_AXES=8 = 40` | `log1p(abs(stride))`; pad with 0 |
| `buf_bytes` | 3 scalars | `log1p(total)`, `log1p(max)`, `log1p(mean)` |

### Categoricals

`dtype`, `device`, `reduce_op`, `reduce_dtype` are label-encoded to integers and declared as `categorical_feature` in the `lgb.Dataset` constructor. LightGBM handles categorical splits natively (finds the best subset split rather than treating them as ordinal).

### Full feature vector (82 features)

```
dtype, device, reduce_op, reduce_dtype      (4 categoricals)
has_reduce, dont_use_locals                 (2 booleans)
shape_len, upcasted, group_for_reduces, n_bufs  (4 ints)
log_{global,local,reduce,warp,upcast}_size  (5)
log_n_{mulacc,transcendental,where,cast}    (4)
log_compiled_uops, log_flop_estimate        (2)
log_buf_{total,max,mean}_bytes              (3)
log_{shared,global}_max                    (2)
shape_0..7                                  (8)
axis_type_0..7                              (8)
stride_0_0..stride_4_7                      (40)
```

Total: 82 features.

---

## Training (`model/train.py`)

```sh
uv run python -m model.train --data data/train.jsonl --out model/checkpoints/
uv run python -m model.train --trees 500 --leaves 128 --lr 0.05
```

Saves to `model/checkpoints/model_YYYYMMDD_HHMMSS.lgb`. Prints LightGBM's eval output (every 50 rounds) and final NDCG@5/10/20 on the val set.

Key LightGBM params:
```python
{
  "objective":    "lambdarank",
  "metric":       "ndcg",
  "ndcg_eval_at": [5, 10, 20],
  "label_gain":   list(range(max_label + 1)),
  "num_leaves":   64,
  "learning_rate": 0.05,
  "min_child_samples": 5,
}
```

Results on the full 36k-record dataset (500 trees):
- Val NDCG@10: **0.865** (kernels never seen during training)
- Train NDCG@10: **0.945**

---

## Inference (`model/predict.py`)

```python
from model.predict import load_model

cost_model = load_model()               # auto-discovers latest .lgb in model/checkpoints/
cost_model = load_model("path/to.lgb")  # explicit path
score = cost_model(scheduler)           # float, lower = predicted faster
```

`CostModel.__call__` pipeline:
1. `extract_features(scheduler)` — extracts feature dict (tinygrad import, lazy)
2. `encode_features(feat, compiled_uops=0, flop_estimate=0)` — fixed-size array
3. `booster.predict(x)` — LightGBM inference
4. Negate the score: booster gives higher = better rank; harness interface is lower = faster

---

## Evaluation (`model/evaluate.py`)

```sh
uv run python -m model.evaluate --data data/train.jsonl
uv run python -m model.evaluate --data data/train.jsonl --per-kernel
```

Computes NDCG@K using the same linear-gain formula as training. `--per-kernel` prints a breakdown by `kernel_id` to identify which kernel types the model handles poorly.

---

## Integration with validation harness

`experiment/validate/harness.py` has a `--model` CLI flag. Without `--stub`, it auto-loads the latest checkpoint:

```python
from model.predict import load_model
cost_model = load_model(args.model)   # args.model=None → auto-discover
run_with_model(ops, cost_model=cost_model, ...)
```

The pruning logic in `run_with_model` is:
```python
keep_k = ceil(beam_width * prune_factor)   # default: 5 × 4 = 20

def _filter(candidates):
    scored = [(cost_model(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:keep_k]]

set_candidate_filter(_filter)
```

`cost_model` is called on raw `Scheduler` objects before any compilation. If the model prunes well, far fewer than 100 candidates get compiled, making each beam step much faster.

---

## Caveats and known gaps

**Val score is validation, not test.** The 10% kernel split is monitored during training runs (to decide tree count), so it is a validation set, not a held-out test set. A proper test score requires a separate split created before any training decisions.

**Compiled_uops and flop_estimate are 0 at inference.** These two features are only known post-compilation. Setting them to 0 at inference is a discrepancy from training. If the model relies heavily on them, inference accuracy suffers. They are included as features because they are useful for re-ranking after partial compilation, but a future version might train two models: one without these features (for full pre-compilation pruning) and one with them (for post-compilation re-ranking).

**Group size variation.** Transformer sub-kernels can produce groups of 80+ candidates; elementwise ops may produce groups of 3–5. LightGBM handles this naturally but NDCG numbers are dominated by the large groups.

**No early stopping.** Training uses a fixed tree count. Adding `lgb.early_stopping(50)` to `callbacks` and using the val set for early stopping would be a straightforward improvement once a true held-out test set exists.
