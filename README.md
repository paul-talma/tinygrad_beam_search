# Tinygrad Beam Search — Learned Cost Model

A learned cost model that predicts kernel runtime to speed up tinygrad's `beam_search` optimizer,
replacing exhaustive benchmarking with model-guided pruning. Inspired by TVM/Ansor.

## Setup

```sh
uv sync
```

## Project layout

```
collect/          — data collection: run ops under instrumented beam search, dump JSONL
data/             — training data (JSONL)
experiment/
  explore/        — instrumented beam_search hook + op definitions
  validate/       — harness (runs baseline vs model) and report script
  results/        — CSV outputs from validation runs
model/            — feature encoding, training (LightGBM), and inference
tinygrad/         — vendored tinygrad (modified)
```

## Workflow

### 1. Collect training data

```sh
uv run python -m collect.run
```

Runs each op under instrumented beam search and appends feature/label rows to `data/v2/combined.jsonl`.

### 2. Train the cost model

```sh
uv run python -m model.train
```

Trains a LightGBM ranker on the collected data. Saves a `.lgb` checkpoint to `model/checkpoints/`.

### 3. Validate

Run the validation harness to compare baseline beam search against model-guided beam search:

```sh
# All ops, auto-load latest model checkpoint
uv run python -m experiment.validate.harness

# Specific ops only
uv run python -m experiment.validate.harness --ops matmul,conv3x3

# Re-use a previous baseline CSV (skip re-running baseline)
uv run python -m experiment.validate.harness --baseline-csv experiment/results/validation_20260613_193940.csv

# Use a random-stub model to sanity-check harness plumbing
uv run python -m experiment.validate.harness --stub

# Key flags
#   --beam N           beam width (default: 5)
#   --prune N          prune factor — keep top ceil(beam*prune) candidates (default: 4)
#   --model PATH       explicit .lgb checkpoint path
#   --training-data F  JSONL to tag seen/unseen kernels (default: data/v2/combined.jsonl)
#   --model-test-size  use estimated timing for model run (matches baseline mode)
```

Results are saved to `experiment/results/validation_<timestamp>.csv`.

### 4. Summarise results

```sh
# Latest CSV (auto-detected)
uv run python -m experiment.validate.report

# Specific CSV
uv run python -m experiment.validate.report experiment/results/validation_20260613_193940.csv

# Merge two CSVs (e.g. baseline from one run, model from another)
uv run python -m experiment.validate.report baseline.csv model.csv

# LaTeX table (ready to paste into a paper)
uv run python -m experiment.validate.report --latex
```

The report prints one row per op with eight timing columns and three quality-delta columns:

| Column | Meaning |
|--------|---------|
| Heur µs | Heuristic optimizer only — no beam search |
| B-srch s | Baseline beam search wall time |
| B-kern µs | Baseline winner re-timed at full size |
| F-srch s | Model search time (`allow_test_size=False`, full-size timing) |
| F-kern µs | Model-Full winner re-timed at full size |
| E-srch s | Model search time (`allow_test_size=True`, estimated timing) |
| E-kern µs | Model-Est winner re-timed at full size |
| B→F Δ | Kernel quality change: Model-Full vs Baseline (negative = better) |
| B→E Δ | Kernel quality change: Model-Est vs Baseline |
| F vs E | Kernel quality: Model-Full vs Model-Est |

## Key environment variables

| Variable | Effect |
|----------|--------|
| `BEAM=N` | Beam width (overrides tinygrad default) |
| `DEBUG=4` | Print generated kernel source |
| `NOOPT=1` | Disable all kernel optimizations |
