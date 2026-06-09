"""Load JSONL training data and build a LightGBM Dataset for lambdarank training.

A "group" is one (kernel_id, beam_step) tuple — all candidates timed in one beam
search step for one kernel. Relevance labels are assigned within each group:
  - fastest candidate: label = n - 1
  - slowest candidate: label = 0
"""

import json
from itertools import groupby
from operator import itemgetter

import numpy as np

from model.encode import encode_features, FEATURE_NAMES, CATEGORICAL_FEATURES


def load_records(path: str) -> list[dict]:
  with open(path) as f:
    return [json.loads(line) for line in f if line.strip()]


def build_arrays(
  records: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
  """Build (X, y, group_sizes) arrays suitable for LightGBM lambdarank.

  Records are sorted by (kernel_id, beam_step) so that all candidates
  within one group are contiguous — required by LightGBM's group format.

  Returns:
    X:           float32 array of shape (n_records, n_features)
    y:           float32 relevance labels, non-negative integers
    group_sizes: list of int, one entry per (kernel_id, beam_step) group
  """
  sorted_recs = sorted(records, key=lambda r: (r['kernel_id'], r['beam_step']))

  X_rows: list[np.ndarray] = []
  y_rows: list[float] = []
  group_sizes: list[int] = []

  def _group_key(r):
    return (r['kernel_id'], r['beam_step'])

  for _key, grp_iter in groupby(sorted_recs, key=_group_key):
    grp = list(grp_iter)
    n = len(grp)

    # Assign relevance: rank by ascending runtime, fastest gets label n-1
    order = sorted(range(n), key=lambda i: grp[i]['runtime_s'])
    labels = [0] * n
    for rank, idx in enumerate(order):
      labels[idx] = n - 1 - rank  # fastest=n-1, slowest=0

    for rec, label in zip(grp, labels):
      feat = rec['features']
      row = encode_features(
        feat,
        compiled_uops=rec.get('compiled_uops', 0),
        flop_estimate=rec.get('flop_estimate', 0),
      )
      X_rows.append(row)
      y_rows.append(float(label))

    group_sizes.append(n)

  X = np.stack(X_rows, axis=0).astype(np.float32)
  y = np.array(y_rows, dtype=np.float32)
  return X, y, group_sizes


def kernel_split(
  records: list[dict],
  val_fraction: float = 0.1,
  seed: int = 42,
) -> tuple[list[dict], list[dict]]:
  """Split records into train/val by kernel_id (keeps groups intact).

  All records for a given kernel stay in the same split.
  """
  rng = np.random.default_rng(seed)
  kernel_ids = sorted({r['kernel_id'] for r in records})
  rng.shuffle(kernel_ids)
  n_val = max(1, int(len(kernel_ids) * val_fraction))
  val_ids = set(kernel_ids[:n_val])
  train = [r for r in records if r['kernel_id'] not in val_ids]
  val   = [r for r in records if r['kernel_id'] in val_ids]
  return train, val


def build_lgb_dataset(records: list[dict], reference=None):
  """Build a lgb.Dataset from records. Requires lightgbm installed."""
  import lightgbm as lgb

  X, y, group_sizes = build_arrays(records)
  cat_idx = [FEATURE_NAMES.index(c) for c in CATEGORICAL_FEATURES]
  ds = lgb.Dataset(
    X, label=y,
    feature_name=FEATURE_NAMES,
    categorical_feature=cat_idx,
    reference=reference,
    free_raw_data=False,
  )
  ds.set_group(group_sizes)
  return ds
