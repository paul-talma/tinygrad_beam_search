"""Train the LightGBM lambdarank cost model.

Usage:
  uv run python -m model.train --data data/train.jsonl --out model/checkpoints/
  uv run python -m model.train --data data/train.jsonl --trees 500 --leaves 128
"""

import argparse
import datetime
import os
import pathlib

import lightgbm as lgb
import numpy as np

from model.dataset import load_records, kernel_split, build_lgb_dataset, build_arrays
from model.encode import FEATURE_NAMES, CATEGORICAL_FEATURES


def train(
  data_path: str,
  out_dir: str,
  n_estimators: int = 300,
  num_leaves: int = 64,
  learning_rate: float = 0.05,
  min_child_samples: int = 5,
  val_fraction: float = 0.1,
  seed: int = 42,
  verbose: bool = True,
) -> lgb.Booster:
  records = load_records(data_path)
  if not records:
    raise ValueError(f"No records found in {data_path}")

  n_kernels = len({r['kernel_id'] for r in records})
  n_groups = len({(r['kernel_id'], r['beam_step']) for r in records})
  if verbose:
    print(f"Loaded {len(records)} records | {n_kernels} kernels | {n_groups} groups")

  train_recs, val_recs = kernel_split(records, val_fraction=val_fraction, seed=seed)
  if verbose:
    print(f"Train: {len(train_recs)} records, Val: {len(val_recs)} records")

  train_ds = build_lgb_dataset(train_recs)

  # Compute max group size for label_gain (must cover 0..max_label)
  _, y_train, _ = build_arrays(train_recs)
  max_label = int(y_train.max()) if len(y_train) > 0 else 10

  cat_idx = [FEATURE_NAMES.index(c) for c in CATEGORICAL_FEATURES]

  params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 10, 20],
    'label_gain': list(range(max_label + 1)),
    'num_leaves': num_leaves,
    'learning_rate': learning_rate,
    'min_child_samples': min_child_samples,
    'n_jobs': -1,
    'seed': seed,
    'verbose': -1,
    'categorical_feature': cat_idx,
  }

  callbacks = []
  valid_sets = [train_ds]
  valid_names = ['train']

  if val_recs:
    val_ds = build_lgb_dataset(val_recs, reference=train_ds)
    valid_sets.append(val_ds)
    valid_names.append('val')

  if verbose:
    callbacks.append(lgb.log_evaluation(period=50))

  booster = lgb.train(
    params,
    train_ds,
    num_boost_round=n_estimators,
    valid_sets=valid_sets,
    valid_names=valid_names,
    callbacks=callbacks,
  )

  pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
  ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  model_path = os.path.join(out_dir, f'model_{ts}.lgb')
  booster.save_model(model_path)
  if verbose:
    print(f"\nSaved model to {model_path}")

  # Final NDCG on val set
  if val_recs:
    X_val, y_val, grp_val = build_arrays(val_recs)
    scores = booster.predict(X_val)
    ndcg = _compute_ndcg(scores, y_val, grp_val, k_values=[5, 10, 20])
    if verbose:
      for k, v in ndcg.items():
        print(f"  NDCG@{k}: {v:.4f}")

  return booster


def _compute_ndcg(scores: np.ndarray, labels: np.ndarray, group_sizes: list[int], k_values: list[int]) -> dict[int, float]:
  """Compute mean NDCG@K across all groups."""
  from math import log2

  results: dict[int, float] = {}
  for k in k_values:
    ndcg_scores = []
    offset = 0
    for g in group_sizes:
      grp_scores = scores[offset:offset + g]
      grp_labels = labels[offset:offset + g]
      offset += g

      # Sort by descending predicted score
      pred_order = np.argsort(-grp_scores)
      ideal_order = np.argsort(-grp_labels)

      def dcg(order, top_k):
        total = 0.0
        for i, idx in enumerate(order[:top_k]):
          total += (2 ** grp_labels[idx] - 1) / log2(i + 2)
        return total

      idcg = dcg(ideal_order, k)
      if idcg == 0:
        ndcg_scores.append(1.0)
      else:
        ndcg_scores.append(dcg(pred_order, k) / idcg)

    results[k] = float(np.mean(ndcg_scores))
  return results


def main() -> None:
  parser = argparse.ArgumentParser(description='Train LightGBM lambdarank cost model')
  parser.add_argument('--data',   default='data/train.jsonl', help='Training data JSONL path')
  parser.add_argument('--out',    default='model/checkpoints/', help='Checkpoint output directory')
  parser.add_argument('--trees',  type=int, default=300, help='Number of boosting rounds')
  parser.add_argument('--leaves', type=int, default=64, help='LightGBM num_leaves')
  parser.add_argument('--lr',     type=float, default=0.05, help='Learning rate')
  parser.add_argument('--val',    type=float, default=0.1, help='Validation split fraction')
  args = parser.parse_args()

  train(
    data_path=args.data,
    out_dir=args.out,
    n_estimators=args.trees,
    num_leaves=args.leaves,
    learning_rate=args.lr,
    val_fraction=args.val,
  )


if __name__ == '__main__':
  main()
