"""Offline NDCG evaluation of a trained cost model.

Usage:
  uv run python -m model.evaluate --data data/train.jsonl
  uv run python -m model.evaluate --model model/checkpoints/model_20240101_120000.lgb --data data/train.jsonl --k 5,10,20
"""

import argparse
from collections import defaultdict
from math import log2

import lightgbm as lgb
import numpy as np

from model.dataset import load_records, build_arrays
from model.predict import load_model


def evaluate(
  model_path: str | None = None,
  data_path: str = 'data/train.jsonl',
  k_values: list[int] | None = None,
  per_kernel: bool = False,
) -> dict[int, float]:
  """Score a dataset and report NDCG@K.

  Args:
    model_path: path to .lgb checkpoint, or None to auto-discover.
    data_path:  JSONL file to evaluate on.
    k_values:   list of K values for NDCG@K (default [5, 10, 20]).
    per_kernel: if True, also print per-kernel breakdowns.

  Returns:
    dict mapping K -> mean NDCG@K
  """
  if k_values is None:
    k_values = [5, 10, 20]

  cost_model = load_model(model_path)
  booster = cost_model._booster

  records = load_records(data_path)
  if not records:
    raise ValueError(f"No records found in {data_path}")

  X, y, group_sizes = build_arrays(records)
  scores = booster.predict(X)

  overall = _ndcg_by_group(scores, y, group_sizes, k_values)

  print(f"Dataset: {data_path}  ({len(records)} records, {len(group_sizes)} groups)")
  print(f"Model:   {cost_model._path}")
  print()
  for k in k_values:
    print(f"  NDCG@{k:<3d}: {overall[k]:.4f}")

  if per_kernel:
    # Re-group by kernel_id for per-kernel breakdown
    from itertools import groupby
    from operator import itemgetter
    sorted_recs = sorted(records, key=lambda r: (r['kernel_id'], r['beam_step']))
    # Build a parallel labels/scores array indexed the same way as build_arrays
    # (build_arrays already sorts by kernel_id, beam_step, so this matches)
    print(f"\n{'kernel_id':>16}  {'groups':>6}  {'NDCG@10':>8}")
    kernel_groups: dict[str, list[int]] = defaultdict(list)
    kernel_labels: dict[str, list[float]] = defaultdict(list)
    kernel_scores: dict[str, list[float]] = defaultdict(list)
    offset = 0
    sorted_recs2 = sorted(records, key=lambda r: (r['kernel_id'], r['beam_step']))
    from itertools import groupby as _groupby
    for (kid, _bstep), grp_iter in _groupby(sorted_recs2, key=lambda r: (r['kernel_id'], r['beam_step'])):
      grp = list(grp_iter)
      g = len(grp)
      kernel_groups[kid].append(g)
      kernel_labels[kid].extend(y[offset:offset + g].tolist())
      kernel_scores[kid].extend(scores[offset:offset + g].tolist())
      offset += g

    for kid in sorted(kernel_groups):
      gsizes = kernel_groups[kid]
      klabels = np.array(kernel_labels[kid])
      kscores = np.array(kernel_scores[kid])
      kndcg = _ndcg_by_group(kscores, klabels, gsizes, [10])
      print(f"  {kid[:16]:>16}  {len(gsizes):>6}  {kndcg[10]:>8.4f}")

  return overall


def _ndcg_by_group(
  scores: np.ndarray,
  labels: np.ndarray,
  group_sizes: list[int],
  k_values: list[int],
) -> dict[int, float]:
  results: dict[int, float] = {}
  for k in k_values:
    ndcg_vals = []
    offset = 0
    for g in group_sizes:
      gs = scores[offset:offset + g]
      gl = labels[offset:offset + g]
      offset += g

      pred_order = np.argsort(-gs)
      ideal_order = np.argsort(-gl)

      def dcg(order):
        total = 0.0
        for i, idx in enumerate(order[:k]):
          total += float(gl[idx]) / log2(i + 2)  # linear gain matches label_gain=[0,1,2,...]
        return total

      idcg = dcg(ideal_order)
      ndcg_vals.append(1.0 if idcg == 0 else dcg(pred_order) / idcg)

    results[k] = float(np.mean(ndcg_vals))
  return results


def main() -> None:
  parser = argparse.ArgumentParser(description='Evaluate cost model NDCG offline')
  parser.add_argument('--model',  default=None,                help='Path to .lgb checkpoint (default: auto-discover)')
  parser.add_argument('--data',   default='data/train.jsonl',  help='JSONL data file')
  parser.add_argument('--k',      default='5,10,20',           help='Comma-separated K values for NDCG@K')
  parser.add_argument('--per-kernel', action='store_true',     help='Show per-kernel NDCG breakdown')
  args = parser.parse_args()

  k_values = [int(x.strip()) for x in args.k.split(',')]
  evaluate(
    model_path=args.model,
    data_path=args.data,
    k_values=k_values,
    per_kernel=args.per_kernel,
  )


if __name__ == '__main__':
  main()
