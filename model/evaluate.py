"""Offline NDCG evaluation of a trained cost model.

Usage:
  uv run python -m model.evaluate --data data/train.jsonl
  uv run python -m model.evaluate --model model/checkpoints/model_20240101_120000.lgb --data data/train.jsonl --k 5,10,20
  uv run python -m model.evaluate --data data/v2/combined.jsonl --per-kernel --save model/results/
"""

import argparse
import datetime
import json
import os
import pathlib
from collections import defaultdict
from math import log2

import lightgbm as lgb
import numpy as np

from model.dataset import load_records, build_arrays
from model.encode import FEATURE_NAMES
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


def _save_results(
  save_dir: str,
  data_path: str,
  model_path: str,
  n_records: int,
  n_groups: int,
  overall: dict[int, float],
  per_kernel_ndcg: dict[str, float] | None,
) -> None:
  pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
  ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

  results = {
    'timestamp': ts,
    'dataset': data_path,
    'model': model_path,
    'n_features': len(FEATURE_NAMES),
    'n_records': n_records,
    'n_groups': n_groups,
    'ndcg': {str(k): round(v, 6) for k, v in overall.items()},
  }
  if per_kernel_ndcg:
    results['per_kernel_ndcg10'] = {k: round(v, 6) for k, v in sorted(per_kernel_ndcg.items())}

  json_path = os.path.join(save_dir, f'eval_{ts}.json')
  with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

  md_path = os.path.join(save_dir, f'eval_{ts}.md')
  ndcg5  = overall.get(5,  float('nan'))
  ndcg10 = overall.get(10, float('nan'))
  ndcg20 = overall.get(20, float('nan'))

  lines = [
    f'# Cost Model Evaluation — {ts}',
    '',
    '## Dataset',
    f'- File: `{data_path}`',
    f'- Records: {n_records:,}',
    f'- Groups (kernel × beam_step): {n_groups:,}',
    '',
    '## Model',
    f'- Path: `{model_path}`',
    f'- Features: {len(FEATURE_NAMES)}',
    '',
    '## NDCG Results',
    '',
    '| Metric   | Value  |',
    '|----------|--------|',
    f'| NDCG@5   | {ndcg5:.4f} |',
    f'| NDCG@10  | {ndcg10:.4f} |',
    f'| NDCG@20  | {ndcg20:.4f} |',
    '',
    '## Feature Engineering (vs. baseline 68 features)',
    '',
    '**Added (Ansor-inspired, +8):**',
    '- `log_arithmetic_intensity` — FLOPs per byte (roofline position)',
    '- `log_reduce_per_output` — reduction work per output element',
    '- `n_global_axes`, `n_local_axes`, `n_reduce_axes` — loop structure counts',
    '- `n_zero_stride_pairs`, `n_unit_stride_pairs`, `n_large_stride_pairs` — memory access patterns from stride matrix',
    '',
    '**Pruned (−2):**',
    '- `log_warp_size` — near-constant on CUDA, low variance',
    '- `log_buf_mean_bytes` — derivable from total/count, redundant',
    '',
    f'**Net: {len(FEATURE_NAMES)} features**',
    '',
    '## Interpretation',
    '',
  ]

  if ndcg10 >= 0.90:
    interp = (
      f'NDCG@10 = {ndcg10:.4f} is strong. The model reliably ranks the fastest kernels '
      'near the top of each beam search group, making it suitable as a beam search cost model.'
    )
  elif ndcg10 >= 0.80:
    interp = (
      f'NDCG@10 = {ndcg10:.4f} is good. The model generally identifies fast kernels, '
      'though ranking near the top of each group is occasionally imprecise. '
      'Consider increasing trees/leaves or adding more training data.'
    )
  else:
    interp = (
      f'NDCG@10 = {ndcg10:.4f} is below target (0.80). The model is learning a ranking signal '
      'but may not be reliable enough for beam search pruning. '
      'Investigate per-kernel breakdown for poorly ranked op types and consider additional features.'
    )
  lines.append(interp)

  if per_kernel_ndcg:
    lines += [
      '',
      '### Per-kernel NDCG@10',
      '',
      '| kernel_id (prefix) | NDCG@10 |',
      '|--------------------|---------|',
    ]
    for kid, v in sorted(per_kernel_ndcg.items()):
      lines.append(f'| `{kid[:16]}` | {v:.4f} |')

  with open(md_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')

  print(f'\nResults saved to:')
  print(f'  {json_path}')
  print(f'  {md_path}')


def main() -> None:
  parser = argparse.ArgumentParser(description='Evaluate cost model NDCG offline')
  parser.add_argument('--model',      default=None,                help='Path to .lgb checkpoint (default: auto-discover)')
  parser.add_argument('--data',       default='data/train.jsonl',  help='JSONL data file')
  parser.add_argument('--k',          default='5,10,20',           help='Comma-separated K values for NDCG@K')
  parser.add_argument('--per-kernel', action='store_true',         help='Show per-kernel NDCG breakdown')
  parser.add_argument('--save',       default=None, metavar='DIR', help='Save JSON + markdown report to this directory')
  args = parser.parse_args()

  k_values = [int(x.strip()) for x in args.k.split(',')]
  overall = evaluate(
    model_path=args.model,
    data_path=args.data,
    k_values=k_values,
    per_kernel=args.per_kernel or bool(args.save),
  )

  if args.save:
    cost_model = load_model(args.model)
    records = load_records(args.data)
    _, _, group_sizes = build_arrays(records)

    # Collect per-kernel NDCG@10 if available (re-compute from evaluate's internals)
    per_kernel_ndcg: dict[str, float] | None = None
    if args.per_kernel or args.save:
      import lightgbm as lgb
      from itertools import groupby as _groupby
      booster = cost_model._booster
      X, y, _ = build_arrays(records)
      scores = booster.predict(X)
      sorted_recs = sorted(records, key=lambda r: (r['kernel_id'], r['beam_step']))
      kernel_groups: dict[str, list[int]] = defaultdict(list)
      kernel_labels: dict[str, list[float]] = defaultdict(list)
      kernel_scores: dict[str, list[float]] = defaultdict(list)
      offset = 0
      for (kid, _bstep), grp_iter in _groupby(sorted_recs, key=lambda r: (r['kernel_id'], r['beam_step'])):
        grp = list(grp_iter)
        g = len(grp)
        kernel_groups[kid].append(g)
        kernel_labels[kid].extend(y[offset:offset + g].tolist())
        kernel_scores[kid].extend(scores[offset:offset + g].tolist())
        offset += g
      per_kernel_ndcg = {}
      for kid in kernel_groups:
        gsizes = kernel_groups[kid]
        kndcg = _ndcg_by_group(np.array(kernel_scores[kid]), np.array(kernel_labels[kid]), gsizes, [10])
        per_kernel_ndcg[kid] = kndcg[10]

    _save_results(
      save_dir=args.save,
      data_path=args.data,
      model_path=cost_model._path,
      n_records=len(records),
      n_groups=len(group_sizes),
      overall=overall,
      per_kernel_ndcg=per_kernel_ndcg,
    )


if __name__ == '__main__':
  main()
