"""Load a trained cost model and wrap it as cost_model(scheduler) -> float.

The returned callable is compatible with experiment.validate.harness.run_with_model():
  lower score = predicted faster kernel.

Usage:
  from model.predict import load_model

  cost_model = load_model()               # auto-discovers latest checkpoint
  cost_model = load_model("path/to.lgb")  # explicit path

  # At inference time (no compiled_uops / flop_estimate available):
  score = cost_model(scheduler)  # float, lower = faster
"""

import os
import pathlib

import lightgbm as lgb
import numpy as np

from model.encode import encode_features


_CHECKPOINT_DIR = pathlib.Path(__file__).parent / 'checkpoints'


def _find_latest_checkpoint(directory: str | pathlib.Path | None = None) -> str:
  d = pathlib.Path(directory) if directory else _CHECKPOINT_DIR
  candidates = sorted(d.glob('*.lgb'), key=lambda p: p.stat().st_mtime, reverse=True)
  if not candidates:
    raise FileNotFoundError(f"No .lgb checkpoint files found in {d}")
  return str(candidates[0])


class CostModel:
  """Wraps a LightGBM booster as a callable suitable for the validation harness."""

  def __init__(self, booster: lgb.Booster, path: str = ''):
    self._booster = booster
    self._path = path

  def __call__(self, scheduler) -> float:
    """Score one beam search candidate. Lower = predicted faster."""
    from collect.features import extract_features
    feat = extract_features(scheduler)
    x = encode_features(feat, compiled_uops=0, flop_estimate=0).reshape(1, -1)
    score = float(self._booster.predict(x)[0])
    return -score  # negate: booster gives higher=better, harness wants lower=faster

  def __repr__(self) -> str:
    return f"CostModel(path={self._path!r})"


def load_model(path: str | None = None, checkpoint_dir: str | pathlib.Path | None = None) -> CostModel:
  """Load a LightGBM cost model from disk.

  Args:
    path: explicit path to a .lgb file. If None, auto-discovers the most recently
          modified checkpoint in `checkpoint_dir` (or model/checkpoints/ by default).
    checkpoint_dir: directory to search when `path` is None.

  Returns:
    CostModel callable: cost_model(scheduler) -> float
  """
  if path is None:
    path = _find_latest_checkpoint(checkpoint_dir)
  booster = lgb.Booster(model_file=path)
  return CostModel(booster, path=path)
