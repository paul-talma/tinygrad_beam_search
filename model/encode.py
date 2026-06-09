"""Fixed-size feature encoding for LightGBM.

Converts a feature dict from collect.features.extract_features() into a
fixed-length numpy array. Stateless: vocabularies are fixed constants,
so the same encoding is used identically in training and inference.
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Shape constants
# ---------------------------------------------------------------------------

MAX_AXES = 8   # pad/truncate full_shape and axis_types to this length
MAX_BUFS = 5   # pad/truncate stride_matrix rows to this many buffers

# ---------------------------------------------------------------------------
# Categorical vocabularies — fixed, not fitted from data
# ---------------------------------------------------------------------------

AXIS_TYPE_VOCAB: list[str] = ['GLOBAL', 'LOCAL', 'REDUCE', 'WARP', 'UPCAST', 'SPECIAL']
_AXIS_TYPE_IDX: dict[str, int] = {a: i for i, a in enumerate(AXIS_TYPE_VOCAB)}

DTYPE_VOCAB: list[str] = [
  'float16', 'bfloat16', 'float32', 'float64',
  'int8', 'int16', 'int32', 'int64',
  'uint8', 'uint16', 'uint32', 'uint64',
  'unsigned int', 'bool', 'unknown',
]
_DTYPE_IDX: dict[str, int] = {d: i for i, d in enumerate(DTYPE_VOCAB)}

DEVICE_VOCAB: list[str] = ['METAL', 'CUDA', 'AMD', 'NV', 'HIP', 'CPU', 'LLVM', 'CL', 'CLANG', 'GPU', 'unknown']
_DEVICE_IDX: dict[str, int] = {d: i for i, d in enumerate(DEVICE_VOCAB)}

REDUCE_OP_VOCAB: list[str] = ['none', 'ADD', 'MAX', 'MUL', 'MIN', 'XOR']
_REDUCE_OP_IDX: dict[str, int] = {r: i for i, r in enumerate(REDUCE_OP_VOCAB)}

REDUCE_DTYPE_VOCAB: list[str] = ['none'] + DTYPE_VOCAB
_REDUCE_DTYPE_IDX: dict[str, int] = {d: i for i, d in enumerate(REDUCE_DTYPE_VOCAB)}

# ---------------------------------------------------------------------------
# Feature names — same order as encode_features() output
# (used as column names when building lgb.Dataset)
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = (
  # categoricals (4)
  ['dtype', 'device', 'reduce_op', 'reduce_dtype'] +
  # scalar booleans (2)
  ['has_reduce', 'dont_use_locals'] +
  # scalar ints (4)
  ['shape_len', 'upcasted', 'group_for_reduces', 'n_bufs'] +
  # log-transformed size products (5)
  ['log_global_size', 'log_local_size', 'log_reduce_size', 'log_warp_size', 'log_upcast_size'] +
  # log-transformed counts (4)
  ['log_n_mulacc', 'log_n_transcendental', 'log_n_where', 'log_n_cast'] +
  # log-transformed post-compile features (2)
  ['log_compiled_uops', 'log_flop_estimate'] +
  # log-transformed buffer aggregates (3)
  ['log_buf_total_bytes', 'log_buf_max_bytes', 'log_buf_mean_bytes'] +
  # device constants (2)
  ['log_shared_max', 'log_global_max'] +
  # full_shape — log2(x+1) per slot, MAX_AXES slots (8)
  [f'shape_{i}' for i in range(MAX_AXES)] +
  # axis_types — ordinal per slot, -1 = padding (8)
  [f'axis_type_{i}' for i in range(MAX_AXES)] +
  # stride_matrix — log1p per cell, MAX_BUFS × MAX_AXES cells (40)
  [f'stride_{b}_{a}' for b in range(MAX_BUFS) for a in range(MAX_AXES)]
)

# Column names of categorical features (LightGBM uses these to enable native cat handling)
CATEGORICAL_FEATURES: list[str] = ['dtype', 'device', 'reduce_op', 'reduce_dtype']

assert len(FEATURE_NAMES) == 4 + 2 + 4 + 5 + 4 + 2 + 3 + 2 + MAX_AXES + MAX_AXES + MAX_BUFS * MAX_AXES


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _safe_log2(x: float) -> float:
  return math.log2(x + 1.0)


def _safe_log1p(x: float) -> float:
  return math.log1p(max(0.0, float(x)))


def encode_features(feat: dict, compiled_uops: int = 0, flop_estimate: int = 0) -> np.ndarray:
  """Encode one feature dict to a fixed-length float32 numpy array.

  `compiled_uops` and `flop_estimate` come from the JSONL record's top-level fields,
  not from feat itself. Pass 0 at inference time (before compilation).
  """
  out = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
  i = 0

  # -- categoricals --
  out[i] = _DTYPE_IDX.get(feat.get('dtype', 'unknown'), len(DTYPE_VOCAB) - 1);  i += 1
  out[i] = _DEVICE_IDX.get(feat.get('device', 'unknown'), len(DEVICE_VOCAB) - 1); i += 1
  reduce_op_key = feat.get('reduce_op') or 'none'
  out[i] = _REDUCE_OP_IDX.get(reduce_op_key, 0); i += 1
  reduce_dtype_key = feat.get('reduce_dtype') or 'none'
  out[i] = _REDUCE_DTYPE_IDX.get(reduce_dtype_key, 0); i += 1

  # -- booleans --
  out[i] = float(bool(feat.get('has_reduce', False))); i += 1
  out[i] = float(bool(feat.get('dont_use_locals', False))); i += 1

  # -- scalar ints --
  out[i] = float(feat.get('shape_len', 0)); i += 1
  out[i] = float(feat.get('upcasted', 0)); i += 1
  out[i] = float(feat.get('group_for_reduces', 0)); i += 1
  buf_bytes: list[int] = feat.get('buf_bytes', [])
  out[i] = float(len(buf_bytes)); i += 1

  # -- log size products --
  for key in ('global_size', 'local_size', 'reduce_size', 'warp_size', 'upcast_size'):
    out[i] = _safe_log2(feat.get(key, 1)); i += 1

  # -- log counts --
  for key in ('n_mulacc', 'n_transcendental', 'n_where', 'n_cast'):
    out[i] = _safe_log1p(feat.get(key, 0)); i += 1

  # -- post-compile features (0 at inference) --
  out[i] = _safe_log1p(compiled_uops); i += 1
  out[i] = _safe_log1p(flop_estimate); i += 1

  # -- buffer aggregates --
  if buf_bytes:
    out[i] = _safe_log1p(sum(buf_bytes)); i += 1
    out[i] = _safe_log1p(max(buf_bytes)); i += 1
    out[i] = _safe_log1p(sum(buf_bytes) / len(buf_bytes)); i += 1
  else:
    i += 3

  # -- device constants --
  out[i] = _safe_log2(feat.get('shared_max', 0)); i += 1
  out[i] = _safe_log2(feat.get('global_max', 0)); i += 1

  # -- full_shape (MAX_AXES slots) --
  full_shape: list = feat.get('full_shape', [])
  for slot in range(MAX_AXES):
    if slot < len(full_shape):
      v = full_shape[slot]
      out[i] = _safe_log2(float(v) if isinstance(v, (int, float)) else 0.0)
    i += 1

  # -- axis_types (MAX_AXES slots, ordinal, -1 = padding) --
  axis_types: list[str] = feat.get('axis_types', [])
  for slot in range(MAX_AXES):
    if slot < len(axis_types):
      out[i] = float(_AXIS_TYPE_IDX.get(axis_types[slot], len(AXIS_TYPE_VOCAB) - 1))
    else:
      out[i] = -1.0
    i += 1

  # -- stride_matrix (MAX_BUFS × MAX_AXES, log1p) --
  stride_matrix: list[list[int]] = feat.get('stride_matrix', [])
  for b in range(MAX_BUFS):
    row = stride_matrix[b] if b < len(stride_matrix) else []
    for a in range(MAX_AXES):
      if a < len(row):
        out[i] = _safe_log1p(abs(row[a]))
      i += 1

  assert i == len(FEATURE_NAMES), f"encoder wrote {i} values, expected {len(FEATURE_NAMES)}"
  return out
