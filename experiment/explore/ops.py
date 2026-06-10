"""~10 representative workloads for beam search exploration.

Each factory pre-realizes its inputs so every call to the returned lambda
produces the same kernel AST (enabling beam search cache hits across
repeated runs). All ops use float16 — typical for GPU workloads.
"""
from collections.abc import Callable
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

_f16 = dtypes.float16

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
OPS: dict[str, Callable] = {}

def _reg(name: str):
  def deco(fn):
    OPS[name] = fn()
    return fn
  return deco


# ---------------------------------------------------------------------------
# Matmuls
# ---------------------------------------------------------------------------
@_reg("matmul_256")
def _():
  a = Tensor.randn(256, 256, dtype=_f16).realize()
  b = Tensor.randn(256, 256, dtype=_f16).realize()
  return lambda: a @ b

@_reg("matmul_1024")
def _():
  a = Tensor.randn(1024, 1024, dtype=_f16).realize()
  b = Tensor.randn(1024, 1024, dtype=_f16).realize()
  return lambda: a @ b

@_reg("matmul_2048")
def _():
  a = Tensor.randn(2048, 2048, dtype=_f16).realize()
  b = Tensor.randn(2048, 2048, dtype=_f16).realize()
  return lambda: a @ b

@_reg("matmul_4096")
def _():
  a = Tensor.randn(4096, 4096, dtype=_f16).realize()
  b = Tensor.randn(4096, 4096, dtype=_f16).realize()
  return lambda: a @ b

@_reg("matmul_8192")
def _():
  a = Tensor.randn(8192, 8192, dtype=_f16).realize()
  b = Tensor.randn(8192, 8192, dtype=_f16).realize()
  return lambda: a @ b

@_reg("matmul_rect")
def _():
  a = Tensor.randn(512, 1024, dtype=_f16).realize()
  b = Tensor.randn(1024, 2048, dtype=_f16).realize()
  return lambda: a @ b

# ---------------------------------------------------------------------------
# Convolutions
# ---------------------------------------------------------------------------
@_reg("conv_3x3")
def _():
  x = Tensor.randn(4, 64, 32, 32, dtype=_f16).realize()
  w = Tensor.randn(128, 64, 3, 3, dtype=_f16).realize()
  return lambda: x.conv2d(w, padding=1)

@_reg("conv_5x5")
def _():
  x = Tensor.randn(4, 32, 32, 32, dtype=_f16).realize()
  w = Tensor.randn(64, 32, 5, 5, dtype=_f16).realize()
  return lambda: x.conv2d(w, padding=2)

# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------
@_reg("elem_relu")
def _():
  x = Tensor.randn(4096, 4096, dtype=_f16).realize()
  return lambda: x.relu()

@_reg("elem_fused")
def _():
  x = Tensor.randn(2048, 2048, dtype=_f16).realize()
  y = Tensor.randn(2048, 2048, dtype=_f16).realize()
  return lambda: (x + y).relu()

# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------
@_reg("reduce_sum")
def _():
  x = Tensor.randn(1024, 4096, dtype=_f16).realize()
  return lambda: x.sum(axis=1)

# ---------------------------------------------------------------------------
# Attention-like
# ---------------------------------------------------------------------------
@_reg("attention")
def _():
  B, H, T, D = 2, 8, 128, 64
  scale = D ** -0.5
  q = Tensor.randn(B, H, T, D, dtype=_f16).realize()
  k = Tensor.randn(B, H, T, D, dtype=_f16).realize()
  v = Tensor.randn(B, H, T, D, dtype=_f16).realize()
  def _attn():
    scores = (q @ k.transpose(-2, -1)) * scale
    return scores.softmax(-1) @ v
  return _attn
