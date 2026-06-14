"""Unified op registry for data collection.

All ops live here — no distinction between "benchmark" and "gap" ops.
Each factory pre-realizes its inputs so repeated calls hit the program cache.
"""
from collections.abc import Callable

OpFactory = Callable[[], Callable]


def _tinygrad(dtype=None):
  from tinygrad import Tensor, dtypes
  return Tensor, (dtypes.float16 if dtype is None else dtype)


def _matmul(M: int, K: int, N: int, dtype=None):
  Tensor, dtype = _tinygrad(dtype)
  a = Tensor.randn(M, K, dtype=dtype).realize()
  b = Tensor.randn(K, N, dtype=dtype).realize()
  return lambda: a @ b


def _conv2d(N: int, C_in: int, H: int, W: int, C_out: int, kH: int, kW: int,
            stride: int = 1, padding: int = 0, dtype=None):
  Tensor, dtype = _tinygrad(dtype)
  x = Tensor.randn(N, C_in, H, W, dtype=dtype).realize()
  w = Tensor.randn(C_out, C_in, kH, kW, dtype=dtype).realize()
  return lambda: x.conv2d(w, stride=stride, padding=padding)


def _attention(B: int, heads: int, seq: int, head_dim: int, dtype=None):
  Tensor, dtype = _tinygrad(dtype)
  q = Tensor.randn(B, heads, seq, head_dim, dtype=dtype).realize()
  k = Tensor.randn(B, heads, seq, head_dim, dtype=dtype).realize()
  v = Tensor.randn(B, heads, seq, head_dim, dtype=dtype).realize()
  scale = head_dim ** -0.5
  def attn():
    scores = (q @ k.transpose(-2, -1)) * scale
    return scores.softmax(-1) @ v
  return attn


def _softmax(N: int, M: int, dtype=None):
  Tensor, dtype = _tinygrad(dtype)
  x = Tensor.randn(N, M, dtype=dtype).realize()
  return lambda: x.softmax(-1)


def _elementwise(N: int, M: int, dtype=None):
  Tensor, dtype = _tinygrad(dtype)
  x = Tensor.randn(N, M, dtype=dtype).realize()
  return lambda: x.relu()


def _elementwise_gelu(N: int, M: int, dtype=None):
  Tensor, dtype = _tinygrad(dtype)
  x = Tensor.randn(N, M, dtype=dtype).realize()
  return lambda: x.gelu()


def _elementwise_silu(N: int, M: int, dtype=None):
  Tensor, dtype = _tinygrad(dtype)
  x = Tensor.randn(N, M, dtype=dtype).realize()
  return lambda: x.silu()


def _reduction(N: int, M: int, dtype=None):
  Tensor, dtype = _tinygrad(dtype)
  x = Tensor.randn(N, M, dtype=dtype).realize()
  return lambda: x.sum()


def _reduction_rows(N: int, M: int, dtype=None):
  """Row-wise reduction (sum over last axis) — different loop structure than global sum."""
  Tensor, dtype = _tinygrad(dtype)
  x = Tensor.randn(N, M, dtype=dtype).realize()
  return lambda: x.sum(axis=-1)


def _layer_norm(N: int, D: int, dtype=None):
  Tensor, dtype = _tinygrad(dtype)
  x = Tensor.randn(N, D, dtype=dtype).realize()
  w = Tensor.randn(D, dtype=dtype).realize()
  b = Tensor.randn(D, dtype=dtype).realize()
  return lambda: x.layernorm().mul(w).add(b)


def _rms_norm(N: int, D: int, dtype=None):
  Tensor, dtype = _tinygrad(dtype)
  x = Tensor.randn(N, D, dtype=dtype).realize()
  return lambda: x * (x.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()


def _f32():
  from tinygrad import dtypes
  return dtypes.float32


ALL_OPS: dict[str, OpFactory] = {
  # --- matmul ---
  "matmul_512":            lambda: _matmul(512, 512, 512),
  "matmul_1024":           lambda: _matmul(1024, 1024, 1024),
  "matmul_2048":           lambda: _matmul(2048, 2048, 2048),
  "matmul_4096":           lambda: _matmul(4096, 4096, 4096),
  "matmul_8192":           lambda: _matmul(8192, 8192, 8192),
  "matmul_16384":          lambda: _matmul(16384, 16384, 16384),
  "matmul_4096_f32":       lambda: _matmul(4096, 4096, 4096, dtype=_f32()),
  "matmul_ffn_4096_16384": lambda: _matmul(2048, 4096, 16384),
  "matmul_ffn_16384_4096": lambda: _matmul(2048, 16384, 4096),
  "matmul_rect_512_2048":  lambda: _matmul(512, 512, 2048),   # encoder FFN
  "matmul_rect_64_4096":   lambda: _matmul(64, 2048, 4096),   # decode step

  # --- convolution ---
  "conv_small":        lambda: _conv2d(1, 32, 32, 32, 64, 3, 3),
  "conv_medium":       lambda: _conv2d(1, 64, 64, 64, 128, 3, 3),
  "conv_medium_b4":    lambda: _conv2d(4, 64, 64, 64, 128, 3, 3, padding=1),
  "conv_medium_b8":    lambda: _conv2d(8, 64, 64, 64, 128, 3, 3, padding=1),
  "conv_large":        lambda: _conv2d(1, 128, 128, 128, 256, 3, 3),
  "conv_large_b4":     lambda: _conv2d(4, 128, 64, 64, 256, 3, 3, padding=1),
  "conv_xlarge":       lambda: _conv2d(1, 256, 128, 128, 512, 3, 3),
  "conv_1x1_large":    lambda: _conv2d(1, 256, 56, 56, 256, 1, 1),
  "conv_strided":      lambda: _conv2d(1, 64, 56, 56, 128, 3, 3, stride=2),
  "conv_strided_b4":   lambda: _conv2d(4, 64, 56, 56, 128, 3, 3, stride=2),
  "conv_5x5":          lambda: _conv2d(1, 32, 32, 32, 64, 5, 5, padding=2),
  "conv_5x5_b8":       lambda: _conv2d(8, 32, 32, 32, 64, 5, 5, padding=2),
  "conv_7x7":          lambda: _conv2d(1, 3, 224, 224, 64, 7, 7, stride=2, padding=3),
  "conv_3x3_b4":       lambda: _conv2d(4, 64, 32, 32, 128, 3, 3, padding=1),
  "conv_5x5_b4":       lambda: _conv2d(4, 32, 32, 32, 64, 5, 5, padding=2),

  # --- attention ---
  "attn_256":          lambda: _attention(1, 8, 256, 64),
  "attn_512":          lambda: _attention(1, 8, 512, 64),
  "attn_1024":         lambda: _attention(1, 8, 1024, 64),
  "attn_b2_t128":      lambda: _attention(2, 8, 128, 64),
  "attn_b4_t256":      lambda: _attention(4, 8, 256, 64),
  "attn_b8_t128":      lambda: _attention(8, 8, 128, 64),
  "attn_b1_t64":       lambda: _attention(1, 8, 64, 64),
  "attn_b1_t512":      lambda: _attention(1, 8, 512, 64),

  # --- elementwise ---
  "relu_small":        lambda: _elementwise(256, 256),
  "relu_medium":       lambda: _elementwise(1024, 1024),
  "relu_large":        lambda: _elementwise(4096, 4096),
  "gelu_medium":       lambda: _elementwise_gelu(1024, 1024),
  "gelu_large":        lambda: _elementwise_gelu(4096, 4096),
  "silu_medium":       lambda: _elementwise_silu(1024, 1024),
  "silu_large":        lambda: _elementwise_silu(4096, 4096),

  # --- softmax ---
  "softmax_medium":    lambda: _softmax(1024, 1024),
  "softmax_large":     lambda: _softmax(4096, 4096),

  # --- reduction / norm ---
  "reduce_small":         lambda: _reduction(256, 256),
  "reduce_medium":        lambda: _reduction(1024, 1024),
  "reduce_large":         lambda: _reduction(4096, 4096),
  "reduce_large_f32":     lambda: _reduction(4096, 4096, dtype=_f32()),
  "reduce_rows_medium":   lambda: _reduction_rows(1024, 1024),
  "reduce_rows_large":    lambda: _reduction_rows(4096, 4096),
  "layernorm_256":        lambda: _layer_norm(256, 512),
  "layernorm_512":        lambda: _layer_norm(512, 1024),
  "layernorm_512_f32":    lambda: _layer_norm(512, 1024, dtype=_f32()),
  "layernorm_1024":       lambda: _layer_norm(1024, 2048),
  "rmsnorm_small":        lambda: _rms_norm(512, 1024),
  "rmsnorm_large":        lambda: _rms_norm(2048, 4096),
  "rmsnorm_large_f32":    lambda: _rms_norm(2048, 4096, dtype=_f32()),
}


def get_op(name: str) -> Callable:
  try:
    return ALL_OPS[name]()
  except KeyError as e:
    raise KeyError(f"unknown op {name!r}. Available: {sorted(ALL_OPS)}") from e


def list_op_names() -> list[str]:
  return list(ALL_OPS)
