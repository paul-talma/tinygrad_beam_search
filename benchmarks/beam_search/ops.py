"""Op definitions for beam search benchmarking.

Each op factory pre-realizes its input tensors so that repeated calls produce
the same kernel AST (enabling program-cache hits after the first compile).
"""
from collections.abc import Callable

def _tinygrad():
  from tinygrad import Tensor, dtypes
  return Tensor, dtypes.float16


def _matmul(M: int, K: int, N: int):
  Tensor, dtype = _tinygrad()
  a = Tensor.randn(M, K, dtype=dtype).realize()
  b = Tensor.randn(K, N, dtype=dtype).realize()
  return lambda: a @ b


def _conv2d(N: int, C_in: int, H: int, W: int, C_out: int, kH: int, kW: int):
  Tensor, dtype = _tinygrad()
  x = Tensor.randn(N, C_in, H, W, dtype=dtype).realize()
  w = Tensor.randn(C_out, C_in, kH, kW, dtype=dtype).realize()
  return lambda: x.conv2d(w)


def _attention(B: int, heads: int, seq: int, head_dim: int):
  Tensor, dtype = _tinygrad()
  q = Tensor.randn(B, heads, seq, head_dim, dtype=dtype).realize()
  k = Tensor.randn(B, heads, seq, head_dim, dtype=dtype).realize()
  v = Tensor.randn(B, heads, seq, head_dim, dtype=dtype).realize()
  scale = head_dim ** -0.5
  def attn():
    scores = (q @ k.transpose(-2, -1)) * scale
    return scores.softmax(-1) @ v
  return attn


def _elementwise(N: int, M: int):
  Tensor, dtype = _tinygrad()
  x = Tensor.randn(N, M, dtype=dtype).realize()
  return lambda: x.relu()


def _reduction(N: int, M: int):
  Tensor, dtype = _tinygrad()
  x = Tensor.randn(N, M, dtype=dtype).realize()
  return lambda: x.sum()


def _layer_norm(N: int, D: int):
  Tensor, dtype = _tinygrad()
  x = Tensor.randn(N, D, dtype=dtype).realize()
  w = Tensor.randn(D, dtype=dtype).realize()
  b = Tensor.randn(D, dtype=dtype).realize()
  return lambda: x.layernorm().mul(w).add(b)


# ---------------------------------------------------------------------------
# Complex ops for contemporary models
# ---------------------------------------------------------------------------

def _transformer_block(seq: int, hidden: int, heads: int, ffn_dim: int):
  """Standard transformer block: LayerNorm → MHA → residual → LayerNorm → FFN → residual.
  Sized after BERT-base (hidden=768, heads=12, ffn_dim=3072)."""
  Tensor, dtype = _tinygrad()
  B = 1
  head_dim = hidden // heads
  x   = Tensor.randn(B, seq, hidden, dtype=dtype).realize()
  wq  = Tensor.randn(hidden, hidden, dtype=dtype).realize()
  wk  = Tensor.randn(hidden, hidden, dtype=dtype).realize()
  wv  = Tensor.randn(hidden, hidden, dtype=dtype).realize()
  wo  = Tensor.randn(hidden, hidden, dtype=dtype).realize()
  w1  = Tensor.randn(hidden, ffn_dim, dtype=dtype).realize()
  w2  = Tensor.randn(ffn_dim, hidden, dtype=dtype).realize()
  scale = head_dim ** -0.5

  def block():
    # Attention sub-layer
    h = x.layernorm()
    q = (h @ wq).reshape(B, seq, heads, head_dim).transpose(1, 2)
    k = (h @ wk).reshape(B, seq, heads, head_dim).transpose(1, 2)
    v = (h @ wv).reshape(B, seq, heads, head_dim).transpose(1, 2)
    attn = ((q @ k.transpose(-2, -1)) * scale).softmax(-1) @ v
    x2 = x + (attn.transpose(1, 2).reshape(B, seq, hidden) @ wo)
    # FFN sub-layer
    h2 = x2.layernorm()
    return x2 + (h2 @ w1).gelu() @ w2

  return block


def _llama_block(seq: int, hidden: int, heads: int, kv_heads: int, ffn_dim: int):
  """LLaMA-style block: RMSNorm → GQA → residual → RMSNorm → SwiGLU FFN → residual.
  Uses grouped-query attention (GQA) and the SwiGLU activation from LLaMA-2/3."""
  Tensor, dtype = _tinygrad()
  B = 1
  head_dim = hidden // heads
  groups   = heads // kv_heads
  x    = Tensor.randn(B, seq, hidden, dtype=dtype).realize()
  wq   = Tensor.randn(hidden, heads    * head_dim, dtype=dtype).realize()
  wk   = Tensor.randn(hidden, kv_heads * head_dim, dtype=dtype).realize()
  wv   = Tensor.randn(hidden, kv_heads * head_dim, dtype=dtype).realize()
  wo   = Tensor.randn(hidden, hidden, dtype=dtype).realize()
  wg   = Tensor.randn(hidden, ffn_dim, dtype=dtype).realize()  # SwiGLU gate
  wu   = Tensor.randn(hidden, ffn_dim, dtype=dtype).realize()  # SwiGLU up
  wd   = Tensor.randn(ffn_dim, hidden, dtype=dtype).realize()  # SwiGLU down
  scale = head_dim ** -0.5

  def rms_norm(t):
    return t * (t.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()

  def block():
    # GQA self-attention
    h  = rms_norm(x)
    q  = (h @ wq).reshape(B, seq, heads,    head_dim).transpose(1, 2)
    k  = (h @ wk).reshape(B, seq, kv_heads, head_dim).transpose(1, 2)
    v  = (h @ wv).reshape(B, seq, kv_heads, head_dim).transpose(1, 2)
    # Expand KV for grouped-query attention
    k  = k.reshape(B, kv_heads, 1, seq, head_dim).expand(B, kv_heads, groups, seq, head_dim).reshape(B, heads, seq, head_dim)
    v  = v.reshape(B, kv_heads, 1, seq, head_dim).expand(B, kv_heads, groups, seq, head_dim).reshape(B, heads, seq, head_dim)
    attn = ((q @ k.transpose(-2, -1)) * scale).softmax(-1) @ v
    x2   = x + (attn.transpose(1, 2).reshape(B, seq, hidden) @ wo)
    # SwiGLU FFN
    h2 = rms_norm(x2)
    return x2 + ((h2 @ wg).silu() * (h2 @ wu)) @ wd

  return block


def _depthwise_sep_conv(N: int, C: int, H: int, W: int, C_out: int):
  """Depthwise-separable convolution as used in MobileNet/EfficientNet.
  Two fused kernels: depthwise 3×3 (groups=C) then pointwise 1×1."""
  Tensor, dtype = _tinygrad()
  x  = Tensor.randn(N, C, H, W, dtype=dtype).realize()
  dw = Tensor.randn(C, 1, 3, 3, dtype=dtype).realize()   # depthwise
  pw = Tensor.randn(C_out, C, 1, 1, dtype=dtype).realize()  # pointwise
  return lambda: x.conv2d(dw, groups=C).conv2d(pw)


def _causal_attention(B: int, heads: int, seq: int, head_dim: int):
  """Scaled dot-product attention with a causal (lower-triangular) mask,
  as used in decoder-only models (GPT, LLaMA inference)."""
  Tensor, dtype = _tinygrad()
  q     = Tensor.randn(B, heads, seq, head_dim, dtype=dtype).realize()
  k     = Tensor.randn(B, heads, seq, head_dim, dtype=dtype).realize()
  v     = Tensor.randn(B, heads, seq, head_dim, dtype=dtype).realize()
  # Pre-build mask so it's part of the realized input, not the searched kernel
  mask  = Tensor.ones(seq, seq, dtype=dtype).tril().realize()
  scale = head_dim ** -0.5

  def attn():
    scores = (q @ k.transpose(-2, -1)) * scale
    scores = scores + (1 - mask) * -1e4   # apply causal mask
    return scores.softmax(-1) @ v

  return attn


# ---------------------------------------------------------------------------
# Registry and suites
# ---------------------------------------------------------------------------

OpFactory = Callable[[], Callable]

MATMUL_OPS: dict[str, OpFactory] = {
  "matmul_1024": lambda: _matmul(1024, 1024, 1024),
  "matmul_2048": lambda: _matmul(2048, 2048, 2048),
  "matmul_8192": lambda: _matmul(8192, 8192, 8192),
  "matmul_16384": lambda: _matmul(16384, 16384, 16384),
}

VANILLA_ATTENTION_OPS: dict[str, OpFactory] = {
  "attn_256": lambda: _attention(1, 8, 256, 64),
  "attn_512": lambda: _attention(1, 8, 512, 64),
}

SINGLE_LAYER_CONVOLUTION_OPS: dict[str, OpFactory] = {
  "conv_small":  lambda: _conv2d(1, 32, 32, 32, 64, 3, 3),
  "conv_medium": lambda: _conv2d(1, 64, 64, 64, 128, 3, 3),
  "conv_large":  lambda: _conv2d(1, 128, 128, 128, 256, 3, 3),
  "conv_xlarge": lambda: _conv2d(1, 256, 128, 128, 512, 3, 3),
}

ATTENTION_RELATED_BLOCK_OPS: dict[str, OpFactory] = {
  "relu_large":    lambda: _elementwise(4096, 4096),
  "reduce_large":  lambda: _reduction(4096, 4096),
  "layernorm_512": lambda: _layer_norm(512, 1024),
}

LARGER_CONVOLUTION_MODULE_OPS: dict[str, OpFactory] = {
  "depthwise_sep_conv": lambda: _depthwise_sep_conv(N=1, C=128, H=56, W=56, C_out=256),
}

LARGER_TRANSFORMER_BLOCK_OPS: dict[str, OpFactory] = {
  # "transformer_block": lambda: _transformer_block(seq=128, hidden=768, heads=12, ffn_dim=3072),
  "transformer_block_1024": lambda: _transformer_block(seq=1024, hidden=1024, heads=16, ffn_dim=4096),
  "transformer_block_2048": lambda: _transformer_block(seq=2048, hidden=1536, heads=24, ffn_dim=6144),
  # "llama_block": lambda: _llama_block(seq=256, hidden=2048, heads=16, kv_heads=4, ffn_dim=5504),
  "llama_block_1024": lambda: _llama_block(seq=1024, hidden=3072, heads=24, kv_heads=8, ffn_dim=8192),
  "llama_block_2048": lambda: _llama_block(seq=2048, hidden=4096, heads=32, kv_heads=8, ffn_dim=11008),
  # "causal_attn_512": lambda: _causal_attention(B=1, heads=8, seq=512, head_dim=64),
  "causal_attn_1024": lambda: _causal_attention(B=1, heads=16, seq=1024, head_dim=64),
  "causal_attn_2048": lambda: _causal_attention(B=1, heads=32, seq=2048, head_dim=64),
}

def _merge_suites(*suites: dict[str, OpFactory]) -> dict[str, OpFactory]:
  merged: dict[str, OpFactory] = {}
  for suite in suites: merged.update(suite)
  return merged

BENCHMARK_SUITES: dict[str, dict[str, OpFactory]] = {
  "matmuls": MATMUL_OPS,
  "vanilla_attentions": VANILLA_ATTENTION_OPS,
  "single_layer_convolutions": SINGLE_LAYER_CONVOLUTION_OPS,
  "attention_related_blocks": ATTENTION_RELATED_BLOCK_OPS,
  "larger_convolution_modules": LARGER_CONVOLUTION_MODULE_OPS,
  "larger_transformer_blocks": LARGER_TRANSFORMER_BLOCK_OPS,
}
BENCHMARK_SUITES["default"] = _merge_suites(
  MATMUL_OPS,
  SINGLE_LAYER_CONVOLUTION_OPS,
  VANILLA_ATTENTION_OPS,
  ATTENTION_RELATED_BLOCK_OPS,
)
BENCHMARK_SUITES["all"] = _merge_suites(
  MATMUL_OPS,
  VANILLA_ATTENTION_OPS,
  SINGLE_LAYER_CONVOLUTION_OPS,
  ATTENTION_RELATED_BLOCK_OPS,
  LARGER_CONVOLUTION_MODULE_OPS,
  LARGER_TRANSFORMER_BLOCK_OPS,
)
OP_NAMES = tuple(BENCHMARK_SUITES["all"])

def get_suite_names() -> tuple[str, ...]:
  return tuple(BENCHMARK_SUITES)

def get_suite_ops(suite: str) -> list[str]:
  if suite not in BENCHMARK_SUITES:
    raise KeyError(f"unknown benchmark suite {suite!r}")
  return list(BENCHMARK_SUITES[suite])

def get_op(name: str):
  try:
    return BENCHMARK_SUITES["all"][name]()
  except KeyError as e:
    raise KeyError(f"unknown benchmark op {name!r}") from e

def get_ops() -> dict:
  return {name: factory() for name, factory in BENCHMARK_SUITES["all"].items()}
