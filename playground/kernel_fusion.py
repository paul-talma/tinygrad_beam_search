"""
Kernel fusion demo for tinygrad.

Kernel fusion is tinygrad's mechanism for combining multiple tensor ops into a
single GPU kernel, avoiding intermediate buffer writes and reads. This file
demonstrates three scenarios and explains how to visualize the fusion graph.

--- Visualization ---

Two complementary views are available:

1.  DEBUG=4  (in code, via Context)
    Prints the generated GPU kernel source for each scheduled kernel.
    One kernel function = full fusion; multiple functions = partial or no fusion.
    Use: `with Context(DEBUG=4): tensor.numpy()`

2.  VIZ=1  (as an env var before running this script)
    Launches the interactive graph-rewrite visualizer that shows the UOp DAG
    before and after each rewrite pass (pattern-matcher step).
    Use: `VIZ=1 python3 experimentation/kernel_fusion.py`
    The browser UI lets you step through rewrites to see exactly when ops are
    fused into a single SINK node.

Run with DEBUG=4 to see generated kernels:
    DEBUG=4 python3 experimentation/kernel_fusion.py

Run with VIZ=1 for the interactive graph visualizer:
    VIZ=1 python3 experimentation/kernel_fusion.py

IMPORTANT — schedule_linear() side effect:
    Calling .schedule_linear() transforms the tensor's UOp graph in-place
    (remapping buffer references).  Do NOT call it before .numpy()/.realize()
    on the same tensor — the input buffers will be remapped and you'll get
    zeros.  Always realize first if you want to inspect the schedule.
"""

import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import Context


def separator(title: str) -> None:
  print(f"\n{'='*60}")
  print(f"  {title}")
  print('='*60)


# ── fixed inputs (no randn kernels polluting the count) ──────────────────────
# .copy() on each Tensor() call avoids buffer aliasing across scenarios.
rng = np.random.default_rng(42)
a_np = rng.standard_normal(1024).astype(np.float32)
b_np = rng.standard_normal(1024).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Elementwise fusion
#
# tinygrad defers execution until .numpy() / .realize().  When it finally
# schedules the work it sees that relu(a), relu(b), mul, and add all have the
# same shape and no data dependency between them, so it emits ONE kernel:
#
#   out[i] = max(a[i], 0) * 2.0 + max(b[i], 0)
#
# Look for "scheduled 1 kernels" and a SINGLE kernel function in the output.
# The generated code contains all four ops: two relu clamps, one multiply,
# one add — in one pass over the data.
# ─────────────────────────────────────────────────────────────────────────────
separator("Scenario 1 — Elementwise fusion  [expect: 1 compute kernel]")

a = Tensor(a_np.copy())
b = Tensor(b_np.copy())
fused = a.relu() * 2.0 + b.relu()

with Context(DEBUG=4):
  result_fused = fused.numpy()

# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Broken fusion with .realize()
#
# .realize() forces immediate materialization, writing intermediate results to
# device memory.  Each group of lazy ops before a .realize() becomes its own
# kernel; the dependency on materialized buffers prevents the scheduler from
# fusing across the boundary.
#
# Kernel 1: relu(a2)  → writes a_relu to device memory
# Kernel 2: relu(b2)  → writes b_relu to device memory
# Kernel 3: a_relu * 2.0 + b_relu  → reads those buffers, writes final result
#
# Look for "scheduled 2 kernels" for each realize() and "scheduled 1 kernels"
# for the final step — 3 compute dispatches total vs 1 in Scenario 1.
# ─────────────────────────────────────────────────────────────────────────────
separator("Scenario 2 — Broken fusion with .realize()  [expect: 3 total kernels]")

a2 = Tensor(a_np.copy())
b2 = Tensor(b_np.copy())

print("-- realize a_relu:")
with Context(DEBUG=4):
  a_relu = a2.relu().realize()

print("-- realize b_relu:")
with Context(DEBUG=4):
  b_relu = b2.relu().realize()

print("-- final step (mul + add):")
with Context(DEBUG=4):
  result_broken = (a_relu * 2.0 + b_relu).numpy()

assert np.allclose(result_fused, result_broken, atol=1e-5), "outputs differ!"
print("Both paths produce identical outputs.")

# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — Reduction fusion
#
# A reduction (sum, mean, max, …) and a following scalar elementwise op can
# still be fused: tinygrad embeds the scalar multiply INSIDE the reduction
# kernel, applying it to the final accumulated value before the single thread
# writes to the output buffer.
#
# Notice in the generated code: the multiply-by-2 appears in the final-write
# if-block:   *(data0_1+0) = ((acc1+0))*2.0f)
# not in a separate kernel.
# ─────────────────────────────────────────────────────────────────────────────
separator("Scenario 3 — Post-reduction fusion  [expect: 1 compute kernel]")

a3 = Tensor(a_np.copy())
fused_reduction = a3.sum() * 2.0

with Context(DEBUG=4):
  result_reduction = fused_reduction.numpy()

expected = float(a_np.sum()) * 2.0
assert abs(result_reduction.item() - expected) < 1e-1, "reduction output wrong"
print(f"Result: {result_reduction.item():.4f}  (expected: {expected:.4f})")
