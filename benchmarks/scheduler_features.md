# Scheduler Object — Attribute Reference

`tinygrad/codegen/opt/postrange.py`

The `Scheduler` is the object passed into `beam_search` (and `hand_coded_optimizations`).
It wraps a kernel UOp AST and a renderer, and exposes all the properties that the optimizer
reads to decide which `Opt` actions to try. Every attribute below is used somewhere in the
live tinygrad pipeline — this document records what it is, why tinygrad needs it, and what
it exposes for feature engineering.

---

## What a kernel is

A kernel in tinygrad is a **single loop nest**: a set of nested `for` loops whose body
computes arithmetic and writes results to buffers. Each loop corresponds to one `RANGE`
node in the kernel's UOp AST. The full set of `RANGE` nodes, together with their sizes
and types, completely defines the loop nest structure.

For a `(4,8) @ (8,4)` matmul the loop nest is:

```
for i in range(4):      # RANGE — AxisType.GLOBAL (output row)
  for j in range(4):    # RANGE — AxisType.GLOBAL (output col)
    for k in range(8):  # RANGE — AxisType.REDUCE (accumulation)
      C[i,j] += A[i,k] * B[k,j]
```

Every `Scheduler` attribute is derived from this loop nest: `full_shape` is the list of
loop bounds `[4, 4, 8]`, `axis_types` is `[GLOBAL, GLOBAL, REDUCE]`, and so on.

### What defines the loop nest structure

Three things determine which RANGE nodes exist and how they are typed:

1. **The computation itself.** Each independent output dimension becomes a GLOBAL loop;
   each accumulation dimension becomes a REDUCE loop. A fused elementwise-after-reduce
   kernel (e.g. `relu(matmul(A, B))`) shares the same GLOBAL loops for both the matmul
   and the relu — tinygrad unifies them into one loop nest.

2. **Opts applied by beam search.** `apply_opt` rewrites the AST by splitting a RANGE
   node into two (via `shift_to`), which adds a new RANGE and changes the type of the
   original. For example, `Opt(LOCAL, axis=0, arg=4)` splits the size-4 GLOBAL axis into
   a LOCAL axis of size 4 (consumed by the workgroup) — changing both `full_shape` and
   `axis_types` in place. This is why `Scheduler` attributes must be read _after_ opts
   are applied to reflect what will actually be compiled.

3. **Tensor core opts.** `_apply_tc_opt` injects a `WARP` RANGE to represent the lane
   dimension of a WMMA instruction, further restructuring the nest.

### What a kernel cannot express: the softmax case

Some computations inherently require two sequential passes over the data and cannot be
collapsed into a single loop nest. Softmax is the canonical example:

```
max_val = max(x, axis=-1)          # pass 1: full reduction to find max
result  = exp(x - max_val) / sum(exp(x - max_val))  # pass 2: needs complete max_val
```

Pass 2 cannot start until pass 1 is complete across _all_ elements of the reduction
dimension. tinygrad handles this by emitting **separate kernels** — one per pass — each
with its own `Scheduler`. For a `(4, 8)` softmax, beam search sees 6 kernels total
(including copies and intermediate buffers), of which two are reduction kernels:

| kernel            | axes                 | REDUCE op | ALU ops                    |
| ----------------- | -------------------- | --------- | -------------------------- |
| max reduction     | `REDUCE[8], LOOP[4]` | `Ops.MAX` | MUL, ADD                   |
| exp+sum+normalize | `REDUCE[8], LOOP[4]` | `Ops.ADD` | MUL, ADD, EXP2, RECIPROCAL |

The second kernel computes `exp(x - max_val)`, sums it (the ADD-reduce), and divides
(RECIPROCAL) — all in one loop nest because the `exp` and `sum` share the same reduction
axis and the `max_val` dependency is resolved by the time this kernel runs.

This is the general rule: **a kernel boundary appears wherever a later computation
depends on the complete result of a reduction**. Within those boundaries, tinygrad fuses
as much as possible into a single loop nest.

---

## Attribute tree

```
Scheduler
├── Core identity
│   ├── ast                  UOp — the kernel computation graph
│   └── ren                  Renderer — the device target
│
├── Optimization state
│   ├── applied_opts         list[Opt] — opts applied so far
│   ├── dont_use_locals      bool — NOLOCALS flag
│   └── opt_range            counter — source of fresh RANGE IDs
│
├── Axis structure  (all derived from RANGE nodes in ast)
│   ├── rngs                 list[UOp] — the loop axes in order
│   ├── shape_len            int — number of axes
│   ├── full_shape           list[int|sym] — size of each axis
│   └── axis_types           list[AxisType] — type of each axis
│
├── Axis selectors
│   ├── axes_of(*types)      list[int] — axis indices matching types
│   ├── ranges_of(*types)    list[UOp] — RANGE UOps matching types
│   ├── upcastable_dims      list[int] — axes eligible for UPCAST
│   └── unrollable_dims      list[int] — axes eligible for UNROLL
│
├── Kernel structure  (derived from deeper AST traversal)
│   ├── reduceops            list[UOp] — all REDUCE nodes
│   ├── reduceop             UOp|None — canonical single REDUCE
│   ├── bufs                 list[UOp] — INDEX nodes (one per buffer)
│   └── output_shape         list[int|sym] — shape with reduce dims → 1
│
└── Derived scalars
    ├── upcasted             int — number of UPCAST+UNROLL axes
    ├── group_for_reduces    int — number of GROUP_REDUCE axes
    └── upcast_size()        int — product of UPCAST+UNROLL sizes
```

`AxisType` values: `GLOBAL`, `LOCAL`, `WARP`, `LOOP`, `GROUP_REDUCE`, `REDUCE`, `UPCAST`, `UNROLL`, `THREAD`

---

## Core identity

### `ast: UOp`

The full kernel computation graph as a UOp DAG. Every other property on `Scheduler` is
derived from it by traversal.

**Why tinygrad records it:** `apply_opt` rewrites `self.ast` in-place by substituting
RANGE nodes (via `ast.substitute`). After opts are applied, `get_optimized_ast` stamps
final metadata (`KernelInfo`) onto it and hands it to `full_rewrite_to_sink` for
lowering. The AST is the single source of truth for the kernel at every stage.

---

### `ren: Renderer`

The device renderer. Carries hardware constants and capability flags.

**Why tinygrad records it:** `apply_opt` consults `ren` to validate constraints before
applying opts:

- `ren.has_local` — gating LOCAL and GROUP opts
- `ren.has_threads` — gating THREAD opt
- `ren.shared_max` — checking GROUP/GROUPTOP won't exceed shared memory
- `ren.global_max` — checking THREAD count is legal
- `ren.tensor_cores` — selecting the right TC configuration
- `ren.target.device` — skipping TF32 on non-CUDA, detecting AMX

**For features:** Device type (as embedding), `shared_max`, and `global_max[0]` are hard
constraints on the legal action space. Include them so the model doesn't predict impossible
configurations.

---

## Optimization state

### `applied_opts: list[Opt]`

The sequence of `Opt(op, axis, arg)` applied to this scheduler instance.

**Why tinygrad records it:**

- Beam search persists the winning opt sequence to disk cache (`diskcache_put`) and
  replays it on cache hit (`apply_opt` loop in `beam_search`).
- The heuristic optimizer checks `applied_opts[-1].op is OptOps.THREAD` to know when
  to stop the threading loop.
- `get_optimized_ast` embeds the list into `KernelInfo` so downstream passes know what
  was applied.

---

### `dont_use_locals: bool`

Set to `True` when the `NOLOCALS` opt is applied.

**Why tinygrad records it:** Controls downstream code generation — `gpudims.py` skips
local memory assignment when this flag is set. Several opt validity checks in `apply_opt`
also gate on it (`check(not self.dont_use_locals, ...)`).

---

### `opt_range: counter`

An integer counter used to generate unique IDs for new RANGE nodes created during
`shift_to`.

**Why tinygrad records it:** Every RANGE in the AST must have a unique ID tuple so the
UOp deduplication machinery does not collapse distinct loop axes together.

---

## Axis structure

These four properties are all derived from the same source: the `RANGE` nodes in
`ast.backward_slice`, sorted by `(axis_to_pos[axis_type], *axis_id)`.

### `rngs: list[UOp]`

The ordered list of RANGE UOps that define the loop nest.

**Why tinygrad records it:** `apply_opt` looks up axes by index into `rngs`
(`self.rngs[real_axis]`) before rewriting the AST. The heuristic also iterates `rngs`
directly to find axes absent from buffer index expressions (to decide what to localize).

---

### `shape_len: int`

`len(rngs)`. The depth of the kernel's loop nest — how many nested loops the kernel
contains in total, across all operations being computed inside it.

A kernel _is_ a loop nest. For a `(M,K) @ (K,N)` matmul the loops are
`for i in M / for j in N / for k in K`, giving `shape_len = 3`. If tinygrad fuses
multiple operations into one kernel (e.g. a reduction followed by an elementwise op),
their loops are unified and `shape_len` reflects the combined depth. Two independent
matmuls would never share a kernel — `create_schedule` emits separate kernels for
independent outputs, each with its own Scheduler.

**Why tinygrad records it:** Beam search uses it to skip actions that target out-of-range
axes (`ax >= s.shape_len`).

---

### `full_shape: list[int|sym]`

The size of each axis, in `rngs` order. Symbolic for dynamic shapes; plain `int` for
static ones.

**Why tinygrad records it:** Used pervasively — divisibility checks before applying opts
(`rng.src[0].divides(amount)`), upcast budget checks (`up > max_up`), local memory size
checks, heuristic thresholds (e.g. `full_shape[axis] <= 7` for masked upcast).

**For features:** `log2` of per-type size products. Key derived quantities:

- `global_size = prod(full_shape[i] for i in axes_of(GLOBAL))` — parallelism / output size
- `reduce_size = prod(full_shape[i] for i in axes_of(REDUCE))` — inner loop trip count
- `reduce_size / global_size` — arithmetic intensity proxy (roofline position)

---

### `axis_types: list[AxisType]`

The type of each axis. Determines how the loop maps to hardware.

| type           | hardware mapping                               |
| -------------- | ---------------------------------------------- |
| `GLOBAL`       | GPU global dispatch index / CPU outer loop     |
| `LOCAL`        | shared-memory tile loop (GPU workgroup)        |
| `WARP`         | warp-level parallelism (tensor core lanes)     |
| `LOOP`         | uncategorized sequential loop                  |
| `GROUP_REDUCE` | staged partial reduction through shared memory |
| `REDUCE`       | sequential accumulation (inner reduce loop)    |
| `UPCAST`       | fully unrolled into vector registers           |
| `UNROLL`       | unrolled reduce loop                           |
| `THREAD`       | explicit thread axis (CPU threading)           |

**Why tinygrad records it:** Every opt validity check consults axis type. E.g. UPCAST
requires `GLOBAL/LOCAL/LOOP`; UNROLL requires `GROUP_REDUCE/REDUCE`; LOCAL requires
`GLOBAL/LOOP`. Beam search accumulates `up` (product of UPCAST+UNROLL sizes) and `lcl`
(product of WARP+LOCAL+GROUP_REDUCE sizes) from `axis_types` to enforce budget limits.

**For features:** The count of axes per type and the product of sizes per type are
compact, fixed-size feature vectors regardless of `shape_len`. The ordering (which types
come first) encodes relative importance for memory locality.

---

## Axis selectors

### `axes_of(*types) → list[int]`

Indices of axes whose type is in `types`. Used everywhere to find, e.g., the reduce axes
or the upcastable axes.

### `ranges_of(*types) → list[UOp]`

The RANGE UOps for axes of the given types.

### `upcastable_dims: list[int]`

`axes_of(GLOBAL, LOCAL, LOOP)` filtered to axes with integer size > 1.

**Why tinygrad records it:** The heuristic iterates this to find candidate axes for
UPCAST opts. Beam search implicitly covers the same axes through the precomputed `actions`
list filtered by `real_axis`.

### `unrollable_dims: list[int]`

`axes_of(GROUP_REDUCE, REDUCE)` filtered to axes with integer size > 1.

**Why tinygrad records it:** Same role as `upcastable_dims` but for UNROLL opts. The
heuristic checks `unrollable_dims[-1]` (the innermost reducible axis) to decide whether
to unroll the reduce loop.

---

## Kernel structure

### `reduceops: list[UOp]`

All `REDUCE` UOps in the AST (via `backward_slice`).

**Why tinygrad records it:** Multi-reduce kernels have constraints — GROUP inside another
REDUCE is currently disallowed. The TC opt also checks `reduceops[0]` to find the
`Ops.ADD` + `Ops.MUL` pattern required for WMMA.

### `reduceop: UOp | None`

A canonical single REDUCE node (returns `None` for elementwise kernels).

**Why tinygrad records it:** The heuristic uses it to detect matvec patterns, check
shared-memory budget for GROUP opts, and identify FMA-fusable ADD-reduces. `None` means
the kernel is elementwise — no inner accumulation loop.

**For features:** Presence/absence is the largest kernel class split (elementwise vs.
reduction). Reduce op type (`Ops.ADD` vs `Ops.MAX`) distinguishes FMA-fusable from
non-fusable reductions. `reduceop.dtype` captures output precision.

---

### `bufs: list[UOp]`

The `INDEX` UOps in the AST (reversed toposort), one per buffer access. Each has:

- `src[0]`: a `PARAM` UOp identifying the buffer (arg = parameter index, dtype = pointer type)
- `src[1]`: the index expression — an affine combination of RANGE nodes

**Why tinygrad records it:** The heuristic reads index expressions directly to decide
which axes to localize and which to upcast. Concretely:

```python
# check whether a range appears in a buffer's index (for LOCAL decision)
rng not in b.src[1].get_idx().backward_slice

# sum strides to rank upcast candidates
for c in idx.split_uop(Ops.ADD):
    if c.op is Ops.MUL and c.src[0] is rng: sum_strides += c.src[1].arg
```

**For features:** The stride matrix — which axes appear in each buffer's index, and with
what coefficient — is the richest fixed-size structural feature available. Extracting it:
walk `idx.src[1].toposort()`, collect `MUL(RANGE, CONST)` pairs (stride > 1) and bare
`RANGE` nodes (stride = 1). Key signals:

- REDUCE axis absent from an input → broadcast (free reuse, very cache-friendly)
- REDUCE axis with stride=1 in an input → streaming inner loop (cache-friendly)
- GLOBAL axis with large stride in an input → strided GPU load (coalescing risk)

---

### `output_shape: list[int|sym]`

`full_shape` with reduce-type axes replaced by 1.

**Why tinygrad records it:** The heuristic gates GROUP opts and upcast decisions on the
total output element count (`prod(output_shape[i] for i in upcastable_dims)`). An output
too small means there's not enough parallelism to benefit from grouping.

---

## Derived scalars

### `upcasted: int`

`len(axes_of(UPCAST, UNROLL))`. The number of axes collapsed into registers.

**Why tinygrad records it:** The heuristic conditions unroll decisions on
`not k.axes_of(AxisType.UNROLL)` and upcast decisions on `not k.upcasted`.

### `group_for_reduces: int`

`len(axes_of(GROUP_REDUCE))`. The depth of staged partial reductions.

**Why tinygrad records it:** The heuristic returns immediately after applying GROUP opts
(`if k.group_for_reduces: return k`) — GROUP changes the kernel structure enough that
the remaining heuristics don't apply. Beam search also deducts it from `lcl` budget.

### `upcast_size() → int`

`prod(full_shape[i] for i in axes_of(UPCAST, UNROLL))`. Total vectorization width.

**Why tinygrad records it:** Beam search enforces `up // tc_up <= BEAM_UPCAST_MAX`. The
heuristic conditions further upcast opts on `upcast_size() < 32` (or `< 64` for unroll).

---

## Conditionally set

### `tensor_core: TensorCore`

Set on the instance by `_apply_tc_opt` when a TC opt successfully matches. Carries:
`dims`, `dtype_in`, `dtype_out`, `threads`, `opts`.

**Why tinygrad records it:** Beam search reads `hasattr(s2, 'tensor_core')` to compute
`tc_up = prod(tc.dims) // tc.threads` and deduct pre-consumed upcast capacity from the
budget. Its presence also gates further TC-incompatible opts (no GROUP with TC).

---

## Not a property but worth knowing: op mix

Walk `ast.toposort()` and bin nodes by `GroupOp.ALU`. This is not a Scheduler property
but is cheap to extract and captures arithmetic character:

- `MUL` + `ADD` (or `MULACC`) → FMA-fusable, benefits from vectorization
- `EXP2`, `LOG2`, `SIN`, `SQRT`, `RECIPROCAL` → transcendental, ~10-20× slower than FMA,
  limits vectorization benefit
- `WHERE` → predication/masking, can inhibit loop elimination
- `CAST`/`BITCAST` → type conversion overhead

A kernel with high transcendental count and identical shape to a pure matmul will have
very different runtime behavior — the op mix is what distinguishes them.
