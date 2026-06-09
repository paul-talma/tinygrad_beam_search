# Scheduling Pipeline

The scheduler converts a lazy UOp graph into a linear sequence of kernel calls. Entry point: `create_linear_with_vars(big_sink)` in `tinygrad/schedule/__init__.py`.

## High-level flow

```
Tensor UOp graph
  ↓  get_kernel_graph()       [rangeify.py]
Kernel graph (CALL nodes with AFTER ordering edges)
  ↓  create_schedule()        [__init__.py]
Linearized CALL sequence      (Kahn topological sort)
  ↓  memory_plan_rewrite()    [memory.py]
Final linear with shared arenas
```

`create_linear_with_vars` wraps `get_kernel_graph` + `create_schedule` + `memory_plan_rewrite`, handles schedule caching (keyed on `function.key`), and registers the linear with `TinyJit` if a capture is in progress.

---

## Stage 0 — Multi-device rewriting (`multi.py`, `allreduce.py`)

Runs first inside `get_kernel_graph` as `multi_pm`.

- Resolves `MULTI`/`MSTACK`/`MSELECT` ops: shards or replicates tensors across devices.
- Expands `ALLREDUCE` into one of three collective implementations chosen at runtime:
    - **Naive**: copy each shard to all devices and reduce element-wise.
    - **Ring allreduce**: rotate chunks around the ring in a reduce-scatter then allgather pass.
    - **All-to-all**: each device directly sends its chunk to every other device.
    - Selection heuristic: ring/all2all only when `ndev > 2` and `numel > 256_000`.

---

## Stage 1 — Earliest rewrites (`rangeify.py: earliest_rewrites`)

Applied bottom-up.

- Merge adjacent `RESHAPE` nodes.
- Inline `FUNCTION` calls (substitute params with args).
- Resolve `TUPLE`/`GETTUPLE`.
- **`split_reduceop`**: if a reduction's input is large enough (> `REDUCEOP_SPLIT_THRESHOLD` = 32 768 elements), split it into two kernels — first reducing along a split dimension, then doing the final reduce. Increases global thread parallelism.
- Fix store hazards (source reads from the same buffer being written → insert `contiguous()`).
- Handle zero-size tensors, `COPY`, `DETACH`, and `CONTIGUOUS_BACKWARD` cleanup.

---

## Stage 2 — Rangeify (`indexing.py: run_rangeify`)

The core fusion/recompute decision. Converts the tensor DAG into a loop-based representation by assigning `RANGE` nodes (loop iterators) to every op.

### Step 2a — Build realize map

`pm_generate_realize_map` marks which ops must produce a concrete buffer:

- `CONTIGUOUS`, `COPY`, `STORE` always realize.
- Sources of `COPY`, `MSELECT`, `MSTACK` realize.
- Self-referential stores (WAR hazard) realize.

### Step 2b — Assign ranges (bottom-up over toposort)

For each op, determine its output ranges:

| Situation                             | Action                                                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Op is in realize map                  | Create fresh ranges — this is a **materialization point**                                         |
| One consumer                          | Inherit consumer's ranges — **fusion**                                                            |
| Multiple consumers                    | Try to merge; if ranges differ across consumers, create fresh ranges — **forced materialization** |
| Ending ranges present (from `EXPAND`) | Also force partial materialization                                                                |

Input ranges are derived from output ranges by inverting the op semantics:

- **Movement ops** (`SHRINK`, `PERMUTE`, `FLIP`, `EXPAND`, `RESHAPE`, `PAD`) — `apply_movement_op` transforms the range indices symbolically (e.g. PERMUTE reorders them, SHRINK offsets them, RESHAPE recomputes indices via mod/div arithmetic).
- **REDUCE** — creates new `AxisType.REDUCE` ranges for the reduced axes.
- **Elementwise** — passes output ranges through unchanged.

The range map `rctx.range_map[x] = (input_rngs, output_rngs)` is built for the entire graph.

### Step 2c — Apply rangeify

`pm_apply_rangeify` rewrites the graph: inserts `BUFFERIZE` nodes at materialization points, applies range indices via `INDEX`, replaces `PAD` with `WHERE` (for valid masking), and removes movement ops (now baked into index math).

---

## Stage 3 — Symbolic simplification + buffer folding

```python
graph_rewrite(tsink, symbolic + pm_reduce_simplify + pm_const_buffer_folding + pm_remove_bufferize)
```

- `symbolic`: simplify index expressions (mod/div arithmetic, range bounds).
- `pm_const_buffer_folding`: fold constants through `BUFFERIZE`; collapse dead axes (axes not referenced by any index → replace range with 1 and reshape).
- **`remove_bufferize`** — the cost function for eliminating intermediate buffers:
    1. Never remove non-`removable` buffers (e.g. `CONTIGUOUS` targets).
    2. Keep if accessed from **> 3 distinct source buffers** (kernel would be too wide).
    3. Keep if **any reduce accesses a buffer** (fusing a reduction that reads external memory is risky).
    4. With `PCONTIG > 2`: partial contiguous mode — can keep some axes local while fusing others.
    5. Otherwise remove: substitute the closed ranges back into the consumer, eliminating the intermediate store.

This is the main **recompute-vs-materialize heuristic** and is explicitly noted in the codebase as the place where cost decisions live (see the `# *** here is where we compute the cost ***` comment).

---

## Stage 4 — Buffer limit enforcement (`pm_limit_bufs`)

Metal allows 31 buffers per kernel; WebGPU allows 8. If a kernel would exceed the device limit, force some inputs to materialize (insert `BUFFERIZE` with fresh ranges) to split the kernel.

---

## Stage 5 — Bufferize → STORE (`pm_add_buffers`)

Convert `BUFFERIZE` nodes into actual memory operations:

- **Global** (`AddrSpace.GLOBAL`): allocate a new `BUFFER` (`LUNIQUE` placeholder), emit a `STORE … END` sequence, return an `AFTER` node encoding the dependency.
- **Local** (`AddrSpace.LOCAL`): allocate shared memory (`placeholder`), emit store + barrier.
- **Disk/TinyFS** targets: convert to `BUFFER_VIEW`.

---

## Stage 6 — Kernel splitting (`split_store` / `split_kernels`)

For every `STORE` or `END` with no open ranges, `split_store` extracts a self-contained kernel:

1. Runs `to_define_global` bottom-up: replaces `BUFFER` nodes with `PARAM` (numbered args), renumbers `RANGE` nodes from 0, collects variable bindings.
2. Wraps the result in `SINK` with `KernelInfo`.
3. Wraps that in `CALL` with the buffer list.

After all kernels are split, WAR (write-after-read) dependencies are detected: if kernel A writes buffer S and kernel B reads S, B's write-AFTER is updated to also wait for A.

---

## Stage 7 — Topological sort (`create_schedule`)

Kahn's algorithm on the `AFTER`/`CALL` dependency graph:

1. Parse `AFTER` nodes to build `children` and `in_degree` maps.
2. Start with all zero-in-degree kernels in a deque.
3. Pop, emit, decrement children, add newly-zero nodes.

The result is a `UOp(Ops.LINEAR, src=(kernel_calls...))`.

---

## Stage 8 — Memory planning (`memory_plan_rewrite`)

Suballocates intermediate buffers into shared arenas to reduce peak memory usage.

1. Compute first/last appearance of each buffer across the linear schedule.
2. Separate copy-engine and compute buffers into different lanes (prevents introducing copy→compute→copy ordering cycles).
3. TLSF (two-level segregated fit) allocator assigns non-overlapping offsets within a per-device-lane arena.
4. Replace each `BUFFER` with a `BUFFER_VIEW` into the shared arena.

---

## Key data structures

| Node                            | Meaning                                                                                                 |
| ------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `RANGE(shape, idx, AxisType)`   | A loop variable; AxisType is LOOP, REDUCE, UPCAST, UNROLL, GLOBAL, LOCAL, THREAD, WARP                  |
| `BUFFERIZE(src, *ranges)`       | Marks a value that may need to be stored; has a removability flag                                       |
| `AFTER(buf, *stores_or_afters)` | Dependency: buf is available after these ops complete                                                   |
| `CALL(sink, *buffers)`          | A self-contained kernel call                                                                            |
| `IndexingContext`               | Holds `realize_map` (which ops materialize and which axes) and `range_map` (per-op input/output ranges) |

---

## The recompute-vs-materialize problem (research target)

The current cost model in `remove_bufferize` uses simple integer thresholds (buffer count, presence of reduces). It does **not** consider:

- **Roofline model**: arithmetic intensity of the fused kernel vs. memory bandwidth cost of materializing.
- **Reuse distance**: how many times a recomputed value is actually used.
- **Register pressure**: deeply fused kernels may spill registers, reducing occupancy.
- **Tile/cache effects**: a materialize can be cache-friendly if the next consumer accesses the same memory in a compatible pattern.
- **GPU-specific occupancy**: local memory usage, warp divergence from complex index expressions.

The heuristic also lives in one small function (`remove_bufferize`, ~50 lines in `indexing.py:235–299`), making it a well-contained target for a learned or analytical replacement.
