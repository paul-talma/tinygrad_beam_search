# How tinygrad's scheduler handles fusion

## The core mental model: fusion is the default, realization is the exception

tinygrad starts with a **maximally fused** lazy UOp graph — every tensor op just adds a node; nothing executes. The scheduler's job is to decide **where to force materializations** (insert intermediate buffers), which is what "splits" the fused graph into separate kernels. It's not "should we fuse X and Y?" — it's "do we have to break apart this giant fused graph?"

---

## Phase 1: `get_kernel_graph` — turning the tensor graph into kernel boundaries

`rangeify.py:579` — `get_kernel_graph` is the entry point. It runs a sequence of rewrite passes:

**1. `run_rangeify`** (`indexing.py`) — converts all movement ops (reshape, permute, pad, shrink, expand) into index arithmetic using `RANGE` loop nodes. After this pass, ops that are "compatible" (same loop structure) naturally collapse into a single computation because they share the same `RANGE` nodes. This is the mechanism of elementwise fusion — no explicit "fuse X with Y" logic, just range sharing.

**2. `pm_generate_realize_map`** (`indexing.py:28`) — identifies what must be materialized:
- `CONTIGUOUS`, `COPY`, and `STORE` always force realization — these are the "hard cuts"
- Sources of `MSELECT`/`MSTACK` (multi-device ops) also force realization

**3. `remove_bufferize`** (`rangeify.py:243`) — the **fusion cost function**. A `STAGE` node represents a potential intermediate buffer. This function decides whether to keep it (don't fuse = separate kernel) or remove it (fuse into one kernel). It returns `None` (keep buffer) when:
- More than 3 input buffers accessed — `rangeify.py:275`
- Any reduce reads from a buffer — `rangeify.py:285` (reduces that read intermediate values have different memory access patterns and can cause correctness/bandwidth issues if fused)
- The `PCONTIG` flag enables more aggressive partial-contig fusion via local shared memory

**4. `split_reduceop`** (`rangeify.py:100`) — a perf-driven unfusion heuristic. For large reductions where output is small, it splits into two kernels (a partial reduce + a final reduce) to improve GPU occupancy. The threshold is `prod(input) / prod(output) >= 32768` by default.

**5. `limit_bufs`** (`rangeify.py:361`) — hardware-driven unfusion. Metal has a 31-buffer limit, WebGPU has 8. If a fused kernel would exceed the limit, it inserts intermediate `STAGE` nodes to break it up.

**6. `split_kernels`** (`rangeify.py:574`) — the final pass. Any `STORE`/`END` node with no open `RANGE` loops is a complete, independent computation. `split_store` wraps each one into a `CALL` (kernel) node.

---

## Phase 2: `create_schedule` — topological ordering

`__init__.py:21` — once `get_kernel_graph` has emitted a set of `CALL` nodes with explicit dependency edges (`AFTER` nodes encoding producer→consumer relationships), `create_schedule` does a standard Kahn's-algorithm topological sort to produce a linear execution order.

---

## Summary table

| Mechanism | Where | Direction |
|---|---|---|
| Range sharing via `run_rangeify` | `indexing.py` | Fuses compatible ops automatically |
| `CONTIGUOUS`/`COPY`/`STORE` rules | `indexing.py:28` | Forces realization (unfuses) |
| `remove_bufferize` cost function | `rangeify.py:243` | Keeps intermediate buffer if >3 inputs or buffer-in-reduce |
| `split_reduceop` | `rangeify.py:100` | Unfuses large reductions into 2-phase |
| `limit_bufs` | `rangeify.py:361` | Unfuses when hardware buffer limit exceeded |
| `split_kernels` | `rangeify.py:574` | Emits one kernel per closed computation |

The key insight from the CLAUDE.md: *"recompute-vs-materialize decision is unsolved"* — the current cost model in `remove_bufferize` is a rough heuristic (buffer count, reduce-access check), not a roofline-aware model. That's flagged as the biggest open research problem in the scheduler.
