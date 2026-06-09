# Tinygrad Project — Session Context

## Background

Two-person, five-week project.

## Tinygrad Stack (read: mesozoic-egg.github.io/tinygrad-notes + docs.tinygrad.org)

- Frontend: Tensor ops build a lazy UOp DAG (op, dtype, src, arg)
- Scheduler: partitions UOp graph into ExecItems (one per kernel); handles fusion; CONTIGUOUS acts as fusion barrier
- Lowering: PatternMatcher rewrites UOp AST → source string; beam search (BEAM=n) tries OptOps combinations
- Execution: compiled binary → GPU dispatch; TinyJit captures and replays the command graph

## ShapeTracker

Zero-copy shape manipulation via (shape, strides, offset, mask) Views. Multi-view ShapeTracker handles cases not expressible as a single affine index map.

## Project Ideas Considered

1. Scheduler cost model — formalize the recompute-vs-materialize decision using a roofline-aware cost model. Explicitly acknowledged as open by tinygrad team. High research interest.
2. Learned beam search cost model — predict kernel runtime from features to replace exhaustive benchmarking. Connects to TVM/Ansor literature.
3. Flash Attention as a primitive — pattern-match the attention subgraph, emit a hand-written kernel (Option A). Scope risk: reliable pattern detection in the UOp DAG.
4. New backend (ROCm/HIP, Vulkan, WebGPU, LLVM IR)
5. ONNX import, INT8 quantization, sparse tensors

## Speed Bottlenecks (from docs.tinygrad.org/developer/speed)

- Compile speed: UOp graph rewrite passes (Python), on par with torch.compile
- Execution speed: not a bottleneck (TinyJit is fast)
- Model speed (scheduler): biggest training bottleneck; recompute-vs-materialize is unsolved
- Kernel speed (codegen): no explicit SRAM tiling; limited tensor core support; no TMA support

## Useful Debug Flags

DEBUG=2 (launch stats), DEBUG=4 (generated source), DEBUG=5 (UOp AST), NOOPT=1, BEAM=n, VIZ=1
