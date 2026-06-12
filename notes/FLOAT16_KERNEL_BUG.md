# Bug Report: CUDA float16 kernels fail to compile on NVRTC 12.3+

**Affected component:** `tinygrad/renderer/cstyle.py` — `CUDARenderer`  
**Symptom:** All kernels using `dtypes.float16` (half precision) silently fail to compile when running on Linux with an NVIDIA GPU and CUDA ≥ 12.3. Beam search benchmarks report every op as failed.  
**Platform confirmed:** NVIDIA TITAN V (sm_70), CUDA 12.4, NVRTC 12.4  
**Not affected:** macOS (uses Metal, not NVRTC)

---

## Symptom

Running the beam search benchmark on Linux with a CUDA GPU:

```
python benchmarks/beam_search/benchmark_revised.py --ops relu_large --beams 0
  ✗ relu_large beam=0: compile FAILED
```

The subprocess exit code is non-zero and stdout is empty (no JSON). No Python exception is raised in the orchestrator — the failure is silent from the benchmark's perspective.

Running `run_single.py` directly reveals the underlying error:

```
tinygrad.device.CompileError: Nvrtc Error 6, NVRTC_ERROR_COMPILATION
ptxas fatal   : Unresolved extern function '_ZN6__halfC1Ef'
```

`_ZN6__halfC1Ef` is the C++ mangled name for `__half::__half(float)` — the constructor that converts a `float` to CUDA's `__half` (float16) type.

The same error appears for all `__half` constructors:

| Mangled name | Constructor |
|---|---|
| `_ZN6__halfC1Ef` | `__half(float)` |
| `_ZN6__halfC1Ed` | `__half(double)` |
| `_ZN6__halfC1Ei` | `__half(int)` |

---

## Root Cause

### Background

NVRTC is NVIDIA's runtime compiler: it takes a CUDA C++ source string and compiles it in-process to PTX (an intermediate GPU assembly format), which is then assembled to GPU machine code by `ptxas`. NVRTC compiles exactly one source string with no linker step.

tinygrad's CUDA renderer generates kernels that store float16 results using C++ cast syntax:

```c
#include <cuda_fp16.h>
// ...
output[i] = (half)(some_float_expression);
```

The `(half)(val)` C++ cast calls the `__half::__half(float)` constructor, defined in `cuda_fp16.hpp`.

### The CUDA 12.3 change

Before CUDA 12.3, the `__half` constructors were marked `inline` in the CUDA headers. NVRTC would expand them directly into the kernel — no call instruction emitted, no external reference needed.

In CUDA 12.3, NVIDIA changed the macro that controls this:

```cpp
// cuda_fp16.h, lines 4112-4118
#if defined(__CUDACC_RTC__) && CUDA_VERSION >= 12.3
  #define __CUDA_FP16_INLINE__           // empty — NOT inline
#else
  #define __CUDA_FP16_INLINE__ inline
#endif
```

`__CUDACC_RTC__` is defined whenever NVRTC is the compiler. So on CUDA 12.3+, every `__half` constructor becomes a non-inline `__device__` function.

### Why non-inline breaks NVRTC

When a `__device__` function is not inlined, the compiler emits a **call instruction** that references the function by its mangled name. For NVRTC this is fatal: NVRTC produces PTX for a single kernel and has no linker. `ptxas` sees the call to `_ZN6__halfC1Ef`, can't find the function body anywhere in the PTX, and aborts.

The function body *is* present in the included `cuda_fp16.hpp` — but without the `inline` keyword the compiler treats it as a separately-compiled callable rather than something to expand at every callsite.

### Verification

Compiling the cast syntax vs. the intrinsic directly via NVRTC on the affected machine:

```python
# (half)(val)   →  FAIL: ptxas fatal: Unresolved extern function '_ZN6__halfC1Ef'
# __float2half(val)  →  OK
```

The intrinsic `__float2half` is a proper GPU built-in that NVRTC resolves correctly regardless of the `__CUDA_FP16_INLINE__` setting.

---

## Fix

**File:** `tinygrad/renderer/cstyle.py`  
**Class:** `CUDARenderer`

Added a `render_cast` override that replaces C++ constructor cast syntax with the NVRTC-safe intrinsic when the target type is `half`:

```python
def render_cast(self, dt:DType, val:str) -> str:
    # NVRTC 12.3+ sets __CUDA_FP16_INLINE__ to empty, making __half constructors non-inline device
    # functions that ptxas can't resolve. Use __float2half() intrinsic instead of C++ cast syntax.
    if dt == dtypes.half: return f"__float2half((float)({val}))"
    return super().render_cast(dt, val)
```

The `(float)(...)` wrapper ensures correctness for all source types:
- Source is `float`: the `(float)` cast is a no-op, `__float2half` receives a float directly.
- Source is `double` or `int`: explicitly narrowed to `float` first, matching `__float2half`'s expected argument type.

This override applies to scalar `half` only (`dt == dtypes.half`). Vectorized casts (e.g. float4 → half4) already go through `__builtin_convertvector` via a separate pattern in `base_rewrite` and are unaffected.

The fix is correct for both NVRTC and `nvcc` compilation paths — `__float2half` is valid in both contexts.

---

## Testing

After the fix:

```
python benchmarks/beam_search/run_single.py --op relu_large --beam 0 --mode compile --n-exec 1 --n-warmup 0
{"op": "relu_large", "beam": 0, "mode": "compile", "compile_time_s": 0.057}

python benchmarks/beam_search/run_single.py --op matmul_1024 --beam 2 --mode compile --n-exec 1 --n-warmup 0
{"op": "matmul_1024", "beam": 2, "mode": "compile", "compile_time_s": 33.16}
```

Core test suite (`test/test_tiny.py`): 17 passed, 2 skipped — no regressions.
