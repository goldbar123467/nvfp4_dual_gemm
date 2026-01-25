# Gap 4: Precision Policy for Fusion

## Reference Implementation Precision
```python
# From task.py ref_kernel:
ref1 = torch.empty(..., dtype=torch.float32)  # FP32 accumulator
ref2 = torch.empty(..., dtype=torch.float32)  # FP32 accumulator

# GEMM in FP32
res1 = torch._scaled_mm(..., out_dtype=torch.float32)
res2 = torch._scaled_mm(..., out_dtype=torch.float32)

# SiLU in FP32, multiply in FP32
c_ref = (torch.nn.functional.silu(ref1) * ref2).to(torch.float16)
```

## SiLU Formula
```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

Numerical considerations:
- exp(-x) can overflow for large negative x
- Division can lose precision in FP16
- Recommended: compute in FP32, cast result
```

## Precision Policy Options

### Option A: Full FP32 (Safest - RECOMMENDED)
```cpp
// Accumulator: FP32
using AccumulatorType = float;

// Epilogue fusion in FP32
float d1 = gemm1_result;  // FP32
float d2 = gemm2_result;  // FP32
float silu_d1 = d1 / (1.0f + expf(-d1));  // FP32 SiLU
float result = silu_d1 * d2;  // FP32 multiply
half output = __float2half(result);  // Cast to FP16
```
- Pros: Matches reference exactly
- Cons: More register pressure, slightly slower epilogue

### Option B: Mixed Precision (Faster, riskier)
```cpp
// Accumulator: FP32 (required for Tensor Cores)
using AccumulatorType = float;

// Epilogue in FP16
half d1 = __float2half(gemm1_result);
half d2 = __float2half(gemm2_result);
half silu_d1 = __hmul(d1, __hrcp(__hadd(__float2half(1.0f), hexp(__hneg(d1)))));
half output = __hmul(silu_d1, d2);
```
- Pros: Faster epilogue, less register pressure
- Cons: May exceed error tolerance on some inputs

### Option C: Hybrid (Balanced)
```cpp
// Accumulator: FP32
// SiLU: FP32 (for exp stability)
// Final multiply: FP16

float d1 = gemm1_result;
float d2 = gemm2_result;
float silu_d1 = d1 / (1.0f + expf(-d1));  // FP32 SiLU
half output = __float2half(silu_d1) * __float2half(d2);  // FP16 mul
```

## Validation Gate
```python
def test_precision_policy():
    # Test on exact benchmark shapes
    for m, n, k, l in [(256,4096,7168,1), (512,4096,7168,1)]:
        inputs = generate_input(m, n, k, l, seed=42)
        golden = ref_kernel(inputs)
        kernel_out = my_kernel(inputs)
        
        # Must pass with leaderboard tolerances
        assert torch.allclose(kernel_out, golden, rtol=1e-3, atol=1e-3), \
            f"Precision mismatch at {m}x{n}x{k}"
        
        # Also check max absolute error
        max_err = (kernel_out - golden).abs().max()
        print(f"{m}x{n}x{k}: max_err={max_err:.6f}")
```

## Decision: Lock This Down
```
CHOSEN POLICY: Option A (Full FP32)
- Accumulator: FP32
- SiLU: FP32
- Multiply: FP32
- Output cast: FP16

Rationale: Matches reference exactly, guarantees correctness.
Optimize for speed only AFTER correctness is confirmed.
```

## Implementation Notes

### CUTLASS Accumulator Type
When using CUTLASS for the dual GEMM, set the accumulator type explicitly:
```cpp
using ElementAccumulator = float;  // FP32 accumulator
using ElementCompute = float;      // FP32 for epilogue math
using ElementOutput = cutlass::half_t;  // FP16 output
```

### Epilogue Functor
Custom epilogue must preserve FP32 through SiLU computation:
```cpp
template <typename Fragment>
CUTLASS_DEVICE void operator()(
    Fragment& d1_frag,  // FP32 from GEMM1
    Fragment& d2_frag   // FP32 from GEMM2
) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Fragment::kElements; ++i) {
        float d1 = d1_frag[i];
        float d2 = d2_frag[i];
        // SiLU in FP32
        float silu = d1 / (1.0f + expf(-d1));
        // Multiply in FP32, then cast
        output[i] = __float2half(silu * d2);
    }
}
```

### Register Pressure Considerations
FP32 epilogue uses 2x registers vs FP16. Monitor occupancy:
- Target: 50%+ occupancy for Hopper
- If register-bound, consider reducing tile size before switching to FP16 epilogue

### Error Budget
With leaderboard tolerances (rtol=1e-3, atol=1e-3):
- FP32 epilogue: ~0.0001 max error (well within budget)
- FP16 epilogue: ~0.01 max error (may exceed on edge cases)
- Hybrid: ~0.001 max error (borderline)

Choose FP32 for guaranteed correctness.
