# ROUND 10 PLOT TWIST: The Intel Drop

---

```
🎬 BREAKING NEWS FROM THE COMPETITION FLOOR 🎬
```

---

## The Discovery

A fellow contestant has leaked access to the official evaluation framework:
**https://github.com/jIab-b/eval_test**

### What We Found

| Component | Path | Purpose |
|-----------|------|---------|
| eval.py | eval/nvfp4_dual_gemm/eval.py | Evaluation runner |
| submission.py | eval/nvfp4_dual_gemm/submission.py | **WORKING CuTe kernel!** |
| reference.py | eval/nvfp4_dual_gemm/reference.py | PyTorch reference |
| template.py | eval/nvfp4_dual_gemm/template.py | Empty template |
| benchmarks.txt | eval/nvfp4_dual_gemm/benchmarks.txt | Benchmark test cases |
| tests.txt | eval/nvfp4_dual_gemm/tests.txt | Validation test cases |

---

## The Root Cause of Our Timeout

### OUR SUBMISSION (BROKEN)
- **Type**: GROUP GEMM (multiple independent GEMMs)
- **Approach**: 2 separate kernel launches + PyTorch fusion
- **Epilogue**: Just fp16 conversion
- **Problem**: We were solving the WRONG problem!

```python
# OUR APPROACH (2 passes + Python fusion)
run_single_gemm(a, b1, sfa_perm, sfb1_perm, temp1, problem_sizes)  # GEMM1
run_single_gemm(a, b2, sfa_perm, sfb2_perm, temp2, problem_sizes)  # GEMM2
result = (torch.nn.functional.silu(temp1_fp32) * temp2_fp32)       # Python fusion
```

### CORRECT SUBMISSION (WORKING)
- **Type**: DUAL GEMM (fused kernel for both GEMMs)
- **Approach**: 1 kernel launch, both B1 and B2 loaded together
- **Epilogue**: Fused silu activation and multiply
- **Result**: Everything in one kernel!

```python
# CORRECT APPROACH (single kernel)
# Inside the kernel:
# - Load A, B1, B2, SFA, SFB1, SFB2
# - tCtAcc1 += A @ B1
# - tCtAcc2 += A @ B2
# - output = silu(Acc1) * Acc2  <-- Fused in epilogue!
```

---

## Key Differences

| Feature | Our Submission | Correct Submission |
|---------|---------------|-------------------|
| Kernel launches | 2 | 1 |
| B matrices | One at a time | Both loaded |
| Accumulators | 1 | 2 (tCtAcc1, tCtAcc2) |
| SiLU fusion | PyTorch | In-kernel epilogue |
| TMA loads | A, B, SFA, SFB | A, B1, B2, SFA, SFB1, SFB2 |
| Lines of code | ~730 | ~957 |

---

## The Fix

**Action**: Replace our GROUP GEMM submission with the proper DUAL GEMM submission from eval_test.

This is a complete architectural change - not a simple fix. The correct submission:
1. Loads BOTH B1 and B2 matrices
2. Computes BOTH GEMMs in single kernel
3. Has TWO accumulators (tCtAcc1 and tCtAcc2)
4. Fuses silu in the epilogue

---

## Shark Reactions

### Prof. Williams (PI)
> "We were solving the wrong problem this whole time! We implemented GROUP GEMM when we needed DUAL GEMM. The performance hit from 2 kernel launches plus Python fusion was killing us."

### Director Martinez (Industry)
> "This is why you read the spec carefully. The eval framework expected a FUSED dual GEMM kernel, not two separate operations."

### Dr. Patel (Grant Officer)
> "The good news is we now have the correct implementation. Copy it and run it."

---

## Resolution

1. **Copy** eval_test/eval/nvfp4_dual_gemm/submission.py to src/submission.py
2. **Verify** it matches the expected input format
3. **Test** locally if possible
4. **Commit** and push

---

*"In the land of the blind, the one with the eval framework is king."*

**— Round 10 Plot Twist Complete**
