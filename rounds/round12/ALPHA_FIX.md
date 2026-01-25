# ROUND 12 - AGENT ALPHA FIX DOCUMENTATION

## Problem Analysis

The error occurred because `custom_kernel` in `submission.py` was receiving a 4-element tuple from the production evaluation system, but the code expected 10 elements:

```
ValueError: not enough values to unpack (expected 10, got 4)
  File "submission.py", line 938
    a, b1, b2, _, _, _, sfa_permuted, sfb1_permuted, sfb2_permuted, c = data
```

## Investigation Findings

### Format 1: eval_test Format (10 elements)
From `eval_test/eval/nvfp4_dual_gemm/reference.py`:
```python
(a, b1, b2, sfa_cpu, sfb1_cpu, sfb2_cpu, sfa_perm, sfb1_perm, sfb2_perm, c)
```
- Flat tuple with all tensors directly accessible
- Used by local testing and eval_test framework

### Format 2: Production GROUP GEMM Format (4 elements)
From `nvfp4_grouped_gemm/task.py` and archive analysis:
```python
(abc_tensors, sfasfb_cpu_tensors, sfasfb_reordered_tensors, problem_sizes)
```

For GROUP GEMM, this is:
- Element 0: `list[(a, b, c)]` - List of tensor tuples per group
- Element 1: `list[(sfa_cpu, sfb_cpu)]` - CPU scale factors (unused by kernel)
- Element 2: `list[(sfa_perm, sfb_perm)]` - Permuted/reordered scale factors
- Element 3: `list[(m, n, k, l)]` - Problem sizes per group

### DUAL GEMM Adaptation
For DUAL GEMM (`C = silu(A @ B1) * (A @ B2)`), the 4-element format represents 2 "groups":
- Group 0: GEMM1 = A @ B1 -> `(a, b1, c)`, `(sfa_perm, sfb1_perm)`
- Group 1: GEMM2 = A @ B2 -> `(a, b2, c)`, `(sfa_perm, sfb2_perm)`

Both groups share the same `a`, `c`, and `sfa_perm` tensors.

## Solution Implemented

Modified `custom_kernel()` in `/home/ubuntu/projects/Shark-Tank-for-GPUMODE.COM/src/submission.py` to:

1. **Detect format by length**: `len(data) == 10` vs `len(data) == 4`

2. **Handle 10-element format** (unchanged):
   ```python
   a, b1, b2, _, _, _, sfa_permuted, sfb1_permuted, sfb2_permuted, c = data
   ```

3. **Handle 4-element GROUP GEMM format**:
   - Detect if `abc_tensors` contains tuples (GROUP style) or flat tensors
   - For GROUP style with 2 groups:
     ```python
     (a, b1, c) = abc_tensors[0]
     (_, b2, _) = abc_tensors[1]  # a and c are shared
     (sfa_permuted, sfb1_permuted) = sfasfb_reordered_tensors[0]
     (_, sfb2_permuted) = sfasfb_reordered_tensors[1]  # sfa is shared
     ```
   - For flat tuple style: `(a, b1, b2, c)` or `(a, b1, b2)` with c allocation

## Key Design Decisions

1. **No torch.compile with max-autotune**: Per constraints, avoided to prevent 180s timeout

2. **Preserved kernel caching**: Uses existing `compile_kernel()` pattern

3. **Defensive unpacking**: Multiple format variations handled with clear error messages

4. **Single kernel execution**: After format detection, all paths converge to same kernel call:
   ```python
   compiled_func(a_ptr, b1_ptr, b2_ptr, sfa_ptr, sfb1_ptr, sfb2_ptr, c_ptr, (m, n, k, l))
   ```

## Files Modified

- `/home/ubuntu/projects/Shark-Tank-for-GPUMODE.COM/src/submission.py` - Lines 932-998

## Testing Recommendations

1. Test with 10-element format (eval_test):
   ```bash
   cd /home/ubuntu/projects/Shark-Tank-for-GPUMODE.COM/eval_test
   python -m eval.nvfp4_dual_gemm.eval test tests.txt
   ```

2. Test with 4-element GROUP format (if production test available)

3. Verify kernel correctness with reference implementation comparison

---
**Agent ALPHA - "The Investigator"**
*Round 12 Deathmatch Entry*
