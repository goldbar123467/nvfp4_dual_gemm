# SHARK TANK ROUND 4: CONTESTANT C - MINIMAL FIX

---

```
 __  __ ___ _   _ ___ __  __    _    _        _____ _____  __
|  \/  |_ _| \ | |_ _|  \/  |  / \  | |      |  ___|_ _\ \/ /
| |\/| || ||  \| || || |\/| | / _ \ | |      | |_   | | \  /
| |  | || || |\  || || |  | |/ ___ \| |___   |  _|  | | /  \
|_|  |_|___|_| \_|___|_|  |_/_/   \_\_____|  |_|   |___/_/\_\

"After 3 rounds of failures, simplicity is the only path forward."
```

---

## EXECUTIVE SUMMARY

After 3 failed optimization rounds, I propose we ask a different question:

**What is the ABSOLUTE MINIMUM change to make this kernel correct?**

I present two options:
1. **Two-Pass Approach**: Call the existing kernel TWICE, fuse results externally
2. **Minimal Single-Pass**: Duplicate the mainloop with minimal changes

**My Recommendation**: Start with Two-Pass to validate correctness, THEN optimize.

---

## PART 1: MINIMAL CODE ANALYSIS

### Current Kernel Structure (What We Have)

```
Lines 41-61:   kernel() signature - takes ONE B matrix, ONE SFB
Lines 119-140: SharedStorage - allocates space for sA, sB, sSFA, sSFB (single)
Lines 143-157: Pipeline setup - single producer/consumer pair
Lines 231-251: TMA partitions for A, B, SFA, SFB (single)
Lines 313-352: Main loop - computes ONE GEMM (A @ B)
Lines 354-389: Epilogue - stores result (no SiLU, no multiply)
Lines 528-539: custom_kernel() - passes ONE b, ONE sfb per group
```

### What's Missing

| Component | Current State | Required |
|-----------|---------------|----------|
| B2 input | Missing | N x K x L tensor |
| SFB2 input | Missing | Scale factors for B2 |
| Second GEMM | Missing | A @ B2 computation |
| SiLU activation | Missing | `x * sigmoid(x)` on GEMM1 |
| Element-wise multiply | Missing | `silu(GEMM1) * GEMM2` |

---

## PART 2: TWO-PASS APPROACH

### Concept

Run the existing kernel TWICE:
1. First pass: Compute `temp1 = A @ B1`
2. Second pass: Compute `temp2 = A @ B2`
3. External fusion kernel: `C = silu(temp1) * temp2`

### Implementation

**Changes to submission.py (custom_kernel function):**

```python
def custom_kernel(data: input_t) -> output_t:
    """Execute the block-scaled group GEMM kernel."""
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data

    # Unpack B1/B2 and SFB1/SFB2
    # Current: abc_tensors[i] = (a, b, c)  # b is single
    # New: abc_tensors[i] = (a, b1, b2, c)
    # sfasfb_reordered_tensors[i] = (sfa, sfb1, sfb2)

    num_groups = len(problem_sizes)

    # Allocate temp buffers for GEMM1 results
    temp1_tensors = []
    for i, (m, n, k, l) in enumerate(problem_sizes):
        temp1 = torch.empty((m, n, l), dtype=torch.float32, device="cuda")
        temp1_tensors.append(temp1)

    # PASS 1: Compute A @ B1 -> temp1
    abc_ptrs_pass1 = []
    sfasfb_ptrs_pass1 = []
    for i, ((a, b1, b2, c), (sfa, sfb1, sfb2), _) in enumerate(
        zip(abc_tensors, sfasfb_reordered_tensors, problem_sizes)
    ):
        abc_ptrs_pass1.append((a.data_ptr(), b1.data_ptr(), temp1_tensors[i].data_ptr()))
        sfasfb_ptrs_pass1.append((sfa.data_ptr(), sfb1.data_ptr()))

    # Launch kernel for GEMM1
    _launch_kernel(problem_sizes, abc_ptrs_pass1, sfasfb_ptrs_pass1)

    # PASS 2: Compute A @ B2 -> c (use output as temp2)
    abc_ptrs_pass2 = []
    sfasfb_ptrs_pass2 = []
    for i, ((a, b1, b2, c), (sfa, sfb1, sfb2), _) in enumerate(
        zip(abc_tensors, sfasfb_reordered_tensors, problem_sizes)
    ):
        abc_ptrs_pass2.append((a.data_ptr(), b2.data_ptr(), c.data_ptr()))
        sfasfb_ptrs_pass2.append((sfa.data_ptr(), sfb2.data_ptr()))

    # Launch kernel for GEMM2
    _launch_kernel(problem_sizes, abc_ptrs_pass2, sfasfb_ptrs_pass2)

    # FUSION: c = silu(temp1) * c
    for i in range(num_groups):
        c = abc_tensors[i][3]  # output tensor
        temp1 = temp1_tensors[i]
        c.copy_(torch.nn.functional.silu(temp1) * c)

    return [abc_tensors[i][3] for i in range(num_groups)]
```

### Lines Changed: ~50 lines in custom_kernel()

### Pros
- **Zero changes to GPU kernel code** (lines 40-390)
- **Easy to validate** - can compare against reference
- **Very low risk** - reuses proven kernel
- **Fast to implement** - 1-2 hours

### Cons
- **Two kernel launches** - extra overhead
- **No A matrix reuse** - loads A twice from global memory
- **Extra temp buffer** - allocates temp1 for GEMM1 results
- **Fusion kernel is separate** - not fused into epilogue

### Performance Estimate
- Current: 373 us (for g=8, K=7168)
- Two-pass: ~750-800 us (2x kernel + fusion overhead)
- This is SLOWER but CORRECT

---

## PART 3: MINIMAL SINGLE-PASS APPROACH

### Concept

Modify the kernel to compute both GEMMs in sequence, reusing A tiles.

### Changes Required

#### 1. Kernel Signature (Lines 41-61)

Add B2 and SFB2 parameters:

```python
@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b1: cute.CopyAtom,     # RENAMED from tma_atom_b
    mB1_nkl: cute.Tensor,           # RENAMED from mB_nkl
    tma_atom_b2: cute.CopyAtom,     # NEW
    mB2_nkl: cute.Tensor,           # NEW
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb1: cute.CopyAtom,   # RENAMED
    mSFB1_nkl: cute.Tensor,         # RENAMED
    tma_atom_sfb2: cute.CopyAtom,   # NEW
    mSFB2_nkl: cute.Tensor,         # NEW
    # ... rest unchanged
):
```

**Lines to modify: 44-60** (add 4 new parameters)

#### 2. Shared Memory for B2 (Lines 119-140)

```python
    sB1 = smem.allocate_tensor(...)  # RENAMED
    sB2 = smem.allocate_tensor(      # NEW - duplicate sB allocation
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )
    sSFB1 = smem.allocate_tensor(...)  # RENAMED
    sSFB2 = smem.allocate_tensor(...)  # NEW - duplicate sSFB allocation
```

**Lines to modify: 125-140** (add 2 tensor allocations)

#### 3. Second Accumulator in TMEM (Lines 256-284)

```python
    # Existing accumulator for GEMM1
    tCtAcc1 = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

    # NEW: Second accumulator for GEMM2
    acc_tmem_ptr2 = acc_tmem_ptr + size_of_first_accumulator
    tCtAcc2 = cute.make_tensor(acc_tmem_ptr2, tCtAcc_fake.layout)
```

**Lines to modify: 259-265** (add second accumulator)

**Note: This requires doubling num_tmem_alloc_cols from 512 to 1024**

#### 4. Main Loop - Run Twice (Lines 313-352)

Option A: Run mainloop twice (simplest)

```python
    # Main loop for GEMM1 (existing code)
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

        for k_tile in range(k_tile_cnt):
            # ... existing load A, B1, SFA, SFB1
            # ... existing MMA into tCtAcc1
        acc_empty.commit()

    # Main loop for GEMM2 (NEW - duplicate of above with B2/SFB2)
    if warp_idx == 0:
        acc_empty2 = acc_producer.acquire_and_advance()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

        for k_tile in range(k_tile_cnt):
            # Load A (reuse from GEMM1? or reload)
            # Load B2, SFB2 (instead of B1, SFB1)
            # MMA into tCtAcc2
        acc_empty2.commit()
```

**Lines to modify: 313-352** (duplicate entire mainloop, ~40 lines)

Option B: Interleave in same loop (more complex but reuses A)

```python
    for k_tile in range(k_tile_cnt):
        # Load A tile once
        cute.copy(tma_atom_a, tAgA[(None, k_tile)], tAsA[(None, ab_empty.index)], ...)
        cute.copy(tma_atom_sfa, tAgSFA[(None, k_tile)], tAsSFA[(None, ab_empty.index)], ...)

        # Load B1 and compute GEMM1
        cute.copy(tma_atom_b1, tB1gB1[(None, k_tile)], tB1sB1[(None, ab_empty.index)], ...)
        cute.copy(tma_atom_sfb1, tB1gSFB1[(None, k_tile)], tB1sSFB1[(None, ab_empty.index)], ...)
        # ... wait and MMA into tCtAcc1

        # Load B2 and compute GEMM2 (A is already in shared memory!)
        cute.copy(tma_atom_b2, tB2gB2[(None, k_tile)], tB2sB2[(None, ab_empty.index)], ...)
        cute.copy(tma_atom_sfb2, tB2gSFB2[(None, k_tile)], tB2sSFB2[(None, ab_empty.index)], ...)
        # ... wait and MMA into tCtAcc2
```

**This is more complex but reuses A tiles!**

#### 5. Epilogue - Fuse SiLU and Multiply (Lines 354-389)

```python
    # Load GEMM1 result from tmem
    cute.copy(tiled_copy_t2r, tDtAcc1, tDrAcc1)
    acc1_vec = tDrAcc1.load()

    # Load GEMM2 result from tmem
    cute.copy(tiled_copy_t2r, tDtAcc2, tDrAcc2)
    acc2_vec = tDrAcc2.load()

    # Fuse: C = silu(acc1) * acc2
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    silu_result = acc1_vec * (1.0 / (1.0 + cute.exp(-acc1_vec)))
    result = silu_result * acc2_vec

    tDrC.store(result.to(c_dtype))
```

**Lines to modify: 362-370** (add second accumulator read, SiLU, multiply)

### Total Lines Changed: ~100-150 lines

### Pros
- **Single kernel launch**
- **Can reuse A tiles** (if interleaved)
- **Fused epilogue** (no separate fusion kernel)

### Cons
- **Moderate complexity** - many changes
- **Doubles TMEM usage** (two accumulators)
- **Risk of bugs** - after 3 failed rounds, this matters!

---

## PART 4: RISK ASSESSMENT

### Historical Failure Analysis

| Round | Approach | Claimed Improvement | Actual Result |
|-------|----------|---------------------|---------------|
| 1 | Pipeline stages | 1.5x faster | 30% SLOWER |
| 2 | Tile size tuning | 2-3x faster | Compile error |
| 3 | TMA epilogue | 20% faster | Not tested (found bug) |

**Pattern**: Every "optimization" attempt has failed.

### Risk Matrix for Round 4

| Approach | Lines Changed | Risk Level | Reason |
|----------|---------------|------------|--------|
| Two-Pass | ~50 | LOW | No GPU kernel changes |
| Single-Pass (Sequential) | ~150 | MEDIUM | Duplicates proven code |
| Single-Pass (Interleaved) | ~200 | HIGH | Complex synchronization |

### Recommendation: START WITH TWO-PASS

1. **Validate correctness first** - get a passing test
2. **Measure baseline** - see actual dual-GEMM performance
3. **Then optimize** - with confidence that it works

---

## PART 5: IMPLEMENTATION PLAN

### Phase 1: Two-Pass Implementation (2-3 hours)

**Step 1: Update input handling** (30 min)
- Modify `custom_kernel()` to unpack B1, B2, SFB1, SFB2
- Allocate temp1 tensor for GEMM1 results

**Step 2: First kernel pass** (30 min)
- Call compiled kernel with B1, SFB1
- Store to temp1 tensor

**Step 3: Second kernel pass** (30 min)
- Call compiled kernel with B2, SFB2
- Store to output tensor C

**Step 4: Fusion** (30 min)
- Implement `C = silu(temp1) * C`
- Can use PyTorch for simplicity

**Step 5: Validation** (1 hour)
- Run against reference implementation
- Fix any issues

### Phase 2: Single-Pass Optimization (4-6 hours, AFTER Phase 1)

**Only after Two-Pass is validated:**

1. Add B2/SFB2 to kernel signature
2. Allocate second set of shared memory
3. Allocate second accumulator
4. Duplicate mainloop (or interleave)
5. Fuse SiLU and multiply in epilogue
6. Validate against reference

---

## PART 6: RECOMMENDATION

Given the history of 3 failed optimization rounds, I make the following recommendation:

### IMMEDIATE ACTION: Two-Pass Approach

```
Priority: CORRECTNESS > SIMPLICITY > PERFORMANCE
```

**Why Two-Pass First:**

1. **Zero risk** - doesn't touch GPU kernel code
2. **Fast to implement** - 2-3 hours
3. **Easy to validate** - can compare with reference
4. **Provides baseline** - actual dual-GEMM timing

**Estimated Performance:**
- Two-Pass: ~800 us (2x single GEMM + fusion)
- Target: ~19 us (for g=8, K=7168)
- Gap: 42x (better than current 0x since kernel is broken)

### SUBSEQUENT ACTION: Single-Pass Optimization

Once Two-Pass is validated:

1. **Measure the overhead** - how much does two-pass cost?
2. **Identify bottleneck** - is it A reloads or fusion overhead?
3. **Apply minimal changes** - sequential mainloop first
4. **Validate at each step** - catch bugs early

---

## CONCLUSION

**The kernel is broken. It computes `C = A @ B` when it should compute `C = silu(A @ B1) * (A @ B2)`.**

After 3 rounds of failed optimizations, the safest path forward is:

1. **Two-Pass**: Call kernel twice + external fusion (LOW RISK)
2. **Validate**: Confirm correctness against reference
3. **Measure**: Get actual dual-GEMM baseline
4. **Then Optimize**: Apply minimal single-pass changes

**My pitch: Do the SIMPLEST thing that could possibly work, then iterate.**

---

*"The best optimization is computing the right thing."*
*- Contestant C, Shark Tank Round 4*

---

## APPENDIX: Exact Lines to Modify (Two-Pass)

### File: `/home/clark/nvfp4_dual_gemm_repo/nvfp4_group_gemm/submission.py`

```
Line 528-539: custom_kernel() function
  - Unpack B1/B2 from abc_tensors
  - Unpack SFB1/SFB2 from sfasfb_reordered_tensors
  - Allocate temp1 tensors
  - Call kernel twice (once with B1/SFB1, once with B2/SFB2)
  - Apply fusion: C = silu(temp1) * C
```

**No changes to kernel() function (lines 40-390)**
**No changes to my_kernel() function (lines 392-496)**
**No changes to compile_kernel() function (lines 499-525)**

This is the MINIMAL fix.
