# SHARK TANK ROUND 4: CONTESTANT A - SEQUENTIAL DUAL GEMM

---

```
  ____  _____ ___  _   _ _____ _   _ _____ ___    _    _
 / ___|| ____/ _ \| | | | ____| \ | |_   _|_ _|  / \  | |
 \___ \|  _|| | | | | | |  _| |  \| | | |  | |  / _ \ | |
  ___) | |__| |_| | |_| | |___| |\  | | |  | | / ___ \| |___
 |____/|_____\__\_\\___/|_____|_| \_| |_| |___/_/   \_\_____|

  ____  _   _    _    _       ____ _____ __  __ __  __
 |  _ \| | | |  / \  | |     / ___| ____|  \/  |  \/  |
 | | | | | | | / _ \ | |    | |  _|  _| | |\/| | |\/| |
 | |_| | |_| |/ ___ \| |___ | |_| | |___| |  | | |  | |
 |____/ \___/_/   \_\_____| \____|_____|_|  |_|_|  |_|
```

---

## THE PITCH: SEQUENTIAL DUAL GEMM

**Philosophy: Get it correct first, then optimize.**

The Sequential approach is the most straightforward path to a correct implementation. We run the existing, proven GEMM mainloop twice - once for B1, once for B2 - then fuse the SiLU and multiply in the epilogue.

---

## WHY SEQUENTIAL?

1. **Minimum Risk**: Reuses the existing, working mainloop
2. **Easiest to Debug**: Clear separation between GEMM1 and GEMM2
3. **Incremental Path**: Can be optimized later to interleaved/parallel approaches
4. **Proven Pattern**: This is how cuBLAS handles multi-GEMM when fusion isn't critical

---

## APPROACH OVERVIEW

```
Current Flow:
  Load A, B tiles -> GEMM accumulate -> Store C

Sequential Dual GEMM Flow:
  Phase 1: Load A, B1 tiles -> GEMM1 accumulate -> acc1 in TMEM
  Phase 2: Load A, B2 tiles -> GEMM2 accumulate -> acc2 in TMEM
  Phase 3: Apply SiLU to acc1, multiply by acc2 -> Store C
```

---

## CODE CHANGES REQUIRED

### 1. Function Signature Changes

**Current:**
```python
tensor_of_abc_ptrs: cute.Tensor,          # Contains [A, B, C] pointers
tensor_of_sfasfb_ptrs: cute.Tensor,       # Contains [SFA, SFB] pointers
```

**New:**
```python
tensor_of_abc_ptrs: cute.Tensor,          # Contains [A, B1, B2, C] pointers (4 items now!)
tensor_of_sfasfb_ptrs: cute.Tensor,       # Contains [SFA, SFB1, SFB2] pointers (3 items now!)
```

**Changes in `custom_kernel()`:**
```python
# Current (lines 535-539):
abc_ptrs.append((a.data_ptr(), b.data_ptr(), c.data_ptr()))
sfasfb_ptrs.append((sfa_reordered.data_ptr(), sfb_reordered.data_ptr()))

# New:
abc_ptrs.append((a.data_ptr(), b1.data_ptr(), b2.data_ptr(), c.data_ptr()))
sfasfb_ptrs.append((sfa.data_ptr(), sfb1.data_ptr(), sfb2.data_ptr()))
```

**Tensor layouts change:**
```python
# Current:
tensor_of_abc_ptrs = cute.make_tensor(
    ptr, cute.make_layout((num_groups, 3), stride=(3, 1))
)
tensor_of_sfasfb_ptrs = cute.make_tensor(
    ptr, cute.make_layout((num_groups, 2), stride=(2, 1))
)

# New:
tensor_of_abc_ptrs = cute.make_tensor(
    ptr, cute.make_layout((num_groups, 4), stride=(4, 1))  # A, B1, B2, C
)
tensor_of_sfasfb_ptrs = cute.make_tensor(
    ptr, cute.make_layout((num_groups, 3), stride=(3, 1))  # SFA, SFB1, SFB2
)
```

### 2. Tensor Initialization (lines 86-91 and 188-209)

Need to create tensors for B2 and SFB2:

```python
# Current: only B and SFB
mB_nkl_iter = cute.make_ptr(ab_dtype, tensor_of_abc_ptrs[group_idx, 1], ...)
sfb_nkl_iter = cute.make_ptr(sf_dtype, tensor_of_sfasfb_ptrs[group_idx, 1], ...)

# New: add B2 and SFB2
mB1_nkl_iter = cute.make_ptr(ab_dtype, tensor_of_abc_ptrs[group_idx, 1], ...)
mB2_nkl_iter = cute.make_ptr(ab_dtype, tensor_of_abc_ptrs[group_idx, 2], ...)
sfb1_nkl_iter = cute.make_ptr(sf_dtype, tensor_of_sfasfb_ptrs[group_idx, 1], ...)
sfb2_nkl_iter = cute.make_ptr(sf_dtype, tensor_of_sfasfb_ptrs[group_idx, 2], ...)

# Output C is now at index 3
mC_mnl_iter = cute.make_ptr(c_dtype, tensor_of_abc_ptrs[group_idx, 3], ...)
```

### 3. TensorMap Updates (lines 183-226)

Add TensorMaps for B2 and SFB2:

```python
# Add tensormap pointers (increase num_tensormaps from 4 to 6)
num_tensormaps = 6  # A, B1, B2, SFA, SFB1, SFB2

tensormap_b2_smem_ptr = tensormap_sfb_smem_ptr + bytes_per_tensormap // 8
tensormap_sfb2_smem_ptr = tensormap_b2_smem_ptr + bytes_per_tensormap // 8

# Initialize and update B2, SFB2 tensormaps in warp 0
tensormap_manager.init_tensormap_from_atom(tma_atom_b, tensormap_b2_smem_ptr, 0)
tensormap_manager.init_tensormap_from_atom(tma_atom_sfb, tensormap_sfb2_smem_ptr, 0)
tensormap_manager.update_tensormap(
    (real_tensor_a, real_tensor_b1, real_tensor_b2, real_tensor_sfa, real_tensor_sfb1, real_tensor_sfb2),
    ...
)
```

### 4. Main Loop Restructure (lines 314-352)

**Current Structure:**
```python
# Single GEMM loop
for k_tile in range(k_tile_cnt):
    # Load A, B, SFA, SFB
    # Run MMA on all K blocks
```

**New Structure - Sequential:**
```python
# ============ PHASE 1: GEMM1 (A @ B1) ============
acc1_empty = acc_producer.acquire_and_advance()
tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

for k_tile in range(k_tile_cnt):
    ab_empty = ab_producer.acquire_and_advance()

    # Load A and B1 tiles
    cute.copy(tma_atom_a, tAgA[(None, k_tile)], tAsA[(None, ab_empty.index)], ...)
    cute.copy(tma_atom_b, tBgB1[(None, k_tile)], tBsB[(None, ab_empty.index)], ...)  # B1!
    cute.copy(tma_atom_sfa, tAgSFA[(None, k_tile)], tAsSFA[(None, ab_empty.index)], ...)
    cute.copy(tma_atom_sfb, tBgSFB1[(None, k_tile)], tBsSFB[(None, ab_empty.index)], ...)  # SFB1!

    ab_full = ab_consumer.wait_and_advance()

    # Copy scale factors to TMEM
    cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_staged, tCtSFA_compact_s2t)
    cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t_staged, tCtSFB_compact_s2t)

    # Run MMA for GEMM1
    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator)
        cute.gemm(tiled_mma, tCtAcc1, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc1)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

    ab_full.release()

acc1_empty.commit()

# ============ PHASE 2: GEMM2 (A @ B2) ============
acc2_empty = acc_producer.acquire_and_advance()
tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

for k_tile in range(k_tile_cnt):
    ab_empty = ab_producer.acquire_and_advance()

    # Load A (again) and B2 tiles
    cute.copy(tma_atom_a, tAgA[(None, k_tile)], tAsA[(None, ab_empty.index)], ...)
    cute.copy(tma_atom_b, tBgB2[(None, k_tile)], tBsB[(None, ab_empty.index)], ...)  # B2!
    cute.copy(tma_atom_sfa, tAgSFA[(None, k_tile)], tAsSFA[(None, ab_empty.index)], ...)
    cute.copy(tma_atom_sfb, tBgSFB2[(None, k_tile)], tBsSFB[(None, ab_empty.index)], ...)  # SFB2!

    ab_full = ab_consumer.wait_and_advance()

    # Copy scale factors to TMEM
    cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_staged, tCtSFA_compact_s2t)
    cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t_staged, tCtSFB_compact_s2t)

    # Run MMA for GEMM2
    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator)
        cute.gemm(tiled_mma, tCtAcc2, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc2)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

    ab_full.release()

acc2_empty.commit()
```

---

## MEMORY MANAGEMENT: WHERE TO STORE ACC1 WHILE COMPUTING ACC2?

### Option A: Two TMEM Accumulators (Preferred if space permits)

The current kernel allocates TMEM for one accumulator:
```python
num_tmem_alloc_cols = 512
```

For 128x128 tile with FP32 accumulator:
- acc shape = 128 * 128 = 16,384 elements
- FP32 = 4 bytes each = 65,536 bytes = 64 KB per accumulator
- TMEM size on Blackwell: ~256 KB per SM (speculative, need to verify)

**If TMEM can fit two accumulators:**
```python
num_tmem_alloc_cols = 1024  # Double allocation

# Allocate two separate accumulators
acc1_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
tCtAcc1 = cute.make_tensor(acc1_tmem_ptr, tCtAcc_fake.layout)

# Second accumulator at offset
acc2_tmem_ptr = acc1_tmem_ptr + acc_size
tCtAcc2 = cute.make_tensor(acc2_tmem_ptr, tCtAcc_fake.layout)
```

### Option B: Store ACC1 to Shared Memory (If TMEM limited)

If TMEM cannot fit two accumulators:

```python
# After GEMM1 completes, copy acc1 from TMEM to SMEM
sAcc1 = smem.allocate_tensor(
    element_type=cutlass.Float32,
    layout=acc_layout,
    byte_alignment=128,
)

# Copy TMEM -> SMEM (T2S)
cute.copy(tiled_copy_t2s, tCtAcc1, sAcc1)
cute.arch.barrier()  # Sync before reusing TMEM

# Now compute GEMM2 into same TMEM slot
# In epilogue, load sAcc1 back
```

**SMEM requirement:** 128 * 128 * 4 = 64 KB
**Available SMEM on Blackwell SM100:** 228 KB configurable

This is feasible, but adds memory traffic.

### Option C: Store ACC1 to Registers (Not Recommended)

Moving 64KB through registers would cause massive register pressure and likely spilling. Not recommended for 128x128 tiles.

**My Recommendation: Try Option A first (two TMEM accumulators)**

---

## SILU IMPLEMENTATION

### SiLU Definition
```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

### In CuTe/CUTLASS - Fused in Epilogue

The epilogue currently does:
```python
# Load acc from TMEM to registers
cute.copy(tiled_copy_t2r, tDtAcc, tDrAcc)
acc_vec = tDrAcc.load()
# Convert and store
tDrC.store(acc_vec.to(c_dtype))
```

**New Epilogue with SiLU Fusion:**
```python
# Load BOTH accumulators from TMEM to registers
cute.copy(tiled_copy_t2r, tDtAcc1, tDrAcc1)
cute.copy(tiled_copy_t2r, tDtAcc2, tDrAcc2)

acc1_vec = tDrAcc1.load()  # FP32
acc2_vec = tDrAcc2.load()  # FP32

# Apply SiLU to acc1
# SiLU(x) = x / (1 + exp(-x))
# Using cutlass intrinsics or manual computation

def silu_fp32(x):
    # Method 1: Direct computation
    neg_x = -x
    exp_neg_x = cute.exp(neg_x)           # exp(-x)
    one_plus_exp = 1.0 + exp_neg_x        # 1 + exp(-x)
    sigmoid = 1.0 / one_plus_exp          # sigmoid(x)
    return x * sigmoid                     # x * sigmoid(x)

# Or use fast approximation:
def silu_fast(x):
    # Approximation: x * sigmoid(x) = x * (0.5 + 0.5 * tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
    # This is the GELU approximation pattern adapted for SiLU
    return x * (1.0 / (1.0 + cute.exp(-x)))

# Apply to vector
silu_acc1 = silu_fp32(acc1_vec)

# Element-wise multiply with acc2
result = silu_acc1 * acc2_vec

# Convert to FP16 and store
tDrC.store(result.to(c_dtype))
```

### Efficient SiLU with PTX

For maximum performance, use PTX intrinsics:
```python
# In cutlass, use __expf() for fast exponential
# sigmoid(x) = 1 / (1 + __expf(-x))
# silu(x) = x * sigmoid(x)
```

Or leverage cutlass::epilogue::thread::SiLU if available in the API.

---

## EPILOGUE CHANGES IN DETAIL

**Current Epilogue (lines 354-389):**
```python
# Epilogue
op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc[None,0,0])
thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
tDtAcc = thr_copy_t2r.partition_S(tCtAcc[None,0,0])
tDgC = thr_copy_t2r.partition_D(tCgC[None,0,0])

tDrAcc = cute.make_rmem_tensor(tDgC.shape, cutlass.Float32)
tDrC = cute.make_rmem_tensor(tDgC.shape, c_dtype)

tmem.relinquish_alloc_permit()
acc_full = acc_consumer.wait_and_advance()

cute.copy(tiled_copy_t2r, tDtAcc, tDrAcc)
acc_vec = tDrAcc.load()
tDrC.store(acc_vec.to(c_dtype))

# ... SIMT copy to global memory
cute.copy(simt_atom, cute.flatten(tDrC), cute.flatten(tDgC), pred=cute.flatten(tDpC))
```

**New Epilogue:**
```python
# Epilogue - Load both accumulators
op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)

# Partition for acc1
tiled_copy_t2r_1 = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc1[None,0,0])
thr_copy_t2r_1 = tiled_copy_t2r_1.get_slice(tidx)
tDtAcc1 = thr_copy_t2r_1.partition_S(tCtAcc1[None,0,0])

# Partition for acc2
tiled_copy_t2r_2 = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc2[None,0,0])
thr_copy_t2r_2 = tiled_copy_t2r_2.get_slice(tidx)
tDtAcc2 = thr_copy_t2r_2.partition_S(tCtAcc2[None,0,0])

tDgC = thr_copy_t2r_1.partition_D(tCgC[None,0,0])

# Register tensors
tDrAcc1 = cute.make_rmem_tensor(tDgC.shape, cutlass.Float32)
tDrAcc2 = cute.make_rmem_tensor(tDgC.shape, cutlass.Float32)
tDrC = cute.make_rmem_tensor(tDgC.shape, c_dtype)

tmem.relinquish_alloc_permit()
acc1_full = acc_consumer.wait_and_advance()
acc2_full = acc_consumer.wait_and_advance()

# Load both accumulators to registers
cute.copy(tiled_copy_t2r_1, tDtAcc1, tDrAcc1)
cute.copy(tiled_copy_t2r_2, tDtAcc2, tDrAcc2)

acc1_vec = tDrAcc1.load()
acc2_vec = tDrAcc2.load()

# FUSED SILU + MULTIPLY
# silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
neg_acc1 = -acc1_vec
exp_neg = cute.exp(neg_acc1)  # Need to verify this API exists
sigmoid_acc1 = 1.0 / (1.0 + exp_neg)
silu_acc1 = acc1_vec * sigmoid_acc1

result = silu_acc1 * acc2_vec

# Convert to FP16 and store
tDrC.store(result.to(c_dtype))

# SIMT copy to global memory (same as before)
cute.copy(simt_atom, cute.flatten(tDrC), cute.flatten(tDgC), pred=cute.flatten(tDpC))

acc1_full.release()
acc2_full.release()
```

---

## PROS AND CONS

### PROS

1. **Correctness First**: Maximum confidence in getting the right answer
2. **Minimal Risk**: Reuses proven mainloop, minimal changes to complex MMA logic
3. **Easy to Debug**: Clear separation - if GEMM1 works, just need to verify GEMM2 and epilogue
4. **Incremental**: Can be refactored to interleaved approach later
5. **Clear Memory Model**: Two sequential phases, predictable memory access patterns
6. **Works with Existing Infrastructure**: Same TMA, pipelines, and MMA operations

### CONS

1. **Loads A Twice**: Each K-tile of A is loaded from GMEM twice (once per GEMM)
   - Memory traffic increase: ~2x for A matrix
   - For M=256, K=7168, L=1: A = 0.9 MB, so ~1.8 MB extra

2. **No A Reuse**: The key optimization opportunity (reusing A from SMEM) is missed
   - Interleaved approach could keep A in SMEM between B1 and B2 loads

3. **Longer Latency**: Two sequential mainloops = 2x the compute time
   - Cannot overlap GEMM1 and GEMM2 computation

4. **TMEM Pressure**: May need to store acc1 to SMEM if TMEM is limited
   - Adds 64KB SMEM usage and extra memory traffic

5. **Pipeline Underutilization**: Producer/consumer pipeline does same work twice

---

## ESTIMATED COMPLEXITY

### Difficulty: MEDIUM

| Component | Effort | Risk |
|-----------|--------|------|
| Function signature changes | Low | Low |
| Add B2/SFB2 tensormaps | Medium | Low |
| Duplicate mainloop | Low | Low |
| Second TMEM accumulator | Medium | Medium |
| SiLU implementation | Medium | Low |
| Epilogue fusion | Medium | Medium |
| Testing/Debugging | Medium | Low |

**Total Estimated Time:** 2-4 hours for a working implementation

**Comparison to Alternatives:**
- Sequential (this approach): Medium complexity, low risk
- Interleaved: High complexity, medium risk
- Ping-Pong Warp Specialization: Very high complexity, high risk

---

## PERFORMANCE ESTIMATE

### Current State (Single GEMM, Wrong Computation)
- g=8, K=7168: ~373-456 us

### Sequential Dual GEMM (Correct Computation)
- **Compute Time**: ~2x current (two full mainloops)
- **Memory Traffic**: ~1.5x current (A loaded twice, but A is smaller than B1+B2)
- **Estimated**: 600-900 us for the same benchmark

### Why Performance Drops Initially
This is expected! We're now computing the CORRECT thing (2 GEMMs + SiLU + multiply) instead of the wrong thing (1 GEMM).

### Path to Target Performance
Once correctness is verified, optimize with:
1. Interleaved K-loop (reuse A from SMEM) -> saves ~30% memory traffic
2. Parallel accumulation (if TMEM permits) -> saves compute overlap
3. TMA epilogue -> saves 5-10% on stores

---

## SUMMARY

The Sequential Dual GEMM approach prioritizes **correctness over performance**. It's the safest path to a working kernel that computes `C = silu(A @ B1) * (A @ B2)`.

**Key Implementation Steps:**
1. Add B2, SFB2 pointers to tensor layouts
2. Create tensormaps for B2, SFB2
3. Allocate second TMEM accumulator (or spill to SMEM)
4. Run mainloop twice: once for B1->acc1, once for B2->acc2
5. Fuse SiLU and multiply in epilogue

**Verdict:** This approach trades performance for reliability. It's the right choice when correctness is paramount and debugging time matters.

---

*"Make it work, make it right, make it fast - in that order."*
*- Kent Beck*

---

**CONTESTANT A: SEQUENTIAL DUAL GEMM - Ready for Shark Tank Round 4**
