# SHARK TANK ROUND 4: CONTESTANT B - INTERLEAVED DUAL GEMM

---

```
  ___ _   _ _____ _____ ____  _     _____    ___     _______ ____
 |_ _| \ | |_   _| ____|  _ \| |   | ____|  / \ \   / / ____|  _ \
  | ||  \| | | | |  _| | |_) | |   |  _|   / _ \ \ / /|  _| | | | |
  | || |\  | | | | |___|  _ <| |___| |___ / ___ \ V / | |___| |_| |
 |___|_| \_| |_| |_____|_| \_\_____|_____/_/   \_\_/  |_____|____/

       ____  _   _    _    _       ____ _____ __  __ __  __
      |  _ \| | | |  / \  | |     / ___| ____|  \/  |  \/  |
      | | | | | | | / _ \ | |    | |  _|  _| | |\/| | |\/| |
      | |_| | |_| |/ ___ \| |___ | |_| | |___| |  | | |  | |
      |____/ \___/_/   \_\_____| \____|_____|_|  |_|_|  |_|
```

---

## THE APPROACH: INTERLEAVED DUAL GEMM

The key insight: **Load A tile ONCE, compute BOTH GEMMs with it before moving on.**

This is the most efficient approach because:
1. A matrix bandwidth is reused 2x
2. Both accumulators progress together through K dimension
3. Epilogue can fuse silu + multiply after both GEMMs complete
4. Better cache locality than sequential approach

---

## ALGORITHM OVERVIEW

```python
# Initialize TWO accumulators in TMEM
acc1 = zeros(M_tile, N_tile)  # For A @ B1
acc2 = zeros(M_tile, N_tile)  # For A @ B2

for k_tile in range(k_tile_cnt):
    # === PRODUCER: Load A tile ONCE ===
    load_A_tile_to_smem(k_tile)
    load_SFA_tile_to_smem(k_tile)

    # === PRODUCER: Load B1 tile ===
    load_B1_tile_to_smem(k_tile)
    load_SFB1_tile_to_smem(k_tile)

    # === CONSUMER: Compute partial GEMM1 ===
    acc1 += A_smem @ B1_smem  # A already in shared memory!

    # === PRODUCER: Load B2 tile (A is REUSED!) ===
    load_B2_tile_to_smem(k_tile)
    load_SFB2_tile_to_smem(k_tile)

    # === CONSUMER: Compute partial GEMM2 ===
    acc2 += A_smem @ B2_smem  # SAME A tile, already loaded!

# === EPILOGUE: Fuse silu + multiply ===
C = silu(acc1) * acc2
```

---

## DETAILED CODE CHANGES REQUIRED

### 1. Kernel Signature Changes

Current kernel takes:
- `mB_nkl: cute.Tensor` - Single B matrix
- `mSFB_nkl: cute.Tensor` - Single scale factor

New kernel needs:
```python
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b1: cute.CopyAtom,      # TMA for B1
    mB1_nkl: cute.Tensor,             # B1 matrix
    tma_atom_b2: cute.CopyAtom,       # TMA for B2
    mB2_nkl: cute.Tensor,             # B2 matrix
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb1: cute.CopyAtom,     # TMA for SFB1
    mSFB1_nkl: cute.Tensor,           # Scale factors for B1
    tma_atom_sfb2: cute.CopyAtom,     # TMA for SFB2
    mSFB2_nkl: cute.Tensor,           # Scale factors for B2
    # ... rest of parameters
):
```

### 2. TMA Descriptor Setup (6 total, up from 4)

```python
# Current: 4 tensormaps (A, B, SFA, SFB)
num_tensormaps = 4  # CHANGE TO 6

# New: 6 tensormaps (A, B1, B2, SFA, SFB1, SFB2)
num_tensormaps = 6

# Tensormap pointers
tensormap_a_smem_ptr = tensormap_smem_ptr
tensormap_b1_smem_ptr = tensormap_a_smem_ptr + bytes_per_tensormap // 8
tensormap_b2_smem_ptr = tensormap_b1_smem_ptr + bytes_per_tensormap // 8  # NEW
tensormap_sfa_smem_ptr = tensormap_b2_smem_ptr + bytes_per_tensormap // 8
tensormap_sfb1_smem_ptr = tensormap_sfa_smem_ptr + bytes_per_tensormap // 8
tensormap_sfb2_smem_ptr = tensormap_sfb1_smem_ptr + bytes_per_tensormap // 8  # NEW
```

### 3. Shared Memory Layout for Interleaved B1/B2

Two options:

**Option A: Separate SMEM buffers (recommended)**
```python
# Allocate separate shared memory for B1 and B2
sB1 = smem.allocate_tensor(
    element_type=ab_dtype,
    layout=b_smem_layout_staged.outer,
    byte_alignment=128,
    swizzle=b_smem_layout_staged.inner,
)
sB2 = smem.allocate_tensor(
    element_type=ab_dtype,
    layout=b_smem_layout_staged.outer,
    byte_alignment=128,
    swizzle=b_smem_layout_staged.inner,
)

# Similarly for scale factors
sSFB1 = smem.allocate_tensor(...)
sSFB2 = smem.allocate_tensor(...)
```

**Option B: Ping-pong within same buffer (complex, not recommended)**

### 4. Accumulator Management - THE KEY CHALLENGE

Current TMEM allocation:
```python
num_tmem_alloc_cols = 512  # Single accumulator
```

**For interleaved dual GEMM, we need TWO accumulators:**

```python
# CRITICAL: Check if we have enough TMEM for 2 accumulators
# Current layout: acc_shape based on 128x128 MMA tile
# FP32 accumulator = 4 bytes per element
# 128 x 128 = 16384 elements per accumulator
# 16384 * 4 = 65536 bytes = 64KB per accumulator
# 2 accumulators = 128KB

# Option 1: Double the TMEM allocation
num_tmem_alloc_cols = 1024  # May exceed TMEM limit!

# Option 2: Share columns, different row offsets
# Acc1: columns 0-511
# Acc2: columns 512-1023
acc1_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
tCtAcc1 = cute.make_tensor(acc1_tmem_ptr, tCtAcc_fake.layout)

# Offset for second accumulator
acc2_offset = tcgen05.find_tmem_tensor_col_offset(tCtAcc1)
acc2_tmem_ptr = cute.recast_ptr(acc1_tmem_ptr + acc2_offset, dtype=cutlass.Float32)
tCtAcc2 = cute.make_tensor(acc2_tmem_ptr, tCtAcc_fake.layout)
```

**TMEM Capacity Analysis:**
- B200 TMEM: ~256KB per SM
- Current: 512 columns for 1 accumulator
- Proposed: 1024 columns for 2 accumulators
- This SHOULD fit, but needs verification

### 5. Main Loop Modifications

```python
# Main loop - INTERLEAVED approach
if warp_idx == 0:
    # Initialize BOTH accumulators
    acc_empty1 = acc_producer.acquire_and_advance()
    acc_empty2 = acc_producer.acquire_and_advance()  # Second accumulator
    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

    for k_tile in range(k_tile_cnt):
        # === PHASE 1: Load A tile (used by BOTH GEMMs) ===
        ab_empty = ab_producer.acquire_and_advance()

        cute.copy(tma_atom_a, tAgA[(None, k_tile)], tAsA[(None, ab_empty.index)],
            tma_bar_ptr=ab_empty.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_a_gmem_ptr, cute.AddressSpace.generic))
        cute.copy(tma_atom_sfa, tAgSFA[(None, k_tile)], tAsSFA[(None, ab_empty.index)],
            tma_bar_ptr=ab_empty.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_sfa_gmem_ptr, cute.AddressSpace.generic))

        # === PHASE 2: Load B1 tile and compute GEMM1 ===
        cute.copy(tma_atom_b1, tB1gB1[(None, k_tile)], tB1sB1[(None, ab_empty.index)],
            tma_bar_ptr=ab_empty.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_b1_gmem_ptr, cute.AddressSpace.generic))
        cute.copy(tma_atom_sfb1, tB1gSFB1[(None, k_tile)], tB1sSFB1[(None, ab_empty.index)],
            tma_bar_ptr=ab_empty.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_sfb1_gmem_ptr, cute.AddressSpace.generic))

        ab_full = ab_consumer.wait_and_advance()

        # Copy scale factors to TMEM
        cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_staged, tCtSFA_compact_s2t)
        cute.copy(tiled_copy_s2t_sfb1, tCsSFB1_compact_s2t_staged, tCtSFB1_compact_s2t)

        # Compute GEMM1: A @ B1 -> acc1
        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
            tiled_mma.set(tcgen05.Field.SFB, tCtSFB1[sf_kblock_coord].iterator)
            cute.gemm(tiled_mma, tCtAcc1, tCrA[kblock_coord], tCrB1[kblock_coord], tCtAcc1)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        # === PHASE 3: Load B2 tile (A is ALREADY in smem!) ===
        # Note: We can overlap B2 load with GEMM1 computation
        cute.copy(tma_atom_b2, tB2gB2[(None, k_tile)], tB2sB2[(None, ab_empty.index)],
            tma_bar_ptr=ab_empty.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_b2_gmem_ptr, cute.AddressSpace.generic))
        cute.copy(tma_atom_sfb2, tB2gSFB2[(None, k_tile)], tB2sSFB2[(None, ab_empty.index)],
            tma_bar_ptr=ab_empty.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_sfb2_gmem_ptr, cute.AddressSpace.generic))

        # Wait for B2 load
        ab_full2 = ab_consumer.wait_and_advance()

        # Copy SFB2 to TMEM
        cute.copy(tiled_copy_s2t_sfb2, tCsSFB2_compact_s2t_staged, tCtSFB2_compact_s2t)

        # Compute GEMM2: A @ B2 -> acc2 (REUSING A from smem!)
        # Reset accumulate flag for acc2
        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile > 0)
        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)  # SAME SFA!
            tiled_mma.set(tcgen05.Field.SFB, tCtSFB2[sf_kblock_coord].iterator)
            cute.gemm(tiled_mma, tCtAcc2, tCrA[kblock_coord], tCrB2[kblock_coord], tCtAcc2)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        ab_full.release()
        ab_full2.release()

    acc_empty1.commit()
    acc_empty2.commit()
```

### 6. Epilogue: Fused SiLU + Multiply

```python
# Epilogue - Read BOTH accumulators and fuse
# Read acc1 from TMEM
tDrAcc1 = cute.make_rmem_tensor(tDgC.shape, cutlass.Float32)
cute.copy(tiled_copy_t2r, tDtAcc1, tDrAcc1)
acc1_vec = tDrAcc1.load()

# Read acc2 from TMEM
tDrAcc2 = cute.make_rmem_tensor(tDgC.shape, cutlass.Float32)
cute.copy(tiled_copy_t2r, tDtAcc2, tDrAcc2)
acc2_vec = tDrAcc2.load()

# Fused SiLU + Multiply
# silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
def silu_fused_multiply(acc1, acc2):
    # SiLU on acc1
    silu_acc1 = acc1 / (1.0 + cute.exp(-acc1))
    # Multiply with acc2
    return silu_acc1 * acc2

result_vec = silu_fused_multiply(acc1_vec, acc2_vec)

# Store result
tDrC = cute.make_rmem_tensor(tDgC.shape, c_dtype)
tDrC.store(result_vec.to(c_dtype))
```

---

## SYNCHRONIZATION STRATEGY

### Pipeline Barriers - Critical for Interleaved

```python
# We need separate pipelines or careful barrier management
# Option 1: Separate pipelines for B1 and B2 (cleaner, more overhead)
ab_producer_b1, ab_consumer_b1 = pipeline.PipelineTmaUmma.create(...)
ab_producer_b2, ab_consumer_b2 = pipeline.PipelineTmaUmma.create(...)

# Option 2: Single pipeline, careful ordering (trickier)
# - Load A+SFA+B1+SFB1, compute GEMM1
# - Load B2+SFB2 (A is still valid in smem), compute GEMM2
# - Release barrier only after BOTH GEMMs use A
```

### A Tile Lifetime Management

```
K-tile 0:
[Load A0] -> [Load B1_0] -> [GEMM1] -> [Load B2_0] -> [GEMM2] -> [Release A0]
                                         ^
                                         |
                              A0 must still be valid here!

K-tile 1:
[Load A1] -> [Load B1_1] -> [GEMM1] -> [Load B2_1] -> [GEMM2] -> [Release A1]
```

Key insight: A tile buffer must NOT be overwritten until BOTH GEMMs have consumed it.

---

## MEMORY ANALYSIS

### Shared Memory Requirements

| Buffer | Size | Count | Total |
|--------|------|-------|-------|
| A tile (128x256) | 16KB | 1 | 16KB |
| B1 tile (128x256) | 16KB | 1 | 16KB |
| B2 tile (128x256) | 16KB | 1 | 16KB |
| SFA tile | ~2KB | 1 | 2KB |
| SFB1 tile | ~2KB | 1 | 2KB |
| SFB2 tile | ~2KB | 1 | 2KB |
| Tensormaps (6x128B) | 768B | 1 | 768B |
| Barriers | ~256B | 1 | 256B |
| **Total** | | | **~55KB** |

B200 SMEM per SM: 228KB (configurable up to 228KB)
**Verdict: Fits comfortably!**

### TMEM Requirements (Accumulators)

| Buffer | Size | Count | Total |
|--------|------|-------|-------|
| Acc1 (128x128 FP32) | 64KB | 1 | 64KB |
| Acc2 (128x128 FP32) | 64KB | 1 | 64KB |
| SFA TMEM | ~8KB | 1 | 8KB |
| SFB1 TMEM | ~8KB | 1 | 8KB |
| SFB2 TMEM | ~8KB | 1 | 8KB |
| **Total** | | | **~152KB** |

B200 TMEM: 256KB per SM
**Verdict: Fits, but tight. May need to reduce to 1 stage if issues.**

---

## PROS OF INTERLEAVED APPROACH

1. **A Matrix Bandwidth Efficiency**: Load A once, use twice = 2x reduction in A memory traffic

2. **Better Cache Locality**: A tile stays hot in L1/L2 while we switch between B1 and B2

3. **Natural Pipelining**: Can overlap B2 load with GEMM1 computation:
   ```
   Time: |--Load A--|--Load B1--|--GEMM1--|--Load B2--|--GEMM2--|
                                    ^
                                    |
                              B2 load can start here!
   ```

4. **Single Epilogue Pass**: Both accumulators are ready at the same time, fuse naturally

5. **Same Number of Barriers**: Don't need complex ping-pong logic

---

## CONS OF INTERLEAVED APPROACH

1. **Double TMEM Usage**: Need 2 accumulators = 2x TMEM footprint
   - Risk: May exceed TMEM capacity for certain configurations
   - Mitigation: Current 512 cols -> 1024 cols should fit on B200

2. **More Complex Main Loop**: Must manage A tile lifetime carefully
   - Cannot release A buffer until BOTH GEMMs have consumed it
   - Requires careful barrier placement

3. **Two Scale Factor Sets**: Need SFB1 and SFB2 in TMEM simultaneously
   - Current: One SFB tensor
   - New: Two SFB tensors

4. **Pipeline Complexity**: Need to coordinate loads of B1 and B2 with shared A
   - Not as simple as "double everything"
   - Requires understanding of barrier semantics

5. **Debugging Difficulty**: Interleaved execution harder to trace than sequential

---

## COMPARISON TO SEQUENTIAL APPROACH

| Metric | Sequential | Interleaved | Winner |
|--------|------------|-------------|--------|
| A memory traffic | 2x | 1x | **Interleaved** |
| TMEM usage | 1 acc | 2 acc | Sequential |
| Implementation complexity | Lower | Higher | Sequential |
| Epilogue complexity | Same | Same | Tie |
| Potential performance | Good | Better | **Interleaved** |
| Risk | Lower | Medium | Sequential |

**Recommendation**: Interleaved is the better approach for performance, but Sequential is safer for initial correctness. Consider implementing Sequential first, then upgrading to Interleaved.

---

## IMPLEMENTATION PRIORITY

### Phase 1: Make It Work (Sequential First)
1. Add B2, SFB2 support to kernel signature
2. Compute GEMM1 fully
3. Compute GEMM2 fully
4. Add SiLU + multiply in epilogue
5. **Validate correctness**

### Phase 2: Make It Fast (Interleave)
1. Restructure main loop for interleaving
2. Add second accumulator
3. Manage A tile lifetime correctly
4. Optimize barrier usage
5. **Validate performance**

---

## CONCLUSION

The Interleaved Dual GEMM approach is the **theoretically optimal** way to implement `C = silu(A @ B1) * (A @ B2)` because it minimizes A matrix memory traffic. However, it requires:

1. Doubling TMEM allocation for accumulators
2. Careful A tile lifetime management
3. More complex synchronization

**My recommendation**: This IS the right approach for maximum performance. The complexity is manageable, and the 2x reduction in A memory traffic is worth it.

For Round 4, if we're prioritizing CORRECTNESS first:
- Start with Sequential (simpler)
- Then optimize to Interleaved (better performance)

If we're going for PERFORMANCE from the start:
- Go directly to Interleaved
- Accept the additional complexity

---

*"Load once, compute twice. That's the interleaved way."*
*- Contestant B, Shark Tank Round 4*
