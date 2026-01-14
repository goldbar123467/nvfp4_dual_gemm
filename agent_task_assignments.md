# NVFP4 Dual GEMM - Agent Task Assignments
## 4 Parallel Research Tracks + Implementation Phase

---

## TRACK 1: CUTLASS Doc Scraper Agent

### Mission
Extract complete NVFP4 GEMM patterns from CUTLASS documentation and examples.

### URLs to Fetch (Priority Order)

1. **CRITICAL** - Example 72 source:
   ```
   https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu
   ```
   FETCHED - saved to /home/claude/cutlass_nvfp4_baseline.cu

2. **CRITICAL** - Example 72a (BF16 output variant):
   ```
   https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu
   ```

3. **HIGH** - Sub-byte GEMM Colfax tutorial:
   ```
   https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/
   ```

4. **HIGH** - Blackwell functionality doc:
   ```
   https://github.com/NVIDIA/cutlass/blob/main/media/docs/blackwell_functionality.md
   ```

5. **MEDIUM** - Scale factor layout implementation:
   ```
   https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/detail/sm100_blockscaled_layout.hpp
   ```

### Extract These Code Patterns
- [ ] Element type declarations (nv_float4_t, float_e2m1_t)
- [ ] Scale factor layout construction (Sm1xxBlkScaledConfig)
- [ ] MMA tile shapes that work for NVFP4
- [ ] Arguments struct for block-scaled GEMM
- [ ] Reference kernel implementation

---

## TRACK 2: cuBLAS Scale Factor Doc Agent

### Mission
Understand the block scaling factor memory layout requirements.

### URLs to Fetch

1. **CRITICAL** - cuBLAS 1D Block Scaling section:
   ```
   https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
   ```

2. **HIGH** - Triton block-scaled matmul (has nvfp4 scale layout):
   ```
   https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
   ```

### Key Information to Extract
- [ ] 16-element block size for FP4
- [ ] (32, 4) atom_m layout = 128 row blocks
- [ ] Interleaved format visualization
- [ ] How your task's `create_scale_factor_tensors` maps to this

### Scale Factor Layout Summary (from your task):
```python
# Your task does this permutation:
atom_m = (32, 4)  # 128 rows per block
atom_k = 4        # 4 scale blocks per K tile

# Permutation order: (3, 4, 1, 5, 2, 0)
# Input: (l, ceil(mn/128), ceil(sf_k/4), 32, 4, 4)
# Output: (32, 4, rest_m, 4, rest_k, l)
```

---

## TRACK 3: EVT/Epilogue Fusion Agent

### Mission
Design the SiLU fusion pattern for dual GEMM output.

### URLs to Fetch

1. **CRITICAL** - EVT Colfax tutorial:
   ```
   https://research.colfax-intl.com/epilogue_visitor_tree/
   ```
   FETCHED - key patterns extracted

2. **HIGH** - CUTLASS activation functions:
   ```
   https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/epilogue/thread/activation.h
   ```

3. **HIGH** - CUTLASS EVT operations:
   ```
   https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/epilogue/fusion/operations.hpp
   ```

4. **MEDIUM** - SM90 callbacks (EVT implementation):
   ```
   https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp
   ```

### Design the EVT for: C = SiLU(A@B1) * (A@B2)

**Computation Graph:**
```
    A@B1 -----> SiLU() ---\
                           *----> C (fp16)
    A@B2 ----------------/
```

**Implementation Options:**
1. Sequential: GEMM1 -> store -> GEMM2 -> load+fuse
2. Custom dual mainloop (hard)
3. Grouped GEMM with shared A (investigate)

---

## TRACK 4: Performance/Optimization Agent

### Mission
Calculate theoretical limits and optimize tile sizes.

### Calculations Needed

1. **Memory vs Compute Bound Analysis**
   ```
   For each problem size (M, N, K, L):

   Compute: FLOPS = 2 * 2 * M * N * K (dual GEMM)
   Memory: Bytes = M*K/2 + 2*N*K/2 + M*N*2 + scale_factors

   B200 FP4 peak: 18 PFLOPS per GPU
   B200 HBM BW: ~8 TB/s

   Arithmetic Intensity = FLOPS / Bytes
   If AI > Peak_FLOPS/Peak_BW: Memory bound
   ```

2. **Speed of Light Targets**
   | M | N | K | SOL [us] | FLOPS | Bytes | AI | Bound? |
   |---|---|---|----------|-------|-------|-----|--------|
   | 256 | 4096 | 7168 | 4.708 | ? | ? | ? | ? |
   | 512 | 4096 | 7168 | 8.714 | ? | ? | ? | ? |
   | 256 | 3072 | 4096 | 2.125 | ? | ? | ? | ? |
   | 512 | 3072 | 7168 | 6.535 | ? | ? | ? | ? |

3. **Tile Size Tuning**
   ```
   Default: MmaTileShape = Shape<_128,_128,_256>
   Try: Shape<_128,_256,_256> for wider N
   Try: Shape<_256,_128,_256> for wider M
   ```

### Profiling Commands
```bash
# Build profiler
cd cutlass/build
make cutlass_profiler

# Profile your kernel
./tools/profiler/cutlass_profiler \
  --operation=gemm \
  --m=512 --n=4096 --k=7168 \
  --A=f4_e2m1 --B=f4_e2m1 --C=f32 --D=f16 \
  --accum=f32 \
  --use-cuda-graphs

# nsight-compute for bottleneck analysis
ncu --set full ./your_kernel --m=512 --n=4096 --k=7168
```

---

## IMPLEMENTATION PHASE CHECKLIST

### Day 1: Foundation (8 hours)
- [ ] Clone CUTLASS, verify SM100 build
- [ ] Run Example 72, verify correctness
- [ ] Understand scale factor layout transformation
- [ ] Port reference implementation to standalone test

### Day 1 Evening: Single GEMM Working (4 hours)
- [ ] Modify Example 72 for FP16 output
- [ ] Match your task's data layouts
- [ ] Verify correctness against ref_kernel

### Day 2 Morning: Dual GEMM Structure (6 hours)
- [ ] Implement Option A: Sequential GEMMs
- [ ] Add SiLU fusion to second epilogue
- [ ] Benchmark baseline

### Day 2 Afternoon: Optimization (6 hours)
- [ ] Profile bottlenecks
- [ ] Try fusing A loads
- [ ] Tune tile sizes
- [ ] Submit best result

---

## CRITICAL GOTCHAS

1. **K divisibility**: K must be divisible by 256 (stated in task)
2. **Scale factor alignment**: M and N must be divisible by 128
3. **FP4 packing**: Two elements per byte, use float4_e2m1fn_x2
4. **Output dtype**: fp16, accumulator is fp32
5. **Scale factor dtype**: e4m3fnuz (unsigned zero variant)

---

## FILES CREATED

1. `/home/claude/nvfp4_research_ops.md` - Full research document
2. `/home/claude/cutlass_nvfp4_baseline.cu` - Annotated baseline code
3. This file: `/home/ubuntu/projects/nvfp4_dual_gemm/agent_task_assignments.md`

---

## AGENT COORDINATION

### Agent Mail Project
- Project key: `/home/ubuntu/projects/nvfp4_dual_gemm`
- Coordinating agent: LilacSpring (Claude Opus 4.5)

### File Reservations
Before editing any file, agents must reserve via:
```
mcp__mcp-agent-mail__file_reservation_paths
```

### Handoff Protocol
Use subject prefixes:
- `[TASK]` - New task assignment
- `[HANDOFF]` - Passing work to another agent
- `[DONE]` - Task completed
- `[BLOCKED]` - Need help/waiting on dependency
- `[QUESTION]` - Clarification needed
