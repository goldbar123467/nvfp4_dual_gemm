# SHARK TANK ROUND 7: RESEARCH EDITION - RESULTS

---

```
 ____  _____ ____  _____    _    ____   ____ _   _
|  _ \| ____/ ___|| ____|  / \  |  _ \ / ___| | | |
| |_) |  _| \___ \|  _|   / _ \ | |_) | |   | |_| |
|  _ <| |___ ___) | |___ / ___ \|  _ <| |___|  _  |
|_| \_\_____|____/|_____/_/   \_\_| \_\\____|_| |_|

 ____  _____ ____  _   _ _   _____ ____
|  _ \| ____/ ___|| | | | | |_   _/ ___|
| |_) |  _| \___ \| | | | |   | | \___ \
|  _ <| |___ ___) | |_| | |___| |  ___) |
|_| \_\_____|____/ \___/|_____|_| |____/
```

---

## THE SMOKING GUN 🔫

### 75% OF THREADS ARE IDLE

```python
# submission.py lines 316-354
if warp_idx == 0:  # ← ONLY WARP 0 DOES WORK!
    for k_tile in range(k_tile_cnt):
        # TMA loads
        # S2T copies
        # MMA operations ← ALL COMPUTE HERE
```

**The kernel uses 128 threads (4 warps), but only Warp 0 (32 threads) does any work in the main loop.**

| Threads | Role | Status |
|---------|------|--------|
| 32 (Warp 0) | TMA + MMA + Everything | WORKING |
| 96 (Warps 1-3) | Nothing | **IDLE** |

**This alone explains a 4x performance gap.**

---

## RESEARCH FINDINGS

### Researcher 1: Roofline Analysis

```
Workload Type: MEMORY-BOUND
Arithmetic Intensity: 555 FLOPs/byte
Ridge Point (B200): 4,444 FLOPs/byte

Theoretical Minimum:
├─ Compute (5 PFLOPS): 14.66 µs
├─ Memory (shared B):  13.05 µs
└─ Memory (unique B):  58.7 µs

Current: 479 µs → 8-37x slower than theoretical floor
```

### Researcher 2: Kernel Structure

**Timeline of Current Execution:**
```
PHASE 4: MAIN LOOP (WARP 0 ONLY) ← 75% THREADS IDLE
┌─────────────────────────────────────────────────────┐
│ for k_tile in range(28):  # K=7168/256              │
│   ┌─────────────────────────────────────────────┐   │
│   │ Warp 0: TMA load → wait → S2T → 4x MMA     │   │
│   │ Warps 1-3: COMPLETELY IDLE                  │   │
│   └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Other inefficiencies found:**
- Single-stage pipeline (`num_ab_stage = 1`) - no double buffering
- SIMT epilogue stores (16-bit, not vectorized)
- Linear group search O(n) instead of O(1)

### Researcher 3: Comparison Analysis

| Implementation | Performance |
|----------------|-------------|
| cuBLAS FP4 | 6,787 TFLOPS |
| SGLang FP4 | 206.9 µs/layer |
| FlashInfer FP4 | 481.9 µs/layer |
| **Our kernel** | **479 µs** (≈ FlashInfer unoptimized) |

**Winning solutions use:**
- Warp specialization (producer/consumer warps)
- TMA for ALL memory (including stores)
- Persistent kernels
- Fused dual-GEMM operations

### Researcher 4: Untried Approaches

**HIGH IMPACT (Not Yet Tried):**

| Optimization | Expected Gain | Risk |
|--------------|---------------|------|
| **True Dual-GEMM Fusion** | 30%+ (halves A bandwidth) | High |
| **Warp Specialization** | 2-4x (use all threads) | Medium |
| **TMA Store Epilogue** | 20-30% | Medium |
| **128-bit Vector Stores** | 10-20% | Low |

**All 4 warps participating would give us 4x more throughput.**

---

## ROOT CAUSE ANALYSIS

### Why are we 25x slower than target?

| Factor | Multiplier | Explanation |
|--------|------------|-------------|
| 75% threads idle | **4x** | Only warp 0 works |
| No load/compute overlap | **1.5-2x** | Single-stage pipeline |
| SIMT vs TMA stores | **1.2-1.5x** | Inefficient epilogue |
| Two separate GEMMs | **1.3x** | A matrix loaded twice |
| **Combined** | **9-24x** | Matches observed 25x gap |

---

## PRIORITY OPTIMIZATIONS FOR ROUND 8

### Priority 1: USE ALL WARPS (4x potential)
```python
# CURRENT:
if warp_idx == 0:
    # all work

# NEEDED:
# Warp 0: TMA loads (producer)
# Warps 1-3: MMA compute (consumers)
```

### Priority 2: TRUE DUAL-GEMM FUSION (1.3x potential)
Load A matrix once, compute A@B1 and A@B2 with shared tiles.

### Priority 3: TMA STORE EPILOGUE (1.2x potential)
Replace SIMT 16-bit stores with TMA bulk stores.

---

## WHAT THIS MEANS

**The kernel is a REFERENCE IMPLEMENTATION, not optimized.**

The `if warp_idx == 0` pattern is typical of example code that:
- Demonstrates the API
- Is NOT meant for production use
- Leaves optimization as an exercise

**We've been optimizing around the edges while the core is 4x underutilized.**

---

## RECOMMENDED NEXT STEPS

1. **Round 8A**: Modify kernel to use warp specialization
   - Warp 0 = Producer (TMA loads)
   - Warps 1-3 = Consumers (MMA compute)

2. **Round 8B**: Fuse dual-GEMM into single kernel
   - Load A once
   - Compute both A@B1 and A@B2
   - Fuse SiLU + multiply in epilogue

3. **Round 8C**: TMA store epilogue
   - Replace SIMT stores with TMA

---

## LEARNINGS

| Round | Approach | What We Learned |
|-------|----------|-----------------|
| 1 | Pipeline stages | Compute-bound, not memory-bound |
| 2 | Tile sizes | Hardware constraint (128x128) |
| 3 | Debug | The kernel was wrong |
| 4 | Two-pass | Fixed correctness |
| 5 | Streams | Not allowed |
| 6 | Pre-allocation | Python isn't the bottleneck |
| **7** | **Research** | **75% threads idle - use all warps!** |

---

*"The fastest optimization is using the hardware you already have."*
*— Round 7 Research Team*

