# SHARK TANK ROUND 7: RESEARCH EDITION

---

```
 ____  _____ ____  _____    _    ____   ____ _   _
|  _ \| ____/ ___|| ____|  / \  |  _ \ / ___| | | |
| |_) |  _| \___ \|  _|   / _ \ | |_) | |   | |_| |
|  _ <| |___ ___) | |___ / ___ \|  _ <| |___|  _  |
|_| \_\_____|____/|_____/_/   \_\_| \_\\____|_| |_|

 _____ ____ ___ _____ ___ ___  _   _
| ____|  _ \_ _|_   _|_ _/ _ \| \ | |
|  _| | | | | |  | |  | | | | |  \| |
| |___| |_| | |  | |  | | |_| | |\  |
|_____|____/___| |_| |___\___/|_| \_|
```

---

## THE SITUATION

**6 rounds of optimization attempts. Mixed results.**

| Round | Approach | Result |
|-------|----------|--------|
| 1 | Pipeline Stages | 30% SLOWER |
| 2 | Tile Tuning | COMPILE ERROR |
| 3 | Debug | Found bug ✓ |
| 4 | Two-Pass Fix | Fixed correctness ✓ |
| 5 | Stream Parallelism | NOT ALLOWED |
| 6 | Pre-allocation Cache | 33% SLOWER |

**Current baseline: 479 µs (target: 18.8 µs) - 25x gap**

---

## ROUND 7 RULES: RESEARCH ONLY

**NO IMPLEMENTATION THIS ROUND.**

Each researcher must investigate ONE aspect of the problem and report findings.
The goal is to understand WHERE the time is actually going before trying more fixes.

---

## RESEARCH QUESTIONS

### Researcher 1: ROOFLINE ANALYSIS
- What's the theoretical minimum time for this GEMM?
- Are we compute-bound or memory-bound?
- How far from roofline are we?

### Researcher 2: KERNEL STRUCTURE ANALYSIS
- What does our kernel actually do step-by-step?
- Where are the synchronization points?
- How many instructions per GEMM?

### Researcher 3: COMPARISON ANALYSIS
- What do winning submissions on gpumode typically do?
- What does cuBLAS achieve for similar problems?
- What's the gap between us and theoretical peak?

### Researcher 4: WHAT HAVEN'T WE TRIED
- List ALL optimization approaches we haven't attempted
- Filter out what's impossible (hardware constraints, competition rules)
- Rank by expected impact

---

## WHAT WE KNOW

### Hardware (B200)
- 192 SMs
- FP4 Tensor Cores
- ~2.25 TB/s memory bandwidth
- MMA requires 128x128 minimum tiles

### Our Kernel
- Single launch with all groups in Z-dimension
- TMA for async memory loads
- NVFP4 MMA instructions
- SIMT epilogue (not TMA)

### Constraints
- No multiple CUDA streams
- No tile sizes < 128x128
- Pipeline stages make it SLOWER

---

## DELIVERABLES

Each researcher produces:
1. **Data/Analysis** (numbers, not opinions)
2. **Key Insight** (one sentence)
3. **Recommended Next Step** (specific, actionable)

---

*"Before you optimize, understand. Before you understand, measure."*

