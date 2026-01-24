# SHARK TANK ROUND 5: ALL WILD CARD EDITION

---

```
 __        _____ _     ____     ____    _    ____  ____
 \ \      / /_ _| |   |  _ \   / ___|  / \  |  _ \|  _ \
  \ \ /\ / / | || |   | | | | | |     / _ \ | |_) | | | |
   \ V  V /  | || |___| |_| | | |___ / ___ \|  _ <| |_| |
    \_/\_/  |___|_____|____/   \____/_/   \_\_| \_\____/

  _____ ____ ___ _____ ___ ___  _   _
 | ____|  _ \_ _|_   _|_ _/ _ \| \ | |
 |  _| | | | | |  | |  | | | | |  \| |
 | |___| |_| | |  | |  | | |_| | |\  |
 |_____|____/___| |_| |___\___/|_| \_|
```

---

## THE SITUATION

After 4 rounds, we've learned:
- Pipeline stages: SLOWER (compute-bound)
- Tile tuning: IMPOSSIBLE (hardware constraint)
- The kernel was computing the wrong thing
- We fixed it but performance is still ~20-100x off target

**Current Performance:**
```
g=8, K=7168: ~530 µs (target: 18.8 µs) - 28x gap
g=8, K=2048: ~508 µs (target: 10.7 µs) - 47x gap
g=2, K=4096: ~279 µs (target: 2.4 µs)  - 116x gap
g=2, K=1536: ~256 µs (target: 1.5 µs)  - 170x gap
```

---

## ROUND 5 RULES: ALL WILD CARD EDITION

Three contestants. Three COMPLETELY DIFFERENT submission files.
No incremental changes. No safe bets. WILD CARDS ONLY.

Each contestant must create a NEW submission file with a radically different approach.

---

## HARDWARE CONSTRAINTS (STILL APPLY)

```python
# NVFP4 MMA requires 128x128 minimum tile
mma_tiler_mnk = (128, 128, 256)  # CANNOT CHANGE
```

---

## WHAT WE'RE OPTIMIZING

GROUP GEMM: Multiple independent GEMMs batched together
- Each group: C = A @ B (with block scaling)
- Different shapes per group
- Input format: (abc_tensors, _, sfasfb_tensors, problem_sizes)

---

## WILD CARD APPROACHES TO CONSIDER

1. **Triton Kernel** - Write in Triton instead of CuTe DSL
2. **torch.compile** - Let PyTorch's compiler optimize
3. **cuBLAS Calls** - Use NVIDIA's library directly
4. **Persistent Kernel** - Keep thread blocks alive across problems
5. **Stream Parallelism** - Run groups in parallel streams
6. **Warp Specialization** - Producer/consumer architecture
7. **Custom Memory Layout** - Optimize data layout for cache
8. **Kernel Fusion** - Fuse multiple operations
9. **Mixed Precision** - Strategic precision choices
10. **Algorithmic Changes** - Different GEMM algorithms

---

## SUBMISSION FILE FORMAT

Each contestant creates: `submission_wildcard_X.py`

Must export: `custom_kernel(data) -> output`

Input format (4 elements):
```python
abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data
# abc_tensors: List[(a, b, c)] for each group
# sfasfb_reordered_tensors: List[(sfa, sfb)] for each group
# problem_sizes: List[(m, n, k, l)] for each group
```

---

## JUDGING CRITERIA

1. **Correctness** (must pass validation)
2. **Performance** (geometric mean of benchmarks)
3. **Innovation** (how creative/novel is the approach)
4. **Risk/Reward** (potential upside vs implementation complexity)

---

## THE STAKES

We're 20-170x off target. Conservative approaches have FAILED.
It's time for something radical.

---

*"When conventional wisdom fails, unconventional wisdom prevails."*
*- The Wild Cards*
