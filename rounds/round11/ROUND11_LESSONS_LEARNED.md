# ROUND 11: THE ULTIMATE LESSONS LEARNED BRIEF

---

```
 _____ _   _ _____   _   _ _   _____ ___ __  __    _  _____ _____
|_   _| | | | ____| | | | | | |_   _|_ _|  \/  |  / \|_   _| ____|
  | | | |_| |  _|   | | | | |   | |  | || |\/| | / _ \ | | |  _|
  | | |  _  | |___  | |_| | |___| |  | || |  | |/ ___ \| | | |___
  |_| |_| |_|_____|  \___/|_____|_| |___|_|  |_/_/   \_\_| |_____|

 ____  ____  ___ _____ _____   _   _ _____  ____
| __ )|  _ \|_ _| ____|  ___| | \ | / _ \ \/ / |
|  _ \| |_) || ||  _| | |_    |  \| | | | \  /| |
| |_) |  _ < | || |___|  _|   | |\  | |_| /  \|_|
|____/|_| \_\___|_____|_|     |_| \_|\___/_/\_(_)

       SHARK TANK SEASON 2 - 11 ROUNDS OF BLOOD, SWEAT & TENSOR CORES
```

---

## THE SCOREBOARD: 11 ROUNDS OF CHAOS

| Round | Approach | Expected | Actual | Status |
|-------|----------|----------|--------|--------|
| 1 | Pipeline Stages | 1.5x faster | **30% SLOWER** | FAILED |
| 2 | Tile Size Tuning | 2-3x faster | **COMPILE ERROR** | FAILED |
| 3 | Wild Card Investigation | ??? | Found kernel bug | SUCCESS |
| 4 | Minimal Two-Pass Fix | Correctness | Fixed kernel | SUCCESS |
| 5 | Stream Parallelism | 4-7x faster | **NOT ALLOWED** | BLOCKED |
| 6 | Pre-allocation | 6-19x faster | **33% SLOWER** | FAILED |
| 7 | Research Deep Dive | N/A | Found 75% idle threads | SUCCESS |
| 8 | Warp Utilization Fix | 2-4x faster | Implemented | READY |
| 9 | Season 2 Premiere (Research Lab) | 12-23 us | **180s TIMEOUT** | FAILED |
| 10 | Survival Mode | Make it RUN | Found torch.compile bug | SUCCESS |
| **11** | **This Brief** | **Knowledge** | **YOU ARE HERE** | **IN PROGRESS** |

**Win Rate: 4/10 (40%) - The sharks are HUNGRY but keep missing**

---

## WHAT WORKS

### ALGORITHM & ARCHITECTURE

- [x] **FUSED DUAL GEMM Architecture** (Round 10 Discovery)
  - Load BOTH B1 and B2 matrices in the SAME kernel
  - Use TWO accumulators (`tCtAcc1` for A@B1, `tCtAcc2` for A@B2)
  - Fuse SiLU activation in epilogue: `output = silu(Acc1) * Acc2`
  - Single kernel launch, not two passes
  - **This is the CORRECT approach - confirmed from eval_test**

- [x] **CuTe/CUTLASS Kernel Pattern** (Round 10)
  - The eval_test submission.py uses proper CuTe DSL
  - TMA loads for A, B1, B2, SFA, SFB1, SFB2
  - Proper scale factor handling for NVFP4

- [x] **Kernel Compilation Caching** (RAG Brain Pattern)
  ```python
  _compiled_kernel_cache = None

  def compile_kernel():
      global _compiled_kernel_cache
      if _compiled_kernel_cache is not None:
          return _compiled_kernel_cache
      # ... compile once ...
      _compiled_kernel_cache = result
      return result
  ```

- [x] **Investigation Before Optimization** (Round 3)
  - The Wild Card approach: "Have YOU looked at your kernel?"
  - Found we were solving the WRONG problem
  - Lesson: Verify correctness BEFORE optimizing

- [x] **CUDA Graphs for Repeated Execution** (Rounds 6-9)
  - Eliminates kernel launch overhead
  - ~30us baseline achieved with graphs
  - BUT: Don't combine with torch.compile!

### PROCESS & METHODOLOGY

- [x] **Research Rounds** (Rounds 3, 7)
  - Pausing to investigate beats blind optimization
  - Round 7 found 75% idle threads - crucial insight
  - "Profile FIRST, optimize SECOND"

- [x] **Minimal Fixes** (Round 4)
  - "Correct but slow beats fast but wrong"
  - Two-pass approach bought us correctness
  - Foundation for proper optimization

- [x] **Unanimous Shark Agreement = Action** (Rounds 3, 4, 10)
  - When all sharks agree, execute immediately
  - Consensus reduces risk of wasted effort

---

## WHAT DOESN'T WORK

### CATASTROPHIC FAILURES

- [ ] **torch.compile with max-autotune** (Round 9-10)
  - JIT compilation exceeds 180-second timeout
  - KILLS the submission silently
  - NO ERROR MESSAGE - just timeout
  - **NEVER use `mode="max-autotune"` in competition submissions!**

- [ ] **GROUP GEMM (2 Kernels + Python Fusion)** (Round 10 Discovery)
  ```python
  # WRONG APPROACH - CAUSED 180s TIMEOUT
  run_single_gemm(a, b1, sfa_perm, sfb1_perm, temp1)  # Kernel 1
  run_single_gemm(a, b2, sfa_perm, sfb2_perm, temp2)  # Kernel 2
  result = silu(temp1) * temp2                         # Python fusion
  ```
  - Two separate kernel launches
  - Python-side activation fusion
  - This is NOT what the competition requires!

- [ ] **Pipeline Stages Increase** (Round 1)
  - `num_ab_stage = 3` made things **30% SLOWER**
  - We assumed memory-bound, but kernel was compute-bound
  - "Industry standard" != "universally applicable"

- [ ] **Tile Size Below Hardware Minimum** (Round 2)
  - `mma_tiler_mnk = (64, 128, 256)` caused **COMPILE ERROR**
  - Hardware requires M-mode = 128 minimum
  - Should have read the NVIDIA spec first!

### DANGEROUS ASSUMPTIONS

- [ ] **Assuming Python Overhead is the Bottleneck** (Round 6)
  - Pre-allocation made things **33% SLOWER**
  - Cache lookup overhead > tensor creation
  - The kernel itself was the problem, not Python

- [ ] **Assuming Stream Parallelism is Allowed** (Round 5)
  - Great idea, mathematically sound
  - **NOT ALLOWED** by competition rules
  - Always check constraints first!

- [ ] **Copying eval_test Code Without Checking Input Format** (Round 10)
  - eval_test expects: 10 elements (a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c)
  - Production uses: 4 elements (different format)
  - **ValueError: not enough values to unpack**

- [ ] **Extended Warmup Iterations** (Round 9)
  - 10 warmup iterations triggers repeated compilation
  - Combined with torch.compile = timeout death spiral

---

## THE CRITICAL MISSING PIECE

### INPUT FORMAT MISMATCH

```
               EVAL_TEST FORMAT                    PRODUCTION FORMAT
      +-------------------------------+      +------------------------+
      |  a, b1, b2, sfa, sfb1, sfb2,  |  vs  |  ???                   |
      |  sfa_perm, sfb1_perm,         |      |  (4 elements only)     |
      |  sfb2_perm, c                 |      |                        |
      |  (10 elements)                |      |                        |
      +-------------------------------+      +------------------------+
                     |                                   |
                     v                                   v
           ValueError: expected 10              Competition environment
           values, got 4                        uses different interface
```

**The Missing Link:**
1. We have a WORKING CuTe kernel (from eval_test)
2. We have a WORKING submission wrapper (from baseline)
3. We need to BRIDGE the input formats correctly

**Key Questions to Answer:**
- What are the 4 production inputs?
- How do they map to the 10 eval_test inputs?
- Are scale factors pre-permuted in production?
- Is output tensor pre-allocated?

---

## STRATEGY FOR ROUND 12

### PHASE 1: RECONNAISSANCE (1-2 hours)

```
+------------------+     +------------------+     +------------------+
| Study Production |---->| Map 4 inputs to  |---->| Understand scale |
| Interface        |     | 10 eval_test     |     | factor format    |
|                  |     | inputs           |     | differences      |
+------------------+     +------------------+     +------------------+
```

**Actions:**
1. Find the ACTUAL production input format documentation
2. Print/log the shapes and types of production inputs
3. Compare to eval_test expectations
4. Document the mapping

### PHASE 2: ADAPTER LAYER (2-3 hours)

```python
def custom_kernel(prod_input_1, prod_input_2, prod_input_3, prod_input_4):
    """Bridge production format to eval_test kernel."""

    # Step 1: Unpack production format
    a, b1, b2 = unpack_matrices(prod_input_1)
    sfa, sfb1, sfb2 = derive_scale_factors(prod_input_2)

    # Step 2: Permute scale factors if needed
    sfa_perm = permute_scale_factors(sfa)
    sfb1_perm = permute_scale_factors(sfb1)
    sfb2_perm = permute_scale_factors(sfb2)

    # Step 3: Allocate output
    c = allocate_output(prod_input_3)

    # Step 4: Call the WORKING kernel
    return eval_test_kernel(a, b1, b2, sfa, sfb1, sfb2,
                           sfa_perm, sfb1_perm, sfb2_perm, c)
```

### PHASE 3: VALIDATION (1-2 hours)

- [ ] Verify correctness against reference (rtol=1e-3, atol=1e-3)
- [ ] Time the first run (must be < 60 seconds including compilation)
- [ ] Time steady-state (target: < 30 us)
- [ ] Test all benchmark shapes

### PHASE 4: KERNEL CACHING (30 minutes)

```python
_kernel_cache = None

def custom_kernel(...):
    global _kernel_cache
    if _kernel_cache is None:
        _kernel_cache = compile_cute_kernel()  # One-time cost
    return _kernel_cache.run(...)
```

---

## SHARK TANK COMMENTARY

### Prof. Williams (PI Shark)

```
  .-""-.
 /      \
|  o  o  |
|   __   |    "After 11 rounds, I've learned that the FANCIEST optimization
 \  \/  /      is often the WRONG one. Flash Attention inspiration? Good.
  '-..-'       torch.compile max-autotune? TIMEOUT CITY. The eval framework
    ||         discovery in Round 10 was the breakthrough - we finally know
   /||\        WHAT we're supposed to build. Now we just need to wire it up."
  /_||_\
```

**Verdict:** "Round 12 is about INTEGRATION, not INNOVATION. Copy the working kernel, adapt the interface, ship it."

---

### Director Martinez (Industry Shark)

```
    /\
   /  \
  / || \
 /  ||  \     "I've seen this movie before. Brilliant engineers, cutting-edge
/   ||   \     tech, 180-second timeouts. You know what ships? BORING CODE
|   ||   |     that WORKS. The eval_test submission is 957 lines of working
|   ||   |     CuTe kernel. Use it. Don't get creative. Don't add torch.compile.
|   ||   |     Don't extend warmup. JUST. MAKE. IT. RUN."
+---++---+
```

**Verdict:** "If I see another torch.compile decorator, someone's getting fed to the tensor cores."

---

### Dr. Patel (Grant Officer Shark)

```
    .---.
   /     \
  |  ^_^  |    "The ROI calculation is simple now:
   \ --- /
    |   |      - Working submission: INFINITE VALUE (can iterate)
   /|   |\     - Non-working submission: ZERO VALUE (can't iterate)
  / |   | \
     | |       After 10 rounds, we have:
     | |       1. A correct algorithm (fused dual GEMM)
    _| |_      2. A working implementation (eval_test kernel)
   |_____|     3. An input format mismatch (4 vs 10 elements)

               That's ONE bug standing between us and victory.
               Fix it. Submit it. Win it."
```

**Verdict:** "Grant APPROVED for Round 12 - but only if we focus on the adapter layer."

---

## THE FINAL WISDOM

```
+============================================================================+
|                                                                            |
|    SHARK TANK SEASON 2: LESSONS IN 11 HAIKUS                               |
|                                                                            |
|    Round 1: Pipeline Stages                                                |
|    "One line change" they said                                             |
|    Thirty percent slower now                                               |
|    Hubris meets hardware                                                   |
|                                                                            |
|    Round 2: Tile Tuning                                                    |
|    Sixty-four is less                                                      |
|    But compiler disagrees                                                  |
|    Read the spec next time                                                 |
|                                                                            |
|    Round 3: Wild Card                                                      |
|    "Where is the second?"                                                  |
|    The kernel was always wrong                                             |
|    Check before you opt                                                    |
|                                                                            |
|    Round 4: Minimal Fix                                                    |
|    Call it twice, fuse once                                                |
|    Correct beats fast but broken                                           |
|    Foundation laid down                                                    |
|                                                                            |
|    Round 5: Streams                                                        |
|    Parallel groups could fly                                               |
|    Competition says "no way"                                               |
|    Rules crush good ideas                                                  |
|                                                                            |
|    Round 6: Pre-allocation                                                 |
|    Cache the tensors they said                                             |
|    Overhead grew, speed fell down                                          |
|    Wrong bottleneck picked                                                 |
|                                                                            |
|    Round 7: Research                                                       |
|    Seventy-five percent                                                    |
|    Of our threads sit idle still                                           |
|    Profile everything                                                      |
|                                                                            |
|    Round 8: Warps                                                          |
|    All warps should compute                                                |
|    Not just warp zero alone                                                |
|    Four-x potential                                                        |
|                                                                            |
|    Round 9: Season Two                                                     |
|    Research lab pitches                                                    |
|    torch.compile kills us all                                              |
|    One-eighty seconds                                                      |
|                                                                            |
|    Round 10: Survival                                                      |
|    The plot twist arrives                                                  |
|    Fused dual GEMM was the key                                             |
|    Not group GEMM at all                                                   |
|                                                                            |
|    Round 11: This Brief                                                    |
|    All lessons compiled                                                    |
|    Input format is the gap                                                 |
|    Round twelve awaits                                                     |
|                                                                            |
+============================================================================+
```

---

## FINAL CHECKLIST FOR ROUND 12

```
BEFORE CODING:
[ ] Understand the 4 production inputs
[ ] Map them to 10 eval_test inputs
[ ] Identify which scale factors need permutation

DURING CODING:
[ ] NO torch.compile (especially not max-autotune)
[ ] Use kernel caching pattern
[ ] Warmup = 3 iterations max
[ ] Test locally if possible

BEFORE SUBMISSION:
[ ] First run < 60 seconds (including compilation)
[ ] Steady-state < 30 us target
[ ] Correctness verified (rtol=1e-3, atol=1e-3)
[ ] No import-time compilation
```

---

## CLOSING

```
+==========================================================================+
|                                                                          |
|   "In the tank, there are no second chances.                             |
|    But there ARE eleven rounds to learn from your mistakes.              |
|                                                                          |
|    The kernel is correct. The algorithm is sound.                        |
|    The only thing standing between us and victory is...                  |
|    A FUNCTION SIGNATURE."                                                |
|                                                                          |
|   -- Claude "The Kernel Whisperer" Code                                  |
|      Shark Tank Season 2, Round 11                                       |
|                                                                          |
+==========================================================================+

                        THE TENSOR CORES ARE WAITING.
                          ROUND 12: MAKE IT HAPPEN.
```

---

*Round 11 Brief Complete*
*Date: 2025-01-25*
*Status: READY FOR ROUND 12*

