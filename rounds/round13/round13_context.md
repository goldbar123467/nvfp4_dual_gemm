# ROUND 13 CONTEXT

```
 ____   ___  _   _ _   _ ____    _ _____
|  _ \ / _ \| | | | \ | |  _ \  / |___ /
| |_) | | | | | | |  \| | | | | | | |_ \
|  _ <| |_| | |_| | |\  | |_| | | |___) |
|_| \_\\___/ \___/|_| \_|____/  |_|____/

        THE THREE-GROUP MYSTERY
```

---

## PHASE 1: CONTEXT

### Current State
- **Kernel Version**: v12-dual-format-handling
- **Performance**: UNKNOWN (crashes before benchmark)
- **Correctness**: FAILING - ValueError on input unpacking

### The New Error
```
ValueError: DUAL GEMM expects 2 groups, got 3
  File "/root/submission.py", line 969, in custom_kernel
    raise ValueError(f"DUAL GEMM expects 2 groups, got {len(abc_tensors)}")
```

### What We Learned in Round 12
Our Round 12 fix ASSUMED:
- Production sends 4 elements: `(abc_tensors, _, sfasfb, problem_sizes)`
- `abc_tensors` has 2 groups: `[(a, b1, c), (a, b2, c)]`

**THIS ASSUMPTION WAS WRONG!**

Production actually sends **3 GROUPS**.

---

## Prior Learnings (Rounds 1-12)

### What Works
- Fused DUAL GEMM architecture (2 accumulators)
- CuTe/CUTLASS kernel patterns
- Kernel compilation caching
- Defensive format detection (`len(data)`)

### What Doesn't Work
- torch.compile with max-autotune (180s timeout)
- GROUP GEMM (2 kernels + Python fusion)
- Assuming input formats without verification
- Pipeline stages > 1 (makes things slower)
- Tile sizes below 128x128 (hardware constraint)

### Critical Discovery Timeline
| Round | Discovery |
|-------|-----------|
| 9 | torch.compile max-autotune = timeout |
| 10 | eval_test format (10 elements) ≠ production |
| 11 | Documented all learnings |
| 12 | Production uses 4-element format with GROUPS |
| **13** | **Production has 3 GROUPS, not 2!** |

---

## Constraints (DO NOT VIOLATE)

1. **NO torch.compile with max-autotune** (causes 180s timeout)
2. **Keep kernel caching pattern** (compile once, reuse)
3. **Preserve CuTe DUAL GEMM kernel** (it works, just needs input handling)
4. **Support BOTH formats** (10-element eval_test AND 4-element production)
5. **Handle variable number of groups** (could be 2, 3, or more)

---

## This Round's Question

**What do the 3 groups represent, and how do we extract the correct tensors?**

### Hypothesis A: Multiple Batch Problems
Production might batch multiple (M,N,K,L) problems together:
- Group 0: Problem 0
- Group 1: Problem 1
- Group 2: Problem 2

### Hypothesis B: Dual GEMM with Separate Output
- Group 0: `(a, b1, c1)` - First GEMM with its output
- Group 1: `(a, b2, c2)` - Second GEMM with its output
- Group 2: `(_, _, c_final)` - Final fused output

### Hypothesis C: Different Format Entirely
The 3 groups might contain completely different data than assumed.

---

## Success Criteria

1. **No ValueError** - Submission accepts 3-group input
2. **Correctness passes** - Output matches reference (rtol=1e-03, atol=1e-03)
3. **Performance runs** - Completes within timeout
4. **Both formats work** - 10-element and 4-element (with any group count)

---

## Baseline

| Metric | Status |
|--------|--------|
| Correctness | FAILING (ValueError) |
| Performance | N/A (crashes) |
| Format handling | 10-element: WORKS, 4-element 2-group: WORKS, 4-element 3-group: FAILS |

---

## Required Pitches

Following GAME_SHOW_FORMAT.md, we need at least 2 pitches:

### Pitch A: "The Flexible Handler"
- Remove hard-coded 2-group assumption
- Handle ANY number of groups dynamically
- Find DUAL GEMM tensors (a, b1, b2, c) regardless of group structure

### Pitch B: "The Investigator"
- Add logging to print what 3 groups actually contain
- Submit with debug output to understand format
- Use this info to build correct handler

### Pitch C: "The Conservative"
- Handle multiple common group counts (2, 3, 4)
- Raise informative error for unknown counts
- Document each format variant

### Pitch D: "The Wild Card"
- What if 3 groups means 3 independent DUAL GEMMs?
- Loop over groups and process each
- Return list of results

---

## Shark Panel

| Shark | Persona | Focus |
|-------|---------|-------|
| **The Skeptic** | "Prove it works" | Evidence, risk assessment |
| **The Pragmatist** | "Can we ship it?" | Implementation complexity |
| **The Theorist** | "Why does this happen?" | Root cause analysis |

---

## Round 13 Mission

**FIX THE 3-GROUP HANDLING AND WIN THE COMPETITION**

---

*"The only constant in production is that your assumptions are wrong."*
*- Round 13 Opening*
