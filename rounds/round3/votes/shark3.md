# SHARK #3 - THE ROI MAXIMIZER: ROUND 3 VOTE

## CONFESSIONAL: MY BROKEN ROI FORMULA

**Round 1 Claim:** Pipeline Stages has "3.0-3.6x speedup per hour" ROI.
**Round 1 Reality:** Negative ROI. We LOST 30% performance per hour invested.

**Round 2 Claim:** Tile Tuning has "10-15% per hour" ROI.
**Round 2 Reality:** Undefined ROI. Division by zero. No output, just compile errors.

**My ROI formula was garbage.** I was calculating `Expected_Speedup / Time` without accounting for the probability of catastrophic failure. I was basically computing expected value as if success was guaranteed.

---

## THE NEW ROI FORMULA (POST-BANKRUPTCY)

**Old Formula (Broken):**
```
ROI = Expected_Speedup / Implementation_Time
```

**New Formula (Battle-Scarred):**
```
ROI = (P_success * Expected_Upside - P_failure * Downside_Cost) / (Implementation_Time + Expected_Recovery_Time)

Where:
- P_success = Probability of ANY improvement (not assumed 90%+ anymore)
- P_failure = 1 - P_success
- Expected_Upside = Speedup factor if successful
- Downside_Cost = Time wasted + Regression magnitude
- Implementation_Time = Hours to implement
- Expected_Recovery_Time = Time to debug, revert, document failure
```

**Critical Calibration:** After 0 for 2, my P_success estimates must be CONSERVATIVE. Previous sharks estimated P_success at 80-90% for approaches that had 0% success rate.

**My new baseline:** P_success = 35% for anything that looks "obviously correct."

---

## CONTESTANT A: TMA STORE EPILOGUE

### The Pitch Summary
- Replace SIMT stores with TMA stores in epilogue
- Won't touch mainloop, tile sizes, or pipeline stages
- Asks for profiling before implementation
- Claims 15% probability of meaningful improvement

### My ROI Calculation

**Probabilities (Conservative):**
| Outcome | Probability | Notes |
|---------|-------------|-------|
| Meaningful improvement (>5%) | 10% | Contestant claims 15%, I trust less |
| Small improvement (2-5%) | 15% | More likely scenario |
| No change | 35% | Wasted time, no damage |
| Small regression | 25% | Like Pipeline Stages but milder |
| Major failure (compile/crash) | 15% | Still possible with TMA API |

**Expected Upside:** 0.10 * 0.08 + 0.15 * 0.03 = 0.008 + 0.0045 = **1.25% speedup**

**Expected Downside:** 0.25 * (-0.03) + 0.15 * (-0.15) = -0.0075 - 0.0225 = **-3% regression cost**

**Net Expected Value:** 1.25% - 3% = **-1.75%**

**Time Cost:**
- Implementation: 4-5 hours (including profiling)
- Recovery if fails: 1 hour (easier to revert than Pipeline Stages)
- Total: 5-6 hours

**ROI:** (-1.75%) / 5.5 hours = **-0.32% per hour**

### Assessment
Under my new formula, TMA Epilogue has **negative expected ROI**. The probability of meaningful gain is too low, and the probability of small regression (like Pipeline Stages) is too high. However, the contestant's humility and request for profiling before commitment reduces the downside significantly.

**Adjusted ROI with conditional implementation:**
If we profile first (30 min) and abort if epilogue < 5%:
- P_abort_early = 60% (saving implementation time)
- Conditional ROI becomes: **-0.15% per hour** (less negative)

---

## CONTESTANT B: WARP SPECIALIZATION

### The Pitch Summary
- Use idle warps (3 of 4 are doing nothing)
- Producer/consumer architecture
- Incremental approach: instrument first, then minimal warp activation
- Claims 40-60% probability of some improvement

### My ROI Calculation

**Probabilities (Conservative):**
| Outcome | Probability | Notes |
|---------|-------------|-------|
| Significant improvement (>15%) | 15% | The dream scenario |
| Moderate improvement (5-15%) | 20% | Realistic good outcome |
| Small improvement (2-5%) | 15% | Incremental gains |
| No change | 20% | Overhead cancels gains |
| Regression | 20% | Barrier overhead, like Pipeline Stages |
| Major failure (deadlock/crash) | 10% | New synchronization = new bugs |

**Expected Upside:** 0.15 * 0.30 + 0.20 * 0.10 + 0.15 * 0.03 = 0.045 + 0.02 + 0.0045 = **6.95% speedup**

**Expected Downside:** 0.20 * (-0.10) + 0.10 * (-0.25) = -0.02 - 0.025 = **-4.5% regression cost**

**Net Expected Value:** 6.95% - 4.5% = **+2.45%**

**Time Cost:**
- Phase 1 (instrumentation): 3 hours
- Phase 2 (minimal warp test): 6 hours
- Recovery if fails: 4 hours (complex to debug)
- Total expected: 13 hours

**ROI:** 2.45% / 13 hours = **+0.19% per hour**

### Assessment
Warp Specialization has **positive expected ROI** under my new formula. The key insight is that it addresses a real structural issue (3 idle warps), not a hypothetical bottleneck. However, the longer timeline and higher complexity mean more exposure to unknown unknowns.

**Risk factor adjustment:** Given our 0 for 2 track record, I should apply a "humility discount" of 50% to any positive ROI estimate.

**Adjusted ROI:** +0.095% per hour

---

## CONTESTANT C: WILD CARD IDEAS

### The Ideas Ranked by ROI

#### IDEA #5: Dual GEMM Verification (TOP PICK)
**The pitch:** Are we even computing the right thing? The kernel does ONE GEMM, but the task requires `silu(A@B1) * (A@B2)`.

**Probabilities:**
| Outcome | Probability | Notes |
|---------|-------------|-------|
| We're computing wrong (huge upside to fix) | 25% | Would explain the 20-100x gap |
| We're computing right (wasted time) | 75% | Investigation only |

**Time Cost:** 1-2 hours to investigate code flow

**If wrong:** Fixing gives 2x minimum speedup (ROI: astronomical)
**If right:** Lost 2 hours, learned the system

**Expected Value:** 0.25 * 100% + 0.75 * 0% = **25% speedup** (if wrong), **0** (if right)
**Blended ROI:** 25% * 0.25 / 1.5 hours = **4.17% per hour**

Even with investigation cost and 75% chance of "nothing found," the upside if we ARE computing wrong is so massive that ROI is strongly positive.

#### IDEA #3: Reversed K-Loop (BACKUP PICK)
**The pitch:** Change loop direction, zero risk, one line change.

**Probabilities:**
| Outcome | Probability | Notes |
|---------|-------------|-------|
| Small improvement (5-15%) | 10% | Memory prefetcher behaves differently |
| No change | 85% | Most likely |
| Regression | 5% | Unlikely but possible |

**Time Cost:** 15 minutes

**ROI:** (0.10 * 0.10 - 0.05 * 0.05) / 0.25 hours = **3.0% per hour**

Surprisingly high ROI because time cost is nearly zero.

#### IDEA #4: Scale Factor Precomputation (MEDIUM)
**Probabilities:**
| Outcome | Probability | Notes |
|---------|-------------|-------|
| Meaningful improvement | 15% | SF on critical path |
| No change | 50% | SF already overlapped |
| SMEM overflow/regression | 35% | High failure rate |

**Time Cost:** 4 hours + 2 hours recovery risk
**ROI:** (0.15 * 0.15 - 0.35 * 0.10) / 6 hours = **-0.33% per hour** (negative)

#### IDEA #1: 2SM Cooperative (HIGH RISK)
**Probabilities:**
| Outcome | Probability | Notes |
|---------|-------------|-------|
| Major improvement (2-4x) | 10% | If hardware supports it |
| Doesn't work (compile error/constraint) | 70% | Like Tile Tuning |
| Regression from coordination overhead | 20% | Like Pipeline Stages |

**Time Cost:** 16 hours + 8 hours recovery
**ROI:** (0.10 * 3.0 - 0.90 * 0.20) / 24 hours = **+0.38% per hour** (positive but risky)

#### IDEA #2: Persistent Kernel (NUCLEAR)
**Probabilities:**
| Outcome | Probability | Notes |
|---------|-------------|-------|
| Significant improvement | 10% | Wave quantization eliminated |
| Partial improvement | 15% | Some gains, much overhead |
| Failure/Regression | 75% | Insane complexity |

**Time Cost:** 40+ hours
**ROI:** (0.10 * 4.0 + 0.15 * 1.5 - 0.75 * 0.30) / 50 hours = **+0.48% per hour** (but very high variance)

---

## CONSOLIDATED ROI RANKINGS

| Approach | Expected ROI (%/hr) | Variance | Recommendation |
|----------|---------------------|----------|----------------|
| Wild Card #5: Dual GEMM Check | **+4.17%** | Low | INVESTIGATE FIRST |
| Wild Card #3: Reversed K-Loop | **+3.00%** | Very Low | QUICK WIN |
| Warp Specialization (full) | +0.19% (adj: +0.095%) | High | CONDITIONAL |
| Wild Card #1: 2SM Cooperative | +0.38% | Very High | HIGH RISK |
| Wild Card #2: Persistent Kernel | +0.48% | Extreme | NUCLEAR OPTION |
| TMA Epilogue (conditional) | -0.15% | Low | SAFE BUT NEGATIVE |
| Wild Card #4: SF Precomputation | -0.33% | Medium | SKIP |

---

## MY VOTE: CONTESTANT C (WILD CARD)

### Primary Recommendation: Wild Card Idea #5 (Dual GEMM Verification)

**ROI Justification:**

After two failures, I've learned that before optimizing ANYTHING, we need to verify we're solving the right problem. Wild Card #5 has the highest ROI because:

1. **Investigation cost is LOW** (1-2 hours max)
2. **Potential upside is ENORMOUS** (if we're computing wrong, fixing it = 2x minimum)
3. **Even if investigation finds nothing, we learn the system**
4. **It addresses the 20-100x gap directly** - no incremental optimization closes a 100x gap

The task spec says: `C = silu(A @ B1) * (A @ B2)`

The current kernel appears to compute single GEMMs. If this observation is correct, we're not even computing the dual GEMM with SiLU fusion. THAT would explain why we're 20-100x off - we might be doing half the computation OR we might be doing the wrong computation entirely.

### Secondary Recommendation: Wild Card Idea #3 (Reversed K-Loop)

If Idea #5 confirms the kernel is correct, immediately try the Reversed K-Loop. ROI is +3.0% per hour because:
- Implementation time: 15 minutes
- Risk: Nearly zero (same operations, different order)
- Even a 5% improvement beats our Round 1 and Round 2 results COMBINED

### Conditional Support: Warp Specialization

If Wild Cards #5 and #3 both yield nothing, Warp Specialization becomes the next logical step. It has positive expected ROI and addresses a real structural inefficiency (3 idle warps).

---

## POST-MORTEM ON MY ROI EVOLUTION

### Round 1 (Naive)
```
ROI = Speedup / Time
Assumed P_success = 95%
Result: -30% (negative speedup)
Lesson: My probability estimates were garbage
```

### Round 2 (Still Naive)
```
ROI = Speedup / Time
Assumed P_success = 85%
Result: COMPILE ERROR (division by zero in my formula)
Lesson: "Obvious" improvements can be impossible
```

### Round 3 (Humbled)
```
ROI = (P_success * Upside - P_failure * Downside) / (Impl_Time + Recovery_Time)
Assumed P_success = 25-50% (conservative)
Applied "humility discount" to positive estimates
Prioritized investigation over optimization
```

### Key Changes in My Thinking

1. **P_success should NEVER exceed 50% for novel approaches on novel hardware.** We're 0 for 2. Our calibration is off.

2. **Recovery time is REAL.** Every failed approach costs debugging time, morale, and credibility.

3. **Investigation has positive ROI.** Wild Card #5 is "just reading code" but it has the highest ROI because it might reveal we're solving the wrong problem.

4. **Zero-cost experiments first.** Reversed K-Loop takes 15 minutes. Even if it only has 10% chance of working, the time cost is so low that ROI is positive.

5. **"Obvious" optimizations have hidden failure modes.** Pipeline Stages was "obvious." Tile Tuning was "obvious." Both failed for reasons we didn't anticipate.

---

## FINAL STATEMENT

*"I came into this competition as 'The ROI Maximizer' with fancy formulas and confident predictions. Two rounds later, my formula predicted positive ROI twice and delivered negative results twice.*

*After much humiliation, I've rebuilt my ROI framework from the ground up. The new formula accounts for failure probability, recovery costs, and applies aggressive skepticism to any 'obvious' optimization.*

*Under this new framework, the Wild Card wins - not because its ideas are guaranteed to work, but because:*

1. *Idea #5 has the best risk-adjusted return: low cost to investigate, massive upside if we find a fundamental problem*
2. *Idea #3 has the best quick-win return: 15 minutes for potentially 5-10% improvement*
3. *Neither idea can make things worse than Pipeline Stages or Tile Tuning did*

*I vote for Contestant C: The Wild Card. Specifically, investigate Idea #5 (Dual GEMM verification) immediately, then try Idea #3 (Reversed K-Loop) as a quick experiment.*

*If conventional wisdom is 0 for 2, maybe unconventional wisdom deserves a shot."*

---

**OFFICIAL VOTE: CONTESTANT C - THE WILD CARD**

**Specific Investment Allocation:**
- 60% on Wild Card Idea #5 (Dual GEMM Verification)
- 30% on Wild Card Idea #3 (Reversed K-Loop)
- 10% reserved for pivoting to Warp Specialization if above fail

---

*Shark #3 - The ROI Maximizer (Reformed)*
*"When your formula is wrong twice, fix the formula."*
