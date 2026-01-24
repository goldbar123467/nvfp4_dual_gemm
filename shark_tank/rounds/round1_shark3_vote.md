# SHARK #3 VOTE: THE ROI MAXIMIZER

**Shark Name:** The ROI Maximizer
**Mantra:** "Speedup per hour of engineering time. That's what separates billion-dollar products from vanity projects."

---

## EXECUTIVE SUMMARY

I've analyzed all four contestants through one lens: **bang for your buck**. What speedup can you get for each hour of engineering investment? Which optimization eliminates the most waste-per-dollar-of-effort?

The answer is clear: **CONTESTANT #1 (PIPELINE STAGES)** is the runaway winner. It's not the fastest in absolute speedup potential, but it's the ONLY one that delivers 1.5x+ speedup in under 30 minutes. That's **3x the speedup-per-hour of every competitor**.

---

## DETAILED SCORING

### CONTESTANT #1: PIPELINE STAGES (num_ab_stage tuning)

**Expected Speedup:** 1.5x - 1.8x
**Implementation Time:** < 30 minutes
**Speedup Per Hour:** **3.0 - 3.6x improvement/hour**

| Criterion | Score | Justification |
|-----------|-------|---|
| **Expected Speedup (40% weight)** | 8/10 | Conservative 1.5-1.8x is solid. Not flashy, but reliable. Eliminates tensor core starvation—a known bottleneck in single-stage pipelines. |
| **Implementation Feasibility (25%)** | 10/10 | **LITERALLY ONE VARIABLE CHANGE.** `num_ab_stage = 1` → `num_ab_stage = 3`. The infrastructure already exists: `PipelineTmaUmma`, `make_smem_layout_*`, barrier allocation all parameterized. This is not a one-line hack—it's a one-line *configuration*. |
| **Risk Level (20%)** | 9/10 | Minimal risk. Easy profiling path (test stages 2, 3, 4). Easy rollback. B200 has 256KB SMEM; estimated ~36KB per stage leaves massive headroom. No algorithmic changes. No new dependencies. |
| **Pitch Quality (15%)** | 9/10 | Technically precise. Honest about limitations. Good visualization of tensor core starvation. Realistic memory calculations. Conservative speedup estimates. |
| **WEIGHTED SCORE** | **8.9/10** | **This is champion material.** |

---

### CONTESTANT #2: TILE SIZE TUNING (Adaptive mma_tiler_mnk)

**Expected Speedup:** 1.8x - 2.5x (geometric mean)
**Implementation Time:** 2-3 days (~20 hours)
**Speedup Per Hour:** **0.09 - 0.125x improvement/hour**

| Criterion | Score | Justification |
|-----------|-------|---|
| **Expected Speedup (40% weight)** | 9/10 | Higher absolute speedup (1.8-2.5x) is attractive. Addresses real issue: 89% SM waste on small-M problems. Math is sound: wave efficiency + tile efficiency. |
| **Implementation Feasibility (25%)** | 6/10 | MODERATE difficulty. Requires: adaptive tile selection heuristic, kernel recompilation for each tile config, SMEM layout updates, caching strategy. The "simple heuristic" in their pitch is deceptively complex—you need to profile 4-5 different tile configurations per problem class. Real implementation is 2-3 days, not "surprisingly manageable." |
| **Risk Level (20%)** | 6/10 | MEDIUM-HIGH risk. JIT compilation overhead (unpredictable—could harm small-batch performance). Some tile configs may have alignment issues they're not accounting for. Regression possible if heuristic is wrong for a problem size. Need extensive validation. The pitch downplays this with "keep 128x128x256 as fallback"—but that fallback means you're not getting speedup for those cases. |
| **Pitch Quality (15%)** | 7/10 | Good data on SM utilization waste. Wave efficiency math is correct. But pitch undersells implementation complexity and overestimates feasibility of "simple heuristic." Claims 2-3 days but that's optimistic. |
| **WEIGHTED SCORE** | **7.2/10** | **Solid optimizer, not a game-changer for ROI.** |

---

### CONTESTANT #3: TMA STORE EPILOGUE

**Expected Speedup:** 1.12x - 1.15x (12-15% overall from 3.9x epilogue improvement)
**Implementation Time:** 2-4 hours
**Speedup Per Hour:** **0.28 - 0.58x improvement/hour**

| Criterion | Score | Justification |
|-----------|-------|---|
| **Expected Speedup (40% weight)** | 7/10 | Modest speedup (12-15% overall). The math is right—epilogue is currently 20% of kernel time, 3.9x improvement there = ~12% overall. But this optimization has **limited ceiling**: epilogue is pure overhead, but it's *already* a small portion of runtime. Good engineering, limited upside. |
| **Implementation Feasibility (25%)** | 8/10 | Implementation is straightforward: TMA store descriptor, SMEM staging, async fence. They've already built TMA load infrastructure; stores use the same pattern. 2-4 hours is realistic. Good documentation available (CUTLASS examples 49, 71). |
| **Risk Level (20%)** | 7/10 | LOW-MEDIUM risk. TMA is proven tech. But risks exist: alignment issues on strides, SMEM pressure, TMA descriptor bugs, cluster synchronization for 2SM stores. Fallback path exists (keep SIMT). However, if TMA store fails to deliver, you've spent 4 hours for ~0% speedup. |
| **Pitch Quality (15%)** | 8/10 | Clear problem statement (epilogue overhead). Honest about register pressure and memory address calculation waste. Good technical details. Conservative speedup estimates are appreciated. |
| **WEIGHTED SCORE** | **7.3/10** | **Solid engineering, mediocre ROI.** |

---

### CONTESTANT #4: WARP SPECIALIZATION

**Expected Speedup:** 1.25x - 1.40x
**Implementation Time:** 3-4 weeks (~120-160 hours)
**Speedup Per Hour:** **0.0078 - 0.0117x improvement/hour**

| Criterion | Score | Justification |
|-----------|-------|---|
| **Expected Speedup (40% weight)** | 7/10 | Modest speedup (25-40%) seems high until you read the fine print. PyTorch reports 10-15% on Flash Attention. Tawa claims 3.78x but that's from 104→393 TFLOPs/s (that's 3.78x total, NOT on top of an already-optimized kernel). For an already-functional GEMM, 25-40% is optimistic. Realistic: 1.15-1.25x. |
| **Implementation Feasibility (25%)** | 4/10 | **HIGH complexity.** This is NOT "leverage CUTLASS primitives." Warp specialization for **dual GEMM with fused SiLU** is novel work. They gloss over "Dual GEMM Coordination: High" complexity. You need: warp group partitioning, named barriers, register reallocation, dual consumer interleaving, epilogue fusion synchronization. This is 3-4 weeks of careful debugging, not "follow the template." CUTLASS examples are for single GEMM, not dual GEMM fusion. |
| **Risk Level (20%)** | 4/10 | **HIGH risk.** Barrier deadlocks (medium-high probability). Debugging complexity is "High" per their own assessment. Dual GEMM synchronization is unproven territory. If register spilling occurs, occupancy drops and you lose the 1.25x. If barrier ordering is wrong, you have silent correctness issues. No easy rollback—refactoring this deep means rewriting weeks of work. |
| **Pitch Quality (15%)** | 6/10 | Flashy pitch, but oversells feasibility. The "Week 1/2/3/4" timeline is fiction. References are good (PyTorch, CUTLASS). But comparison table claiming 1.35x speedup while others get 1.1x is cherry-picked: pipelining (this work's foundation) is 1.5x. The "architectural transformation" language is marketing, not engineering. |
| **WEIGHTED SCORE** | **5.5/10** | **High risk, mediocre reward. The worst ROI by far.** |

---

## FINAL RANKING

| Rank | Contestant | Score | Speedup/Hour | Verdict |
|------|-----------|-------|------|---------|
| **🥇 #1** | **Pipeline Stages** | **8.9/10** | **3.0-3.6x/hr** | **THE WINNER** |
| **🥈 #2** | TMA Store Epilogue | 7.3/10 | 0.28-0.58x/hr | Solid but incremental |
| **🥉 #3** | Tile Size Tuning | 7.2/10 | 0.09-0.125x/hr | Good optimization, wrong timing |
| **4️⃣** | Warp Specialization | 5.5/10 | 0.0078-0.0117x/hr | Ambitious but impractical |

---

## MY VOTE: CONTESTANT #1 - PIPELINE STAGES

### THE DECISION

I'm voting for **CONTESTANT #1: PIPELINE STAGES** because it delivers **maximum speedup per hour of engineering effort**.

### THE MATH

- **Pipeline Stages:** 1.5x speedup in 0.5 hours = **3x speedup/hour**
- **TMA Epilogue:** 1.12x speedup in 3 hours = **0.04x speedup/hour**
- **Tile Tuning:** 2x speedup in 20 hours = **0.1x speedup/hour**
- **Warp Specialization:** 1.3x speedup in 150 hours = **0.0087x speedup/hour**

Pipeline stages is **35x more efficient than TMA**, **30x more efficient than tile tuning**, and **345x more efficient than warp specialization**.

### THE REASONING

1. **Proven Technique:** Every high-performance GEMM uses multi-stage pipelining. This isn't experimental; it's foundational.

2. **Zero Algorithmic Risk:** You're not changing the kernel logic—just changing one configuration value. The infrastructure already exists and is battle-tested.

3. **Easy Profiling & Iteration:** Test stages 2, 3, 4 in parallel. Profile each. Pick the best. If something breaks, you revert a single line.

4. **Foundation for Everything Else:** Once you have 3-4 pipeline stages, the OTHER optimizations (TMA epilogue, tile tuning, warp specialization) become MORE effective, not less. You don't eliminate them; you sequence them strategically.

5. **Timeline Reality:** The other contestants want 2 days to 4 weeks. I can ship pipeline stages in under 30 minutes, declare victory, then systematically tackle the next optimization with fresh data about what actually matters.

### WHY NOT THE OTHERS?

- **Tile Tuning (Contestant #2):** Real speedup is 1.8x max, but implementation is 20 hours of heuristic development + JIT caching complexity. Pass for now. Do this AFTER pipeline stages when profiling is cleaner.

- **TMA Epilogue (Contestant #3):** Solid engineering. But epilogue is already only 20% of runtime. Even 3.9x improvement there = 12% overall. You'd get MORE speedup from just adding one more pipeline stage (which costs 0 hours of dev time).

- **Warp Specialization (Contestant #4):** Ambitious. But 150+ hours for 1.3x speedup is brutal ROI. The dual GEMM + SiLU fusion aspect is novel and risky. You'll spend 4 weeks and hit register spilling or barrier deadlocks. Do NOT attempt this as Step 1.

---

## DEAL TERMS: WHAT I WANT TO SEE

If Contestant #1 is awarded the optimization slot, here's what I require:

### Phase 1: Validation (Required Before Shipping)
1. **Profile with stages 2, 3, 4** for each benchmark shape
2. **Check SMEM utilization**: Verify < 200KB per SM (256KB available)
3. **Measure register pressure**: Ensure no spilling
4. **Validate correctness**: Run full test suite with each stage count
5. **Thermal check**: Confirm no thermal throttling at peak stage count

### Phase 2: Implementation Standards
1. **Document the choice**: Why did we pick stage=3 (or whatever)? Not stage 2? Not stage 4?
2. **Future-proof the code**: Make `num_ab_stage` a tunable parameter in the submission, not a magic constant
3. **Add comments**: Explain what each stage's SMEM layout is for
4. **Provide fallback**: Keep `num_ab_stage = 1` logic available for emergency rollback

### Phase 3: Integration Roadmap
1. **Don't ship this alone.** Immediately start tile tuning in parallel (different engineer)
2. **Measure again after tile tuning:** Does multi-stage pipeline still provide 1.5x, or does it drop to 1.2x once tile configs are right?
3. **Queue up TMA epilogue**: If pipeline + tile tuning plateaus, implement TMA store as the next optimization
4. **Save warp specialization for v2**: Only if we hit a wall and can't squeeze more without architectural changes

### Phase 4: Success Metrics
- **Target:** ≥ 1.4x speedup on geometric mean of benchmarks
- **Timeline:** Ship in 1 week (including profiling, validation, and doc)
- **Regression threshold:** If ANY benchmark drops more than 5%, investigate before shipping
- **Future maintainability:** Code must be clear enough for a new engineer to understand in under an hour

---

## CLOSING ARGUMENT

Sharks, optimization is a *sequence*, not a menu. You don't go for the most interesting dish; you go for the highest-calorie-per-dollar meal first.

**Pipeline stages is that meal.** One variable change. 50% speedup potential. 30 minutes of work. Proven to work on every GPU since Hopper.

Once that's done, we have clean baselines and real profiling data. THEN we do tile tuning (which becomes MORE effective with better pipelining). THEN TMA epilogue (which now overlaps with a busier mainloop).

By the time we're done with those three, warp specialization might not even be necessary—we'll have hit our perf targets.

**Fund Contestant #1. Let's stop leaving low-hanging fruit on the table.**

---

## FINAL SCORES SUMMARY

```
Pipeline Stages:          8.9/10 ✓ FUND THIS
TMA Epilogue:             7.3/10   (Queue for next round)
Tile Tuning:              7.2/10   (Parallel work, different eng)
Warp Specialization:      5.5/10   (Reserve for v2 or desperation)
```

**SHARK #3 VOTES FOR: CONTESTANT #1**

*"The ROI Maximizer has spoken. Let's maximize it."*
