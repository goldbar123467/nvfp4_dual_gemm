# SHARK TANK: Escalation Paths

## Overview

This document defines what to do when things go wrong. Not every round succeeds, and the process must handle failures gracefully.

---

## Escalation Decision Tree

```
START: Something went wrong
│
├─► All pitches are bad
│   └─► See: "No Valid Pitches"
│
├─► Winner failed during implementation
│   └─► See: "Implementation Failure"
│
├─► Winner made things worse
│   └─► See: "Performance Regression"
│
├─► Correctness is broken
│   └─► See: "Correctness Failure" (HIGHEST PRIORITY)
│
├─► Competition rules block approach
│   └─► See: "External Blocker"
│
├─► Sharks fundamentally disagree
│   └─► See: "Evaluation Deadlock"
│
└─► Multiple rounds failed
    └─► See: "Persistent Failure"
```

---

## Correctness Failure (HIGHEST PRIORITY)

### Symptoms
- Output no longer matches reference
- Validation tests fail
- rtol/atol tolerance exceeded

### IMMEDIATE ACTIONS

1. **STOP all optimization work**
2. **Rollback to last known-correct version**
3. **Do NOT benchmark incorrect code**
4. **Do NOT proceed to next round**

### Root Cause Analysis

Before any further work:

1. Identify exactly which change broke correctness
2. Understand WHY it broke (not just what broke)
3. Document the failure in learnings

### Path Forward

| Situation | Action |
|-----------|--------|
| Recent change broke it | Rollback that specific change |
| Unclear which change | Binary search through commits |
| Fundamental issue discovered | Research Round to understand |

### Evidence from Rounds 1-8

Round 3 discovered the kernel was computing the wrong thing for the first 3 rounds. All performance numbers before that were meaningless.

**Lesson**: Correctness is non-negotiable. A fast wrong answer is worthless.

---

## Performance Regression

### Symptoms
- Benchmark shows kernel got SLOWER
- Implementation "succeeded" but results are worse

### Immediate Actions

1. Verify the benchmark is correct (same test, same hardware)
2. Verify correctness still passes (regression might have changed the computation)
3. Document the regression with exact numbers

### Analysis Questions

| Question | Why It Matters |
|----------|----------------|
| Did ALL benchmarks regress, or just some? | Partial regression suggests size-dependent issue |
| How much regression? | >20% suggests fundamental problem, <10% might be noise |
| Was the hypothesis wrong? | Optimization might address the wrong bottleneck |

### Path Forward

| Regression | Action |
|------------|--------|
| <5% | Might be measurement noise. Re-run benchmarks. |
| 5-20% | Rollback. Add to "Things That Don't Work" list. |
| >20% | Rollback. Mandatory analysis of why hypothesis was wrong. |

### Evidence from Rounds 1-8

- Round 1: Pipeline stages made kernel 30-46% slower (compute-bound, not memory-bound)
- Round 6: Pre-allocation made kernel 33% slower (Python wasn't the bottleneck)

**Lesson**: "Obvious" optimizations often backfire on specialized kernels.

---

## Implementation Failure

### Symptoms
- Code won't compile
- Runtime error during execution
- Implementation took much longer than estimated

### Compile Error Path

1. **Check constraint violation**: Did we hit a hardware limit?
2. **Check API usage**: Did we misuse a library?
3. **Check types**: Are tensor types compatible?

| Compile Error | Likely Cause | Action |
|---------------|--------------|--------|
| "expects M-mode to be 128" | Tile size constraint | This is a hardware limit. Cannot use smaller tiles. |
| "undefined symbol" | Missing import or link | Fix the build, not a fundamental issue |
| Type mismatch | Wrong tensor dtype | Check data type compatibility |

### Runtime Error Path

1. Identify the exact error message
2. Check if it's a known failure mode
3. Determine if the approach is fundamentally blocked

### Estimation Error Path (Took too long)

If implementation is taking >2x the estimated time:

1. **Checkpoint**: Document what's done and what's remaining
2. **Re-evaluate**: Is the complexity estimate wrong?
3. **Decide**: Continue, pivot, or abandon?

### Path Forward

| Situation | Action |
|-----------|--------|
| Compile error due to constraint | Pitch is invalid. Try second-place pitch. |
| Fixable error | Fix it and continue |
| Taking too long | Human decides: continue or try second-place |

### Evidence from Rounds 1-8

Round 2: Compile error "expects M-mode to be 128" revealed a hardware constraint that was not in the original pitch. The pitch was fundamentally impossible.

---

## No Valid Pitches

### Symptoms
- All pitches violate constraints
- All pitches score below 4.0
- No pitch addresses the actual problem

### Analysis Questions

1. Are the constraints correct? (Maybe we learned a wrong lesson)
2. Is the problem statement correct? (Maybe we're asking the wrong question)
3. Do we have enough information? (Maybe we need to research)

### Path Forward

| Situation | Action |
|-----------|--------|
| Constraints are wrong | Re-evaluate learnings, update constraints |
| Problem statement is wrong | Research Round to find real problem |
| Not enough information | Research Round to gather data |
| Pitches are all weak | Request new pitches with clearer guidelines |

### Triggering a Research Round

A Research Round is mandatory when:
- 3+ consecutive rounds have failed
- No pitch scores above 4.0
- Root cause is unknown

Research Round format:
- No implementation
- Goal is information gathering
- Pitches are investigation strategies, not optimizations
- Success = actionable insight discovered

---

## External Blocker

### Symptoms
- Competition rules prevent the approach
- Hardware limitation blocks the approach
- External dependency is unavailable

### Immediate Actions

1. **Document the blocker precisely** (exact error message, rule citation)
2. **Confirm the blocker is real** (not a misunderstanding)
3. **Add to constraints list** (so future pitches don't repeat)

### Path Forward

| Blocker Type | Action |
|--------------|--------|
| Competition rule | Add to constraints, approach is permanently blocked |
| Hardware limit | Add to constraints, approach is permanently blocked |
| Missing dependency | Research if dependency can be added |
| Temporary issue | Wait and retry if appropriate |

### Evidence from Rounds 1-8

Round 5: Stream parallelism would have given 4-7x speedup but was blocked by competition rule "Your code contains work on another stream". This is a permanent constraint.

---

## Evaluation Deadlock

### Symptoms
- Sharks disagree fundamentally (not just on scores)
- 3-way tie with no clear winner
- Sharks are arguing about premises, not pitches

### Immediate Actions

1. **Identify the disagreement**: What exactly do sharks disagree on?
2. **Check if it's a factual question**: Can we resolve it with evidence?
3. **Check if it's a values question**: Do sharks weight criteria differently?

### Path Forward

| Disagreement | Action |
|--------------|--------|
| Factual (e.g., "Is this compute-bound?") | Research to get the answer |
| Scoring (different weights) | Use the defined weights, sharks can't change them |
| Fundamental (different hypotheses) | Add 4th shark as tiebreaker |

### Adding a Fourth Shark

If 3 sharks cannot reach resolution:

1. Add a 4th shark with "Wild Card" persona
2. Wild Card reviews all pitches fresh
3. Include Wild Card's score in the average
4. If still tied, human orchestrator decides

---

## Persistent Failure

### Symptoms
- 3+ rounds have failed in a row
- Each round tried a different approach
- No progress toward target

### Mandatory Actions

1. **Full learnings review**: What have we actually learned?
2. **Constraint audit**: Are our constraints correct?
3. **Assumption audit**: What are we assuming that might be wrong?

### Path Forward: The Reset Protocol

When persistent failure occurs:

1. **Research Round** (mandatory)
   - Profile the kernel extensively
   - Question all assumptions
   - Look for undiscovered constraints

2. **Wild Card Round** (if Research Round doesn't help)
   - Unconventional approaches only
   - "Obvious" solutions banned
   - At least one pitch must question the problem statement

3. **Human Intervention** (if Wild Card fails)
   - Human reviews all learnings
   - Human may redefine the problem
   - Human may bring external expertise

### Evidence from Rounds 1-8

Rounds 1-2 both failed. Round 3 was a Wild Card round that discovered the kernel was fundamentally wrong. This changed everything.

---

## Pivot vs. Persist Decision Framework

### When to Persist (Keep Trying Same Direction)

- The approach is sound but implementation was flawed
- We learned something that can improve the next attempt
- The failure was due to a fixable mistake

### When to Pivot (Try Something Different)

- The approach was fundamentally wrong (not just poorly executed)
- We hit an external blocker (competition rules, hardware limits)
- Multiple attempts in same direction have failed

### Decision Matrix

| Failure Reason | Persist or Pivot? |
|----------------|-------------------|
| Implementation bug | Persist (fix the bug) |
| Wrong hypothesis about bottleneck | Pivot (find real bottleneck) |
| Hardware constraint | Pivot (cannot overcome) |
| Competition rule | Pivot (cannot overcome) |
| Estimation error (took longer) | Persist (if still promising) |
| Performance regression | Pivot (hypothesis was wrong) |

---

## Escalation Authority Matrix

| Escalation | Who Decides | When |
|------------|-------------|------|
| Rollback code | Implementation executor | On failure |
| Try second-place pitch | Round orchestrator | On winner failure |
| Add 4th shark | Round orchestrator | On deadlock |
| Research Round | Round orchestrator | On 3+ failures or no valid pitches |
| Wild Card Round | Round orchestrator | When conventional approaches exhausted |
| Redefine problem | Human oversight | When persistent failure |
| Override vote | Human oversight | With documented justification |

---

## Quick Reference: Escalation Triggers

```
CORRECTNESS BROKEN
└─► STOP. Rollback. Do not benchmark.

REGRESSION > 20%
└─► Rollback. Analyze why. Add to "doesn't work" list.

COMPILE ERROR
└─► Check if constraint violation. If so, try second-place.

NO VALID PITCHES
└─► Research Round to understand problem better.

3+ CONSECUTIVE FAILURES
└─► Mandatory Research Round.

SHARKS DEADLOCKED
└─► Add 4th shark. If still tied, human decides.

COMPETITION RULE BLOCKS
└─► Add to permanent constraints. Approach is dead.
```

---

## Escalation Documentation

Every escalation must be documented:

```markdown
## Escalation Record

**Round**: [N]
**Date**: [DATE]
**Escalation Type**: [from the types above]

### What Happened
[Description of the failure]

### Evidence
[Error messages, benchmark numbers, etc.]

### Decision Made
[What action was taken]

### Outcome
[What resulted from the action]

### Lessons Learned
[What we know now that we didn't before]
```

---

*"The escalation path is not failure - it's the process working. Every escalation teaches us something."*
