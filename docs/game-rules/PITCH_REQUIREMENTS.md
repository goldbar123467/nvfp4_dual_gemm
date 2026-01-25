# SHARK TANK: Pitch Requirements

## Overview

This document defines what every pitch MUST include to be considered valid, and what automatically disqualifies a pitch.

A pitch is a formal proposal for a specific optimization approach. Vague ideas are not pitches.

---

## Required Elements

Every valid pitch MUST include ALL of the following:

### 1. Problem Statement

**What it is**: A clear description of the bottleneck, inefficiency, or issue being addressed.

**Requirements**:
- Specific and measurable (not "the kernel is slow")
- Explains WHY this is a problem
- Connects to actual performance data if possible

**Good Example**:
> "75% of threads (96 of 128) are idle during MMA compute because only warp 0 executes the compute loop. This leaves 75% of available compute capacity unused."

**Bad Example**:
> "The kernel could be faster."

---

### 2. Proposed Solution

**What it is**: The specific technical approach to solve the problem.

**Requirements**:
- Concrete and implementable (someone could code this from the description)
- States what code changes are needed
- States which files/functions are affected

**Good Example**:
> "Move the MMA compute (`cute.gemm(tiled_mma, ...)`) outside the `if warp_idx == 0` block so all 128 threads participate. Add a `cute.arch.barrier()` after S2T copies to ensure data is ready before MMA."

**Bad Example**:
> "Optimize the kernel to use more threads."

---

### 3. Expected Performance Impact

**What it is**: A quantified estimate of improvement WITH reasoning.

**Requirements**:
- Numeric estimate (percentage or multiplier)
- Reasoning for why this improvement is expected
- NOT a guess - must be derived from analysis

**Good Example**:
> "Expected 2-4x improvement. Currently 32 threads do MMA, after fix 128 threads will. The MMA instruction is cooperative and requires all threads. This directly addresses the 4x thread utilization gap identified in Round 7."

**Bad Example**:
> "Expected significant improvement."

---

### 4. Implementation Complexity Estimate

**What it is**: How hard this will be to implement.

**Requirements**:
- One of: Low / Medium / High
- Estimated time to implement
- List of technical challenges

| Level | Time Estimate | Characteristics |
|-------|---------------|-----------------|
| Low | <2 hours | Small changes, well-understood patterns, clear path |
| Medium | 2-6 hours | Moderate changes, some unknowns, may need debugging |
| High | 6+ hours | Major changes, architectural impact, research needed |

**Good Example**:
> "Medium complexity. Estimated 3-4 hours. Challenges: (1) Ensure barrier placement doesn't cause deadlock, (2) Verify TMA loads still work with changed thread participation, (3) Test all benchmark sizes."

---

### 5. Risk Assessment

**What it is**: What can go wrong and how likely.

**Requirements**:
- One of: Low / Medium / High
- Specific risks identified
- Mitigation strategy for each risk

| Level | Likelihood | Impact |
|-------|------------|--------|
| Low | Unlikely to cause problems | Easy to recover if issues |
| Medium | Some failure modes possible | May require debugging |
| High | Significant chance of problems | Could break correctness |

**Good Example**:
> "Medium risk. Risks: (1) Barrier placement could cause deadlock if not all threads reach it - mitigate by verifying thread convergence, (2) MMA results could differ if thread participation changes semantics - mitigate by running correctness tests before benchmarking."

---

### 6. Evidence or Precedent

**What it is**: Why we believe this will work.

**Requirements**:
- At least one of:
  - Prior success in similar context
  - Profiling data supporting the hypothesis
  - Documentation/literature reference
  - Theoretical analysis

**Good Example**:
> "Evidence: (1) Round 7 profiling showed 75% idle threads, (2) CUTLASS documentation states cute.gemm is a cooperative warp operation requiring all threads, (3) Similar fix in other CUTLASS kernels showed proportional speedup."

**Bad Example**:
> "This is a standard optimization technique."

---

### 7. Rollback Plan

**What it is**: How to undo the change if it fails.

**Requirements**:
- Specific rollback steps
- What state we return to

**Good Example**:
> "Rollback: Revert to submission_v7_final.py. Git commit before changes provides restore point. No external state changes."

---

## Pitch Template

```markdown
# Pitch [X]: [NAME]

## Problem Statement
[What is the bottleneck or issue?]

## Proposed Solution
[Specific technical approach]

### Code Changes
- File: [path]
  - Change: [description]
- File: [path]
  - Change: [description]

## Expected Impact
[X]% / [X]x improvement

**Reasoning**: [Why we expect this improvement]

## Implementation Complexity
**Level**: Low / Medium / High
**Time Estimate**: [X] hours

**Challenges**:
1. [Challenge 1]
2. [Challenge 2]

## Risk Assessment
**Level**: Low / Medium / High

**Risks**:
1. [Risk 1] - Mitigation: [strategy]
2. [Risk 2] - Mitigation: [strategy]

## Evidence/Precedent
1. [Evidence 1]
2. [Evidence 2]

## Rollback Plan
[How to undo if it fails]
```

---

## Automatic Disqualification

A pitch is **automatically disqualified** if it:

### 1. Violates Known Hardware Constraints

**Examples from Rounds 1-8**:
- Tile size M < 128 (Round 2: "MmaMXF4NVF4Op requires M=128")
- Tile size N < 128 (likely same constraint)
- Any assumption that NVFP4 MMA can use different tile shapes

**Before pitching**: Check SHARK_TANK_LEARNINGS.md for hardware constraints.

---

### 2. Violates Competition Rules

**Examples from Rounds 1-8**:
- Using multiple CUDA streams (Round 5: "Your code contains work on another stream")
- Any approach that would fail the competition validator

**Before pitching**: Read the competition rules in task.md.

---

### 3. Has Been Tried Before Without New Evidence

**Examples from Rounds 1-8**:
- Pipeline stages > 1 (Round 1: Made things 30% slower)
- Pre-allocation without profiling evidence (Round 6: Made things 33% slower)

**Exception**: A previously failed approach CAN be pitched if:
- New evidence suggests the original analysis was wrong
- The approach is modified to address the failure reason
- The pitch explicitly acknowledges the prior failure and explains why this time is different

**Before pitching**: Check round results for prior attempts.

---

### 4. Contains Only Vague Claims

Disqualified if the pitch:
- Has no specific code changes
- Has no numeric impact estimate
- Says "should improve performance" without reasoning
- Cannot be implemented from the description alone

---

### 5. Proposes Something Impossible

**Examples**:
- "Use Triton for NVFP4" (Triton cannot access NVFP4 MMA)
- "Reduce M to 64" (Hardware requires M=128)
- "Use cuBLAS FP4" (May not exist or be accessible)

---

## Pitch Validation Checklist

Before a pitch enters voting, verify:

```
REQUIRED ELEMENTS
[ ] Has specific problem statement
[ ] Has specific proposed solution
[ ] Has code changes identified
[ ] Has numeric impact estimate with reasoning
[ ] Has complexity estimate (Low/Medium/High + hours)
[ ] Has risk assessment (Low/Medium/High + specific risks)
[ ] Has evidence or precedent
[ ] Has rollback plan

DISQUALIFICATION CHECKS
[ ] Does NOT violate hardware constraints
[ ] Does NOT violate competition rules
[ ] Has NOT been tried before (or has new evidence)
[ ] Is NOT vague or unimplementable
[ ] Is NOT impossible

VALIDATION RESULT: VALID / INVALID
```

---

## Special Pitch Types

### Research Pitch

Sometimes the best pitch is to investigate rather than optimize.

**When valid**:
- Multiple prior rounds have failed
- Root cause is unclear
- Need profiling data to proceed

**Requirements**:
- Specific investigation approach (not "let's look at the code")
- What we hope to learn
- How the findings will inform the next round

**Example** (Round 7):
> "Profile thread utilization with Nsight to determine if the kernel is compute-bound, memory-bound, or latency-bound. Expected output: Per-warp activity breakdown and bottleneck identification."

---

### Wild Card Pitch

An unconventional approach that challenges assumptions.

**When valid**:
- Conventional approaches have failed
- There's reason to question the current approach
- The wild card has specific investigation steps

**Example** (Round 3):
> "Verify the kernel computes the correct output by comparing against the reference implementation. The 20-100x gap may not be a performance problem - it may be a correctness problem."

---

### Minimal Fix Pitch

A zero-risk approach that prioritizes correctness over performance.

**When valid**:
- Correctness is broken or uncertain
- Prior rounds have had multiple failures
- Need a stable baseline before optimizing

**Example** (Round 4):
> "Call the existing GEMM kernel twice instead of modifying it. Zero GPU code changes = zero new bugs. Fuse SiLU in Python where correctness is trivial to verify."

---

## Lessons from Rounds 1-8

These pitches would have been disqualified if this document existed:

| Round | Pitch | Disqualification Reason |
|-------|-------|------------------------|
| 1 | Pipeline Stages | Lacked evidence that memory latency was the bottleneck |
| 2 | Tile Size 64x128 | Violated hardware constraint (M must be 128) |
| 5 | Stream Parallelism | Violated competition rules (no multiple streams) |
| 5 | Triton Rewrite | Impossible (Triton cannot access NVFP4 MMA) |
| 6 | Pre-allocation | Lacked profiling evidence that Python was the bottleneck |

---

*"The pitch is the contract. If it's not specific enough to implement, it's not a pitch."*
