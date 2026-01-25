# Round [N]: [DESCRIPTIVE TITLE]

> Copy this template for each new round. Fill in all sections before proceeding.

---

## Pre-Round Checklist

- [ ] Previous round results documented
- [ ] Learnings document updated
- [ ] Baseline performance verified
- [ ] Known constraints listed
- [ ] Success criteria defined
- [ ] At least 2 pitches prepared
- [ ] 3 sharks assigned

---

## Context

### Current State
| Metric | Value |
|--------|-------|
| Kernel Version | v[X] |
| Correctness | PASSING / FAILING |
| Best Benchmark | [X] us |
| Target | [X] us |
| Gap to Target | [X]x |

### What We Know
[Summary of prior learnings relevant to this round]

### What We're Trying to Solve
[Specific question or bottleneck this round addresses]

### Constraints (Do Not Violate)
1. [Constraint 1 - e.g., "NVFP4 MMA requires M=128"]
2. [Constraint 2 - e.g., "No multiple CUDA streams"]
3. [Constraint 3 - e.g., "Must pass correctness with rtol=1e-03, atol=1e-03"]

---

## Contestants

### Pitch A: [NAME]

**Problem Identified**: [What bottleneck or issue does this address?]

**Proposed Solution**: [Specific technical approach]

**Expected Impact**: [X]% improvement

**Implementation Complexity**: Low / Medium / High

**Risk Level**: Low / Medium / High

**Evidence/Precedent**: [Why do we think this will work?]

**Rollback Plan**: [How to undo if it fails]

---

### Pitch B: [NAME]

**Problem Identified**: [What bottleneck or issue does this address?]

**Proposed Solution**: [Specific technical approach]

**Expected Impact**: [X]% improvement

**Implementation Complexity**: Low / Medium / High

**Risk Level**: Low / Medium / High

**Evidence/Precedent**: [Why do we think this will work?]

**Rollback Plan**: [How to undo if it fails]

---

### Pitch C: [NAME]

**Problem Identified**: [What bottleneck or issue does this address?]

**Proposed Solution**: [Specific technical approach]

**Expected Impact**: [X]% improvement

**Implementation Complexity**: Low / Medium / High

**Risk Level**: Low / Medium / High

**Evidence/Precedent**: [Why do we think this will work?]

**Rollback Plan**: [How to undo if it fails]

---

### Pitch D: [NAME]

**Problem Identified**: [What bottleneck or issue does this address?]

**Proposed Solution**: [Specific technical approach]

**Expected Impact**: [X]% improvement

**Implementation Complexity**: Low / Medium / High

**Risk Level**: Low / Medium / High

**Evidence/Precedent**: [Why do we think this will work?]

**Rollback Plan**: [How to undo if it fails]

---

## Pitch Validation

Before proceeding to voting, verify each pitch:

| Pitch | Has Specific Solution | Has Impact Estimate | Has Evidence | No Constraint Violations | Valid? |
|-------|----------------------|---------------------|--------------|--------------------------|--------|
| A | Y/N | Y/N | Y/N | Y/N | Y/N |
| B | Y/N | Y/N | Y/N | Y/N | Y/N |
| C | Y/N | Y/N | Y/N | Y/N | Y/N |
| D | Y/N | Y/N | Y/N | Y/N | Y/N |

Invalid pitches are disqualified from voting.

---

## Shark Votes

### Scoring Criteria
| Criterion | Weight | Description |
|-----------|--------|-------------|
| Expected Speedup | 40% | How much improvement is claimed, and how believable? |
| Implementation Complexity | 25% | How hard to implement? (10 = trivial, 0 = impossible) |
| Risk Level | 20% | What can go wrong? (10 = safe, 0 = dangerous) |
| Pitch Quality | 15% | Is the reasoning clear and well-supported? |

### Shark 1: The Skeptic

| Pitch | Speedup (40%) | Complexity (25%) | Risk (20%) | Quality (15%) | **Total** |
|-------|---------------|------------------|------------|---------------|-----------|
| A | /10 | /10 | /10 | /10 | **/10** |
| B | /10 | /10 | /10 | /10 | **/10** |
| C | /10 | /10 | /10 | /10 | **/10** |
| D | /10 | /10 | /10 | /10 | **/10** |

**Vote**: Pitch [X]

**Reasoning**: [Why this pitch?]

---

### Shark 2: The Pragmatist

| Pitch | Speedup (40%) | Complexity (25%) | Risk (20%) | Quality (15%) | **Total** |
|-------|---------------|------------------|------------|---------------|-----------|
| A | /10 | /10 | /10 | /10 | **/10** |
| B | /10 | /10 | /10 | /10 | **/10** |
| C | /10 | /10 | /10 | /10 | **/10** |
| D | /10 | /10 | /10 | /10 | **/10** |

**Vote**: Pitch [X]

**Reasoning**: [Why this pitch?]

---

### Shark 3: The Theorist

| Pitch | Speedup (40%) | Complexity (25%) | Risk (20%) | Quality (15%) | **Total** |
|-------|---------------|------------------|------------|---------------|-----------|
| A | /10 | /10 | /10 | /10 | **/10** |
| B | /10 | /10 | /10 | /10 | **/10** |
| C | /10 | /10 | /10 | /10 | **/10** |
| D | /10 | /10 | /10 | /10 | **/10** |

**Vote**: Pitch [X]

**Reasoning**: [Why this pitch?]

---

## Vote Tally

| Pitch | Shark 1 | Shark 2 | Shark 3 | **Average** | **Votes** |
|-------|---------|---------|---------|-------------|-----------|
| A | /10 | /10 | /10 | **/10** | X |
| B | /10 | /10 | /10 | **/10** | X |
| C | /10 | /10 | /10 | **/10** | X |
| D | /10 | /10 | /10 | **/10** | X |

---

## Winner

**Winning Pitch**: [NAME]

**Final Score**: [X]/10

**Vote Count**: [X] of 3 sharks

**Why It Won**: [Summary of shark reasoning]

---

## Implementation

### Code Changes Made
```
[List files modified and summary of changes]
```

### Correctness Verification
| Test | Result |
|------|--------|
| Output matches reference | PASS / FAIL |
| rtol=1e-03 | PASS / FAIL |
| atol=1e-03 | PASS / FAIL |

**Correctness Status**: PASSING / FAILING

> If FAILING, STOP HERE. Do not benchmark. Document the failure and proceed to escalation.

---

## Results

### Benchmark Results

| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| g=8, K=7168 | X us | X us | +/-X% |
| g=8, K=2048 | X us | X us | +/-X% |
| g=2, K=4096 | X us | X us | +/-X% |
| g=2, K=1536 | X us | X us | +/-X% |
| **Geometric Mean** | X us | X us | **+/-X%** |

### Outcome

**Status**: SUCCESS / FAILURE / BLOCKED / PARTIAL

**Summary**: [One sentence describing what happened]

---

## Lessons Learned

### What Worked
- [Lesson 1]
- [Lesson 2]

### What Didn't Work
- [Lesson 1]
- [Lesson 2]

### New Constraints Discovered
- [Constraint 1]
- [Constraint 2]

### Questions for Next Round
- [Question 1]
- [Question 2]

---

## Next Steps

[ ] Update SHARK_TANK_LEARNINGS.md with lessons
[ ] Update submission.py to new version (if success)
[ ] Create Round [N+1] context document
[ ] Or: Trigger escalation if needed (see ESCALATION_PATHS.md)

---

*Round completed: [DATE]*
*Orchestrator: [NAME/AGENT]*
