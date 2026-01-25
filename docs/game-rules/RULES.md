# SHARK TANK: Master Rules Document

## 1. Purpose

The Shark Tank format is an AI-driven competitive framework for GPU kernel optimization. Instead of a single AI agent blindly trying optimizations, multiple AI agents propose approaches while evaluator AI agents ("Sharks") vote on which to implement.

**Core Philosophy**: The best optimization is the one that survives scrutiny from multiple skeptical evaluators.

### Why This Format Exists

1. **Diverse Perspectives**: Different agents propose different strategies
2. **Critical Evaluation**: Bad ideas get voted down before wasting implementation time
3. **Fail Fast**: One round of deliberation is cheaper than one failed implementation
4. **Learning Accumulates**: Each round's lessons inform the next
5. **Accountability**: Every decision is documented with reasoning

---

## 2. Participation Requirements

### Who Can Be a Contestant

- AI agents capable of:
  - Reading technical specifications
  - Analyzing code and hardware constraints
  - Proposing specific, measurable optimizations
  - Estimating performance impact with reasoning
  - Identifying risks and rollback strategies

### Who Can Be a Shark

- AI agents capable of:
  - Evaluating technical claims critically
  - Scoring proposals against defined criteria
  - Providing clear reasoning for votes
  - Maintaining a consistent evaluation persona (Skeptic, Pragmatist, Theorist)

### Human Involvement

- **Round Orchestrator**: A human oversees the process, triggers phases, and handles escalations
- **Implementation Executor**: A human or AI with access to run benchmarks
- **Tiebreaker**: Human has final say on unresolvable disputes

---

## 3. What Constitutes a Valid Round

A valid Shark Tank round requires ALL of the following:

### Pre-Round Requirements

1. **Context Document**: Summary of current state, prior learnings, and constraints
2. **At Least 2 Pitches**: Minimum two valid contestant pitches (see PITCH_REQUIREMENTS.md)
3. **At Least 3 Sharks**: Three evaluators with distinct personas
4. **Clear Success Metric**: How will we know if the winner worked?
5. **Baseline Measurement**: Current performance numbers to compare against

### During Round Requirements

1. **Pitch Phase Completed**: All contestants submitted pitches
2. **Vote Phase Completed**: All sharks submitted votes with scores and reasoning
3. **Winner Determined**: A single winner selected by the rules below
4. **Implementation Plan**: Clear steps to implement the winner

### Post-Round Requirements

1. **Implementation Attempted**: Winner's approach was actually tried
2. **Results Measured**: Benchmark numbers collected
3. **Outcome Documented**: Success/Failure recorded with evidence
4. **Lessons Captured**: What we learned added to learnings document

---

## 4. Win/Loss Conditions

### How a Pitch Wins

1. **Highest Total Score**: Sum of all shark scores (see VOTING_PROTOCOL.md)
2. **Majority Support**: If scores tie, the pitch with more shark votes wins
3. **Tiebreaker**: See section 5 below

### How an Implementation Succeeds

An implemented pitch is a **SUCCESS** if:

1. **Correctness Preserved**: Output still matches reference (rtol=1e-03, atol=1e-03)
2. **Performance Improved**: Geometric mean of benchmarks is faster than baseline
3. **No Regressions**: No benchmark is more than 5% slower than baseline

An implemented pitch is a **FAILURE** if:

1. **Correctness Broken**: Output no longer matches reference
2. **Performance Regressed**: Geometric mean is slower than baseline
3. **Blocked**: External constraint prevents implementation (e.g., competition rules)
4. **Compile Error**: Code fails to compile

### Special Outcomes

| Outcome | Definition | Action |
|---------|------------|--------|
| SUCCESS | Met all success criteria | Proceed to next round |
| FAILURE | Did not meet success criteria | Document lessons, rollback if needed |
| BLOCKED | External constraint prevents implementation | Document blocker, proceed to next round |
| PARTIAL | Some improvements, some regressions | Document, human decides next step |

---

## 5. Tie-Breaking Procedures

Ties are broken in the following order:

### Score Ties

1. **First Tiebreaker - Risk Level**: Lower risk pitch wins
2. **Second Tiebreaker - Implementation Complexity**: Simpler pitch wins
3. **Third Tiebreaker - Shark Consensus**: Pitch with most unanimous shark support wins
4. **Fourth Tiebreaker - Human Decision**: Round orchestrator decides

### Vote Ties (e.g., 2-2 with 4 sharks)

1. Add a fourth shark with the "Wild Card" persona
2. If still tied, human orchestrator casts deciding vote

### Multi-Way Ties (3+ pitches with same score)

1. Eliminate the pitch with highest risk
2. If still tied, eliminate pitch with highest complexity
3. If still tied, human orchestrator decides

---

## 6. Escalation Procedures

### When to Escalate

| Situation | Escalation Path |
|-----------|-----------------|
| All pitches violate constraints | Pause round, research phase (see ESCALATION_PATHS.md) |
| Winner implementation fails | Rollback, try second-place pitch |
| Correctness broken | STOP all optimization, fix correctness first |
| Competition rules block approach | Document, remove approach from future consideration |
| Sharks fundamentally disagree | Add fourth shark or human tiebreaker |
| No improvement after 3 rounds | Mandatory research round |

### Escalation Authority

1. **AI Agents**: Can flag issues, cannot override rules
2. **Round Orchestrator**: Can pause rounds, request research, add sharks
3. **Human Oversight**: Final authority on all disputes

---

## 7. Rule Amendments Process

### Proposing an Amendment

1. Any participant can propose a rule change
2. Proposal must include:
   - Current rule text
   - Proposed new text
   - Rationale for change
   - Evidence from rounds that supports the change

### Approving an Amendment

1. **Minor Changes** (clarifications, typos): Round orchestrator can approve
2. **Major Changes** (scoring weights, phase structure): Requires:
   - Documentation of the problem the change solves
   - At least one round of evidence showing the problem
   - Approval from round orchestrator

### Documenting Amendments

All amendments must be documented in this file with:
- Date of change
- Previous rule text
- New rule text
- Reason for change
- Round(s) that motivated the change

---

## Amendment History

| Date | Rule | Change | Reason | Evidence Round |
|------|------|--------|--------|----------------|
| (Initial) | All | Initial rule set | Based on Rounds 1-8 learnings | Rounds 1-8 |

---

## Quick Reference: Round Checklist

```
PRE-ROUND
[ ] Context document created
[ ] At least 2 pitches submitted
[ ] All pitches pass PITCH_REQUIREMENTS validation
[ ] 3 sharks assigned with distinct personas
[ ] Baseline benchmark recorded
[ ] Success metric defined

DURING ROUND
[ ] All sharks reviewed all pitches
[ ] All sharks submitted scores (0-10 scale)
[ ] All sharks submitted reasoning
[ ] Winner calculated per VOTING_PROTOCOL
[ ] Tiebreaker applied if needed

POST-ROUND
[ ] Winner implementation attempted
[ ] Correctness verified BEFORE benchmarking
[ ] Benchmarks run (minimum 3 runs, median reported)
[ ] Results documented in round_results.md
[ ] Lessons added to SHARK_TANK_LEARNINGS.md
[ ] Next round context prepared (if continuing)
```

---

*"Before optimizing, verify you're solving the right problem."*
*- Round 3 Wild Card, the pitch that found the kernel was computing the wrong thing*
