# SHARK TANK: Game Show Format

## Overview

Each Shark Tank round consists of 5 phases executed in sequence. This document defines the complete structure for running a round from start to finish.

---

## Round Phases

```
┌────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: CONTEXT          [15-30 min]                                 │
│  └─► Prepare round context, verify constraints, check prior learnings  │
│                                                                        │
│  PHASE 2: PITCH            [30-60 min]                                 │
│  └─► Contestants submit optimization proposals                         │
│                                                                        │
│  PHASE 3: DELIBERATION     [30-45 min]                                 │
│  └─► Sharks review, analyze, and score pitches                         │
│                                                                        │
│  PHASE 4: VOTE             [15 min]                                    │
│  └─► Sharks submit final votes, winner determined                      │
│                                                                        │
│  PHASE 5: IMPLEMENT        [2-8 hours]                                 │
│  └─► Winner is implemented, tested, benchmarked                        │
│                                                                        │
│  PHASE 6: DEBRIEF          [15-30 min]                                 │
│  └─► Document results, capture learnings, prepare next round           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Context (15-30 minutes)

### Purpose
Ensure all participants understand the current state before proposing new optimizations.

### Required Inputs
1. **Current Performance**: Baseline benchmark numbers
2. **Prior Round Results**: What was tried, what worked/failed
3. **Known Constraints**: Hardware limits, competition rules, etc.
4. **Target Goal**: What are we trying to achieve?

### Deliverable
A `roundN_context.md` file containing:
- Current kernel version
- Baseline performance numbers
- Summary of all prior learnings
- Constraints that must not be violated
- Specific question for this round (e.g., "How do we improve thread utilization?")

### Context Template

```markdown
# Round N Context

## Current State
- Kernel Version: vX
- Performance: [benchmark results]
- Correctness: PASSING / FAILING

## Prior Learnings
[Summary of what we've learned from previous rounds]

## Constraints (Do Not Violate)
- [Constraint 1]
- [Constraint 2]

## This Round's Question
[What are we trying to solve?]

## Success Criteria
[How will we know if the winner worked?]
```

---

## Phase 2: Pitch (30-60 minutes)

### Purpose
Generate diverse optimization proposals from multiple perspectives.

### Number of Contestants
- **Minimum**: 2 pitches
- **Recommended**: 4 pitches
- **Maximum**: 6 pitches (more becomes unwieldy for sharks to evaluate)

### Contestant Selection Methods

**Option A: Role-Based Contestants**
| Contestant | Focus |
|------------|-------|
| The Incrementalist | Safe, small improvements |
| The Refactorer | Code structure changes |
| The Hardware Expert | GPU-specific optimizations |
| The Wild Card | Unconventional approaches |

**Option B: Strategy-Based Contestants**
| Contestant | Approach |
|------------|----------|
| Pitch A | Address the suspected bottleneck directly |
| Pitch B | Alternative hypothesis about the bottleneck |
| Pitch C | Higher-risk, higher-reward approach |
| Pitch D | Research/investigation if unclear |

### Contestant Briefing

Each contestant receives:
1. The round context document
2. Access to the current codebase
3. All prior learnings documentation
4. The pitch requirements checklist (see PITCH_REQUIREMENTS.md)

### Deliverable
Each contestant produces a `roundN_pitchX.md` file meeting all requirements.

---

## Phase 3: Deliberation (30-45 minutes)

### Purpose
Allow sharks to thoroughly evaluate each pitch before voting.

### Number of Sharks
- **Required**: 3 sharks
- **Tiebreaker**: 4th shark added if needed

### Shark Personas

| Shark | Persona | Key Questions They Ask |
|-------|---------|------------------------|
| **The Skeptic** | "Prove it works" | Where's the evidence? What could go wrong? |
| **The Pragmatist** | "Can we ship it?" | How long to implement? What's the risk? |
| **The Theorist** | "Why does this happen?" | What's the root cause? Does the math check out? |

### Shark Briefing

Each shark receives:
1. The round context document
2. All submitted pitches
3. The voting protocol (see VOTING_PROTOCOL.md)
4. Their persona description

### Deliberation Activities
1. Read all pitches
2. Identify claims that need verification
3. Check pitches against known constraints
4. Calculate scores using the voting rubric
5. Prepare reasoning for each score

### Deliverable
Each shark produces notes on each pitch (used in vote phase).

---

## Phase 4: Vote (15 minutes)

### Purpose
Formally select the winning pitch through structured evaluation.

### Voting Process
1. Each shark scores each pitch on 4 criteria (0-10 scale)
2. Sharks submit votes independently (no discussion)
3. Scores are tallied per VOTING_PROTOCOL.md
4. Winner is announced with reasoning

### Deliverable
A `roundN_sharkX_vote.md` file for each shark containing:
- Scores for each pitch
- Reasoning for each score
- Final vote selection

Plus a `roundN_vote_tally.md` summarizing all votes and the winner.

---

## Phase 5: Implement (2-8 hours)

### Purpose
Actually build and test the winning optimization.

### Implementation Steps

```
1. SETUP (15 min)
   └─► Create working branch
   └─► Verify baseline still passes correctness

2. IMPLEMENT (1-4 hours)
   └─► Make the code changes
   └─► Follow the pitch's implementation plan

3. VERIFY CORRECTNESS (30 min)
   └─► Run validation tests
   └─► Check output matches reference
   └─► IF FAILS: STOP and document

4. BENCHMARK (30 min)
   └─► Run full benchmark suite
   └─► Minimum 3 runs per benchmark
   └─► Report median results

5. COMPARE (15 min)
   └─► Calculate improvement/regression
   └─► Determine success/failure
```

### Critical Rule: Correctness Before Performance

**ALWAYS verify correctness BEFORE benchmarking.**

Evidence from Round 3: The kernel was 20-100x slower than target because it was computing the wrong thing. Three rounds were wasted optimizing a broken kernel.

### Deliverable
A `roundN_implementation.md` documenting:
- Code changes made
- Correctness verification results
- Benchmark results
- Comparison to baseline

---

## Phase 6: Debrief (15-30 minutes)

### Purpose
Capture learnings and prepare for the next round.

### Debrief Activities
1. Document the outcome (SUCCESS/FAILURE/BLOCKED)
2. Update SHARK_TANK_LEARNINGS.md with new knowledge
3. Identify what to try next (if continuing)
4. Archive round artifacts

### Deliverable
A `roundN_results.md` summarizing:
- Winner and vote counts
- Implementation outcome
- Performance numbers (before/after)
- Key lessons learned
- Recommendations for next round

---

## Time Allocations Summary

| Phase | Minimum | Typical | Maximum |
|-------|---------|---------|---------|
| Context | 15 min | 20 min | 30 min |
| Pitch | 30 min | 45 min | 60 min |
| Deliberation | 30 min | 35 min | 45 min |
| Vote | 10 min | 15 min | 20 min |
| Implement | 2 hours | 4 hours | 8 hours |
| Debrief | 15 min | 20 min | 30 min |
| **Total** | **3.5 hours** | **6 hours** | **10.5 hours** |

---

## Inter-Round Gap Procedures

### Between Rounds (Same Day)
1. Review previous round results
2. Update context document
3. Adjust constraints if needed
4. Begin next round

### Between Rounds (Different Days)
1. Full re-read of learnings document
2. Re-verify baseline performance (code may have changed)
3. Re-check competition rules (may have updated)
4. Fresh context document

### After a Failed Round
1. Document failure thoroughly
2. Add failure to "Things That Don't Work" list
3. Consider: Should runner-up be tried, or new pitches needed?
4. If 3+ consecutive failures: Mandatory Research Round

---

## Special Round Types

### Research Round

When the path forward is unclear, run a Research Round:

**Purpose**: Investigate rather than optimize

**Contestant Pitches**:
- Profile the kernel with Nsight
- Analyze thread utilization
- Check for hidden bottlenecks
- Review competition leaderboard approaches

**Success Criteria**: New actionable information discovered

**Evidence**: Round 7 was a research round that discovered 75% of threads were idle.

### Wild Card Round

When conventional approaches have failed:

**Purpose**: Try unconventional ideas

**Contestant Requirements**:
- At least one pitch must be something never tried before
- "Obvious" approaches are not allowed

**Evidence**: Round 3's Wild Card pitch found the fundamental bug.

---

## Artifact Checklist

At the end of each round, these files should exist:

```
shark_tank/rounds/
├── roundN_context.md         # Phase 1 output
├── roundN_pitchA.md          # Phase 2 output (per contestant)
├── roundN_pitchB.md
├── roundN_pitchC.md
├── roundN_pitchD.md
├── roundN_shark1_vote.md     # Phase 4 output (per shark)
├── roundN_shark2_vote.md
├── roundN_shark3_vote.md
├── roundN_vote_tally.md      # Phase 4 summary
├── roundN_implementation.md  # Phase 5 output
└── roundN_results.md         # Phase 6 output
```

---

*"The format is the firewall. It prevents one bad idea from wasting eight hours of implementation time."*
