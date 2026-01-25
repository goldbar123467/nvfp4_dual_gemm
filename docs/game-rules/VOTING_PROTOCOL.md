# SHARK TANK: Voting Protocol

## Scoring System

Each shark evaluates each pitch on four weighted criteria. Scores are 0-10 for each criterion.

### Scoring Criteria

| Criterion | Weight | What It Measures |
|-----------|--------|------------------|
| **Expected Speedup** | 40% | How much improvement is claimed AND how believable is the claim? |
| **Implementation Complexity** | 25% | How hard is this to implement? (10 = trivial, 0 = impossible) |
| **Risk Level** | 20% | What can go wrong? (10 = very safe, 0 = extremely risky) |
| **Pitch Quality** | 15% | Is the reasoning clear, specific, and well-supported? |

### Why These Weights?

- **Speedup (40%)**: We're here to optimize. If it doesn't improve performance, it doesn't matter how easy it is.
- **Complexity (25%)**: Simpler implementations are less likely to introduce bugs and can be iterated faster.
- **Risk (20%)**: After Rounds 1-6, we learned that risk management matters. Many "obvious" optimizations made things worse.
- **Quality (15%)**: Well-reasoned pitches with evidence are more likely to succeed.

---

## Score Calculation

### Individual Shark Score

For each pitch, a shark calculates:

```
Total = (Speedup * 0.40) + (Complexity * 0.25) + (Risk * 0.20) + (Quality * 0.15)
```

**Example**:
| Criterion | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Speedup | 8 | 0.40 | 3.20 |
| Complexity | 7 | 0.25 | 1.75 |
| Risk | 6 | 0.20 | 1.20 |
| Quality | 9 | 0.15 | 1.35 |
| **Total** | | | **7.50** |

### Final Score

The final score for a pitch is the **average of all shark scores**:

```
Final Score = (Shark1 + Shark2 + Shark3) / 3
```

---

## Scoring Rubrics

### Expected Speedup (40%)

| Score | Description |
|-------|-------------|
| 10 | >10x improvement with strong evidence (profiling data, prior success) |
| 9 | 5-10x improvement with evidence |
| 8 | 2-5x improvement with evidence |
| 7 | 50-100% improvement with evidence |
| 6 | 20-50% improvement with evidence |
| 5 | 10-20% improvement with evidence |
| 4 | 5-10% improvement with evidence |
| 3 | <5% improvement OR claims without evidence |
| 2 | Uncertain improvement, speculation only |
| 1 | Likely no improvement |
| 0 | Likely regression |

**Key Question**: "How much faster will this make the kernel, and why do we believe it?"

### Implementation Complexity (25%)

| Score | Description |
|-------|-------------|
| 10 | One-line change, no new logic |
| 9 | Small change (<20 lines), well-understood pattern |
| 8 | Moderate change (20-50 lines), clear implementation path |
| 7 | Moderate change with some unknowns |
| 6 | Significant change (50-100 lines), some research needed |
| 5 | Significant change with multiple unknowns |
| 4 | Major change (100-200 lines), substantial research needed |
| 3 | Major refactor, architectural changes required |
| 2 | Near-rewrite, many unknowns |
| 1 | Requires fundamental architectural redesign |
| 0 | Impossible with current infrastructure |

**Key Question**: "How long will this take to implement correctly?"

### Risk Level (20%)

| Score | Description |
|-------|-------------|
| 10 | Zero risk of breaking correctness, easy rollback |
| 9 | Very low risk, changes are isolated |
| 8 | Low risk, well-understood changes |
| 7 | Low-moderate risk, some edge cases possible |
| 6 | Moderate risk, requires careful testing |
| 5 | Moderate risk, touches critical code paths |
| 4 | Moderate-high risk, multiple failure modes |
| 3 | High risk, could break correctness |
| 2 | High risk, could cause compile errors |
| 1 | Very high risk, violates known constraints |
| 0 | Will definitely fail (violates hardware requirements, competition rules, etc.) |

**Key Question**: "What could go wrong, and how bad would it be?"

### Pitch Quality (15%)

| Score | Description |
|-------|-------------|
| 10 | Specific solution, clear reasoning, strong evidence, addresses potential issues |
| 9 | Specific solution with clear reasoning and some evidence |
| 8 | Specific solution with clear reasoning |
| 7 | Specific solution, reasoning present but incomplete |
| 6 | Solution provided, some vagueness in approach |
| 5 | Solution provided but lacks specificity |
| 4 | Vague solution, hand-wavy reasoning |
| 3 | Very vague, more of an idea than a plan |
| 2 | Just a concept, no implementation details |
| 1 | Unclear what is even being proposed |
| 0 | No coherent pitch |

**Key Question**: "Do I understand exactly what to implement and why it should work?"

---

## Tie-Breaking Procedures

### Score Ties

When two or more pitches have the same final score:

1. **First Tiebreaker - Lower Risk Wins**
   - Compare the average Risk scores
   - Lower risk = safer bet

2. **Second Tiebreaker - Lower Complexity Wins**
   - Compare the average Complexity scores
   - Simpler = faster to implement and iterate

3. **Third Tiebreaker - More Unanimous Support Wins**
   - Compare standard deviation of shark scores
   - Lower variance = more agreement

4. **Fourth Tiebreaker - Human Decision**
   - Round orchestrator reviews and decides

### Vote Ties (Shark Majority)

When counting shark votes (which pitch each shark voted for):

- **3 sharks, all different votes**: Use score-based winner
- **3 sharks, 2-1 split**: Majority wins
- **4 sharks, 2-2 split**: Use score-based winner

---

## Abstention Rules

### When a Shark Can Abstain

A shark may abstain from scoring a pitch if:

1. **Conflict of Interest**: Shark "authored" a similar pitch in a prior round
2. **Insufficient Expertise**: Pitch requires domain knowledge shark lacks
3. **Incomplete Information**: Pitch is missing critical details (note: this may also mean the pitch should be disqualified)

### How Abstention Works

- Abstaining shark does not score the pitch
- Final score is calculated from remaining shark scores
- If >1 shark abstains on same pitch, pitch is flagged for review

### Abstention Limits

- A shark can abstain on at most 1 pitch per round
- If a shark needs to abstain on multiple pitches, replace the shark

---

## Unanimous vs. Majority Rules

### Standard Voting

- **Majority Required**: Pitch with highest average score wins
- **No Minimum Threshold**: Even a 5.1 vs 5.0 decides it

### Veto Power (Special Cases)

A shark can invoke a **veto** if a pitch violates:

1. Known hardware constraints (e.g., "M must be 128")
2. Known competition rules (e.g., "No multiple streams")
3. Prior proven failures (e.g., "We tried this in Round 2")

**Veto process**:
- Shark states the veto and cites evidence
- If evidence is valid, pitch is disqualified
- If evidence is disputed, human orchestrator decides

### Unanimous Rejection

If all sharks score a pitch below 4.0:
- Pitch is flagged as "weak"
- If all pitches are below 4.0, trigger Research Round (see ESCALATION_PATHS.md)

---

## Override Rules

### Can Sharks Override the Vote?

**No.** Once votes are tallied, the highest-scoring pitch wins. Sharks cannot override after the fact.

### Can Humans Override the Vote?

**Yes, but rarely.** Human override is permitted when:

1. New information emerges after voting (e.g., constraint discovered)
2. Scoring error was made (e.g., math mistake)
3. Pitch was misunderstood by all sharks

**Override process**:
- Human documents the reason for override
- Original vote is preserved in record
- Override is logged in round results

---

## Vote Recording Format

Each shark's vote should be recorded as follows:

```markdown
## Shark [N]: [Persona Name]

### Scores

| Pitch | Speedup (40%) | Complexity (25%) | Risk (20%) | Quality (15%) | **Total** |
|-------|---------------|------------------|------------|---------------|-----------|
| A: [Name] | X/10 | X/10 | X/10 | X/10 | **X.XX** |
| B: [Name] | X/10 | X/10 | X/10 | X/10 | **X.XX** |
| C: [Name] | X/10 | X/10 | X/10 | X/10 | **X.XX** |
| D: [Name] | X/10 | X/10 | X/10 | X/10 | **X.XX** |

### Vote: Pitch [X]

### Reasoning
[2-3 sentences explaining why this pitch won their vote]

### Concerns
[Any risks or issues they want flagged for implementation]
```

---

## Vote Tally Format

After all sharks vote, create a summary:

```markdown
## Round [N] Vote Tally

### Final Scores

| Pitch | Shark 1 | Shark 2 | Shark 3 | **Average** | **Rank** |
|-------|---------|---------|---------|-------------|----------|
| A: [Name] | X.XX | X.XX | X.XX | **X.XX** | #X |
| B: [Name] | X.XX | X.XX | X.XX | **X.XX** | #X |
| C: [Name] | X.XX | X.XX | X.XX | **X.XX** | #X |
| D: [Name] | X.XX | X.XX | X.XX | **X.XX** | #X |

### Vote Distribution

| Pitch | Votes |
|-------|-------|
| A | X |
| B | X |
| C | X |
| D | X |

### Winner: Pitch [X] - [NAME]

**Score**: X.XX / 10
**Votes**: X of 3

**Summary**: [One sentence on why this pitch won]
```

---

## Quick Reference: Scoring Checklist

Before submitting scores, each shark should verify:

- [ ] I scored all valid pitches
- [ ] My scores are 0-10 integers
- [ ] I documented reasoning for my vote
- [ ] I flagged any constraint violations
- [ ] I noted any concerns for implementation
- [ ] My weighted calculation is correct

---

*"The format removes bias. The weights encode our learnings. The math decides."*
