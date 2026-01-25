# SHARK 1 VOTE: Prof. Williams (The Principal Investigator)

**Stakeholder Interest**: Novelty and Publishability

---

## Evaluation Criteria

As a PI, I ask:
1. Is this novel enough for a paper?
2. Does this advance state of the art?
3. Can we cite this as a contribution?
4. Will this impress at MLSYS/SC?

---

## Pitch Evaluations

### Pitch A: Dr. Chen (Triton Fused Dual-GEMM)

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| Novelty | 7/10 | Triton for FP4 dual-GEMM is novel, but Triton itself isn't |
| Technical Depth | 9/10 | Excellent memory traffic analysis |
| Publishability | 7/10 | Could be workshop paper material |
| Risk-Adjusted | 6/10 | Triton FP4 on SM100 is unproven |

**Total: 7.25/10**

*Comment*: Strong theoretical foundation, but using Triton instead of native CUDA limits the novelty. A paper would need to focus on the analysis, not the implementation.

---

### Pitch B: Dr. Santos (Persistent + Fused Epilogue)

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| Novelty | 5/10 | Both techniques are well-known |
| Technical Depth | 7/10 | Good practical analysis |
| Publishability | 4/10 | Engineering report, not research |
| Risk-Adjusted | 7/10 | Proven patterns, lower risk |

**Total: 5.75/10**

*Comment*: This is engineering, not research. Useful for production, but I can't publish "we used existing techniques."

---

### Pitch C: Dr. Kim (Conservative Optimization)

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| Novelty | 2/10 | Configuration changes aren't research |
| Technical Depth | 5/10 | Solid but shallow |
| Publishability | 1/10 | Cannot publish this |
| Risk-Adjusted | 9/10 | Lowest risk, but lowest ceiling |

**Total: 4.25/10**

*Comment*: I appreciate the caution, but this won't advance our research agenda. We need publishable results.

---

### Pitch D: Dr. Okonkwo (CUTLASS Dual-Accumulator)

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| Novelty | 9/10 | Cross-domain insight, novel for NVFP4 |
| Technical Depth | 9/10 | Deep understanding of Flash Attention parallel |
| Publishability | 9/10 | "Flash Attention patterns for FP4 GLU" - publishable! |
| Risk-Adjusted | 6/10 | High complexity, but high reward |

**Total: 8.25/10**

*Comment*: THIS is research. The Flash Attention parallel is a genuine insight. Even if implementation fails, the framing is publishable.

---

## Final Ranking

| Rank | Pitch | Score | Notes |
|------|-------|-------|-------|
| 1 | D (Okonkwo) | 8.25 | Most publishable, genuine insight |
| 2 | A (Chen) | 7.25 | Good analysis, Triton limits novelty |
| 3 | B (Santos) | 5.75 | Engineering, not research |
| 4 | C (Kim) | 4.25 | Can't publish configuration tuning |

---

## My Votes (2 selections)

### Primary Vote: **Pitch D (Dr. Okonkwo)**

**Reasoning**: The dual-accumulator CUTLASS approach with EVT fusion is genuinely novel for NVFP4 workloads. The Flash Attention parallel provides a narrative hook for publication. Even partial success gives us a workshop paper.

### Secondary Vote: **Pitch A (Dr. Chen)**

**Reasoning**: The memory traffic analysis is rigorous. If Triton FP4 works on Blackwell, we have a compelling story about democratizing high-performance FP4 kernels.

---

## Submission Preferences

If my votes win:
- **Combined (D+A)**: CUTLASS dual-accumulator with Triton fallback
- **Single D**: CUTLASS-only approach
- **Single A**: Triton-only approach

The combined submission hedges our bets while maintaining publishability.

---

*"Research impact matters. We need results that advance the field, not just improve our benchmark."*

— Prof. Williams
