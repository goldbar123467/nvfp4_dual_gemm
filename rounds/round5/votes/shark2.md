# SHARK 2: THE PRAGMATIST - ROUND 5 WILD CARD VOTE

---

## Implementation Time Estimates

| Pitch | Time to MVP | Time to Production |
|-------|-------------|-------------------|
| Wild Card A (Triton) | 1-2 hours (epilogue only) / 10+ hours (full kernel) | N/A (full) / 3 hours (epilogue) |
| Wild Card B (torch.compile) | 30 min - 1 hour | 2-4 hours |
| Wild Card C (Streams) | 1-2 hours | 2-3 hours |

---

## Risk/Reward Analysis

### Wild Card A: Triton Kernel

**Reward**: Best case ~160 us (2.7x improvement), realistic ~500 us (no improvement)
**Risk**: HIGH - Triton has NO native FP4 support

| Factor | Score |
|--------|-------|
| Upside | LOW - Cannot access NVFP4 MMA hardware |
| Downside | HIGH - Software FP4 decode kills performance |
| Testability | MEDIUM - Easy to test, but results predictable |
| Learning | LOW - We already know Triton can't do FP4 |

**Honest assessment from the pitch itself**: "Triton is NOT the right tool for this job." The pitch author admits defeat gracefully. That's honest, but it means this is a non-starter.

The epilogue-only variant could save 10-30 us. That's real but not transformative.

**Risk/Reward Ratio**: 8:1 (bad)

---

### Wild Card B: torch.compile + CUDA Graphs

**Reward**: Claims 10-50 us (10-50x speedup)
**Risk**: MEDIUM-HIGH - Depends on torch._scaled_mm FP4 support

| Factor | Score |
|--------|-------|
| Upside | POTENTIALLY HIGH - If _scaled_mm works |
| Downside | MEDIUM - Falls back to slow path if not |
| Testability | HIGH - 30 minutes to know if it works |
| Learning | HIGH - Validates PyTorch FP4 infrastructure |

**The catch**: This pitch claims torch._scaled_mm can use "Blackwell FP4 tensor cores" but doesn't verify this. The scale factor format conversion is handwaved. If `torch._scaled_mm` doesn't actually support our specific FP4 format and scale factor layout, the whole thing fails.

Also, the 10-50x speedup claim is optimistic. Python overhead and launch overhead are real, but they're not 90% of our 400-530 us. The GEMM computation itself is substantial.

**Risk/Reward Ratio**: 3:1 (moderate)

---

### Wild Card C: Stream-Parallel Execution

**Reward**: 4-7x speedup for 8-group workloads (80-120 us from 530 us)
**Risk**: LOW-MEDIUM - Uses proven CUDA technique

| Factor | Score |
|--------|-------|
| Upside | MODERATE - 4-7x is significant |
| Downside | LOW - We're not touching the kernel |
| Testability | HIGH - Easy to test, clear pass/fail |
| Learning | HIGH - Confirms whether groups are truly sequential |

**The key insight**: This pitch correctly identifies that we're processing groups sequentially when B200 has 192 SMs that could run them in parallel. The math checks out:
- 512 CTAs across 8 groups
- 192 SMs
- ~2.7 waves instead of 8 sequential waves

**What I like**:
1. We're NOT touching the kernel - zero risk of kernel bugs
2. It's a PROVEN technique (cuBLAS uses this)
3. Quick to implement and test
4. If it fails, we learn something valuable

**Risk/Reward Ratio**: 1.5:1 (good)

---

## MY VOTE: WILD CARD C (Stream-Parallel Execution)

### Pragmatic Reasoning

**1. It ACTUALLY addresses a real problem**

The pitch correctly identifies that our current kernel processes groups sequentially. This is verifiable. If the analysis is correct, we're leaving 4-7x performance on the table just from launch architecture.

**2. Fastest path to testable hypothesis**

I can test this in 1-2 hours. The code changes are minimal:
- Create stream pool
- Launch groups on separate streams
- Synchronize

If it works, great. If it doesn't, we've only spent 2 hours and we've learned something.

**3. Zero kernel risk**

We're not modifying the CuTe kernel at all. No compile errors. No register pressure changes. No tile size changes. We're just changing HOW we launch it.

**4. The claims are CONSERVATIVE**

Wild Card B claims 10-50x. Wild Card C claims 4-7x. The conservative estimate is more believable and still meaningful.

**5. Compound potential**

If stream parallelism gives us 4-7x, AND we later optimize the kernel, the gains stack. Stream parallelism is orthogonal to other optimizations.

---

## Why NOT the others?

### Wild Card A (Triton)

The pitch author literally says "DO NOT INVEST in Full Triton Kernel" and "Triton is NOT the right tool for this job." I respect that honesty. The epilogue-only variant saves 10-30 us - not worth the complexity right now.

### Wild Card B (torch.compile)

The claims are too optimistic and the critical dependency (`torch._scaled_mm` with FP4) is unverified. If it works, it's magic. But "if it works" is doing a lot of heavy lifting. The scale factor format conversion is also glossed over - our cuBLAS blocked format is non-trivial.

The CUDA graph aspect is good, but that's also what Wild Card C proposes, and C doesn't require us to rewrite the compute path.

---

## Expected Outcome

| Scenario | Probability | Result |
|----------|-------------|--------|
| Stream parallelism works as claimed | 60% | 80-120 us (4-6x speedup) |
| Partial benefit due to contention | 30% | 200-300 us (2x speedup) |
| No benefit (already parallel somehow) | 10% | 400-530 us (no change) |

Even the worst case is just "no change" - we don't make things worse.

---

## Final Word

Four rounds of failed optimizations taught me: **test the simple hypothesis first**.

Wild Card C asks a simple question: "Are we leaving parallelism on the table?"

That's testable in 2 hours. If the answer is yes, we get 4-7x. If the answer is no, we've only lost 2 hours and we've eliminated a hypothesis.

That's pragmatism.

---

**SHARK 2 VOTES: WILD CARD C (Stream-Parallel Execution)**

*"Don't rewrite the engine. Just stop running it one cylinder at a time."*

