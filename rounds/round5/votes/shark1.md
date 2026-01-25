# SHARK 1 VOTE: THE SKEPTIC'S ANALYSIS

## Round 5 Wild Card Evaluation

After four rounds of failures, I've learned one thing: NEVER trust "expected" performance numbers. Let me tear apart each of these pitches.

---

## WILD CARD A: TRITON KERNEL

### Score: 3/10

### What They Claim
- Full Triton kernel could hit ~160 us (optimistic) or ~500 us (realistic)
- Triton epilogue fusion could save 10-30 us

### My Skeptical Analysis

**The good**: Wild Card A is HONEST. They admit right in the pitch: "Triton is NOT the right tool for this job." They acknowledge no native FP4 support, no TMA access, no TMEM access. They even recommend AGAINST their own full kernel approach.

**The bad**: Even their "small win" epilogue fusion saves only 10-30 us out of 400-530 us. That's a 2-7% improvement. After 4 rounds of failure, we need orders of magnitude, not percentages.

**Why I'm skeptical**:
1. The pitch admits it CAN'T access NVFP4 MMA hardware - the ENTIRE POINT of using FP4
2. Software FP4 decode means we'd lose all benefit of FP4 compression
3. Their "honest assessment" is basically "don't pick me for the main task"
4. Even the epilogue fusion is tiny - 10-30 us when we need to cut 400+ us

**The real problem**: Triton simply cannot access the specialized B200 FP4 hardware. This is like trying to win a Formula 1 race in a Honda Civic. Doesn't matter how good the driver is.

**Verdict**: Points for honesty. But honesty about being wrong doesn't make it right.

---

## WILD CARD B: TORCH.COMPILE + CUDA GRAPHS

### Score: 4/10

### What They Claim
- 10-50 us latency achievable (10-50x speedup!)
- "Let the compiler do the hard work"
- "Low risk" with ~50 lines of new code

### My Skeptical Analysis

**The good**: The approach is elegant. CUDA Graphs for launch overhead elimination is a proven technique. torch.compile HAS shown impressive results in other contexts.

**The bad**: This pitch is built on a MASSIVE assumption that goes unverified.

**Why I'm skeptical**:

1. **The `torch._scaled_mm` assumption**: The ENTIRE pitch hinges on `torch._scaled_mm` supporting NVFP4 (float4_e2m1fn_x2) format on B200. Does it? They say "Native B200 FP4 support via torch._scaled_mm is critical" - but never VERIFY it exists! After Round 3 where we discovered we were calling the WRONG KERNEL, I don't trust assumptions.

2. **The "10-50x speedup" is fantasy math**:
   - They claim Python loop overhead is ~50 us and can be eliminated
   - They claim kernel launch is ~10 us x N and can become ~5 us
   - They claim epilogue fusion saves ~25 us
   - Adding these up does NOT get you from 400-530 us to 10-50 us. The math doesn't work.

3. **Where's the actual compute time?**: Even if ALL overhead is eliminated, the actual GEMM computation takes time. They handwave this away. The CuTe kernel is ~150-200 us of actual compute. Where does that go in their "10-50 us" target?

4. **"Let the compiler do the hard work" is cope**: The compiler can't magically make FP4 tensor cores run faster. It can only reduce overhead AROUND the kernel.

5. **Scale factor layout**: They show `to_blocked_format()` but this is non-trivial. Our permuted scale factors are already complex. Getting them into cuBLAS blocked format for `_scaled_mm` could be a massive headache.

**The real problem**: Their best case assumes EVERYTHING works perfectly. After 4 rounds of "it should work" followed by failures, I don't believe "should work" anymore.

**Verdict**: Sounds great on paper. So did pipeline stages. So did tile tuning.

---

## WILD CARD C: STREAM-PARALLEL EXECUTION

### Score: 6/10

### What They Claim
- 4-7x speedup by running 8 groups in parallel
- "We're not changing the kernel - we're changing how we LAUNCH it"
- ~80-120 us target (down from ~530 us)

### My Skeptical Analysis

**The good**:
1. This is ACTUALLY a different dimension of optimization
2. The B200 resource analysis is solid - 192 SMs CAN run multiple groups
3. Memory bandwidth math checks out - 142 MB at 4.5 TB/s = 30 us is not bottleneck
4. "Zero risk of kernel bugs" is TRUE - they're not touching the kernel code
5. Stream parallelism is a PROVEN technique in production systems

**The bad**: The pitch overestimates the benefit and underestimates complications.

**Why I'm skeptical (but less so)**:

1. **Is the current kernel REALLY sequential across groups?**: The pitch claims groups are processed "one after another" but the CuTe kernel grid linearizes ALL groups into ONE launch. Are they truly sequential, or is the GPU scheduler already parallelizing some? Need to verify.

2. **Stream overhead compounds**: Creating 8 streams, 8 events, launching 8 separate kernels, synchronizing 8 times... that's 8x the fixed overhead per kernel. If launch overhead is 10 us, that's 80 us just in launch overhead.

3. **The 4-7x claim is optimistic**: They show 8 groups at ~530 us going to ~80-120 us. That's 4-6x. But their own math shows 512 CTAs on 192 SMs = 2.7 waves. That's not perfect parallelism. More like 3x theoretical maximum from wave reduction alone.

4. **CUDA Graph capture adds complexity**: Graph capture requires fixed tensor sizes. Our problem has variable M, N, K across groups. Per-shape graph caching means memory overhead and cold-start penalties.

5. **The kernel wasn't designed for multi-stream**: The CuTe kernel uses global memory for tensormaps and problem indices. Multiple instances might contend on these resources.

**HOWEVER**:

Unlike the other pitches, this one:
- Uses PROVEN techniques (streams, graphs)
- Doesn't require touching the kernel
- Has realistic expectations (4-7x, not 10-50x)
- Admits its own limitations (2-group cases see less benefit)
- Actually analyzed the hardware resources properly

**The real insight**: Everyone else tried to make the GEMM faster. Wild Card C asks: why are we running one GEMM at a time when we have 192 SMs?

**Verdict**: Most realistic of the three. Still skeptical of the claimed speedup, but the approach is sound.

---

## COMPARATIVE ANALYSIS

| Pitch | Claimed Speedup | My Realistic Estimate | Key Risk | Can Actually Work? |
|-------|-----------------|----------------------|----------|-------------------|
| A: Triton | 2-7% (epilogue only) | 2-5% | No FP4 hardware access | Marginally |
| B: torch.compile | 10-50x | 1-2x if lucky | `_scaled_mm` doesn't support FP4 | Probably not |
| C: Streams | 4-7x | 2-4x | Stream overhead, contention | Yes |

---

## MY VOTE: WILD CARD C (STREAM-PARALLEL)

### Why C wins my vote:

1. **Orthogonal approach**: After 4 rounds of kernel-level failures, trying something OUTSIDE the kernel is smart. We're not repeating past mistakes.

2. **Proven technique**: Multi-stream execution and CUDA Graphs are battle-tested in production. This isn't speculative - cuBLAS, cuDNN, and every major ML framework use these patterns.

3. **Minimal risk**: Worst case, it doesn't help and we fall back to sequential. We don't break anything. Unlike Rounds 1-3, we can't make things worse.

4. **Honest expectations**: 4-7x claimed is more believable than 10-50x. And even if we get 2-3x, that's STILL better than what we have.

5. **Implementation reality**: 2 hours, no kernel changes, uses existing PyTorch APIs. This we can actually BUILD and TEST quickly.

### Why NOT the others:

**Not A (Triton)**: They admitted defeat in their own pitch. "Don't bet on me for the full kernel." When the pitcher tells you not to invest, listen to them.

**Not B (torch.compile)**: Built on an unverified assumption (`torch._scaled_mm` supporting NVFP4). After Round 3's revelation that we were calling the wrong kernel, I refuse to trust another pitch that says "it should use the hardware" without PROVING it.

---

## FINAL SKEPTIC'S NOTE

Even with my vote for Wild Card C, I'm not confident. After 4 rounds of failure:
- Round 1: Pipeline stages made it 30% SLOWER
- Round 2: Tile tuning caused COMPILE ERROR (hardware constraints)
- Round 3: Wrong kernel entirely
- Round 4: Fixed but still slow

Wild Card C at least doesn't repeat these patterns. It's a lateral move - changing HOW we launch rather than WHAT we launch. That's the only reason it gets my vote.

My prediction: 2-3x speedup at best, with ~150-200 us final latency. Still 10x off target, but better than 400-530 us.

---

**SHARK 1 VOTE: WILD CARD C**

*"I'm not voting for the best option. I'm voting for the least likely to fail catastrophically."*

---

| Pitch | Score |
|-------|-------|
| Wild Card A (Triton) | 3/10 |
| Wild Card B (torch.compile) | 4/10 |
| Wild Card C (Streams) | 6/10 |

**Winner: Wild Card C**
