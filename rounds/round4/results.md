# SHARK TANK ROUND 4: THE FIX

---

```
  ____   ___  _   _ _   _ ____    _  _
 |  _ \ / _ \| | | | \ | |  _ \  | || |
 | |_) | | | | | | |  \| | | | | | || |_
 |  _ <| |_| | |_| | |\  | |_| | |__   _|
 |_| \_\\___/ \___/|_| \_|____/     |_|

 _____ _   _ _____   _____ _____  __
|_   _| | | | ____| |  ___|_ _\ \/ /
  | | | |_| |  _|   | |_   | | \  /
  | | |  _  | |___  |  _|  | | /  \
  |_| |_| |_|_____| |_|   |___/_/\_\
```

---

## PREVIOUSLY ON SHARK TANK...

*[Dramatic recap music]*

**ROUND 1: Pipeline Stages**
- "One line change! 1.5x speedup!"
- **RESULT:** 30% SLOWER

**ROUND 2: Tile Size Tuning**
- "The sharks didn't listen to me!"
- **RESULT:** COMPILE ERROR

**ROUND 3: The Wild Card**
- "Have YOU even looked at your kernel?"
- **RESULT:** DISCOVERED THE KERNEL IS WRONG

---

## TONIGHT'S EPISODE: "THE RECKONING"

*[Host walks onto stage, looking nervous]*

**HOST:** "Welcome back, folks. After three rounds of... let's call them 'learning experiences'... we finally know what's wrong."

*[Points to giant screen]*

**HOST:** "The task asks for: `C = silu(A @ B1) * (A @ B2)`"

*[Screen shows current kernel]*

**HOST:** "The kernel computes: `C = A @ B`"

*[Audience gasps]*

**HOST:** "That's right. We've been optimizing HALF a kernel. Tonight, we FIX IT."

---

## THE PITCHES

### CONTESTANT A: SEQUENTIAL DUAL GEMM

**SEQUENTIAL:** "I propose the straightforward approach. Two separate accumulator tensors. Two separate mainloops. Clean and simple."

**SHARKS:** "What's the catch?"

**SEQUENTIAL:** "Double the TMEM allocation. Double the register pressure. It will work, but it won't be fast."

**SHARK 1:** "How much slower?"

**SEQUENTIAL:** "Maybe 50-70% of theoretical peak. But at least it will be CORRECT."

---

### CONTESTANT B: INTERLEAVED DUAL GEMM

**INTERLEAVED:** "Load A once, use it twice. For each K-tile, compute both partial products. Maximum A reuse."

**SHARKS:** "Sounds complex."

**INTERLEAVED:** "It is. Need ping-pong between two accumulators. Need careful synchronization. But the A bandwidth savings are real - we load A ONCE instead of TWICE."

**SHARK 2:** "Implementation time?"

**INTERLEAVED:** "4-6 hours minimum. Major kernel restructuring."

---

### CONTESTANT C: MINIMAL FIX (TWO-PASS)

**MINIMAL:** "Hear me out. The kernel WORKS for single GEMM. We've verified this. So..."

*[Dramatic pause]*

**MINIMAL:** "...call it twice."

**SHARKS:** "Wait, what?"

**MINIMAL:** "Call the kernel with B1. Store result. Call again with B2. Fuse SiLU and multiply in PyTorch."

**SHARK 1:** "That's... that's cheating!"

**MINIMAL:** "Is it? Round 3 taught us: correct but slow beats wrong. This approach:
- Zero GPU kernel changes (zero risk of new bugs)
- 1 hour implementation time
- Guaranteed correctness
- Provides baseline for future optimization"

**SHARK 3:** "But performance..."

**MINIMAL:** "We're currently 20-100x off target. Getting to 2x off target is progress. We can optimize a working kernel. We CAN'T optimize a broken one."

---

## THE VOTE

**SHARK 1 (Performance Oracle):**
"Sequential: 6.2/10 - Works, but double resource usage hurts.
Interleaved: 8.5/10 - Best theoretical performance, but HIGH implementation risk.
Minimal Fix: 7.9/10 - Zero GPU changes = zero new bugs."

"After Rounds 1-3... I vote **MINIMAL FIX**."

---

**SHARK 2 (Pragmatic Engineer):**
"Sequential: Implementation time 3h, risk LOW, value 6.8/10
Interleaved: Implementation time 6h, risk HIGH, value 7.2/10
Minimal Fix: Implementation time 1h, risk ZERO, value 9.15/10"

"Implementation speed × correctness confidence = winner. **MINIMAL FIX**."

---

**SHARK 3 (ROI Maximizer):**
"Let me calculate ROI per hour of work:

Sequential: ~30% improvement / 3 hours = 10 ROI units
Interleaved: ~60% improvement / 6 hours = 10 ROI units
Minimal Fix: Correct output / 1 hour = INFINITE ROI (can't divide by zero risk)"

"The math is clear. **MINIMAL FIX**."

---

## THE RESULT

```
+=================================================================+
|                                                                 |
|    ROUND 4 WINNER: MINIMAL FIX (TWO-PASS)                       |
|                                                                 |
|    UNANIMOUS VOTE (3-0)                                         |
|                                                                 |
|    "Call the kernel twice. Fuse in Python.                      |
|     Correctness first, optimization second."                    |
|                                                                 |
+=================================================================+
```

---

## IMPLEMENTATION PLAN

### Phase 1: Adapt Input Format
The task provides: `(a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c)`
The kernel expects: `(abc_tensors, _, sfasfb_reordered_tensors, problem_sizes)`

Need to bridge this gap.

### Phase 2: Two-Pass Execution
```python
# Pass 1: GEMM1 = A @ B1
temp1 = kernel(a, b1, sfa_perm, sfb1_perm)

# Pass 2: GEMM2 = A @ B2
temp2 = kernel(a, b2, sfa_perm, sfb2_perm)

# Fuse: C = silu(GEMM1) * GEMM2
c = torch.nn.functional.silu(temp1) * temp2
```

### Phase 3: Validation
- Compare against reference implementation
- rtol=1e-03, atol=1e-03

---

## SEASON SCORECARD

| Round | Winner | Expected | Actual | Status |
|-------|--------|----------|--------|--------|
| 1 | Pipeline Stages | 1.5x faster | 30% SLOWER | FAILED |
| 2 | Tile Size Tuning | 2-3x faster | COMPILE ERROR | FAILED |
| 3 | Wild Card | ??? | Found the bug | SUCCESS |
| 4 | Minimal Fix | Correctness | TBD | IN PROGRESS |

---

## HOST'S CLOSING

**HOST:** "And there you have it, folks! After three rounds of humiliation, the sharks have learned a valuable lesson."

*[Turns to camera]*

**HOST:** "Sometimes the best optimization is making sure your code does what it's supposed to do."

*[Credits roll]*

---

```
NEXT: THE IMPLEMENTATION

Will the Two-Pass approach work?
Will validation pass?
Will we finally close the performance gap?

STAY TUNED FOR THE EXCITING CONCLUSION OF...
SHARK TANK: GPU OPTIMIZATION EDITION
```

---

*"Correct but slow beats fast but wrong. Every. Single. Time."*
*- The Sharks, Round 4*
