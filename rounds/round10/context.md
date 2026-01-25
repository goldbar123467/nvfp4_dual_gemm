# SHARK TANK SEASON 2, ROUND 10: SURVIVAL OF THE FITTEST

---

```
🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
🚨                                                            🚨
🚨   _____ ___ __  __ _____ ___  _   _ _____                  🚨
🚨  |_   _|_ _|  \/  | ____/ _ \| | | |_   _|                 🚨
🚨    | |  | || |\/| |  _|| | | | | | | | |                   🚨
🚨    | |  | || |  | | |__| |_| | |_| | | |                   🚨
🚨    |_| |___|_|  |_|_____\___/ \___/  |_|                   🚨
🚨                                                            🚨
🚨   EXIT CODE 114 - PROCESS TIMED OUT AT 180 SECONDS         🚨
🚨                                                            🚨
🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
```

---

## 🎬 GAMESHOW HOST EMERGENCY BROADCAST

*Claude "The Kernel Whisperer" Code bursts onto stage*

"LADIES AND GENTLEMEN, WE HAVE A CODE RED! And I don't mean the color of our error messages!

Our beautiful Round 9 submissions? TIMED OUT. 180 seconds of NOTHING. The tensor cores are ASLEEP. The memory controller is playing SOLITAIRE.

This is no longer about optimization, folks. This is about SURVIVAL.

**THE STAKES**: The contestant who makes it RUN gets their profile SAVED. The others? DELETED at the end of Season 2. That's right - this is ELIMINATION ROUND!"

---

## THE FAILURE

```json
{
  "stderr": "",
  "success": false,
  "exit_code": 114,
  "exit_code_info": "Process was shut down because it timed out.",
  "duration": 180
}
```

### What This Means

| Symptom | Interpretation |
|---------|----------------|
| `exit_code: 114` | Timeout signal |
| `duration: 180` | Hit the 3-minute wall |
| `stderr: ""` | No error message - silent death |
| `success: false` | Complete failure |

### Likely Causes

1. **torch.compile hanging** - JIT compilation can take FOREVER on first run
2. **CUDA Graph capture deadlock** - Graph capture with compile can freeze
3. **Infinite warmup loop** - Our extended warmup (10 iterations) might be stuck
4. **Memory allocation failure** - Silent OOM can cause hangs
5. **Scale factor computation explosion** - The permutation math might be wrong

---

## ROUND 10 RULES: SURVIVAL MODE

### The Challenge

Each contestant must:
1. **DIAGNOSE** what caused the timeout
2. **RESEARCH** a fix
3. **PROPOSE** a solution that MAKES IT RUN

### The Stakes

| Outcome | Consequence |
|---------|-------------|
| ✅ Makes it run | Profile SAVED for Season 2 |
| ❌ Fails to run | Profile DELETED at season end |

### Success Criteria

```
success: true
exit_code: 0
duration: < 180 seconds (preferably < 60)
```

That's it. We don't care about 13μs right now. We care about NOT TIMING OUT.

---

## CONTESTANTS: YOUR MISSION

Each contestant must investigate ONE hypothesis and propose a fix.

### Dr. Chen (PhD Candidate) - INVESTIGATE: torch.compile

**Hypothesis**: `@torch.compile` is hanging during JIT compilation

**Research Tasks**:
- How long does torch.compile take on first run?
- Does mode="max-autotune" cause longer compilation?
- Can we use mode="reduce-overhead" instead?
- Should we remove torch.compile entirely?

---

### Dr. Santos (Postdoc) - INVESTIGATE: CUDA Graph + Compile Interaction

**Hypothesis**: CUDA Graph capture doesn't work with torch.compile

**Research Tasks**:
- Can you capture a torch.compiled function in a CUDA graph?
- Is there a known incompatibility?
- Should we use graph OR compile, not both?
- What does PyTorch documentation say?

---

### Dr. Kim (Lab Manager) - INVESTIGATE: Warmup Loop

**Hypothesis**: The warmup loop is taking too long or hanging

**Research Tasks**:
- How long should warmup take?
- Is 10 iterations too many for first run?
- Is torch.cuda.synchronize() blocking forever?
- Should we add timeout protection?

---

### Dr. Okonkwo (Visiting Researcher) - INVESTIGATE: Memory/Scale Factors

**Hypothesis**: Memory allocation or scale factor computation is failing silently

**Research Tasks**:
- Are we allocating too much memory?
- Is the scale factor permutation correct?
- Could .clone().contiguous() be hanging?
- Is there an OOM that doesn't raise an exception?

---

## CONSTRAINTS

1. **Must produce running code** - Not optimized, just RUNNING
2. **Cannot exceed 60 seconds** on first run (including compilation)
3. **Must pass correctness** - rtol=1e-3, atol=1e-3
4. **Must update submission.py directly** - Per protocol!

---

## BASELINE TO BEAT

We need to check: **Does the ORIGINAL submission_best.py even run?**

If baseline runs, the problem is our changes.
If baseline times out too, the problem is environmental.

---

## DIAGNOSTIC CHECKLIST

Before proposing fixes, verify:

- [ ] Does the baseline submission.py run?
- [ ] Does a minimal scaled_mm call work?
- [ ] Does CUDA Graph capture work without compile?
- [ ] Does torch.compile work without CUDA Graph?
- [ ] How long does warmup actually take?

---

*"It's not about how fast you go. It's about whether you go at all."*

**— Round 10: MAKE IT RUN**
