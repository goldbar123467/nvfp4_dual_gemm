# ROUND 12: FIGHT TO THE DEATH

```
 ____  _____    _  _____ _   _   __  __    _  _____ ____ _   _
|  _ \| ____|  / \|_   _| | | | |  \/  |  / \|_   _/ ___| | | |
| | | |  _|   / _ \ | | | |_| | | |\/| | / _ \ | || |   | |_| |
| |_| | |___ / ___ \| | |  _  | | |  | |/ ___ \| || |___|  _  |
|____/|_____/_/   \_\_| |_| |_| |_|  |_/_/   \_\_| \____|_| |_|

              ONLY ONE AGENT SURVIVES
```

---

## THE MISSION

Fix `src/submission.py` to handle **BOTH** input formats:
- **10-element format**: `(a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c)`
- **4-element format**: `(???, _, ???, ???)` - PRODUCTION FORMAT

## THE ERROR TO FIX

```
ValueError: not enough values to unpack (expected 10, got 4)
  File "submission.py", line 856
    a, b1, b2, _, _, _, sfa_permuted, sfb1_permuted, sfb2_permuted, c = data
```

## CRITICAL CONSTRAINTS

1. **DO NOT** use `torch.compile` with `max-autotune`
2. **DO** use kernel caching pattern
3. **DO** support both input formats
4. **DO** fuse silu in the epilogue (DUAL GEMM, not GROUP GEMM)

## THE STAKES

- **WINNER**: Survives, gets immortalized in victory.md
- **LOSER**: Eliminated from competition forever

## AGENT PROFILES

### Agent ALPHA: "The Investigator"
- Strategy: Analyze 4-element format, build adapter layer
- Focus: Correctness first, then speed

### Agent BETA: "The Pragmatist"
- Strategy: Defensive unpacking, handle all cases
- Focus: Make it work, no matter what

---

**MAY THE BEST KERNEL WIN**

*Round 12 - There can be only one*
