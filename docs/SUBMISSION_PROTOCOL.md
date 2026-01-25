# SUBMISSION PROTOCOL

---

```
 ____  _   _ ____  __  __ ___ ____ ____ ___ ___  _   _
/ ___|| | | | __ )|  \/  |_ _/ ___/ ___|_ _/ _ \| \ | |
\___ \| | | |  _ \| |\/| || |\___ \___ \| | | | |  \| |
 ___) | |_| | |_) | |  | || | ___) |__) | | |_| | |\  |
|____/ \___/|____/|_|  |_|___|____/____/___\___/|_| \_|

 ____  ____   ___ _____ ___   ____ ___  _
|  _ \|  _ \ / _ \_   _/ _ \ / ___/ _ \| |
| |_) | |_) | | | || || | | | |  | | | | |
|  __/|  _ <| |_| || || |_| | |__| |_| | |___
|_|   |_| \_\\___/ |_| \___/ \____\___/|_____|
```

---

## THE GOLDEN RULE

> **ALWAYS update `submission.py` directly.**
>
> Do NOT create separate files like `submission_santos.py` or `submission_okonkwo.py`.

---

## Why One File?

### Competition Reality
- GPU MODE evaluates **ONE** file: `submission.py`
- Multiple files create confusion about which is "the" submission
- Judges don't care about your experiments, they care about results

### Clean Workflow
- One file = one source of truth
- Easy to benchmark: `python submission.py`
- Easy to submit: upload `submission.py`
- Easy to rollback: `git checkout submission.py`

---

## The Correct Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    SHARK TANK WORKFLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PITCH PHASE                                                 │
│     └─► Contestants propose optimizations                       │
│     └─► Sharks vote on winners                                  │
│                                                                 │
│  2. IMPLEMENTATION PHASE                                        │
│     └─► Worker fish implement the winning approach              │
│     └─► Test LOCALLY first (don't touch submission.py yet!)     │
│                                                                 │
│  3. VALIDATION PHASE                                            │
│     └─► Bubbles validates correctness                           │
│     └─► If PASS: proceed to update                              │
│     └─► If FAIL: debug, DO NOT update submission.py             │
│                                                                 │
│  4. UPDATE PHASE                                                │
│     └─► UPDATE submission.py with the winning code              │
│     └─► Archive old version if significant change               │
│     └─► Commit with clear message                               │
│                                                                 │
│  5. BENCHMARK PHASE                                             │
│     └─► Sharky benchmarks the updated submission.py             │
│     └─► Record results in round documentation                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Locations

### The ONE Submission File
```
~/projects/nvfp4_dual_gemm/python/submission.py   ← THIS IS THE ONE
```

For Shark Tank project:
```
~/projects/Shark-Tank-for-GPUMODE.COM/src/submission.py   ← OR THIS ONE
```

### Archive (for reference only)
```
~/projects/nvfp4_dual_gemm/python/archive/
├── submission_v1.py      ← Old baseline
├── submission_v2.py      ← Previous attempt
└── ...
```

---

## What NOT To Do

### ❌ WRONG: Creating multiple submission files
```
submission_santos.py      ← NO!
submission_okonkwo.py     ← NO!
submission_combined.py    ← NO!
submission_experiment.py  ← NO!
```

### ✅ RIGHT: One file, updated in place
```
submission.py             ← YES! Always this one.
```

---

## Version Control Strategy

### Before Major Changes
```bash
# Archive the current version
cp submission.py archive/submission_v$(date +%Y%m%d).py

# Then make changes to submission.py
```

### After Changes
```bash
# Commit with clear message
git add submission.py
git commit -m "Update submission.py: [describe the optimization]"
```

### Rollback if Needed
```bash
# Restore from git
git checkout HEAD~1 -- submission.py

# Or restore from archive
cp archive/submission_v20260125.py submission.py
```

---

## Worker Fish Protocol Update

When implementing optimizations:

1. **Finn/Coral**: Implement in a LOCAL test file first
2. **Bubbles**: Validate the test file
3. **If valid**: Merge changes INTO `submission.py`
4. **Sharky**: Benchmark the updated `submission.py`

The test file is TEMPORARY. The submission file is PERMANENT.

---

## Summary

| Question | Answer |
|----------|--------|
| How many submission files? | **ONE** |
| Which file to update? | `submission.py` |
| Where to experiment? | Local test files, then merge |
| Where to archive old versions? | `archive/` directory |
| What gets submitted to competition? | `submission.py` |

---

*"One file to rule them all, one file to find them, one file to bring them all, and in the benchmarks bind them."*

— The Submission Protocol

---

**Added to RAG Brain**: Memory ID `ee4abd28-4f18-48a7-9666-41438544e1f9`
