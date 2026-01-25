# SHARK TANK ROUND 6: THE PERFORMANCE HUNT - RESULTS

---

```
 _   _ _   _    _    _   _ ___ __  __  ___  _   _ ____
| | | | \ | |  / \  | \ | |_ _|  \/  |/ _ \| | | / ___|
| | | |  \| | / _ \ |  \| || || |\/| | | | | | | \___ \
| |_| | |\  |/ ___ \| |\  || || |  | | |_| | |_| |___) |
 \___/|_| \_/_/   \_\_| \_|___|_|  |_|\___/ \___/|____/

__     _____  _____ _____
\ \   / / _ \|_   _| ____|
 \ \ / / | | | | | |  _|
  \ V /| |_| | | | | |___
   \_/  \___/  |_| |_____|
```

---

## THE PITCHES

### CONTESTANT 1: PERSISTENT KERNEL
*"Keep thread blocks alive across groups"*

**The Promise:** 1.4x speedup (479µs → 340µs)

**Why it lost:**
- The kernel ALREADY uses single launch with Z-dim blocking
- Atomic work queues add complexity, not speed
- 1.4x is nowhere near the 25x we need

**Scores:** Skeptic: 3/10 | Pragmatist: 7/10 | Theorist: 3/10

---

### CONTESTANT 2: WARP SPECIALIZATION
*"Producer/consumer architecture"*

**The Promise:** 2.5-4x speedup (479µs → 120-192µs)

**Why it lost:**
- Earlier 3-stage pipeline was SLOWER (compute-bound, not memory-bound)
- Fixes per-tile overlap, not per-group overhead
- The bottleneck isn't TMA latency

**Scores:** Skeptic: 6/10 | Pragmatist: 5/10 | Theorist: 4/10

---

### CONTESTANT 3: SPLIT-K PARALLELISM
*"More CTAs = better occupancy"*

**The Promise:** 3-4x speedup from better SM utilization

**Why it lost:**
- Reduction overhead could eat the gains
- Low occupancy isn't the bottleneck (60µs is plenty of compute time)
- Adds complexity with workspace + reduction kernel

**Scores:** Skeptic: 4/10 | Pragmatist: 6/10 | Theorist: 5/10

---

### CONTESTANT 4: REDUCE LAUNCH OVERHEAD
*"Python is the bottleneck, not CUDA"*

**The Promise:** 6-19x speedup by eliminating Python overhead

**Why it WON:**
- Found the SMOKING GUN: torch.tensor() calls in hot path
- 3 tensor creations × 10-20µs = 30-60µs overhead PER CALL
- Doesn't touch the kernel (zero risk)
- Quick to implement (4-6 hours)

**Scores:** Skeptic: 8/10 | Pragmatist: 8/10 | Theorist: 8/10

---

## THE VOTE

```
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  ROUND 6 WINNER: REDUCE LAUNCH OVERHEAD                        ║
║                                                                ║
║  UNANIMOUS VOTE (3-0)                                          ║
║                                                                ║
║  Shark 1 (Skeptic):    "Finally attacking the actual          ║
║                         bottleneck. Python garbage, not CUDA." ║
║                                                                ║
║  Shark 2 (Pragmatist): "4-6 hours, low risk, 6-19x upside.    ║
║                         This is where the ROI lives."          ║
║                                                                ║
║  Shark 3 (Theorist):   "The math checks out: 60µs/group with  ║
║                         2-5µs compute = 50µs Python overhead." ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

## THE SMOKING GUN

Found in `submission.py` lines 607-620:

```python
# EVERY CALL creates NEW tensors:
tensor_of_problem_sizes = torch.tensor(problem_sizes, ...)  # ~15µs
tensor_of_abc_ptrs = torch.tensor(abc_ptrs, ...)            # ~15µs
tensor_of_sfasfb_ptrs = torch.tensor(sfasfb_ptrs, ...)      # ~15µs
tensor_of_tensormap = torch.empty(tensormap_shape, ...)      # ~5µs

# Total: ~50µs of Python overhead PER KERNEL CALL
```

---

## THE FIX: PRE-ALLOCATION

### Step 1: Cache metadata tensors by shape
```python
_tensor_cache = {}

def get_cached_tensors(num_groups, total_clusters):
    key = (num_groups, total_clusters)
    if key not in _tensor_cache:
        _tensor_cache[key] = {
            'problem_sizes': torch.empty((num_groups, 4), dtype=torch.int32, device="cuda"),
            'abc_ptrs': torch.empty((num_groups, 3), dtype=torch.int64, device="cuda"),
            'sfasfb_ptrs': torch.empty((num_groups, 2), dtype=torch.int64, device="cuda"),
            'tensormap': torch.empty((total_clusters, 4, 16), dtype=torch.int64, device="cuda"),
        }
    return _tensor_cache[key]
```

### Step 2: Use copy_ instead of new tensor
```python
cached = get_cached_tensors(num_groups, total_clusters)
cached['problem_sizes'].copy_(torch.tensor(problem_sizes, device="cpu").cuda())
# Or better: use pinned memory + async copy
```

### Step 3: Pre-compute CuTe pointers
```python
# Cache make_ptr() objects, update addresses only
```

---

## EXPECTED IMPACT

| Benchmark | Current | Expected | Speedup |
|-----------|---------|----------|---------|
| g=8, K=7168 | 479 µs | 50-80 µs | 6-10x |
| g=8, K=2048 | 451 µs | 45-75 µs | 6-10x |
| g=2, K=4096 | 247 µs | 30-50 µs | 5-8x |
| g=2, K=1536 | 218 µs | 25-45 µs | 5-9x |

---

## SEASON SCORECARD

| Round | Winner | Expected | Actual | Status |
|-------|--------|----------|--------|--------|
| 1 | Pipeline Stages | 1.5x faster | 30% SLOWER | FAILED |
| 2 | Tile Size Tuning | 2-3x faster | COMPILE ERROR | FAILED |
| 3 | Wild Card | ??? | Found the bug | SUCCESS |
| 4 | Minimal Fix | Correctness | Fixed kernel | SUCCESS |
| 5 | Stream Parallelism | 4-7x faster | NOT ALLOWED | BLOCKED |
| 6 | **Reduce Overhead** | **6-19x faster** | **TBD** | **IN PROGRESS** |

---

*"The fastest code is the code that doesn't run."*
*- Contestant 4, explaining why eliminating Python overhead beats optimizing CUDA*

