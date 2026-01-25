# SHARK TANK PITCH: REDUCE LAUNCH OVERHEAD

## The Problem: 25x Performance Gap

**Current State:** 479us for 8 groups (target: 18.8us)

The math does not add up. At ~60us per group, we are spending 12-20x more time on overhead than actual GEMM compute (which should be 2-5us).

---

## 1. THE CORE INSIGHT: Python is the Bottleneck, Not CUDA

Looking at `custom_kernel()` in `/home/clark/nvfp4_dual_gemm_repo/nvfp4_group_gemm/submission.py`:

```python
def custom_kernel(data: input_t) -> output_t:
    # OVERHEAD #1: Python dispatch and data unpacking
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data

    # OVERHEAD #2: Compile check on EVERY call (dict lookup + tuple creation)
    compiled_func = compile_kernel(problem_sizes)  # Lines 597, 532

    # OVERHEAD #3: Creating Python lists on GPU hot path
    abc_ptrs = []
    sfasfb_ptrs = []
    for i, ((a, b, c), ...) in enumerate(zip(...)):  # Lines 601-607
        abc_ptrs.append((a.data_ptr(), b.data_ptr(), c.data_ptr()))

    # OVERHEAD #4: Tensor creation on EVERY kernel call
    tensor_of_problem_sizes = torch.tensor(problem_sizes, dtype=torch.int32, device="cuda")  # Line 609
    tensor_of_abc_ptrs = torch.tensor(abc_ptrs, dtype=torch.int64, device="cuda")           # Line 610
    tensor_of_sfasfb_ptrs = torch.tensor(sfasfb_ptrs, dtype=torch.int64, device="cuda")     # Line 611
    tensor_of_tensormap = torch.empty(tensormap_shape, dtype=torch.int64, device="cuda")    # Line 622

    # OVERHEAD #5: make_ptr() calls create Python objects
    cute_ptr_of_tensor_of_abc_ptrs = make_ptr(...)  # 4x per call (Lines 624-631)
```

**Estimated Overhead Per Call:**
- List/tuple creation: ~5us
- torch.tensor() to GPU (3x): ~15us each = 45us
- torch.empty(): ~5us
- make_ptr() (4x): ~10us
- Python interpreter overhead: ~10us
- **Total Python overhead: ~75us per call**

For 8 groups: **600us of pure Python overhead** (more than measured!)

---

## 2. PROFILING STRATEGY

### Phase 1: Isolate Python vs CUDA overhead (5 min)

```python
import time
import torch

def profile_kernel_overhead(data):
    torch.cuda.synchronize()

    # Measure Python prep time (no GPU work)
    t0 = time.perf_counter()
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data
    compiled_func = compile_kernel(problem_sizes)
    abc_ptrs = [(t[0].data_ptr(), t[1].data_ptr(), t[2].data_ptr())
                for t in abc_tensors]
    # ... rest of prep
    t1 = time.perf_counter()
    python_prep_us = (t1 - t0) * 1e6

    # Measure tensor creation time
    t2 = time.perf_counter()
    tensor_of_problem_sizes = torch.tensor(problem_sizes, dtype=torch.int32, device="cuda")
    tensor_of_abc_ptrs = torch.tensor(abc_ptrs, dtype=torch.int64, device="cuda")
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    tensor_create_us = (t3 - t2) * 1e6

    print(f"Python prep: {python_prep_us:.1f}us")
    print(f"Tensor creation: {tensor_create_us:.1f}us")
```

### Phase 2: Use torch.profiler with stacks

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    for _ in range(10):
        custom_kernel(data)

# Look for "aten::empty", "aten::to" calls
print(prof.key_averages().table(sort_by="cpu_time_total"))
```

---

## 3. SPECIFIC CODE CHANGES

### Change A: Pre-allocate Metadata Tensors (HIGH IMPACT)

**Current:** Creates 4 tensors per call
**Fix:** Allocate once, reuse via cache

```python
# Global cache for pre-allocated tensors
_metadata_cache = {}

def _get_metadata_tensors(num_groups, total_clusters, device="cuda"):
    key = (num_groups, total_clusters)
    if key not in _metadata_cache:
        _metadata_cache[key] = {
            'problem_sizes': torch.empty((num_groups, 4), dtype=torch.int32, device=device),
            'abc_ptrs': torch.empty((num_groups, 3), dtype=torch.int64, device=device),
            'sfasfb_ptrs': torch.empty((num_groups, 2), dtype=torch.int64, device=device),
            'tensormap': torch.empty((total_clusters, 4, 16), dtype=torch.int64, device=device),
        }
    return _metadata_cache[key]

# In hot path: just copy data to pre-allocated tensors
cache = _get_metadata_tensors(num_groups, total_clusters)
cache['problem_sizes'].copy_(torch.tensor(problem_sizes, dtype=torch.int32))  # CPU->GPU copy
```

**Savings: ~50us per call**

---

### Change B: Pre-compute CuTe Pointers (MEDIUM IMPACT)

**Current:** Creates 4 make_ptr() objects per call
**Fix:** Cache the raw pointers, update only data_ptr values

```python
# Cache make_ptr shell objects
_ptr_cache = {}

def _get_cached_ptrs():
    if 'ptrs' not in _ptr_cache:
        _ptr_cache['ptrs'] = {
            'abc': make_ptr(cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16),
            'sfasfb': make_ptr(cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16),
            'sizes': make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
            'tensormap': make_ptr(cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16),
        }
    return _ptr_cache['ptrs']
```

**Savings: ~10us per call**

---

### Change C: Eliminate Python Loops in Hot Path (HIGH IMPACT)

**Current:** Python loops with append()
```python
abc_ptrs = []
for i, ((a, b, c), ...) in enumerate(zip(...)):
    abc_ptrs.append((a.data_ptr(), b.data_ptr(), c.data_ptr()))
```

**Fix:** Vectorized tensor operations
```python
# Pre-extract pointers using torch stack
abc_ptrs_tensor = torch.tensor(
    [[t[0].data_ptr(), t[1].data_ptr(), t[2].data_ptr()] for t in abc_tensors],
    dtype=torch.int64
)
# Even better: keep pointers in a pre-allocated tensor and update in-place
```

**Savings: ~15us per call**

---

### Change D: CUDA Graphs for Entire Pipeline (ULTIMATE FIX)

The `submission_best.py` already attempts this but incorrectly. The proper approach:

```python
class GroupGEMMGraphCache:
    def __init__(self):
        self.graphs = {}  # Key: (num_groups, total_clusters)
        self.static_buffers = {}

    def capture_graph(self, data, problem_sizes):
        key = self._make_key(problem_sizes)

        # Allocate static buffers
        self.static_buffers[key] = self._alloc_static_buffers(problem_sizes)

        # Warmup
        self._run_warmup(data, problem_sizes)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_kernel(data, problem_sizes, self.static_buffers[key])

        self.graphs[key] = graph
        return graph

    def replay(self, key, data):
        # Update input pointers in static buffers
        self._update_pointers(self.static_buffers[key], data)
        # Replay graph
        self.graphs[key].replay()
```

**Savings: ~90% of Python overhead = ~540us for 8 groups**

---

## 4. EXPECTED SPEEDUP

| Optimization | Per-Call Savings | 8-Group Total |
|-------------|------------------|---------------|
| Pre-alloc metadata tensors | 50us | 400us |
| Cache CuTe pointers | 10us | 80us |
| Vectorize pointer extraction | 15us | 120us |
| CUDA Graphs (if applicable) | 90% of remaining | ~450us |

**Conservative Estimate (A+B+C only):** 479us -> 79us (6x speedup)
**Aggressive Estimate (with CUDA Graphs):** 479us -> 25us (19x speedup)

Target is 18.8us. With these changes we can reach **25-30us**, putting us within striking distance.

---

## 5. WHY THE SHARKS SHOULD INVEST

### The ROI is Clear

1. **Low Risk:** Pure Python changes, no CUDA kernel modifications required
2. **High Reward:** 6-19x speedup from eliminating overhead
3. **Quick Implementation:** 2-4 hours of focused work
4. **Reusable Pattern:** Same overhead exists in every CUTLASS Python binding

### This is the Critical Path

The kernel itself is already optimized. The NVFP4 MMA is running. TMA loads are pipelined. We have hit the wall, and that wall is **Python**.

### The Competition is Leaving Money on the Table

Looking at the leaderboard (13.304us top), others have likely hit the same wall. Whoever solves the Python overhead problem first wins.

---

## IMPLEMENTATION PRIORITY

1. **Day 1 (2 hrs):** Implement profiling harness, confirm overhead hypothesis
2. **Day 1 (2 hrs):** Pre-allocate metadata tensors (Change A)
3. **Day 2 (2 hrs):** Vectorize pointer extraction + cache CuTe ptrs (Changes B+C)
4. **Day 2 (4 hrs):** CUDA Graph implementation (Change D)

**Total Investment:** 10 engineering hours
**Expected Return:** 6-19x performance improvement
**ROI:** >100x (if this gets us to leaderboard top)

---

## APPENDIX: Code Locations

- Main kernel: `/home/clark/nvfp4_dual_gemm_repo/nvfp4_group_gemm/submission.py`
- Compile function: Lines 503-527
- Hot path with overhead: Lines 576-646
- Metadata tensor creation: Lines 609-622
- make_ptr calls: Lines 624-631

---

*"We're not optimizing the kernel. We're optimizing around the kernel."*
