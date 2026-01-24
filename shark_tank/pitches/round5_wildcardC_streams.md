# WILD CARD C: STREAM-PARALLEL / PERSISTENT KERNEL APPROACH

---

```
 __        _____ _     ____     ____    _    ____  ____     ____
 \ \      / /_ _| |   |  _ \   / ___|  / \  |  _ \|  _ \   / ___|
  \ \ /\ / / | || |   | | | | | |     / _ \ | |_) | | | | | |
   \ V  V /  | || |___| |_| | | |___ / ___ \|  _ <| |_| | | |___
   \_/\_/  |___|_____|____/   \____/_/   \_\_| \_\____/   \____|

  ____ _____ ____  _____    _    __  __ ____
 / ___|_   _|  _ \| ____|  / \  |  \/  / ___|
 \___ \ | | | |_) |  _|   / _ \ | |\/| \___ \
  ___) || | |  _ <| |___ / ___ \| |  | |___) |
 |____/ |_| |_| \_\_____/_/   \_\_|  |_|____/
```

---

## THE INSIGHT: WE'RE LEAVING PARALLELISM ON THE TABLE

### Current State
- **Current kernel:** ~400-530 us (runs ALL groups SEQUENTIALLY)
- **Target:** 2-19 us
- **Gap:** 20-170x

### The Problem Nobody Addressed
The CuTe kernel launches ONE grid that processes ALL groups SEQUENTIALLY.
Each CTA delinearizes its `bidz` to find which group it belongs to.

```python
# Current approach (from submission.py lines 68-83):
for _, (cta_m, cta_n) in enumerate(cta_mn_list):
    if cta_rest >= (cta_m * cta_n):
        group_idx += 1
        cta_rest -= cta_m * cta_n
    else:
        # ... process this group
```

This means: **Groups are processed in sequence, not in parallel!**

### The B200 Reality Check
The NVIDIA B200 has:
- **192 SMs** (Streaming Multiprocessors)
- **18,432 CUDA cores**
- **4.5 TB/s HBM3e bandwidth**

For our typical problem sizes:
- Group with M=256, N=4096, K=7168: ~32 CTAs (2 x 32 tiles at 128x128)
- **8 groups = ~256 CTAs**
- B200 can easily run **multiple groups simultaneously**

---

## THE PROPOSAL: STREAM-PARALLEL GROUP GEMM

### Approach 1: Multi-Stream Execution

Instead of one kernel with sequential group processing:

```python
# CURRENT: Sequential (what we have)
compiled_kernel(all_groups_together)  # CTAs process groups one-by-one

# PROPOSED: Stream-Parallel
streams = [torch.cuda.Stream() for _ in range(num_groups)]
for i, group in enumerate(groups):
    with torch.cuda.stream(streams[i]):
        run_single_group_gemm(group)
torch.cuda.synchronize()
```

### Approach 2: CUDA Graphs for Launch Overhead Elimination

The kernel launch overhead (~5-10 us per launch) compounds when launching many kernels.
CUDA Graphs capture the execution pattern and replay it with near-zero overhead:

```python
# First call: capture the graph
with torch.cuda.graph(g):
    for i, group in enumerate(groups):
        with torch.cuda.stream(streams[i]):
            run_single_group_gemm(group)

# Subsequent calls: replay instantly
g.replay()  # Near-zero launch overhead!
```

### Approach 3: Persistent Kernel (Advanced)

Keep thread blocks alive across ALL groups:

```cuda
__global__ void persistent_group_gemm() {
    while (true) {
        // Atomically claim next work unit
        int work_id = atomicAdd(&global_work_counter, 1);
        if (work_id >= total_work_units) break;

        // Decode which group and tile this is
        int group_idx = decode_group(work_id);
        int tile_idx = decode_tile(work_id);

        // Process the tile
        compute_tile(group_idx, tile_idx);
    }
}
```

---

## THEORETICAL SPEEDUP ANALYSIS

### Current: Sequential Processing
```
Time = sum(time_per_group)
     = T1 + T2 + T3 + ... + T8
     = ~500 us (8 groups)
```

### With Stream Parallelism
```
Time = max(time_per_group) + overhead
     = max(T1, T2, ..., T8) + stream_overhead
     = ~70 us + ~10 us = ~80 us
```

**Theoretical speedup: 6-7x** (with 8 groups running in parallel)

### Why Not 8x?
1. **Memory bandwidth contention**: All groups compete for HBM3e bandwidth
2. **SM contention**: If total CTAs > 192 SMs, some serialization occurs
3. **Stream synchronization overhead**: ~5-10 us per stream
4. **CUDA Graph capture overhead**: One-time cost, amortized

---

## B200 RESOURCE ANALYSIS

### Can B200 Run 8 GEMMs in Parallel?

**Scenario: 8 groups, each M=256, N=4096, K=7168**

Per-group CTA count:
- M tiles: ceil(256 / 128) = 2
- N tiles: ceil(4096 / 128) = 32
- Total CTAs per group: 2 x 32 = 64 CTAs

Total CTAs for 8 groups: 64 x 8 = **512 CTAs**

B200 has **192 SMs**. With 128 threads per CTA:
- Each SM can run ~1-2 CTAs concurrently (depending on register pressure)
- 512 CTAs / 192 SMs = ~2.7 waves

**Conclusion: Yes, B200 CAN run multiple groups in parallel!**

Not perfectly parallel (2.7 waves instead of 1), but much better than 8 sequential waves.

### Memory Bandwidth Consideration

Per-group bandwidth requirement:
- A matrix: M x K x 0.5 bytes = 256 x 7168 x 0.5 = 0.9 MB
- B matrix: N x K x 0.5 bytes = 4096 x 7168 x 0.5 = 14.7 MB
- C matrix: M x N x 2 bytes = 256 x 4096 x 2 = 2.1 MB
- Total: ~17.7 MB per group

8 groups total: ~142 MB
HBM3e bandwidth: 4.5 TB/s
Time to load 8 groups: 142 MB / 4.5 TB/s = **0.03 ms = 30 us**

**Memory is NOT the bottleneck for parallel execution!**

---

## IMPLEMENTATION SKETCH

### Phase 1: Multi-Stream Wrapper (Low Risk)

```python
import torch
from typing import List, Tuple

class StreamParallelGroupGEMM:
    def __init__(self, max_groups: int = 16):
        self.streams = [torch.cuda.Stream() for _ in range(max_groups)]
        self.events = [torch.cuda.Event() for _ in range(max_groups)]

    def execute(self, groups: List[Tuple]) -> List[torch.Tensor]:
        results = []
        num_groups = len(groups)

        # Launch all groups in parallel
        for i, group_data in enumerate(groups):
            stream = self.streams[i % len(self.streams)]
            with torch.cuda.stream(stream):
                # Each group gets its own kernel launch
                result = run_single_group_gemm(group_data)
                results.append(result)
                self.events[i].record(stream)

        # Wait for all to complete
        for i in range(num_groups):
            self.events[i].synchronize()

        return results
```

### Phase 2: CUDA Graph Capture (Medium Risk)

```python
class GraphCapturedGroupGEMM:
    def __init__(self):
        self.graph = None
        self.graph_exec = None
        self.streams = None
        self.captured = False

    def capture_graph(self, groups: List[Tuple]):
        self.streams = [torch.cuda.Stream() for _ in range(len(groups))]

        # Warmup required before capture
        for _ in range(3):
            self._run_warmup(groups)

        # Capture the graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.streams[0]):
            self._run_parallel(groups)

        self.captured = True

    def replay(self):
        assert self.captured
        self.graph.replay()
        torch.cuda.current_stream().wait_stream(self.streams[0])
```

### Phase 3: Hybrid Approach (Best of Both)

```python
def custom_kernel(data: input_t) -> output_t:
    abc_tensors, _, sfasfb_tensors, problem_sizes = data
    num_groups = len(abc_tensors)

    # For small group counts: use stream parallelism
    if num_groups <= 8:
        return stream_parallel_execute(abc_tensors, sfasfb_tensors, problem_sizes)

    # For large group counts: use original sequential (less overhead)
    return sequential_execute(abc_tensors, sfasfb_tensors, problem_sizes)
```

---

## RISKS AND MITIGATIONS

### Risk 1: Memory Bandwidth Saturation
**Concern:** 8 parallel GEMMs might saturate HBM3e bandwidth.
**Mitigation:** Analysis shows 142 MB total, 4.5 TB/s bandwidth = 30 us load time. Not a bottleneck.

### Risk 2: SM Contention
**Concern:** Too many CTAs competing for SMs.
**Mitigation:** 512 CTAs on 192 SMs = 2.7 waves. Still ~3x better than 8 sequential waves.

### Risk 3: Stream Overhead
**Concern:** Creating/managing streams has overhead.
**Mitigation:** Pre-allocate stream pool. Use CUDA Graphs to eliminate per-call overhead.

### Risk 4: cuBLAS Uses Streams Differently
**Concern:** Our CuTe kernel might not benefit from streams like cuBLAS does.
**Mitigation:** Test empirically. Worst case: fall back to sequential.

### Risk 5: Tensor Memory (TMEM) Contention
**Concern:** TMEM is a shared resource on SM100.
**Mitigation:** Each CTA gets its own TMEM allocation. Not shared across CTAs.

---

## EXPECTED PERFORMANCE

| Metric | Sequential | Stream-Parallel | Speedup |
|--------|------------|-----------------|---------|
| 8 groups, K=7168 | ~530 us | ~80-120 us | 4-6x |
| 8 groups, K=2048 | ~508 us | ~70-100 us | 5-7x |
| 2 groups, K=4096 | ~279 us | ~150 us | ~2x |
| 2 groups, K=1536 | ~256 us | ~140 us | ~2x |

**Note:** 2-group cases see smaller benefit (less parallelism to exploit).

---

## COMPARISON WITH OTHER APPROACHES

| Approach | Expected Gain | Risk | Implementation Time |
|----------|---------------|------|---------------------|
| Pipeline Stages | -30% (FAILED) | Proven | N/A |
| Tile Tuning | N/A (IMPOSSIBLE) | Hardware | N/A |
| Warp Specialization | 10-20% | High | 6+ hours |
| TMA Epilogue | 5-10% | Medium | 3 hours |
| **Stream Parallelism** | **4-7x** | **Medium** | **2 hours** |

---

## THE PITCH

**WILD CARD C:** "The B200 is a massively parallel processor. We're treating it like a sequential one. The current kernel runs 8 groups one after another when the hardware can run them simultaneously. Stream parallelism is a PROVEN technique - it's how cuBLAS scales across problem sets. Why aren't we using it?"

**KEY INSIGHT:** "Everyone's focused on making individual GEMMs faster. I'm focused on running them IN PARALLEL. Different strategy, potentially bigger win."

**RISK/REWARD:** "Medium risk, high reward. We're not touching the kernel - we're just launching it smarter."

---

## WHY THIS COULD WORK WHEN OTHERS FAILED

1. **Pipeline Stages failed** because the kernel is compute-bound, not memory-bound.
   - Stream parallelism is ORTHOGONAL to pipeline depth.

2. **Tile sizes failed** because of hardware constraints.
   - Stream parallelism doesn't change tile sizes.

3. **We're not changing the kernel** - we're changing how we LAUNCH it.
   - Zero risk of kernel bugs.
   - Zero risk of compile errors.

---

## CONCLUSION

The stream-parallel approach attacks a DIFFERENT dimension of the problem:
- Previous optimizations tried to make individual GEMMs faster
- This optimization runs multiple GEMMs simultaneously

**Expected outcome:** 4-7x speedup for 8-group workloads.
**Implementation time:** 2 hours.
**Risk level:** Medium (well-understood technique, new to this kernel).

---

*"While everyone else is tuning the engine, I'm adding more engines."*
*- Wild Card C*

---

## APPENDIX: CODE PROTOTYPE

```python
# submission_wildcard_streams.py

import torch
from typing import List, Tuple
import functools

class StreamPool:
    """Reusable pool of CUDA streams."""
    _instance = None

    @classmethod
    def get(cls, num_streams: int = 16):
        if cls._instance is None or len(cls._instance.streams) < num_streams:
            cls._instance = cls(num_streams)
        return cls._instance

    def __init__(self, num_streams: int):
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.events = [torch.cuda.Event() for _ in range(num_streams)]


def run_group_on_stream(
    stream: torch.cuda.Stream,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    sfa_perm: torch.Tensor,
    sfb_perm: torch.Tensor,
    problem_size: Tuple[int, int, int, int]
):
    """Run a single group GEMM on a specific stream."""
    with torch.cuda.stream(stream):
        # Call the existing compiled kernel for this single group
        compiled_func = compile_kernel([problem_size])

        abc_ptrs = [(a.data_ptr(), b.data_ptr(), c.data_ptr())]
        sfasfb_ptrs = [(sfa_perm.data_ptr(), sfb_perm.data_ptr())]

        tensor_of_problem_sizes = torch.tensor([problem_size], dtype=torch.int32, device="cuda")
        tensor_of_abc_ptrs = torch.tensor(abc_ptrs, dtype=torch.int64, device="cuda")
        tensor_of_sfasfb_ptrs = torch.tensor(sfasfb_ptrs, dtype=torch.int64, device="cuda")

        # ... launch kernel (abbreviated for clarity)


def custom_kernel_stream_parallel(data) -> List[torch.Tensor]:
    """
    Stream-parallel Group GEMM implementation.

    Launches each group on a separate CUDA stream for parallel execution.
    """
    abc_tensors, _, sfasfb_tensors, problem_sizes = data
    num_groups = len(abc_tensors)

    pool = StreamPool.get(num_groups)

    # Launch all groups in parallel on separate streams
    for i, ((a, b, c), (sfa, sfb), size) in enumerate(
        zip(abc_tensors, sfasfb_tensors, problem_sizes)
    ):
        stream = pool.streams[i]
        run_group_on_stream(stream, a, b, c, sfa, sfb, size)
        pool.events[i].record(stream)

    # Wait for all to complete
    for i in range(num_groups):
        pool.events[i].synchronize()

    return [abc_tensors[i][2] for i in range(num_groups)]
```

---

## APPENDIX: CUDA GRAPH VARIANT

```python
class CUDAGraphGroupGEMM:
    """
    CUDA Graph-accelerated Group GEMM.

    Captures the multi-stream execution pattern and replays it
    with near-zero launch overhead.
    """

    def __init__(self, num_groups: int):
        self.num_groups = num_groups
        self.graph = None
        self.static_tensors = None
        self.streams = [torch.cuda.Stream() for _ in range(num_groups)]

    def capture(self, sample_data):
        """Capture the execution graph (call once with representative data)."""
        abc_tensors, _, sfasfb_tensors, problem_sizes = sample_data

        # Allocate static buffers for graph capture
        self.static_tensors = self._allocate_static(abc_tensors, sfasfb_tensors)

        # Warmup
        for _ in range(3):
            self._run_parallel(sample_data)

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.graph(self.graph, stream=s):
            self._run_parallel(sample_data)

    def replay(self, data):
        """Replay the captured graph (fast path)."""
        # Copy input data to static buffers
        self._copy_inputs(data)

        # Replay graph
        self.graph.replay()
        torch.cuda.current_stream().synchronize()

        return self._get_outputs()

    def _run_parallel(self, data):
        """Internal: run groups on parallel streams."""
        abc_tensors, _, sfasfb_tensors, problem_sizes = data
        for i in range(self.num_groups):
            with torch.cuda.stream(self.streams[i]):
                run_single_group_gemm(
                    abc_tensors[i],
                    sfasfb_tensors[i],
                    problem_sizes[i]
                )
```

---
