# SHARK TANK ROUND 4: FIX THE KERNEL

---

```
  ____   ___  _   _ _   _ ____    _  _
 |  _ \ / _ \| | | | \ | |  _ \  | || |
 | |_) | | | | | | |  \| | | | | | || |_
 |  _ <| |_| | |_| | |\  | |_| | |__   _|
 |_| \_\\___/ \___/|_| \_|____/     |_|

    _____ ___ __  __   _____ ___    _____ _____  __
   |_   _|_ _|  \/  | | ____|_ _|  |  ___|_ _\ \/ /
     | |  | || |\/| | |  _|  | |   | |_   | | \  /
     | |  | || |  | | | |___ | |   |  _|  | | /  \
     |_| |___|_|  |_| |_____|___|  |_|   |___/_/\_\
```

---

## THE SITUATION

After 3 rounds of failed "optimizations," the Wild Card discovered the REAL problem:

**THE KERNEL DOESN'T IMPLEMENT THE TASK SPECIFICATION**

---

## WHAT THE TASK REQUIRES

```python
C = silu(A @ B1) * (A @ B2)
```

Input tensors:
- `a`: M x K x L (nvfp4)
- `b1`: N x K x L (nvfp4)
- `b2`: N x K x L (nvfp4)
- `sfa`, `sfb1`, `sfb2`: Scale factors (fp8)
- `c`: M x N x L output (fp16)

Algorithm:
1. Compute GEMM1 = A @ B1 (with block scaling)
2. Apply SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
3. Compute GEMM2 = A @ B2 (with block scaling)
4. Multiply: C = silu(GEMM1) * GEMM2

---

## WHAT THE KERNEL CURRENTLY DOES

```python
C = A @ B  # Just ONE gemm
```

Missing:
- Second B matrix (B2)
- Second scale factor (SFB2)
- SiLU activation
- Element-wise multiply

---

## HARDWARE CONSTRAINTS (CONFIRMED)

```python
mma_tiler_mnk = (128, 128, 256)  # CANNOT CHANGE - hardware requirement
num_ab_stage = 1                  # More stages = SLOWER (compute-bound)
```

---

## ROUND 4 GOAL

**Actually implement the dual GEMM with SiLU fusion.**

This is NOT an optimization round. This is a CORRECTNESS round.

---

## IMPLEMENTATION APPROACHES TO CONSIDER

### Approach A: Sequential Dual GEMM
- Compute GEMM1 fully, store to shared memory
- Apply SiLU
- Compute GEMM2 fully
- Multiply and store

Pros: Simple, easy to understand
Cons: May not reuse A efficiently

### Approach B: Interleaved Dual GEMM
- For each K-tile:
  - Load A tile to shared memory
  - Compute partial GEMM1 (A @ B1_tile)
  - Compute partial GEMM2 (A @ B2_tile) - reuse A!
- Fuse silu + multiply in epilogue

Pros: Reuses A tiles, better memory efficiency
Cons: Need two accumulators

### Approach C: Ping-Pong Dual GEMM
- Use warp specialization
- One consumer computes GEMM1
- Other consumer computes GEMM2
- Fuse in epilogue

Pros: Maximum parallelism
Cons: Complex synchronization

---

## SUCCESS CRITERIA

1. **Correctness**: Passes validation against reference implementation
2. **Completeness**: Actually computes `silu(A @ B1) * (A @ B2)`
3. **Performance**: Should be CLOSER to target (even if not optimal yet)

---

## CONSTRAINTS FOR CONTESTANTS

1. Must use existing NVFP4 MMA infrastructure
2. Must respect 128x128 tile size requirement
3. Must handle all benchmark shapes (M=40-384, N=3072-7168, K=1536-7168)
4. Must fuse SiLU and multiply efficiently (not separate kernels)

---

*"Round 4: Where we finally compute the right thing."*
