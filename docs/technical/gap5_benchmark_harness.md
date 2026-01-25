# Gap 5: Benchmark Harness Pitfalls

## Common Performance Killers

### 1. Hidden Copies and Casts
```python
# BAD: Hidden copy
def kernel_wrapper(a, b1, b2, ...):
    a = a.contiguous()  # COPY if not contiguous!
    return my_kernel(a, b1, b2, ...)

# GOOD: Validate upfront, fail fast
def kernel_wrapper(a, b1, b2, ...):
    assert a.is_contiguous(), "A must be contiguous"
    return my_kernel(a, b1, b2, ...)
```

### 2. Output Allocation in Timed Region
```python
# BAD: Allocating during timing
def timed_kernel(inputs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    
    c = torch.empty(M, N, L, dtype=torch.float16, device='cuda')  # ALLOCATION!
    my_kernel(inputs, c)
    
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

# GOOD: Pre-allocate
def timed_kernel(inputs, c_preallocated):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    
    my_kernel(inputs, c_preallocated)  # No allocation
    
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)
```

### 3. Hidden Synchronizations
```python
# BAD: print causes sync
def kernel_wrapper(inputs):
    result = my_kernel(inputs)
    print(f"Result shape: {result.shape}")  # SYNC!
    return result

# BAD: .item() causes sync
def kernel_wrapper(inputs):
    result = my_kernel(inputs)
    if result[0,0,0].item() < 0:  # SYNC!
        pass
    return result

# GOOD: No mid-computation syncs
def kernel_wrapper(inputs):
    return my_kernel(inputs)  # Pure kernel call
```

### 4. Per-Batch Kernel Launches
```python
# BAD: L separate kernel launches
def kernel_wrapper(a, b1, b2, c, L):
    for l in range(L):
        my_kernel(a[:,:,l], b1[:,:,l], b2[:,:,l], c[:,:,l])

# GOOD: Single kernel handles all L
def kernel_wrapper(a, b1, b2, c):
    my_batched_kernel(a, b1, b2, c)  # Handles L internally
```

### 5. Warmup Missing
```python
# BAD: First run includes JIT/cache warming
times = []
for i in range(100):
    times.append(timed_kernel(inputs, c))
avg_time = sum(times) / len(times)  # First runs are slow!

# GOOD: Explicit warmup
# Warmup (not timed)
for _ in range(10):
    my_kernel(inputs, c)
torch.cuda.synchronize()

# Timed runs
times = []
for i in range(100):
    times.append(timed_kernel(inputs, c))
avg_time = sum(times) / len(times)
```

## Correct Benchmark Template
```python
def benchmark_kernel(m, n, k, l, num_warmup=10, num_runs=100):
    # Generate inputs ONCE
    inputs = generate_input(m, n, k, l, seed=42)
    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, _ = inputs
    
    # Pre-allocate output
    c = torch.empty(m, n, l, dtype=torch.float16, device='cuda')
    
    # Validate inputs (no copies needed)
    assert a.is_contiguous() and b1.is_contiguous() and b2.is_contiguous()
    
    # Warmup
    for _ in range(num_warmup):
        my_kernel(a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm, c)
    torch.cuda.synchronize()
    
    # Timed runs with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(num_runs):
        start.record()
        my_kernel(a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm, c)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
    }
```

## Validation Gate Checklist
- [ ] No .contiguous() calls in hot path
- [ ] Output pre-allocated before timing
- [ ] No print/logging in timed region
- [ ] No .item()/.cpu()/.numpy() in timed region
- [ ] Single kernel launch for all L batches
- [ ] Warmup runs before timing
- [ ] Using CUDA events (not time.time())
- [ ] torch.cuda.synchronize() after timing
