# Gap 2: Scale Factor Layout Mapping

## Overview

This document defines the **SINGLE AUTHORITATIVE** mapping between two scale factor formats used in the nvfp4-dual-gemm project. Getting this mapping wrong causes silent numerical corruption - this is the #1 correctness landmine.

## The Two Formats

### Format 1: Reference (`to_blocked`)

The `_scaled_mm` PyTorch operation uses a blocked layout created by `to_blocked()`:

```python
def to_blocked(input_matrix):
    """
    Convert scale factors to blocked layout for _scaled_mm.
    
    Input: (rows, cols) scale factor matrix
    Output: Flattened blocked representation
    """
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    
    # Reshape into 128x4 blocks
    blocks = input_matrix.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    
    # Further rearrange within blocks: (block, 4, 32, 4) -> (block, 32, 4, 4) -> (block, 32, 16)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    
    return rearranged.flatten()
```

Key properties:
- Groups rows into 128-row blocks
- Groups columns into 4-column blocks
- Within each block: interleaves 32-row groups with 4-column groups
- Final shape is flattened for `_scaled_mm` consumption

### Format 2: Kernel Layout (Atom Layout)

The custom CUTLASS kernel expects scale factors in "atom layout":

```python
# Reference input shape: sfa_ref_cpu [M, K//16, L]
#   - M: number of rows
#   - K//16: number of scale blocks along K dimension (16 elements per scale)
#   - L: batch dimension

# Kernel expected shape: sfa_permuted [32, 4, M//128, 4, K//64, L]
#   - 32: rows within a 32-row atom
#   - 4: which 32-row group within 128-row block (4 * 32 = 128)
#   - M//128: which 128-row block
#   - 4: scale blocks within a K tile (4 * 16 = 64 elements)
#   - K//64: which K tile
#   - L: batch dimension

atom_m = (32, 4)  # 128 rows per block = 32 * 4
atom_k = 4        # 4 scale blocks per K tile = 64 elements
```

## Authoritative Index Mapping

### Forward Mapping: Reference to Permuted

```python
def ref_to_permuted_index(i: int, kb: int, l: int, M: int, K: int) -> tuple:
    """
    Map reference index (row, k_block, batch) to permuted tensor indices.
    
    Args:
        i: Row index in [0, M)
        kb: K-block index in [0, K//16)
        l: Batch index in [0, L)
        M: Total rows (must be multiple of 128)
        K: Total K dimension (must be multiple of 64)
    
    Returns:
        (mm32, mm4, mm, kk4, kk, l) - indices into permuted tensor
    """
    # Decompose row index into hierarchical components
    mm = i // 128           # Which 128-row block [0, M//128)
    mm4 = (i % 128) // 32   # Which 32-row group within 128-row block [0, 4)
    mm32 = i % 32           # Position within 32-row group [0, 32)
    
    # Decompose k_block index into hierarchical components
    kk = kb // 4            # Which 4-block group (K tile) [0, K//64)
    kk4 = kb % 4            # Position within 4-block group [0, 4)
    
    return (mm32, mm4, mm, kk4, kk, l)
```

### Inverse Mapping: Permuted to Reference

```python
def permuted_to_ref_index(mm32: int, mm4: int, mm: int, kk4: int, kk: int, l: int) -> tuple:
    """
    Map permuted tensor indices back to reference index.
    
    Args:
        mm32: Position within 32-row group [0, 32)
        mm4: Which 32-row group within 128-row block [0, 4)
        mm: Which 128-row block [0, M//128)
        kk4: Position within 4-block group [0, 4)
        kk: Which K tile [0, K//64)
        l: Batch index
    
    Returns:
        (i, kb, l) - indices into reference tensor
    """
    i = mm * 128 + mm4 * 32 + mm32
    kb = kk * 4 + kk4
    return (i, kb, l)
```

### Verification Identity

For any valid indices:
```python
sfa_permuted[mm32, mm4, mm, kk4, kk, l] == sfa_ref_cpu[i, kb, l]
```

where `(mm32, mm4, mm, kk4, kk, l) = ref_to_permuted_index(i, kb, l, M, K)`

## Permutation Function

```python
def permute_scales(sfa_ref: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """
    Permute scale factors from reference layout to kernel atom layout.
    
    Args:
        sfa_ref: Scale factors in shape [M, K//16, L]
        M: Total rows
        K: Total K dimension
    
    Returns:
        sfa_permuted: Scale factors in shape [32, 4, M//128, 4, K//64, L]
    """
    L = sfa_ref.shape[2] if sfa_ref.dim() == 3 else 1
    sf_k = K // 16  # Number of scale blocks along K
    
    # Reshape: [M, K//16, L] -> [M//128, 128, K//64, 4, L]
    #                        -> [M//128, 4, 32, K//64, 4, L]
    sfa = sfa_ref.view(M // 128, 128, K // 64, 4, L)
    sfa = sfa.view(M // 128, 4, 32, K // 64, 4, L)
    
    # Permute: [M//128, 4, 32, K//64, 4, L] -> [32, 4, M//128, 4, K//64, L]
    sfa_permuted = sfa.permute(2, 1, 0, 4, 3, 5).contiguous()
    
    return sfa_permuted
```

## Unit Test Gate

This test MUST pass before any kernel integration:

```python
import torch
import random

def test_scale_mapping():
    """
    Comprehensive test for scale factor layout mapping.
    This is a correctness gate - failure here means silent data corruption.
    """
    M, K, L = 256, 512, 2
    sf_k = K // 16  # 32 scale blocks along K
    
    # Create reference with unique values for each position
    sfa_ref = torch.arange(M * sf_k * L, dtype=torch.float32).reshape(M, sf_k, L)
    
    # Permute to kernel format
    sfa_permuted = permute_scales(sfa_ref, M, K)
    
    # Verify expected shape
    expected_shape = (32, 4, M // 128, 4, K // 64, L)
    assert sfa_permuted.shape == expected_shape, \
        f"Shape mismatch: {sfa_permuted.shape} vs {expected_shape}"
    
    # Verify random samples (statistical coverage)
    errors = []
    for _ in range(100):
        i = random.randint(0, M - 1)
        kb = random.randint(0, sf_k - 1)
        l = random.randint(0, L - 1)
        
        mm32, mm4, mm, kk4, kk, b = ref_to_permuted_index(i, kb, l, M, K)
        
        ref_val = sfa_ref[i, kb, l].item()
        perm_val = sfa_permuted[mm32, mm4, mm, kk4, kk, b].item()
        
        if ref_val != perm_val:
            errors.append(f"Mismatch at ref({i},{kb},{l}) -> perm({mm32},{mm4},{mm},{kk4},{kk},{b}): {ref_val} vs {perm_val}")
    
    assert len(errors) == 0, f"Found {len(errors)} mismatches:\n" + "\n".join(errors[:10])
    
    # Verify exhaustive for small dimensions
    M_small, K_small, L_small = 128, 64, 1
    sf_k_small = K_small // 16
    sfa_ref_small = torch.arange(M_small * sf_k_small * L_small, dtype=torch.float32).reshape(M_small, sf_k_small, L_small)
    sfa_perm_small = permute_scales(sfa_ref_small, M_small, K_small)
    
    for i in range(M_small):
        for kb in range(sf_k_small):
            for l in range(L_small):
                mm32, mm4, mm, kk4, kk, b = ref_to_permuted_index(i, kb, l, M_small, K_small)
                assert sfa_ref_small[i, kb, l] == sfa_perm_small[mm32, mm4, mm, kk4, kk, b], \
                    f"Exhaustive check failed at ({i},{kb},{l})"
    
    print("All scale mapping tests passed!")
    return True


def test_inverse_mapping():
    """Test that inverse mapping correctly recovers original indices."""
    M, K = 256, 512
    
    for _ in range(100):
        i_orig = random.randint(0, M - 1)
        kb_orig = random.randint(0, K // 16 - 1)
        l_orig = random.randint(0, 1)
        
        # Forward then inverse
        mm32, mm4, mm, kk4, kk, l = ref_to_permuted_index(i_orig, kb_orig, l_orig, M, K)
        i_rec, kb_rec, l_rec = permuted_to_ref_index(mm32, mm4, mm, kk4, kk, l)
        
        assert (i_orig, kb_orig, l_orig) == (i_rec, kb_rec, l_rec), \
            f"Inverse mapping failed: {(i_orig, kb_orig, l_orig)} != {(i_rec, kb_rec, l_rec)}"
    
    print("Inverse mapping test passed!")
    return True
```

## Common Bugs and Pitfalls

### Bug 1: Dimension Order Confusion

**Wrong:**
```python
# Swapped mm4 and mm32 positions
sfa_permuted = sfa.permute(1, 2, 0, 4, 3, 5)  # WRONG ORDER
```

**Correct:**
```python
sfa_permuted = sfa.permute(2, 1, 0, 4, 3, 5)  # 32 first, then 4
```

### Bug 2: Integer Division vs Modulo

**Wrong:**
```python
mm4 = i // 32  # Forgets the 128-block boundary
```

**Correct:**
```python
mm4 = (i % 128) // 32  # Correctly handles 128-block wrapping
```

### Bug 3: K-Block Grouping

**Wrong:**
```python
kk = kb % 4   # Swapped division and modulo
kk4 = kb // 4
```

**Correct:**
```python
kk = kb // 4   # Which 4-block group
kk4 = kb % 4   # Position within group
```

### Bug 4: Shape Assumptions

The mapping assumes:
- M is divisible by 128
- K is divisible by 256 (task.md constraint, implies divisible by 64)
- Scale block size is 16 (so K//16 scale blocks)

**Note**: The task specification requires K % 256 == 0. The atom layout groups
4 scale blocks per K tile (4 * 16 = 64 elements), but the overall K constraint
is stricter at 256.

Always validate these constraints before permutation.

## Relationship to to_blocked()

The `to_blocked()` function and atom layout serve different consumers:

| Aspect | to_blocked() | Atom Layout |
|--------|--------------|-------------|
| Consumer | `_scaled_mm` | Custom CUTLASS kernel |
| Output | Flattened 1D | 6D tensor |
| Block size | 128x4 | 128x64 (in elements) |
| Scale grouping | 32x16 | 32x4 per atom |

Both start from the same reference format `[M, K//16, L]` but produce different layouts optimized for their respective consumers.

## Integration Checklist

Before using scale factors in the kernel:

1. [ ] Reference scales are in shape `[M, K//16, L]`
2. [ ] M is divisible by 128
3. [ ] K is divisible by 256 (task constraint)
4. [ ] `test_scale_mapping()` passes
5. [ ] `test_inverse_mapping()` passes
6. [ ] Kernel output matches reference `_scaled_mm` output (end-to-end validation)

## Scale Factor Data Type

**CRITICAL**: Scale factors must use `torch.float8_e4m3fn` dtype to match the
reference implementation. See `python/constants.py` for the single source of truth.

The task specification mentioned `e4m3fnuz` but the reference implementation
uses `e4m3fn`. We use `e4m3fn` to ensure validation passes against the reference.
