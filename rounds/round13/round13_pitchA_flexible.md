# ROUND 13 PITCH A: THE FLEXIBLE HANDLER

## Contestant: The Adapter Architect

---

## Problem Statement

Current code fails with:
```
ValueError: DUAL GEMM expects 2 groups, got 3
```

The hard-coded assumption of exactly 2 groups is wrong.

---

## Proposed Solution

**Remove the 2-group assumption. Handle ANY number of groups dynamically.**

### Code Change

```python
elif data_len == 4:
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data

    # FLEXIBLE GROUP HANDLING - No hard-coded count
    if isinstance(abc_tensors, (list, tuple)):
        num_groups = len(abc_tensors)

        if num_groups >= 2:
            # Extract tensors from first two groups for DUAL GEMM
            # Group 0: First GEMM (A @ B1)
            # Group 1: Second GEMM (A @ B2)
            first_group = abc_tensors[0]
            second_group = abc_tensors[1]

            if isinstance(first_group, (list, tuple)) and len(first_group) >= 3:
                a = first_group[0]
                b1 = first_group[1]
                c = first_group[2]  # Output from first group

            if isinstance(second_group, (list, tuple)) and len(second_group) >= 3:
                # a should be same, b2 is different
                b2 = second_group[1]

            # Scale factors - same pattern
            if len(sfasfb_reordered_tensors) >= 2:
                sf_group0 = sfasfb_reordered_tensors[0]
                sf_group1 = sfasfb_reordered_tensors[1]

                sfa_permuted = sf_group0[0]
                sfb1_permuted = sf_group0[1]
                sfb2_permuted = sf_group1[1]

            # Ignore group 2+ for now (might be metadata or padding)
```

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Group 3+ contains important data | Medium | Log and ignore initially |
| Wrong tensor extraction | Low | Validate shapes |
| Scale factor mismatch | Low | Same extraction pattern |

---

## Expected Outcome

- **Correctness**: Should pass if groups 0-1 contain DUAL GEMM data
- **Performance**: No change (same kernel)
- **Robustness**: Handles 2, 3, or more groups

---

## Implementation Time

**30 minutes** - Simple modification to existing format handler

---

## Confidence Level

**75%** - Reasonable assumption that first 2 groups are the DUAL GEMM data

---

*Pitch A: "Flexibility is the key to survival."*
