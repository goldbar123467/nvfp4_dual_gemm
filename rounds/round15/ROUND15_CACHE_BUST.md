# ROUND 15: THE CACHE BUST

```
 ____   ___  _   _ _   _ ____    _ ____
|  _ \ / _ \| | | | \ | |  _ \  / | ___|
| |_) | | | | | | |  \| | | | | | |___ \
|  _ <| |_| | |_| | |\  | |_| | | |___) |
|_| \_\\___/ \___/|_| \_|____/  |_|____/

        COMPLETE REWRITE TO BUST CACHE
```

---

## THE PROBLEM

Production kept returning:
```
ModuleNotFoundError: No module named 'common.scale_helpers'
```

Even though our local file had NO such import! Production was caching old versions.

---

## THE SOLUTION: COMPLETE REWRITE

1. **Updated VERSION** to unique timestamp: `v15-group-gemm-fallback-20260125-184900`

2. **Completely rewrote** GROUP GEMM handler from scratch

3. **Changed all debug markers** from `[R14]` to `[R15]`

4. **Zero external imports** - only uses `torch._scaled_mm` (built-in)

---

## THE NEW GROUP GEMM HANDLER

```python
# ROUND 15: GROUP GEMM FALLBACK - NO EXTERNAL IMPORTS
for grp_idx in range(num_groups):
    grp_abc = abc_tensors[grp_idx]
    grp_sf = sfasfb_reordered_tensors[grp_idx]

    a_grp, b_grp, c_grp = grp_abc[0], grp_abc[1], grp_abc[2]
    sfa_grp, sfb_grp = grp_sf[0], grp_sf[1]

    for batch_idx in range(batch_dim):
        # Slice and run scaled GEMM
        gemm_result = torch._scaled_mm(
            a_slice, b_slice.T, sfa_slice, sfb_slice,
            out_dtype=torch.float32
        )
        c_grp[:, :, batch_idx] = gemm_result.to(torch.float16)

return group_outputs[-1]
```

---

## COMMIT

```
5a1156f - ROUND 15: Complete rewrite of GROUP GEMM handler [CACHE BUST 20260125-185200]
```

---

## EXPECTED BEHAVIOR

When you see `[R15]` in the debug output, you know you're running the new code:
```
[R15] GROUP_GEMM: 2 independent problems
[R15] Grp0: a=torch.Size([96, 64, 1]) b=torch.Size([128, 64, 1]) c=torch.Size([96, 128, 1]) L=1
[R15] Grp1: a=torch.Size([128, 256, 1]) b=torch.Size([256, 256, 1]) c=torch.Size([128, 256, 1]) L=1
[R15] GROUP_GEMM complete, returning group 1 output
```

If you still see `[R14]` or the import error, production is STILL caching.

---

*"When in doubt, bust the cache."*
