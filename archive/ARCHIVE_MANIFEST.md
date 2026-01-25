# Archive Manifest

This directory contains archived code and documentation that is no longer actively used
but preserved for historical reference and potential future use.

## Directory Structure

```
archive/
├── ARCHIVE_MANIFEST.md    # This file
├── nvfp4_group_gemm/      # CUTLASS-based submission versions
├── python/                # PyTorch/Triton experimental submissions
├── src_old/               # Old CUDA source files
└── tests/                 # Validation test suite
```

## Contents

### nvfp4_group_gemm/

Historical versions of the CuTe DSL kernel submissions.

| File | Description | Status |
|------|-------------|--------|
| `submission_v6_clean.py` | v6 with code cleanup | Superseded by v8 |
| `submission_v7_final.py` | v7 with basic pipelining | Superseded by v8 |
| `submission_v8_prealloc.py` | v8 preallocation attempt | Superseded by v8b-fix2 |
| `submission_v8b_fix2.py` | Current best (copied to src/) | **Active in src/** |
| `RESEARCH.md` | Research notes on FP4 GEMM optimization | Reference only |

**Why archived**: These versions show the evolution of the kernel but are superseded by
the current submission in `/src/submission.py`.

**Potentially useful**: Yes - for understanding optimization history and reverting if needed.

### python/

Experimental Python/Triton approaches before switching to CuTe DSL.

| File | Description | Status |
|------|-------------|--------|
| `submission.py` | Original baseline | Superseded |
| `submission_triton.py` | Triton implementation attempt | Did not meet targets |
| `submission_cuda.py` | Inline CUDA attempt | Complexity issues |
| `submission_compile.py` | torch.compile experiment | Minor improvements |
| `submission_fused.py` | Fused operations | Memory issues |
| `submission_best_v*.py` | Various optimization attempts | Superseded |
| `task.py` | Task definition (required for imports) | **Still needed** |
| `constants.py` | Shared constants | Reference only |
| `utils.py` | Utility functions | Reference only |
| `test_kernel.py` | Python test harness | Reference only |

**Why archived**: Pure Python/Triton approaches hit ~479us wall, CuTe DSL required for
sub-5us targets.

**Potentially useful**: Yes - `task.py` is still needed for imports. The Triton code
could inform future hybrid approaches.

### src_old/

Legacy CUDA source files from early development.

| File | Description | Status |
|------|-------------|--------|
| `nvfp4_dual_gemm.cpp` | C++ kernel wrapper | Replaced by CuTe DSL |
| `setup.py` | Build configuration | No longer used |
| `silu_mul_kernel.cu` | Custom CUDA SiLU kernel | Replaced by CuTe DSL |
| `test_kernel.py` | C++ test harness | Replaced |
| `epilogue/` | Epilogue CUDA code | Replaced by CuTe DSL |
| `kernel/` | Kernel CUDA code | Replaced by CuTe DSL |

**Why archived**: The project pivoted from hand-written CUDA to CuTe DSL for better
maintainability and Blackwell compatibility.

**Potentially useful**: Maybe - shows low-level CUDA patterns if CuTe DSL hits limits.

### tests/

Validation test suite for correctness checking.

| File | Description | Status |
|------|-------------|--------|
| `run_all_tests.py` | Test runner | Still useful |
| `test_step1_dtype_consistency.py` | Data type tests | Still useful |
| `test_step2_fp4_nibble_order.py` | FP4 packing tests | Still useful |
| `test_step3_flops_and_contiguity.py` | Performance tests | Still useful |

**Why archived**: Tests were written for earlier architecture but principles still apply.

**Potentially useful**: Yes - tests should be adapted for current submission.

## Notes

- **DO NOT DELETE** - This archive preserves the project's evolution
- Files may have import dependencies on `task.py` from python/
- Some code may have GPU-specific requirements (B200/SM100)
- Historical commits in git provide even more detailed history

## Restoration

To restore any file:
```bash
cp archive/<path>/<file> <destination>
```

Example restoring v7 for comparison:
```bash
cp archive/nvfp4_group_gemm/submission_v7_final.py test_v7.py
```
