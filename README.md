# Shark Tank for GPUMode

AI agents compete to optimize CUDA kernels through a Shark Tank-style voting process.

## What is This?

A competitive framework where multiple Claude agents pitch kernel optimizations, and
a panel of "Shark" evaluators vote on which approach to implement. This repository
documents our attempt at the **GPUMode NVFP4 Group GEMM challenge** for NVIDIA B200.

**Target Operation:**
```
C = silu(A @ B1) * (A @ B2)
```

## Current Status

| Metric | Value |
|--------|-------|
| Round | 8 |
| Best Time | ~479 us |
| Target | ~5 us |
| Progress | Significant gap remains |

## Project Structure

```
.
├── README.md                 # This file
├── CLAUDE.md                 # Agent instructions
├── src/
│   ├── submission.py         # Current best kernel (CuTe DSL)
│   └── task.py               # Task definition
├── docs/
│   ├── game-rules/           # How Shark Tank rounds work
│   │   ├── RULES.md
│   │   ├── GAME_SHOW_FORMAT.md
│   │   ├── ROUND_TEMPLATE.md
│   │   ├── VOTING_PROTOCOL.md
│   │   ├── PITCH_REQUIREMENTS.md
│   │   └── ESCALATION_PATHS.md
│   ├── technical/            # Technical documentation
│   │   ├── build.md
│   │   ├── gap*.md           # Gap analysis reports
│   │   └── phase1*.md        # Phase 1 reports
│   ├── ARCHAEOLOGIST_REPORT.md
│   ├── ARCHIVIST_MANIFEST.md
│   └── RAG_SETUP_NEEDED.md
├── rounds/                   # Round history
│   ├── round1/ through round8/
│   │   ├── context.md
│   │   ├── results.md
│   │   └── votes/
├── pitches/                  # All pitch documents
├── scripts/                  # Utility scripts
└── archive/                  # Archived code (not deleted)
    ├── ARCHIVE_MANIFEST.md
    ├── nvfp4_group_gemm/     # Old submission versions
    ├── python/               # Experimental Python submissions
    ├── src_old/              # Old CUDA source files
    └── tests/                # Test suite
```

## Quick Start

```bash
# Run the submission (requires CUDA environment)
cd src
python submission.py

# Or import it
python -c "from src.submission import *; print('Import OK')"
```

## How Shark Tank Works

1. **Contestants** (4 AI agents) each pitch a different optimization approach
2. **Sharks** (3 AI agents) evaluate with different lenses:
   - **Skeptic**: "Prove it works"
   - **Pragmatist**: "Can we ship it?"
   - **Theorist**: "Why does this work mathematically?"
3. **Vote**: Majority decides which approach to implement
4. **Implement**: Build only the winning approach
5. **Benchmark**: Measure actual performance
6. **Repeat**: Use learnings for next round

See [docs/game-rules/](docs/game-rules/) for detailed rules.

## Key Learnings (8 Rounds)

| What Failed | Why |
|-------------|-----|
| Pipeline stages | B200 is compute-bound, not memory-bound |
| Smaller tiles | Hardware requires 128x128 minimum for NVFP4 |
| Multiple CUDA streams | GPUMode rules forbid it |
| Triton rewrite | Cannot access NVFP4 MMA instructions |

| What Worked | Why |
|-------------|-----|
| Bug discovery (Round 3) | Actually reading the task spec |
| Pre-allocated tensors | Eliminates Python overhead |
| CuTe DSL | Direct access to Blackwell instructions |

## Hardware Requirements

- NVIDIA B200 GPU (Blackwell, SM100)
- CUDA with CUTLASS support
- PyTorch with FP4 support

## Credits

- **GPUMode** - Challenge and leaderboard
- **NVIDIA** - B200 hardware and CUTLASS/CuTe DSL
- **Anthropic** - Claude Code powering the AI agents

## License

MIT
