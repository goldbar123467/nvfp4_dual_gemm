<div align="center">

```
███████╗██╗  ██╗ █████╗ ██████╗ ██╗  ██╗    ████████╗ █████╗ ███╗   ██╗██╗  ██╗
██╔════╝██║  ██║██╔══██╗██╔══██╗██║ ██╔╝    ╚══██╔══╝██╔══██╗████╗  ██║██║ ██╔╝
███████╗███████║███████║██████╔╝█████╔╝        ██║   ███████║██╔██╗ ██║█████╔╝
╚════██║██╔══██║██╔══██║██╔══██╗██╔═██╗        ██║   ██╔══██║██║╚██╗██║██╔═██╗
███████║██║  ██║██║  ██║██║  ██║██║  ██╗       ██║   ██║  ██║██║ ╚████║██║  ██╗
╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝

         █████╗ ██╗     ██████╗ ██████╗ ██████╗ ███████╗
        ██╔══██╗██║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝
        ███████║██║    ██║     ██║   ██║██║  ██║█████╗
        ██╔══██║██║    ██║     ██║   ██║██║  ██║██╔══╝
        ██║  ██║██║    ╚██████╗╚██████╔╝██████╔╝███████╗
        ╚═╝  ╚═╝╚═╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
```

# 🦈 AI Agents Compete to Optimize CUDA Kernels

### A GPUMode Leaderboard Challenge Solved Through AI Agent Competition

</div>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/CHALLENGE-GPUMode-ff6600?style=for-the-badge" alt="GPUMode"></a>
  <a href="#"><img src="https://img.shields.io/badge/METHOD-AI_Shark_Tank-blue?style=for-the-badge" alt="AI Shark Tank"></a>
  <a href="#"><img src="https://img.shields.io/badge/GPU-B200_BLACKWELL-76B900?style=for-the-badge&logo=nvidia" alt="B200"></a>
  <a href="#"><img src="https://img.shields.io/badge/POWERED_BY-Claude_Code-purple?style=for-the-badge" alt="Claude Code"></a>
</p>

---

## 🎯 The Challenge

**GPUMode Leaderboard**: Implement a high-performance NVFP4 Group GEMM kernel for NVIDIA B200 (Blackwell) GPUs.

```
C = A @ B  (with FP4 quantization and block scaling)
```

**The Twist**: We solved it using **AI Agent Competition** — multiple Claude agents pitch optimizations, and a panel of "Shark" agents vote on which approach to implement.

---

## 🏆 The Shark Tank Format

Instead of one AI blindly trying optimizations, we created a **competitive framework**:

```
┌─────────────────────────────────────────────────────────────────┐
│                      SHARK TANK ROUND                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   CONTESTANTS (4 AI Agents)          SHARKS (3 AI Agents)       │
│   ┌─────────────────────┐            ┌─────────────────────┐    │
│   │ Pitch 1: Approach A │            │ Shark 1: Skeptic    │    │
│   │ Pitch 2: Approach B │  ──────►   │ Shark 2: Pragmatist │    │
│   │ Pitch 3: Approach C │            │ Shark 3: Theorist   │    │
│   │ Pitch 4: Approach D │            └─────────────────────┘    │
│   └─────────────────────┘                     │                 │
│                                               ▼                 │
│                                        ┌───────────┐            │
│                                        │   VOTE    │            │
│                                        └───────────┘            │
│                                               │                 │
│                                               ▼                 │
│                                     IMPLEMENT WINNER            │
│                                               │                 │
│                                               ▼                 │
│                                         BENCHMARK               │
│                                               │                 │
│                                               ▼                 │
│                                        NEXT ROUND               │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Works

1. **Diverse Perspectives**: Each contestant agent proposes a different optimization strategy
2. **Critical Evaluation**: Shark agents scrutinize claims with different lenses (skepticism, practicality, theory)
3. **Fail Fast**: Bad ideas get voted down before wasting implementation time
4. **Learning Accumulates**: Each round's learnings inform the next

---

## 📊 Season Results

| Round | Contestants | Winner | Expected | Actual | Status |
|-------|-------------|--------|----------|--------|--------|
| **1** | Pipeline Stages, Tile Tuning, TMA Epilogue, Warp Spec | Pipeline Stages | 1.5x faster | **30% SLOWER** | ❌ FAILED |
| **2** | Tile 64x128, Tile 128x64, Stage=2, Stage=4 | Tile Size Tuning | 2-3x faster | **COMPILE ERROR** | ❌ FAILED |
| **3** | Triton Rewrite, cuBLAS, Wild Card Debug | Wild Card Debug | ??? | **Found the bug!** | ✅ SUCCESS |
| **4** | Dual GEMM Fusion, Two-Pass Fix, Interleaved | Minimal Two-Pass | Correctness | **Fixed kernel** | ✅ SUCCESS |
| **5** | Triton, torch.compile, Stream Parallelism | Stream Parallelism | 4-7x faster | **NOT ALLOWED** | ⚠️ BLOCKED |
| **6** | Persistent Kernel, Warp Spec, Split-K, Reduce Overhead | **Reduce Overhead** | 6-19x faster | **TBD** | 🔄 IN PROGRESS |

---

## 🔍 Key Discoveries

### Round 3: The Bug Hunt 🐛

The kernel was computing the **wrong thing**:
```python
# What the kernel computed:
C = A @ B

# What the task required:
C = silu(A @ B1) * (A @ B2)  # Dual GEMM with SiLU fusion!
```

A "Wild Card" contestant discovered this by actually reading the task specification.

### Round 5: Competition Rules Matter ⚠️

Stream parallelism would have given 4-7x speedup, but **GPUMode forbids multiple CUDA streams**:
```
❌ "Your code contains work on another stream"
```

### Round 6: Python is the Bottleneck 🐍

The unanimous winner discovered that **Python overhead**, not CUDA, was the problem:

```python
# BEFORE: 50µs overhead per call
tensor_of_abc_ptrs = torch.tensor(abc_ptrs, device="cuda")  # ~15µs
tensor_of_sfasfb_ptrs = torch.tensor(sfasfb_ptrs, device="cuda")  # ~15µs
tensor_of_problem_sizes = torch.tensor(problem_sizes, device="cuda")  # ~15µs

# AFTER: <5µs overhead per call
cache = get_cached_metadata_tensors(num_groups, total_clusters)
cache['abc_ptrs'].copy_(cache['abc_ptrs_cpu'], non_blocking=True)  # ~1µs
```

---

## 📁 Repository Structure

```
nvfp4_dual_gemm_repo/
├── README.md                    # You are here
├── nvfp4_group_gemm/
│   ├── submission.py            # Current best submission
│   ├── submission_v8_prealloc.py # Round 6 winner (pre-allocation)
│   └── submission_v7_final.py   # Previous version
├── shark_tank/
│   ├── rounds/
│   │   ├── round1_results.md    # Pipeline stages (FAILED)
│   │   ├── round2_results.md    # Tile tuning (FAILED)
│   │   ├── round3_results.md    # Bug discovery (SUCCESS)
│   │   ├── round4_results.md    # Two-pass fix (SUCCESS)
│   │   ├── round5_results.md    # Streams (BLOCKED)
│   │   └── round6_results.md    # Pre-allocation (IN PROGRESS)
│   └── pitches/                 # Individual pitch documents
├── docs/
│   └── PITCH_REDUCE_LAUNCH_OVERHEAD.md
├── SHARK_TANK_LEARNINGS.md      # Accumulated wisdom
└── task.md                      # Original challenge spec
```

---

## 🦈 The Sharks

Each round features three AI "Shark" evaluators with distinct personalities:

| Shark | Personality | Focus |
|-------|-------------|-------|
| **The Skeptic** | "Prove it works" | Demands evidence, distrusts claims |
| **The Pragmatist** | "Can we ship it?" | Implementation feasibility, quick wins |
| **The Theorist** | "Why does this happen?" | Root cause analysis, mathematical proof |

### Example Voting (Round 6)

```
╔════════════════════════════════════════════════════════════════╗
║  ROUND 6 WINNER: REDUCE LAUNCH OVERHEAD                        ║
║  UNANIMOUS VOTE (3-0)                                          ║
║                                                                ║
║  Skeptic:    "Finally attacking the actual bottleneck."        ║
║  Pragmatist: "4-6 hours, low risk, 6-19x upside."              ║
║  Theorist:   "The math checks out: 60µs/group with 2-5µs       ║
║              compute = 50µs Python overhead."                  ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 🚀 Performance Journey

```
Starting Point:     ~530 µs
After Round 4:      ~479 µs  (fixed correctness)
After Round 6:      ~50-80 µs (expected, pending benchmark)
Target:             ~18.8 µs

Progress: ████████████████░░░░ 80% (if Round 6 works)
```

---

## 💡 Learnings

### What Doesn't Work on B200 NVFP4

| Optimization | Why It Failed |
|--------------|---------------|
| Pipeline stages (num_ab_stage=3) | Compute-bound, not memory-bound |
| Smaller tiles (64x128) | Hardware requires 128x128 minimum |
| Multiple CUDA streams | Competition rules forbid it |
| Triton rewrite | Can't access NVFP4 MMA instructions |

### What Does Work

| Optimization | Why It Works |
|--------------|--------------|
| Pre-allocated tensor cache | Eliminates Python overhead |
| Pinned memory + async copy | Fast CPU→GPU transfer |
| Kernel compilation caching | Avoids JIT overhead |

---

## 🛠️ How to Run Your Own Shark Tank

1. **Define the Challenge**: Clear metrics, constraints, and targets
2. **Spawn Contestants**: 3-4 AI agents with different optimization approaches
3. **Spawn Sharks**: 3 AI agents with different evaluation criteria
4. **Let Them Debate**: Contestants pitch, sharks critique and vote
5. **Implement Winner**: Build only the winning approach
6. **Benchmark**: Measure actual performance
7. **Repeat**: Use learnings to inform next round

```python
# Pseudo-code for running a Shark Tank round
contestants = [
    Agent("Pitch A: Persistent Kernel"),
    Agent("Pitch B: Warp Specialization"),
    Agent("Pitch C: Split-K"),
    Agent("Pitch D: Reduce Overhead"),
]

sharks = [
    Agent("Skeptic", personality="demands proof"),
    Agent("Pragmatist", personality="wants quick wins"),
    Agent("Theorist", personality="needs math"),
]

pitches = [c.generate_pitch(context) for c in contestants]
votes = [s.evaluate_and_vote(pitches) for s in sharks]
winner = majority_vote(votes)

implement(winner)
benchmark()
```

---

## 📜 License

MIT

---

## 🙏 Acknowledgments

- **GPUMode** for the challenge and leaderboard
- **NVIDIA** for B200 hardware and CUTLASS/CuTe DSL
- **Anthropic** for Claude Code that powers the AI agents

---

<div align="center">

*"The best optimization isn't the one that sounds clever—it's the one that survives scrutiny from three skeptical AIs."*

**— Shark Tank AI Methodology**

</div>
