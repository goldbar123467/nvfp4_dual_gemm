# SHARK TANK ROUND 3: THE GAME SHOW RECAP

---

```
  ____    _    __  __ _____   ____  _   _  _____        __
 / ___|  / \  |  \/  | ____| / ___|| | | |/ _ \ \      / /
| |  _  / _ \ | |\/| |  _|   \___ \| |_| | | | \ \ /\ / /
| |_| |/ ___ \| |  | | |___   ___) |  _  | |_| |\ V  V /
 \____/_/   \_\_|  |_|_____| |____/|_| |_|\___/  \_/\_/

         _____ ___ __  __ _____   _____ ___
        |_   _|_ _|  \/  | ____| |_   _/ _ \
          | |  | || |\/| |  _|     | || | | |
          | |  | || |  | | |___    | || |_| |
          |_| |___|_|  |_|_____|   |_| \___/

     ____ _____ _____   __        _____ _____ ____  ____
    / ___| ____|_   _|  \ \      / /_ _|  ___|  _ \|  _ \
   | |  _|  _|   | |     \ \ /\ / / | || |_  | | | | | | |
   | |_| | |___  | |      \ V  V /  | ||  _| | |_| | |_| |
    \____|_____| |_|       \_/\_/  |___|_|   |____/|____/
```

---

## EPISODE 3: "THE WILD CARD RISES"

### PREVIOUSLY ON SHARK TANK...

*[Dramatic music plays]*

**ROUND 1: "THE SAFE BET"**
```
ANNOUNCER: The sharks unanimously backed Pipeline Stages!
           "One line change! 1.5x speedup! Can't lose!"

*[Tests run]*

RESULTS SCREEN: -30% PERFORMANCE
                TENSOR CORES: "We were already fed, idiots"

SHARKS: *shocked faces*
```

**ROUND 2: "THE I-TOLD-YOU-SO"**
```
TILE TUNING: "The sharks didn't listen to ME! Now they'll see!"
SHARKS: "You were right all along! UNANIMOUS YES!"

*[Compile button pressed]*

COMPILER: "Error: expects M-mode to be 128, but got 64"
COMPILER: "Did you even READ the hardware spec?"

TILE TUNING: "But... NVIDIA said..."
COMPILER: "NVIDIA said 128. You said 64. I said no."

SHARKS: *even more shocked faces*
```

---

## TONIGHT'S EPISODE: ROUND 3

*[Host walks on stage]*

**HOST**: "Welcome back to Shark Tank: GPU Optimization Edition! I'm your host, and WOW, what a season it's been!"

*[Audience laughter]*

**HOST**: "Our sharks are now 0 for 2. Pipeline Stages made things WORSE. Tile Tuning made things NOT COMPILE. At this point, I'm pretty sure my grandmother could pick better optimizations."

*[Sharks look sheepish]*

**HOST**: "But tonight, we have a NEW contestant! Please welcome... THE WILD CARD!"

*[Pyrotechnics, smoke machine, air horn sounds]*

---

## THE PITCHES

### CONTESTANT A: TMA STORE EPILOGUE

**TMA**: "I'm still here. I've survived two rounds of other people's failures."

**SHARKS**: "What's your speedup?"

**TMA**: "Honestly? 0-5%. Maybe. I'm not even sure the epilogue matters."

**SHARKS**: "That's... refreshingly honest?"

**TMA**: "Look, I've seen what happens when you overpromise. Pipeline said 1.5x, got -30%. Tile said 2-3x, got compile error. I'm saying 'probably nothing but at least I won't break it.'"

**SHARKS**: "We appreciate the humility but that's not very exciting."

---

### CONTESTANT B: WARP SPECIALIZATION

**WARP**: "You know what's funny? You rejected me twice for being 'too complex.'"

**SHARKS**: "..."

**WARP**: "You know what the 'simple' approaches got you?"

**SHARKS**: "..."

**WARP**: "NEGATIVE THIRTY PERCENT AND A COMPILE ERROR."

*[Audience cheers]*

**WARP**: "Maybe - and hear me out - maybe CAREFUL complexity is better than RECKLESS simplicity?"

**SHARK 1**: "I... I never thought of it that way."

**WARP**: "Of course you didn't. You were too busy saying 'one line change, what could go wrong?'"

---

### CONTESTANT C: THE WILD CARD

*[Lights go out. Single spotlight. Smoke machine.]*

**WILD CARD**: "Sharks. I've looked at your kernel."

**SHARKS**: "...okay?"

**WILD CARD**: "Have YOU looked at your kernel?"

**SHARKS**: "Of course we—"

**WILD CARD**: "Then tell me: WHERE IS THE SECOND GEMM?"

*[Dramatic music sting]*

**SHARKS**: "The... the what?"

**WILD CARD**: "The task is `C = silu(A @ B1) * (A @ B2)`. That's TWO GEMMs. One with B1, one with B2, multiplied together with a SiLU activation."

**SHARKS**: "Right..."

**WILD CARD**: "Your kernel computes ONE GEMM. Where's the other one? Where's the SiLU? Where's the multiply?"

**SHARKS**: *[look at each other nervously]*

**WILD CARD**: "You've spent two rounds optimizing a kernel that MIGHT NOT BE DOING THE RIGHT COMPUTATION. No wonder you're 20-100x off target."

*[Audience gasps]*

**SHARK 2**: "Oh no."

**SHARK 3**: "Oh no."

**SHARK 1**: "...oh no."

**WILD CARD**: "Before you optimize another line of code, maybe VERIFY IT'S THE RIGHT CODE?"

---

## THE VOTE

**HOST**: "Sharks, it's time to vote!"

**SHARK 1 (Performance Oracle)**:
"I gave Pipeline 8.4. I gave Tiles 8.7. They both failed. Wild Card is asking the question we should have asked in Round 1: Are we even solving the right problem? **I vote Wild Card.**"

**SHARK 2 (Pragmatic Engineer)**:
"I thought pragmatic meant 'ship fast.' Now I think pragmatic means 'verify you're building the right thing.' Wild Card's investigation has the best risk-adjusted value. **I vote Wild Card.**"

**SHARK 3 (ROI Maximizer)**:
"My ROI formula said Pipeline was 3x speedup per hour. Actual ROI: negative. My formula is broken. Wild Card's 'check if kernel is correct' costs 1 hour and could explain our 20-100x gap. **I vote Wild Card.**"

---

## THE RESULT

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    ROUND 3 WINNER: THE WILD CARD                                 ║
║                                                                  ║
║    UNANIMOUS VOTE (3-0)                                          ║
║                                                                  ║
║    "Maybe check if the kernel does what it's supposed to?"       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## SEASON RECAP

| Round | Winner | What We Expected | What We Got | Lesson |
|-------|--------|------------------|-------------|--------|
| 1 | Pipeline Stages | 1.5x faster | **30% SLOWER** | "Industry standard" doesn't mean "universally applicable" |
| 2 | Tile Size Tuning | 2-3x faster | **COMPILE ERROR** | Maybe read the hardware requirements first |
| 3 | Wild Card | ??? | **TBD** | Maybe verify the kernel is correct before optimizing it |

---

## WHAT HAPPENS NEXT

The Wild Card's top recommendations:

1. **INVESTIGATE**: Is the kernel actually computing `silu(A@B1) * (A@B2)` or just one GEMM?

2. **ZERO-RISK EXPERIMENT**: Try reversing the K-loop order (one line change, can't make things worse)

3. **PROFILE FIRST**: Before any changes, measure where time actually goes

---

## HOST'S CLOSING

**HOST**: "And that's a wrap on Round 3! The Wild Card has done what no contestant could do before: make the sharks question whether they were even solving the right problem."

*[Turns to camera]*

**HOST**: "Join us next time on Shark Tank: GPU Optimization Edition, where we might finally make something faster instead of slower or broken!"

*[Credits roll]*

---

```
NEXT EPISODE:
"The Investigation"
Will the kernel be doing dual GEMM?
Or have we been optimizing a lie?

FIND OUT NEXT TIME ON...
SHARK TANK: GPU OPTIMIZATION EDITION
```

---

*"The best optimization is optimizing the right thing."*
*- The Wild Card*

