# COMPETITION RULES & EVALUATION MECHANICS: ARC PRIZE 2026 (ARC-AGI-3)

## 1. Executive Summary & Objective
The ARC Prize 2026 (ARC-AGI-3) Kaggle competition measures genuine fluid intelligence in interactive environments. Unlike static ARC benchmarks, ARC-AGI-3 evaluates an agent's ability to interact with dynamic, unknown grid environments, induce physical and causal rules through active intervention, build a working world model, and achieve win conditions with maximum action efficiency.

**Central Optimization Goal:** Maximize final leaderboard evaluation score with **Relative Human Action Efficiency (RHAE)** as the core scoring multiplier.

---

## 2. Official Timeline & Prize Structure ($850,000 Total Pool)

### 2.1 Key Deadlines (11:59 PM UTC)
- **March 25, 2026:** Competition Launch & Official Start.
- **June 30, 2026:** Milestone #1 Deadline (Open-Source Agent Evaluation).
- **September 30, 2026:** Milestone #2 Deadline (Open-Source Agent Evaluation).
- **October 26, 2026:** Team Merger & Entry Deadline.
- **November 2, 2026:** Final Submission Deadline (Notebook freeze & final code submission).
- **November 8, 2026:** Research Paper Submission Deadline.
- **December 4, 2026:** Official Winners Announcement & Leaderboard Unblinding.

### 2.2 Prize Pool Allocation
- **Grand Prize ($700,000):** Awarded to the first eligible team achieving a verified 100% score.
- **Top Score Leaderboard Prizes ($75,000):** 1st ($40k), 2nd ($15k), 3rd ($10k), 4th ($5k), 5th ($5k).
- **Milestone Prizes ($75,000):** Rewarding open-source architectures published early.

---

## 3. Hardware, Environment & Runtime Constraints

### 3.1 Hardware Infrastructure (Kaggle Sandbox)
- **Primary Accelerator:** NVIDIA T4 x2 (Dual GPU)
  - VRAM: 16 GB per T4 = 32 GB total VRAM.
  - Interconnect: PCIe (no NVLink; tensor parallelism has PCIe overhead).
- **Alternative / Exclusive Accelerator:** NVIDIA RTX 6000 (`g4-standard-48` instance)
  - VRAM: Up to 48 GB dedicated high-throughput compute.
  - Reserved exclusively for ARC-AGI-3 scoring notebooks.
- **CPU & Host Memory:**
  - 4 vCPUs.
  - ~30 GB System RAM.
- **Storage:**
  - Standard Kaggle working scratch disk (~20 GB writeable at `/kaggle/working/`).
  - Kaggle Datasets mounted read-only under `/kaggle/input/`.

### 3.2 Global Execution & Timeout Limits
- **Wall-Clock Time Limit:** Exactly **9 Hours (32,400 seconds)** for full notebook execution.
- **Game Evaluation Scale:** **110 hidden evaluation games**.
- **Time Budget Per Game:**
  $$\text{Average Time Budget} = \frac{32,400\text{ s}}{110\text{ games}} \approx 294.5\text{ seconds/game} \ (\approx 4.9\text{ minutes/game})$$
- **System Pacing Requirement:** Hard per-game watchdog ($\le 240$s) to prevent catastrophic timeout on later games.
- **Gateway Idle Timeout:** The gateway closes scorecards if an agent idles for $>15$ minutes (`STALE_MINUTES = 15`).

### 3.3 Network & Licensing Isolation
- **Network Access:** **Strictly Disabled (Zero Internet Access)**.
  - No external API calls permitted (OpenAI, Anthropic, Google Cloud APIs are prohibited during evaluation).
  - All model weights, tokenizers, Python wheels, and dependencies must be pre-packaged into offline Kaggle Datasets.
- **Licensing & Reproducibility:**
  - Open-source, open-weights only (Apache 2.0, MIT, CC0).
  - Proprietary or closed-weight models are disqualified.
  - Final winning submissions must be completely reproducible from scratch.

---

## 4. Dataset Architecture & The 110-Game Hidden Split

### 4.1 Dataset Hierarchy
| Cohort | Game Count | Access | Role in Development |
| :--- | :--- | :--- | :--- |
| **Public Demo Set** | 25 games | Fully open | Smoke testing, API integration validation only. Never optimize directly on this cohort. |
| **Local Offline Suite** | 252 environments | Local repo (`arc-interactive/`) | Structural analysis of framework mechanics. Strict anti-overfitting rules apply. |
| **Competition Hidden Evaluation** | **110 hidden games** | **Completely private** | **The true scoring benchmark.** |

### 4.2 Leaderboard Partitioning
The 110 hidden games are split 50/50:
1. **Public Leaderboard (55 games):** Evaluated during each submission to generate the public score.
2. **Private Leaderboard (55 games):** Evaluated simultaneously but hidden until the competition closes. Determines final prize awards.

---

## 5. Dual-Phase Submission Lifecycle & Gateway Sidecar Architecture

### 5.1 Phase A vs. Phase B Lifecycle
Kaggle evaluates submissions via two distinct execution phases:

1. **Phase A: Interactive Commit ("Save & Run All")**
   - Environment flag `KAGGLE_IS_COMPETITION_RERUN` is **NOT set** (or false).
   - The gateway sidecar container is **offline**.
   - **Requirement:** The notebook must immediately ($<5$ seconds) emit a minimal valid dummy `submission.parquet` and exit cleanly (code 0). Failure to do so prevents notebook saving.
2. **Phase B: Official Competition Rerun**
   - Triggered when clicking "Submit to Competition".
   - Environment flag `KAGGLE_IS_COMPETITION_RERUN == "true"`.
   - Kaggle spins up the background **gateway sidecar container** at `http://gateway:8001`.
   - The notebook must handshake with the gateway (`curl http://gateway:8001/api/games`), step through actions, and finalize the scorecard.

### 5.2 Required Parquet Schema: `/kaggle/working/submission.parquet`
```python
columns = ["row_id", "game_id", "end_of_game", "score"]
```
- `row_id` (int64 / string): Unique record identifier.
- `game_id` (string): Short identifier of the evaluated game.
- `end_of_game` (bool): Boolean flag (`True` on completion).
- `score` (float64): Numerical score achieved on the environment (0.0 to 115.0).

### 5.3 Submission Limits
- Strictly **1 submission per 24-hour rolling window** for ARC-AGI-3 (to prevent test set probing).
- Final submission selection: Exactly 2 selected notebooks for private leaderboard evaluation.

---

## 6. Official Scoring Formula & Linear Level Weighting

### 6.1 Linear Level Weight Escalation
The ARC-AGI-3 scoring engine (`arc_agi.scorecard.EnvironmentScoreCalculator`) weights levels linearly by their index $w_i = i$:
$$\text{weight}_i = i \quad \text{for } i \in \{1, 2, \dots, N\}$$

$$\text{total\_score} = \sum_{i=1}^{N} \text{level\_score}_i \times i$$
$$\text{total\_weights} = \sum_{i=1}^{N} i = \frac{N(N+1)}{2}$$
$$\text{max\_weights} = \sum_{i \in \text{completed}} i$$

$$\text{raw\_score} = \frac{\text{total\_score}}{\text{total\_weights}}, \quad \text{max\_score} = \frac{\text{max\_weights}}{\text{total\_weights}} \times 100$$
$$\text{Environment Score} = \min(\text{raw\_score}, \text{max\_score})$$

### 6.2 Level Score & RHAE Formula
For each level $i$:
$$\text{level\_score}_i = \begin{cases} 
\min\left(115.0, \ \left(\frac{H_i}{A_i}\right)^2 \times 100\right) & \text{if level } i \text{ completed} \\ 
0.0 & \text{if level } i \text{ failed} 
\end{cases}$$
where $H_i$ is median human baseline actions and $A_i$ is AI actions taken.

### 6.3 The Quadratic RHAE Cliff
The quadratic term $\left(\frac{H}{A}\right)^2$ penalizes action bloat severely:

| Action Ratio ($A_i / H_i$) | Efficiency Description | RHAE Score |
| :--- | :--- | :--- |
| **$0.93\times$** | Faster than human median | **$1.15$ (Superhuman Cap)** |
| **$1.00\times$** | Human parity | **$1.00$ (100.0%)** |
| **$1.25\times$** | 25% extra actions | **$0.64$ (64.0%)** |
| **$1.50\times$** | 50% extra actions | **$0.44$ (44.4%)** |
| **$2.00\times$** | $2\times$ human actions | **$0.25$ (25.0%)** |
| **$3.00\times$** | $3\times$ human actions | **$0.11$ (11.1%)** |
| **$5.00\times$** | $5\times$ human actions | **$0.04$ (4.0%)** |
| **$10.00\times$** | Random walk / blind search | **$0.01$ (1.0%)** |

> [!CRITICAL]
> **Scoring Rule Takeaways:**
> 1. In a 5-level game, Level 5 represents $33.3\%$ of total weight, while Level 1 represents only $6.7\%$.
> 2. An agent that only solves Level 1 cannot exceed $6.67\%$ score regardless of efficiency! Solving later levels is mandatory.
> 3. Resetting via `ACTION 0` increments total actions and penalizes RHAE quadratically.

---

## 7. Action Space & Engine Interface

### 7.1 Action Enums (`arcengine.enums.GameAction`)
| Action ID | Name | Parameters | Semantic Behavior |
| :--- | :--- | :--- | :--- |
| `0` | `RESET` | None | Reinitializes the current level state; increments action counter. |
| `1` | `ACTION1` | None | Up / Cardinal Direction North. |
| `2` | `ACTION2` | None | Down / Cardinal Direction South. |
| `3` | `ACTION3` | None | Left / Cardinal Direction West. |
| `4` | `ACTION4` | None | Right / Cardinal Direction East. |
| `5` | `ACTION5` | None | Interact / Select / Collect / Fire / Toggle. |
| `6` | `ACTION6` | `{"x": int, "y": int}` | Parameterized Coordinate Click within $[0, 63] \times [0, 63]$. |
| `7` | `ACTION7` | None | Undo / Reversible Step. |

### 7.2 Dynamic Action Whitelist
- At every step, the agent must check `FrameData.available_actions`.
- Submitting an action outside this list causes an illegal action rejection.
- `ACTION 6` requires valid integer coordinates: `{"x": x, "y": y}` clamped to `[0, 63]`. Sending empty data or floats causes Pydantic validation failure.
