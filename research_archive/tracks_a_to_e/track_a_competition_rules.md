# RESEARCH REPORT: TRACK A — COMPETITION RULES AND EVALUATION MECHANICS
**ARC Prize 2026 — ARC-AGI-3 Kaggle Competition**
**Specialist:** Track A Specialist (Competition Rules & Evaluation Mechanics)
**Date:** October 13, 2026

---

### EXECUTIVE SUMMARY

The **ARC Prize 2026 — ARC-AGI-3** competition represents a fundamental paradigm shift from static grid transformation tasks (ARC-AGI-1 and 2) to **dynamic, interactive POMDP environments** where agents receive no upfront instructions, natural language prompts, or internal environment states. Agents must explore, infer latent causal rules, maintain revisable world models, plan, and act under strict compute and action budgets.

This report provides an exhaustive, mathematical, and mechanical analysis of the competition framework, scoring infrastructure, Kaggle offline evaluation sandbox, hardware quotas, and known failure modes that cause local agents to collapse on the hidden leaderboard.

---

### 1. OFFICIAL COMPETITION TIMELINE & PRIZE STRUCTURE

#### 1.1 Key Milestones & Deadlines (All deadlines 11:59 PM UTC)
* **March 25, 2026:** Competition Launch & Official Start.
* **June 30, 2026:** Milestone #1 Deadline (ARC-AGI-3 Open-Source Agent Evaluation).
* **September 30, 2026:** Milestone #2 Deadline (ARC-AGI-3 Open-Source Agent Evaluation).
* **October 26, 2026:** Team Merger & Entry Deadline (Final date to accept rules and form teams).
* **November 2, 2026:** Final Submission Deadline (Notebook freeze & final code submission).
* **November 8, 2026:** Research Paper Submission Deadline (For eligible research paper prize tracks).
* **December 4, 2026:** Official Winners Announcement & Leaderboard Unblinding.

#### 1.2 Prize Pool Architecture ($850,000 Total Pool)
* **Grand Prize ($700,000):** Awarded to the first eligible team whose open-source agent achieves a verified **100% score** on the ARC-AGI-3 evaluation suite. If unclaimed, rolls over to future iterations.
* **Top Score Leaderboard Prizes ($75,000 Guaranteed):**
  * 1st Place: $40,000
  * 2nd Place: $15,000
  * 3rd Place: $10,000
  * 4th Place: $5,000
  * 5th Place: $5,000
* **Milestone Prizes ($75,000 Guaranteed):** Distributed across Milestone #1 and Milestone #2 to reward teams who publish high-performing open-source architectures early in the competition cycle.

---

### 2. SUBMISSION FORMAT & DUAL-PHASE EVALUATION PROTOCOL

#### 2.1 Kernel / Notebook Submission Architecture
The competition is an **offline code competition** evaluated exclusively through Kaggle Notebooks. 

#### 2.2 The Dual-Phase Lifecycle: Phase A vs. Phase B
Kaggle evaluates submissions via a two-phase execution lifecycle that agents must explicitly handle:

1. **Phase A: Interactive Commit ("Save & Run All")**
   * **Trigger:** Participant clicks "Save Version" / "Save & Run All".
   * **Environment Flag:** `os.getenv("KAGGLE_IS_COMPETITION_RERUN")` is **NOT set** (or false).
   * **Behavior:** The gateway sidecar container is **offline**. The notebook must NOT attempt to connect to the gateway or run long evaluations.
   * **Mandatory Requirement:** The notebook must immediately (within <5 seconds) emit a minimal, valid dummy `submission.parquet` file to `/kaggle/working/submission.parquet` and exit cleanly (code 0). Failure to emit this artifact causes commit failure, preventing submission.
2. **Phase B: Official Competition Rerun**
   * **Trigger:** Participant clicks "Submit to Competition" on the completed notebook version.
   * **Environment Flag:** `KAGGLE_IS_COMPETITION_RERUN == "true"`.
   * **Infrastructure:** Kaggle spins up a background **gateway sidecar container** at `http://gateway:8001`.
   * **Startup Handshake:** The submission notebook must poll the gateway sidecar until ready:
     ```bash
     curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games
     ```
   * **Execution:** The agent queries `GET http://gateway:8001/api/games`, instantiates games via the `Arcade` client or REST endpoints, steps through actions, and closes scorecards.
   * **Output:** The gateway sidecar intercepts actions and dumps the authoritative `submission.parquet` directly into `/kaggle/working/`.

#### 2.3 Required Output Parquet Schema
The file `/kaggle/working/submission.parquet` must adhere strictly to the following 4-column schema:
```python
columns = ["row_id", "game_id", "end_of_game", "score"]
```
* `row_id` (int64 / string): Unique index identifier per record (e.g. `0, 1, 2...` or `"1_0"`).
* `game_id` (string): Unique identifier for the evaluated game environment (e.g. `"ls20"`, `"bt11"`).
* `end_of_game` (bool): Boolean flag indicating whether the game run concluded (`True` on completion/termination).
* `score` (float64): Numerical score achieved on the environment (normalized 0.0 to 1.15 or level fraction).

---

### 3. RUNTIME & COMPUTE CONSTRAINTS

#### 3.1 Global Sandbox Constraints
* **Global Execution Limit:** **9 hours (32,400 seconds)** wall-clock limit. If execution exceeds 32,400 seconds, the kernel receives `SIGKILL`, the submission fails with a Timeout Error, and the entire scorecard receives 0.
* **Network Isolation:** **Strictly offline** (`isInternetEnabled: False`). Outbound sockets to external domains fail immediately with connection timeouts.
* **CPU & Host Memory:** 4 vCPUs (Intel Xeon / AMD EPYC), **~30 GB system RAM**.
* **Scratch Disk:** Ephemeral storage at `/kaggle/working/` (~20 GB writable).

#### 3.2 Available Hardware Accelerators
Participants configure the accelerator in notebook metadata (`kernel-metadata.json` / `build_notebook.py`):
1. **CPU Only (`"none"`):** 4 vCPUs, 30 GB RAM. Suitable only for symbolic/heuristic agents.
2. **Dual NVIDIA T4 (`"nvidiaTeslaT4"`):** 2x 16 GB VRAM (**32 GB total VRAM**). Standard default for small quantized models (e.g. 7B/14B Q4).
3. **Single NVIDIA P100 (`"nvidiaTeslaP100"`):** 1x 16 GB high-bandwidth HBM2 VRAM.
4. **NVIDIA RTX 6000 (`"nvidiaRtx6000"`):** GCP `g4-standard-48` equivalent, **up to 48 GB / 96 GB VRAM**. Reserved specifically for ARC-AGI-3 notebooks. Consumes Kaggle GPU quota rapidly; required for heavy 27B-class LLMs (e.g., Qwen 2.5 27B / 32B quantized).

---

### 4. LICENSING & OPEN-WEIGHTS REQUIREMENTS

* **Zero External APIs:** Direct API calls to OpenAI (GPT-4o), Anthropic (Claude 3.5 Sonnet), Google (Gemini 1.5 Pro), or any proprietary provider are strictly prohibited and technically blocked by network isolation.
* **Permissible Code Licenses:** To claim competition rankings and prizes, submitter code must be released under permissive open-source licenses: **Apache 2.0**, **MIT**, **CC0**, or **MIT-0**.
* **Open-Weights Compatibility:** All foundation models must have publicly available, permissible open weights (e.g., Apache 2.0, MIT, Llama Community License).
* **Offline Asset Packaging:** Model weights, tokenizers, inference engines (e.g. `llama.cpp` wheels, `vLLM` binaries), and dependencies must be pre-packaged into Kaggle Datasets mounted under `/kaggle/input/...` and installed offline with `--no-index --find-links`.

---

### 5. BENCHMARK ARCHITECTURE & LEADERBOARD SPLIT

The full ARC-AGI-3 evaluation corpus comprises **135 interactive game environments**:
* **25 Public Demonstration Games:** Fully open-source environments distributed in `arc-interactive/environment_files/` for local development, diagnostic testing, and heuristic verification.
* **110 Hidden Evaluation Games:** Environments never disclosed to participants:
  * **55 Games for the Public Leaderboard (50%):** Evaluated during each Phase B submission to produce the real-time public leaderboard standing.
  * **55 Games for the Final Private Leaderboard (50%):** Evaluated simultaneously but blinded until the final competition unblinding on December 4, 2026.
* **Significance:** A public score based on the 25 demo games provides zero guarantee of leaderboard performance. Solutions that overfit to known mechanics inevitably collapse when facing the 110 novel hidden environments.

---

### 6. EXACT SCORING MECHANICS & ENVIRONMENT SCORECARD CALCULATION

The authoritative scoring logic is implemented in `arc_agi.scorecard.EnvironmentScoreCalculator` and `arc_agi.scorecard.EnvironmentScorecard`.

#### 6.1 Level Structure & Progression
* Each game contains $N$ levels ($N = \text{win\_levels}$, typically between 1 and 5).
* Game states transition through: `NOT_PLAYED` $\to$ `PLAYING` / `NOT_FINISHED` $\to$ `WIN` or `GAME_OVER`.
* An environment run is considered `completed` if and only if `state == GameState.WIN`.

#### 6.2 Level Score Formula
For each level $i \in \{1, \dots, N\}$ (1-indexed):
$$\text{level\_score}_i = \begin{cases} \min\left(115.0, \left(\frac{\text{baseline\_actions}_i}{\text{actions\_taken}_i}\right)^2 \times 100\right) & \text{if level } i \text{ completed} \\ 0.0 & \text{if level } i \text{ failed / uncompleted} \end{cases}$$

#### 6.3 Environment Score Formula: Linear Level Weighting
A crucial property of the ARC-AGI-3 scoring engine is that **levels are weighted linearly by their index** $w_i = i$:
$$\text{weight}_i = i \quad \text{for } i \in \{1, 2, \dots, N\}$$

The total score and total weights are computed as:
$$\text{total\_score} = \sum_{i=1}^{N} \text{level\_score}_i \times i$$
$$\text{total\_weights} = \sum_{i=1}^{N} i = \frac{N(N+1)}{2}$$
$$\text{max\_weights} = \sum_{i \in \text{completed}} i$$

The raw weighted average is:
$$\text{raw\_score} = \frac{\text{total\_score}}{\text{total\_weights}}$$
$$\text{max\_score} = \frac{\text{max\_weights}}{\text{total\_weights}} \times 100$$
$$\text{Environment Score} = \min(\text{raw\_score}, \text{max\_score})$$

#### 6.4 Key Mathematical Insights on Environment Scoring
1. **Linear Weight Escalation:** In a 5-level game:
   * Level 1 weight = 1 (6.7% of total weight)
   * Level 2 weight = 2 (13.3% of total weight)
   * Level 3 weight = 3 (20.0% of total weight)
   * Level 4 weight = 4 (26.7% of total weight)
   * Level 5 weight = 5 (33.3% of total weight)
   * *Conclusion:* Completing earlier levels with superhuman efficiency is worth very little if the agent cannot solve later levels.
2. **The Completion Ceiling (`max_score` Cap):**
   * If an agent completes only Level 1 with maximum efficiency (115%):
     $$\text{raw\_score} = \frac{115 \times 1}{15} = 7.67\%$$
     $$\text{max\_score} = \frac{1}{15} \times 100 = 6.67\%$$
     $$\text{Final Score} = \min(7.67\%, 6.67\%) = 6.67\%$$
   * The `max_score` cap guarantees that action efficiency cannot artificially inflate the score beyond the fraction of completed level weights!

#### 6.5 Scorecard Aggregation Across Environments
* For an environment played across multiple runs/episodes, the engine takes the **maximum score achieved across runs**:
  $$\text{env\_score} = \max_{\text{run}} (\text{run.score})$$
* The aggregate competition score across all environments $E$ is the arithmetic mean:
  $$\text{Competition Score} = \frac{1}{|E|} \sum_{e \in E} \text{env\_score}_e$$

---

### 7. RELATIVE HUMAN ACTION EFFICIENCY (RHAE) FORMULA

#### 7.1 Mathematical Definition
$$\text{RHAE}(h_l, a_l, \text{is\_won}) = \begin{cases} \min\left(1.15, \left(\frac{h_l}{a_l}\right)^2\right) & \text{if } \text{is\_won} = \text{True} \text{ and } a_l > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
where:
* $h_l$: Median human baseline action count for level $l$.
* $a_l$: Total AI actions taken on level $l$.
* $\text{is\_won}$: Boolean indicating successful level completion.

#### 7.2 The Quadratic Cliff: Extreme Sensitivity to Action Bloat
The quadratic term $\left(\frac{h_l}{a_l}\right)^2$ penalizes inefficient exploration brutally:
| Action Ratio ($a_l / h_l$) | Explanation | RHAE Score |
|---|---|---|
| **$0.93\times$** | Faster than human median | **$1.15$ (Capped Max)** |
| **$1.00\times$** | Exactly matches human median | **$1.00$ (100.0%)** |
| **$1.25\times$** | 25% more actions than human | **$0.64$ (64.0%)** |
| **$1.50\times$** | 50% more actions than human | **$0.44$ (44.4%)** |
| **$2.00\times$** | 2x human actions | **$0.25$ (25.0%)** |
| **$3.00\times$** | 3x human actions | **$0.11$ (11.1%)** |
| **$5.00\times$** | 5x human actions | **$0.04$ (4.0%)** |
| **$10.00\times$**| 10x human actions (blind search) | **$0.01$ (1.0%)** |

---

### 8. ACTION BUDGETS & ENGINE MECHANICS

#### 8.1 Action Space Definition (`arcengine.enums.GameAction`)
ARC-AGI-3 specifies 8 discrete actions:
* **ACTION 0: `RESET` (SimpleAction):** Resets the current level state. Increments the action counter and reset counter.
* **ACTION 1: `UP` (SimpleAction)**
* **ACTION 2: `DOWN` (SimpleAction)**
* **ACTION 3: `LEFT` (SimpleAction)**
* **ACTION 4: `RIGHT` (SimpleAction)**
* **ACTION 5: `INTERACT` (SimpleAction):** Context-dependent interaction (Toggle, Collect, Place, Activate, Grab).
* **ACTION 6: `CLICK / PLACE` (ComplexAction):** Coordinate-parameterized action targeting the 64x64 canvas:
  ```python
  ComplexAction(x=int, y=int)  # 0 <= x <= 63, 0 <= y <= 63
  ```
  Requires passing a data payload: `{"x": x, "y": y}`.
* **ACTION 7: `UNDO` (SimpleAction):** Reverts the preceding action if supported by the game engine.

#### 8.2 Dynamic Action Whitelist
* Agents must inspect `FrameData.available_actions` at every step. Submitting an action outside this list causes an illegal action rejection.
* If `ACTION 6` is chosen, the agent must include valid integer coordinates `x` and `y` clamped to `[0, 63]`. Sending empty data or floats causes Pydantic validation failure.

---

### 9. TOP 10 PITFALLS CAUSING LEADERBOARD COLLAPSE

1. **Overfitting to the 25 Public Demo Games:** Color assumptions and coordinate fingerprints fail completely on the 110 hidden games.
2. **Action Space Blindness:** Ignoring `available_actions` and issuing invalid action IDs.
3. **Malformed `ACTION 6` Coordinates:** Missing `data={"x": x, "y": y}` or passing out-of-bounds coordinates.
4. **Global Wall-Clock Exhaustion (9-Hour Limit):** 110 hidden games in 32,400s requires $\le 294$s average per game.
5. **Memory Leaks and OOM Termination (Exit Code 137):** Retaining full raw frame histories across 110 games consumes all 30 GB of system RAM.
6. **Gateway Sidecar Race Conditions in Phase B:** Failing to poll `http://gateway:8001/api/games` until ready before starting.
7. **Scorecard Abandonment & Gateway Idle Timeouts:** Gateway marks scorecard stale after $>15$ minutes of inactivity (`STALE_MINUTES = 15`).
8. **Quadratic RHAE Action Degradation:** Taking 200+ actions on a 20-action baseline drops RHAE to 0.01.
9. **Linear Level Weight Penalty:** Early levels represent only a fraction of total game points; completing higher levels is mandatory.
10. **Phase A Commit Failure:** Failing to emit a dummy `submission.parquet` within $<5$s during notebook commit prevents Kaggle from saving the notebook version.
