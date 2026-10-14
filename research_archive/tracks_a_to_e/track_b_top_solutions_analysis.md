# ARC-AGI-3 COMPETITIVE LANDSCAPE & TOP PUBLIC SOLUTIONS ANALYSIS

**To:** Parent Agent (`e2521de7-f050-4c42-b9e4-5c1062a99a1a`)  
**From:** Track B Specialist: Analysis of Top Public Solutions for ARC-AGI-3  
**Date:** October 13, 2026  
**Subject:** Exhaustive Technical Breakdown of Top ARC-AGI-3 Community Solutions, Architectures, Leaderboard Transfer Gaps, and Ranked Transferable Mechanisms

---

## 1. EXECUTIVE SUMMARY & COMPETITIVE BENCHMARK OVERVIEW

ARC-AGI-3 marks a fundamental paradigm shift from static grid puzzles (ARC-AGI-1/2) to **interactive, turn-based fluid intelligence**. Agents interact with unfamiliar, black-box 2D environments (typically 64×64 grids, 16-color palettes) with zero natural-language instructions, no stated goals, and no predefined action semantics.

### Key Benchmark Constraints & Metrics
* **Official Evaluation Metric — Relative Human Action Efficiency (RHAE):**  
  $$\text{RHAE} = \text{aggregate score normalized against the upper-median human baseline}$$
  Human players establish median action counts per level. Agents are evaluated on efficiency ($\sim (\text{Human Actions} / \text{Agent Actions})^2$, capped at 115% per level). A score of 100% means human parity.
* **Evaluation Cohorts & The "Grand Transfer Gap":**
  * **Public Demo Set:** 25 games (openly inspectable in local environments).
  * **Hidden Evaluation Set:** 110 completely separate hidden games (55 games for Public Leaderboard; 55 games for Final Private Leaderboard).
  * **The Transfer Cliff:** Top research systems achieved **100.00 RHAE** on the 25 public games using heavy frontier models and prompt caching. However, on the official Kaggle Competition Leaderboard (under strict 9-hour offline limits on 110 games), the **winning Milestone 1 score was 1.21%** (Tufa Labs' *The Duck*).
  * **Community Evidence:** Local/public performance translates to only **20%–45%** on held-out distributions, and drops even further when offline compute/action budgets are restricted.

---

## 2. DEEP-DIVE PROFILES OF TOP SOLUTIONS & CONTENDERS

---

### Profile 1: Tufa Labs' "The Duck" (1st Place Milestone 1 Winner — $25,000)
* **Authors:** Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal Tešnar, Stefano Viel.
* **Lineage:** Successor to Tufa Labs' earlier *Stochastic Goose* agent.
* **Base Model & Quantization:** **Qwen 3.6 27B** in **FP8** (deployed on Kaggle single-GPU H100/A100 instances); also evaluated with Qwen 2.5 32B/72B and Claude/GPT via OpenRouter during development.
* **Inference Engine:** **vLLM** local offline inference server on Kaggle (`taaf-duck-harness-kaggle-share`).
* **Harness Architecture & Agent Loop:**
  * Built on **TAAF** (*Tufa ARC-AGI Framework*), executing the GameAPI.
  * Treats ARC-AGI-3 as an **interactive coding problem** using a live Python Read-Eval-Print Loop (REPL).
  * Game states are converted into Python variables. The LLM writes Python scripts to inspect arrays, define helper routines, test hypotheses, and execute game actions (`step()`).
* **Observation Representation:**
  * Multi-representation perception: rendered 2D images, raw ASCII text matrices, and programmatic bounding-box segmentation/crop tools to inspect sub-grids.
* **Memory & Context Management:**
  * **"Infinite Play via Message Eviction":** A sliding-window message eviction protocol. When context limits approach capacity, older dialogue turns are dropped while the persistent system prompt and recent action history are preserved.
* **Tool Use & Execution:**
  * Interactive Python REPL. The agent writes arbitrary Python code to inspect pixel differences, run flood-fill algorithms, and formulate moves.
* **Exploration & Planning Policy:**
  * Code-driven trial and error. Multi-pass reasoning loop: fast exploratory passes to probe action affordances followed by deeper strategic execution.
* **Action Budget & Pacing:**
  * Dynamic allocation across the 9-hour Kaggle offline submission limit ($\approx 4.9$ minutes per game across 110 games). Early cutoff on games that fail to yield reward signals.
* **Reported Scores & Transfer Gap:**
  * Public training run score: $\approx 1.60$.
  * Kaggle Milestone 1 Official Leaderboard Score: **1.21%** (initially logged 1.30%, adjusted to 1.21%).
* **Failure Modes & Overfitting Indicators:**
  * Frequent REPL syntax/runtime errors; infinite loops in generated scripts; context amnesia (eviction discards early mechanic discoveries unless manually stored in persistent script variables); prompt overfitting tuned around mechanics prominent in the 25 public games.

---

### Profile 2: Team Reki (2nd Place Milestone 1 Winner — $7,500)
* **Architecture Paradigm:** **"Vision-LLM-as-Policy"** with multi-step action queuing.
* **Base Model & Quantization:** **Gemma-4-31B** (FP8 / AWQ locally served).
* **Inference Engine:** **vLLM** offline server.
* **Agent Loop & Prompting:**
  * Direct observation-to-action policy without a code REPL.
  * Ingests recent board frames as coordinate-labeled rendered images.
  * On each inference step, prompts the model to output a single structured JSON payload with three fields:
    1. `"what_changed"` (visual frame diff analysis)
    2. `"plan"` (short natural language rationale)
    3. `"action_queue"` (an array of 1 to 4 sequential actions)
* **Observation Representation:**
  * Labeled image grids overlaid with coordinate ticks, plus state metadata.
* **Action Queuing & Pacing:**
  * **1-to-4 Action Queue:** Emitting micro-plans of 1–4 actions per LLM call drastically reduces token consumption, cuts inference latency by $\approx 60\%$, and preserves trajectory coherence.
* **Memory Design:**
  * **Episodic Reflection Memory:** Summarizes progress and tested hypotheses every $\sim 10$ steps, compressing historical transitions into compact natural language notes.
* **Tool Use & Heuristics:**
  * **JSON Self-Repair:** Heuristic regex and JSON fixer to repair malformed LLM outputs.
  * **Legal-Action Constraint Checker:** Prevents issuing out-of-bounds or invalid actions.
  * **NumPy Click-Heuristic Fallback (Zero-GPU):** When the agent is exploring, a fast NumPy heuristic identifies small, rare-colored, button-like connected components and probes them before spending expensive LLM calls.
  * **Dead-Signature Detector:** Prunes repeated futile actions (e.g., walking into a solid wall).
* **Reported Scores & Transfer Gap:**
  * 2nd Place Milestone 1 ($\approx 1.05\%$ LB score).
* **Failure Modes & Overfitting Indicators:**
  * Visual spatial hallucination (misidentifying coordinates on 64×64 grids); heuristic bias (the button-click prior fails completely on maze navigation, gravity puzzles, or painting tasks); dead-signature false positives terminating productive multi-step probes.

---

### Profile 3: Md Boktiar Mahbub Murad — "forge" (3rd Place Milestone 1 Winner — $5,000)
* **Architecture Paradigm:** Vision-Language Policy Agent with strict schema enforcement and modular ablation toggling.
* **Base Model & Quantization:** **Gemma-4-31B** / modular open-weight models.
* **Inference Engine:** **vLLM** local execution pipeline.
* **Key Mechanisms:**
  * Structured frame-diff tracking between $t-1$ and $t$.
  * Strict JSON schema validation with fallback parsing.
  * Modular architecture with environment-variable feature flags (allowing rapid offline ablation of visual overlays, reflection prompts, and action batching).
* **Milestone 1 Standing:** 3rd Place ($5,000).

---

### Profile 4: Kepler & Tycho (NIMI-Research / Wensen Wu, arXiv:2610.00834)
* **Architecture Paradigm:** **Auditable World Models** & **Certify-Then-Replay Protocol**.
* **Base Model:** Frozen Claude Opus 5 / GPT-5.6 Sol / open-weight models.
* **Lineage:** *Tycho* (empiricist observation logger) $\to$ *Kepler* (executable world model synthesizer).
* **Agent Loop & Philosophy:**
  * Does not emit raw environment actions directly.
  * Maintains an explicit, executable Python world model (`world_model.py`), research notes (`notes.md`), and an append-only transaction ledger (`events.jsonl`).
  * Hypotheses are formalized as **Deterministic Moore Machines**.
* **Modular Toolsuite:**
  * `observe.py`: Read-only queries of current raw observation frames.
  * `freequery.py`: Zero-environment-cost state computation.
  * `freebacktest.py`: Regression testing candidate world models against the complete historical transaction log.
  * `bfs.py`: Algorithmic pathfinding (BFS/A*) executed *inside* the verified world model.
  * `freecommit.py`: The single guarded action channel that executes verified plans in the real environment.
* **Dual Verification Gates:**
  1. **Retrospective Transition Checks:** A candidate world model is rejected if it fails to reproduce even 1 historical observed transition in `events.jsonl` (100% regression fidelity required).
  2. **Conditional Prediction Checks:** Before executing any action, the world model must predict state $S_{t+1}$. If actual $S_{t+1}$ diverges, execution halts and triggers model revision.
* **Reported Scores & Compute Cost:**
  * **100.00 RHAE** server-verified across all 25 public ARC-AGI-3 games (181/183 levels completed with median-human efficiency).
  * Actions used: 8,256 environment actions.
  * Tokens consumed: $\approx 858 \text{M}$ tokens ($\approx 97.4\%$ prompt cache hit rate). API cost: $\$777.72$.
* **Critical Evaluation Pitfalls Documented:**
  1. *Source-Code Leakage:* An unisolated query channel accidentally exposed environment source code, yielding an invalid "perfect run" before being caught and patched.
  2. *Control-Group Resilience:* When tools were removed, autonomous agents re-implemented the missing tools inside shell/python sub-shells.
  3. *Masked Planners:* Auto-repair logic mathematically patched transition outputs to pass tests while leaving the core planning logic defective.
  4. *Leaderboard Infeasibility:* Consuming 858M tokens and $\$777$ API compute cannot be run within Kaggle's 9-hour single-GPU offline environment without massive distillation and model compaction.

---

### Profile 5: Memento 3 (Huawei Noah’s Ark Lab & UCL, arXiv:2610.11794)
* **Architecture Paradigm:** **Model-Based Recursive Self-Improvement (RSI) through Reflective Rulebooks**.
* **Base Model:** Frozen Claude Opus / frontier coding models (weights strictly frozen).
* **Persistent Semantic Memory (The Rulebook):**
  * Maintains a natural-language markdown document (`rulebook.md`) detailing working hypotheses of physics, object affordances, obstacle behaviors, and win conditions.
  * Explicitly flags unverified assumptions as "tentative" until verified by empirical interaction.
* **Code Compilation & Zero-Token Planning:**
  * Compiles the natural-language rulebook into an executable Python simulator (`simulator.py`).
  * Plans paths using standard algorithms (BFS, Dijkstra, A*) running inside the local simulator. **Zero LLM inference tokens are burned during planning and trajectory search.**
* **Dual Verification Gates:**
  1. *Semantic Alignment Audit:* Frozen LLM inspects whether the compiled code faithfully represents the natural-language rulebook.
  2. *Cell-Exact Replay Verification:* The compiled code simulator is executed across all historical interaction frames; it must achieve 100% cell-exact prediction accuracy.
* **Reported Score:**
  * **100.0 RHAE** across all 25 public games, completing levels with only **44% of human action counts**.
* **Failure Modes & Bottlenecks:**
  * Brittle when environments exhibit non-deterministic, noisy, or continuous mechanics; compilation latency scales quadratically with history length; heavy frontier model requirement.

---

### Profile 6: Symbolica's Arcgentica (Symbolica AI / Agentica SDK)
* **Architecture Paradigm:** Hierarchical Orchestrator-Subagent with **Recursive Language Models (RLMs)**.
* **Base Model:** Claude 3.5 Sonnet / Opus 4.6 orchestrated via Agentica SDK.
* **Agent Loop:**
  * Top-level orchestrator maintains abstract objectives and task queues.
  * Recursively spawns specialized subagents to write Python test scripts, inspect sub-regions, or search for state transformations.
  * Subagents execute in isolated REPLs and return compressed structured summaries back to the orchestrator, preventing context-window explosion.
* **Performance:**
  * State-of-the-art on ARC-AGI-2 (85.28%) and Day-1 zero-shot $>36\%$ on ARC-AGI-3 public games.
* **Failure Modes:**
  * Significant orchestration token overhead; recursive call depth limits; API latency overhead incompatible with tight per-game execution budgets.

---

### Profile 7: Barada Sahu (Kaggle Leaderboard Rank 3 Contender)
* **Architecture Paradigm:** High-Efficiency Hybrid Policy Agent.
* **Focus:** Extreme action-budget efficiency, aggressive loop pruning, and low-latency local inference.
* **Key Mechanism:** Balances fast rule-based action generation during routine states with targeted LLM intervention only when novel transitions or dead ends occur, preserving Kaggle runtime across all 110 games.

---

## 3. RANKED LIST OF GAME-AGNOSTIC, HIGH-TRANSFER ARCHITECTURAL IDEAS

### Rank 1: Certify-Then-Replay Executable World Modeling (Kepler / Memento 3)
* **The Mechanism:** Maintain an explicit Python simulator (`world_model.py`) separated from real environment execution. Before committing any action:
  1. *Backtest:* Verify the model reproduces 100% of historical recorded transitions.
  2. *Predict:* Simulate the next step; compare predicted vs actual frame.
* **Why it Transfers:** Operates purely on mathematical consistency, completely agnostic to visual themes, colors, or game mechanics. Eliminates hallucinated assumptions.

### Rank 2: 1-to-N Action Plan Queuing (Reki)
* **The Mechanism:** Instead of invoking the LLM every single step, the planner outputs a coherent queue of 1 to 4 actions.
* **Why it Transfers:** Drastically cuts inference latency and token costs by $50\%\text{--}75\%$, preserves multi-step goal momentum, and vastly improves RHAE on routine steps.

### Rank 3: Dual Semantic Memory + Compiled Simulator (Memento 3)
* **The Mechanism:** Decouple reasoning into two tiers:
  * Tier 1: Natural-language markdown rulebook (`rulebook.md`) documenting object roles, affordances, and hypotheses.
  * Tier 2: A fast Python script compiled from the rulebook that runs algorithmic pathfinding (BFS/A*).
* **Why it Transfers:** Allows zero-token, lightning-fast planning for routine movement while reserving expensive LLM calls exclusively for hypothesis updates when unexpected transitions occur.

### Rank 4: Dead-Signature Loop Breaking & Action Masking (Reki / Kepler)
* **The Mechanism:** Track a hash of $(S_t, a_t) \to S_{t+1}$. If an action produces a null transition ($\Delta S = 0$) or returns to an identical state cycle, permanently prune that action from that state signature.
* **Why it Transfers:** Prevents the #1 failure mode on unseen games: infinite repetition loops that exhaust the action budget.

### Rank 5: Multi-Representation Perception with Dynamic Cropping (The Duck)
* **The Mechanism:** Provide the model with both coordinate-aligned ASCII matrices and dynamic bounding-box crops of active sub-regions.
* **Why it Transfers:** Overcomes the spatial blindness of pure text grids while avoiding the coordinate hallucination of downsampled 64×64 vision models.

### Rank 6: Strict Action-Budget Management & Fast Failure Cutoffs (Barada Sahu / Duck)
* **The Mechanism:** Implement strict per-game time and action budgets ($\le 5$ minutes, $\le 60$ exploratory probes). If an agent shows zero progress or score increase within the exploratory window, terminate gracefully and save budget for solvable games.
* **Why it Transfers:** Prevents bimodal "score-zero sinkholes" from depleting the 9-hour Kaggle runtime budget across the 110 evaluation games.
