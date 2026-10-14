# FAILURE TAXONOMY: SYSTEMATIC DIAGNOSIS & GENERAL CAPABILITY GAPS

## 1. Overview
In ARC-AGI-3, describing an error as "the model failed" or "game X was lost" provides zero engineering utility. Every unsuccessful level or sub-optimal trajectory must be diagnosed into standardized, capability-based failure modes.

This document establishes the official 15 failure categories, their empirical signatures, root causes, and general architectural remedies.

---

## 2. Standardized 15 Failure Categories

| Code | Failure Category | Empirical Signature | Root Cause | Proposed General Remedy | Validation Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F01** | **Perception Failure** | Misaligned discrete grid coordinates; merged distinct sprites; bounding box drift. | Dense tile patterns mimicking background; non-standard sprite scaling (e.g., $5\times 5$, $7\times 7$ non-power-of-2). | Multi-scale morphological boundary detection; frame border perimeter statistics. | In Progress |
| **F02** | **Object-Role Failure** | Agent tries to move an immobile wall; treats a deadly hazard as an avatar or collectible. | Speculative or static role assignment based on single-frame appearance. | Dynamic affordance induction: classify roles based on movement deltas during active interaction. | Validated in v0.2 |
| **F03** | **Action-Semantics Failure** | Agent executes Action 1 expecting translation, but action triggers rotation or state toggle. | Hardcoded cardinal movement assumptions without verifying action effects. | Active Action-Semantics Probing: sample non-fatal actions on Level 1 step 1-3 to classify action type. | Planned (Track D) |
| **F04** | **Causal-Model Failure** | Predicted post-state differs from actual environment transition; unmodeled chain reactions. | Cellular automata or physics model fails to account for secondary displacements (e.g., falling gravity). | Verification Delta: compute $S_{t+1}^{actual} \oplus S_{t+1}^{predicted}$; trigger instant replanning on mismatch. | In Progress |
| **F05** | **Memory Failure** | Agent rediscovers known rules on Level 2; repeats identical actions that failed on Level 1. | Epistemic state wiped between levels or steps. | Solved-Level Memory: persist invariant physical laws, affordances, and causal rules across levels. | Validated in v0.2 |
| **F06** | **Context Failure** | Agent hallucinates previous objects or loops because critical transition history was truncated. | Naive FIFO history pruning overflowing context window. | Structured Context Compaction: retain permanent world model + action-effect ledger + recent 3-5 diffs. | Designed (Track C) |
| **F07** | **Exploration Failure** | Agent burns 30+ actions random-walking without changing state or resolving uncertainty. | Lack of information-gain objective; blind testing without hypothesis tracking. | Epistemic Curiosity & Dead-Signature Culling: filter provably null transitions in $O(1)$. | Validated in v0.2 |
| **F08** | **Planning Failure** | Rules and goals are understood, but agent takes a circuitous 50-step path or traps a box. | Weak search heuristic; lack of dead-lock detection in push puzzles. | Multi-tier symbolic planners (A*, BFS, Sokoban IDA* / macro-actions) invoked as exact tools. | Partially Validated |
| **F09** | **Goal-Inference Failure** | Agent navigates the arena endlessly without triggering win condition or collecting target items. | Win condition is unstated and differs from simple exit-reaching (e.g. topological enclosure). | Invariant Goal Induction: test visual invariants (color reduction, centroid alignment, parity matching). | Planned (Track D) |
| **F10** | **Dynamic-Environment Failure** | Agent gets killed by moving hazards; conveyor belts or falling blocks disrupt planned trajectory. | World model assumes static transition $T(s, a) \to s'$ without autonomous environment ticks. | Decoupled dynamic entity tracking: model environmental delta $\Delta_{env}$ independently of agent action $a$. | Open Gap |
| **F11** | **Hidden-State Failure** | Agent fails memory matching games or fog-of-war puzzles; cannot track card flips. | Agent does not store concealed identities or phase states. | Explicit Latent State Ledger: track reveal history, card positions, and card values in durable memory. | Partially Validated |
| **F12** | **Efficiency Failure (RHAE)** | Game is WON, but agent used $3\times$ human baseline actions, yielding low RHAE score (<20%). | Excessive exploratory actions on higher levels ($L \ge 2$); sub-optimal pathing. | Zero-exploration policy on Level $\ge 2$: strictly execute inferred policy directly to goal. | Focus for Sprints 3-5 |
| **F13** | **Runtime / Timeout Failure** | Notebook crashes or times out after 9 hours; individual game exceeds 10 minutes. | Unbounded MCTS rollouts; slow Python loops; LLM autoregressive token explosion. | Dynamic System Pacer: hard budget cap per game ($T \le 240$s); fallback to fast heuristic on timeout. | Validated in starter |
| **F14** | **Harness Bug** | Unhandled exception; action index out of bounds; numpy dimension mismatch. | Fragile type assertions; unhandled `available_actions` subsets. | Robust Defensive Harness: legal action normalization and try-except recovery fallbacks. | Validated in v0.2 |
| **F15** | **Overfitting Risk** | Improvement on public games yields zero or negative transfer on held-out games. | Game-specific color, coordinate, or fingerprint heuristics. | Anti-overfitting gating: test all changes across held-out Cohort 2 & 3 before merging. | Active Protocol |

---

## 3. Failure Diagnosis Workflow
When an evaluation run completes:
1. Generate `benchmark_results.json`.
2. For every game where `is_won == False` or `rhae_score < 0.40`:
   - Inspect the action count trajectory.
   - Assign primary and secondary Failure Codes (e.g., `F08 + F12`).
   - Log the root mechanism gap in `EXPERIMENT_LOG.md`.
