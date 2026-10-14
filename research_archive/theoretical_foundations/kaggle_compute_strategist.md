# Comprehensive Investigation & Competitive Analysis: ARC-AGI-3 Open-Source Solutions, Compute Budgeting, and RHAE Optimization

## Executive Summary

In the **ARC Prize 2026 (ARC-AGI-3)** Kaggle competition, AI agents encounter turn-based, video-game-like interactive POMDP environments where physical laws, object affordances, and winning conditions must be inferred through interaction. Unlike traditional static benchmarks or classic reinforcement learning environments with millions of trial-and-error episodes, ARC-AGI-3 evaluates agents via **Relative Human Action Efficiency (RHAE)**:

$$S_l = \begin{cases} \min\left(1.15, \left(\frac{h_l}{a_l}\right)^2\right) & \text{if level } l \text{ is won and } a_l \le 5 h_l \\ 0.0 & \text{otherwise} \end{cases}$$

Where $h_l$ is the first-exposure human baseline action count, and $a_l$ is the agent's action count. 

Every live interaction in the environment (including `RESET`!) increments $a_l$. The official Kaggle evaluation budget provides **9 hours (32,400 seconds)** across approximately **135 hidden games** (~240 seconds per game). Top competitors achieve high RHAE by recognizing that **wall-clock time is virtually unpenalized within the container budget, whereas live environment actions incur a catastrophic quadratic penalty**. 

Below is an in-depth investigation of the leading open-source systems, followed by concrete mathematical derivations and actionable architectural recommendations.

---

## 1. Competitive Landscape & Open-Source Solutions

### 1.1 Tufa Labs — "The Duck" (Milestone #1 Winner)
* **Core Philosophy**: "Harness engineering over pure model weights." The Duck proved that surrounding an open-weights model with an expressive agentic harness achieves higher reasoning capability than raw model scale.
* **Python REPL as Universal Interface**: Instead of generating single-token action predictions, the agent interacts with the game inside a live Python REPL. Game states and observation tensors are exposed as Python variables. The model writes Python scripts to inspect objects, compute distances, and output actions.
* **Triple-Channel Perception**:
  1. *Rendered RGB Image*: Global spatial layout and visual cues.
  2. *Raw ASCII Grid*: Precise coordinate mapping and discrete color indices.
  3. *Segmentation Zoom Tool*: Crops bounding boxes around dynamic clusters for detailed inspection.
* **Infinite Play via Eviction**: To prevent context exhaustion over long episodes, The Duck implements a continuous eviction policy, popping the oldest history while preserving the immutable system instructions and recent active turns.

### 1.2 Reki (Milestone #1 Top-3)
* **Core Philosophy**: Constrained vision-policy execution with strong syntactic and operational guards.
* **VLM Engine**: Driven by fine-tuned Gemma-4-31B.
* **Action Constraints & Self-Repair**: Uses strict JSON schema enforcement with automatic self-repair loops, completely preventing invalid action IDs or malformed payloads from reaching the game engine.
* **Action Plan Queue & Reflection Memory**: Rather than single-step reactive generation, Reki outputs a 1–4 action trajectory alongside explicit reflection tokens evaluating whether past actions succeeded or failed.

### 1.3 Barada Sahu (Rank 3 Competitor)
* **Core Philosophy**: Verification-driven execution bursts and divergence protection.
* **Sahu Verification Delta Controller**: Plans multi-step macro bursts (1–4 actions), but annotates each step with an expected post-action topological state hash $\mathcal{H}_{\text{predicted}}(s_{t+k})$.
* **Immediate Rollback on Divergence**: After each step in the burst, the live hash $\mathcal{H}_{\text{live}}(s_{t+k})$ is compared against $\mathcal{H}_{\text{predicted}}$. If any discrepancy or null transition ($\Delta\Omega = 0$) occurs, execution halts immediately and remaining queued actions are purged, preventing multi-step action cascades from compounding live penalties.

### 1.4 Kepler (NeurIPS 2026 / arXiv:2610.00834)
* **Core Philosophy**: "Auditable and Executable World Models via Active Abstraction" (built on Tycho, arXiv:2607.28287).
* **Auditable Hypotheses**: Replaces opaque latent embeddings with human-readable and machine-executable Python world models.
* **Dual Verification Protocol**:
  1. *Retrospective Transition Checks*: Verifies that candidate transition logic reproduces 100% of recorded historical transitions $(s_\tau, a_\tau) \to s_{\tau+1}$.
  2. *Conditional Predictive Checks*: Evaluates whether forward projections maintain conservation laws (mass, color invariance, boundary walls).
* **Evaluation Integrity**: Highlighted benchmark failure modes (accidental source leakage, brittle heuristic crutches) and demonstrated that auditable programmatic world models achieve robust generalization.

### 1.5 Memento 3 (UCL / Huawei Noah's Ark Lab — arXiv:2610.11794)
* **Core Philosophy**: "Code as Model" and Recursive Self-Improvement (RSI) using frozen LLMs.
* **Dual World Model Architecture**:
  1. *Reflective Rulebook (`world_model.md`)*: Semantic Markdown document maintaining evolving hypotheses regarding physics, objects, rules, and win conditions.
  2. *Executable Realization*: Compiled Python module containing `simulate_step(grid, action, state) -> (next_grid, next_state)`.
* **Cell-Exact Replay Verification**: A zero-tolerance gate requiring candidate transition code to achieve cell-exact (pixel-perfect) reproduction of all observed historical transitions before being promoted to the planner.
* **Solved-Level Memory (Siriki M85)**: Upon winning a level, verified physical invariants and winning action tails are archived into persistent memory. Subsequent levels inherit this verified core, eliminating re-exploration.
* **Execution as Code**: Once the world model is verified, the agent synthesizes a deterministic controller (A*, MCTS, feedback policy) that runs at native execution speed with zero LLM API latency or live exploratory waste.
* **Benchmark Result**: Cleared all 25 public ARC-AGI-3 games with a mean RHAE of 100.0, using only 44% of human baseline action counts.

### 1.6 Symbolica Arcgentica (Symbolica AI)
* **Core Philosophy**: Type-safe runtime object orchestration via the Agentica SDK.
* **Persistent Runtime Environment**: Models interact inside a stateful Python runtime where variables, helper functions, and visual tensors persist across multiple prompt turns without serialization overhead.
* **Hierarchical Orchestration**: A central orchestrator delegates discrete sub-problems (connected component analysis, invariant detection, trajectory optimization) to specialized subagents, which return compressed semantic summaries. This prevents context bloat and maintains high signal-to-noise ratios.

---

## 2. Preventing Live Action Waste via Offline Computation

### 2.1 The Mathematics of Action Waste
The sensitivity of the level RHAE score $S_l$ with respect to live actions taken $a_l$ reveals why exploratory trial-and-error is lethal:

$$\frac{dS_l}{da_l} = \frac{d}{da_l} \left[ \left(\frac{h_l}{a_l}\right)^2 \right] = -2 \frac{h_l^2}{a_l^3}$$

This cubic sensitivity profile demonstrates:
1. **At the human baseline ($a_l = h_l$)**:
   $$\left. \frac{dS_l}{da_l} \right|_{a_l = h_l} = -\frac{2}{h_l}$$
   If $h_l = 10$, a single wasted exploratory action costs **20% of the entire level score**.
2. **If $a_l = 2 h_l$**:
   The score collapses to $S_l = (h_l / 2h_l)^2 = 0.25$ (a 75% score destruction).
3. **If $a_l > 5 h_l$**:
   The score drops discontinuously to **0.0**, regardless of whether the level is ultimately completed.

### 2.2 Offline Simulation as a Zero-Penalty Sandbox
Offline computation allows an agent to trade milliseconds of CPU/GPU execution for guaranteed action parsimony:
* **In-Memory Forward Rollouts**: Once a candidate transition function `simulate_step` is compiled, exploring $10^5$ trajectory branches in Python/C++ costs $0$ live environment actions.
* **$\mathcal{O}(1)$ Dead-Signature Culling**:
  By encoding state-action contexts into 64-bit packed bitmasks:
  $$\text{Signature} = (\text{Action} \ll 52) \mid (\text{Color} \ll 48) \mid (\text{Mass} \ll 32) \mid (\text{ContextHash})$$
  Any action that produces a null state transition ($\Delta\Omega = 0$, such as bumping into a static boundary wall or selecting an unreactive cell) is permanently cached in an instantaneous bitset. Offline planners cull dead actions before generating a single candidate step.
* **Sahu Verification Delta Rollback**:
  When dispatching a macro-action burst $(a_1, a_2, \dots, a_k)$, the agent checks $\mathcal{H}_{\text{live}}(s_{t+i}) \equiv \mathcal{H}_{\text{predicted}}(s_{t+i})$. If step 1 fails to yield the predicted state, execution stops immediately. The agent wastes at most 1 live action instead of $k$ compounding errors.

---

## 3. Time-Budgeting Algorithms (System Pacer) and Level Allocation

### 3.1 The Global Budget Envelope
* Total Kaggle evaluation time: $T_{\text{total}} = 32,400\text{ s}$ (9.0 hours).
* Hidden evaluation game set: $N_{\text{games}} \approx 135$ games.
* Nominal budget per game:
  $$\bar{T}_{\text{game}} = \frac{32,400}{135} = 240.0\text{ seconds/game}$$
* Games consist of $L \in [3, 7]$ levels (mean $L \approx 4.5$).

### 3.2 The Asymmetric Level-Weighting Formula
The official ARC-AGI-3 game score is **linearly weighted by the 1-indexed level number**:

$$E_{\text{game}} = \frac{\sum_{l=1}^{L} l \cdot S_l}{\sum_{l=1}^{L} l} = \frac{2}{L(L+1)} \sum_{l=1}^{L} l \cdot S_l$$

The effective weight of level $l$ is:

$$w_l = \frac{2l}{L(L+1)}$$

For a representative 5-level game ($L = 5$):
* **Level 1**: $w_1 = \frac{1}{15} \approx \mathbf{6.67\%}$
* **Level 2**: $w_2 = \frac{2}{15} \approx \mathbf{13.33\%}$
* **Level 3**: $w_3 = \frac{3}{15} \approx \mathbf{20.00\%}$
* **Level 4**: $w_4 = \frac{4}{15} \approx \mathbf{26.67\%}$
* **Level 5**: $w_5 = \frac{5}{15} \approx \mathbf{33.33\%}$

**Crucial Insight**: Higher levels account for **93.33%** of the total game score. Level 1 contributes only **6.67%**.

### 3.3 The Asymmetric System Pacer Policy
Because the physical laws, color roles, and mechanics discovered in Level 1 are invariant across all levels of that game, top systems allocate time using an asymmetric exploration/exploitation schedule:

```
[ Game Start (240s Total Budget) ]
  │
  ├── Level 1: Model Induction & Hypothesis Synthesis (80s - 120s Compute)
  │     ├── High live action tolerance (exploration cost bounded to ≤ 6.67% of game)
  │     ├── Heavy LLM code generation + AST Sandbox compilation
  │     ├── Cell-Exact Replay Verification on all Level 1 transitions
  │     └── Solved-Level Memory (Siriki M85) distillation
  │
  └── Levels 2..L: Pure Symbolic Exploitation (20s - 35s Compute / Level)
        ├── 0 live exploration actions
        ├── High-speed A* / Epistemic-Damped MCTS over verified world model
        ├── Near-optimal trajectories: a_l ≤ h_l  ==>  S_l ≥ 1.00
        └── Captures maximum score across 93.3% of game weight
```

### 3.4 Dynamic Pacing Modes & Franzen M2 Tail-Fade
To guard against timeouts across varying puzzle difficulties, System Pacer dynamically tracks remaining time $T_{\text{rem}} = T_{\text{total}} - t_{\text{elapsed}}$ and amortized per-game time $\tau = T_{\text{rem}} / N_{\text{rem\_games}}$:

| Remaining Budget Fraction | Amortized Time $\tau$ | Pacer Mode | Behavior |
|---|---|---|---|
| $> 40\%$ | $\tau \ge 120\text{s}$ | **FULL_COGNITIVE** | Deep LLM code synthesis, reflective rulebook generation, full MCTS rollouts. |
| $15\% - 40\%$ | $60\text{s} \le \tau < 120\text{s}$ | **BALANCED_SEARCH** | Template-driven code generation, lightweight MCTS, A* pathfinding. |
| $5\% - 15\%$ | $20\text{s} \le \tau < 60\text{s}$ | **RAPID_GREEDY** | Deterministic CCL heuristics, reactive greedy stepping, 0 LLM calls. |
| $< 5\%$ | $\tau < 20\text{s}$ | **EMERGENCY_SKIP** | Immediate graceful exit to ensure `submission.parquet` serializes safely. |

* **Hard Level Timeout Ceiling**: Regardless of mode, each level is capped at $T_{\text{level}} \le \min(35.0\text{s}, \tau / 4.0)$, eliminating infinite solver hangs on pathological puzzles.

---

## 4. Why Spending Compute Time Per Step Dramatically Improves RHAE

### 4.1 Test-Time Compute Scaling & Geodesic Convergence
In an interactive puzzle POMDP, the state space graph $\mathcal{G} = (\mathcal{S}, \mathcal{A})$ contains exponentially branching non-optimal paths. 

Let $C$ denote the offline compute (rollout count / search tree expansions) invested per decision:
* A greedy or reactive agent ($C \to 0$) samples actions according to immediate perceptual heuristics, accumulating stochastic exploratory detours. The expected path length grows as a sub-optimal random walk:
  $$\mathbb{E}[a_l \mid C \to 0] = h_l + \mathcal{O}(D_{\text{branching}})$$
* An agent performing deep offline search (MCTS / A* / symbolic constraint satisfaction) expands the search tree until it identifies the true geodesic (shortest path) $a_l^* \le h_l$. By the Chernoff-Hoeffding bound on MCTS selection error:
  $$P(\text{suboptimal action} \mid C) \le K \exp(-\beta C)$$
  As compute $C$ increases, the probability of executing a suboptimal live action decays exponentially.

### 4.2 The Quadratic Payoff of Geodesic Search
Consider a level where the human baseline is $h_l = 8$:
* **Fast Heuristic Agent (0.01s compute/step)**:
  Takes $a_l = 24$ actions to reach the goal.
  $$S_l = \left(\frac{8}{24}\right)^2 = \left(\frac{1}{3}\right)^2 \approx \mathbf{0.111}$$
* **Deep Offline Planner (2.0s compute/step)**:
  Spends 16 seconds of CPU time running A* / MCTS over the verified world model, finding the optimal 8-step geodesic ($a_l = 8$).
  $$S_l = \left(\frac{8}{8}\right)^2 = \mathbf{1.000}$$

**Conclusion**: Spending 16 seconds of container wall-clock time yields a **9× (800%) improvement in RHAE score**. Because the 9-hour Kaggle sandbox provides $\approx 240$ seconds per game, hoarding compute time is counterproductive; spending it offline directly optimizes the exponent in RHAE.

### 4.3 Context-Window Hysteresis (Franzen Hysteresis)
When utilizing LLMs for offline planning, generation latency is often dominated by prompt prefill. If an agent naively trims context on every turn, it invalidates the GPU's KV cache (Radix cache), slowing token generation.

The **Franzen Hysteresis** mechanism solves this:
1. **High Watermark (e.g., 118,000 tokens)**: Context is permitted to accumulate without any pruning, ensuring a 100% warm KV prefix cache across consecutive REPL thoughts.
2. **Blockwise Drain to Low Watermark (e.g., 59,000 tokens)**: When the high watermark is breached, a single blockwise eviction trims older conversational turns while preserving:
   * The immutable environment prompt.
   * The persistent helper function registry (e.g., CCL, flood-fill).
   * The active reflective rulebook (`world_model.md`).
3. **Outcome**: The agent spends $>90\%$ of its compute budget on forward inference and Python execution rather than repeated prompt processing.

---

## 5. Actionable Architectural Recommendations

To achieve top-tier RHAE within the Kaggle constraints, competitive systems should be structured according to the following five-stage architecture:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   HYBRID ARCHITECTURE FOR ARC-AGI-3                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [ Stage 1: Zero-NN Dual-Channel Perception ]                          │
│    - Fast C++/NumPy Connected Component Labeling (CCL)                 │
│    - Dynamic border timer bar masking (isolates internal board)        │
│                                                                        │
│  [ Stage 2: Reflective Rulebook & Executable World Model ]             │
│    - Dual representation: Structured Markdown + Python simulate_step   │
│    - Cell-Exact Replay Verification Gate (100% historical accuracy)    │
│    - Solved-Level Memory (Siriki M85) for cross-level invariant reuse  │
│                                                                        │
│  [ Stage 3: Verification Delta Controller & Macro-Queue ]              │
│    - Dispatches 1..4 action bursts with predicted state hashes         │
│    - Halts immediately upon hash divergence or null transition         │
│                                                                        │
│  [ Stage 4: Epistemic-Damped UCT MCTS & A* Engine ]                    │
│    - UCT(s,a) = Q + C_p * exp(-lambda*V(s)) * sqrt(ln N / (1+n))       │
│    - 64-bit Bitwise Dead-Signature Culling (O(1) null-transition gate)  │
│    - Semantic Evaluator: D4 Symmetry, Enclosure, Sinkhorn Distance     │
│                                                                        │
│  [ Stage 5: Asymmetric System Pacer ]                                  │
│    - Allocates 40%-50% of game compute to Level 1 model induction      │
│    - Enforces 20s-35s fast symbolic exploitation on Levels 2..L        │
│    - Franzen M2 Tail-Fade (Cognitive -> Balanced -> Rapid -> Emergency)│
│    - Hard 35s per-level timeout gate                                   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Checklist for Implementation:
1. **Never explore in Levels $l \ge 2$**: Treat Level 1 as the exclusive learning phase. Inherit verified physics into higher levels via Solved-Level Memory.
2. **Gate execution behind Cell-Exact Replay**: Never execute multi-step plans from a world model that cannot reproduce past transitions with zero pixel discrepancy.
3. **Use the Sahu Verification Delta**: Always pair macro-actions with anticipated state hashes to halt before bad decisions compound.
4. **Cull dead actions in $\mathcal{O}(1)$**: Maintain a bitset of null transitions ($\Delta\Omega = 0$) to eliminate wasteful moves into walls.
5. **Pace against the global clock**: Use a dynamic pacer that adjusts solver modes according to remaining time, ensuring all ~135 games are attempted while completing well within the 9-hour limit.