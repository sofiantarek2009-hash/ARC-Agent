# Research Report: Goal Deduction and Objective Function Formulation for ARC-AGI-3 under Unstated Objectives

**Target Investigation**: Autonomous Goal Inference, Topological Entropy Heuristics, and Solved-Level Memory Transfer in ARC-AGI-3.  
**Theoretical Paradigm**: Epistemic Active Probing, Algorithmic Information Theory, Discrete Optimal Transport, and Inductive Logic Programming (Siriki M85).

---

## Executive Summary

In ARC-AGI-3, agents face an extreme epistemic challenge: environments provide **no explicit reward function**, **no linguistic specification**, and **no stated goal**. The agent perceives only raw discrete color grids $G_t \in \{0, \dots, 15\}^{H \times W}$, a discrete action space $\mathcal{A} \subseteq \{1, \dots, 7\}$, and binary episode status signals (`PLAYING`, `WIN`, `GAME_OVER`).

To solve levels with **Relative Human Action Efficiency (RHAE)** approaching or exceeding 1.0 (the 1.15 cap), an agent cannot afford brute-force reinforcement learning or unguided random walks. This report formulates the mathematical and algorithmic solutions to the three core challenges:
1. **Level 1 Goal Deduction**: Inferring the hidden win condition via Spelke core knowledge priors, kinematic response inversion, spatial HUD disentanglement, and Bayesian hypothesis updating.
2. **Topological Shannon Entropy as a Universal Objective**: Formulating $H(G) = H_{\text{color}} + \alpha \cdot H_{\text{spatial}} + \beta \cdot H_{\text{degree}}$ as an intrinsic dissipation metric where $\Delta H \le -\epsilon$ acts as a universal reward signal guiding the agent toward structured, low-entropy attractor ground states.
3. **Solved-Level Memory Transfer (Siriki M85 Paradigm)**: Distilling the causal win transition $(s_{T_k-1}, a_{T_k-1}) \to s_{T_k}$ into a verified invariant predicate $\phi^*$, and executing geodesic shortest-path solvers on Levels $2 \dots N$ to achieve optimal human-parity action efficiency.

---

## 1. Level 1 Win Condition Inference (Completely Unstated Goal)

### 1.1 The Epistemic Dilemma & Core Knowledge Decomposition
At $t=0$ on Level 1, the agent observes an initial grid $s_0$. The reward function $R(s, a)$ is identical to zero for all non-terminal states. To infer the goal without wasting action budget, the agent deploys a **Zero-Parameter Perception Module (CCL)** coupled with Chomsky/Spelke core knowledge priors:
- **Objectness & Cohesion**: The grid is partitioned into discrete connected components $\mathcal{C} = \{C_1, \dots, C_m\}$ using 4- or 8-connectivity over color equivalence classes:
  $$C_i = \{ (r, c) \in \mathbb{Z}^2 \mid G(r, c) = c_i, \; \text{connected to } (r_0, c_0) \}$$
- **Kinematic Response Inversion (Avatar Identification)**:
  The agent executes exploratory directional micro-probes ($a \in \{1, 2, 3, 4\}$). The avatar component $C_{\text{avatar}}$ is identified as the unique entity whose centroid coordinates $\mathbf{x} = (\bar{r}, \bar{c})$ satisfy:
  $$\frac{\partial \mathbf{x}(C_i)}{\partial a} \approx \Delta_a \quad \text{where } \Delta_1 = (-1, 0), \; \Delta_2 = (1, 0), \; \Delta_3 = (0, -1), \; \Delta_4 = (0, 1)$$
  If no component shifts under $\mathcal{A}_{1..4}$, the environment is categorized as a pure click/puzzle environment dominated by $\text{ACTION6}(x, y)$.

### 1.2 The Four Primary ARC-AGI-3 Win Condition Archetypes

#### Archetype A: Centroid Reachability & Kinematic Navigation
- **Observation Characteristics**: Avatar $C_{\text{avatar}}$ present; static collidable walls $C_{\text{wall}}$ (colors 3, 4, 8); salient compact target centroids $C_{\text{target}}$ (colors 11 [cyan/light blue], 14 [magenta], 7 [orange]).
- **Goal Deduction Mechanism**:
  1. *Centroid Extraction*: Compute centroids $\mathbf{x}_k = \frac{1}{|C_k|} \sum_{(r, c) \in C_k} (r, c)$.
  2. *Topological Attraction Predicate*:
     $$\phi_{\text{reach}}(s) \iff \|\mathbf{x}(C_{\text{avatar}}) - \mathbf{x}(C_{\text{target}})\|_1 \le \delta_{\text{touch}}$$
  3. *Sequential Waypoints & Detours* (e.g., `tw01`, `bp01`, `va03`): If touching intermediate targets (color 10, 12) causes them to mutate color or disappear without triggering `WIN`, they are pushed to a sequential waypoint stack $\mathcal{W} = [w_1, \dots, w_k, \text{Goal}]$.

#### Archetype B: HUD / Preview Card Disentanglement & Template Matching
- **Observation Characteristics**: Distinct structural separation between the outer perimeter/HUD strip ($r < 4$ or $r > 60$, or top header $r \le 8$) and the interior game canvas.
- **HUD Disentanglement Operator**:
  $$\text{Grid} = G_{\text{interior}} \oplus G_{\text{HUD}}, \quad \text{where } G_{\text{interior}} = G[4:60, 4:60]$$
- **Cross-Information Matching** (e.g., `sq01` Sequencing, `rs01` Rule Switcher, `pt01` Pattern Match):
  The HUD displays a sequence card or preview template $S_{\text{HUD}}$. The goal predicate is formulated as maximum normalized mutual information or zero Hamming distance between active canvas subregions and $S_{\text{HUD}}$:
  $$\phi_{\text{match}}(s) \iff \mathcal{D}_{\text{Hamming}}(G_{\text{target\_region}}, S_{\text{HUD}}) = 0$$
  When a candidate interior entity matching the HUD active token is manipulated (via click or touch), HUD advancement confirms the sequencing hypothesis.

#### Archetype C: Tile Clearing, State Toggling, and Parity Solvers
- **Observation Characteristics**: Regular matrix of hidden tiles (e.g., color 3 backing in `mm01..mm04`) or binary toggle boards (`lo01`, `lo02`, `dm01`).
- **Goal Deduction Mechanism**:
  1. *Lattice Grid Induction*: Calculate tile dimension $ts = \text{round}(\sqrt{\text{Area}})$ and grid origin $(ox, oy)$ to extract canonical tile centers.
  2. *Color Revelation & Matching Size Deduction*:
     Clicking tile $\mathbf{x}_i$ reveals hidden color $c_i$. The agent tracks active flipped tiles in round $\mathcal{F}$. If $|\mathcal{F}| = 2$ causes tiles to clear (turn to background 0 or permanent revealed state), match size $k=2$; if 3 identical colors are required (`mm02`, `mm03`), $k=3$.
  3. *Linear Toggle Invariance (GF(2) Gaussian Elimination)*:
     For Lights Out / toggling puzzles, clicking coordinate $\mathbf{x}$ applies a linear toggle operator over $\mathbb{F}_2$:
     $$\mathbf{A} \mathbf{x} \equiv \mathbf{b} \pmod 2$$
     The unstated goal is deduced as the zero vector $\mathbf{b} = \mathbf{0}$ (clearing all lights) or target pattern vector.

#### Archetype D: Topological Routing & Source-to-Sink Connectivity
- **Observation Characteristics**: Static emitter source $v_s$ (e.g., color 14 battery/pump), sink terminals $v_t \in \mathcal{T}$ (e.g., color 11 lamps/drains), and clickable intermediate conduits/relays (color 10, pipes in `pu01`, relays in `rp01..rp04`).
- **Goal Deduction Mechanism**:
  1. The agent identifies source and sink as endpoints having boundary degree 1 or unique color tags.
  2. The action affordance reveals that $\text{ACTION6}$ toggles conduit connectivity matrix $\mathbf{A}_{\text{relays}}$, while $\text{ACTION5}$ propagates a flow/pulse along connected components.
  3. The deduced goal is the **Minimum Steiner Tree / Connected Component Closure**:
     $$\phi_{\text{connect}}(s) \iff \forall t \in \mathcal{T}, \; \exists \text{ path } \pi(v_s, t) \subseteq \mathcal{G}_{\text{active}}$$

### 1.3 Active Epistemic Probing with Dead-Signature Culling
To prevent action exhaustion during goal inference, the agent maintains an $O(1)$ **Dead-Signature Filter**:
$$\text{sig}(a, s) = \text{Hash}\big(a, \; \text{mass}(s), \; \text{color}(s), \; \text{ContextHash}(s)\big)$$
If transition $(s, a) \to s'$ produces zero delta ($\|s' - s\|_0 = 0$), $\text{sig}(a, s)$ is permanently marked as dead. This eliminates redundant exploration of dead walls and unresponsive clicks, guaranteeing that all spent actions yield non-zero epistemic information gain:
$$IG(a) = H(P(\Phi)) - \mathbb{E}_{s'}[H(P(\Phi \mid s', a))]$$

---

## 2. Topological Shannon Entropy $H(G)$ as a Universal Objective Function

When no extrinsic reward is provided by the environment, how can an agent distinguish between productive actions and degenerate behaviors? The answer lies in **Topological Shannon Entropy Minimization**.

### 2.1 Formal Definition of Structural Shannon Entropy

The structural entropy of a grid state $G \in \{0, \dots, C-1\}^{H \times W}$ is formulated as:
$$H(G) = H_{\text{color}}(G) + \alpha \cdot H_{\text{spatial}}(G) + \beta \cdot H_{\text{degree}}(G)$$
where $\alpha \approx 0.5$ (or normalized weights $w_{\text{color}}=0.4, w_{\text{spatial}}=0.3, w_{\text{degree}}=0.3$).

```
                      GRID STATE OBSERVATION G
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
 1. Color Entropy        2. Spatial Entropy       3. Boundary Entropy
     H_color                 H_spatial                H_degree
  -Σ p(c) log2 p(c)      -Σ p(d_k) log2 p(d_k)    -Σ p(deg) log2 p(deg)
 (Palette purity /      (Dispersion of mass /    (Perimeter smoothness /
  fragment clearance)     clustering order)        loop enclosure)
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                     COMPOSITE FUNCTIONAL H(G)
                  H(G) = H_color + α·H_spatial + β·H_degree
                                 │
                     TRANSITION REWARD FUNCTION
               R(s, a, s') = max(0, -ΔH) · I[ΔH ≤ -ε]
```

#### 1. Color Distribution Entropy $H_{\text{color}}$
Measures the diversity and fragmentation of active colors on the interior canvas (excluding canvas background $c=0$):
$$p(c) = \frac{\sum_{r, c} \mathbb{I}[G(r, c) = c]}{\sum_{r, c} \mathbb{I}[G(r, c) \neq 0]}, \quad c \in \{1, \dots, 15\}$$
$$H_{\text{color}}(G) = - \sum_{c: p(c) > 0} p(c) \log_2\big(p(c) + 10^{-12}\big)$$
- *Physical Meaning*: Elimination of puzzle pieces, matching memory cards, or painting cells to a uniform target palette monotonically drives $H_{\text{color}} \to 0$.

#### 2. Spatial Dispersion Entropy $H_{\text{spatial}}$
Measures the spatial scattering of object centroids relative to the geometric center of mass:
Let $\{o_1, \dots, o_K\}$ be the extracted connected components with centroids $\mathbf{x}_k = (\bar{r}_k, \bar{c}_k)$ and global centroid $\bar{\mathbf{\mu}} = \frac{1}{K} \sum_{k=1}^K \mathbf{x}_k$.
The normalized dispersion distribution is:
$$p_k = \frac{\|\mathbf{x}_k - \bar{\mathbf{\mu}}\|_2}{\sum_{j=1}^K \|\mathbf{x}_j - \bar{\mathbf{\mu}}\|_2}$$
$$H_{\text{spatial}}(G) = - \sum_{k: p_k > 0} p_k \log_2\big(p_k + 10^{-12}\big)$$
Alternatively, under the $K \times K$ spatial partition formulation:
$$p_{i, j} = \frac{\text{Mass}(B_{i, j})}{\sum \text{Mass}(B)}, \quad H_{\text{spatial}}(G) = - \sum_{i, j} p_{i, j} \log_2 p_{i, j}$$
- *Physical Meaning*: High entropy reflects scattered shards and chaotic obstacle arrangements. Low entropy reflects assembled objects, consolidated blocks, and aligned target configurations.

#### 3. Boundary Adjacency Degree Entropy $H_{\text{degree}}$
For each cell, the boundary degree $d(r, c) \in \{0, 1, 2, 3, 4\}$ counts orthogonal neighbors with mismatching colors:
$$d(r, c) = \sum_{(dr, dc) \in \mathcal{N}_4} \mathbb{I}[G(r+dr, c+dc) \neq G(r, c)]$$
$$H_{\text{degree}}(G) = - \sum_{d=0}^4 p(d) \log_2\big(p(d) + 10^{-12}\big)$$
- *Physical Meaning*: Penalizes noisy, jagged, high-frequency contours; rewards clean contiguous boundaries, geometric compact hulls, and closed topological loops.

### 2.2 Why Entropy Dissipation Governs Goal-Seeking
The core principle governing ARC-AGI-3 is that **human puzzle design is fundamentally negentropic**:
1. **The Ground State Hypothesis**: A puzzle starts in a disordered, dislocated, or incomplete microstate. The completion condition is an ordered "ground state" or low-entropy attractor characterized by symmetry ($D_4$ invariance), compact grouping, or complete clearance.
2. **Intrinsic Transition Reward**:
   $$\Delta H = H(s_{t+1}) - H(s_t)$$
   A transition is intrinsically rewarded if and only if it exhibits measurable negentropic progress:
   $$R_{\text{intrinsic}}(s_t, a_t, s_{t+1}) = \begin{cases} -\Delta H + \kappa & \text{if } \Delta H \le -\epsilon \\ 0 & \text{otherwise} \end{cases}$$
3. **Equivalence with Discrete Optimal Transport (Sinkhorn-Knopp $W_2$)**:
   When objects undergo fusion (e.g., merging 4 shards into 1 central shape) or move toward target pads, the 2-Wasserstein distance $W_2(\mu_t, \mu_{\text{target}})$ to the target measure decreases in direct correlation with the drop in spatial entropy.

### 2.3 Safeguards Against Pathological Collapse (The Trivial Erasure Dilemma)
A naive entropy minimizer might attempt to clear every cell to background color 0, achieving $H(G) = 0$ degenerately. The Project Override architecture prevents this via:
- **Mass Conservation Priors**: State transitions that destroy avatar entities or critical key colors incur a penalty of $-\infty$ (or $-2.0$ dead penalty).
- **Epistemic-Damped UCT Selection**: MCTS penalizes states where epistemic variance increases:
  $$\text{UCT}(s, a) = Q(s, a) + C_p \cdot \exp\big(-\lambda V(s)\big) \sqrt{\frac{\ln N(s)}{1 + n(s, a)}}$$
- **Dead-Signature Pruning**: Actions that produce no effect or produce catastrophic irreversible boundary loss are pruned at expansion time.

---

## 3. Solved-Level Memory: Mathematical Formulation of Cross-Level Transfer

Once Level 1 is solved, the transition $(s_{T_1-1}, a_{T_1-1}) \to s_{T_1}$ that triggered `WIN` or `levels_completed = 1` provides ground truth verification of the underlying physical mechanics and win conditions.

```
 LEVEL 1: EXPLORATORY SOLVE
 ┌──────────────────────────────────────────────┐
 │  [Exploration & Dead Probing]  ──►  [Tail]   │ ──► WIN Observed!
 └──────────────────────────────────────┬───────┘
                                        ▼
                  SOLVED-LEVEL MEMORY (SIRIKI M85)
 ┌─────────────────────────────────────────────────────────────┐
 │ 1. Extract Terminal Tail: τ_tail = (a_{T-k}, ..., a_{T-1})  │
 │ 2. Synthesize Invariant Rulebook Predicate: φ*             │
 │    - Dynamic kinematics (tile_step, sliding ice, push)     │
 │    - Goal color & role identity (targets, switches, sinks)  │
 └──────────────────────────────┬──────────────────────────────┘
                                ▼
 LEVEL 2..N: GEODESIC EXECUTION (OPTIMAL HUMAN PARITY)
 ┌─────────────────────────────────────────────────────────────┐
 │ If ContextHash(s_0^{k+1}) == ContextHash(s_0^k):            │
 │   - Execute deterministic instant replay in O(1) actions.   │
 │ Else:                                                       │
 │   - Instantiate explicit objective J(π) targeting φ*.       │
 │   - Zero exploration: execute A* / BFS geodesic path.       │
 │   - RHAE = min(1.15, (Baseline_human / N_AI)^2) >= 1.0!     │
 └─────────────────────────────────────────────────────────────┘
```

### 3.1 Trajectory Decomposition & The Winning Action Tail
Every successful trajectory $\tau$ on Level $k$ can be uniquely factored into exploratory overhead and a causal tactical tail:
$$\tau_k = \tau_{\text{exploration}}^{(k)} \mathbin{\Vert} \tau_{\text{tail}}^{(k)}$$
where $\tau_{\text{tail}}^{(k)} = \big(a_{T_k - L}, \dots, a_{T_k - 1}\big)$ for window length $L \in [1, K_{\text{tail}}]$ (typically $L=8$).
The tail represents the minimal sufficient sequence of actions that drove the environment from the penultimate pre-goal configuration into the winning attractor basin.

### 3.2 Inductive Rulebook Synthesis via Minimum Description Length (MDL)
Let the environment state be formalized as a relational tuple $\mathcal{S} = (\mathcal{E}, \mathcal{R}, \mathcal{K})$, where $\mathcal{E}$ is the entity set, $\mathcal{R}$ are spatial relations (adjacency, distance, containment), and $\mathcal{K}$ is kinematic physics.

The causal win transition satisfies an unknown predicate $\phi^*(s) = \top$. The agent performs program induction over a domain-specific hypothesis grammar $\mathcal{L}_{\text{ARC}}$:
$$\phi^* = \arg\min_{\phi \in \mathcal{L}_{\text{ARC}}} \Big[ \text{Length}(\phi) + \lambda \sum_{j=1}^k \mathcal{L}_{\text{inconsistency}}(\phi, \tau_j) \Big]$$

#### Induced Invariant Classes:
1. **Kinematic Invariants**:
   - Step size $\Delta_{\text{step}} = ts$ (e.g. 6 pixels per discrete grid cell).
   - Momentum / Sliding physics: $\text{is\_sliding} \in \{\text{True}, \text{False}\}$. If displacement $\|\mathbf{x}_{t+1} - \mathbf{x}_t\|_1 > 1.5 \cdot ts$, slide physics is activated permanently across all subsequent levels.
2. **Entity Role Invariants**:
   - Avatar color $c_{\text{player}} = \text{const}$.
   - Obstacle colors $\mathcal{C}_{\text{obstacle}} = \{2, 3, 4, 8\} \cup \mathcal{C}_{\text{trail}}$.
   - Goal color $c_{\text{goal}} \in \{11, 14, 7\}$.
3. **Objective Predicate**:
   $$\phi^* = \begin{cases}
   \forall t \in \mathcal{T}_{\text{targets}}, \; \text{visited}(t) \land \mathbf{x}_{\text{player}} = \mathbf{x}_{\text{goal}} & \text{(Waypoints + Exit)} \\
   \forall c \in \text{Cards}, \; \text{cleared}(c) & \text{(Memory Match)} \\
   \mathbf{A} \mathbf{x}_{\text{clicks}} \equiv \mathbf{b}_{\text{target}} \pmod 2 & \text{(Linear Toggle)} \\
   \text{Flow}(\text{Source}, \text{Sinks}) = \text{MaxFlow} & \text{(Steiner Pulse)}
   \end{cases}$$

### 3.3 Zero-Shot Generalization to Levels $2 \dots N$

When Level $k+1$ begins, the agent queries the `SolvedLevelMemory` block:

#### Case 1: Identical / Cyclic Hash Match
$$\text{ContextHash}\big(s_0^{(k+1)}\big) == \text{ContextHash}\big(s_0^{(k)}\big)$$
In deterministic repeating or static puzzle stages, the agent bypasses all reasoning and executes a **deterministic instant replay** of $\tau_k$. Exploration action count $= 0$.

#### Case 2: Parametric / Topological Generalization
When Level $k+1$ features a novel map layout, larger grid dimensions, or increased entity counts (e.g., $N+1$ pairs in `mm01`, altered maze walls in `ic01..ic03`, or additional relay nodes in `rp01`):
1. The agent does **not** re-explore to learn what the goal is. The goal is already bound to $\phi^*$.
2. The problem transforms from an unstated reinforcement learning problem into an **explicit optimal control problem**:
   $$\pi^*_{k+1} = \arg\min_{\pi} \sum_{t=0}^{T} \text{Cost}(s_t, a_t) \quad \text{subject to } \phi^*(s_T) = \top$$
3. The agent activates specialized deterministic solvers directly:
   - *Discrete A\* / BFS with Collision Hulls*: For maze navigation and sliding ice puzzles.
   - *Minimum Steiner Tree / Pulse Routing*: For relay and plumbing networks.
   - *Kuhn-Munkres (Hungarian) Pair Matching*: For domino and memory tiles.
   - *GF(2) Gaussian Elimination*: For Lights Out toggles.

### 3.4 Mathematical Guarantee of Optimal Human-Parity (RHAE)
The ARC-AGI-3 official scoring metric defines Relative Human Action Efficiency as:
$$\text{RHAE} = \min\left(1.15, \; \left(\frac{N_{\text{human}}}{N_{\text{AI}}}\right)^2\right) \cdot \mathbb{I}[\text{Level Won}]$$

- On Level 1, exploratory probing may result in $N_{\text{AI}}^{(1)} \ge N_{\text{human}}$, producing a modest RHAE.
- On Levels $2 \dots N$, the transferred invariant rulebook $\phi^*$ allows the agent to execute the **mathematically optimal geodesic path**:
  $$N_{\text{AI}}^{(k)} = \text{dist}_{\text{geodesic}}(s_0^{(k)}, \phi^*) \le N_{\text{human}}^{(k)}$$
- Consequently, for all transferred levels $k \ge 2$:
  $$\frac{N_{\text{human}}^{(k)}}{N_{\text{AI}}^{(k)}} \ge 1.0 \implies \text{RHAE}^{(k)} \to 1.15 \quad (\text{Superhuman Action Efficiency Ceiling})$$

---

## 4. Synthesis: Complete Algorithmic Architecture

The integrated pipeline operates across three synchronized tiers:

| Component | Responsibility | Mathematical Formulation |
| :--- | :--- | :--- |
| **Zero-NN Perception** | Entity segmentation, centroid tracking, timer bar / HUD mask | CCL over $\mathbb{Z}^2$, bounding boxes, moments of inertia |
| **Dead-Signature Filter** | $O(1)$ pruning of null-action transitions | $\text{sig}(a, s) \in \mathcal{S}_{\text{dead}} \implies \text{Prune}(a)$ |
| **Structural Entropy Scorer** | Universal goal-seeking heuristic when reward is unstated | $H(G) = H_{\text{color}} + \alpha H_{\text{spatial}} + \beta H_{\text{degree}}$ |
| **Epistemic MCTS Planner** | Exploration-exploitation trade-off under uncertainty | $\text{UCT}(s, a) = Q + C_p e^{-\lambda V(s)} \sqrt{\ln N / (1+n)}$ |
| **Solved-Level Memory** | Cross-level invariant distillation and trajectory transfer | $\phi^* = \text{MDL}(\tau_1 \dots \tau_k), \; \pi_{k+1}^* = \text{Geodesic}(\phi^*)$ |

This closes the loop: an unstated objective on Level 1 is discovered via epistemic probing guided by topological entropy dissipation; the verified transition is crystallized into an invariant predicate; and Levels $2 \dots N$ are cleared with optimal human-parity action efficiency.