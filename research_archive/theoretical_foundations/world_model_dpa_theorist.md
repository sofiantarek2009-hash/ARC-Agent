# Technical Investigation: Kepler, Memento 3, and Discrete Programmatic Automata (DPA)

**To**: Parent Agent (`e2521de7-f050-4c42-b9e4-5c1062a99a1a`)  
**From**: Research Subagent  
**Subject**: Synthesis of Auditable World Models, Transition Functions $T(s, a) \to s'$, and DPA Kinematics in ARC-AGI-3

---

### Executive Summary

In interactive reasoning benchmarks like **ARC-AGI-3**, black-box neural policy models suffer catastrophic failures due to action budget exhaustion, lack of verifiable grounding, and vulnerability to the quadratic Relative Human Action Efficiency (**RHAE**) penalty. 

Three complementary frameworks address this challenge:
1. **Kepler** (*Wu, NeurIPS 2026*): An open-source audit harness that represents hypotheses as executable Python world models ($T(s, a) \to s'$), enforcing an auditable contract via **Retrospective Transition Checks** (100% historical replay verification) and **Conditional Prediction Checks** via a guarded `commit.py` tool.
2. **Memento 3** (*Reflective Rulebook Paradigm*): An externalized, structured hypothesis representation (`world_model.md` / `Rulebook`) compiled via an AST security sandbox, paired with **Solved-Level Memory (Siriki M85)** to transfer confirmed physical invariants and winning action tails across level progression without model fine-tuning.
3. **Discrete Programmatic Automata (DPA)**: A discrete symbolic physics paradigm that models autonomous and interactive grid entities via cellular automata updates, integer velocity vectors $\vec{v} = (v_r, v_c)$, deterministic boundary reflections, and time-indexed spatiotemporal hazard tubes $\mathcal{F}(t)$.

Below is the detailed technical investigation addressing each of the four core questions.

---

### 1. Deducing CA Rules & Movement Vectors from 1–3 Transitions (or Level 1 Probes)

#### A. Mathematical Formulation: State-Delta Decomposition
Let the grid at step $t$ be $S_t \in \Sigma^{H \times W}$, where $\Sigma = \{0, 1, \dots, 15\}$ represents discrete color tokens. When action $a_t$ is executed, the empirical observation is $S_{t+1}$.
The state transition delta tensor is:
$$\Delta S_t = S_{t+1} \ominus S_t, \quad \Delta \Omega = \|\Delta S_t\|_0 = \sum_{r, c} \mathbb{I}[S_{t+1}(r, c) \neq S_t(r, c)]$$

By applying **Connected Component Labeling (CCL)** with border timer-bar exclusion (excluding the outer 4-pixel countdown boundary), the changed coordinates $C_{\text{diff}} = \{(r, c) \mid \Delta S_t(r, c) \neq 0\}$ partition into:
- **Pre-image / Vacated cells ($E$)**: Cells occupied by a non-background color in $S_t$ that reverted to background in $S_{t+1}$.
- **Image / Target cells ($W$)**: Cells occupied by background in $S_t$ that acquired a sprite color in $S_{t+1}$.

#### B. Deduction from $\le 2$ Probe Actions (The Orthogonal Reversible Probe)
On Level 1, where action budgets are forgiving and failure states are distant, the agent probes with orthogonal action pairs:
1. **Step 1 (Forward Probe)**: Execute Action 1 (Up).
   - If $\Delta \Omega = 0$: Action 1 is a null/dead action (blocked by boundary/wall, or non-functional). Register in `DeadSignatureFilter`.
   - If $\Delta \Omega > 0$: Identify sprite $\mathcal{S}_i$ with color $c_i$ and mass $m_i$ whose centroid shifted:
     $$\vec{v}(a_1) = \text{centroid}(W) - \text{centroid}(E) = (\Delta r, \Delta c)$$
   - If $\vec{v}(a_1) = (-1, 0)$, the agent has simultaneously deduced:
     - **Controllable Avatar Identity**: The sprite of color $c_i$ is the player avatar.
     - **Action Semantics**: Action 1 is the discrete translation vector $(-1, 0)$ (Up).
     - **Passability**: Cell $(r-1, c)$ is passable empty space.
2. **Step 2 (Complementary Reversal Probe)**: Execute Action 2 (Down).
   - If $S_{t+2} \equiv S_t$: Vector $\vec{v}(a_2) = (1, 0)$ is confirmed, the environment is deterministic, and the exploration was **zero-loss** (avatar returns to start).
3. **Step 3 (Lateral Probe)**: Execute Action 4 (Right).
   - Confirms lateral movement $\vec{v}(a_4) = (0, 1)$ and by reflection $\vec{v}(a_3) = (0, -1)$.

#### C. Inductive Logic CA Rule Synthesis
For local cellular automata interactions (e.g., Sokoban pushing, ice sliding, hazard destruction):
- The transition function is parameterized as local $3 \times 3$ update kernels:
  $$S_{t+1}(r, c) = f(N_{3 \times 3}(S_t, r, c), a_t)$$
- With $|\Sigma| \le 16$, 1–3 transitions provide sufficient negative and positive constraints to solve the rule via Inductive Logic Programming (ILP) or constraint satisfaction:
  - **Sokoban Push**: If avatar collides with box of color $c_{\text{box}}$ at $(r', c')$ and box cell $(r' + v_r, c' + v_c)$ is empty, both avatar and box translate by $\vec{v}$.
  - **Slide / Inertia**: If avatar continues moving in direction $\vec{v}$ until hitting wall $c_{\text{wall}}$, the update rule applies iteratively: $S' = \text{raycast}(S, \text{pos}, \vec{v})$.

---

### 2. Classifying Action Modalities: Translation, Toggle, Rotation, or Parameter Click

The agent discriminates the exact mechanical modality of an action by computing morphological, topological, and algebraic invariants across the transition delta:

| Action Modality | Morphological Signature ($\Delta S$) | Conservation of Mass | Centroid Displacement ($\|\Delta \vec{x}\|$) | Geometric / Algebraic Test |
| :--- | :--- | :--- | :--- | :--- |
| **Grid Translation** | Disjoint Vacated ($E$) and Populated ($W$) sets | Strictly Conserved ($\|E\| = \|W\|$) | $\|\vec{d}\| \ge 1$ (Integer grid shift) | Normalized shape Blake2b hash is identical: $\text{Hash}(E) \equiv \text{Hash}(W)$ |
| **State Toggle** | In-place color mutation ($M = E \cap W \neq \emptyset$) | Unconserved (Colors flip in-place) | $\|\vec{d}\| \approx 0$ | Bounding box is static; local cell flips $c_A \leftrightarrow c_B$. Solved via GF(2) linear system $A x = b \pmod 2$. |
| **Discrete Rotation** | Bounding box aspect ratio transposes ($h \leftrightarrow w$) | Strictly Conserved | Stationary pivot: $\text{dist}(\text{centroid}, \text{pivot}) < \epsilon$ | Coordinate transformation matches $D_4$ dihedral group: $W - p = \mathbf{R}_\theta (E - p)$ for $\theta \in \{90^\circ, 180^\circ, 270^\circ\}$. |
| **Parameter Click** | Requires $(x, y)$ coordinate payload | Variable (Reveals hidden tile, stamps mark, or drops domino) | Directly bounded by $(x, y)$ | Diff bounding box strictly contains or centers on input coordinate $(x, y)$. Non-parameter actions (1–5) yield $\Delta \Omega = 0$. |

#### Detailed Detection Logic:
1. **Discrete Grid Translation**:
   - Compute normalized coordinates: $C_{\text{norm}} = C - \min(C)$.
   - If $\text{shape\_hash}(S_t[E]) == \text{shape\_hash}(S_{t+1}[W])$ and $\sum S_t[E] == \sum S_{t+1}[W]$, it is classified as a pure translation.
2. **State Toggle (Lights-Out / Door Mechanics)**:
   - Evaluated using `SymbolicPuzzleSolver`. If the bounding box is localized ($\le 3 \times 3$) and repeated clicks at $(x, y)$ invert the state back to $S_t$, it is a binary involution ($x^2 = I$).
   - Multi-cell toggle networks are mapped to a binary adjacency matrix $A \in \{0, 1\}^{N \times N}$ and solved using Gaussian elimination over **GF(2)**:
     $$A \mathbf{x} \equiv \mathbf{b} \pmod 2$$
3. **Discrete Rotation**:
   - Tested using `SemanticEvaluator` D4 dihedral orbit check. If the eigenvalues of the coordinate covariance matrix remain invariant while eigenvectors rotate by $\pi/2$, the action is classified as a $90^\circ$ rotation around centroid $(r_0, c_0)$.
4. **Parameter Click (ACTION6)**:
   - If actions 1–4 are invalid or produce null transitions ($\Delta \Omega = 0$), the agent inspects ACTION6.
   - Using `SymbolicPuzzleSolver.deduce_centers()`, the agent detects regular card/tile lattices by taking $\text{tile\_size} = \sqrt{\text{card\_area}}$ and checking grid purity. If state transitions only occur when coordinates coincide with tile centers, the action is classified as a Parameterized Click.

---

### 3. Implementing the 'Commit Tool' & 'Verification Delta'

#### A. Architecture of Kepler's `commit.py` & Sahu Verification Delta
In standard RL, agents execute actions directly in `env.step()`. If the agent's mental model is wrong, it takes catastrophic or wasteful steps that degrade the RHAE score.

The **Commit Tool** decouples planning from physical execution:
1. **Offline Simulation**: The agent generates an action macro $\pi = [a_1, a_2, \dots, a_k]$ and simulates it entirely inside its compiled world model:
   $$\hat{s}_0 = s_{\text{live}}, \quad (\hat{s}_{i+1}, \hat{\text{state}}_{i+1}) = \text{simulate\_step}(\hat{s}_i, a_{i+1}, \hat{\text{state}}_i)$$
2. **Offline Goal & Safety Certification**: The trajectory is verified offline:
   - Does $\hat{s}_k$ achieve the subgoal?
   - Does $\hat{s}_i$ intersect any forbidden hazard sets $\mathcal{F}(i)$?
   - Are any steps null transitions ($\Delta \Omega = 0$)?
3. **Guarded Commit with Verification Delta**: The agent passes the planned burst to the execution controller (`MacroActionQueue`). Before each action is forwarded to `env.step()`, a verification contract is enforced:

```python
class MacroActionQueue:
    def execute_burst(self, env_step_fn, hash_fn, interior_mask_fn):
        executed_count = 0
        while self.plan:
            step = self.plan.pop(0)  # Contains action, expected_hash, expected_grid
            executed_count += 1
            
            # 1. Execute live environment step
            next_grid, is_done, info = env_step_fn(step.action, step.action_data)
            
            # 2. Exclude border timer UI to prevent false divergence
            live_gameplay = interior_mask_fn(next_grid)
            expected_gameplay = interior_mask_fn(step.expected_grid)
            
            # 3. Check for No-Op / Dead Transition
            if np.array_equal(live_gameplay, prev_gameplay):
                self.dead_filter.record_dead_signature(step.signature)
                self.clear()  # Void remaining burst
                return VerificationResult(completed=False, diverged_at=executed_count, is_noop=True)
                
            # 4. Sahu Verification Delta: Hash comparison
            live_hash = hash_fn(live_gameplay)
            if step.expected_state_hash is not None and live_hash != step.expected_state_hash:
                # Simulation reality and live reality diverged!
                # Instantly void remaining queue, protect RHAE score, and return counterexample
                self.clear()
                counterexample = RecordedTransition(
                    step_index=executed_count,
                    grid=prev_grid,
                    action=step.action,
                    next_grid=next_grid,
                    predicted_grid=step.expected_grid
                )
                self.world_model_engine.refine_hypothesis(counterexample)
                return VerificationResult(completed=False, diverged_at=executed_count, counterexample=counterexample)
```

#### B. The 100% Cell-Exact Replay Verifier
Before any synthesized Python transition code (`simulate_step`) is approved for planning, Kepler's **ReplayVerifier** runs:
$$\text{Accuracy} = \frac{1}{|H|} \sum_{(s_t, a_t, s_{t+1}) \in H} \mathbb{I}[\text{simulate\_step}(s_t, a_t) \equiv s_{t+1}] \equiv 1.0$$
If a single cell in the entire historical replay diverges ($\text{diff\_count} > 0$), planning approval is **rejected**, and the exact coordinate `CellMismatch(step, row, col, expected, predicted)` is fed back to the code generator for iterative repair.

---

### 4. Handling Stochastic / Dynamic Sprites Without the 'Noisy TV' Trap

#### A. The 'Noisy TV' Problem in Discrete Grid Worlds
In curiosity-driven or transition-loss-minimizing world models, an agent encounters the "Noisy TV" trap when an environmental component generates unpredictable, uncontrollable entropy (e.g., oscillating decorative textures, random particle emitters, or uncoupled dynamic hazards). If the agent attempts to predict every pixel in $S_{t+1}$, planning either halts or fixates endlessly on fitting meaningless visual noise.

#### B. DPA Resolution: Controllable vs. Autonomous Decomposition
Discrete Programmatic Automata (DPA) resolves this by decomposing entities into three formal algebraic classes:
1. **Controllable Sub-state ($\mathcal{S}_{\text{ctrl}}$)**: The avatar and objects directly coupled to agent actions (e.g., pushed boxes, flipped switches). State changes synchronously with $a \in \mathcal{A}$.
2. **Autonomous Deterministic Automata ($\mathcal{S}_{\text{auto}}$)**: Entities whose motion is governed by discrete cellular automata rules parameterized purely by time $t$, independent of agent action:
   - **Linear Bouncers**: $\vec{x}_{t+1} = \vec{x}_t + \vec{v}_t$, where $\vec{v}$ reflects upon colliding with static masks:
     $$v_r \leftarrow -v_r \quad \text{if } x_r + v_r \in \text{WallMask}$$
   - **Gravity Falling Blocks**: $\vec{v} = (1, 0)$ subject to floor terminal rest.
   - **Cyclic Patrols**: Periodic state loops with period $T \le 16$.
3. **Exogenous / Stochastic Noise ($\mathcal{S}_{\text{noise}}$)**: Uncoupled background visual variations (e.g. flashing border lamps, decorative timers).

#### C. Spatiotemporal Hazard Tubes ($\mathcal{F}(t)$)
Rather than expanding the controllable state space combinatorially to include hazard sprites, DPA projects them forward into a **Spatiotemporal Collision Tube**:
$$\mathcal{F}(t) = \bigcup_{h \in \text{Hazards}} \left( \text{Trajectory}_h(t) \oplus \mathcal{B}_{\text{safety}} \right)$$
where $\mathcal{B}_{\text{safety}} = \{(-1, 0), (1, 0), (0, -1), (0, 1)\}$ is an adjacency dilation buffer preventing edge-grazing collisions.

The avatar's path planner (A* Navigation) then solves for trajectory $p(t)$:
$$\min_{\pi} \text{Cost}(\pi) \quad \text{s.t.} \quad p(t) \notin \mathcal{F}(t) \quad \forall t \in [0, H]$$
The agent treats the hazard as a **time-varying impassable obstacle**, completely bypassing the need to predict or control its internal entropy.

#### D. Partial Prediction Masks & Relational Topological Hashing
To prevent verification delta failures caused by uncoupled dynamic sprites:
1. **Partial Prediction Masks**: In `commit.py`, predictions specify a boolean mask $M_{\text{valid}}$. Cells marked as dynamic noise are excluded from verification delta equality checks:
   $$\| (S_{\text{live}} \ominus \hat{S}) \odot M_{\text{valid}} \|_0 == 0$$
2. **Relational Invariant Hashing (Blake2b-64)**: Implemented in `ZeroNNPerception.hash_scene()`. Instead of hashing raw grid pixels, the hash encodes:
   - Canonical component sorting by mass, color, and normalized shape.
   - Pairwise centroid displacements and Chebyshev distances between task-salient components.
   - Exogenous background noise and rigid translation shifts produce identical scene hashes, completely insulating the planner from the Noisy TV trap.

---

### Implementation File Reference Map

For inspection in the repository:
- `project_override/world_model/world_model_engine.py`: Contains `Rulebook`, `WorldModelCompiler`, `ReplayVerifier`, and `SolvedLevelMemory` (Siriki M85).
- `project_override/orchestration/macro_queue.py`: Implements `MacroActionQueue` and the Sahu Verification Delta Controller.
- `project_override/core/dpa_physics_predictor.py`: Implements `DPAPhysicsPredictor` (linear bouncers, deterministic intercept geometry, forbidden hazard tubes).
- `project_override/core/symbolic_puzzle_solver.py`: Implements ACTION6 click candidate extraction, tile center geometry deduction, and GF(2) linear toggle systems.
- `project_override/core/dead_signature.py`: Implements 64-bit packed bitwise action signature encoding and $O(1)$ dead-action culling.
- `project_override/core/zero_nn_ccl.py`: Implements translation-invariant Blake2b-64 relational topological scene hashing and border-timer masking.

All architectural modules are documented and validated against the internal test suites (`tests/test_world_model_and_repl.py`, `tests/test_astar_and_puzzle_solvers.py`).