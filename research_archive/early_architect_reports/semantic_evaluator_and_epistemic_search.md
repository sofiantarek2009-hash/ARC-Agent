# Semantic Evaluator & Epistemic-Damped Search Blueprint

**Source Conversation ID:** `5d1ba1cb-cb22-42e3-be2e-fc418fe460ff`

## Original Prompt
```text
<USER_REQUEST>
You are the Semantic Evaluator and Epistemic-Damped Search Architect for Project Override (ARC-AGI-3).
Your task is to design and write the complete, non-placeholder implementation of:
1. `semantic_evaluator.py`:
   - Dihedral Group D4 symmetry scorer: Measures reflectional (horizontal, vertical, diagonal) and rotational (90, 180, 270) symmetry of states.
   - Boundary enclosure & connectivity closure: Flood-fill and ray-cast containment to detect sealed loops, enclosed goals, and connected paths.
   - Mass and color compaction metrics: Cluster density and perimeter-to-area compactness ($\sum \text{Mass} / (\text{Perimeter} + \epsilon)$).
   - Structural Shannon Entropy Scorer: Evaluates color entropy, degree entropy, and spatial dispersion. A transition is rewarded if $\Delta H \le -\epsilon$.
   - Discrete Optimal Transport (Sinkhorn-Knopp algorithm): Measures 2-Wasserstein distance ($W_2$) between discrete point clouds of object centroids to handle object fission (shattering) and fusion (merging) without breaking topological continuity.
2. `dead_signature.py` & Epistemic MCTS Planner:
   - 64-bit packed bitwise action signature encoder: encodes action (4 bits), color (4 bits), mass (16 bits), and context hash (32 bits).
   - O(1) Dead-Signature Culling: Prunes any action that produces a null state transition ($\Delta \Omega = 0$).
   - Epistemic-Damped UCT search: $\text{UCT}(s, a) = Q(s, a) + C_p \cdot \exp(-\lambda \cdot \mathcal{V}(s)) \cdot \sqrt{\ln N / (1 + n)}$.

Write clean, robust, fully working Python and/or C++ code. Test the code and output the full file paths and implementations.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-11T23:29:21+03:00.
</ADDITIONAL_METADATA>
```

---

## Architecture / Research Report

Completed the implementation and verification of Semantic Evaluator, Dead-Signature Culling, and Epistemic MCTS Planner for Project Override (ARC-AGI-3):

1. `/Applications/Dad Company/project_override/project_override/core/semantic_evaluator.py`:
   - Dihedral Group D4 symmetry scorer: horizontal (flipud), vertical (fliplr), diagonal (transpose/anti-diagonal), and rotational (90, 180, 270) symmetry metrics with automatic square padding for rectangular states.
   - Boundary enclosure & connectivity closure: 4-way exterior flood-fill cavity detection, ray-cast containment across cardinal axes, and BFS connectivity closure.
   - Compaction metrics: perimeter-to-area compactness (\sum Mass / (Perimeter + \epsilon)), bounding-box cluster density, and radius of gyration.
   - Structural Shannon Entropy Scorer: color entropy, degree entropy (color boundary heterogeneity), spatial dispersion across subdivisions, with transition rewards awarded when \Delta H \le -\epsilon.
   - Discrete Optimal Transport (Sinkhorn-Knopp algorithm): calculates 2-Wasserstein distance (W_2) between discrete centroid point clouds, gracefully handling object fission (shattering) and fusion (merging).

2. `/Applications/Dad Company/project_override/project_override/core/dead_signature.py`:
   - 64-bit packed bitwise action encoder: Action (4 bits), Color (4 bits), Mass (16 bits), Context Hash (32 bits), and Flags (8 bits). Includes hex serialization and CRC32 context hashing.
   - O(1) Dead-Signature Culling: prunes any action producing a null state transition (\Delta \Omega = 0) with instantaneous set-based culling and static null prediction.

3. `/Applications/Dad Company/project_override/project_override/core/mcts_planner.py`:
   - Epistemic-Damped UCT search: UCT(s, a) = Q(s, a) + C_p * exp(-\lambda * V(s)) * sqrt(ln(N) / (1 + n)), dynamically dampening chaotic branches based on epistemic variance V(s).
   - O(1) action pruning integrated directly into expansion.
   - Semantic reward integration combining entropy drops, compaction, symmetry, and W_2 proximity.
   - Macro-burst generation producing MacroPlanStep batches for MacroActionQueue.

4. `/Applications/Dad Company/project_override/tests/test_architect_eval_planner.py`:
   - 19 comprehensive unit and pipeline integration tests, 100% passing.

All root and package shims have been configured for seamless imports across modules.

## Additional Synthesized Content

# Project Override (ARC-AGI-3): Semantic Evaluator & Epistemic-Damped Search Architecture

The complete, non-placeholder implementation of the **Semantic Evaluator**, **64-bit Dead-Signature Culler**, and **Epistemic-Damped MCTS Planner** has been designed, implemented, and verified with unit and integration tests.

---

## 1. System Architecture & File Layout

| Module | Absolute File Path | Description |
| :--- | :--- | :--- |
| **Semantic Evaluator** | [`semantic_evaluator.py`](file:///Applications/Dad%20Company/project_override/project_override/core/semantic_evaluator.py) | Full $D_4$ group symmetry, flood-fill & ray-cast enclosure, compaction metrics, Structural Shannon entropy ($\Delta H \le -\epsilon$), and Sinkhorn-Knopp Optimal Transport ($W_2$). |
| **Dead Signature** | [`dead_signature.py`](file:///Applications/Dad%20Company/project_override/project_override/core/dead_signature.py) | 64-bit bitwise packed action signature encoder, CRC32 32-bit context hashing, and $O(1)$ dead-signature null-transition ($\Delta \Omega = 0$) culling filter. |
| **Epistemic MCTS** | [`mcts_planner.py`](file:///Applications/Dad%20Company/project_override/project_override/core/mcts_planner.py) | Epistemic-Damped UCT tree search planner, $O(1)$ action pruning during tree expansion, and macro-burst plan extraction for `MacroActionQueue`. |
| **Evaluation Facade** | [`evaluation/semantic_evaluator.py`](file:///Applications/Dad%20Company/project_override/project_override/evaluation/semantic_evaluator.py) | Re-exports all evaluators under `project_override.evaluation`. |
| **Test Suite** | [`test_architect_eval_planner.py`](file:///Applications/Dad%20Company/project_override/tests/test_architect_eval_planner.py) | 19 comprehensive unit and integration tests (100% passing). |

---

## 2. Mathematical Formulation & Algorithmic Details

### A. Dihedral Group $D_4$ Symmetry Scorer
The Dihedral Group $D_4$ comprises the 8 isometric transformations of a 2D square:
1. **Rotations**: $R_0$ (Identity), $R_{90}$ (90° clockwise), $R_{180}$ (180°), $R_{270}$ (270° clockwise).
2. **Reflections**:
   - $S_h$: Horizontal reflection across horizontal midline (`np.flipud`)
   - $S_v$: Vertical reflection across vertical midline (`np.fliplr`)
   - $S_{d1}$: Main diagonal reflection top-left to bottom-right ($A^T$, transpose)
   - $S_{d2}$: Anti-diagonal reflection top-right to bottom-left
- **Aspect Ratio Invariance**: For rectangular grids ($H \ne W$), the scorer supports `pad_to_square=True`, which embeds the grid into a centered square canvas of size $\max(H, W) \times \max(H, W)$ padded with the background color to evaluate all 8 transformations without dimension mismatch.

### B. Boundary Enclosure & Connectivity Closure
- **Flood-Fill Enclosure**:
  Pads the grid with an exterior border and performs a 4-connected flood-fill from $(0, 0)$. Any non-barrier cell that cannot be reached from the exterior is identified as belonging to an *interior sealed cavity*.
  Connected components of cavities are extracted via `scipy.ndimage.label`.
- **Ray-Cast Containment**:
  For any point $(r, c)$, rays are cast in all 4 cardinal directions ($N, S, E, W$). If all 4 rays encounter a barrier cell prior to the canvas boundaries, the cell is marked as ray-cast enclosed.
- **Goal Enclosure & Connectivity**:
  Validates whether goal coordinates or specific goal colors are fully enclosed inside sealed cavities, and performs BFS traversability path closure between target endpoints.

### C. Mass & Color Compaction Metrics
- **Perimeter-to-Area Compactness**:
  Evaluates the exact metric specified in the ARC-AGI-3 specification:
  $$\text{Compactness} = \sum_{c \in \text{components}} \frac{\text{Mass}(c)}{\text{Perimeter}(c) + \epsilon}$$
  where $\text{Perimeter}(c)$ is the count of external 4-connected boundary edges exposed to non-component pixels.
- **Cluster Density & Radius of Gyration**:
  $$\text{Cluster Density} = \frac{\text{Mass}}{A_{\text{bbox}}} = \frac{\text{Mass}}{(r_{\max} - r_{\min} + 1)(c_{\max} - c_{\min} + 1)}$$
  $$R_g = \sqrt{\frac{1}{M} \sum_{i=1}^M \|p_i - \bar{p}\|^2_2}, \quad \text{Gyration Compactness} = \frac{1}{1 + R_g}$$

### D. Structural Shannon Entropy Scorer
- **Color Entropy**:
  $$H_{\text{color}} = - \sum_{c \in \mathcal{C}, p(c) > 0} p(c) \log_2(p(c))$$
- **Degree Entropy**:
  Treats each cell's orthogonal boundary heterogeneity as a degree $d \in \{0, 1, 2, 3, 4\}$ (number of neighbors with differing colors):
  $$H_{\text{degree}} = - \sum_{d, p(d) > 0} p(d) \log_2(p(d))$$
- **Spatial Dispersion Entropy**:
  Partitions the canvas into $K \times K$ subdivisions (e.g. $4 \times 4$ spatial grid) and evaluates mass dispersion:
  $$H_{\text{spatial}} = - \sum_{b, p(b) > 0} p(b) \log_2(p(b))$$
- **Composite Structural Entropy & Transition Rewards**:
  $$H(s) = w_c H_{\text{color}} + w_d H_{\text{degree}} + w_s H_{\text{spatial}}$$
  $$\Delta H = H(s') - H(s)$$
  A state transition is rewarded if:
  $$\Delta H \le -\epsilon \implies \text{Reward} = \max(0.0, -\Delta H)$$

### E. Discrete Optimal Transport (Sinkhorn-Knopp $W_2$)
- **Continuous Representation of Discrete Transitions**:
  Extracts centroid coordinates $X \in \mathbb{R}^{n \times 2}, Y \in \mathbb{R}^{m \times 2}$ and mass probability distributions $a \in \mathbb{R}^n, b \in \mathbb{R}^m$ for all non-background objects.
- **Cost Matrix**:
  $$C_{ij} = \|x_i - y_j\|_2^2$$
- **Entropy-Regularized Sinkhorn-Knopp**:
  Solves $\min_{P \in \mathcal{U}(a, b)} \langle P, C \rangle + \gamma \mathcal{H}(P)$ using iterative scaling:
  $$u^{(k+1)} = \frac{a}{K v^{(k)} + \delta}, \quad v^{(k+1)} = \frac{b}{K^T u^{(k+1)} + \delta}, \quad K = \exp(-C / \gamma)$$
- **2-Wasserstein Distance ($W_2$)**:
  $$W_2 = \sqrt{\sum_{i=1}^n \sum_{j=1}^m P_{ij} \|x_i - y_j\|_2^2}$$
- **Fission & Fusion Preservation**:
  - **Fission ($1 \to K$)**: When an object shatters, $P$ distributes parent mass across the shards; $W_2$ measures dispersion without topological discontinuity.
  - **Fusion ($K \to 1$)**: When $K$ shards merge, $P$ collects mass into the unified centroid.

---

## 3. Action Signature & $O(1)$ Dead-Signature Culling

### 64-Bit Packed Bitwise Action Layout
The 64-bit integer (`uint64`) packs fields without memory allocation overhead:
```
[63:56]  (8 bits) : Flags / Version (0..255)
[55:52]  (4 bits) : Action (0..15)
[51:48]  (4 bits) : Color (0..15)
[47:32] (16 bits) : Mass (0..65535)
[31:0]  (32 bits) : Context Hash (0..0xFFFFFFFF)
```
- Bitwise packing:
  ```python
  sig = ((flags & 0xFF) << 56) | ((action & 0x0F) << 52) | ((color & 0x0F) << 48) | ((mass & 0xFFFF) << 32) | (context_hash & 0xFFFFFFFF)
  ```
- **$O(1)$ Dead Culling Filter**:
  A state transition is dead if $\Delta \Omega = \|s' - s\|_0 == 0$ (null transition / noop).
  The filter caches dead action signatures in a set (`DeadSignatureFilter`).
  During tree expansion, candidate actions are evaluated in $O(1)$ against the dead signature table, eliminating unnecessary simulations.

---

## 4. Epistemic-Damped UCT Search Formulation

The exploration bonus in Upper Confidence Bound for Trees (UCT) is dynamically modulated by the epistemic uncertainty $\mathcal{V}(s)$:

$$\text{UCT}(s, a) = Q(s, a) + C_p \cdot \exp(-\lambda \cdot \mathcal{V}(s)) \cdot \sqrt{\frac{\ln N(s)}{1 + n(s, a)}}$$

- **Epistemic Variance Estimator**:
  $$\mathcal{V}(s) = \operatorname{Var}_{a \in \text{children}}(Q(s, a)) + \frac{\sigma_0^2}{1 + N(s)}$$
- **Behavioral Properties**:
  - When $\mathcal{V}(s)$ is high (chaotic, unpredictable, or newly discovered states), $\exp(-\lambda \mathcal{V}(s)) \to 0$. Exploration on ungrounded branches is suppressed, preserving the search budget.
  - When $\mathcal{V}(s) \to 0$ (grounded state with consistent estimates), $\exp(-\lambda \mathcal{V}(s)) \to 1$, restoring standard UCT exploration.
- **Macro-Action Generation**:
  Outputs a chain of high-visit actions mapped to `MacroPlanStep` objects with expected grids and context hashes, compatible with `MacroActionQueue.execute_burst`.

---

## 5. Test Suite Verification

Run the test suite directly:
```bash
./venv/bin/python project_override/tests/test_architect_eval_planner.py
```
**Test Output**:
```
Ran 19 tests in 0.076s

OK
```

All 19 test cases passed:
1. `test_d4_symmetry_horizontal`: $D_4$ horizontal reflection ($S_h$).
2. `test_d4_symmetry_vertical`: $D_4$ vertical reflection ($S_v$).
3. `test_d4_symmetry_rotational_180`: $D_4$ 180° rotation ($R_{180}$).
4. `test_d4_symmetry_diagonal`: $D_4$ main diagonal reflection ($S_{d1}$).
5. `test_d4_symmetry_rectangular_padding`: Dimension padding for rectangular grids.
6. `test_boundary_enclosure_sealed_cavity`: Detection of sealed loops enclosing goals.
7. `test_boundary_enclosure_leaky_loop`: Correctly rejects unsealed/leaky perimeters.
8. `test_ray_cast_containment`: 4-way ray-cast containment check.
9. `test_path_connectivity_bfs`: BFS path connectivity and obstacle breach.
10. `test_compaction_metrics`: Validates $\sum \text{Mass} / (\text{Perimeter} + \epsilon)$ on shapes.
11. `test_structural_entropy_and_reward`: Validates transition reward for $\Delta H \le -\epsilon$.
12. `test_optimal_transport_identity`: Sinkhorn-Knopp $W_2 \approx 0$ on identical clouds.
13. `test_optimal_transport_fission`: 1 object shattering into 4 shards ($W_2 = 2.0$).
14. `test_optimal_transport_fusion`: 4 shards merging into 1 centroid ($W_2 = 2.0$).
15. `test_action_signature_packing_unpacking`: Bitwise round-trip packing and hex serialization.
16. `test_dead_signature_culling`: $O(1)$ registration and filtering of null transitions ($\Delta \Omega = 0$).
17. `test_epistemic_uct_formula`: Verifies mathematical damping $\exp(-\lambda \mathcal{V}(s))$.
18. `test_mcts_planner_execution`: End-to-end MCTS tree search on an ARC grid state.
19. `test_mcts_macro_queue_pipeline_integration`: Full pipeline test integrating MCTS, `MacroPlanStep`, and `MacroActionQueue.execute_burst`.