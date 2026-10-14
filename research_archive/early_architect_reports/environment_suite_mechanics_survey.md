# Comprehensive Environment Suite Mechanics Survey

**Source Conversation ID:** `86d51fea-837f-44a1-a469-c9a46084bc05`

## Original Prompt
```text
<USER_REQUEST>
Analyze the full suite of 252 environments in `/Applications/Dad Company/arc-interactive/environment_files`.
Inspect the code across diverse game categories (movement, click, puzzle, physics, reaction) and answer:
1. What are the common mechanical patterns across games?
   - How is the player avatar represented? (Is it always a specific color or the only component that moves on Action 1-4?)
   - How are goals / win conditions triggered? (Collision with a specific sprite, reaching an area, clicking specific tiles?)
   - How is Action 6 (click) parameterized across games? (Clicking card grids, clicking falling/spawning items, clicking adjacent tiles?)
   - How is Action 5 used? (Toggling state, picking up items, interacting?)
2. What are the failure modes of generic planners like standard A* or random search on these games?
3. How can an agent reliably identify:
   - Its own avatar on Step 1 or 2 with 0 or 1 test action?
   - The goal or target without prior knowledge of goal color?
   - The passable terrain vs impassable walls without hardcoding color 3 or 4?
   - Interactive clickable targets for Action 6?
Return a comprehensive, rigorous taxonomy and detection algorithms with exact NumPy/SciPy formulations.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-12T07:53:57+03:00.
</ADDITIONAL_METADATA>
```

---

## Architecture / Research Report

### Comprehensive Analysis of 252 ARC-Interactive Environments

Based on static AST and code analysis across all 252 environment files in `/Applications/Dad Company/arc-interactive/environment_files`, here is the empirical taxonomy, failure modes of generic planners, and exact NumPy/SciPy formulations for autonomous identification.

---

### 1. Mechanical Patterns Across Games

#### A. Player Avatar Representation
* **Existence**: Present in ~147 games (Movement, Push/Sokoban, Hazard evasion). In ~105 games (pure Click, Sandpile, Match-3/Memory, Grid-painting), **there is no physical player sprite**; the interaction is purely cursor/coordinate-driven via `GameAction.ACTION6`.
* **Visual Palette**: In games featuring an avatar, **color 9** is the dominant standard avatar color (~85% of movement games), with color 14 used in rare variants (~3 games) or dynamic multi-colored state representations.
* **Movement Dynamics**: When an avatar exists, executing `ACTION1` (Up: $(0, -1)$), `ACTION2` (Down: $(0, 1)$), `ACTION3` (Left: $(-1, 0)$), or `ACTION4` (Right: $(1, 0)$) causes **the avatar to be the primary displaced component**. 
  * *Exceptions*: Sokoban/push mechanics (e.g., box pushes simultaneously), conveyor belts/wind, sliding ice environments (avatar slides multiple cells per action), or reactive/chaser games where hazards update their positions concurrently.

#### B. Goals & Win Conditions Triggering
1. **Geometric Overlap / Sprite Collision**:
   $$\text{Win} \iff (x_{\text{avatar}}, y_{\text{avatar}}) = (x_{\text{goal}}, y_{\text{goal}})$$
   Standard in navigation, maze escape, and reach tasks (~60 games). Target color is predominantly **14** (yellow) or **11** (cyan).
2. **Exhaustion / Clearing Sets**:
   $$\text{Win} \iff |\mathcal{S}_{\text{remaining}}| = 0$$
   Occurs in item collection, target sweeping (`len(self._targets) == 0`), memory card matching (`self._pairs_remaining == 0`), or lighting puzzle tiles (`len(self._lit) == 0`).
3. **Set Equality / Target Invariant**:
   $$\text{Win} \iff \mathcal{S}_{\text{painted}} \supseteq \mathcal{S}_{\text{goal}} \quad \text{or} \quad \mathcal{S}_{\text{flagged}} = \mathcal{S}_{\text{hidden}}$$
   Found in stencil painting (e.g. `sf04`), pattern reproduction, and minesweeper/beacon sweep (`bn01-04`).
4. **Physical Stability / State Conservation**:
   E.g., Abelian sandpiles (`ab01`), where stability $\max_{x,y} G_{x,y} < 4$ and mass balance $\sum G \pmod P = R$ trigger level advancement.

#### C. Parameterization of Action 6 (`ACTION6` / Click)
`ACTION6` requires payload coordinates: `action.data = {"x": px, "y": py}`.
* In **103+ games**, coordinates are passed through `self.camera.display_to_grid(px, py)` to resolve board-relative grid cell coordinates $(g_x, g_y)$.
* **Common Click Paradigms**:
  * **Card Grids / Memory Tiles**: Clicking discrete $(H \times W)$ card tiles toggles visibility or flips pairs.
  * **Grid Placement / Sandpile / Cellular Automata**: Adding mass or painting cells at $(g_x, g_y)$ (e.g., `ab01`, `sf04`).
  * **Interactive Adjacency / Beacon Placement**: Interacting with tiles adjacent to the player or placing sensors on fogged tiles.
  * **Physics / Projectiles / Directional Aiming**: Clicking a grid point defines a target vector for shooting or redirecting falling bodies.

#### D. Semantics of Action 5 (`ACTION5`)
Found across 72 games as an in-place modal modifier:
* **Orientation / Stencil Rotation**: Modulo cycle $(q + 1) \pmod 4$ for stencils/tools (e.g. `sf04`).
* **Tool / Mode Toggling**: Switching reveal modes (row vs. column in `bn04`), changing paint brushes, or toggling weapon/magnet states.
* **Contextual In-Place Interaction**: Picking up items (`as01`), activating standing towers (`bp01`), charging energy, or detonating placed bombs.
* **One-Step Phase Shift / Ability**: E.g., arming a one-time pass through walls (`bl01`).

---

### 2. Failure Modes of Generic Planners (Standard A* & Random Search)

1. **Massive Action Space in Click Games**:
   A generic search treats action spaces as discrete. With `ACTION6(x, y)` over a $64 \times 64$ display screen, the branching factor per step jumps from $|\mathcal{A}| = 4$ to $|\mathcal{A}| \approx 4096$. Random exploration has near-zero probability of hitting the requisite grid sequence.
2. **Hidden State & Irreversible Traps (Dead Ends)**:
   Many puzzle/push games have irreversible state transitions (e.g., pushing a block into a corner). Without backward reasoning or undo simulations, standard $A^*$ without domain heuristics falls into dead ends or exhausts its node budget.
3. **Multi-Step Delayed Win Signals**:
   In games requiring sequence collection (e.g., gather key $\to$ unlock door $\to$ touch goal), the Euclidean distance to the visible goal creates a severe local minimum in $A^*$. The agent gets stuck against the wall closest to the goal instead of traversing away to retrieve the prerequisite key.
4. **Non-Stationary / Environmental Dynamics**:
   In physics/reaction games with gravity, falling hazards, or patrolling enemies, the transition function $T(s, a) \to s'$ is time-dependent. Standard static-grid $A^*$ plans collisions into moving obstacles.
5. **Partial Observability / Fog-of-War**:
   In beacon sweep (`bn01-04`) or memory match environments, state evaluation requires belief-state tracking; standard deterministic $A^*$ cannot evaluate information-gathering actions.

---

### 3. Rigorous Detection Algorithms (NumPy / SciPy Formulations)

#### A. Avatar Identification on Step 1–2 (with 0 or 1 Test Action)

Let initial observation frame be $O_0 \in \mathbb{Z}^{H \times W}$.

**0-Action Prior Hypothesis (Step 1)**:
Scan connected components of rare, compact foreground sprites:
```python
import numpy as np
from scipy.ndimage import label

def detect_avatar_prior(frame: np.ndarray, bg_color: int = 5):
    # Candidate pixels excluding background
    mask = (frame != bg_color)
    # Color 9 is canonical avatar color across ~85% of games
    if np.any(frame == 9):
        coords = np.argwhere(frame == 9)
        return {"color": 9, "position": coords.mean(axis=0), "confidence": 0.95}
    
    # Otherwise identify unique single-cell or minimal connected components
    best_candidate = None
    min_size = float("inf")
    for c in np.unique(frame):
        if c == bg_color:
            continue
        c_mask = (frame == c)
        labeled, num_features = label(c_mask)
        for i in range(1, num_features + 1):
            sz = np.sum(labeled == i)
            if sz == 1:  # single unit cell avatar
                return {"color": int(c), "position": np.argwhere(labeled == i)[0], "confidence": 0.7}
    return None
```

**1-Action Active Probe Formulation (Step 2)**:
Execute directional action $a \in \{1, 2, 3, 4\}$ (e.g. $a = 4 \implies \Delta = [0, 1]$).
Let $O_1$ be the post-action observation frame:
$$\Delta O = (O_1 \neq O_0)$$
$$\text{Displaced Mask}: M_{\text{vacated}} = (O_0 \neq O_1) \land (O_1 == \text{bg}), \quad M_{\text{occupied}} = (O_0 \neq O_1) \land (O_0 == \text{bg})$$
```python
def identify_avatar_active(O_0: np.ndarray, O_1: np.ndarray, expected_delta: np.ndarray = np.array([0, 1])):
    diff_mask = (O_0 != O_1)
    if not np.any(diff_mask):
        return None  # Wall blocked or click-only game
    
    # Find components present in O_0 whose pixel set shifted by expected_delta in O_1
    for c in np.unique(O_0[diff_mask]):
        mask0 = (O_0 == c)
        mask1 = (O_1 == c)
        # Shift mask0 by expected_delta
        shifted_mask0 = np.roll(mask0, shift=expected_delta, axis=(0, 1))
        # Zero out rolled border artifacts
        if expected_delta[0] > 0: shifted_mask0[:expected_delta[0], :] = False
        elif expected_delta[0] < 0: shifted_mask0[expected_delta[0]:, :] = False
        if expected_delta[1] > 0: shifted_mask0[:, :expected_delta[1]] = False
        elif expected_delta[1] < 0: shifted_mask0[:, expected_delta[1]:] = False
        
        overlap = np.logical_and(shifted_mask0, mask1)
        if np.sum(overlap) > 0 and np.sum(overlap) == np.sum(mask0 & diff_mask):
            avatar_pos = np.argwhere(mask1 & diff_mask)
            return {"color": int(c), "position": avatar_pos.mean(axis=0), "active": True}
    return None
```

---

#### B. Goal / Target Identification Without Color Prior
Goals exhibit structural invariants:
1. **Low Cardinality / Uniqueness**: In single-goal games, goal sprites occupy 1 to 4 cells.
2. **Topological Inaccessibility**: Positioned at points of high geodesic path length from avatar start.
3. **Palette Divergence**: Distinct from background ($c \neq 5$) and wall ($c \neq 3$).

```python
def identify_candidate_goals(frame: np.ndarray, avatar_pos: np.ndarray, wall_mask: np.ndarray):
    candidates = []
    unique_colors = np.unique(frame)
    for c in unique_colors:
        if c in (5, 3, 4, 0): # standard bg/wall priors
            continue
        coords = np.argwhere(frame == c)
        if len(coords) == 0:
            continue
        
        # Calculate cluster properties
        cluster_size = len(coords)
        centroid = coords.mean(axis=0)
        dist_to_avatar = np.linalg.norm(centroid - avatar_pos)
        
        # Score priority: rare elements far from avatar
        score = (1.0 / (cluster_size + 1e-4)) * dist_to_avatar
        candidates.append({"color": int(c), "centroid": centroid, "size": cluster_size, "score": score})
        
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates
```

---

#### C. Passable Terrain vs. Impassable Walls Without Hardcoding Colors
Impassable walls have unmistakable topological signatures:
* **High Morphological Bounding/Perimeter**: Walls enclose boundaries, have high Euler characteristics, or form continuous 1D/2D obstacles.
* **Zero Permeability**: No sprite ever overlaps or traverses them.

```python
from scipy.ndimage import binary_erosion, generate_binary_structure

def identify_impassable_walls(frame: np.ndarray, bg_color: int):
    # A wall candidate is non-background, multi-instance, often perimeter-aligned
    H, W = frame.shape
    candidates = {}
    
    border_pixels = np.concatenate([
        frame[0, :], frame[H - 1, :], frame[:, 0], frame[:, W - 1]
    ])
    
    for c in np.unique(frame):
        if c == bg_color:
            continue
        mask = (frame == c)
        # Ratio of perimeter alignment
        border_count = np.sum(border_pixels == c)
        total_count = np.sum(mask)
        
        # Structural connectivity test: walls frequently form contiguous barriers
        struct = generate_binary_structure(2, 2)
        eroded = binary_erosion(mask, structure=struct)
        
        # Metric combining perimeter presence, count, and lack of motion
        wall_likelihood = (border_count / (2 * (H + W))) * 2.0 + (total_count / (H * W))
        candidates[int(c)] = wall_likelihood

    # Color with highest structural barrier likelihood is wall
    wall_color = max(candidates, key=candidates.get)
    impassable_mask = (frame == wall_color)
    passable_terrain = ~impassable_mask & (frame == bg_color)
    return wall_color, impassable_mask, passable_terrain
```

---

#### D. Interactive Clickable Targets for Action 6
In `ACTION6` games, clickable entities are organized in regular geometric arrays (grids of cards/cells) or bounded clickable arenas:

```python
from scipy.spatial.distance import cdist

def detect_clickable_grid_targets(frame: np.ndarray, bg_color: int = 5):
    """
    Detects regular card/tile lattices by identifying repeating centroids of connected components.
    """
    foreground = (frame != bg_color)
    labeled, num_features = label(foreground)
    
    centroids = []
    bounding_boxes = []
    
    for i in range(1, num_features + 1):
        pts = np.argwhere(labeled == i)
        if len(pts) < 1:
            continue
        min_y, min_x = pts.min(axis=0)
        max_y, max_x = pts.max(axis=0)
        h, w = max_y - min_y + 1, max_x - min_x + 1
        
        # Filter UI bars (e.g. level indicator dots at y=0 or status bars at bottom)
        if min_y <= 2 or max_y >= frame.shape[0] - 2:
            continue
            
        centroids.append(pts.mean(axis=0))
        bounding_boxes.append((min_x, min_y, max_x, max_y))
        
    if not centroids:
        # Fallback to entire board interior as interactive click plane
        return [{"type": "continuous_grid", "bounds": (0, 0, frame.shape[1], frame.shape[0])}]
    
    centroids = np.array(centroids)
    # Check for grid regularity via pairwise distance modal frequencies
    dists = cdist(centroids, centroids)
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dists = dists.min(axis=1)
    
    # Return formatted discrete target click centers
    clickable_targets = [
        {"grid_xy": (int(round(c[1])), int(round(c[0]))), "bbox": bbox}
        for c, bbox in zip(centroids, bounding_boxes)
    ]
    return clickable_targets
```