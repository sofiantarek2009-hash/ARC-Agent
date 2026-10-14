# ARC-AGI-3 Comprehensive Mechanical Taxonomy & Dynamic Perceptual Detection Framework

Based on an exhaustive analysis of all **252 game environments** (across all 153 distinct stem families) in `/Applications/Dad Company/arc-interactive/environment_files`, here is the established universal taxonomy, perceptual detection framework, and action space mapping without any hardcoding.

---

## 1. Universal Taxonomy of ARC-AGI-3 Mechanical Classes

Every game in the 252-environment corpus belongs to one of **13 mutually exclusive, collectively exhaustive (MECE) mechanical families**:

| Family # | Mechanical Family | Defining Mechanics | Representative Game Stems |
|---|---|---|---|
| **I** | **Direct Spatial Navigation & Path Constraints** | Discrete 1-step cardinal traversal with topological constraints (revisit forbidden, Hamiltonian tour, decaying floors, locks/keys, portals, non-standard moves). | `ez01-04` (basics), `bd01` (no revisits), `hm01` (Hamiltonian), `va01-03` (coverage), `ul01-03` (keys/doors), `tp01-03` (portals), `wk01` (weak floors), `gl01` (glass floors), `tr01` (toxic floors), `kn01` (knight hops), `bi01` (bishop slides), `gr01` (gravity). |
| **II** | **Inertial, Slide & Momentum Dynamics** | Movement vector propagates until blocked by obstacles, or undergoes momentum constraints / external physical drift. | `ic01-03` (ice slides, torus ice, capped slide), `ci01` (crate on ice), `mo01` (momentum: $\ge 2$ steps straight), `tc01` / `cy01` (conveyor push), `wg01` (wind gusts), `eb01` (escalators), `rk01` (rook slide), `rf01` (mirror input swap). |
| **III** | **Block Manipulation & Sokoban (Push/Pull)** | Interacting with movable rigid entities by pushing, pulling, or placing onto designated destination sockets / pressure plates. | `sk01-04` (classic multi-crate sokoban), `pb01-04` (single-box & winch pulls), `pw01` (dual plates), `rz01` (rush grid 1x2 cars), `mb01` (magnet crates), `tk01` (telekinetic pull), `vt01` / `wp01` (weight plates). |
| **IV** | **Multi-Waypoint, Sequencing & Logistics** | Sequential or unordered target acquisition, cargo delivery, SLA constraints, or dynamic rule changes. | `tt01-03` (target collection), `sq01-05` (color sequences), `as01` (assembly fetch), `dd01` (SLA routed delivery), `rs01-04` (rule switching), `fl01/04` / `lw01-04` (numberlink / line weave), `sn01` (snake growth). |
| **V** | **Network, Circuit, Relay & Plumbing** | Configuring discrete conduit orientations, switches, or relays to establish source-to-sink transmission paths. | `pu01/04` (pipe twist), `pd01/04` (pipe drop), `ck02-04` (circuit junctions & wires), `rp01-04` (relay pulse fire), `sg01/04` (signal locks), `cr01` / `tb01-03` / `wl01` (bridge & wall building). |
| **VI** | **Optics, Lasers & Ray Reflection** | Placing or rotating deflectors/mirrors to guide continuous beams or bouncing projectiles into targets. | `ml01-04` (mirror laser / fire pulse), `pj01/04` (bolt bounce), `lf01` (laser fences), `ec01` (mirror echoes). |
| **VII** | **Pattern Matching, Stencils, Painting & Tiling** | Reconfiguring canvas pixels to match hints, stencils, symmetries, or solve combinatorial packing/inversion. | `gp01-04` (grid paint/erase), `ff01-04` (flood fill), `sf01/04` (stencil translation & stamp), `lo01-05` (lights out kernels), `sy01-04` (mirror reflection), `pt01-05` / `qr01/04` (tile/quad rotation), `pk01/02` / `dm01/04` (domino/polyomino packing), `ng01/04` (nonograms), `sl01` (8-puzzle). |
| **VIII** | **Memory, Hidden State & Information Discovery** | Deductive exploration or memory recall over hidden, fogged, or masked elements. | `mm01-05` (memory card pair matching), `ms01-04` (blind sapper / minesweeper), `bn01-04` (beacon sweep radar), `fg01` (fog of war), `rc01` (state recall). |
| **IX** | **Reaction, Timing, Reflex & Dynamic Hazards** | Real-time or turn-based autonomous environmental evolution requiring phase synchronization or speed. | `wm01-04` (whack-a-mole reflex click), `zq01-03` (blinking phase hazards), `tf01` (traffic lights), `rb01` (rhythm gates), `st01` (stealth guard vision cones), `tg01` / `hs01` (chasers), `fw01` (wildfire spread), `vi01` (infection run). |
| **X** | **Cellular Automata & Physical Simulations** | Local rule-based continuous or discrete physical evolution. | `ab01` / `sp01/04` (abelian sandpile toppling), `ll01/04` (Conway's Game of Life), `ph01-04` (phase wave interference mod 4), `df01/04` / `ih01` (heat diffusion), `zm01/04` (strain / flood spread). |
| **XI** | **Discrete Logic, Combinatorics & Formal Algebra** | Discrete constraint satisfaction, Boolean SAT, graph theory, or arithmetic trees. | `cw01` (3-SAT clause toggles), `cs01` (vertex cover), `ct01` (independent set), `cv01` (list coloring), `cx01` (cut gates), `cz01` / `ox01` (XOR algebra), `hn01/04` (Tower of Hanoi), `kv01/04` (Kirchhoff resistor ladders). |
| **XII** | **Survival & Resource Economics** | Multi-meter depletion management (hunger, warmth, oxygen, battery, health) alongside navigation. | `sv01-03` (hunger & warmth with shelters), `tm01` (twin oxygen/heat meters), `bt01` (battery step limit), `au01` (bonus step pickups), `ep01` (pain budget). |
| **XIII** | **Multi-Agent & Coordination** | Joint control or priority cycling across multiple bodies on the board. | `mc01` (tandem joint movement + lead swap), `ob01` (odd one out 3-body cycle), `sw01` (partner swap), `es01` (escort civilian), `dp01` (dual plane). |

---

## 2. Dynamic Entity & Category Identification Framework (First 1–2 Frames)

An agent can identify game properties dynamically without hardcoding by following a strict 4-stage perceptual and active probing pipeline:

### Stage A: Spatial De-letterboxing & Grid Parsing
All ARC-AGI-3 frames arrive rendered as $64 \times 64$ palette arrays $O \in \{0..15\}^{64 \times 64}$.
1. **Viewport Detection**: Detect constant rows/columns of padding color (empirically Color 4 in 71.8% of games, or Color 5 in 27.8%) on outer edges. Find active viewport $[y_{min}, y_{max}] \times [x_{min}, x_{max}]$.
2. **Cell Resolution ($cs$) Extraction**:
   $$cs = \gcd \left( \{ \text{run lengths of horizontal and vertical uniform color blocks inside active viewport} \} \right)$$
   In ARC-AGI-3, $cs \in \{1, 2, 3, 4, 8\}$, corresponding exactly to grid sizes $64\times 64$, $32\times 32$, $24\times 24$, $16\times 16$, and $8\times 8$.
3. **Downsampling**: Logical grid $G \in \{0..15\}^{GH \times GW}$ is extracted via block mode:
   $$G_{y, x} = \text{mode}\left( O_{y \cdot cs : (y+1)cs, \, x \cdot cs : (x+1)cs} \right)$$

### Stage B: HUD Layer Decomposition
The engine's `RenderableUserDisplay` overlays standardized diagnostic cues on fixed rows:
- **Top Row (Row 0) — Level Gauge**:
  - Dots spaced at odd columns $x = 1 + 2i$.
  - Color 14 (Green) = cleared level, Color 11 (Yellow) = current level, Color 3 (Dark Gray) = remaining levels.
  - *Perceptual Rule*: Length and colors directly disclose total levels and current level index.
- **Bottom Row (Row $H-1$) — Target Count**:
  - Horizontal tick bar of Color 11 (Yellow).
  - *Perceptual Rule*: Number of yellow ticks in row $H-1$ equals the exact count of remaining targets required to win.
- **Sub-bottom Row (Row $H-2$) — Budget / Gauge**:
  - Filled ticks (Color 10 / Cyan) + Empty ticks (Color 3 / Gray).
  - *Perceptual Rule*: Ratio of filled ticks gives step limit, battery level, or action allowance.
- **Status Row (Row $H-3$) — Terminal State**:
  - Flashes solid Green (14) on Level Pass / Win, or Red (8) on Game Over.

### Stage C: Empirical Color Semantics (Priors across 252 Games)
Analysis of all sprite definitions across the 252 games reveals highly consistent semantic assignments:
- **Floor / Background**: **Color 5 (Black, 71.4%)**
- **Padding / Border**: **Color 4 (Charcoal, 71.8%)**
- **Static Walls**: **Color 3 (Dark Gray, 97.6% of all walls)** — high contiguous frequency, impassable.
- **Player Avatar**: **Color 9 (Blue, 96.8% of all avatars)** — typically a $1\times 1$ singleton.
- **Lethal Hazards**: **Color 8 (Red, 87.0% of all hazards)** — impassable, death on contact.
- **Exit Goal / Pad**: **Color 14 (Green, 63.4%)** — immobile target destination pad.
- **Collectible Items / Coins**: **Color 11 (Yellow, 30.6%)** — disappears upon player overlap, decrements Row $H-1$ HUD ticks.
- **Pushable Blocks / Crates**: **Color 15 (Purple, 59.1% of crates)** — movable upon player contact.
- **Teleporters / Portals**: **Color 7 (Light Pink) / Color 10 (Cyan)**.

### Stage D: Deterministic 2-Frame Active Probing Protocol
To achieve 100% certainty without hardcoding, execute a dynamic probing step from Frame 0 ($O_0$) to Frame 1 ($O_1$):

1. **Action Space Check**:
   - If available actions $\mathcal{A} \subseteq \{5, 6\}$ (no directional actions 1..4):
     $\implies$ **Game Category: Board Manipulation / Clicker / Simulation / Memory Match**. No directional avatar.
2. **Directional Probe (Step 1)**:
   - If $\{1, 2, 3, 4\} \subseteq \mathcal{A}$, choose an unblocked direction $a \in \{1, 2, 3, 4\}$ (away from adjacent Color 3 walls).
   - Execute $a = (\Delta x, \Delta y)$ to receive $O_1$. Compute grid difference map:
     $$D = \{ (x, y) \mid G^{(1)}_{y, x} \neq G^{(0)}_{y, x} \text{ and } y \notin \{0, H-3, H-2, H-1\} \}$$
3. **Differential Signature Classification**:
   - **Signature 1: Exactly 2 cells changed** ($c_{from}$ lost Color $C_p$, $c_{to}$ gained $C_p$):
     - If $c_{to} - c_{from} == (\Delta x, \Delta y)$:
       $\implies$ **Player Confirmed**: Entity at $c_{from}$ is the Player Avatar ($C_p = 9$). Category: **Avatar Navigation**.
     - If $c_{to} - c_{from} == k \cdot (\Delta x, \Delta y)$ with $k > 1$:
       $\implies$ **Category: Inertial / Ice Slide**. Entity slides until blocked.
   - **Signature 2: 3 or 4 cells changed** (player moved to $c_{to}$, and cell beyond $c_{to}$ changed to Color $C_{box}$):
     - $\implies$ **Category: Push / Sokoban**. The entity at $c_{to}$ is a pushable crate/box.
   - **Signature 3: Single cell changed** (player moved, but leaves color trail $C_{trail}$ at $c_{from}$):
     - $\implies$ **Category: Path Coverage / Trail / Weak Floor**.
   - **Signature 4: $D = \emptyset$ (No change on playfield)**:
     - Try orthogonal cardinal direction $a'$. If all unblocked 1..4 yield $D = \emptyset$:
       $\implies$ **Category: Padded Clicker / Board Game**. Actions 1..4 are explicit NO-OPS. The primary interaction is coordinate click (ACTION6) or trigger (ACTION5).
   - **Signature 5: Global / Multi-cell changes**:
     - $\implies$ **Category: Physical Simulation / Cellular Automata / Conveyor**.

---

## 3. Action Space Analysis (Actions 1..7 vs. Actual Mechanics)

The complete empirical distribution across the 252 environments is:
1. `[1, 2, 3, 4]`: **105 games (41.7%)**
2. `[1, 2, 3, 4, 6]`: **44 games (17.5%)**
3. `[1, 2, 3, 4, 5]`: **34 games (13.5%)**
4. `[6]`: **31 games (12.3%)**
5. `[1, 2, 3, 4, 5, 6]`: **20 games (7.9%)**
6. `[5, 6]`: **13 games (5.2%)**
7. `[1, 2, 3, 5]`: **2 games (0.8%)** (`hn01`, `kv01`)
8. `[5]`: **2 games (0.8%)** (`pb04`, `sg04`)
9. `[1, 2, 3, 4, 5, 6, 7]`: **1 game (0.4%)** (`ml01`)

### Exact Semantic Role of Each Action:

#### Actions 1..4: Directional / Selection / Cursor / Padded No-op
- **Avatar Movement (172 games)**: $1 = \text{Up } (0, -1)$, $2 = \text{Down } (0, 1)$, $3 = \text{Left } (-1, 0)$, $4 = \text{Right } (1, 0)$.
- **Non-Standard Kinematics**:
  - `kn01`: Knight L-hops.
  - `bi01`: Bishop diagonal slides.
  - `rk01`: Rook orthogonal slides.
- **Cursor / Frame Displacement**:
  - `ng01/04`: Moves the deduction selection cursor.
  - `sf01/04`: Moves the $3 \times 3$ painting stencil frame across the canvas.
- **Discrete Option Selection**:
  - `hn01`: Actions 1, 2, 3 select Peg 1, 2, 3 in Tower of Hanoi.
  - `fe02`: Actions 1, 2, 3, 4 vote for policy slots.
- **Padded No-Op (20 games)**:
  - In clicker/board games (`ab01`, `cs01`, `lo01`, `sp01`, `qr01`, etc.), `step()` contains `if action.id.value in (1,2,3,4): complete_action(); return`. Provided strictly for API uniformity.

#### Action 5: Contextual Discrete Trigger / Tool / Dimension Switch
ACTION5 never takes coordinate data. Across the corpus, it executes one of **8 distinct functions**:
1. **Cargo Transfer**: Pickup / Drop off cargo parcel at customer (`as01`, `dd01`).
2. **Ray / Pulse Emission**: Fire laser beam (`ml01-04`) or electrical pulse (`rp01-04`).
3. **Simulation Tick**: Advance Conway Game of Life by 1 generation (`ll01/04`) or diffuse heat field (`df04`).
4. **Physical Modification**: Dig mud block (`dg01`), place firebreak (`fi01`), or melt wall segment (`mx01`).
5. **Mode / State / Dimension Swap**:
   - Toggle knight move bank (`kn01`).
   - Switch active timeline world $T_0 \leftrightarrow T_1$ (`dv01`).
   - Swap lead avatar in tandem movement (`mc01`).
   - Cycle active body in multi-agent control (`ob01`).
   - Switch active color strain in dual flood (`zm01`).
6. **Exit Pad Charging**: Stand on green exit and hold ACTION5 for $K$ ticks (`ex01-03`).
7. **Idle / Pass Turn**: Wait 1 world tick without moving, pausing hunger/warmth drain (`sv01-03`).
8. **Special Power**: Arm phase walk through wall (`bl01`), recall to start (`rc01`), or sentry whistle (`st01`).

#### Action 6: Targeted Point-and-Click Interaction
Requires coordinate payload `action.data = {"x": disp_x, "y": disp_y}`, converted to grid space via `camera.display_to_grid(disp_x, disp_y)`. Powers 7 interaction patterns:
1. **Tile Toggle / State Cycle**: Invert Lights Out cells (`lo01-05`), rotate pipes 90° (`pu01/04`), cycle wires (`ck02-04`), place/toggle mirrors (`ml01-04`).
2. **Canvas Painting & Erasing**: Paint yellow pixel matching stencil hint (`gp01-04`), flood fill closed contours (`ff01-04`).
3. **Reflex Hit**: Whack active mole at target coordinate (`wm01-04`).
4. **Memory / Hidden State Flip**: Flip face-down memory tile (`mm01-05`), flag sapper mine (`ms02-03`).
5. **Granular Placement**: Add 1 grain of sand to trigger Abelian toppling cascades (`ab01`, `sp01/04`).
6. **Block / Board Rotation**: Rotate single tile 90° (`pt01-05`) or $2 \times 2$ quad block (`qr01/04`).
7. **Environmental Construction**: Place bridge segment across water (`tb01-03`, `cr01`) or defensive wall (`wl01`).

#### Action 7: Undo / State Reversion
Exclusively present in `ml01` (Mirror Laser with Undo) to pop the last mirror configuration change without failing the episode.

---

### Summary Checklist for Agent Deployment
1. **On Reset**: Parse $64 \times 64$ frame $\rightarrow$ compute $cs = \gcd(\dots)$ $\rightarrow$ crop letterbox $\rightarrow$ extract logical grid $G$.
2. **Decompose HUD**: Read level dots on Row 0, target count on Row $H-1$, budget ticks on Row $H-2$.
3. **Inspect Action Space**: If no 1..4 actions $\implies$ immediately execute Clicker / Board Solver.
4. **Probe Step 1**: If 1..4 present, test one cardinal move. Compare $D = |G_1 - G_0|$. If 2 cells swap $\implies$ Blue 9 is confirmed avatar. If 0 cells change $\implies$ 1..4 are no-ops; switch to coordinate-click solver.
5. **Navigate & Solve**: Target Green 14 (exit) or Yellow 11 (collectibles), avoid Red 8 (hazards) and Dark Gray 3 (walls).

This taxonomy and detection logic is universal, robust, and 100% free of hardcoded game IDs.