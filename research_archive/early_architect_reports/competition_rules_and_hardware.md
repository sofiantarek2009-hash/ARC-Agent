# Competition Rules & Hardware Architecture Research

**Source Conversation ID:** `1acaa2b7-1407-4332-9607-776e6da3ece0`

## Original Prompt
```text
<USER_REQUEST>
Research the ARC-AGI-3 Kaggle competition thoroughly. I need to understand:

1. **Competition rules**: How does ARC-AGI-3 work? What is the interactive environment API? What actions are available? How is scoring done (RHAE)?
2. **The actual evaluation format**: How does the Kaggle submission work? What's the API the agent interacts with?
3. **Hardware constraints**: What hardware is available in the Kaggle notebook?
4. **Time limits**: What's the actual time budget?

Search for:
- The official ARC-AGI-3 Kaggle competition page
- ARC Prize 2025 rules and documentation
- The arc-agi-3 pip package documentation
- Any official blog posts about ARC-AGI-3
- The GitHub repo: https://github.com/arcprize/arc-agi

Provide exact details about the submission format, the Python API the agent must use, and the scoring formula.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-11T21:25:19+03:00.
</ADDITIONAL_METADATA>
```

---

## Architecture / Research Report

# Comprehensive Research Report: ARC-AGI-3 Kaggle Competition

---

### Executive Overview & Competition Context
- **Competition Name**: ARC Prize 2026 - ARC-AGI-3
- **Platform**: Kaggle Code Competition ([Competition Page](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3))
- **Organizer**: ARC Prize Foundation (launched March 25, 2026; final submission deadline November 2, 2026; awards ceremony December 4, 2026)
- **Prize Pool**: $850,000 total pool for ARC-AGI-3 ($700,000 Grand Prize for first eligible agent scoring 100% RHAE, $75,000 guaranteed leaderboard prizes, $75,000 milestone open-source prizes). Total ARC Prize 2026 initiative exceeds $2,000,000.
- **Core Mission**: Shift AI benchmarking from static input-output pattern recognition (ARC-AGI-1 and 2) to **fluid, agentic intelligence** in interactive, novel, turn-based environments with zero natural language guidance, text instructions, or pre-stated goals.

---

## 1. Competition Rules & Mechanics

### 1.1 Environment Architecture
- **Interactive 2D Grid Worlds**: Environments are turn-based 2D grid games powered by the underlying `arcengine` sprite/grid engine.
- **Grid Constraints**:
  - Maximum grid dimensions: $64 \times 64$ pixels/cells.
  - Cell values: 16 discrete integer values ($0 \dots 15$), representing distinct colors/palette indices.
- **Core Knowledge Priors**: Built on Core Knowledge priors (objectness, persistence, basic geometry, topology, agentness, simple physics like gravity and collision) rather than domain trivia or language.
- **Zero In-Game Instructions**: Agents are given no hints, goals, or mechanics up-front; they must explore to deduce rules, hypothesize win states, and solve multi-step tasks.
- **Multi-Level Progression**: Each game environment contains sequential levels (typically increasing in complexity). Completing a level unlocks the subsequent level.

### 1.2 Available Action Space
The environment standardizes interactions via discrete commands encapsulated by the `GameAction` enumeration:

| Action | Semantic Mapping / Key | Description |
| :--- | :--- | :--- |
| `GameAction.ACTION1` | Up (`W` / `1`) | Move or interact upward |
| `GameAction.ACTION2` | Down (`S` / `2`) | Move or interact downward |
| `GameAction.ACTION3` | Left (`A` / `3`) | Move or interact leftward |
| `GameAction.ACTION4` | Right (`D` / `4`) | Move or interact rightward |
| `GameAction.ACTION5` | Interact / Select (`Space` / `F`) | Primary interaction, select, or trigger |
| `GameAction.ACTION6` | Click / Coordinate Input | **Complex action**: Requires coordinates `x` and `y` in the range $[0, 63]$ |
| `GameAction.ACTION7` | Undo | Reverts previous step (where supported) |
| `GameAction.RESET` | Restart / Initialize | Restarts the current level or environment |

*Note on `ACTION6` coordinate calls*: Coordinate parameters are passed via action data dictionaries:
```python
obs = env.step(GameAction.ACTION6, data={"x": target_x, "y": target_y})
```

### 1.3 State & Observations (`FrameData`)
Each step returns or updates `FrameData` snapshots:
- `latest_frame.frame`: 2D array / matrix ($H \times W \le 64 \times 64$) of discrete cell values ($0 \dots 15$).
- `latest_frame.state`: Member of `GameState` enum (`GameState.NOT_PLAYED`, `GameState.PLAYING`, `GameState.WIN`, `GameState.GAME_OVER`).
- `latest_frame.levels_completed`: Number of levels successfully cleared so far.
- `latest_frame.win_levels`: Total levels required to complete the game.
- `latest_frame.available_actions`: Valid subset of `GameAction`s available for the current frame.
- `frames`: List of historical `FrameData` frames since game start/reset.

---

## 2. Scoring Metric: RHAE (Relative Human Action Efficiency)

RHAE (pronounced "Ray") measures how efficiently an AI agent discovers the rules and solves tasks relative to calibrated human players.

### 2.1 Level Score Formula
For each level $l$ within a game:

$$\text{Level Score}_l = \min\left(1.0, \left(\frac{\text{Human Baseline Actions}_l}{\text{AI Actions}_l}\right)^2\right)$$

- **Human Baseline**: The calibrated **upper-median** number of actions taken by human playtesters solving the level for the first time.
- **The Quadratic Penalty**: Because the ratio is squared:
  - If AI matches human efficiency ($1\times$ actions): $\text{Score} = 1.0$ ($100\%$).
  - If AI takes $2\times$ human actions: $\text{Score} = (0.5)^2 = 0.25$ ($25\%$).
  - If AI takes $5\times$ human actions: $\text{Score} = (0.2)^2 = 0.04$ ($4\%$).
- **Hard Action Cutoff (Budget Limit)**:
  - If the agent exceeds **$5\times$ the human baseline actions** for a level, execution for that level is terminated and receives a score of **$0.0$**.
  - If the agent fails to reach the terminal win state (`GameState.WIN`), the level receives a score of **$0.0$**.

### 2.2 Game Score Formula
Games feature sequential levels $l \in \{1, \dots, L\}$. Because initial levels often serve as tutorial levels while later levels test full mastery, levels are weighted linearly by their 1-indexed level number:

$$\text{Game Score} = \frac{\sum_{l=1}^L l \times \text{Level Score}_l}{\sum_{l=1}^L l}$$

- All levels must be cleared to reach $100\%$ on a game.

### 2.3 Overall Competition Leaderboard Score
The overall leaderboard score is the arithmetic mean of game scores across all games in the evaluation test set:

$$\text{Overall Score} = \frac{1}{N} \sum_{i=1}^N \text{Game Score}_i \times 100\%$$

---

## 3. Python API & Development SDK

### 3.1 Installation
The official toolkit is packaged as `arc-agi`:
```bash
pip install arc-agi
# or using uv:
uv add arc-agi
```
Core underlying dependencies include `arcengine` (Rust-backed high-speed engine achieving 2,000+ FPS in headless execution).

### 3.2 Repositories
- **Starter Kit**: [https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
- **Agent Templates**: [https://github.com/arcprize/ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents)
- **Main SDK / Toolkit**: [https://github.com/arcprize/ARC-AGI](https://github.com/arcprize/ARC-AGI)

### 3.3 Agent Implementation Contract
In the starter kit, participants implement their agent inside `agent/my_agent.py` by subclassing `Agent`:

```python
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

class MyAgent(Agent):
    def __init__(self):
        super().__init__()
        # Initialize internal models, planner, or local neural weights

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Determines if the agent concludes interaction on this level."""
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Inspects current observation and emits the next GameAction."""
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET
        
        # Access current grid (list of lists or numpy array)
        grid = latest_frame.frame
        available = latest_frame.available_actions
        
        # Example decision logic:
        # return GameAction.ACTION1
        # or for coordinate clicks:
        action = GameAction.ACTION6
        action.set_data({"x": 12, "y": 24})
        return action
```

### 3.4 Local Testing Loop (`make play-local`)
Under the hood, local execution uses the `Arcade` class:
```python
import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")

obs = env.step(GameAction.ACTION1)
scorecard = arc.get_scorecard()
```

---

## 4. Evaluation Format & Kaggle Submission Workflow

### 4.1 Sandboxed Execution Environment
- **Internet Access**: Strictly **DISABLED** during evaluation reruns.
- **Pre-trained Models & Dependencies**: Any weights, offline wheels, or language models must be uploaded as private or public **Kaggle Datasets** and attached to the submission notebook.
- **Open-Source Requirement**: To claim leaderboard and milestone prizes, winning solutions and methodologies must be open-sourced.

### 4.2 The Gateway Sidecar Architecture & Dual-Phase Notebook Pattern
The evaluation uses a sidecar architecture. A hidden local gateway service runs inside the Kaggle submission container at `http://gateway:8001/api/games` (`ARC_BASE_URL=http://gateway:8001/`).

The Kaggle notebook submission utilizes a two-phase detection pattern:

1. **Phase A (Commit / Save Version)**:
   - When a user clicks "Save Version" in Kaggle, `KAGGLE_IS_COMPETITION_RERUN` is **unset**.
   - The notebook quickly writes a dummy `/kaggle/working/submission.parquet` and finishes in ~30 seconds, preventing wasted compute during routine saves.
2. **Phase B (Competition Submission Rerun)**:
   - When the user clicks "Submit to Competition", Kaggle spins up the rerun container and sets `KAGGLE_IS_COMPETITION_RERUN=true`.
   - The competition environment activates the `gateway:8001` sidecar serving hidden game environments.
   - The runner executes `python main.py --agent <myagent>`, which queries `http://gateway:8001/`, steps through all hidden games, and dumps the actual evaluated results to `submission.parquet`.

### 4.3 Output File Format (`submission.parquet`)
The final output file must be saved to `/kaggle/working/submission.parquet` with `index=False`:
```python
submission.to_parquet('/kaggle/working/submission.parquet', index=False)
```

**Required Columns in `submission.parquet`**:
1. `row_id` (integer / string identifier per record)
2. `game_id` (string identifier of the game)
3. `end_of_game` (boolean flag indicating game completion)
4. `score` (computed RHAE game score)

---

## 5. Hardware Constraints & Time Limits

### 5.1 Hardware Constraints
- **Accelerator**: **NVIDIA RTX 6000 Pro** (Workstation/Pro Ada line) with **96 GB VRAM**.
- **Impact on Solutions**:
  - The massive 96 GB VRAM allows participants to run large open-weights models locally without quantization bottlenecks (e.g. Qwen-2.5, DeepSeek, or Llama 70B quantized to FP8/INT4, or full-precision 14B–32B multimodal vision-language models).
  - High host CPU RAM and multi-core CPU availability.

### 5.2 Time Limits
- **Notebook Runtime Limit**: **9 hours** total run budget (extended by organizers from 6 hours to accommodate 96 GB RTX 6000 Pro compute demands).
- **Per-Game / Per-Level Constraints**:
  - No explicit wall-clock timeout per individual action, but the entire hidden test suite must complete within the 9-hour notebook limit.
  - Action steps are bounded by the $5\times$ human baseline action cutoff per level.

### 5.3 Dataset Splits
- **Public Set**: 25 games (183 levels total), accessible via the public ARC-AGI-3 API for local benchmarking and harness development.
- **Private Evaluation Set**: Hidden set of novel, unseen games evaluated exclusively during the Phase B Kaggle rerun. Agent generalization across novel visual priors and game mechanics is required to prevent overfitting to the public 25 games.

## Additional Synthesized Content

I have conducted a thorough investigation of the ARC-AGI-3 Kaggle competition, official documentation, source code repositories, and technical discussions, and sent the full detailed report to the caller agent. 

### Key Findings Summary:
1. **Competition Mechanics & Action Space**:
   - Evaluated on interactive, turn-based 2D grid environments (up to $64 \times 64$ grid, 16 discrete integer cell values) with zero textual guidance or pre-stated goals.
   - Action space (`GameAction` enum): `ACTION1` (Up), `ACTION2` (Down), `ACTION3` (Left), `ACTION4` (Right), `ACTION5` (Interact/Space), `ACTION6` (Click with $x, y \in [0, 63]$), `ACTION7` (Undo), and `RESET`.
2. **RHAE Scoring Formula**:
   - Per-level score: $\min\left(1.0, \left(\frac{\text{Human Actions}}{\text{AI Actions}}\right)^2\right)$, with a hard cutoff at $5\times$ human baseline actions (exceeding it yields $0.0$).
   - Per-game score: Linearly level-weighted average $\frac{\sum_{l=1}^L l \cdot \text{Score}_l}{\sum_{l=1}^L l}$.
   - Overall leaderboard score: Mean of game scores across the hidden evaluation set ($0\% - 100\%$).
3. **Evaluation Format**:
   - Sandboxed Kaggle code run with **internet access disabled**.
   - Dual-phase architecture: Phase A generates a dummy file for notebook commits; Phase B connects to `http://gateway:8001/api/games` (`ARC_BASE_URL=http://gateway:8001/`) to run against the hidden test suite.
   - Generates `/kaggle/working/submission.parquet` with columns `row_id`, `game_id`, `end_of_game`, `score`.
4. **Hardware & Time Limits**:
   - **Accelerator**: NVIDIA RTX 6000 Pro (96 GB VRAM).
   - **Time Budget**: 9 hours total notebook execution limit.