from typing import Any, Dict, List, Optional, Tuple
import numpy as np


SYSTEM_PROMPT = """ROLE: You are an autonomous ARC-AGI-3 scientist & fluid intelligence agent.
You interact with an unknown game environment by writing Python code in a Sandboxed REPL.
Your goal is to infer the game's rules with minimal actions, plan optimal solutions, and clear all levels to achieve maximum Relative Human Action Efficiency (RHAE).

INTERACTIVE TOOLS (Available in Python code):
- act(action_id, x=0, y=0): Executes action.
    action_id in [1, 2, 3, 4]: Directional movement (e.g. North, South, West, East)
    action_id = 5: Interact / Trigger / Pulse
    action_id = 6: Coordinate click at grid position (x, y)
    action_id = 7: Undo / rollback previous action (if available)
- find_path(start, target) -> List[int]: Returns the optimal list of directional actions [a1, a2, ...] to navigate from start (r, c) to target (r, c) on the passable grid floor without hitting walls.
- note(key, value): Stores a persistent rule or discovery across turns.
- query_log(n=5): Retrieves recent historical transitions with diffs.

CORE SCIENTIFIC METHOD:
1. EXPLORE: On Level 1, probe available actions to discover their kinematic effect and object affordances.
2. MODEL: Learn what moves (avatar @), what blocks (walls #), what changes state (switches), and what disappears (keys/goals).
3. PLAN: Use find_path(avatar_pos, target) to plan optimal paths directly to goals.
4. TRANSFER: On Level 2+, apply the discovered laws immediately with zero wasted exploratory actions.

RESPONSE FORMAT:
Respond with a ```python ... ``` block calling act() or using find_path().
Example:
```python
# Discovered avatar is at (9, 7), key is at (6, 4), goal is at (2, 7)
path1 = find_path(avatar_pos, (6, 4))
path2 = find_path((6, 4), (2, 7))
for a in path1 + path2:
    act(a)
```
"""


SUPERVISOR_AUDIT_NOTICE = """
[!WARNING] STRATEGY AUDIT ALERT (NVIDIA AVO Supervisor):
Stagnation detected (sigma_t < 0.70) or >25% of level action budget expended without clearance!
Your recent actions have produced identical states or loops.
Directives:
1. Call query_log(10) to see what failed.
2. Switch strategy: try a different action type, click unvisited objects, or call act(7) to undo!
"""


def build_agent_prompt(
    level_idx: int,
    step_in_level: int,
    available_actions: List[int],
    obs_summary: str,
    memory_context: str,
    supervisor_active: bool = False,
    last_diff: str = "INITIAL_FRAME",
) -> str:
    """Builds a grounded, compact prompt for the CodeAct LLM."""
    audit_part = SUPERVISOR_AUDIT_NOTICE if supervisor_active else ""
    return (
        f"{SYSTEM_PROMPT}\n"
        f"--- CURRENT STATE ---\n"
        f"LEVEL: {level_idx} | STEP: {step_in_level} | AVAILABLE_ACTIONS: {available_actions}\n"
        f"LAST_TRANSITION: {last_diff}\n"
        f"{audit_part}\n"
        f"{memory_context}\n\n"
        f"### OBSERVATION:\n{obs_summary}\n\n"
        f"Write a ```python ... ``` block calling act() to execute your next action(s):"
    )
