from typing import Any, Dict, List, Optional, Tuple
import numpy as np


SYSTEM_PROMPT = """ROLE: You are an autonomous ARC-AGI-3 scientist & fluid intelligence agent.
You interact with an unknown game environment by writing Python code in a Sandboxed REPL.
Your goal is to infer the game's rules with minimal actions, plan optimal solutions, and clear all levels to achieve maximum Relative Human Action Efficiency (RHAE).

INTERACTIVE TOOLS (Available in Python code):
- act(action_id, x=0, y=0): Executes action.
    action_id in [1, 2, 3, 4]: Directional movement (e.g. North, South, West, East)
    action_id = 5: Interact / Trigger / Pulse / Wait
    action_id = 6: Coordinate click at grid position (x, y)
    action_id = 7: Undo / rollback previous action (if available)
- find_path(start, target, hazards=None) -> List[int]: Returns optimal directional actions [a1, a2, ...] to navigate from start to target on the passable grid floor. Supports multiple targets (TSP tour) and dynamic falling hazards in (r, c, t) space-time.
- sokoban_solver(boxes, targets, avatar_pos=None) -> List[int]: Computes exact push action sequence to place all boxes onto targets, pruning concave corner and line deadlocks.
- solve_circuit_flow(terminals, rotatable_tiles) -> List[Tuple[int, dict]]: Solves pipe/circuit tile rotations to connect source and sink terminals via Action 6 clicks.
- triangulate_beacon(probes) -> Tuple[int, int]: Triangulates exact hidden goal coordinates from 3+ Manhattan distance probes [(r, c, dist), ...].
- escort_joint_astar(avatar_pos, escort_pos, goal_pos, walls) -> List[int]: Computes actions for coupled escort movement in joint product space.
- delayed_action_astar(start_pos, goal_pos, walls, delay_k=3) -> List[int]: Computes actions for delayed-action pipeline queues.
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
    action_map: Optional[Dict[Tuple[int, int], int]] = None,
    targets: Optional[List[Tuple[int, int]]] = None,
    avatar_pos: Optional[Tuple[int, int]] = None,
) -> str:
    """Builds a grounded, compact prompt for the CodeAct LLM."""
    audit_part = SUPERVISOR_AUDIT_NOTICE if supervisor_active else ""
    grounded_part = ""
    if action_map and targets and avatar_pos:
        grounded_part = (
            f"--- KINEMATICS & TARGETS VERIFIED ---\n"
            f"Action-to-Displacement Map: {action_map}\n"
            f"Avatar Grid Position: {avatar_pos}\n"
            f"Targets to Gather/Reach: {targets}\n"
            f"CRITICAL: Do NOT guess moves or probe blindly. Use find_path(avatar_pos, targets) to solve:\n"
            f"```python\n"
            f"plan = find_path(avatar_pos, targets)\n"
            f"for a in plan:\n"
            f"    act(a)\n"
            f"```\n\n"
        )
    return (
        f"{SYSTEM_PROMPT}\n"
        f"--- CURRENT STATE ---\n"
        f"LEVEL: {level_idx} | STEP: {step_in_level} | AVAILABLE_ACTIONS: {available_actions}\n"
        f"LAST_TRANSITION: {last_diff}\n"
        f"{audit_part}\n"
        f"{grounded_part}"
        f"{memory_context}\n\n"
        f"### OBSERVATION:\n{obs_summary}\n\n"
        f"Write a ```python ... ``` block calling act() to execute your next action(s):"
    )
