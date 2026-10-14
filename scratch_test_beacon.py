import sys
from pathlib import Path
import numpy as np
from collections import deque

sys.path.insert(0, "/Applications/Dad Company")

from arcengine import GameAction
import importlib.util

def load_game_cls(rel_path, cls_name):
    full_path = Path("/Applications/Dad Company") / rel_path
    spec = importlib.util.spec_from_file_location(cls_name, str(full_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)

Bn02 = load_game_cls("arc-interactive/environment_files/bn02/63be02fb/bn02.py", "Bn02")
Bn03 = load_game_cls("arc-interactive/environment_files/bn03/63be02fb/bn03.py", "Bn03")

class BeaconSolver:
    def __init__(self):
        self.reset_level()

    def reset_level(self):
        self.flagged_coords = set()
        self.deployed_positions = set()
        self.player_pos = None
        self.last_action = None

    def choose_action(self, grid, legal_actions):
        h, w = grid.shape
        # 1. Track player pos
        p_pts = np.argwhere((grid == 9) & (np.arange(h)[:, None] >= 4))
        if len(p_pts) > 0:
            self.player_pos = (int(p_pts[0, 0]), int(p_pts[0, 1]))
        elif self.player_pos is not None and self.last_action in [1, 2, 3, 4]:
            deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
            dr, dc = deltas.get(self.last_action, (0, 0))
            nr, nc = self.player_pos[0] + dr, self.player_pos[1] + dc
            if 4 <= nr < 60 and 4 <= nc < 60 and grid[nr, nc] != 3:
                self.player_pos = (nr, nc)
        
        pr, pc = self.player_pos or (32, 32)

        # 1. Click unflagged ghosts
        g_pts = np.argwhere((grid == 2) & (np.arange(h)[:, None] >= 4))
        unflagged = [tuple(p) for p in g_pts if (int(p[0]), int(p[1])) not in self.flagged_coords]
        if unflagged:
            gr, gc = int(unflagged[0][0]), int(unflagged[0][1])
            self.flagged_coords.add((gr, gc))
            if not hasattr(self, "discovered_ghost_locs"):
                self.discovered_ghost_locs = []
            self.discovered_ghost_locs.append((gr, gc))
            self.last_action = 6
            return 6, {"x": gc, "y": gr}

        # 2. If at start of level and no beacon deployed anywhere yet: deploy at spawn
        if len(self.deployed_positions) == 0:
            self.deployed_positions.add((pr, pc))
            self.last_action = 5
            return 5, {}

        # 3. Determine waypoints and dist_thresh
        is_shrinking = np.any(grid[3, 1:25] == 10)
        if is_shrinking:
            waypoints = []
            if hasattr(self, "discovered_ghost_locs") and self.discovered_ghost_locs:
                for gr, gc in self.discovered_ghost_locs:
                    for dr, dc in [(2, 2), (-2, -2), (2, -2), (-2, 2)]:
                        waypoints.append((max(6, min(58, gr + dr)), max(6, min(58, gc + dc))))
            waypoints += [(14, 14), (14, 50), (50, 50), (50, 14), (32, 32)]
            dist_thresh = 3
            uncovered = [
                wp for wp in waypoints
                if not any(abs(wp[0] - dp[0]) + abs(wp[1] - dp[1]) <= dist_thresh for dp in self.deployed_positions)
            ]
        else:
            primary_wps = [(10, 10), (10, 56), (56, 56), (56, 10), (32, 32), (38, 38)]
            dist_thresh = 6
            uncovered = [
                wp for wp in primary_wps
                if not any(abs(wp[0] - dp[0]) + abs(wp[1] - dp[1]) <= dist_thresh for dp in self.deployed_positions)
            ]
            if not uncovered:
                secondary_wps = [(10, 22), (10, 34), (10, 46)]
                uncovered = [
                    wp for wp in secondary_wps
                    if not any(abs(wp[0] - dp[0]) + abs(wp[1] - dp[1]) <= dist_thresh for dp in self.deployed_positions)
                ]

        # Check beacon budget
        beacon_budget = np.sum(grid[1, 1:25] == 12)
        if beacon_budget == 0:
            # No beacons left, can only click or move
            fallback = 1 if 1 in legal_actions else legal_actions[0]
            self.last_action = fallback
            return fallback, {}

        if uncovered:
            uncovered.sort(key=lambda wp: abs(wp[0] - pr) + abs(wp[1] - pc))
            for target in uncovered:
                act = self._step_towards((pr, pc), target, grid)
                if act is not None:
                    self.last_action = act
                    return act, {}
                else:
                    # Arrived at or adjacent to target! Deploy beacon here
                    self.deployed_positions.add((pr, pc))
                    self.last_action = 5
                    return 5, {}

        fallback = 1 if 1 in legal_actions else legal_actions[0]
        self.last_action = fallback
        return fallback, {}

    def _step_towards(self, start, target, grid):
        sr, sc = start
        tr, tc = target
        h, w = grid.shape
        # If target is on a wall, find nearest passable cell
        if grid[tr, tc] == 3:
            passable_nbrs = [
                (tr + dr, tc + dc) for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
                if 4 <= tr + dr < 60 and 4 <= tc + dc < 60 and grid[tr + dr, tc + dc] != 3
            ]
            if passable_nbrs:
                tr, tc = passable_nbrs[0]

        if (sr, sc) == (tr, tc):
            return None

        q = deque([(sr, sc, [])])
        visited = {(sr, sc)}
        while q:
            cr, cc, path = q.popleft()
            if (cr, cc) == (tr, tc) or (abs(cr - tr) + abs(cc - tc) <= 1 and len(path) > 0):
                return path[0] if path else None
            for act, (dr, dc) in [(1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1))]:
                nr, nc = cr + dr, cc + dc
                if 4 <= nr < 60 and 4 <= nc < 60 and (nr, nc) not in visited:
                    if grid[nr, nc] != 3:  # not a wall
                        visited.add((nr, nc))
                        q.append((nr, nc, path + [act]))
        return None

def test_game_with_arcade(game_name):
    from arc_agi import Arcade, OperationMode
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='/Applications/Dad Company/arc-interactive/environment_files')
    env = arcade.make(game_name)
    solver = BeaconSolver()
    total_actions = 0
    obs = env.reset()
    win_levels = getattr(obs, "win_levels", 5)
    last_level = 0
    solver.reset_level()
    print(f"\nTesting {game_name} ({win_levels} levels) via Arcade...")
    
    act_cnt = 0
    level_actions = 0
    while act_cnt < 1000:
        if not obs.frame:
            break
        grid = np.array(obs.frame[-1])
        act, act_data = solver.choose_action(grid, obs.available_actions)
        obs = env.step(act, data=act_data) if act_data else env.step(act)
        act_cnt += 1
        level_actions += 1

        curr_level = getattr(obs, "levels_completed", 0)
        if curr_level > last_level:
            print(f"  Level {last_level}: WON in {level_actions} actions (total {act_cnt})")
            last_level = curr_level
            level_actions = 0
            solver.reset_level()
            if curr_level >= win_levels:
                break
        if str(getattr(obs, "state", "")).upper() in ["WIN", "GAME_OVER"]:
            if "GAME_OVER" in str(getattr(obs, "state", "")).upper():
                print(f"  Level {last_level}: GAME OVER after {level_actions} actions!")
            break
            
    print(f"{game_name} Final: {last_level}/{win_levels} cleared, {act_cnt} total actions")

if __name__ == "__main__":
    test_game_with_arcade("bn02")
    test_game_with_arcade("bn03")
