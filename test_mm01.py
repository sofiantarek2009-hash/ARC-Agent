from arc_agi import Arcade, OperationMode
from arcengine import GameAction
import numpy as np
import collections
import traceback

arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="/Applications/Dad Company/arc-interactive/environment_files")
env = arcade.make("mm01")
obs = env.reset()

class MemoryGameSolver:
    def __init__(self):
        self.reset_level()
        
    def reset_level(self):
        self.rows = None
        self.cols = None
        self.ts = None
        self.centers = []
        self.tile_memory = {}
        self.cleared = set()
        self.turn_first_card = None
        self.waiting_ticks = 0
        self.last_clicked_center = None
        self.last_was_dummy = False

    def choose_action(self, grid):
        if self.waiting_ticks > 0:
            self.waiting_ticks -= 1
            self.last_was_dummy = True
            return {"x": 0, "y": 0}

        self.last_was_dummy = False
        if not self.centers:
            # Click near center of hidden cards region
            pts = np.argwhere((grid == 3) & (np.arange(64)[:, None] >= 4))
            if len(pts) > 0:
                r_mid = int(pts[:, 0].mean())
                c_mid = int(pts[:, 1].mean())
            else:
                r_mid, c_mid = 32, 32
            self.last_clicked_center = (c_mid, r_mid)
            return {"x": c_mid, "y": r_mid}

        if self.turn_first_card is None:
            # Check if any pair in memory is ready to be collected
            color_to_tiles = collections.defaultdict(list)
            for c, col in self.tile_memory.items():
                if c not in self.cleared:
                    color_to_tiles[col].append(c)
            for col, tlist in color_to_tiles.items():
                if len(tlist) >= 2:
                    c1, c2 = tlist[0], tlist[1]
                    self.last_clicked_center = c1
                    return {"x": c1[0], "y": c1[1]}

            unvisited = [c for c in self.centers if c not in self.tile_memory and c not in self.cleared]
            if not unvisited:
                unvisited = [c for c in self.centers if c not in self.cleared]
            if not unvisited:
                unvisited = list(self.centers)
            c_pick = unvisited[0]
            self.last_clicked_center = c_pick
            return {"x": c_pick[0], "y": c_pick[1]}

        else:
            c1, col1 = self.turn_first_card
            matching_in_mem = [c for c, col in self.tile_memory.items() if col == col1 and c != c1 and c not in self.cleared]
            if matching_in_mem:
                c2 = matching_in_mem[0]
                self.last_clicked_center = c2
                return {"x": c2[0], "y": c2[1]}
            else:
                unvisited = [c for c in self.centers if c not in self.tile_memory and c not in self.cleared and c != c1]
                if not unvisited:
                    unvisited = [c for c in self.centers if c not in self.cleared and c != c1]
                if not unvisited:
                    unvisited = list(self.centers)
                c_pick = unvisited[0]
                self.last_clicked_center = c_pick
                return {"x": c_pick[0], "y": c_pick[1]}

    def on_step_result(self, grid_before, grid_after):
        if self.last_was_dummy:
            return

        diff = (grid_before != grid_after) & (np.arange(64)[:, None] >= 4)
        if not np.any(diff):
            return

        # Mask out 5-pixel click cross marker
        cross_mask = np.zeros_like(diff)
        if self.last_clicked_center:
            cx, cy = self.last_clicked_center
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                if 0 <= cy + dy < 64 and 0 <= cx + dx < 64:
                    cross_mask[cy + dy, cx + dx] = True

        card_diff = diff & ~cross_mask
        if not np.any(card_diff):
            card_diff = diff

        vals, cnts = np.unique(grid_after[card_diff], return_counts=True)
        cand_indices = [i for i, v in enumerate(vals) if v not in (5, 3)]
        if not cand_indices:
            return
        best_idx = max(cand_indices, key=lambda i: cnts[i])
        dom_color = vals[best_idx]

        if not self.centers:
            card_mask = card_diff & (grid_after == dom_color)
            c_pts = np.argwhere(card_mask)
            if len(c_pts) > 0:
                tr0, tc0 = c_pts.min(axis=0)
                tr1, tc1 = c_pts.max(axis=0)
                self.ts = max(tr1 - tr0 + 1, tc1 - tc0 + 1)
                hidden_color = grid_before[tr0, tc0]
                h_pts = np.argwhere((grid_before == hidden_color) & (np.arange(64)[:, None] >= 4))
                hr0, hc0 = h_pts.min(axis=0)
                hr1, hc1 = h_pts.max(axis=0)
                self.rows = max(1, (hr1 - hr0 + 1) // self.ts)
                self.cols = max(1, (hc1 - hc0 + 1) // self.ts)
                self.centers = []
                for r in range(self.rows):
                    for c in range(self.cols):
                        self.centers.append((hc0 + int((c + 0.5) * self.ts), hr0 + int((r + 0.5) * self.ts)))
                closest_c = min(self.centers, key=lambda pt: (pt[0]-self.last_clicked_center[0])**2 + (pt[1]-self.last_clicked_center[1])**2)
                self.tile_memory[closest_c] = int(dom_color)
                self.turn_first_card = (closest_c, int(dom_color))
                return

        closest_c = min(self.centers, key=lambda pt: (pt[0]-self.last_clicked_center[0])**2 + (pt[1]-self.last_clicked_center[1])**2)
        self.tile_memory[closest_c] = int(dom_color)
        if self.turn_first_card is None:
            self.turn_first_card = (closest_c, int(dom_color))
        else:
            c1, col1 = self.turn_first_card
            c2, col2 = closest_c, int(dom_color)
            if col1 == col2:
                self.cleared.add(c1)
                self.cleared.add(c2)
                self.turn_first_card = None
            else:
                self.waiting_ticks = 2
                self.turn_first_card = None

if __name__ == "__main__":
    solver = MemoryGameSolver()
    last_level = 0
    action_count = 0

    for step in range(350):
        if not hasattr(obs, "frame") or len(obs.frame) == 0:
            print(f"Step {step}: obs.frame empty! State: {obs.state}")
            break
        grid_curr = obs.frame[-1]

        if obs.levels_completed > last_level:
            print(f"--> Level {last_level+1} CLEARED! Steps taken: {action_count}")
            last_level = obs.levels_completed
            solver.reset_level()
            action_count = 0
            grid_curr = obs.frame[-1]

        if "WIN" in str(obs.state) or obs.levels_completed >= 7:
            print(f"==> GAME WON! Total levels: {obs.levels_completed}")
            break

        act_dict = solver.choose_action(grid_curr)
        obs = env.step(GameAction.from_id(6), data=act_dict)
        if hasattr(obs, "frame") and len(obs.frame) > 0:
            grid_next = obs.frame[-1]
            solver.on_step_result(grid_curr, grid_next)
        action_count += 1

    print(f"Final state: {obs.state}, levels: {obs.levels_completed}, total steps: {step}")
