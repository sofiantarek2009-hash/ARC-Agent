from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class JordanCurveCavityInducer:
    """Detects hollow frames, target receptacles, and enclosed cavities."""

    @staticmethod
    def find_cavities(grid: np.ndarray, bg_color: int, bezel_px: int = 4) -> List[Tuple[int, int]]:
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        cavities: List[Tuple[int, int]] = []
        for r in range(bezel_px + 1, h - bezel_px - 1):
            for c in range(bezel_px + 1, w - bezel_px - 1):
                if int(arr[r, c]) != bg_color:
                    continue
                neighbors = [
                    int(arr[r - 1, c]),
                    int(arr[r + 1, c]),
                    int(arr[r, c - 1]),
                    int(arr[r, c + 1]),
                ]
                non_bg = [n for n in neighbors if n != bg_color]
                if len(non_bg) >= 3 and len(set(non_bg)) == 1:
                    cavities.append((r, c))
        return cavities


class DihedralSymmetrySolver:
    """Full D4 Dihedral Split-Panel Symmetry & Odd-One-Out Pattern Completion Solver."""

    @staticmethod
    def _d4_transforms(panel: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        transforms = [
            ("id", panel),
            ("fliplr", np.fliplr(panel)),
            ("flipud", np.flipud(panel)),
            ("rot180", np.rot90(panel, 2)),
        ]
        if panel.shape[0] == panel.shape[1]:
            transforms.extend([
                ("rot90", np.rot90(panel, 1)),
                ("rot270", np.rot90(panel, 3)),
                ("transpose", panel.T),
                ("antitranspose", np.fliplr(np.rot90(panel, 1))),
            ])
        return transforms

    @classmethod
    def find_symmetry_clicks(
        cls, grid: np.ndarray, bg_color: int, bezel_px: int = 4
    ) -> List[Tuple[int, int]]:
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        core = arr[bezel_px:h - bezel_px, bezel_px:w - bezel_px]
        ch, cw = core.shape
        clicks: List[Tuple[int, int]] = []

        half_w = cw // 2
        left = core[:, :half_w - 1]
        right = core[:, cw - left.shape[1]:]
        if left.shape == right.shape and left.size > 0:
            for _, t_left in cls._d4_transforms(left):
                diff = t_left != right
                n_diff = int(np.sum(diff))
                if 1 <= n_diff <= 48:
                    rs, cs = np.where(diff)
                    for r, c in zip(rs, cs):
                        clicks.append((int(r + bezel_px), int(c + bezel_px + (cw - left.shape[1]))))
                    return clicks

        half_h = ch // 2
        top = core[:half_h - 1, :]
        bot = core[ch - top.shape[0]:, :]
        if top.shape == bot.shape and top.size > 0:
            for _, t_top in cls._d4_transforms(top):
                diff = t_top != bot
                n_diff = int(np.sum(diff))
                if 1 <= n_diff <= 48:
                    rs, cs = np.where(diff)
                    for r, c in zip(rs, cs):
                        clicks.append((int(r + bezel_px + (ch - top.shape[0])), int(c + bezel_px)))
                    return clicks

        return clicks


class LatentMemoryMatcher:
    """Tracks hidden-state tile reveals in Concentration / Memory-Match games."""

    def __init__(self):
        self.known_tiles: Dict[Tuple[int, int], int] = {}
        self.matched_positions: Set[Tuple[int, int]] = set()

    def reset(self) -> None:
        self.known_tiles.clear()
        self.matched_positions.clear()

    def observe_reveal(
        self, prev_grid: np.ndarray, curr_grid: np.ndarray, click_xy: Tuple[int, int], bg_color: int
    ) -> Optional[Tuple[int, int]]:
        cx, cy = click_xy
        f0 = np.asarray(prev_grid, dtype=np.int8)
        f1 = np.asarray(curr_grid, dtype=np.int8)
        if not (0 <= cy < f1.shape[0] and 0 <= cx < f1.shape[1]):
            return None
        new_color = int(f1[cy, cx])
        old_color = int(f0[cy, cx])
        if new_color != old_color and new_color != bg_color:
            key = (int(round(cx / 4.0) * 4), int(round(cy / 4.0) * 4))
            for other_xy, c in list(self.known_tiles.items()):
                if c == new_color and abs(other_xy[0] - key[0]) + abs(other_xy[1] - key[1]) >= 4:
                    if other_xy not in self.matched_positions:
                        self.matched_positions.add(other_xy)
                        self.matched_positions.add(key)
                        return other_xy
            self.known_tiles[key] = new_color
        return None
