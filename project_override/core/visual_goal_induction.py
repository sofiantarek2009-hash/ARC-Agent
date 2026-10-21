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


class BlueprintMatcher:
    """
    Multi-Scale Reference Blueprint & Legend Pattern Completion Inducer.
    Detects when a smaller reference pattern (e.g. 1x, 2x, 3x, 4x scaled) exists in one
    quadrant/panel and emits the missing target coordinates on the main canvas.
    """

    @staticmethod
    def find_blueprint_clicks(
        grid: np.ndarray, bg_color: int, bezel_px: int = 4
    ) -> List[Tuple[int, int]]:
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        if h < 16 or w < 16:
            return []
        core = arr[bezel_px:h - bezel_px, bezel_px:w - bezel_px]
        ch, cw = core.shape

        q_h, q_w = ch // 3, cw // 3
        legend = core[:q_h, :q_w]
        fg_legend = legend != bg_color
        if int(np.sum(fg_legend)) < 3 or int(np.sum(fg_legend)) > (q_h * q_w) // 2:
            return []

        r_idx, c_idx = np.where(fg_legend)
        sub_leg = legend[r_idx.min():r_idx.max() + 1, c_idx.min():c_idx.max() + 1]
        lh, lw = sub_leg.shape
        if lh < 2 or lw < 2:
            return []

        clicks: List[Tuple[int, int]] = []
        for scale in (1, 2, 3, 4):
            scaled = np.repeat(np.repeat(sub_leg, scale, axis=0), scale, axis=1)
            sh, sw = scaled.shape
            if sh >= ch - q_h or sw >= cw - q_w:
                continue
            best_score = 0
            best_pos: Optional[Tuple[int, int]] = None
            for r0 in range(q_h, ch - sh + 1, max(1, scale // 2)):
                for c0 in range(q_w, cw - sw + 1, max(1, scale // 2)):
                    patch = core[r0:r0 + sh, c0:c0 + sw]
                    fg_match = int(np.sum((patch != bg_color) & (scaled != bg_color)))
                    if fg_match > best_score:
                        best_score = fg_match
                        best_pos = (r0, c0)
            total_fg = int(np.sum(scaled != bg_color))
            if best_pos is not None and 0 < best_score < total_fg and best_score >= total_fg // 3:
                br0, bc0 = best_pos
                patch = core[br0:br0 + sh, bc0:bc0 + sw]
                miss_r, miss_c = np.where((scaled != bg_color) & (patch == bg_color))
                for mr, mc in zip(miss_r[::scale], miss_c[::scale]):
                    clicks.append((int(bezel_px + br0 + mr), int(bezel_px + bc0 + mc)))
                if clicks:
                    return clicks
        return clicks


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

        return BlueprintMatcher.find_blueprint_clicks(grid, bg_color=bg_color, bezel_px=bezel_px)


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
