from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

@dataclass
class Entity:
    id: int
    color: int
    pixels: List[Tuple[int, int]]
    mass: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]

    @property
    def shape_signature(self) -> Tuple[int, int, int, int]:
        r0, c0, r1, c1 = self.bbox
        return (self.color, self.mass, r1 - r0 + 1, c1 - c0 + 1)

def hungarian_match(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    if cost_matrix.size == 0:
        return []
    n_rows, n_cols = cost_matrix.shape
    pairs: List[Tuple[int, int]] = []
    used_r: Set[int] = set()
    used_c: Set[int] = set()
    flat_indices = np.argsort(cost_matrix, axis=None)
    for idx in flat_indices:
        r = int(idx // n_cols)
        c = int(idx % n_cols)
        if r not in used_r and c not in used_c:
            if cost_matrix[r, c] < 1e6:
                pairs.append((r, c))
                used_r.add(r)
                used_c.add(c)
                if len(used_r) == n_rows or len(used_c) == n_cols:
                    break
    return pairs

class DynamicHUDMasker:
    """Infers HUD/status regions purely from statistical change frequency along borders."""
    def __init__(self, shape: Tuple[int, int] = (64, 64), enabled: bool = True):
        self.shape = shape
        self.enabled = enabled
        self.hud_mask = np.zeros(shape, dtype=bool)
        self.change_counts = np.zeros(shape, dtype=np.int32)
        self.total_transitions = 0

    def reset_level(self) -> None:
        self.change_counts.fill(0)
        self.total_transitions = 0
        self.hud_mask.fill(False)

    def observe_transition(self, pre_grid: np.ndarray, post_grid: np.ndarray) -> None:
        if not self.enabled or pre_grid.shape != self.shape or post_grid.shape != self.shape:
            return
        diff = pre_grid != post_grid
        changed_px = int(np.sum(diff))
        if 0 < changed_px <= 160:
            self.total_transitions += 1
            self.change_counts[diff] += 1
            if self.total_transitions >= 1:
                h, w = self.shape
                border_zone = np.zeros((h, w), dtype=bool)
                border_zone[:6, :] = True
                border_zone[h - 6 :, :] = True
                border_zone[:, :6] = True
                border_zone[:, w - 6 :] = True
                high_freq = self.change_counts >= 1
                row_ticks = np.sum(border_zone & high_freq, axis=1)
                col_ticks = np.sum(border_zone & high_freq, axis=0)
                mask = border_zone & (
                    self.change_counts >= max(2, int(0.60 * self.total_transitions))
                )
                for r in range(h):
                    if (r < 6 or r >= h - 6) and row_ticks[r] >= 3:
                        mask[r, :] = True
                for c in range(w):
                    if (c < 6 or c >= w - 6) and col_ticks[c] >= 3:
                        mask[:, c] = True
                self.hud_mask = mask

    def mask_grid(self, grid: np.ndarray) -> np.ndarray:
        arr = np.asarray(grid, dtype=np.int8).copy()
        if self.enabled and arr.shape == self.hud_mask.shape and np.any(self.hud_mask):
            arr[self.hud_mask] = 0
        return arr

class ObservationProcessor:
    """Normalizes grids, segments connected objects, and extracts subframe kinematics."""
    def __init__(self, enable_hud_masking: bool = True) -> None:
        self.hud_masker = DynamicHUDMasker(enabled=enable_hud_masking)

    def normalize_grid(self, raw_frame: Any) -> np.ndarray:
        if isinstance(raw_frame, np.ndarray):
            arr = raw_frame
        elif hasattr(raw_frame, "frame"):
            arr = np.asarray(raw_frame.frame)
        elif hasattr(raw_frame, "data"):
            arr = np.asarray(raw_frame.data)
        else:
            arr = np.asarray(raw_frame)
        if arr.ndim == 3 and arr.shape[0] >= 1:
            arr = arr[-1]
        if arr.ndim != 2:
            arr = np.zeros((64, 64), dtype=np.int8)
        return np.asarray(arr, dtype=np.int8)

    def extract_animation_sequence(self, raw_frame: Any) -> List[np.ndarray]:
        frames_list: List[np.ndarray] = []
        raw_seq = None
        for attr in ("frame", "data", "frames", "grids"):
            if hasattr(raw_frame, attr):
                raw_seq = getattr(raw_frame, attr)
                break
        if raw_seq is None:
            raw_seq = raw_frame
        arr = np.asarray(raw_seq)
        if arr.ndim == 3:
            for i in range(arr.shape[0]):
                frames_list.append(np.asarray(arr[i], dtype=np.int8))
        elif arr.ndim == 2:
            frames_list.append(np.asarray(arr, dtype=np.int8))
        if not frames_list:
            frames_list.append(np.zeros((64, 64), dtype=np.int8))
        return frames_list

    def induce_background_color(
        self, grid: np.ndarray, learned_floor_color: Optional[int] = None
    ) -> int:
        if learned_floor_color is not None:
            return int(learned_floor_color)
        arr = np.asarray(grid, dtype=np.int8)
        vals, counts = np.unique(arr, return_counts=True)
        return int(vals[int(np.argmax(counts))]) if len(vals) > 0 else 0

    def extract_entities(
        self,
        grid: np.ndarray,
        bg_color: Optional[int] = None,
        apply_hud_mask: bool = True,
    ) -> List[Entity]:
        arr = self.hud_masker.mask_grid(grid) if apply_hud_mask else np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        if bg_color is None:
            bg_color = self.induce_background_color(arr)
        visited = np.zeros((h, w), dtype=bool)
        if apply_hud_mask and self.hud_masker.enabled and self.hud_masker.hud_mask.shape == (h, w):
            visited |= self.hud_masker.hud_mask
        entities: List[Entity] = []
        ent_id = 0
        for r in range(h):
            for c in range(w):
                if visited[r, c] or int(arr[r, c]) == bg_color:
                    continue
                color = int(arr[r, c])
                stack = [(r, c)]
                visited[r, c] = True
                pixels: List[Tuple[int, int]] = []
                r_min = r_max = r
                c_min = c_max = c
                while stack:
                    cr, cc = stack.pop()
                    pixels.append((cr, cc))
                    if cr < r_min:
                        r_min = cr
                    if cr > r_max:
                        r_max = cr
                    if cc < c_min:
                        c_min = cc
                    if cc > c_max:
                        c_max = cc
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and not visited[nr, nc]
                            and int(arr[nr, nc]) == color
                        ):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                mass = len(pixels)
                mean_r = sum(p[0] for p in pixels) / float(mass)
                mean_c = sum(p[1] for p in pixels) / float(mass)
                entities.append(
                    Entity(
                        id=ent_id,
                        color=color,
                        pixels=pixels,
                        mass=mass,
                        centroid=(mean_r, mean_c),
                        bbox=(r_min, c_min, r_max, c_max),
                    )
                )
                ent_id += 1
        return entities

    def track_entities(
        self, prev_entities: List[Entity], curr_entities: List[Entity]
    ) -> List[Tuple[Entity, Entity, Tuple[float, float]]]:
        if not prev_entities or not curr_entities:
            return []
        cost = np.full((len(prev_entities), len(curr_entities)), 1e7, dtype=np.float64)
        for i, pe in enumerate(prev_entities):
            for j, ce in enumerate(curr_entities):
                if pe.color == ce.color and abs(pe.mass - ce.mass) <= max(2, int(0.25 * pe.mass)):
                    dist = abs(pe.centroid[0] - ce.centroid[0]) + abs(
                        pe.centroid[1] - ce.centroid[1]
                    )
                    if dist <= 24.0:
                        cost[i, j] = dist + abs(pe.mass - ce.mass) * 2.0
        matches = hungarian_match(cost)
        tracked: List[Tuple[Entity, Entity, Tuple[float, float]]] = []
        for r_idx, c_idx in matches:
            pe = prev_entities[r_idx]
            ce = curr_entities[c_idx]
            dr = ce.centroid[0] - pe.centroid[0]
            dc = ce.centroid[1] - pe.centroid[1]
            if abs(dr) > 0.1 or abs(dc) > 0.1:
                tracked.append((pe, ce, (dr, dc)))
        return tracked

    def extract_subframe_kinematics(
        self, prev_grid: np.ndarray, anim_frames: List[np.ndarray], bg_color: int
    ) -> List[Tuple[Entity, Entity, Tuple[float, float]]]:
        """Kepler sp80 visual lesson: extracts first-subframe impulse vector when multi-frame animations occur."""
        if not anim_frames:
            return []
        prev_ents = self.extract_entities(prev_grid, bg_color=bg_color)
        first_ents = self.extract_entities(anim_frames[0], bg_color=bg_color)
        tracked_first = self.track_entities(prev_ents, first_ents)
        if tracked_first or len(anim_frames) == 1:
            return tracked_first
        last_ents = self.extract_entities(anim_frames[-1], bg_color=bg_color)
        return self.track_entities(prev_ents, last_ents)

    def compute_delta_omega(self, prev_grid: np.ndarray, curr_grid: np.ndarray) -> int:
        a = np.asarray(prev_grid, dtype=np.int8)
        b = np.asarray(curr_grid, dtype=np.int8)
        if a.shape != b.shape:
            return 0
        diff = a != b
        if self.hud_masker.enabled and self.hud_masker.hud_mask.shape == diff.shape:
            diff = diff & (~self.hud_masker.hud_mask)
        return int(np.sum(diff))

    def summarize_transition(
        self,
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
        anim_frames: Optional[List[np.ndarray]] = None,
    ) -> str:
        f0 = np.asarray(prev_grid, dtype=np.int8)
        f1 = np.asarray(curr_grid, dtype=np.int8)
        if f0.shape != f1.shape:
            return "SHAPE_CHANGED"
        diff = f0 != f1
        if self.hud_masker.enabled and self.hud_masker.hud_mask.shape == diff.shape:
            diff = diff & (~self.hud_masker.hud_mask)
        rs, cs = np.where(diff)
        if len(rs) == 0:
            return "NO_INTERIOR_CHANGE"
        samples = []
        for idx in range(min(8, len(rs))):
            r, c = int(rs[idx]), int(cs[idx])
            samples.append(f"({r},{c}):{int(f0[r, c])}->{int(f1[r, c])}")
        anim_note = f" anim_len={len(anim_frames)}" if anim_frames and len(anim_frames) > 1 else ""
        return f"changed_px={len(rs)}{anim_note} [" + ", ".join(samples) + "]"

    def encode_compact_observation(self, grid: np.ndarray, max_entities: int = 24) -> str:
        bg = self.induce_background_color(grid)
        ents = self.extract_entities(grid, bg_color=bg)
        ents_sorted = sorted(ents, key=lambda e: (e.mass, e.color))[:max_entities]
        vals, counts = np.unique(grid, return_counts=True)
        palette_str = ", ".join(f"c{int(v)}:{int(cnt)}" for v, cnt in zip(vals, counts))
        ent_parts = [
            f"O{e.id}(c={e.color},m={e.mass},rc=({int(round(e.centroid[0]))},{int(round(e.centroid[1]))}),box={e.bbox})"
            for e in ents_sorted
        ]
        return f"bg={bg} | palette=[{palette_str}] | objects({len(ents)}): " + "; ".join(ent_parts)
