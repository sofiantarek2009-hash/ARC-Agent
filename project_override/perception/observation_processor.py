from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.optimize import linear_sum_assignment

@dataclass
class Entity:
    id: int
    color: int
    pixels: List[Tuple[int, int]]
    mass: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)

    @property
    def shape_signature(self) -> Tuple[int, int, int, int]:
        r0, c0, r1, c1 = self.bbox
        return (self.color, self.mass, r1 - r0 + 1, c1 - c0 + 1)


class DynamicHUDMasker:
    """Infers HUD/status regions purely from statistical change frequency along borders.
    
    Dynamically identifies border rows/columns that oscillate on every step (e.g. countdown
    timers, action counters) without assuming a fixed 4px bezel.
    """
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
            if self.total_transitions >= 2:
                h, w = self.shape
                # Perimeter candidate zone: only outermost 4 lines
                border_zone = np.zeros((h, w), dtype=bool)
                border_zone[:4, :] = True
                border_zone[h - 4 :, :] = True
                border_zone[:, :4] = True
                border_zone[:, w - 4 :] = True
                
                # Cells that change frequently (>60% of transitions)
                high_freq = border_zone & (self.change_counts >= max(2, int(0.60 * self.total_transitions)))
                row_ticks = np.sum(high_freq, axis=1)
                col_ticks = np.sum(high_freq, axis=0)
                
                mask = np.zeros((h, w), dtype=bool)
                for r in range(h):
                    if (r < 4 or r >= h - 4) and row_ticks[r] >= 4:
                        mask[r, :] = True
                for c in range(w):
                    if (c < 4 or c >= w - 4) and col_ticks[c] >= 4:
                        mask[:, c] = True
                self.hud_mask = mask

    def mask_grid(self, grid: np.ndarray) -> np.ndarray:
        arr = np.asarray(grid, dtype=np.int8).copy()
        if self.enabled and arr.shape == self.hud_mask.shape and np.any(self.hud_mask):
            arr[self.hud_mask] = -1  # Masked HUD indicator
        return arr


class ObservationProcessor:
    """Normalizes grids, segments connected objects via scipy CCL, tracks entities via Hungarian
    matching, and extracts accurate transition kinematics without fixed coordinate assumptions.
    """
    def __init__(self, enable_hud_masking: bool = True) -> None:
        self.hud_masker = DynamicHUDMasker(enabled=enable_hud_masking)

    def normalize_grid(self, raw_frame: Any) -> np.ndarray:
        if isinstance(raw_frame, np.ndarray):
            arr = raw_frame
        elif hasattr(raw_frame, "frame"):
            arr = np.asarray(raw_frame.frame)
        elif hasattr(raw_frame, "data"):
            arr = np.asarray(raw_frame.data)
        elif isinstance(raw_frame, list):
            arr = np.asarray(raw_frame)
        else:
            arr = np.zeros((64, 64), dtype=np.int8)

        # Handle 3D tensors: (1, 64, 64) -> (64, 64) or (64, 64, 1) -> (64, 64)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] in (1, 3, 4):
                arr = arr[..., 0]
            else:
                arr = arr[-1]

        return np.asarray(arr, dtype=np.int8)

    def extract_animation_sequence(self, raw_obs: Any) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        if hasattr(raw_obs, "animation_sequence") and raw_obs.animation_sequence:
            for f in raw_obs.animation_sequence:
                frames.append(self.normalize_grid(f))
        elif hasattr(raw_obs, "frame"):
            raw_arr = np.asarray(raw_obs.frame)
            if raw_arr.ndim == 3 and raw_arr.shape[0] > 1:
                for idx in range(raw_arr.shape[0]):
                    frames.append(self.normalize_grid(raw_arr[idx]))
            else:
                frames.append(self.normalize_grid(raw_arr))
        if not frames:
            frames.append(self.normalize_grid(raw_obs))
        return frames

    def induce_canvas_and_floor(
        self, grid: np.ndarray, learned_floor_color: Optional[int] = None
    ) -> Tuple[int, Optional[int]]:
        """Determines canvas background and floor color dynamically.
        Perimeter boundary mode is canvas_bg; most frequent non-canvas color is floor.
        """
        h, w = grid.shape
        perimeter_cells = np.concatenate([
            grid[0, :], grid[h - 1, :], grid[:, 0], grid[:, w - 1]
        ])
        vals, counts = np.unique(perimeter_cells, return_counts=True)
        canvas_bg = int(vals[np.argmax(counts)])

        if learned_floor_color is not None:
            return canvas_bg, learned_floor_color

        interior = grid[2:h-2, 2:w-2] if h > 8 and w > 8 else grid
        int_vals, int_counts = np.unique(interior, return_counts=True)
        floor_color: Optional[int] = None
        for v, cnt in sorted(zip(int_vals, int_counts), key=lambda x: -x[1]):
            val = int(v)
            if val != canvas_bg and cnt >= (interior.size * 0.15):
                floor_color = val
                break
        return canvas_bg, floor_color

    def extract_entities(
        self, grid: np.ndarray, bg_color: int, floor_color: Optional[int] = None
    ) -> List[Entity]:
        """Segments connected components using scipy.ndimage.label per color."""
        entities: List[Entity] = []
        entity_id = 0
        h, w = grid.shape
        unique_colors = np.unique(grid)
        
        # 4-connectivity structure
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        for col in unique_colors:
            c = int(col)
            if c == bg_color or (floor_color is not None and c == floor_color) or c < 0:
                continue
            color_mask = (grid == c)
            labeled_arr, num_features = scipy_label(color_mask, structure=struct)
            for feat_idx in range(1, num_features + 1):
                feat_coords = np.argwhere(labeled_arr == feat_idx)
                if len(feat_coords) == 0:
                    continue
                pixels = [(int(r), int(c_coord)) for r, c_coord in feat_coords]
                min_r, min_c = int(np.min(feat_coords[:, 0])), int(np.min(feat_coords[:, 1]))
                max_r, max_c = int(np.max(feat_coords[:, 0])), int(np.max(feat_coords[:, 1]))
                centroid = (float(np.mean(feat_coords[:, 0])), float(np.mean(feat_coords[:, 1])))
                entities.append(
                    Entity(
                        id=entity_id,
                        color=c,
                        pixels=pixels,
                        mass=len(pixels),
                        centroid=centroid,
                        bbox=(min_r, min_c, max_r, max_c),
                    )
                )
                entity_id += 1
        return entities

    def track_entities(
        self, prev_entities: List[Entity], curr_entities: List[Entity]
    ) -> List[Tuple[Entity, Entity, Tuple[float, float]]]:
        """Optimal Hungarian tracking of entities between frames using scipy linear_sum_assignment."""
        if not prev_entities or not curr_entities:
            return []
        n_prev, n_curr = len(prev_entities), len(curr_entities)
        cost = np.zeros((n_prev, n_curr), dtype=np.float64)

        for i, pe in enumerate(prev_entities):
            for j, ce in enumerate(curr_entities):
                dr = ce.centroid[0] - pe.centroid[0]
                dc = ce.centroid[1] - pe.centroid[1]
                dist = np.sqrt(dr * dr + dc * dc)
                # Penalize color mismatch heavily
                color_penalty = 0.0 if pe.color == ce.color else 500.0
                mass_diff = abs(pe.mass - ce.mass)
                cost[i, j] = dist + color_penalty + (mass_diff * 2.0)

        row_ind, col_ind = linear_sum_assignment(cost)
        tracked: List[Tuple[Entity, Entity, Tuple[float, float]]] = []
        for r_idx, c_idx in zip(row_ind, col_ind):
            if cost[r_idx, c_idx] < 200.0:  # Matches must have compatible color/shape
                pe = prev_entities[r_idx]
                ce = curr_entities[c_idx]
                dr = ce.centroid[0] - pe.centroid[0]
                dc = ce.centroid[1] - pe.centroid[1]
                if abs(dr) > 0.05 or abs(dc) > 0.05:
                    tracked.append((pe, ce, (dr, dc)))
        return tracked

    def compute_delta_omega(self, pre_grid: np.ndarray, post_grid: np.ndarray) -> int:
        p0 = self.hud_masker.mask_grid(pre_grid)
        p1 = self.hud_masker.mask_grid(post_grid)
        return int(np.sum(p0 != p1))

    def extract_transition_kinematics(
        self, pre_grid: np.ndarray, post_grid: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Extracts the avatar displacement vector directly from individual entity displacement
        rather than pooling vacated/occupied pixels together, avoiding the Inverted Compass bug.
        """
        bg_color, floor_color = self.induce_canvas_and_floor(pre_grid)
        prev_ents = self.extract_entities(pre_grid, bg_color, floor_color)
        curr_ents = self.extract_entities(post_grid, bg_color, floor_color)
        
        tracked = self.track_entities(prev_ents, curr_ents)
        if not tracked:
            return None

        # Sort tracked moving entities by mass (avatar is typically a small, invariant entity)
        moving = [(pe, ce, vec) for pe, ce, vec in tracked if max(abs(vec[0]), abs(vec[1])) >= 0.5]
        if not moving:
            return None

        # Pick the most plausible avatar mover (smallest consistent entity or primary mover)
        moving.sort(key=lambda x: x[0].mass)
        pe, ce, (dr, dc) = moving[0]
        int_dr = int(round(dr))
        int_dc = int(round(dc))
        stride = max(abs(int_dr), abs(int_dc))

        return {
            "floor_color": floor_color if floor_color is not None else bg_color,
            "avatar_colors": {pe.color},
            "avatar_color_counts": {pe.color: pe.mass},
            "vector": (int_dr, int_dc),
            "stride": stride,
            "avatar_centroid": (round(ce.centroid[0], 1), round(ce.centroid[1], 1)),
            "half_w": max(1, (ce.bbox[2] - ce.bbox[0]) // 2),
        }

    def summarize_transition(
        self, pre_grid: np.ndarray, post_grid: np.ndarray, anim_frames: List[np.ndarray]
    ) -> str:
        delta = self.compute_delta_omega(pre_grid, post_grid)
        if delta == 0:
            return "NO_OP"
        kin = self.extract_transition_kinematics(pre_grid, post_grid)
        if kin and kin["vector"] != (0, 0):
            dr, dc = kin["vector"]
            return f"MOVED({dr:+d},{dc:+d})_px{delta}"
        return f"DIFF_px{delta}"

    def encode_compact_observation(self, grid: np.ndarray) -> str:
        """Encodes grid into compact symbolic summary for LLM prompt context."""
        h, w = grid.shape
        bg_col, floor_col = self.induce_canvas_and_floor(grid)
        ents = self.extract_entities(grid, bg_col, floor_col)
        
        lines = [f"GRID_SHAPE: {h}x{w} | BG_COLOR: {bg_col} | FLOOR_COLOR: {floor_col}"]
        lines.append(f"TOTAL_ENTITIES: {len(ents)}")
        # Group entities by color
        by_col: Dict[int, List[Entity]] = {}
        for e in ents:
            by_col.setdefault(e.color, []).append(e)
            
        for col, col_ents in sorted(by_col.items()):
            centroids = [f"({e.centroid[0]:.1f},{e.centroid[1]:.1f})" for e in col_ents[:6]]
            extra = f"...+{len(col_ents)-6}" if len(col_ents) > 6 else ""
            lines.append(f"  Color {col} (cnt={len(col_ents)}, mass={col_ents[0].mass}): {', '.join(centroids)}{extra}")
            
        return "\n".join(lines)
