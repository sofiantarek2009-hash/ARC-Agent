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


class HUDSequenceExtractor:
    """Extracts 1D card sequence strips and active highlighted target tokens from perimeter HUDs."""
    def __init__(self, bg_color: int = 0):
        self.bg_color = bg_color

    def extract_sequence(self, grid: np.ndarray) -> Optional[Dict[str, Any]]:
        h, w = grid.shape
        top_strip = grid[:6, 4:w-4]
        if np.any(top_strip != self.bg_color):
            labeled, num = scipy_label(top_strip != self.bg_color)
            if 2 <= num <= 12:
                tokens = []
                for i in range(1, num + 1):
                    pts = np.argwhere(labeled == i)
                    colors = [top_strip[r, c] for r, c in pts]
                    c_mode = max(set(colors), key=colors.count)
                    centroid_c = int(np.mean(pts[:, 1])) + 4
                    tokens.append((centroid_c, c_mode))
                tokens.sort(key=lambda t: t[0])
                token_colors = [t[1] for t in tokens]
                return {
                    "location": "TOP",
                    "tokens": token_colors,
                    "active_color": token_colors[0] if token_colors else None,
                }
        return None


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

        co_movers = []
        dynamic_hazards = []
        for ope, oce, ovec in moving[1:]:
            odr, odc = int(round(ovec[0])), int(round(ovec[1]))
            if (odr == int_dr and odc == int_dc) or (abs(odr) == abs(int_dr) and abs(odc) == abs(int_dc)):
                co_movers.append((int(round(oce.centroid[0])), int(round(oce.centroid[1]))))
            else:
                dynamic_hazards.append({
                    "pos": (int(round(oce.centroid[0])), int(round(oce.centroid[1]))),
                    "velocity": (odr, odc),
                    "mass": oce.mass,
                    "color": oce.color,
                })

        return {
            "floor_color": floor_color if floor_color is not None else bg_color,
            "avatar_colors": {pe.color},
            "avatar_color_counts": {pe.color: pe.mass},
            "vector": (int_dr, int_dc),
            "stride": stride,
            "avatar_centroid": (round(ce.centroid[0], 1), round(ce.centroid[1], 1)),
            "half_w": max(1, (ce.bbox[2] - ce.bbox[0]) // 2),
            "co_movers": co_movers,
            "dynamic_hazards": dynamic_hazards,
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

    def induce_stride(self, grid: np.ndarray, bg_color: int, floor_color: Optional[int]) -> int:
        """Estimates discrete kinematic cell stride from entity dimensions or common divisors."""
        ents = self.extract_entities(grid, bg_color, floor_color)
        if not ents:
            return 5
        from collections import Counter
        dims = []
        for e in ents:
            h = e.bbox[2] - e.bbox[0]
            w = e.bbox[3] - e.bbox[1]
            if 2 <= h <= 16:
                dims.append(h)
            if 2 <= w <= 16:
                dims.append(w)
        if dims:
            counts = Counter(dims)
            mode_dim = counts.most_common(1)[0][0]
            if 3 <= mode_dim <= 16:
                return mode_dim
        return 5

    def render_ascii_stride_grid(
        self,
        grid: np.ndarray,
        bg_col: int,
        floor_col: Optional[int],
        avatar_pos: Optional[Tuple[int, int]] = None,
        stride: int = 5,
    ) -> Tuple[str, Optional[Tuple[int, int]], List[Tuple[int, int, str]]]:
        """Renders compact downsampled 2D ASCII grid with coordinate headers."""
        stride = max(1, min(stride, 16))
        sub_h, sub_w = grid.shape[0] // stride, grid.shape[1] // stride
        if sub_h == 0 or sub_w == 0:
            return "", None, []

        grid_stride = np.zeros((sub_h, sub_w), dtype=int)
        for r in range(sub_h):
            for c in range(sub_w):
                grid_stride[r, c] = grid[r * stride + stride // 2, c * stride + stride // 2]

        av_cell: Optional[Tuple[int, int]] = None
        if avatar_pos is not None:
            av_cell = (avatar_pos[0] // stride, avatar_pos[1] // stride)

        interactive_items: List[Tuple[int, int, str]] = []

        header = "    " + " ".join(f"{c:2d}" for c in range(sub_w))
        lines = [header]

        # Identify perimeter border columns/rows to avoid listing frame borders as targets
        border_r = set()
        border_c = set()
        if sub_h > 2:
            if np.count_nonzero(grid_stride[0, :] == grid_stride[0, 0]) >= sub_w * 0.7:
                border_r.add(0)
            if np.count_nonzero(grid_stride[sub_h - 1, :] == grid_stride[sub_h - 1, 0]) >= sub_w * 0.7:
                border_r.add(sub_h - 1)
        if sub_w > 2:
            if np.count_nonzero(grid_stride[:, 0] == grid_stride[0, 0]) >= sub_h * 0.7:
                border_c.add(0)
            if np.count_nonzero(grid_stride[:, sub_w - 1] == grid_stride[0, sub_w - 1]) >= sub_h * 0.7:
                border_c.add(sub_w - 1)

        for r in range(sub_h):
            row_chars = []
            for c in range(sub_w):
                val = grid_stride[r, c]
                if (r, c) == av_cell:
                    ch = " @"
                elif val == bg_col:
                    ch = " #"
                elif floor_col is not None and val == floor_col:
                    ch = " ."
                else:
                    ch = f" {val}"
                    if r not in border_r and c not in border_c:
                        interactive_items.append((r, c, f"Color_{val}"))
                row_chars.append(ch)
            lines.append(f"{r:2d} |" + "".join(row_chars))

        return "\n".join(lines), av_cell, interactive_items

    def encode_compact_observation(
        self,
        grid: np.ndarray,
        avatar_pos: Optional[Tuple[int, int]] = None,
        stride: int = 5,
    ) -> str:
        """Encodes grid into 2D spatial ASCII map and semantic object list for LLM context."""
        h, w = grid.shape
        bg_col, floor_col = self.induce_canvas_and_floor(grid)
        ents = self.extract_entities(grid, bg_col, floor_col)

        ascii_map, av_cell, items = self.render_ascii_stride_grid(
            grid, bg_col, floor_col, avatar_pos=avatar_pos, stride=stride
        )

        lines = [f"GRID_SHAPE: {h}x{w} (cell stride = {stride}) | BG_COLOR: {bg_col} (#) | FLOOR_COLOR: {floor_col} (.)"]
        if av_cell is not None:
            lines.append(f"PLAYER_AVATAR: @ at grid coordinate {av_cell} (world pixel: {avatar_pos})")
        else:
            lines.append("PLAYER_AVATAR: @ (probing initial motion to locate)")

        if items:
            items_str = ", ".join(f"{tag} at ({r},{c})" for r, c, tag in items[:8])
            lines.append(f"INTERACTIVE_OBJECTS / TARGETS: {items_str}")

        lines.append(f"\n2D SPATIAL MAP (Legends: @=Avatar, .=Passable Floor, #=Wall/Void, Letters=Items/Goals):\n{ascii_map}")

        return "\n".join(lines)
