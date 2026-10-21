from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class Entity:
    id: int
    color: int
    pixels: List[Tuple[int, int]]
    mass: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]


def px_to_board_coord(r_px: float, c_px: float, scale: int, shape: Tuple[int, int] = (64, 64)) -> Tuple[int, int]:
    s = max(1, int(scale))
    h, w = shape
    br = max(0, min((h // s) - 1, int(round(r_px / s))))
    bc = max(0, min((w // s) - 1, int(round(c_px / s))))
    return br, bc


def board_to_px_coord(br: int, bc: int, scale: int, shape: Tuple[int, int] = (64, 64)) -> Tuple[int, int]:
    s = max(1, int(scale))
    h, w = shape
    r_px = max(0, min(h - 1, int(br * s + s // 2)))
    c_px = max(0, min(w - 1, int(bc * s + s // 2)))
    return r_px, c_px


def hungarian_match(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    if cost_matrix.size == 0:
        return []
    n_rows, n_cols = cost_matrix.shape
    pairs = []
    used_r, used_c = set(), set()
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


class ObservationProcessor:
    """Zero-color-assumption CCL + Motion-Anchored Sprite Tracker + Remote Trigger Detector."""

    def __init__(self, bezel_px: int = 4):
        self.bezel_px = bezel_px

    def induce_background_color(self, grid: np.ndarray, learned_floor_color: Optional[int] = None) -> int:
        if learned_floor_color is not None:
            return int(learned_floor_color)
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        core = arr[self.bezel_px:h - self.bezel_px, self.bezel_px:w - self.bezel_px] if h > 8 else arr
        vals, counts = np.unique(core, return_counts=True)
        return int(vals[int(np.argmax(counts))]) if len(vals) > 0 else 0

    def extract_entities(
        self,
        grid: np.ndarray,
        bg_color: Optional[int] = None,
        ignore_bezel: bool = True,
    ) -> List[Entity]:
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        if bg_color is None:
            bg_color = self.induce_background_color(arr)

        visited = np.zeros((h, w), dtype=bool)
        if ignore_bezel and h >= 16 and w >= 16:
            b = self.bezel_px
            visited[:b, :] = True
            visited[h - b:, :] = True
            visited[:, :b] = True
            visited[:, w - b:] = True

        entities: List[Entity] = []
        ent_id = 0
        for r in range(h):
            for c in range(w):
                if visited[r, c] or int(arr[r, c]) == bg_color:
                    continue
                color = int(arr[r, c])
                stack = [(r, c)]
                visited[r, c] = True
                pixels = []
                r_min = r_max = r
                c_min = c_max = c
                while stack:
                    cr, cc = stack.pop()
                    pixels.append((cr, cc))
                    if cr < r_min: r_min = cr
                    if cr > r_max: r_max = cr
                    if cc < c_min: c_min = cc
                    if cc > c_max: c_max = cc
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(arr[nr, nc]) == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                mass = len(pixels)
                mean_r = sum(p[0] for p in pixels) / mass
                mean_c = sum(p[1] for p in pixels) / mass
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

    def encode_ascii_entity_graph(self, grid: np.ndarray, bg_color: int) -> str:
        ents = self.extract_entities(grid, bg_color=bg_color, ignore_bezel=True)
        ents_sorted = sorted(ents, key=lambda e: (e.mass, e.color))[:20]
        parts = [
            f"E{e.id}(c={e.color},m={e.mass},rc=({int(round(e.centroid[0]))},{int(round(e.centroid[1]))}))"
            for e in ents_sorted
        ]
        return "; ".join(parts)

    def compute_delta_omega(self, prev_grid: np.ndarray, curr_grid: np.ndarray) -> int:
        a = np.asarray(prev_grid, dtype=np.int8)
        b = np.asarray(curr_grid, dtype=np.int8)
        if a.shape != b.shape:
            return 0
        h, w = a.shape
        if h >= 16 and w >= 16:
            bz = self.bezel_px
            return int(np.sum(a[bz:h - bz, bz:w - bz] != b[bz:h - bz, bz:w - bz]))
        return int(np.sum(a != b))

    def infer_rigid_translation(
        self,
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
        known_avatar_colors: Optional[Set[int]] = None,
    ) -> Optional[Tuple[int, Set[int], Tuple[int, int], Tuple[float, float], Set[int], List[Tuple[int, int]]]]:
        """
        Universal Rigid-Body Differential Tracker + Remote Causal Trigger Detector.
        Returns:
          (floor_color, avatar_colors, (dr, dc), post_centroid, pushed_box_colors, remote_changed_centroids)
        """
        f0 = np.asarray(prev_grid, dtype=np.int8)
        f1 = np.asarray(curr_grid, dtype=np.int8)
        if f0.shape != f1.shape:
            return None
        h, w = f0.shape
        diff = f0 != f1
        if h >= 16 and w >= 16:
            bz = self.bezel_px
            mask = np.zeros((h, w), dtype=bool)
            mask[bz:h - bz, bz:w - bz] = True
            diff = diff & mask

        total_changed = int(np.sum(diff))
        if total_changed < 2 or total_changed > 900:
            return None

        best_floor = None
        best_score = 0
        for c in range(16):
            stepped_onto = int(np.sum(diff & (f0 == c) & (f1 != c)))
            vacated = int(np.sum(diff & (f0 != c) & (f1 == c)))
            score = min(stepped_onto, vacated)
            if score > best_score:
                best_score = score
                best_floor = c

        if best_floor is None or best_score == 0:
            return None

        b_post = np.argwhere(diff & (f0 == best_floor) & (f1 != best_floor))
        b_pre_all = np.argwhere(diff & (f0 != best_floor) & (f1 == best_floor))
        if len(b_post) == 0 or len(b_pre_all) == 0:
            return None

        post_center = b_post.mean(axis=0)
        dists = np.linalg.norm(b_pre_all - post_center, axis=1)
        near_mask = dists <= 18.0
        b_pre = b_pre_all[near_mask] if np.any(near_mask) else b_pre_all

        pre_center = b_pre.mean(axis=0)
        dr = int(round(float(post_center[0] - pre_center[0])))
        dc = int(round(float(post_center[1] - pre_center[1])))
        if dr == 0 and dc == 0:
            return None

        all_diff_pts = np.argwhere(diff)
        remote_dists = np.linalg.norm(all_diff_pts - post_center, axis=1)
        remote_pts = all_diff_pts[remote_dists > 16.0]
        remote_centroids: List[Tuple[int, int]] = []
        if len(remote_pts) >= 2:
            rr = int(round(float(remote_pts[:, 0].mean())))
            rc = int(round(float(remote_pts[:, 1].mean())))
            remote_centroids.append((rr, rc))

        pushed_box_colors: Set[int] = set()
        moving_colors = {int(c) for c in np.unique(f1[b_post[:, 0], b_post[:, 1]]) if int(c) != best_floor}
        if known_avatar_colors:
            extra_colors = moving_colors - known_avatar_colors
            for ec in extra_colors:
                overlap = int(np.sum(diff & (f0 == ec) & np.isin(f1, list(known_avatar_colors))))
                if overlap > 0:
                    pushed_box_colors.add(ec)
            avatar_colors = known_avatar_colors
        else:
            avatar_colors = moving_colors

        if not avatar_colors:
            return None

        return (
            best_floor,
            avatar_colors,
            (dr, dc),
            (float(post_center[0]), float(post_center[1])),
            pushed_box_colors,
            remote_centroids,
        )
