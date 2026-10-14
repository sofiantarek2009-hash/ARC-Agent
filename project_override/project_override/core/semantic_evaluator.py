"""Project Override: Semantic Evaluator (ARC-AGI-3)
Mathematical, non-placeholder implementation of semantic evaluation metrics:
1. Dihedral Group D4 Symmetry Scorer (Reflectional & Rotational)
2. Boundary Enclosure & Connectivity Closure (Flood-fill & Ray-cast containment)
3. Mass and Color Compaction Metrics (Cluster density & Perimeter-to-area compactness)
4. Structural Shannon Entropy Scorer (Color, Degree, Spatial Dispersion & Transition Rewards)
5. Discrete Optimal Transport (Sinkhorn-Knopp 2-Wasserstein W2 distance for Object Fission/Fusion)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy import ndimage


# =====================================================================
# 1. Dihedral Group D4 Symmetry Scorer
# =====================================================================

@dataclass
class D4SymmetryResult:
    """Detailed breakdown of Dihedral Group D4 symmetry scores in [0.0, 1.0]."""
    horizontal_reflection: float    # Flipud (horizontal axis)
    vertical_reflection: float      # Fliplr (vertical axis)
    diagonal_reflection_main: float # Transpose (top-left to bottom-right)
    diagonal_reflection_anti: float # Anti-diagonal (top-right to bottom-left)
    rotational_90: float            # 90 degrees clockwise
    rotational_180: float           # 180 degrees
    rotational_270: float           # 270 degrees clockwise
    identity: float = 1.0           # 0 degrees (always 1.0)
    composite_score: float = 0.0    # Mean of non-trivial symmetries
    max_symmetry: float = 0.0       # Maximum non-trivial symmetry
    dominant_symmetry: str = "none" # Highest scoring non-trivial symmetry
    is_symmetric: bool = False      # True if any non-trivial symmetry >= 1.0 - 1e-5

    def to_dict(self) -> Dict[str, Union[float, str, bool]]:
        return {
            "horizontal_reflection": self.horizontal_reflection,
            "vertical_reflection": self.vertical_reflection,
            "diagonal_reflection_main": self.diagonal_reflection_main,
            "diagonal_reflection_anti": self.diagonal_reflection_anti,
            "rotational_90": self.rotational_90,
            "rotational_180": self.rotational_180,
            "rotational_270": self.rotational_270,
            "composite_score": self.composite_score,
            "max_symmetry": self.max_symmetry,
            "dominant_symmetry": self.dominant_symmetry,
            "is_symmetric": self.is_symmetric,
        }


class D4SymmetryScorer:
    """Measures reflectional (horizontal, vertical, diagonal) and rotational

    (90, 180, 270) symmetry of 2D grid states according to the D4 dihedral group.
    """

    def __init__(self, background_color: int = 0, pad_to_square: bool = True) -> None:
        self.background_color = background_color
        self.pad_to_square = pad_to_square

    def _pad_square(self, grid: np.ndarray) -> np.ndarray:
        """Pads a rectangular grid to a square canvas centered with background."""
        h, w = grid.shape
        if h == w or not self.pad_to_square:
            return grid
        dim = max(h, w)
        pad_h = dim - h
        pad_w = dim - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return np.pad(
            grid,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=self.background_color,
        )

    def _compute_match_ratio(self, g1: np.ndarray, g2: np.ndarray) -> float:
        """Computes element-wise match ratio between two grids."""
        if g1.shape != g2.shape:
            return 0.0
        total = g1.size
        if total == 0:
            return 1.0
        return float(np.mean(g1 == g2))

    def evaluate(self, grid: Union[List[List[int]], np.ndarray]) -> D4SymmetryResult:
        """Evaluates all 8 D4 dihedral group transformations of the grid state."""
        arr = np.asarray(grid, dtype=np.int32)
        if arr.ndim != 2:
            raise ValueError(f"Grid must be 2D, got shape {arr.shape}")
        if arr.size == 0:
            return D4SymmetryResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Pad to square if required to make 90/270/diagonal transforms dimension-preserving
        base = self._pad_square(arr)

        # 1. Horizontal reflection: flip across horizontal axis (flipud)
        t_h = np.flipud(base)
        score_h = self._compute_match_ratio(base, t_h)

        # 2. Vertical reflection: flip across vertical axis (fliplr)
        t_v = np.fliplr(base)
        score_v = self._compute_match_ratio(base, t_v)

        # 3. Main diagonal reflection: transpose (top-left to bottom-right)
        t_d1 = base.T
        score_d1 = self._compute_match_ratio(base, t_d1)

        # 4. Anti-diagonal reflection: flip across anti-diagonal
        t_d2 = np.flipud(np.fliplr(base.T))
        score_d2 = self._compute_match_ratio(base, t_d2)

        # 5. Rotations: 90, 180, 270 degrees clockwise
        # Note: np.rot90 rotates counter-clockwise by default, so k=3 is 90 deg clockwise
        t_r90 = np.rot90(base, k=3)
        score_r90 = self._compute_match_ratio(base, t_r90)

        t_r180 = np.rot90(base, k=2)
        score_r180 = self._compute_match_ratio(base, t_r180)

        t_r270 = np.rot90(base, k=1)
        score_r270 = self._compute_match_ratio(base, t_r270)

        scores = {
            "horizontal_reflection": score_h,
            "vertical_reflection": score_v,
            "diagonal_reflection_main": score_d1,
            "diagonal_reflection_anti": score_d2,
            "rotational_90": score_r90,
            "rotational_180": score_r180,
            "rotational_270": score_r270,
        }

        composite = float(np.mean(list(scores.values())))
        dominant_name = max(scores, key=scores.get)
        max_score = float(scores[dominant_name])
        is_sym = bool(max_score >= 1.0 - 1e-5)

        return D4SymmetryResult(
            horizontal_reflection=score_h,
            vertical_reflection=score_v,
            diagonal_reflection_main=score_d1,
            diagonal_reflection_anti=score_d2,
            rotational_90=score_r90,
            rotational_180=score_r180,
            rotational_270=score_r270,
            composite_score=composite,
            max_symmetry=max_score,
            dominant_symmetry=dominant_name,
            is_symmetric=is_sym,
        )


# =====================================================================
# 2. Boundary Enclosure & Connectivity Closure
# =====================================================================

@dataclass
class EnclosureResult:
    """Result of boundary enclosure and connectivity closure analysis."""
    has_sealed_loop: bool
    sealed_cavity_count: int
    sealed_cavity_areas: List[int]
    total_enclosed_area: int
    is_goal_enclosed: bool
    enclosed_goal_cells: List[Tuple[int, int]]
    cavity_mask: np.ndarray
    ray_cast_enclosed_ratio: float


class BoundaryEnclosureScorer:
    """Detects sealed loops, enclosed goals, and connected paths using

    flood-fill containment and ray-cast containment algorithms.
    """

    def __init__(self, background_color: int = 0) -> None:
        self.background_color = background_color

    def compute_enclosure(
        self,
        grid: Union[List[List[int]], np.ndarray],
        barrier_colors: Optional[Set[int]] = None,
        goal_color: Optional[int] = None,
        goal_coords: Optional[List[Tuple[int, int]]] = None,
    ) -> EnclosureResult:
        """Performs flood-fill enclosure analysis.

        Cells are barriers if their color is in `barrier_colors` (or any non-background
        color if `barrier_colors` is None). Non-barrier cells that cannot reach the canvas
        borders via 4-way flood-fill are defined as sealed interior cavities.
        """
        arr = np.asarray(grid, dtype=np.int32)
        h, w = arr.shape
        if h == 0 or w == 0:
            return EnclosureResult(
                has_sealed_loop=False,
                sealed_cavity_count=0,
                sealed_cavity_areas=[],
                total_enclosed_area=0,
                is_goal_enclosed=False,
                enclosed_goal_cells=[],
                cavity_mask=np.zeros((0, 0), dtype=bool),
                ray_cast_enclosed_ratio=0.0,
            )

        # 1. Identify barrier mask
        if barrier_colors is None:
            barrier_mask = (arr != self.background_color)
            if goal_color is not None:
                barrier_mask = barrier_mask & (arr != goal_color)
        else:
            barrier_mask = np.isin(arr, list(barrier_colors))

        # 2. Outer Flood-Fill: pad with 1 pixel of exterior and flood-fill from border
        padded_barriers = np.pad(barrier_mask, 1, mode="constant", constant_values=False)
        ph, pw = padded_barriers.shape

        visited = np.zeros((ph, pw), dtype=bool)
        queue = [(0, 0)]
        visited[0, 0] = True

        head = 0
        while head < len(queue):
            r, c = queue[head]
            head += 1
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < ph and 0 <= nc < pw:
                    if not visited[nr, nc] and not padded_barriers[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

        # Unpad visited mask
        reachable_from_exterior = visited[1 : ph - 1, 1 : pw - 1]

        # Enclosed interior cavity = non-barrier and not reachable from exterior
        cavity_mask = (~barrier_mask) & (~reachable_from_exterior)

        # Label distinct sealed cavities
        labeled_cavities, num_cavities = ndimage.label(cavity_mask)
        cavity_areas: List[int] = []
        if num_cavities > 0:
            cavity_areas = [
                int(np.sum(labeled_cavities == i)) for i in range(1, num_cavities + 1)
            ]
        total_enclosed_area = int(np.sum(cavity_mask))
        has_sealed_loop = bool(num_cavities > 0 and total_enclosed_area > 0)

        # 3. Check goal enclosure
        enclosed_goal_cells: List[Tuple[int, int]] = []
        is_goal_enclosed = False

        if goal_color is not None:
            goal_mask = (arr == goal_color)
            goal_in_cavity = goal_mask & cavity_mask
            coords = np.argwhere(goal_in_cavity)
            enclosed_goal_cells.extend([(int(r), int(c)) for r, c in coords])
            if np.any(goal_mask):
                is_goal_enclosed = bool(np.sum(goal_in_cavity) == np.sum(goal_mask))

        if goal_coords is not None and len(goal_coords) > 0:
            matching = [
                pt for pt in goal_coords
                if 0 <= pt[0] < h and 0 <= pt[1] < w and cavity_mask[pt[0], pt[1]]
            ]
            enclosed_goal_cells.extend(matching)
            is_goal_enclosed = bool(len(matching) == len(goal_coords))

        # 4. Ray-cast containment metric
        ray_enclosed_count = 0
        non_barrier_count = int(np.sum(~barrier_mask))
        if non_barrier_count > 0:
            for r in range(h):
                for c in range(w):
                    if not barrier_mask[r, c]:
                        if self.is_point_enclosed_raycast(arr, r, c, barrier_mask):
                            ray_enclosed_count += 1
            ray_ratio = float(ray_enclosed_count / non_barrier_count)
        else:
            ray_ratio = 0.0

        return EnclosureResult(
            has_sealed_loop=has_sealed_loop,
            sealed_cavity_count=num_cavities,
            sealed_cavity_areas=cavity_areas,
            total_enclosed_area=total_enclosed_area,
            is_goal_enclosed=is_goal_enclosed,
            enclosed_goal_cells=enclosed_goal_cells,
            cavity_mask=cavity_mask,
            ray_cast_enclosed_ratio=ray_ratio,
        )

    def is_point_enclosed_raycast(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        barrier_mask: Optional[np.ndarray] = None,
    ) -> bool:
        """Ray-cast containment: casts rays in 4 cardinal directions.

        Returns True if all 4 rays encounter a barrier before hitting canvas boundary.
        """
        h, w = grid.shape
        if barrier_mask is None:
            barrier_mask = (grid != self.background_color)

        if barrier_mask[r, c]:
            return True

        # North
        has_north = bool(np.any(barrier_mask[:r, c]))
        # South
        has_south = bool(np.any(barrier_mask[r + 1 :, c]))
        # West
        has_west = bool(np.any(barrier_mask[r, :c]))
        # East
        has_east = bool(np.any(barrier_mask[r, c + 1 :]))

        return has_north and has_south and has_west and has_east

    def check_path_connectivity(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        traversable_colors: Optional[Set[int]] = None,
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """BFS path connectivity check between start and goal coordinates."""
        h, w = grid.shape
        sr, sc = start
        gr, gc = goal
        if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
            return False, []

        if traversable_colors is None:
            # By default, cells with start color or background are traversable
            traversable_mask = (grid == grid[sr, sc]) | (grid == self.background_color)
        else:
            traversable_mask = np.isin(grid, list(traversable_colors))

        traversable_mask[sr, sc] = True
        traversable_mask[gr, gc] = True

        queue = [(sr, sc)]
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        found = False
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            if curr == goal:
                found = True
                break
            cr, cc = curr
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    nxt = (nr, nc)
                    if nxt not in parent and traversable_mask[nr, nc]:
                        parent[nxt] = curr
                        queue.append(nxt)

        if not found:
            return False, []

        path = []
        curr = goal
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return True, path


# =====================================================================
# 3. Mass and Color Compaction Metrics
# =====================================================================

@dataclass
class CompactionResult:
    """Summary of cluster density and perimeter-to-area compactness."""
    compactness_score: float             # Sum(Mass / (Perimeter + epsilon))
    mean_cluster_density: float          # Mean Mass / BoundingBoxArea
    mean_radius_of_gyration: float       # Mean dispersion from centroid
    isoperimetric_quotient: float        # 4 * pi * Area / Perimeter^2
    per_color_compactness: Dict[int, float] = field(default_factory=dict)
    per_color_density: Dict[int, float] = field(default_factory=dict)


class CompactionScorer:
    """Evaluates cluster density and perimeter-to-area compactness metrics

    formula: Compactness = Sum(Mass / (Perimeter + epsilon)).
    """

    def __init__(self, background_color: int = 0, epsilon: float = 1e-4) -> None:
        self.background_color = background_color
        self.epsilon = epsilon

    def _compute_perimeter(self, binary_mask: np.ndarray) -> int:
        """Computes external 4-connected boundary perimeter edges of a binary mask."""
        if not np.any(binary_mask):
            return 0
        h, w = binary_mask.shape
        perimeter = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.zeros_like(binary_mask)
            r_slice_dst = slice(max(0, dr), min(h, h + dr))
            r_slice_src = slice(max(0, -dr), min(h, h - dr))
            c_slice_dst = slice(max(0, dc), min(w, w + dc))
            c_slice_src = slice(max(0, -dc), min(w, w - dc))
            shifted[r_slice_dst, c_slice_dst] = binary_mask[r_slice_src, c_slice_src]
            # External edge occurs where cell is in mask, but neighbor is not in mask
            perimeter += int(np.sum(binary_mask & (~shifted)))
        return perimeter

    def evaluate(self, grid: Union[List[List[int]], np.ndarray]) -> CompactionResult:
        """Computes mass, cluster density, and perimeter-to-area compactness."""
        arr = np.asarray(grid, dtype=np.int32)
        unique_colors = np.unique(arr)
        foreground_colors = [int(c) for c in unique_colors if c != self.background_color]

        if not foreground_colors:
            return CompactionResult(
                compactness_score=0.0,
                mean_cluster_density=0.0,
                mean_radius_of_gyration=0.0,
                isoperimetric_quotient=0.0,
            )

        total_compactness = 0.0
        densities: List[float] = []
        gyrations: List[float] = []
        isoperimetric_scores: List[float] = []

        per_color_compactness: Dict[int, float] = {}
        per_color_density: Dict[int, float] = {}

        for color in foreground_colors:
            mask = (arr == color)
            mass = int(np.sum(mask))
            if mass == 0:
                continue

            # Connected component analysis for individual objects of this color
            labeled, num_components = ndimage.label(mask)
            color_compactness = 0.0

            for comp_idx in range(1, num_components + 1):
                comp_mask = (labeled == comp_idx)
                comp_mass = int(np.sum(comp_mask))
                comp_perim = self._compute_perimeter(comp_mask)

                # Compaction metric: Mass / (Perimeter + epsilon)
                ratio = comp_mass / (comp_perim + self.epsilon)
                color_compactness += ratio

                # Isoperimetric quotient: 4 * pi * A / (P^2 + epsilon)
                iso = (4.0 * np.pi * comp_mass) / (float(comp_perim ** 2) + self.epsilon)
                isoperimetric_scores.append(min(1.0, float(iso)))

                # Cluster density: Mass / BBox Area
                coords = np.argwhere(comp_mask)
                rmin, cmin = coords.min(axis=0)
                rmax, cmax = coords.max(axis=0)
                bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
                densities.append(float(comp_mass / bbox_area))

                # Radius of gyration
                centroid = coords.mean(axis=0)
                sq_dist = np.sum((coords - centroid) ** 2, axis=1)
                rg = float(np.sqrt(np.mean(sq_dist)))
                gyrations.append(rg)

            per_color_compactness[color] = float(color_compactness)
            coords_all = np.argwhere(mask)
            rmin, cmin = coords_all.min(axis=0)
            rmax, cmax = coords_all.max(axis=0)
            bbox_area_color = (rmax - rmin + 1) * (cmax - cmin + 1)
            per_color_density[color] = float(mass / bbox_area_color)

            total_compactness += color_compactness

        mean_density = float(np.mean(densities)) if densities else 0.0
        mean_gyration = float(np.mean(gyrations)) if gyrations else 0.0
        mean_iso = float(np.mean(isoperimetric_scores)) if isoperimetric_scores else 0.0

        return CompactionResult(
            compactness_score=float(total_compactness),
            mean_cluster_density=mean_density,
            mean_radius_of_gyration=mean_gyration,
            isoperimetric_quotient=mean_iso,
            per_color_compactness=per_color_compactness,
            per_color_density=per_color_density,
        )


# =====================================================================
# 4. Structural Shannon Entropy Scorer
# =====================================================================

@dataclass
class StructuralEntropyResult:
    """Decomposition of Structural Shannon Entropy."""
    color_entropy: float          # H_color
    degree_entropy: float         # H_degree (heterogeneity/boundary degree)
    spatial_dispersion: float     # H_spatial (quadrant/spatial partition entropy)
    total_entropy: float          # Weighted combination H(s)

    def to_dict(self) -> Dict[str, float]:
        return {
            "color_entropy": self.color_entropy,
            "degree_entropy": self.degree_entropy,
            "spatial_dispersion": self.spatial_dispersion,
            "total_entropy": self.total_entropy,
        }


@dataclass
class EntropyTransitionResult:
    """Transition evaluation reward: rewarded if Delta H <= -epsilon."""
    delta_h: float
    is_rewarded: bool
    reward: float
    prior_entropy: StructuralEntropyResult
    post_entropy: StructuralEntropyResult


class StructuralEntropyScorer:
    """Evaluates color entropy, degree entropy, and spatial dispersion.

    A state transition (s -> s') is rewarded if Delta H <= -epsilon.
    """

    def __init__(
        self,
        weight_color: float = 0.4,
        weight_degree: float = 0.3,
        weight_spatial: float = 0.3,
        spatial_blocks: int = 4,
        epsilon: float = 1e-4,
    ) -> None:
        self.w_color = weight_color
        self.w_degree = weight_degree
        self.w_spatial = weight_spatial
        self.spatial_blocks = spatial_blocks
        self.epsilon = epsilon

    def compute_color_entropy(self, grid: np.ndarray) -> float:
        """Shannon entropy of color distribution: H_color = -Sum p(c) * log2(p(c))."""
        if grid.size == 0:
            return 0.0
        _, counts = np.unique(grid, return_counts=True)
        probs = counts / float(grid.size)
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    def compute_degree_entropy(self, grid: np.ndarray) -> float:
        """Adjacency degree entropy: measures diversity of color boundary degrees.

        For each cell, the boundary degree d in {0, 1, 2, 3, 4} counts how many
        orthogonal neighbors have a different color.
        """
        h, w = grid.shape
        if h <= 1 and w <= 1:
            return 0.0

        degrees = np.zeros((h, w), dtype=np.int32)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.copy(grid)
            r_slice_dst = slice(max(0, dr), min(h, h + dr))
            r_slice_src = slice(max(0, -dr), min(h, h - dr))
            c_slice_dst = slice(max(0, dc), min(w, w + dc))
            c_slice_src = slice(max(0, -dc), min(w, w - dc))
            diff = np.zeros((h, w), dtype=bool)
            diff[r_slice_dst, c_slice_dst] = (
                grid[r_slice_dst, c_slice_dst] != shifted[r_slice_src, c_slice_src]
            )
            degrees += diff.astype(np.int32)

        _, counts = np.unique(degrees, return_counts=True)
        probs = counts / float(grid.size)
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    def compute_spatial_dispersion(self, grid: np.ndarray) -> float:
        """Spatial dispersion entropy: partitions grid into K x K blocks and measures

        entropy of mass distribution across blocks.
        """
        h, w = grid.shape
        if h == 0 or w == 0:
            return 0.0

        k = self.spatial_blocks
        block_h = max(1, int(np.ceil(h / k)))
        block_w = max(1, int(np.ceil(w / k)))

        non_zero_mask = (grid != 0).astype(np.float32)
        total_mass = float(np.sum(non_zero_mask))
        if total_mass <= 0.0:
            return 0.0

        block_masses: List[float] = []
        for i in range(k):
            for j in range(k):
                r0, r1 = i * block_h, min(h, (i + 1) * block_h)
                c0, c1 = j * block_w, min(w, (j + 1) * block_w)
                if r0 < h and c0 < w:
                    block_masses.append(float(np.sum(non_zero_mask[r0:r1, c0:c1])))

        masses = np.array(block_masses, dtype=np.float64)
        if np.sum(masses) <= 0.0:
            return 0.0
        probs = masses / np.sum(masses)
        non_zero_probs = probs[probs > 1e-12]
        return float(-np.sum(non_zero_probs * np.log2(non_zero_probs)))

    def evaluate(self, grid: Union[List[List[int]], np.ndarray]) -> StructuralEntropyResult:
        """Evaluates all structural Shannon entropy metrics for a grid state."""
        arr = np.asarray(grid, dtype=np.int32)
        h_color = self.compute_color_entropy(arr)
        h_degree = self.compute_degree_entropy(arr)
        h_spatial = self.compute_spatial_dispersion(arr)

        total = float(
            self.w_color * h_color
            + self.w_degree * h_degree
            + self.w_spatial * h_spatial
        )
        return StructuralEntropyResult(
            color_entropy=h_color,
            degree_entropy=h_degree,
            spatial_dispersion=h_spatial,
            total_entropy=total,
        )

    def evaluate_transition(
        self,
        prior_state: Union[List[List[int]], np.ndarray],
        post_state: Union[List[List[int]], np.ndarray],
    ) -> EntropyTransitionResult:
        """Evaluates transition Delta H = H(s') - H(s).

        Rewarded if Delta H <= -epsilon (entropy reduction).
        """
        e_prior = self.evaluate(prior_state)
        e_post = self.evaluate(post_state)
        delta_h = float(e_post.total_entropy - e_prior.total_entropy)

        is_rewarded = bool(delta_h <= -self.epsilon)
        reward = max(0.0, -delta_h) if is_rewarded else 0.0

        return EntropyTransitionResult(
            delta_h=delta_h,
            is_rewarded=is_rewarded,
            reward=reward,
            prior_entropy=e_prior,
            post_entropy=e_post,
        )


# =====================================================================
# 5. Discrete Optimal Transport (Sinkhorn-Knopp algorithm)
# =====================================================================

@dataclass
class OptimalTransportResult:
    """Discrete Optimal Transport 2-Wasserstein distance and transport plan."""
    w2_distance: float
    w2_squared: float
    transport_plan: np.ndarray
    converged: bool
    iterations: int
    source_centroids: np.ndarray
    target_centroids: np.ndarray
    source_masses: np.ndarray
    target_masses: np.ndarray


class SinkhornKnoppOptimalTransport:
    """Measures 2-Wasserstein distance (W2) between discrete point clouds of

    object centroids to handle object fission (shattering) and fusion (merging)
    without breaking topological continuity.
    """

    def __init__(
        self,
        reg: float = 0.1,          # Entropy regularization parameter gamma
        max_iter: int = 200,       # Maximum Sinkhorn iterations
        tol: float = 1e-5,         # Convergence tolerance
        background_color: int = 0, # Color to ignore for object extraction
    ) -> None:
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.background_color = background_color

    def extract_point_cloud(
        self,
        grid: Union[List[List[int]], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Extracts object centroids, masses, and colors from a 2D grid."""
        arr = np.asarray(grid, dtype=np.int32)
        unique_colors = [
            int(c) for c in np.unique(arr) if c != self.background_color
        ]
        centroids: List[Tuple[float, float]] = []
        masses: List[float] = []
        colors: List[int] = []

        for color in unique_colors:
            mask = (arr == color)
            labeled, num_components = ndimage.label(mask)
            for idx in range(1, num_components + 1):
                comp_mask = (labeled == idx)
                comp_mass = float(np.sum(comp_mask))
                coords = np.argwhere(comp_mask)
                centroid = coords.mean(axis=0)  # (row, col)
                centroids.append((float(centroid[0]), float(centroid[1])))
                masses.append(comp_mass)
                colors.append(color)

        if not centroids:
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                [],
            )

        return (
            np.array(centroids, dtype=np.float64),
            np.array(masses, dtype=np.float64),
            colors,
        )

    def compute_wasserstein_w2(
        self,
        source: Union[np.ndarray, List[List[int]]],
        target: Union[np.ndarray, List[List[int]]],
        source_masses: Optional[np.ndarray] = None,
        target_masses: Optional[np.ndarray] = None,
    ) -> OptimalTransportResult:
        """Computes 2-Wasserstein distance between discrete point clouds or grids.

        If source/target are 2D grids, centroids and masses are automatically extracted.
        Handles object fission (1 -> N) and fusion (N -> 1) seamlessly.
        """
        # Auto-extract if input is 2D grid
        if isinstance(source, list) or (isinstance(source, np.ndarray) and source.ndim == 2 and source_masses is None):
            src_pts, src_m, _ = self.extract_point_cloud(source)
        else:
            src_pts = np.asarray(source, dtype=np.float64)
            src_m = np.ones(len(src_pts)) if source_masses is None else np.asarray(source_masses, dtype=np.float64)

        if isinstance(target, list) or (isinstance(target, np.ndarray) and target.ndim == 2 and target_masses is None):
            tgt_pts, tgt_m, _ = self.extract_point_cloud(target)
        else:
            tgt_pts = np.asarray(target, dtype=np.float64)
            tgt_m = np.ones(len(tgt_pts)) if target_masses is None else np.asarray(target_masses, dtype=np.float64)

        n = len(src_pts)
        m = len(tgt_pts)

        # Edge cases: empty point clouds
        if n == 0 and m == 0:
            return OptimalTransportResult(
                w2_distance=0.0,
                w2_squared=0.0,
                transport_plan=np.zeros((0, 0)),
                converged=True,
                iterations=0,
                source_centroids=src_pts,
                target_centroids=tgt_pts,
                source_masses=src_m,
                target_masses=tgt_m,
            )
        if n == 0 or m == 0:
            # Maximum penalty if one cloud is empty
            return OptimalTransportResult(
                w2_distance=1e4,
                w2_squared=1e8,
                transport_plan=np.zeros((n, m)),
                converged=True,
                iterations=0,
                source_centroids=src_pts,
                target_centroids=tgt_pts,
                source_masses=src_m,
                target_masses=tgt_m,
            )

        # Normalize probability measures
        a = src_m / (np.sum(src_m) + 1e-12)
        b = tgt_m / (np.sum(tgt_m) + 1e-12)

        # Cost matrix: squared Euclidean distance C_ij = ||x_i - y_j||_2^2
        diff = src_pts[:, np.newaxis, :] - tgt_pts[np.newaxis, :, :]  # (n, m, 2)
        c_matrix = np.sum(diff ** 2, axis=2)  # (n, m)

        # Trivial single-point case
        if n == 1 and m == 1:
            w2_sq = float(c_matrix[0, 0])
            plan = np.ones((1, 1), dtype=np.float64)
            return OptimalTransportResult(
                w2_distance=float(np.sqrt(max(0.0, w2_sq))),
                w2_squared=w2_sq,
                transport_plan=plan,
                converged=True,
                iterations=1,
                source_centroids=src_pts,
                target_centroids=tgt_pts,
                source_masses=src_m,
                target_masses=tgt_m,
            )

        # Sinkhorn-Knopp algorithm
        # Gibbs kernel K = exp(-C / gamma)
        # Using stabilized scaled kernel
        c_min = np.min(c_matrix)
        c_scaled = (c_matrix - c_min) / max(self.reg, 1e-6)
        kernel = np.exp(-c_scaled)

        u = np.ones(n, dtype=np.float64) / n
        v = np.ones(m, dtype=np.float64) / m
        converged = False
        iteration = 0

        for iteration in range(1, self.max_iter + 1):
            # u = a / (K v + delta)
            kv = np.dot(kernel, v)
            u_next = a / (kv + 1e-14)

            # v = b / (K^T u + delta)
            ktu = np.dot(kernel.T, u_next)
            v_next = b / (ktu + 1e-14)

            # Check convergence on marginals
            err = np.sum(np.abs(u_next * np.dot(kernel, v_next) - a))
            u = u_next
            v = v_next

            if err < self.tol:
                converged = True
                break

        # Compute transport plan P = diag(u) * K * diag(v)
        plan = u[:, np.newaxis] * kernel * v[np.newaxis, :]
        plan = plan / (np.sum(plan) + 1e-14)  # ensure sum to 1

        # W2^2 = Sum_ij P_ij * ||x_i - y_j||^2
        w2_sq = float(np.sum(plan * c_matrix))
        w2_dist = float(np.sqrt(max(0.0, w2_sq)))

        return OptimalTransportResult(
            w2_distance=w2_dist,
            w2_squared=w2_sq,
            transport_plan=plan,
            converged=converged,
            iterations=iteration,
            source_centroids=src_pts,
            target_centroids=tgt_pts,
            source_masses=src_m,
            target_masses=tgt_m,
        )


# =====================================================================
# Unified Semantic Evaluator Facade
# =====================================================================

class SemanticEvaluator:
    """Unified Semantic Evaluator for ARC-AGI-3 cognitive state evaluation."""

    def __init__(self, background_color: int = 0) -> None:
        self.background_color = background_color
        self.symmetry_scorer = D4SymmetryScorer(background_color=background_color)
        self.enclosure_scorer = BoundaryEnclosureScorer(background_color=background_color)
        self.compaction_scorer = CompactionScorer(background_color=background_color)
        self.entropy_scorer = StructuralEntropyScorer()
        self.transport_scorer = SinkhornKnoppOptimalTransport(background_color=background_color)

    def evaluate_state(
        self,
        grid: Union[List[List[int]], np.ndarray],
        goal_color: Optional[int] = None,
        goal_coords: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Runs full suite of semantic evaluators on a state."""
        arr = np.asarray(grid, dtype=np.int32)
        sym = self.symmetry_scorer.evaluate(arr)
        enc = self.enclosure_scorer.compute_enclosure(
            arr, goal_color=goal_color, goal_coords=goal_coords
        )
        comp = self.compaction_scorer.evaluate(arr)
        entropy = self.entropy_scorer.evaluate(arr)

        return {
            "symmetry": sym,
            "enclosure": enc,
            "compaction": comp,
            "entropy": entropy,
        }

    def evaluate_transition(
        self,
        prior_state: Union[List[List[int]], np.ndarray],
        post_state: Union[List[List[int]], np.ndarray],
        target_state: Optional[Union[List[List[int]], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Evaluates state transition dynamics: entropy drop and optimal transport."""
        entropy_trans = self.entropy_scorer.evaluate_transition(prior_state, post_state)
        w2_result = self.transport_scorer.compute_wasserstein_w2(prior_state, post_state)

        target_w2 = None
        if target_state is not None:
            target_w2 = self.transport_scorer.compute_wasserstein_w2(post_state, target_state)

        return {
            "entropy_transition": entropy_trans,
            "dynamics_w2": w2_result,
            "target_w2": target_w2,
        }
