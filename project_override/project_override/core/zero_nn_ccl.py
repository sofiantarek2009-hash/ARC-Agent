"""
zero_nn_ccl.py - Zero-NN Perception Module for ARC-AGI-3 (Project Override)

Provides deterministic, symbolic perception with:
- 4-way and 8-way Connected Component Labeling using scipy.ndimage.
- Comprehensive feature extraction: color, mass, bounding box, centroid, perimeter,
  bounding box hull, convex hull geometry, solidity, and density.
- Border exclusion mask: Separates interior gameplay grid changes from the shrinking
  4-cell outer border action-budget timer bar in ARC-AGI-3.
- Relational Topological Invariant Hashing (Blake2b-64) that is mathematically invariant
  to absolute translation shifts across the grid.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import ndimage


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class BoundingBox:
    """Bounding box for a discrete 2D grid component."""
    min_row: int
    min_col: int
    max_row: int  # Inclusive
    max_col: int  # Inclusive
    height: int
    width: int
    area: int

    @classmethod
    def from_coords(cls, coords: np.ndarray) -> BoundingBox:
        """Create bounding box from (N, 2) array of coordinates."""
        if coords.shape[0] == 0:
            return cls(0, 0, 0, 0, 0, 0, 0)
        min_r = int(np.min(coords[:, 0]))
        max_r = int(np.max(coords[:, 0]))
        min_c = int(np.min(coords[:, 1]))
        max_c = int(np.max(coords[:, 1]))
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        return cls(
            min_row=min_r,
            min_col=min_c,
            max_row=max_r,
            max_col=max_c,
            height=h,
            width=w,
            area=h * w,
        )

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return (min_row, min_col, max_row, max_col)."""
        return (self.min_row, self.min_col, self.max_row, self.max_col)

    def contains(self, r: int, c: int) -> bool:
        """Check if point (r, c) is inside the bounding box."""
        return self.min_row <= r <= self.max_row and self.min_col <= c <= self.max_col

    def overlaps(self, other: BoundingBox) -> bool:
        """Check if this bounding box overlaps with another."""
        if self.area == 0 or other.area == 0:
            return False
        return not (
            self.max_row < other.min_row
            or self.min_row > other.max_row
            or self.max_col < other.min_col
            or self.min_col > other.max_col
        )


@dataclass
class Component:
    """
    Connected component representation with geometric and topological features.
    """
    id: int
    color: int
    mass: int
    bbox: BoundingBox
    centroid: Tuple[float, float]
    perimeter_pixels: int
    perimeter_edges: int
    coords: np.ndarray  # (N, 2) int
    normalized_coords: np.ndarray  # (N, 2) int, relative to (min_row, min_col)
    bbox_mask: np.ndarray  # (height, width) bool
    convex_hull_points: List[Tuple[int, int]]
    convex_hull_area: int
    hull_mask: np.ndarray  # (height, width) bool
    solidity: float
    density: float
    shape_hash: str  # Blake2b-64 hex digest of normalized shape
    is_interior: bool
    is_border: bool
    crosses_border: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert component features to JSON/dict representation."""
        return {
            "id": self.id,
            "color": self.color,
            "mass": self.mass,
            "bbox": self.bbox.as_tuple(),
            "height": self.bbox.height,
            "width": self.bbox.width,
            "centroid": self.centroid,
            "perimeter_pixels": self.perimeter_pixels,
            "perimeter_edges": self.perimeter_edges,
            "convex_hull_area": self.convex_hull_area,
            "solidity": self.solidity,
            "density": self.density,
            "shape_hash": self.shape_hash,
            "is_interior": self.is_interior,
            "is_border": self.is_border,
            "crosses_border": self.crosses_border,
        }


@dataclass
class SceneGraph:
    """
    Relational topological graph of components in a scene.
    Contains relative displacement, adjacency, distances, containment,
    and a translation-invariant Blake2b-64 hash.
    """
    components: List[Component]
    scene_hash: str  # Blake2b-64 hex digest (16 chars)
    scene_hash_uint64: int  # 64-bit integer
    pairwise_relations: Dict[Tuple[int, int], Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert scene graph to dict representation."""
        relations_serialized = {}
        for (i, j), rel in self.pairwise_relations.items():
            relations_serialized[f"{i}->{j}"] = rel
        return {
            "component_count": len(self.components),
            "scene_hash": self.scene_hash,
            "scene_hash_uint64": self.scene_hash_uint64,
            "components": [c.to_dict() for c in self.components],
            "relations": relations_serialized,
        }


# =============================================================================
# Convex Hull & Geometry Utilities (Monotone Chain)
# =============================================================================

def compute_convex_hull_2d(points: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Compute 2D convex hull of integer discrete points using Andrew's Monotone Chain.
    Returns vertices in counter-clockwise order.
    Handles N=0, N=1, N=2, and collinear configurations robustly.
    """
    unique_pts = sorted(set(points))
    if len(unique_pts) <= 2:
        return unique_pts

    def cross(o: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Lower hull
    lower: List[Tuple[int, int]] = []
    for p in unique_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Upper hull
    upper: List[Tuple[int, int]] = []
    for p in reversed(unique_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate lower and upper hull, omitting duplicate endpoints
    return lower[:-1] + upper[:-1]


def rasterize_convex_polygon(
    hull_vertices: Sequence[Tuple[int, int]],
    height: int,
    width: int,
) -> np.ndarray:
    """
    Rasterize a 2D convex polygon into a boolean mask of shape (height, width).
    Points inside or on the boundary of the convex polygon are marked True.
    """
    mask = np.zeros((height, width), dtype=bool)
    n = len(hull_vertices)
    if n == 0 or height <= 0 or width <= 0:
        return mask

    if n == 1:
        r, c = hull_vertices[0]
        if 0 <= r < height and 0 <= c < width:
            mask[r, c] = True
        return mask

    if n == 2:
        # Bresenham-like segment between two points
        r0, c0 = hull_vertices[0]
        r1, c1 = hull_vertices[1]
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        steps = max(dr, dc, 1)
        for s in range(steps + 1):
            r = int(round(r0 + s * (r1 - r0) / steps))
            c = int(round(c0 + s * (c1 - c0) / steps))
            if 0 <= r < height and 0 <= c < width:
                mask[r, c] = True
        return mask

    # General convex polygon (counter-clockwise vertices)
    # A point (r, c) is inside the polygon if cross(v_i, v_{i+1}, p) >= 0 for all edges
    # (or <= 0 depending on orientation).
    edges = []
    for i in range(n):
        v1 = hull_vertices[i]
        v2 = hull_vertices[(i + 1) % n]
        edges.append((v1, v2))

    # Grid coordinates
    rows, cols = np.indices((height, width))
    all_inside = np.ones((height, width), dtype=bool)

    # Test orientation of first non-zero edge
    for v1, v2 in edges:
        # cross((v1[0], v1[1]), (v2[0], v2[1]), (r, c))
        # = (v2[0] - v1[0]) * (c - v1[1]) - (v2[1] - v1[1]) * (r - v1[0])
        cross_val = (v2[0] - v1[0]) * (cols - v1[1]) - (v2[1] - v1[1]) * (rows - v1[0])
        all_inside &= (cross_val >= 0)

    # Also check if reversed winding order was used
    if not np.any(all_inside):
        all_inside = np.ones((height, width), dtype=bool)
        for v1, v2 in edges:
            cross_val = (v2[0] - v1[0]) * (cols - v1[1]) - (v2[1] - v1[1]) * (rows - v1[0])
            all_inside &= (cross_val <= 0)

    return all_inside


# =============================================================================
# ZeroNNPerception Main Class
# =============================================================================

class ZeroNNPerception:
    """
    Zero-NN Symbolic Perception Module for ARC-AGI-3.
    """

    DEFAULT_BORDER_WIDTH = 4

    @staticmethod
    def create_border_mask(shape: Tuple[int, int], border_width: int = DEFAULT_BORDER_WIDTH) -> np.ndarray:
        """
        Create a boolean mask where True indicates the outer border cells.
        Shape: (height, width).
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        b = min(border_width, h // 2, w // 2)
        if b <= 0:
            return mask
        mask[:b, :] = True
        mask[h - b:, :] = True
        mask[:, :b] = True
        mask[:, w - b:] = True
        return mask

    @staticmethod
    def create_interior_mask(shape: Tuple[int, int], border_width: int = DEFAULT_BORDER_WIDTH) -> np.ndarray:
        """
        Create a boolean mask where True indicates the interior gameplay cells.
        """
        return ~ZeroNNPerception.create_border_mask(shape, border_width)

    @classmethod
    def separate_grid(
        cls,
        grid: np.ndarray,
        border_width: int = DEFAULT_BORDER_WIDTH,
        fill_value: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate a grid into interior gameplay grid and border timer grid.
        Returns: (interior_grid, border_grid)
        """
        border_mask = cls.create_border_mask(grid.shape, border_width)
        interior_mask = ~border_mask

        interior_grid = np.full_like(grid, fill_value)
        interior_grid[interior_mask] = grid[interior_mask]

        border_grid = np.full_like(grid, fill_value)
        border_grid[border_mask] = grid[border_mask]

        return interior_grid, border_grid

    @classmethod
    def detect_gameplay_changes(
        cls,
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
        border_width: int = DEFAULT_BORDER_WIDTH,
    ) -> np.ndarray:
        """
        Detect cells that changed between prev_grid and curr_grid, STRICTLY
        within the interior gameplay grid, discarding timer bar updates in the border.
        Returns: boolean mask of shape (H, W) where True = interior gameplay change.
        """
        if prev_grid.shape != curr_grid.shape:
            raise ValueError(f"Shape mismatch: {prev_grid.shape} vs {curr_grid.shape}")
        interior_mask = cls.create_interior_mask(curr_grid.shape, border_width)
        raw_diff = (prev_grid != curr_grid)
        return raw_diff & interior_mask

    @classmethod
    def extract_border_timer_state(
        cls,
        grid: np.ndarray,
        border_width: int = DEFAULT_BORDER_WIDTH,
        background_color: int = 0,
    ) -> Dict[str, Any]:
        """
        Extract the action-budget timer bar status from the outer border.
        Measures:
        - active_pixel_count: total non-background pixels in border
        - border_pixel_capacity: total pixels in border region
        - budget_ratio: fraction of active border cells relative to initial full perimeter
        - colors: list of unique non-background colors in border
        - segments: active segments across top, bottom, left, right borders
        """
        h, w = grid.shape
        b = min(border_width, h // 2, w // 2)
        border_mask = cls.create_border_mask((h, w), b)
        border_cells = grid[border_mask]
        non_bg = border_cells != background_color

        active_count = int(np.sum(non_bg))
        total_border_cells = int(np.sum(border_mask))
        unique_colors = [int(c) for c in np.unique(border_cells[non_bg])]

        # Breakdown by border section
        top_active = int(np.sum(grid[:b, :] != background_color)) if b > 0 else 0
        bottom_active = int(np.sum(grid[h - b:, :] != background_color)) if b > 0 else 0
        left_active = int(np.sum(grid[b:h - b, :b] != background_color)) if b > 0 and h > 2 * b else 0
        right_active = int(np.sum(grid[b:h - b, w - b:] != background_color)) if b > 0 and h > 2 * b else 0

        budget_ratio = float(active_count / total_border_cells) if total_border_cells > 0 else 0.0

        return {
            "active_pixel_count": active_count,
            "total_border_cells": total_border_cells,
            "budget_ratio": budget_ratio,
            "colors": unique_colors,
            "segments": {
                "top": top_active,
                "bottom": bottom_active,
                "left": left_active,
                "right": right_active,
            },
        }

    # -------------------------------------------------------------------------
    # Connected Component Labeling & Feature Extraction
    # -------------------------------------------------------------------------

    @classmethod
    def extract_components(
        cls,
        grid: np.ndarray,
        connectivity: int = 4,
        background_color: Optional[int] = 0,
        per_color: bool = True,
        exclude_border: bool = False,
        border_width: int = DEFAULT_BORDER_WIDTH,
    ) -> List[Component]:
        """
        Perform Connected Component Labeling and feature extraction.

        Parameters:
        - grid: 2D integer numpy array.
        - connectivity: 4 for 4-way orthogonal (von Neumann) or 8 for 8-way (Moore).
        - background_color: Color to treat as background (ignored), or None to include all.
        - per_color: If True, components are partitioned per color (monochromatic).
                     If False, all non-background pixels are grouped together.
        - exclude_border: If True, only extracts components strictly inside interior gameplay grid.
        - border_width: Width of outer border timer mask (default 4).

        Returns:
        - List of Component objects.
        """
        grid = np.asarray(grid, dtype=np.int32)
        if grid.ndim != 2:
            raise ValueError(f"Expected 2D grid, got ndim={grid.ndim}")
        h, w = grid.shape
        if h == 0 or w == 0:
            return []

        # Connectivity structuring element
        if connectivity == 4:
            structure = ndimage.generate_binary_structure(2, 1)
        elif connectivity == 8:
            structure = ndimage.generate_binary_structure(2, 2)
        else:
            raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")

        border_mask = cls.create_border_mask((h, w), border_width)
        components: List[Component] = []
        comp_id_counter = 1

        if per_color:
            unique_colors = np.unique(grid)
            for c in unique_colors:
                c_val = int(c)
                if background_color is not None and c_val == background_color:
                    continue

                binary_mask = (grid == c_val)
                labeled, num_features = ndimage.label(binary_mask, structure=structure)

                for label_idx in range(1, num_features + 1):
                    comp_mask = (labeled == label_idx)
                    comp = cls._extract_single_component(
                        comp_id=comp_id_counter,
                        color=c_val,
                        mask=comp_mask,
                        grid_shape=(h, w),
                        border_mask=border_mask,
                    )
                    if comp is not None:
                        if exclude_border and not comp.is_interior:
                            continue
                        components.append(comp)
                        comp_id_counter += 1
        else:
            # Group all foreground together
            if background_color is not None:
                binary_mask = (grid != background_color)
            else:
                binary_mask = np.ones((h, w), dtype=bool)

            labeled, num_features = ndimage.label(binary_mask, structure=structure)

            for label_idx in range(1, num_features + 1):
                comp_mask = (labeled == label_idx)
                coords = np.argwhere(comp_mask)
                dominant_color = int(grid[coords[0, 0], coords[0, 1]]) if len(coords) > 0 else 0
                comp = cls._extract_single_component(
                    comp_id=comp_id_counter,
                    color=dominant_color,
                    mask=comp_mask,
                    grid_shape=(h, w),
                    border_mask=border_mask,
                )
                if comp is not None:
                    if exclude_border and not comp.is_interior:
                        continue
                    components.append(comp)
                    comp_id_counter += 1

        return components

    @classmethod
    def _extract_single_component(
        cls,
        comp_id: int,
        color: int,
        mask: np.ndarray,
        grid_shape: Tuple[int, int],
        border_mask: np.ndarray,
    ) -> Optional[Component]:
        """Extract all features for a single binary mask component."""
        coords = np.argwhere(mask)
        mass = int(len(coords))
        if mass == 0:
            return None

        # Bounding box
        bbox = BoundingBox.from_coords(coords)

        # Centroid (center of mass)
        centroid_r = float(np.mean(coords[:, 0]))
        centroid_c = float(np.mean(coords[:, 1]))
        centroid = (round(centroid_r, 4), round(centroid_c, 4))

        # Normalized coordinates and local bounding box mask
        norm_coords = coords - np.array([bbox.min_row, bbox.min_col])
        bbox_mask = np.zeros((bbox.height, bbox.width), dtype=bool)
        bbox_mask[norm_coords[:, 0], norm_coords[:, 1]] = True

        # Perimeter calculation
        # 1. Perimeter pixels: pixels having at least one 4-way neighbor outside component
        struct_4 = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(mask, structure=struct_4)
        perimeter_pixel_mask = mask & ~eroded
        perimeter_pixels = int(np.sum(perimeter_pixel_mask))

        # 2. Perimeter edges: count of boundary unit edges facing non-component cells / grid bounds
        # Using convolution / neighbor difference count
        perimeter_edges = cls._compute_perimeter_edges(mask)

        # Convex Hull in local bounding-box coordinates
        local_points = [(int(r), int(c)) for r, c in norm_coords]
        hull_vertices = compute_convex_hull_2d(local_points)
        hull_mask = rasterize_convex_polygon(hull_vertices, bbox.height, bbox.width)
        convex_hull_area = int(np.sum(hull_mask))

        # Solidity & Density
        solidity = float(mass / max(convex_hull_area, 1))
        density = float(mass / max(bbox.area, 1))

        # Canonical translation-invariant shape hash
        shape_hash = cls.hash_component_shape(color, bbox.height, bbox.width, bbox_mask)

        # Border classification
        comp_border_overlap = border_mask[coords[:, 0], coords[:, 1]]
        border_count = int(np.sum(comp_border_overlap))
        is_interior = (border_count == 0)
        is_border = (border_count == mass)
        crosses_border = (0 < border_count < mass)

        return Component(
            id=comp_id,
            color=color,
            mass=mass,
            bbox=bbox,
            centroid=centroid,
            perimeter_pixels=perimeter_pixels,
            perimeter_edges=perimeter_edges,
            coords=coords,
            normalized_coords=norm_coords,
            bbox_mask=bbox_mask,
            convex_hull_points=hull_vertices,
            convex_hull_area=convex_hull_area,
            hull_mask=hull_mask,
            solidity=solidity,
            density=density,
            shape_hash=shape_hash,
            is_interior=is_interior,
            is_border=is_border,
            crosses_border=crosses_border,
        )

    @staticmethod
    def _compute_perimeter_edges(mask: np.ndarray) -> int:
        """
        Count total number of exterior boundary edges facing outside cells or canvas borders.
        """
        h, w = mask.shape
        padded = np.pad(mask.astype(np.int32), pad_width=1, mode='constant', constant_values=0)
        # Differences in horizontal and vertical directions
        diff_r = np.abs(np.diff(padded, axis=0))
        diff_c = np.abs(np.diff(padded, axis=1))
        total_edges = int(np.sum(diff_r) + np.sum(diff_c))
        return total_edges

    # -------------------------------------------------------------------------
    # Relational Topological Invariant Hashing (Blake2b-64)
    # -------------------------------------------------------------------------

    @staticmethod
    def hash_component_shape(color: int, height: int, width: int, bbox_mask: np.ndarray) -> str:
        """
        Blake2b-64 hash of a single component's normalized geometric shape and color.
        Strictly invariant to absolute translation shifts.
        Returns 16-hex-character string (64 bits).
        """
        packed_bits = np.packbits(bbox_mask.astype(np.uint8)).tobytes()
        hasher = hashlib.blake2b(digest_size=8)
        hasher.update(b"COMP_V1:")
        hasher.update(int(color).to_bytes(4, byteorder="big", signed=True))
        hasher.update(int(height).to_bytes(4, byteorder="big", signed=False))
        hasher.update(int(width).to_bytes(4, byteorder="big", signed=False))
        hasher.update(packed_bits)
        return hasher.hexdigest()

    @classmethod
    def hash_scene(cls, components: List[Component]) -> Tuple[str, int]:
        """
        Compute a Relational Topological Invariant Blake2b-64 Hash of a scene.

        Mathematical Invariance:
        If the entire constellation of components is shifted across the grid by (Δr, Δc),
        the resulting hash is 100% IDENTICAL.

        Returns:
        - (hash_hex, hash_uint64): 16-character hex string and 64-bit integer.
        """
        if not components:
            hasher = hashlib.blake2b(b"EMPTY_SCENE", digest_size=8)
            digest = hasher.digest()
            return hasher.hexdigest(), int.from_bytes(digest, "big")

        # 1. Anchor origin: scene minimum row and minimum col
        scene_min_r = min(c.bbox.min_row for c in components)
        scene_min_c = min(c.bbox.min_col for c in components)

        # 2. Canonical sorting of components independent of initial discovery order
        # Key: (color, mass, height, width, shape_hash, rel_r, rel_c)
        def canonical_sort_key(c: Component) -> Tuple[int, int, int, int, str, int, int]:
            rel_r = c.bbox.min_row - scene_min_r
            rel_c = c.bbox.min_col - scene_min_c
            return (c.color, c.mass, c.bbox.height, c.bbox.width, c.shape_hash, rel_r, rel_c)

        sorted_comps = sorted(components, key=canonical_sort_key)

        # 3. Build byte stream for canonical sorted components
        hasher = hashlib.blake2b(digest_size=8)
        hasher.update(b"SCENE_V1:")
        hasher.update(len(sorted_comps).to_bytes(4, byteorder="big", signed=False))

        for c in sorted_comps:
            rel_r = c.bbox.min_row - scene_min_r
            rel_c = c.bbox.min_col - scene_min_c
            hasher.update(int(c.color).to_bytes(4, "big", signed=True))
            hasher.update(int(c.mass).to_bytes(4, "big", signed=False))
            hasher.update(int(rel_r).to_bytes(4, "big", signed=False))
            hasher.update(int(rel_c).to_bytes(4, "big", signed=False))
            hasher.update(int(c.bbox.height).to_bytes(4, "big", signed=False))
            hasher.update(int(c.bbox.width).to_bytes(4, "big", signed=False))
            hasher.update(bytes.fromhex(c.shape_hash))

        # 4. Relational topological pairwise interactions
        # Inter-object centroid vectors, Chebyshev distances, 4/8-adjacency, and containment
        struct_4 = ndimage.generate_binary_structure(2, 1)
        struct_8 = ndimage.generate_binary_structure(2, 2)

        n = len(sorted_comps)
        for i in range(n):
            c_i = sorted_comps[i]
            for j in range(i + 1, n):
                c_j = sorted_comps[j]

                # Translation-invariant relative centroid displacement
                d_r = int(round((c_j.centroid[0] - c_i.centroid[0]) * 1000))
                d_c = int(round((c_j.centroid[1] - c_i.centroid[1]) * 1000))

                # Pairwise minimal coordinate distance
                diff_coords = c_i.coords[:, np.newaxis, :] - c_j.coords[np.newaxis, :, :]
                chebyshev_dist = int(np.min(np.max(np.abs(diff_coords), axis=2)))
                manhattan_dist = int(np.min(np.sum(np.abs(diff_coords), axis=2)))

                # Adjacency checks
                touching_4 = (manhattan_dist == 1)
                touching_8 = (chebyshev_dist == 1)

                # Containment check: is c_j enclosed within bounding box of c_i?
                bbox_contains = (
                    c_i.bbox.min_row <= c_j.bbox.min_row
                    and c_i.bbox.max_row >= c_j.bbox.max_row
                    and c_i.bbox.min_col <= c_j.bbox.min_col
                    and c_i.bbox.max_col >= c_j.bbox.max_col
                )

                hasher.update(d_r.to_bytes(4, "big", signed=True))
                hasher.update(d_c.to_bytes(4, "big", signed=True))
                hasher.update(chebyshev_dist.to_bytes(4, "big", signed=False))
                hasher.update(manhattan_dist.to_bytes(4, "big", signed=False))
                hasher.update((1 if touching_4 else 0).to_bytes(1, "big"))
                hasher.update((1 if touching_8 else 0).to_bytes(1, "big"))
                hasher.update((1 if bbox_contains else 0).to_bytes(1, "big"))

        digest = hasher.digest()
        hash_hex = hasher.hexdigest()
        hash_uint64 = int.from_bytes(digest, byteorder="big", signed=False)
        return hash_hex, hash_uint64

    @classmethod
    def build_scene_graph(cls, components: List[Component]) -> SceneGraph:
        """
        Build full SceneGraph with pairwise relational topology and translation-invariant hash.
        """
        hash_hex, hash_uint64 = cls.hash_scene(components)
        relations: Dict[Tuple[int, int], Dict[str, Any]] = {}

        n = len(components)
        for i in range(n):
            c_i = components[i]
            for j in range(i + 1, n):
                c_j = components[j]

                d_r = round(c_j.centroid[0] - c_i.centroid[0], 4)
                d_c = round(c_j.centroid[1] - c_i.centroid[1], 4)

                diff_coords = c_i.coords[:, np.newaxis, :] - c_j.coords[np.newaxis, :, :]
                chebyshev_dist = int(np.min(np.max(np.abs(diff_coords), axis=2)))
                manhattan_dist = int(np.min(np.sum(np.abs(diff_coords), axis=2)))

                touching_4 = (manhattan_dist == 1)
                touching_8 = (chebyshev_dist == 1)

                bbox_contains_j = (
                    c_i.bbox.min_row <= c_j.bbox.min_row
                    and c_i.bbox.max_row >= c_j.bbox.max_row
                    and c_i.bbox.min_col <= c_j.bbox.min_col
                    and c_i.bbox.max_col >= c_j.bbox.max_col
                )
                bbox_contains_i = (
                    c_j.bbox.min_row <= c_i.bbox.min_row
                    and c_j.bbox.max_row >= c_i.bbox.max_row
                    and c_j.bbox.min_col <= c_i.bbox.min_col
                    and c_j.bbox.max_col >= c_i.bbox.max_col
                )

                rel_dict = {
                    "rel_displacement": (d_r, d_c),
                    "chebyshev_distance": chebyshev_dist,
                    "manhattan_distance": manhattan_dist,
                    "adjacent_4": touching_4,
                    "adjacent_8": touching_8,
                    "i_contains_j": bbox_contains_j,
                    "j_contains_i": bbox_contains_i,
                }
                relations[(c_i.id, c_j.id)] = rel_dict

        return SceneGraph(
            components=components,
            scene_hash=hash_hex,
            scene_hash_uint64=hash_uint64,
            pairwise_relations=relations,
        )
