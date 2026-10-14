"""Project Override: Dead-Signature Encoder & O(1) Transition Culler (ARC-AGI-3)
Implements:
1. 64-bit packed bitwise action signature encoder:
   - Action (4 bits)
   - Color (4 bits)
   - Mass (16 bits)
   - Context Hash (32 bits)
   - Flags / Reserved (8 bits)
2. O(1) Dead-Signature Culling:
   - Prunes any action that produces a null state transition (Delta Omega = 0).
   - Instantaneous bitset/set lookups preventing redundant state evaluations.
"""

from __future__ import annotations

import binascii
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np


# =====================================================================
# Bit Field Layout Constants (64-bit unsigned integer)
# =====================================================================
# Bits  0..31 : Context Hash (32 bits, 0..4294967295)
# Bits 32..47 : Mass (16 bits, 0..65535)
# Bits 48..51 : Color (4 bits, 0..15)
# Bits 52..55 : Action (4 bits, 0..15)
# Bits 56..63 : Flags / Reserved (8 bits, 0..255)

SHIFT_CONTEXT: int = 0
MASK_CONTEXT: int = 0xFFFFFFFF

SHIFT_MASS: int = 32
MASK_MASS: int = 0xFFFF

SHIFT_COLOR: int = 48
MASK_COLOR: int = 0x0F

SHIFT_ACTION: int = 52
MASK_ACTION: int = 0x0F

SHIFT_FLAGS: int = 56
MASK_FLAGS: int = 0xFF

MAX_64_BIT: int = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class ActionSignature:
    """Decoded representation of a 64-bit packed action signature."""
    action: int          # 4 bits: 0..15
    color: int           # 4 bits: 0..15
    mass: int            # 16 bits: 0..65535
    context_hash: int    # 32 bits: 0..0xFFFFFFFF
    flags: int = 0       # 8 bits: 0..255

    def __post_init__(self) -> None:
        if not (0 <= self.action <= MASK_ACTION):
            raise ValueError(f"Action {self.action} out of range [0, {MASK_ACTION}]")
        if not (0 <= self.color <= MASK_COLOR):
            raise ValueError(f"Color {self.color} out of range [0, {MASK_COLOR}]")
        if not (0 <= self.mass <= MASK_MASS):
            raise ValueError(f"Mass {self.mass} out of range [0, {MASK_MASS}]")
        if not (0 <= self.context_hash <= MASK_CONTEXT):
            raise ValueError(f"Context hash {self.context_hash} out of range [0, {MASK_CONTEXT}]")
        if not (0 <= self.flags <= MASK_FLAGS):
            raise ValueError(f"Flags {self.flags} out of range [0, {MASK_FLAGS}]")

    def to_uint64(self) -> int:
        """Packs fields into a single 64-bit unsigned integer."""
        sig = (
            ((self.flags & MASK_FLAGS) << SHIFT_FLAGS)
            | ((self.action & MASK_ACTION) << SHIFT_ACTION)
            | ((self.color & MASK_COLOR) << SHIFT_COLOR)
            | ((self.mass & MASK_MASS) << SHIFT_MASS)
            | ((self.context_hash & MASK_CONTEXT) << SHIFT_CONTEXT)
        )
        return sig & MAX_64_BIT

    def to_hex(self) -> str:
        """Hexadecimal representation formatted as 0x16X."""
        return f"0x{self.to_uint64():016X}"

    def to_bytes(self) -> bytes:
        """8-byte big-endian binary representation."""
        return self.to_uint64().to_bytes(8, byteorder="big", signed=False)

    @classmethod
    def from_uint64(cls, sig: int) -> ActionSignature:
        """Unpacks a 64-bit integer into an ActionSignature instance."""
        sig = sig & MAX_64_BIT
        context_hash = (sig >> SHIFT_CONTEXT) & MASK_CONTEXT
        mass = (sig >> SHIFT_MASS) & MASK_MASS
        color = (sig >> SHIFT_COLOR) & MASK_COLOR
        action = (sig >> SHIFT_ACTION) & MASK_ACTION
        flags = (sig >> SHIFT_FLAGS) & MASK_FLAGS
        return cls(
            action=action,
            color=color,
            mass=mass,
            context_hash=context_hash,
            flags=flags,
        )

    @classmethod
    def from_hex(cls, hex_str: str) -> ActionSignature:
        """Parses hexadecimal string into ActionSignature."""
        val = int(hex_str, 16)
        return cls.from_uint64(val)

    @classmethod
    def from_bytes(cls, b: bytes) -> ActionSignature:
        """Parses 8-byte big-endian binary into ActionSignature."""
        if len(b) != 8:
            raise ValueError(f"Expected 8 bytes, got {len(b)}")
        val = int.from_bytes(b, byteorder="big", signed=False)
        return cls.from_uint64(val)


def pack_signature(
    action: int,
    color: int,
    mass: int,
    context_hash: int,
    flags: int = 0,
) -> int:
    """Fast bitwise packing into a 64-bit integer without dataclass allocation."""
    sig = (
        ((flags & MASK_FLAGS) << SHIFT_FLAGS)
        | ((action & MASK_ACTION) << SHIFT_ACTION)
        | ((color & MASK_COLOR) << SHIFT_COLOR)
        | ((mass & MASK_MASS) << SHIFT_MASS)
        | ((context_hash & MASK_CONTEXT) << SHIFT_CONTEXT)
    )
    return sig & MAX_64_BIT


def unpack_signature(sig: int) -> ActionSignature:
    """Unpacks a 64-bit integer signature."""
    return ActionSignature.from_uint64(sig)


def compute_context_hash(
    grid: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> int:
    """Computes a 32-bit context hash for a grid state or region of interest (ROI).

    Uses fast CRC32 checksum over the byte buffer, masked to 32 bits.
    """
    arr = np.asarray(grid, dtype=np.int32)
    if roi is not None:
        r0, c0, r1, c1 = roi
        h, w = arr.shape
        r0 = max(0, min(h, r0))
        r1 = max(r0, min(h, r1))
        c0 = max(0, min(w, c0))
        c1 = max(c0, min(w, c1))
        sub = arr[r0:r1, c0:c1]
    else:
        sub = arr

    raw_bytes = sub.tobytes()
    return binascii.crc32(raw_bytes) & MASK_CONTEXT


# =====================================================================
# O(1) Dead-Signature Culling & Filter
# =====================================================================

class DeadSignatureFilter:
    """High-performance O(1) Dead-Signature Culling Filter.

    Tracks action signatures that produced null state transitions (Delta Omega = 0).
    Allows MCTS and search planners to prune dead actions instantly before
    performing expensive forward simulations, neural rollouts, or heuristic evaluations.
    """

    def __init__(self, max_capacity: Optional[int] = 1_000_000) -> None:
        self.max_capacity = max_capacity
        self._dead_signatures: Set[int] = set()
        self.total_queries: int = 0
        self.culled_queries: int = 0
        self.total_registered: int = 0

    def __len__(self) -> int:
        return len(self._dead_signatures)

    def is_dead(self, signature: int) -> bool:
        """O(1) lookup to check if an action signature is dead (null transition)."""
        self.total_queries += 1
        is_hit = (signature & MAX_64_BIT) in self._dead_signatures
        if is_hit:
            self.culled_queries += 1
        return is_hit

    def record_transition(
        self,
        signature: int,
        delta_omega: int,
    ) -> bool:
        """Evaluates transition state delta Delta Omega = ||s' - s||_0.

        If Delta Omega == 0, the transition produced NO observable change and is dead.
        Records the signature in O(1) and returns True if marked dead.
        """
        sig_uint64 = signature & MAX_64_BIT
        if delta_omega == 0:
            if self.max_capacity is not None and len(self._dead_signatures) >= self.max_capacity:
                # Evict roughly 10% oldest to maintain bounded memory
                evict_count = self.max_capacity // 10
                for _ in range(evict_count):
                    if self._dead_signatures:
                        self._dead_signatures.pop()

            self._dead_signatures.add(sig_uint64)
            self.total_registered += 1
            return True
        return False

    def mark_dead(self, signature: int) -> None:
        """Directly registers a signature as dead."""
        self._dead_signatures.add(signature & MAX_64_BIT)
        self.total_registered += 1

    def record_dead_signature(self, signature: int) -> None:
        """Alias for mark_dead."""
        self.mark_dead(signature)

    def clear(self) -> None:
        """Clears all cached dead signatures."""
        self._dead_signatures.clear()
        self.total_queries = 0
        self.culled_queries = 0
        self.total_registered = 0

    @property
    def cull_rate(self) -> float:
        """Fraction of queried actions pruned by the dead-signature filter."""
        if self.total_queries == 0:
            return 0.0
        return float(self.culled_queries / self.total_queries)

    def cull_actions(
        self,
        candidate_actions: List[Tuple[int, int, int]],  # (action, color, mass)
        context_hash: int,
    ) -> Tuple[List[Tuple[int, int, int]], List[int]]:
        """Filters candidate actions in O(|A|), culling dead signatures.

        Returns (surviving_actions, culled_signatures).
        """
        surviving: List[Tuple[int, int, int]] = []
        culled: List[int] = []

        for action, color, mass in candidate_actions:
            sig = pack_signature(
                action=action,
                color=color,
                mass=mass,
                context_hash=context_hash,
            )
            if self.is_dead(sig):
                culled.append(sig)
            else:
                surviving.append((action, color, mass))

        return surviving, culled

    def predict_static_null(
        self,
        action: int,
        color: int,
        mass: int,
        grid: np.ndarray,
        target_coords: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """Static heuristic check for guaranteed null transitions (Delta Omega = 0).

        Examples:
        - Flood-filling a cell with the exact color it already has
        - Applying an operation with mass == 0 on an empty canvas
        """
        if mass == 0 and action in (0, 1, 2, 3, 4):  # manipulation on empty region
            return True

        if target_coords is not None:
            r, c = target_coords
            h, w = grid.shape
            if 0 <= r < h and 0 <= c < w:
                if grid[r, c] == color:
                    # Painting with identical color produces Delta Omega = 0
                    return True

        return False
