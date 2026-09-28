"""Ordered centreline paths from a binary band mask, built from a junction graph.

The mask is skeletonised and the skeleton is decomposed into a graph:
*nodes* are junctions (true branch points found with the crossing-number
test) and *branches* are the simple curves between them. Then:

1. Short spurs (skeletonisation artefacts with a free end) are removed.
2. Junction pairs joined by a very short bridge are merged into one node:
   a skeleton represents an X-crossing of two bands as two Y-junctions a
   few pixels apart.
3. At every node, branches are paired by continuation: the two whose
   directions are most nearly opposite pass straight through the node.
   A leftover branch (for example R1 at its T-junction with band-1) ends
   there.
4. Paired branches are chained into whole bands.

A band is a smooth curve, so bands crossing each other or meeting in a T
come out whole and separate. Remaining corners inside a single branch
(two bands meeting at a bend without a branch point) are split, and
broken pieces of one band (for example across a small data gap) are
re-joined only where one continues the other smoothly.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

# Offsets of the 8 neighbours in circular order (for the crossing number).
_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
# Half-window (grid points) for measuring direction along a path.
DIRECTION_WINDOW = 3
# A change of direction sharper than this between the chords either side of a
# point is a corner. Skeletons round corners off, so turning is tested at two
# scales; a genuine band (radius of curvature >= ~2 widths) turns well under
# this at both.
CORNER_ANGLE_DEG = 45.0
CORNER_WINDOWS = (DIRECTION_WINDOW, 2 * DIRECTION_WINDOW)
# Points dropped from each side of a corner: skeletons curl through a corner,
# so the points next to it belong to neither band's true course.
CORNER_TRIM = DIRECTION_WINDOW
# Branches with a free end shorter than this (grid units) are skeleton spurs.
SPUR_GRID = 4
# Junctions linked by a branch shorter than this many band widths are one
# node: an X-crossing of two bands can appear as two Y-junctions joined by a
# short bridge (up to about band width / sin(crossing angle)).
BRIDGE_THICKNESSES = 1.5
MIN_BRIDGE_GRID = 4
# Two branches continue through a node if their directions are opposite within this angle.
PAIR_ANGLE_DEG = 35.0
# Broken pieces of one band are re-joined if their ends are this close
# (grid units) and their directions agree within JOIN_ANGLE_DEG...
JOIN_GAP_GRID = 2 * CORNER_TRIM + 7
JOIN_ANGLE_DEG = 30.0
# ...or up to this far if the gap itself lies along both pieces.
CROSSING_GAP_GRID = 25
# Near a junction the skeleton of two thick bands bends (by 35 deg over
# ~40 mm in tests), so a branch's direction at a node is measured beyond one
# mask thickness from the node, over a span of NODE_SPAN_THICKNESSES.
NODE_SPAN_THICKNESSES = 1.0


def extract_paths(mask: np.ndarray, min_length_grid: float, band_width_grid: float = 0.0) -> list[np.ndarray]:
    """Ordered (row, col) paths, longest first, each at least min_length_grid long.

    band_width_grid: typical band width in grid units; sets the size of the
    junction distortion zone and of crossing bridges. If not given it is
    estimated from the mask thickness.
    """
    skeleton = skeletonize(mask)
    thickness = band_width_grid if band_width_grid > 0 else _mask_thickness(mask, skeleton)
    chains = _chains(skeleton, thickness)
    pieces = [piece for chain in chains for piece in _split_at_corners(chain)]
    joined = _join_continuations(pieces)
    joined = [p for p in joined if _length(p) >= min_length_grid]
    return sorted(joined, key=_length, reverse=True)


# --- geometry helpers -------------------------------------------------------

def _length(path: np.ndarray) -> float:
    return float(np.hypot(*np.diff(path.astype(float), axis=0).T).sum()) if len(path) > 1 else 0.0


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _direction_from(points: np.ndarray, skip: int) -> np.ndarray:
    """Direction pointing away from points[0], measured past the first `skip` points."""
    p = points.astype(float)
    n = len(p)
    skip = min(skip, max(n - 2, 0))
    span = min(2 * DIRECTION_WINDOW, n - 1 - skip)
    if span < 1:
        return p[-1] - p[0]
    return p[skip + span] - p[skip]


# --- skeleton graph ---------------------------------------------------------

def _neighbour_stack(skeleton: np.ndarray) -> np.ndarray:
    padded = np.pad(skeleton, 1)
    h, w = skeleton.shape
    return np.stack([padded[1 + dr: 1 + dr + h, 1 + dc: 1 + dc + w] for dr, dc in _RING])


def _branch_points(skeleton: np.ndarray) -> np.ndarray:
    """Skeleton pixels with crossing number >= 3 (true junctions, not staircase steps)."""
    ring = _neighbour_stack(skeleton).astype(int)
    transitions = np.abs(np.diff(np.concatenate([ring, ring[:1]]), axis=0)).sum(axis=0) // 2
    return skeleton & (transitions >= 3)


def _order_branch(pixels: np.ndarray) -> np.ndarray:
    """Order the pixels of a simple 8-connected curve from one end to the other."""
    index = {tuple(p): i for i, p in enumerate(pixels)}
    neighbours = []
    for r, c in pixels:
        neighbours.append([index[(r + dr, c + dc)] for dr, dc in _RING if (r + dr, c + dc) in index])
    ends = [i for i, nb in enumerate(neighbours) if len(nb) <= 1]
    start = ends[0] if ends else 0
    order, seen = [start], {start}
    while True:
        # Prefer 4-connected steps so a staircase is walked pixel by pixel.
        candidates = [j for j in neighbours[order[-1]] if j not in seen]
        if not candidates:
            break
        r, c = pixels[order[-1]]
        candidates.sort(key=lambda j: abs(pixels[j][0] - r) + abs(pixels[j][1] - c))
        order.append(candidates[0])
        seen.add(candidates[0])
    return pixels[order]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def _mask_thickness(mask: np.ndarray, skeleton: np.ndarray) -> float:
    """Typical thickness of the band mask (grid units)."""
    if not skeleton.any():
        return float(MIN_BRIDGE_GRID)
    return 2.0 * float(np.median(ndimage.distance_transform_edt(mask)[skeleton]))


def _direction_at_node(outward: np.ndarray, thickness: float) -> np.ndarray:
    """Branch direction leaving a node, measured beyond the junction's distortion zone."""
    p = outward.astype(float)
    n = len(p)
    skip = int(round(thickness))
    span = max(int(round(NODE_SPAN_THICKNESSES * thickness)), 2)
    if n <= skip + span:
        # Short branch: use its far half, which is least affected by the junction.
        return p[-1] - p[n // 2] if n >= 4 else p[-1] - p[0]
    return p[skip + span] - p[skip]


def _chains(skeleton: np.ndarray, thickness: float) -> list[np.ndarray]:
    """Decompose a skeleton into whole-band chains (see module docstring)."""
    junction_pixels = _branch_points(skeleton)
    # Adjacent junction pixels form one junction cluster.
    junction_labels, n_junctions = ndimage.label(junction_pixels, structure=np.ones((3, 3)))
    # Removing only the junction pixel does not separate an 8-connected
    # skeleton (branches still touch diagonally around it), so its whole 3x3
    # neighbourhood is cleared before labelling branches.
    junction_zone = ndimage.binary_dilation(junction_pixels, structure=np.ones((3, 3)))
    branch_labels, n_branches = ndimage.label(skeleton & ~junction_zone, structure=np.ones((3, 3)))

    near_junction = ndimage.grey_dilation(junction_labels, size=(5, 5))
    branches: list[np.ndarray] = []
    ends: list[list[int]] = []       # junction id (1-based) at each end, 0 = free end
    for label in range(1, n_branches + 1):
        pixels = _order_branch(np.argwhere(branch_labels == label))
        branches.append(pixels)
        ends.append([int(near_junction[tuple(pixels[0])]), int(near_junction[tuple(pixels[-1])])])

    # 1. Spurs: short branches with a free end.
    keep = [not ((0 in e) and _length(b) < SPUR_GRID and not (e[0] == 0 and e[1] == 0))
            for b, e in zip(branches, ends)]
    # 2. Merge junctions joined by a short bridge.
    bridge_limit = max(MIN_BRIDGE_GRID, BRIDGE_THICKNESSES * thickness)
    nodes = _UnionFind(n_junctions + 1)
    for i, (b, e) in enumerate(zip(branches, ends)):
        if keep[i] and e[0] and e[1] and e[0] != e[1] and _length(b) < bridge_limit:
            nodes.union(e[0], e[1])
            keep[i] = False
    branches = [b for b, k in zip(branches, keep) if k]
    ends = [[nodes.find(j) if j else 0 for j in e] for e, k in zip(ends, keep) if k]

    # 3. Pair branches through each node by continuation.
    at_node: dict[int, list[tuple[int, int]]] = defaultdict(list)   # node -> (branch, end)
    for i, e in enumerate(ends):
        for side in (0, 1):
            if e[side]:
                at_node[e[side]].append((i, side))
    link: dict[tuple[int, int], tuple[int, int]] = {}
    for node, incident in at_node.items():
        directions = {}
        for i, side in incident:
            outward = branches[i] if side == 0 else branches[i][::-1]
            directions[(i, side)] = _direction_at_node(outward, thickness)
        candidates = []
        for a in range(len(incident)):
            for b in range(a + 1, len(incident)):
                ea, eb = incident[a], incident[b]
                if ea[0] == eb[0]:
                    continue  # a loop returning to the same node
                angle = _angle_deg(directions[ea], -directions[eb])
                if angle < PAIR_ANGLE_DEG:
                    candidates.append((angle, ea, eb))
        used = set()
        for _, ea, eb in sorted(candidates):
            if ea in used or eb in used:
                continue
            link[ea], link[eb] = eb, ea
            used.update((ea, eb))

    # 4. Walk the links into chains.
    visited = [False] * len(branches)
    chains = []
    starts = [i for i in range(len(branches)) if (i, 0) not in link or (i, 1) not in link]
    for first in starts + list(range(len(branches))):
        if visited[first]:
            continue
        # Enter the first branch from its unlinked end, if it has one.
        entry = 0 if (first, 0) not in link else 1
        parts, i, side = [], first, entry
        while not visited[i]:
            visited[i] = True
            parts.append(branches[i] if side == 0 else branches[i][::-1])
            exit_end = (i, 1 - side)
            if exit_end not in link:
                break
            i, side = link[exit_end]
        chains.append(np.vstack(parts))
    return chains


# --- corners and re-joining -------------------------------------------------

def _split_at_corners(path: np.ndarray) -> list[np.ndarray]:
    n = len(path)
    if n < 2 * DIRECTION_WINDOW + 1:
        return [path]
    p = path.astype(float)
    turn = np.zeros(n)
    for w in CORNER_WINDOWS:
        for i in range(w, n - w):
            turn[i] = max(turn[i], _angle_deg(p[i] - p[i - w], p[i + w] - p[i]))
    corners = []
    i = 0
    while i < n:
        if turn[i] > CORNER_ANGLE_DEG:
            # The corner is the sharpest point of this run of high turning.
            j = i
            while j < n and turn[j] > CORNER_ANGLE_DEG:
                j += 1
            corners.append(i + int(np.argmax(turn[i:j])))
            i = j
        else:
            i += 1
    bounds = [0] + corners + [n]
    pieces = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        start = a + CORNER_TRIM if a > 0 else a       # a > 0: this end is a corner
        stop = b - CORNER_TRIM if b < n else b
        if stop - start >= 2:
            pieces.append(path[start:stop])
    return pieces


def _end_direction(path: np.ndarray, at_start: bool) -> np.ndarray:
    """Outward trend direction at one end of a path (skipping the bent last points)."""
    inward = path if at_start else path[::-1]
    return -_direction_from(inward, 2 * DIRECTION_WINDOW)


def _join_continuations(pieces: list[np.ndarray]) -> list[np.ndarray]:
    pieces = [p for p in pieces if len(p) >= 2]
    while True:
        best = None
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                for ei in (False, True):          # True = start of piece i
                    for ej in (False, True):
                        a = pieces[i][0] if ei else pieces[i][-1]
                        b = pieces[j][0] if ej else pieces[j][-1]
                        gap = np.hypot(*(a - b))
                        if gap > CROSSING_GAP_GRID:
                            continue
                        da = _end_direction(pieces[i], ei)
                        db = _end_direction(pieces[j], ej)
                        angle = _angle_deg(da, -db)
                        if gap > JOIN_GAP_GRID:
                            bridge = (b - a).astype(float)
                            angle = max(angle, _angle_deg(da, bridge), _angle_deg(-db, bridge))
                        if angle < JOIN_ANGLE_DEG and (best is None or angle < best[0]):
                            best = (angle, i, j, ei, ej)
        if best is None:
            return pieces
        _, i, j, ei, ej = best
        first = pieces[i][::-1] if ei else pieces[i]
        second = pieces[j] if ej else pieces[j][::-1]
        pieces = [p for k, p in enumerate(pieces) if k not in (i, j)] + [np.vstack([first, second])]
