"""Ordered centreline paths from a binary band mask.

The mask is skeletonised, and each skeleton component is decomposed into
simple paths: the longest geodesic path first, then the longest remaining
branches. Short spurs, which skeletonisation always produces, are dropped
by a minimum length.

A longest path can turn a corner at a junction (for example band-1 to R1,
which meet in a T), so paths are then split at sharp turns and the pieces
re-joined only where one continues another smoothly. A band is a smooth
curve; a junction between two bands is a corner.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage.morphology import skeletonize

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# Grid radius cleared around an extracted path before looking for branches,
# so the next path does not re-trace the same band.
CLEARANCE_GRID = 2
# Half-window (grid points) for measuring direction along a path.
DIRECTION_WINDOW = 3
# A change of direction sharper than this between the chords either side of a
# point is a junction corner. Skeletons round corners off, so turning is
# tested at two scales; a genuine band (radius of curvature >= ~2 widths)
# turns well under this at both.
CORNER_ANGLE_DEG = 45.0
CORNER_WINDOWS = (DIRECTION_WINDOW, 2 * DIRECTION_WINDOW)
# Points dropped from each side of a corner: skeletons curl through a corner,
# so the points next to it belong to neither band's true course.
CORNER_TRIM = DIRECTION_WINDOW
# Pieces are re-joined if their ends are this close (grid units; covers the
# clearance gap and the corner trim on both sides)...
JOIN_GAP_GRID = 2 * CLEARANCE_GRID + 2 * CORNER_TRIM + 3
# ...or, across a crossing band, up to this far if the gap itself lies along
# both pieces' direction (so parallel neighbouring bands are never joined).
CROSSING_GAP_GRID = 25
# ...and their directions agree within this angle.
JOIN_ANGLE_DEG = 30.0
# Shortest skeleton piece worth extracting (grid units): enough points to
# survive corner trimming and give a direction.
MIN_PIECE_GRID = 2 * CORNER_TRIM + 3


def _graph(nodes: np.ndarray, shape: tuple[int, int]):
    index = -np.ones(shape, dtype=int)
    index[nodes[:, 0], nodes[:, 1]] = np.arange(len(nodes))
    rows, cols, weights = [], [], []
    for dr, dc in _NEIGHBOURS:
        r, c = nodes[:, 0] + dr, nodes[:, 1] + dc
        ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
        target = np.full(len(nodes), -1)
        target[ok] = index[r[ok], c[ok]]
        linked = target >= 0
        rows.append(np.flatnonzero(linked))
        cols.append(target[linked])
        weights.append(np.full(linked.sum(), np.hypot(dr, dc)))
    rows, cols, weights = map(np.concatenate, (rows, cols, weights))
    return coo_matrix((weights, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()


def _longest_path(graph, members: np.ndarray) -> tuple[list[int], float]:
    """Approximate graph diameter within one component (double sweep)."""
    start = members[0]
    dist = dijkstra(graph, indices=start)
    far = members[np.argmax(np.where(np.isfinite(dist[members]), dist[members], -1))]
    dist, pred = dijkstra(graph, indices=far, return_predecessors=True)
    end = members[np.argmax(np.where(np.isfinite(dist[members]), dist[members], -1))]
    path = [end]
    while path[-1] != far:
        path.append(pred[path[-1]])
    return path[::-1], float(dist[end])


def extract_paths(mask: np.ndarray, min_length_grid: float) -> list[np.ndarray]:
    """Ordered (row, col) paths, longest first, each at least min_length_grid long."""
    # Short pieces are extracted too: a band crossed by others is cut into
    # pieces that only reach the minimum length once re-joined.
    raw = _longest_paths(mask, MIN_PIECE_GRID)
    pieces = [piece for path in raw for piece in _split_at_corners(path)]
    joined = _join_continuations(pieces)
    joined = [p for p in joined if _length(p) >= min_length_grid]
    return sorted(joined, key=_length, reverse=True)


def _length(path: np.ndarray) -> float:
    return float(np.hypot(*np.diff(path.astype(float), axis=0).T).sum()) if len(path) > 1 else 0.0


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


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
    """Outward trend direction at one end of a path.

    The last few skeleton points next to a junction bend towards the other
    band, so the direction is measured over a span that skips them.
    """
    p = path.astype(float) if not at_start else path[::-1].astype(float)
    n = len(p)
    # Skeletons bend towards junctions and crossings over a few points, so
    # skip twice the direction window before measuring.
    skip = min(2 * DIRECTION_WINDOW, max(n - 3, 0))
    span = min(2 * DIRECTION_WINDOW, n - 1 - skip)
    if span < 1:
        return p[-1] - p[0]
    end = n - 1 - skip
    return p[end] - p[end - span]


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
        merged = np.vstack([first, second])
        pieces = [p for k, p in enumerate(pieces) if k not in (i, j)] + [merged]


def _longest_paths(mask: np.ndarray, min_length_grid: float) -> list[np.ndarray]:
    remaining = skeletonize(mask)
    paths: list[np.ndarray] = []
    while True:
        nodes = np.argwhere(remaining)
        if len(nodes) < 2:
            break
        graph = _graph(nodes, mask.shape)
        n_comp, labels = connected_components(graph, directed=False)
        best: tuple[list[int], float] | None = None
        for comp in range(n_comp):
            members = np.flatnonzero(labels == comp)
            if len(members) < 2:
                continue
            path, length = _longest_path(graph, members)
            if best is None or length > best[1]:
                best = (path, length)
        if best is None or best[1] < min_length_grid:
            break
        coords = nodes[best[0]]
        paths.append(coords)
        cleared = np.zeros(mask.shape, dtype=bool)
        cleared[coords[:, 0], coords[:, 1]] = True
        cleared = ndimage.binary_dilation(cleared, iterations=CLEARANCE_GRID)
        remaining &= ~cleared
    return paths
