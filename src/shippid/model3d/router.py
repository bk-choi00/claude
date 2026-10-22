"""Deterministic orthogonal 3D pipe routing and inline-item placement.

1. The P&ID graph (objects + junctions, process pipes; signal lines are not piping) is
   contracted to *major* nodes: placed equipment, junctions, off-page connectors, hubs
   (inline objects with three or more connections, e.g. three-way valves) and free pipe ends.
   Every chain of pipes between two major nodes, through inline items (valves, instruments,
   fittings, strainers / filters / flow meters not placed as equipment), is routed as one
   polyline; chains ending at an inline item (instrument taps, vents, drains) become stubs.
2. Junction / hub positions: the drawing position mapped into the room (affine fit of drawing
   centre -> 3D centre over the placed equipment, falling back to the plain normalised mapping)
   blended with the positions of the nodes they connect to, at the elevation band of their line,
   snapped to a free voxel. The node cell is a *terminal* cell (hard for chains not ending there),
   its six neighbours are reserved (a soft cost for them); a hub's body is registered as an
   obstacle right away. Every equipment nozzle used by a chain gets its own approach (nozzle ->
   stub along the face normal -> centre of a free cell, one cell per nozzle where possible) whose
   cells are terminal cells as well.
3. Each chain is routed with orthogonal A* on a voxel grid (0.25-0.5 m): bend penalty, a cost for
   running away from the preferred elevation band of the medium / line, search bounded to a box
   around the endpoints (widened once), largest pipe size first (plan priority lines before
   everything), global time budget. Hard: equipment boxes inflated by OD/2 + clearance, the real
   bodies of inline items / hubs placed so far (mesh bounds incl. stems and actuators, vertical-run
   orientation) inflated likewise, other pipes within r1 + r2 + gap of their centre-line, other
   chains' terminal cells. Pipes of chains sharing a terminal node are passable (soft) only next to
   that terminal. Only when no such path exists are pipes / items / terminals turned into soft
   costs (recorded in the notes); failures fall back to a Manhattan path through the elevation band.
4. Right after a chain is routed its inline items are placed along the polyline in drawing order,
   near positions proportional to the drawn pipe lengths, on straight (preferably horizontal)
   segments, away from shared junctions, at the candidate position whose body stays clear of
   equipment, other pipes, other items and terminal approaches; bodies are clamped to the room.
   They then become obstacles for the following chains, and the polyline is split back into one
   :class:`PipeRoute3D` per original pipe id.
5. Audit (reported in the notes): segments through equipment, pipe / inline-item clashes,
   pipe-pipe clashes between pipes that share no node, overlapping items, items outside the room.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np
from scipy import ndimage

from ..catalog import SYSTEMS, TYPICAL_DIMENSIONS, pipe_od_m
from ..model import (ActuatorType, Category, Medium, PIDDocument, PIDObject, PipeKind, PipeRoute3D, PipeRun,
                     Placement3D, SymbolClass, Vec3)
from .heuristic import DEFAULT_ELEVATIONS, doc_system, normalized_position, pipe_medium
from .meshes import object_parts
from .plan import RoutingPlan
from .scene import placement_transform
from .solver import axis_height, nozzle_normal, placement_box, port_key, scale_for_od
from .zones import Box3, EngineRoom

S = SymbolClass
Pt = tuple[float, float, float]
Cell = tuple[int, int, int]

PIPE_CLEARANCE = 0.1  # m between pipe wall and equipment / inline item bodies
PIPE_GAP = 0.05  # m between the walls of two pipes that share no terminal
SOFT_PIPE = 6.0  # cost of entering a cell that is only soft-blocked (see _search)
SOFT_BODY = 40.0  # relaxed mode: cost of entering an item body or another chain's terminal cell
SOFT_NEAR = 0.6  # cost of running close to equipment (outside the hard inflation)
MODES = ("clean", "rings", "relaxed")
_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
_TOL = 0.02  # m of overlap below which bodies only touch


# ---------------------------------------------------------------------------- small geometry


def _add(a: Pt, b: Sequence[float], s: float = 1.0) -> Pt:
    return (a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2])


def _dist(a: Pt, b: Pt) -> float:
    return math.dist(a, b)


def _simplify(pts: list[Pt], tol: float = 1e-6) -> list[Pt]:
    """Drop duplicate and collinear interior points."""
    out: list[Pt] = []
    for p in pts:
        if out and _dist(out[-1], p) < tol:
            continue
        out.append(p)
    i = 1
    while i < len(out) - 1:
        a, b, c = out[i - 1], out[i], out[i + 1]
        d1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        d2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
        cross = (d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0])
        if max(abs(x) for x in cross) < tol:
            out.pop(i)  # collinear (straight on, or a back-tracking spike)
            i = max(1, i - 1)
        else:
            i += 1
    return out


def _orthogonalize(pts: Sequence[Pt]) -> list[Pt]:
    """Insert corner points so every segment is parallel to one axis (x, then y, then z)."""
    out: list[Pt] = [pts[0]]
    for p in pts[1:]:
        q = out[-1]
        if sum(1 for i in range(3) if abs(p[i] - q[i]) > 1e-9) > 1:
            cur = list(q)
            for ax in range(3):
                if abs(p[ax] - cur[ax]) > 1e-9:
                    cur[ax] = p[ax]
                    out.append(tuple(cur))  # type: ignore[arg-type]
        else:
            out.append(p)
    return out


def segment_hits_box(a: Pt, b: Pt, box: Box3, shrink: float = 0.02) -> bool:
    """True if segment a-b passes through the interior of ``box`` shrunk by ``shrink`` (slab test)."""
    lo = (box.x0 + shrink, box.y0 + shrink, box.z0 + shrink)
    hi = (box.x1 - shrink, box.y1 - shrink, box.z1 - shrink)
    if any(lo[i] >= hi[i] for i in range(3)):
        return False
    t0, t1 = 0.0, 1.0
    for i in range(3):
        d = b[i] - a[i]
        if abs(d) < 1e-12:
            if a[i] <= lo[i] or a[i] >= hi[i]:
                return False
        else:
            ta, tb = (lo[i] - a[i]) / d, (hi[i] - a[i]) / d
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 >= t1:
                return False
    return True


def box_overlaps(lo: np.ndarray, hi: np.ndarray, LO: np.ndarray, HI: np.ndarray, tol: float = _TOL) -> np.ndarray:
    """Rows of the boxes (LO, HI) that overlap the box (lo, hi) by more than ``tol`` on every axis."""
    if len(LO) == 0:
        return np.zeros(0, dtype=bool)
    return np.all(np.minimum(hi, HI) - np.maximum(lo, LO) > tol, axis=1)


def _box_arr(b: Box3) -> tuple[np.ndarray, np.ndarray]:
    return np.array([b.x0, b.y0, b.z0]), np.array([b.x1, b.y1, b.z1])


_BOUNDS: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _local_bounds(cls: SymbolClass, size: tuple[float, float, float], act: ActuatorType
                  ) -> tuple[np.ndarray, np.ndarray]:
    key = (cls, tuple(round(float(x), 4) for x in size), act)
    b = _BOUNDS.get(key)
    if b is None:
        parts = object_parts(cls, size, act)
        lo = np.min([p.mesh.bounds[0] for p in parts], axis=0)
        hi = np.max([p.mesh.bounds[1] for p in parts], axis=0)
        b = (lo, hi)
        if len(_BOUNDS) > 20000:
            _BOUNDS.clear()
        _BOUNDS[key] = b
    return b


def item_box(pl: Placement3D, obj: Optional[PIDObject]) -> Box3:
    """World AABB of an object's rendered mesh (stems, handwheels, actuators; inline items on
    vertical runs turned as in the scene), not just its nominal size box."""
    cat = obj.category if obj is not None else Category.EQUIPMENT
    act = obj.actuator if obj is not None else ActuatorType.NONE
    lo, hi = _local_bounds(obj.cls if obj is not None else pl.cls, pl.size.as_tuple(), act)
    corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = corners @ placement_transform(pl, cat).T
    mn, mx = w[:, :3].min(axis=0), w[:, :3].max(axis=0)
    return Box3(float(mn[0]), float(mn[1]), float(mn[2]), float(mx[0]), float(mx[1]), float(mx[2]))


# ---------------------------------------------------------------------------- voxel grid


class _BoxSet:
    """Growable list of axis-aligned boxes (with a radius and a tag) and cached numpy views."""

    def __init__(self) -> None:
        self.lo: list[tuple[float, float, float]] = []
        self.hi: list[tuple[float, float, float]] = []
        self.r: list[float] = []
        self.tag: list[Any] = []
        self._arr: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def __len__(self) -> int:
        return len(self.lo)

    def add(self, lo: Sequence[float], hi: Sequence[float], r: float = 0.0, tag: Any = None) -> None:
        self.lo.append((float(lo[0]), float(lo[1]), float(lo[2])))
        self.hi.append((float(hi[0]), float(hi[1]), float(hi[2])))
        self.r.append(float(r))
        self.tag.append(tag)
        self._arr = None

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._arr is None:
            self._arr = (np.array(self.lo, dtype=float).reshape(-1, 3), np.array(self.hi, dtype=float).reshape(-1, 3),
                         np.array(self.r, dtype=float))
        return self._arr


class VoxelGrid:
    """Occupancy over the room: equipment (per inflation), routed pipe segments, inline item bodies,
    terminal cells (node centres and nozzle approaches) and reserved cells next to routing nodes."""

    def __init__(self, room: EngineRoom, res: float, boxes: Sequence[Box3]):
        self.room = room
        self.res = res
        b = room.bounds
        self.origin = (b.x0, b.y0, b.z0)
        self.shape = (max(1, int(math.ceil((b.x1 - b.x0) / res))), max(1, int(math.ceil((b.y1 - b.y0) / res))),
                      max(1, int(math.ceil((b.z1 - b.z0) / res))))
        self.boxes = list(boxes)
        self._blocked: dict[float, np.ndarray] = {}
        self.pipes = np.zeros(self.shape, dtype=bool)  # cells crossed by routed centre-lines
        self.reserved = np.zeros(self.shape, dtype=bool)  # neighbours of routing nodes (soft for others)
        self.term = np.zeros(self.shape, dtype=bool)  # terminal cells (hard for other chains)
        self.term_owner: dict[Cell, set[str]] = defaultdict(set)
        self.ring_owner: dict[Cell, set[str]] = defaultdict(set)
        self.segs = _BoxSet()  # routed segments: radius, tag = chain index
        self.items = _BoxSet()  # inline / hub / isolated bodies: tag = (object id, exempt nodes)
        self.approaches = _BoxSet()  # nozzle approach segments: radius, tag = terminal key

    def center(self, c: Cell) -> Pt:
        return tuple(self.origin[i] + (c[i] + 0.5) * self.res for i in range(3))  # type: ignore[return-value]

    def cell(self, p: Sequence[float]) -> Cell:
        return tuple(min(self.shape[i] - 1, max(0, int(math.floor((p[i] - self.origin[i]) / self.res))))
                     for i in range(3))  # type: ignore[return-value]

    def inside(self, c: Cell) -> bool:
        return all(0 <= c[i] < self.shape[i] for i in range(3))

    def blocked(self, inflate: float) -> np.ndarray:
        """Cells whose centre lies in an equipment box inflated by ``inflate`` (cached per 5 cm)."""
        key = round(math.ceil(inflate / 0.05) * 0.05, 2)
        arr = self._blocked.get(key)
        if arr is None:
            arr = np.zeros(self.shape, dtype=bool)
            for bx in self.boxes:
                lo = (bx.x0 - key, bx.y0 - key, bx.z0 - key)
                hi = (bx.x1 + key, bx.y1 + key, bx.z1 + key)
                sl = []
                for i in range(3):
                    a = int(math.ceil((lo[i] - self.origin[i]) / self.res - 0.5))
                    b = int(math.floor((hi[i] - self.origin[i]) / self.res - 0.5))
                    a, b = max(0, a), min(self.shape[i] - 1, b)
                    sl.append(slice(a, b + 1) if a <= b else slice(0, 0))
                arr[tuple(sl)] = True
            self._blocked[key] = arr
        return arr

    def cell_slices(self, lo: Sequence[float], hi: Sequence[float]) -> Optional[tuple[slice, slice, slice]]:
        """Cells whose centre lies strictly inside the box (lo, hi); None if there are none."""
        out = []
        for i in range(3):
            a = int(math.floor((lo[i] - self.origin[i]) / self.res - 0.5)) + 1
            b = int(math.ceil((hi[i] - self.origin[i]) / self.res - 0.5)) - 1
            a, b = max(0, a), min(self.shape[i] - 1, b)
            if a > b:
                return None
            out.append(slice(a, b + 1))
        return tuple(out)  # type: ignore[return-value]

    def crossed(self, a: Pt, b: Pt) -> list[Cell]:
        """Cells an axis-aligned segment passes through."""
        ca, cb = self.cell(a), self.cell(b)
        rng = [range(min(ca[i], cb[i]), max(ca[i], cb[i]) + 1) for i in range(3)]
        return [(i, j, k) for i in rng[0] for j in rng[1] for k in rng[2]]

    def add_route(self, pts: Sequence[Pt], r: float, chain: int) -> None:
        for a, b in zip(pts, pts[1:]):
            self.segs.add(np.minimum(a, b), np.maximum(a, b), r, chain)
            for c in self.crossed(a, b):
                self.pipes[c] = True

    def free(self, c: Cell, inflate: float) -> bool:
        return not (self.blocked(inflate)[c] or self.pipes[c] or self.reserved[c] or self.term[c])

    def free_cells(self, c: Cell, inflate: float, radius: int = 6, prefer: Optional[Sequence[float]] = None,
                   hard_only: bool = False, owner: Optional[str] = None) -> Iterator[Cell]:
        """Free cells around ``c``, nearest first (shell search, ties broken toward ``prefer``).
        Equipment (inflated), inline bodies and other owners' terminal cells always count as
        occupied; pipes and reserved cells too in a first pass unless ``hard_only``."""
        blocked = self.blocked(inflate)
        LO, HI, _ = self.items.arrays()

        def ok(q: Cell, strict: bool) -> bool:
            if blocked[q]:
                return False
            if self.term[q] and (owner is None or owner not in self.term_owner.get(q, ())):
                return False
            if strict and not hard_only and (self.pipes[q] or self.reserved[q]):
                return False
            if len(LO):
                p = np.array(self.center(q))
                if np.any(np.all((p > LO - inflate) & (p < HI + inflate), axis=1)):
                    return False
            return True

        for strict in (True, False):
            for r in range(0, radius + 1):
                shell: list[tuple[float, Cell]] = []
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        for dk in range(-r, r + 1):
                            if max(abs(di), abs(dj), abs(dk)) != r:
                                continue
                            q = (c[0] + di, c[1] + dj, c[2] + dk)
                            if not self.inside(q):
                                continue
                            d = abs(di) + abs(dj) + abs(dk)
                            if prefer is not None:
                                d -= 0.4 * (di * prefer[0] + dj * prefer[1] + dk * prefer[2])
                            shell.append((d, q))
                for _, q in sorted(shell):
                    if ok(q, strict) and (strict or not ok(q, True)):
                        yield q

    def nearest_free(self, c: Cell, inflate: float, radius: int = 6, prefer: Optional[Sequence[float]] = None,
                     hard_only: bool = False, owner: Optional[str] = None) -> Cell:
        """First of :meth:`free_cells`; falls back to the nearest equipment-free cell, then ``c``."""
        for q in self.free_cells(c, inflate, radius, prefer, hard_only, owner):
            return q
        blocked = self.blocked(inflate)
        for r in range(0, radius + 1):  # last resort: anything outside the inflated equipment
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    for dk in range(-r, r + 1):
                        q = (c[0] + di, c[1] + dj, c[2] + dk)
                        if self.inside(q) and not blocked[q]:
                            return q
        return c


def astar(cost: np.ndarray, start: Cell, goal: Cell, band_k: float, start_dir: int = 6, bend: float = 2.0,
          band_w: float = 0.35, eps: float = 1.25, max_expansions: int = 120_000) -> Optional[list[Cell]]:
    """Orthogonal A* on a cell-cost grid (sub-box coordinates; cost < 0 = impassable).

    Step cost 1 + the entered cell's cost, +``bend`` per change of direction, +``band_w`` x
    |k - band_k| per horizontal step (keeps runs on the preferred elevation band); weighted
    heuristic ``eps`` x Manhattan distance (bounded-suboptimal, much faster).
    """
    sx, sy, sz = cost.shape
    flat = cost.ravel().tolist()
    gi, gj, gk = goal
    inf = float("inf")
    best_g: dict[int, float] = {}
    parent: dict[int, int] = {}
    s_state = ((start[0] * sy + start[1]) * sz + start[2]) * 7 + start_dir
    best_g[s_state] = 0.0
    h0 = abs(start[0] - gi) + abs(start[1] - gj) + abs(start[2] - gk)
    heap = [(eps * h0, h0, 0.0, start[0], start[1], start[2], start_dir)]
    expansions = 0
    goal_state = None
    while heap:
        f, h, g, i, j, k, d = heapq.heappop(heap)
        state = ((i * sy + j) * sz + k) * 7 + d
        if g > best_g.get(state, inf):
            continue
        if i == gi and j == gj and k == gk:
            goal_state = state
            break
        expansions += 1
        if expansions > max_expansions:
            return None
        band_pen = band_w * abs(k - band_k)
        for nd, (di, dj, dk) in enumerate(_DIRS):
            if d < 6 and nd == (d ^ 1):  # no immediate reversal
                continue
            ni, nj, nk = i + di, j + dj, k + dk
            if ni < 0 or nj < 0 or nk < 0 or ni >= sx or nj >= sy or nk >= sz:
                continue
            c = flat[(ni * sy + nj) * sz + nk]
            if c < 0:
                continue
            ng = g + 1.0 + c + (bend if (d < 6 and nd != d) else 0.0) + (band_pen if dk == 0 else 0.0)
            ns = ((ni * sy + nj) * sz + nk) * 7 + nd
            if ng < best_g.get(ns, inf):
                best_g[ns] = ng
                parent[ns] = state
                nh = abs(ni - gi) + abs(nj - gj) + abs(nk - gk)
                heapq.heappush(heap, (ng + eps * nh, nh, ng, ni, nj, nk, nd))
    if goal_state is None:
        return None
    cells: list[Cell] = []
    s = goal_state
    while True:
        c = s // 7
        k = c % sz
        j = (c // sz) % sy
        i = c // (sz * sy)
        cells.append((i, j, k))
        if s == s_state or s not in parent:
            break
        s = parent[s]
    cells.reverse()
    return cells


# ---------------------------------------------------------------------------- graph contraction


@dataclass
class _End:
    node: str  # object id, junction id, or "free:<pipe>:<0|1>"
    key: Optional[str]  # nozzle / port key on that node (objects only)


@dataclass
class _Chain:
    start: _End
    end: Optional[_End]  # None = dead end at the last item
    pipes: list[str]
    rev: list[bool] = field(default_factory=list)  # pipe traversed end -> start
    items: list[tuple[str, str, Optional[str]]] = field(default_factory=list)  # (object id, entry key, exit key)


@dataclass
class RouteStats:
    chains: int = 0
    astar: int = 0
    widened: int = 0
    relaxed: int = 0
    retried: int = 0  # chains re-routed with room for their inline items
    item_clash_chains: int = 0  # chains whose items could not be placed clear of everything
    fallback: int = 0
    stubs: int = 0
    seconds: float = 0.0
    collisions: int = 0
    length_m: float = 0.0
    audit: dict[str, Any] = field(default_factory=dict)


def _pipe_od(p: PipeRun) -> float:
    if p.kind == PipeKind.INSTRUMENT_TAP and not p.size:
        return pipe_od_m("15A")
    size = p.size
    if not size and p.line_number:
        from ..catalog import parse_line_number

        size = (parse_line_number(p.line_number) or {}).get("size")
    return pipe_od_m(size)


def _contract(doc: PIDDocument, placed: set[str]) -> tuple[list[_Chain], set[str], dict[str, list[tuple[PipeRun, int]]],
                                                            list[str]]:
    """Chains between major nodes; returns (chains, hubs, adjacency, notes)."""
    objs = doc.object_by_id()
    jids = {j.id for j in doc.junctions}
    notes: list[str] = []
    adj: dict[str, list[tuple[PipeRun, int]]] = defaultdict(list)  # node -> (pipe, which end 0/1)
    pipes = [p for p in doc.pipes if p.kind != PipeKind.SIGNAL and p.points]

    def node_of(p: PipeRun, which: int) -> str:
        e = p.start if which == 0 else p.end
        n = e.node_id
        if n is not None and (n in objs or n in jids):
            return n
        return f"free:{p.id}:{which}"

    for p in pipes:
        adj[node_of(p, 0)].append((p, 0))
        adj[node_of(p, 1)].append((p, 1))
    deg = {n: len(v) for n, v in adj.items()}
    hubs: set[str] = set()
    for n, d in deg.items():
        if n in objs and n not in placed and d >= 3:
            hubs.add(n)

    def is_major(n: str) -> bool:
        return n in placed or n in jids or n in hubs or n.startswith("free:")

    # components without any major node (loops of inline items) -> promote one hub
    seen: set[str] = set()
    for n in sorted(adj):
        if n in seen:
            continue
        comp, q = [], deque([n])
        seen.add(n)
        while q:
            x = q.popleft()
            comp.append(x)
            for p, w in adj[x]:
                y = node_of(p, 1 - w)
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        if not any(is_major(x) for x in comp):
            hub = max(comp, key=lambda x: (deg.get(x, 0), x))
            hubs.add(hub)
            notes.append(f"{hub}: inline-only pipe loop; object used as routing hub")

    def end_key(p: PipeRun, which: int) -> Optional[str]:
        e = p.start if which == 0 else p.end
        return port_key(e, p.id) if e.object_id and e.object_id in objs else None

    used: set[str] = set()
    chains: list[_Chain] = []
    majors = sorted(n for n in adj if is_major(n))
    for m in majors:
        for p0, w0 in adj[m]:
            if p0.id in used:
                continue
            used.add(p0.id)
            chain = _Chain(start=_End(m, end_key(p0, w0)), end=None, pipes=[p0.id], rev=[w0 == 1])
            p, w = p0, w0
            while True:
                n = node_of(p, 1 - w)
                entry = end_key(p, 1 - w)
                if is_major(n):
                    chain.end = _End(n, entry)
                    break
                others = [(q, wq) for q, wq in adj[n] if q.id != p.id]
                if not others:  # dead end at an inline item
                    chain.items.append((n, entry or f"auto_{p.id}", None))
                    break
                q, wq = others[0]
                if q.id in used:  # loop back into an already routed chain
                    chain.items.append((n, entry or f"auto_{p.id}", None))
                    break
                chain.items.append((n, entry or f"auto_{p.id}", end_key(q, wq) or f"auto_{q.id}"))
                used.add(q.id)
                chain.pipes.append(q.id)
                chain.rev.append(wq == 1)
                p, w = q, wq
            chains.append(chain)
    for p in pipes:
        if p.id not in used:
            notes.append(f"{p.id}: not reachable from any routing node, not routed")
    return chains, hubs, adj, notes


# ---------------------------------------------------------------------------- router context


@dataclass
class _Term:
    """Where a chain starts / ends: a node point or a nozzle with its approach to the grid."""

    key: str  # node id, or "<object id>\\0<nozzle key>"
    node: str
    point: Pt
    normal: Optional[tuple[float, float, float]]
    cell: Cell  # grid cell where the A* path starts / ends
    approach: list[Pt]  # point -> ... -> centre of ``cell``
    cells: list[Cell]  # terminal cells (hard for other chains)
    ring: list[Cell] = field(default_factory=list)  # reserved neighbours of a node (soft for other chains)
    margin: float = 0.0  # inline items keep this far from the terminal along the chain


@dataclass
class _ChainCtx:
    idx: int
    chain: _Chain
    nodes: frozenset[str]  # terminal node ids
    keys: frozenset[str]  # terminal keys
    own: list[Cell]  # terminal + ring cells of its terminals
    terms: list["_Term"] = field(default_factory=list)


@dataclass
class _Ctx:
    doc: PIDDocument
    objs: dict[str, PIDObject]
    room: EngineRoom
    grid: VoxelGrid
    pipes: dict[str, PipeRun]
    placements: dict[str, Placement3D]
    node_pos: dict[str, Pt]
    routing: RoutingPlan
    default_medium: Medium
    deadline: float
    stats: RouteStats
    notes: list[str]
    terms: dict[str, _Term] = field(default_factory=dict)
    chain_keys: list[frozenset[str]] = field(default_factory=list)  # terminal keys per routed chain index
    eq_lo: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    eq_hi: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


def band_z(room: EngineRoom, routing: Optional[RoutingPlan], medium: Medium, line: Optional[str]) -> float:
    """Preferred pipe centre-line elevation for a line / medium."""
    deck, h = DEFAULT_ELEVATIONS.get(medium, DEFAULT_ELEVATIONS[Medium.UNKNOWN])
    if routing is not None:
        by_target = {b.target.strip().upper(): b for b in routing.elevation_bands}
        b = (by_target.get(line.strip().upper()) if line else None) or by_target.get(medium.value.upper())
        if b is not None:
            deck, h = b.deck, b.height_m
    z = room.deck_z(deck) + h
    return min(max(z, 0.3), room.height - 0.3)


def adapt_band(zb: float, za: float, zc: float) -> float:
    """Keep the preferred band unless both ends are well below / above it: then run just above the
    higher end (or just below the lower end) instead of climbing to a far-away band."""
    lo, hi = min(za, zc), max(za, zc)
    return min(max(zb, lo - 0.5), hi + 1.0)


def _drawing_map(doc: PIDDocument, placements: dict[str, Placement3D], room: EngineRoom):
    """Affine page -> room (x, y) map fitted on placed equipment (normalised fallback)."""
    objs = doc.object_by_id()
    us, vs, xs, ys = [], [], [], []
    for oid, pl in placements.items():
        o = objs.get(oid)
        if o is None:
            continue
        u, v = normalized_position(doc, o.center.x, o.center.y)
        b = placement_box(pl)
        cx, cy, _ = b.center
        us.append(u), vs.append(v), xs.append(cx), ys.append(cy)

    def fit(t: list[float], w: list[float], default) -> tuple[float, float]:
        if len(t) >= 3 and np.std(t) > 0.05:
            a, c = np.polyfit(t, w, 1)
            if abs(a) > 1e-3:
                return float(a), float(c)
        return default

    ax, bx = fit(us, xs, (room.length, 0.0))
    ay, by = fit(vs, ys, (-room.breadth, room.breadth / 2))

    def mapping(x: float, y: float) -> tuple[float, float]:
        u, v = normalized_position(doc, x, y)
        X = min(max(ax * u + bx, 0.5), room.length - 0.5)
        Y = min(max(ay * v + by, -room.breadth / 2 + 0.5), room.breadth / 2 - 0.5)
        return X, Y

    return mapping


def _chain_medium_line(ctx: _Ctx, chain: _Chain) -> tuple[Medium, Optional[str], float]:
    ps = [ctx.pipes[p] for p in chain.pipes]
    big = max(ps, key=lambda p: (_pipe_od(p), p.kind == PipeKind.PROCESS))
    od = max(_pipe_od(p) for p in ps)
    return pipe_medium(big, ctx.default_medium), big.line_number, od


# ---------------------------------------------------------------------------- clash tests


def _body_hits(ctx: _Ctx, box: Box3, keys: frozenset[str] = frozenset(), nodes: frozenset[str] = frozenset(),
               extra: Sequence[Box3] = (), skip_items: frozenset[str] = frozenset()) -> int:
    """Number of things a body (inline item, stub segment) would clash with: equipment, registered
    bodies (except those exempt for ``nodes`` / named in ``skip_items``), ``extra`` boxes, routed
    pipes (box of the centre-line inflated by the pipe radius), approaches of terminals other than
    ``keys``, and the room boundary."""
    g = ctx.grid
    lo, hi = _box_arr(box)
    n = int(box_overlaps(lo, hi, ctx.eq_lo, ctx.eq_hi).sum())
    if len(g.items):
        LO, HI, _ = g.items.arrays()
        hit = box_overlaps(lo, hi, LO, HI)
        for i in np.nonzero(hit)[0]:
            oid, ex = g.items.tag[i]
            if not (ex & nodes) and oid not in skip_items:
                n += 1
    for b in extra:
        elo, ehi = _box_arr(b)
        n += int(np.all(np.minimum(hi, ehi) - np.maximum(lo, elo) > _TOL))
    if len(g.segs):
        LO, HI, R = g.segs.arrays()
        n += int(box_overlaps(lo, hi, LO - R[:, None], HI + R[:, None]).sum())
    if len(g.approaches):
        LO, HI, R = g.approaches.arrays()
        hit = box_overlaps(lo, hi, LO - R[:, None], HI + R[:, None])
        n += sum(1 for i in np.nonzero(hit)[0] if g.approaches.tag[i] not in keys)
    rb = ctx.room.bounds
    if box.x0 < rb.x0 - 1e-3 or box.y0 < rb.y0 - 1e-3 or box.z0 < rb.z0 - 1e-3 or box.x1 > rb.x1 + 1e-3 or \
            box.y1 > rb.y1 + 1e-3 or box.z1 > rb.z1 + 1e-3:
        n += 1
    return n


def _segment_box(a: Pt, b: Pt, r: float) -> Box3:
    return Box3(min(a[0], b[0]) - r, min(a[1], b[1]) - r, min(a[2], b[2]) - r,
                max(a[0], b[0]) + r, max(a[1], b[1]) + r, max(a[2], b[2]) + r)


def _path_hits(ctx: _Ctx, pts: Sequence[Pt], r: float, keys: frozenset[str], nodes: frozenset[str],
               skip_start: float = 0.0) -> int:
    """Clashes of a polyline of radius ``r`` (its first ``skip_start`` metres are not checked)."""
    n = 0
    s = 0.0
    for a, b in zip(pts, pts[1:]):
        L = _dist(a, b)
        if s + L <= skip_start + 1e-9:
            s += L
            continue
        if s < skip_start:
            t = (skip_start - s) / L
            a = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)
        n += _body_hits(ctx, _segment_box(a, b, r), keys, nodes)
        s += L
    return n


# ---------------------------------------------------------------------------- terminals


def _join(ctx: _Ctx, p: Pt, q: Pt) -> list[Pt]:
    """Orthogonal connection p -> q choosing the axis order that avoids equipment / bodies best."""
    if _dist(p, q) < 1e-9:
        return [q]
    g = ctx.grid
    LO, HI, _ = g.items.arrays()
    bodies = list(g.boxes) + [Box3(*LO[i], *HI[i]) for i in range(len(LO))]
    best, best_hits = None, None
    for order in ((2, 0, 1), (0, 1, 2), (1, 0, 2), (0, 2, 1), (1, 2, 0), (2, 1, 0)):
        pts = [p]
        cur = list(p)
        for ax in order:
            if abs(q[ax] - cur[ax]) > 1e-9:
                cur[ax] = q[ax]
                pts.append(tuple(cur))  # type: ignore[arg-type]
        hits = sum(1 for a, b in zip(pts, pts[1:]) for bx in bodies if segment_hits_box(a, b, bx))
        if best_hits is None or hits < best_hits:
            best, best_hits = pts, hits
            if hits == 0:
                break
    return best[1:]  # type: ignore[index]


def _mark_term(g: VoxelGrid, key: str, cells: Sequence[Cell]) -> None:
    for c in cells:
        g.term[c] = True
        g.term_owner[c].add(key)


def _node_term(ctx: _Ctx, n: str, pos: Pt, inflate: float, margin: float) -> _Term:
    """Snap a junction / hub / free end to a free cell; its cell is a terminal, its neighbours reserved."""
    g = ctx.grid
    c = g.nearest_free(g.cell(pos), inflate, radius=8, owner=n)
    hard = g.blocked(0.25)
    ring = [q for q in ((c[0] + di, c[1] + dj, c[2] + dk) for di, dj, dk in _DIRS)
            if g.inside(q) and not hard[q] and not g.term[q]]
    _mark_term(g, n, [c])
    for q in ring:
        g.reserved[q] = True
        g.ring_owner[q].add(n)
    p = g.center(c)
    return _Term(n, n, p, None, c, [p], [c], ring, margin)


def _nozzle_term(ctx: _Ctx, key: str, pl: Placement3D, nz_key: str, od: float) -> _Term:
    """Approach of one equipment nozzle: out along the face normal, then to the centre of the
    nearest free cell not used by another terminal (in front of the nozzle where possible)."""
    g = ctx.grid
    nz = pl.nozzles.get(nz_key)
    if nz is None:  # unexpected port: top centre of the box
        b = placement_box(pl)
        nz = Vec3(x=b.center[0], y=b.center[1], z=b.z1)
        pl.nozzles[nz_key] = nz
    n = nozzle_normal(pl, nz)
    a = nz.as_tuple()
    stub = max(0.2, od / 2 + 0.15)
    sa = _add(a, n, stub)

    def plan(c: Cell) -> tuple[list[Pt], list[Cell]]:
        approach = _simplify([a, sa] + _join(ctx, sa, g.center(c)))
        cells: list[Cell] = []
        for p, q in zip(approach, approach[1:]):
            for x in g.crossed(p, q):
                if x not in cells:
                    cells.append(x)
        if c not in cells:
            cells.append(c)
        return approach, cells

    def clear(approach: list[Pt], cells: list[Cell]) -> bool:
        """No other terminal's cells crossed, no other approach within both radii + PIPE_GAP."""
        if any(g.term[x] and key not in g.term_owner.get(x, ()) for x in cells):
            return False
        if len(g.approaches):
            LO, HI, R = g.approaches.arrays()
            for p, q in zip(approach, approach[1:]):
                lo, hi = np.minimum(p, q) - od / 2 - PIPE_GAP, np.maximum(p, q) + od / 2 + PIPE_GAP
                hit = box_overlaps(lo, hi, LO - R[:, None], HI + R[:, None], tol=1e-3)
                if any(g.approaches.tag[i] != key for i in np.nonzero(hit)[0]):
                    return False
        return True

    first = None
    chosen = None
    for k, c in enumerate(g.free_cells(g.cell(sa), od / 2 + PIPE_CLEARANCE, prefer=n, owner=key)):
        cand = (c, *plan(c))
        first = first or cand
        if clear(cand[1], cand[2]):
            chosen = cand
            break
        if k > 60:
            break
    if chosen is None:
        chosen = first if first is not None else (g.nearest_free(g.cell(sa), od / 2 + PIPE_CLEARANCE, prefer=n,
                                                                 owner=key),)
        if len(chosen) == 1:
            chosen = (chosen[0], *plan(chosen[0]))
        ctx.notes.append(f"{pl.object_id}.{nz_key}: nozzle approach could not be kept clear of neighbouring "
                         f"nozzles' approaches")
    c, approach, cells = chosen
    for p, q in zip(approach, approach[1:]):
        g.approaches.add(np.minimum(p, q), np.maximum(p, q), od / 2, key)
    _mark_term(g, key, cells)
    return _Term(key, pl.object_id, a, n, c, approach, cells)


def _term_key(end: _End, placed: dict[str, Placement3D]) -> str:
    return f"{end.node}\x00{end.key}" if end.node in placed else end.node


# ---------------------------------------------------------------------------- search


def _sub_box(grid: VoxelGrid, a: Cell, b: Cell, band_k: int, margin_xy: int, margin_z: int
             ) -> tuple[slice, slice, slice]:
    lo = [min(a[i], b[i]) for i in range(3)]
    hi = [max(a[i], b[i]) for i in range(3)]
    lo[2], hi[2] = min(lo[2], band_k), max(hi[2], band_k)
    m = (margin_xy, margin_xy, margin_z)
    return tuple(slice(max(0, lo[i] - m[i]), min(grid.shape[i], hi[i] + m[i] + 1)) for i in range(3))  # type: ignore


def _obstacles(ctx: _Ctx, sl: tuple[slice, slice, slice], od: float, inflate: float, cc: _ChainCtx
               ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Item bodies, other pipes, and per shared terminal (same junction / hub / free end, or the
    same nozzle) the pipes of chains ending there,
    on the sub-box ``sl`` (pipes inflated by both radii + PIPE_GAP, bodies by ``inflate``)."""
    g = ctx.grid
    shape = tuple(s.stop - s.start for s in sl)
    wlo = np.array([g.origin[i] + sl[i].start * g.res for i in range(3)])
    whi = np.array([g.origin[i] + sl[i].stop * g.res for i in range(3)])
    items = np.zeros(shape, dtype=bool)
    other = np.zeros(shape, dtype=bool)
    shared: dict[str, np.ndarray] = {}

    def mark(arr: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
        gs = g.cell_slices(lo, hi)
        if gs is None:
            return
        loc = []
        for i in range(3):
            a = max(gs[i].start, sl[i].start) - sl[i].start
            b = min(gs[i].stop, sl[i].stop) - sl[i].start
            if a >= b:
                return
            loc.append(slice(a, b))
        arr[tuple(loc)] = True

    if len(g.items):
        LO, HI, _ = g.items.arrays()
        sel = np.nonzero(np.all((LO - inflate < whi) & (HI + inflate > wlo), axis=1))[0]
        for i in sel:
            if g.items.tag[i][1] & cc.nodes:  # a hub this chain ends at
                continue
            mark(items, LO[i] - inflate, HI[i] + inflate)
    if len(g.segs):
        LO, HI, R = g.segs.arrays()
        RR = R + od / 2 + PIPE_GAP
        sel = np.nonzero(np.all((LO - RR[:, None] < whi) & (HI + RR[:, None] > wlo), axis=1))[0]
        for i in sel:
            common = ctx.chain_keys[g.segs.tag[i]] & cc.keys
            if not common:
                mark(other, LO[i] - RR[i], HI[i] + RR[i])
            for n in common:
                if n not in shared:
                    shared[n] = np.zeros(shape, dtype=bool)
                mark(shared[n], LO[i] - RR[i], HI[i] + RR[i])
    return items, other, shared


def _search(ctx: _Ctx, cc: _ChainCtx, ca: Cell, cb: Cell, od: float, zb: float, start_dir: int, wide: bool,
            mode: str, strict: bool = False, block: Sequence[Cell] = ()) -> Optional[tuple[list[Pt], list[Cell]]]:
    """A* between two cells in a sub-box (or the whole grid); returns (points, global cells).

    ``mode="clean"``: equipment (inflated by OD/2 + clearance), item bodies, other pipes, other
    terminals' cells and the reserved neighbours of other nodes are hard; pipes of chains sharing
    a terminal (the same junction / hub, or the same nozzle) are soft (SOFT_PIPE) only next to that
    terminal (its cells, its ring and the neighbours of its search cell).
    ``mode="rings"``: as clean, but other nodes' reserved neighbours are soft.
    ``mode="relaxed"``: only equipment is hard; pipes and reserved cells cost SOFT_PIPE, item
    bodies and other terminals' cells SOFT_BODY.
    Unless ``strict``, equipment-only hard cells next to the terminals are softened, so a terminal
    squeezed between equipment never becomes an unreachable pocket. ``block``: extra hard cells.
    """
    g = ctx.grid
    inflate = od / 2 + PIPE_CLEARANCE
    band_k = int(round((zb - g.origin[2]) / g.res - 0.5))
    band_k = min(max(band_k, 0), g.shape[2] - 1)
    span = max(abs(ca[i] - cb[i]) for i in range(3))
    if wide:
        sl = (slice(0, g.shape[0]), slice(0, g.shape[1]), slice(0, g.shape[2]))
    else:
        sl = _sub_box(g, ca, cb, band_k, max(8, int(0.3 * span)), 4)
    off = (sl[0].start, sl[1].start, sl[2].start)
    eq = g.blocked(inflate)[sl]
    near = g.blocked(inflate + 0.3)[sl]
    cost = np.where(near, SOFT_NEAR, 0.0)
    shape = cost.shape

    def local(c: Cell) -> Optional[Cell]:
        q = (c[0] - off[0], c[1] - off[1], c[2] - off[2])
        return q if all(0 <= q[i] < shape[i] for i in range(3)) else None

    la, lb = local(ca), local(cb)
    if la is None or lb is None:
        return None
    items, other, shared = _obstacles(ctx, sl, od, inflate, cc)
    own = np.zeros(shape, dtype=bool)
    for c in cc.own:
        q = local(c)
        if q is not None:
            own[q] = True
    shared_any = np.zeros(shape, dtype=bool)
    shared_soft = np.zeros(shape, dtype=bool)
    for n, m in shared.items():
        near = np.zeros(shape, dtype=bool)
        for t in cc.terms:
            if t.key != n:
                continue
            for c in t.cells + t.ring:
                q = local(c)
                if q is not None:
                    near[q] = True
            lt = local(t.cell)
            if lt is not None:
                for di, dj, dk in _DIRS:
                    q = (lt[0] + di, lt[1] + dj, lt[2] + dk)
                    if all(0 <= q[i] < shape[i] for i in range(3)):
                        near[q] = True
        shared_any |= m
        shared_soft |= m & near
    term_other = g.term[sl] & ~own
    ring_other = g.reserved[sl] & ~own
    if mode in ("clean", "rings"):
        obst = items | other | (shared_any & ~shared_soft) | term_other
        soft = shared_soft.copy()
        if mode == "clean":
            obst |= ring_other
        else:
            soft |= ring_other
        cost[soft] = np.maximum(cost[soft], SOFT_PIPE)
    else:
        obst = np.zeros(shape, dtype=bool)
        soft = other | shared_any | ring_other
        cost[soft] = np.maximum(cost[soft], SOFT_PIPE)
        body = items | term_other
        cost[body] = np.maximum(cost[body], SOFT_BODY)
    hard = eq | obst
    cost[hard] = -1.0
    for t in (la, lb):
        cost[t] = 0.0
        if strict:
            continue
        for di, dj, dk in _DIRS:
            q = (t[0] + di, t[1] + dj, t[2] + dk)
            if all(0 <= q[i] < shape[i] for i in range(3)) and eq[q] and not obst[q]:
                cost[q] = SOFT_PIPE
    for c in block:
        q = local(c)
        if q is not None and q != la and q != lb:
            cost[q] = -1.0
    lab, _ = ndimage.label(cost >= 0)  # cheap reachability test before the A*
    if lab[la] == 0 or lab[la] != lab[lb]:
        return None
    cells = astar(cost, la, lb, band_k - off[2], start_dir=start_dir, max_expansions=300_000 if wide else 100_000)
    if cells is None:
        return None
    glob = [(c[0] + off[0], c[1] + off[1], c[2] + off[2]) for c in cells]
    return [g.center(c) for c in glob], glob


def _dir_index(n: Optional[Sequence[float]]) -> int:
    if n is None:
        return 6
    ax = max(range(3), key=lambda i: abs(n[i]))
    return ax * 2 + (0 if n[ax] > 0 else 1)


def _compose(ta: _Term, mid: Sequence[Pt], tb: _Term) -> list[Pt]:
    pts = list(ta.approach[:-1]) + list(mid) + list(reversed(tb.approach))[1:]
    return _simplify(_orthogonalize(pts))


def _clashes(ctx: _Ctx, pts: Sequence[Pt]) -> int:
    return sum(1 for p, q in zip(pts, pts[1:]) for bx in ctx.grid.boxes if segment_hits_box(p, q, bx))


def _route_between(ctx: _Ctx, cc: _ChainCtx, ta: _Term, tb: _Term, od: float, zb: float, label: str,
                   modes: Sequence[str] = MODES) -> tuple[list[Pt], str]:
    """Polyline ta -> tb (orthogonal) and the mode that produced it: clean search, rings soft, all
    obstacles soft (noted), then a Manhattan fallback ("fallback", noted)."""
    g = ctx.grid
    ca, cb = ta.cell, tb.cell
    if ca == cb:
        return _compose(ta, [g.center(ca)], tb), "clean"
    sd = _dir_index(ta.normal)
    for mode in modes:
        res = None
        for wide in (False, True):
            if time.time() >= ctx.deadline:
                break
            if wide:
                ctx.stats.widened += 1
            res = _search(ctx, cc, ca, cb, od, zb, sd, wide, mode)
            if res is not None:
                break
        if res is None:
            continue
        pts = _compose(ta, res[0], tb)
        if _clashes(ctx, pts) and time.time() < ctx.deadline:
            strict = _search(ctx, cc, ca, cb, od, zb, sd, False, mode, strict=True)
            if strict is not None:
                spts = _compose(ta, strict[0], tb)
                if _clashes(ctx, spts) < _clashes(ctx, pts):
                    pts = spts
        ctx.stats.astar += 1
        if mode == "relaxed":
            ctx.stats.relaxed += 1
            ctx.notes.append(f"{label}: no path clear of other pipes / inline items / terminals; routed with "
                             f"them as soft costs (may touch or cross them)")
        return pts, mode
    if "relaxed" not in modes:
        return [], "none"
    ctx.stats.fallback += 1
    ctx.notes.append(f"{label}: Manhattan fallback route (no path outside the inflated equipment "
                     f"within the search limits / time budget); may cross equipment")
    sa, sb = g.center(ca), g.center(cb)
    mid = [sa, (sa[0], sa[1], zb), (sb[0], sa[1], zb), (sb[0], sb[1], zb), sb]
    return _compose(ta, mid, tb), "fallback"


def _route_with_leg(ctx: _Ctx, cc: _ChainCtx, ta: _Term, tb: _Term, od: float, zb: float, need: float,
                    at_start: bool) -> tuple[list[Pt], str]:
    """Route with a straight leg of at least ``need`` metres out of one terminal (along the nozzle
    normal, or along a free axis at a junction, preferring the direction of the other end), then A*
    from the leg's end: room for the inline items of a chain whose ends are close together."""
    g = ctx.grid
    t, other = (ta, tb) if at_start else (tb, ta)
    k = max(2, int(math.ceil(need / g.res)))
    if t.normal is not None:
        ax = max(range(3), key=lambda i: abs(t.normal[i]))
        dirs = [tuple((1 if t.normal[ax] > 0 else -1) if i == ax else 0 for i in range(3))]
    else:
        to = np.array(other.point) - np.array(t.point)
        dirs = sorted(_DIRS, key=lambda d: (d[2] != 0, -float(np.dot(d, to))))
    inflate = od / 2 + PIPE_CLEARANCE
    eq = g.blocked(inflate)
    for d in dirs:
        cells = [(t.cell[0] + d[0] * i, t.cell[1] + d[1] * i, t.cell[2] + d[2] * i) for i in range(1, k + 1)]
        if not all(g.inside(c) and not eq[c] and not (g.term[c] and t.key not in g.term_owner.get(c, ()))
                   for c in cells):
            continue
        a, w = g.center(t.cell), g.center(cells[-1])
        if _body_hits(ctx, _segment_box(a, w, od / 2 + PIPE_GAP), cc.keys, cc.nodes):
            continue
        res = None
        for mode in MODES[:2]:
            for wide in (False, True):
                if time.time() >= ctx.deadline:
                    return [], "none"
                if at_start:
                    res = _search(ctx, cc, cells[-1], tb.cell, od, zb, _dir_index(d), wide, mode,
                                  block=[t.cell] + cells[:-1])
                else:
                    res = _search(ctx, cc, ta.cell, cells[-1], od, zb, _dir_index(ta.normal), wide, mode,
                                  block=[t.cell] + cells[:-1])
                if res is not None:
                    break
            if res is not None:
                break
        if res is None:
            continue
        leg = [g.center(c) for c in [t.cell] + cells]
        mid = leg + res[0][1:] if at_start else res[0] + leg[::-1][1:]
        return _compose(ta, mid, tb), mode
    return [], "none"


# ---------------------------------------------------------------------------- inline items


def _cum(pts: Sequence[Pt]) -> list[float]:
    out = [0.0]
    for a, b in zip(pts, pts[1:]):
        out.append(out[-1] + _dist(a, b))
    return out


def _point_at(pts: Sequence[Pt], cum: Sequence[float], s: float) -> tuple[Pt, int]:
    s = min(max(s, 0.0), cum[-1])
    for i in range(len(pts) - 1):
        if s <= cum[i + 1] + 1e-12 or i == len(pts) - 2:
            seg = cum[i + 1] - cum[i]
            t = 0.0 if seg < 1e-12 else (s - cum[i]) / seg
            a, b = pts[i], pts[i + 1]
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t), i
    return pts[-1], len(pts) - 2


def _sub_polyline(pts: Sequence[Pt], cum: Sequence[float], s0: float, s1: float) -> list[Pt]:
    p0, i0 = _point_at(pts, cum, s0)
    p1, i1 = _point_at(pts, cum, s1)
    out = [p0] + [pts[k] for k in range(i0 + 1, i1 + 1) if s0 < cum[k] < s1] + [p1]
    return _simplify(out) if len(_simplify(out)) >= 2 else [p0, p1]


def _fit_items(pts: Sequence[Pt], cum: Sequence[float], lengths: Sequence[float], fracs: Sequence[float],
               od: float, m_start: float, m_end: float, hits: Callable[[int, float, float], int],
               accept: Callable[[int, float, float], None], gap: float = 0.05
               ) -> Optional[list[tuple[float, int]]]:
    """Centre arc positions (and segment index) for items of ``lengths`` near ``fracs`` x total, in
    order, on straight segments (``m_start`` / ``m_end`` metres from the chain ends, clear of bends),
    preferring horizontal segments and, above all, positions where ``hits`` reports no clash."""
    total = cum[-1]
    segs = []
    for i in range(len(pts) - 1):
        s0, s1 = cum[i], cum[i + 1]
        if s1 - s0 < 1e-9:
            continue
        vertical = abs(pts[i + 1][2] - pts[i][2]) > 1e-9
        m0 = 0.0 if i == 0 else od / 2 + 0.02  # clear of the bend (the chain ends have no bend)
        m1 = 0.0 if i == len(pts) - 2 else od / 2 + 0.02
        segs.append((s0 + max(m0, m_start - s0), s1 - max(m1, m_end - (total - s1)), vertical, i))
    out: list[tuple[float, int]] = []
    lb = gap
    for k, (L, f) in enumerate(zip(lengths, fracs)):
        target = f * total
        rest = sum(lengths[k + 1:]) + gap * (len(lengths) - k)
        cands: list[tuple[float, float, int]] = []
        for a0, b0, vertical, i in segs:
            a = max(a0 + L / 2, lb + L / 2)
            b = min(b0 - L / 2, total - rest - L / 2 - gap)
            if a > b + 1e-9:
                continue
            vals = {min(max(target, a), b), a, b}
            n = min(16, int((b - a) / 0.1) + 1)
            if n > 1:
                vals.update(float(v) for v in np.linspace(a, b, n))
            for c in vals:
                cands.append((abs(c - target) + (1.5 if vertical else 0.0), c, i))
        if not cands:
            return None
        cands.sort()
        best: Optional[tuple[int, float, float, int]] = None
        for base, c, i in cands:
            h = hits(k, c, L)
            if best is None or (h, base) < (best[0], best[1]):
                best = (h, base, c, i)
            if h == 0:
                break
        assert best is not None
        out.append((best[2], best[3]))
        accept(k, best[2], L)
        lb = best[2] + L / 2 + gap
    return out


def _inline_size(obj: PIDObject, od: float) -> tuple[float, float, float]:
    base = TYPICAL_DIMENSIONS.get(obj.cls, TYPICAL_DIMENSIONS[S.UNKNOWN])
    if obj.category == Category.INSTRUMENT:
        return base
    return tuple(round(x, 3) for x in scale_for_od(base, od))  # type: ignore[return-value]


def _yaw_of(a: Pt, b: Pt) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return round(math.degrees(math.atan2(dy, dx)) % 360, 3)


def _make_inline(ctx: _Ctx, obj: PIDObject, size: tuple[float, float, float], p_in: Pt, p_out: Pt,
                 entry: str, exit_: Optional[str], reversed_: bool, note: str) -> Placement3D:
    cx, cy, cz = ((p_in[0] + p_out[0]) / 2, (p_in[1] + p_out[1]) / 2, (p_in[2] + p_out[2]) / 2)
    vertical = abs(p_out[2] - p_in[2]) > max(abs(p_out[0] - p_in[0]), abs(p_out[1] - p_in[1]))
    if vertical:
        z0 = min(p_in[2], p_out[2])
        yaw = 0.0
    else:
        z0 = cz - axis_height(obj.cls, size)
        yaw = _yaw_of(p_in, p_out)
        if reversed_:
            yaw = (yaw + 180.0) % 360
    nozzles = {entry: Vec3(x=round(p_in[0], 4), y=round(p_in[1], 4), z=round(p_in[2], 4))}
    if exit_:
        nozzles[exit_] = Vec3(x=round(p_out[0], 4), y=round(p_out[1], 4), z=round(p_out[2], 4))
    return Placement3D(object_id=obj.id, cls=obj.cls, position=Vec3(x=round(cx, 4), y=round(cy, 4), z=round(z0, 4)),
                       yaw_deg=yaw, size=Vec3(x=size[0], y=size[1], z=size[2]),
                       deck=ctx.room.deck_below_z(z0), nozzles=nozzles, rationale=f"[router] {note}")


def _clamp_to_room(ctx: _Ctx, obj: PIDObject, size: tuple[float, float, float],
                   make: Callable[[tuple[float, float, float]], Placement3D]) -> tuple[Placement3D, Box3, bool]:
    """Placement whose body fits the room: the cross-section (width, height incl. actuator) is
    reduced in 15 % steps while the body pokes out. Returns (placement, body box, clamped)."""
    rb = ctx.room.bounds
    pl = make(size)
    box = item_box(pl, obj)
    clamped = False
    for _ in range(8):
        if box.x0 >= rb.x0 - 1e-3 and box.y0 >= rb.y0 - 1e-3 and box.z0 >= rb.z0 - 1e-3 and \
                box.x1 <= rb.x1 + 1e-3 and box.y1 <= rb.y1 + 1e-3 and box.z1 <= rb.z1 + 1e-3:
            break
        size = (size[0], round(size[1] * 0.85, 3), round(size[2] * 0.85, 3))
        pl = make(size)
        box = item_box(pl, obj)
        clamped = True
    return pl, box, clamped


def _items_along(ctx: _Ctx, cc: _ChainCtx, pts: list[Pt], od: float, line: Optional[str], dead_end: bool,
                 m_start: float, m_end: float
                 ) -> tuple[list[Placement3D], list[tuple[float, float]], list[Box3], int]:
    """Place the chain's inline items along ``pts``; returns placements, their (s_in, s_out), bodies
    and the number of clashes left (see :func:`_body_hits`)."""
    chain = cc.chain
    objs = ctx.objs
    cum = _cum(pts)
    total = cum[-1]
    n = len(chain.items)
    if n == 0:
        return [], [], [], 0
    sizes = [_inline_size(objs[i], od) for i, _, _ in chain.items]
    drawn = [max(1e-3, ctx.pipes[p].length) for p in chain.pipes]
    acc, fracs = 0.0, []
    for k in range(n):
        acc += drawn[k]
        fracs.append(acc / sum(drawn) if not dead_end else (k + 1) / (n + 0.5))
    lengths = [s[0] for s in sizes]
    tip = dead_end and objs[chain.items[-1][0]].category == Category.INSTRUMENT
    if tip:
        lengths[-1] = 0.0
    note = f"inline on {line or 'unnumbered line'} (pipes {', '.join(chain.pipes)})"
    chosen: dict[int, Box3] = {}

    def build(k: int, c: float, L: float) -> tuple[Placement3D, Box3, float, float, bool]:
        oid, entry, exit_ = chain.items[k]
        obj = objs[oid]
        size = sizes[k]
        if tip and k == n - 1:
            t = pts[-1]
            prev = pts[-2] if len(pts) > 1 else pts[-1]

            def make_tip(sz: tuple[float, float, float]) -> Placement3D:
                return Placement3D(object_id=oid, cls=obj.cls,
                                   position=Vec3(x=round(t[0], 4), y=round(t[1], 4), z=round(t[2], 4)),
                                   yaw_deg=_yaw_of(prev, t), size=Vec3(x=sz[0], y=sz[1], z=sz[2]),
                                   deck=ctx.room.deck_below_z(t[2]),
                                   nozzles={entry: Vec3(x=round(t[0], 4), y=round(t[1], 4), z=round(t[2], 4))},
                                   rationale=f"[router] instrument on a stub of {line or 'its line'}")
            pl, box, clamped = _clamp_to_room(ctx, obj, size, make_tip)
            return pl, box, total, total, clamped
        if L < size[0]:
            size = (round(max(L, 0.02), 3), size[1], size[2])
        s_in, s_out = c - L / 2, c + L / 2
        p_in, _ = _point_at(pts, cum, s_in)
        p_out, _ = _point_at(pts, cum, s_out)
        rev = entry in ("out", "discharge") or (exit_ in ("in", "suction"))
        pl, box, clamped = _clamp_to_room(ctx, obj, size, lambda sz: _make_inline(ctx, obj, sz, p_in, p_out, entry,
                                                                                    exit_, rev, note))
        return pl, box, s_in, s_out, clamped

    def hits(k: int, c: float, L: float) -> int:
        if tip and k == n - 1:
            return 0
        _, box, _, _, _ = build(k, c, L)
        return _body_hits(ctx, box, cc.keys, frozenset(), [b for j, b in chosen.items() if j != k])

    def accept(k: int, c: float, L: float) -> None:
        chosen[k] = build(k, c, L)[1]

    fit = None
    for shrink in (1.0, 0.7, 0.45):
        ls = [L * shrink for L in lengths]
        for fr in (fracs, [(k + 1) / (n + 1) for k in range(n)]):
            chosen.clear()
            fit = _fit_items(pts, cum, ls, fr, od, m_start, m_end, hits, accept)
            if fit is not None:
                break
        if fit is not None:
            if shrink < 1.0:
                ctx.notes.append(f"chain {chain.pipes[0]}..{chain.pipes[-1]}: inline items shortened x{shrink} to fit")
            lengths = ls
            break
    if fit is None:  # sequential packing ignoring bends
        step = total / (n + 1)
        fit = [(step * (k + 1), 0) for k in range(n)]
        lengths = [min(L, step * 0.8) for L in lengths]
        ctx.notes.append(f"chain {chain.pipes[0]}..{chain.pipes[-1]}: no straight run long enough; "
                         f"inline items packed across bends")
    out: list[Placement3D] = []
    spans: list[tuple[float, float]] = []
    boxes: list[Box3] = []
    left = 0
    for k, ((c, _), L) in enumerate(zip(fit, lengths)):
        pl, box, s_in, s_out, clamped = build(k, c, L)
        if clamped:
            ctx.notes.append(f"{pl.object_id}: body reduced to fit inside the room")
        left += _body_hits(ctx, box, cc.keys, cc.nodes if (tip and k == n - 1) else frozenset(), boxes)
        out.append(pl)
        spans.append((s_in, s_out))
        boxes.append(box)
    return out, spans, boxes, left


def _item_radius(obj: PIDObject, size: tuple[float, float, float]) -> float:
    """Largest distance of an inline item's body from its pipe axis (flanges, stem, actuator)."""
    za = axis_height(obj.cls, size)
    return max(size[1] / 2, size[2] - za, za)


def _stub(ctx: _Ctx, cc: _ChainCtx, start: Pt, normal: Optional[Sequence[float]], length: float, od: float,
          tip: float) -> list[Pt]:
    """Dead-end stub (instrument taps, vents, drains): out along the nozzle normal, then up; from a
    node, along the free axis direction. Among the candidate directions the one whose run (plus
    the instrument at its tip) clashes least with equipment, bodies and other pipes is used."""
    L = max(0.3, length)
    ups = ((0, 0, 1), (0, 0, -1))
    flats = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))
    if normal is not None and abs(normal[2]) < 0.5:
        p1 = _add(start, normal, 0.3)
        cands = [[start, p1, _add(p1, d, L)] for d in ups] + [[start, _add(start, normal, 0.3 + L)]]
        cands += [[start, p1, _add(p1, d, L)] for d in flats if abs(sum(d[i] * normal[i] for i in range(3))) < 0.5]
        skip = od / 2 + 0.03  # the stub leaves the nozzle face: do not count touching its own equipment
    elif normal is not None:
        d0 = (0, 0, -1) if normal[2] < -0.5 else (0, 0, 1)
        p1 = _add(start, d0, 0.3)
        cands = [[start, _add(start, d0, L)]] + [[start, p1, _add(p1, d, L)] for d in flats]
        skip = od / 2 + 0.03
    else:  # junction / hub: leave along a free axis, clear of the node's other pipes
        cands = [[start, _add(start, d, L)] for d in ups[:1] + flats + ups[1:]]
        skip = ctx.terms[cc.chain.start.node].margin if cc.chain.start.node in ctx.terms else 0.3
    r = od / 2

    def score(pts: list[Pt]) -> int:
        h = _path_hits(ctx, pts, r, cc.keys, cc.nodes, skip)
        e = pts[-1]
        return h + _body_hits(ctx, Box3(e[0] - tip, e[1] - tip, e[2] - tip, e[0] + tip, e[1] + tip, e[2] + tip),
                              cc.keys, cc.nodes)

    best = min(range(len(cands)), key=lambda i: (score(cands[i]), i))
    return cands[best]


# ---------------------------------------------------------------------------- router


def route(doc: PIDDocument, placements: Sequence[Placement3D], room: EngineRoom,
          routing: Optional[RoutingPlan] = None, res: float = 0.5, time_budget: float = 120.0
          ) -> tuple[list[PipeRoute3D], list[Placement3D], list[str]]:
    """Route every process pipe; place inline items. Returns (routes, inline placements, notes)."""
    t0 = time.time()
    objs = doc.object_by_id()
    pmap = {p.object_id: p for p in placements}
    grid = VoxelGrid(room, res, [placement_box(p) for p in placements])
    sys_ = doc_system(doc)
    default_medium = SYSTEMS[sys_].medium if sys_ in SYSTEMS else Medium.UNKNOWN
    notes: list[str] = []
    stats = RouteStats()
    chains, hubs, adj, cnotes = _contract(doc, set(pmap))
    notes += cnotes
    stats.chains = len(chains)
    pipes = {p.id: p for p in doc.pipes}
    signal = sum(1 for p in doc.pipes if p.kind == PipeKind.SIGNAL)
    if signal:
        notes.append(f"{signal} signal line(s) are not piping and are not routed")
    ctx = _Ctx(doc, objs, room, grid, pipes, pmap, {}, routing or RoutingPlan(), default_medium, t0 + time_budget,
               stats, notes)
    eq = [placement_box(p) for p in placements]
    ctx.eq_lo = np.array([[b.x0, b.y0, b.z0] for b in eq], dtype=float).reshape(-1, 3)
    ctx.eq_hi = np.array([[b.x1, b.y1, b.z1] for b in eq], dtype=float).reshape(-1, 3)

    # --- positions of junctions, hubs and free ends
    mapping = _drawing_map(doc, pmap, room)
    jpos = {j.id: j.position for j in doc.junctions}
    nodes: dict[str, Pt] = {}
    node_z: dict[str, float] = {}
    node_od: dict[str, float] = {}
    for n, lst in adj.items():
        if n in pmap:
            continue
        if n in jpos:
            px, py = jpos[n].x, jpos[n].y
        elif n in objs:
            px, py = objs[n].center.x, objs[n].center.y
        else:
            pid, which = n.split(":")[1], int(n.split(":")[2])
            pts = pipes[pid].points
            pp = pts[0] if which == 0 else pts[-1]
            px, py = pp.x, pp.y
        if not (n in jpos or n in hubs or n.startswith("free:")):
            continue
        x, y = mapping(px, py)
        big = max((p for p, _ in lst), key=lambda p: _pipe_od(p))
        node_z[n] = band_z(room, routing, pipe_medium(big, default_medium), big.line_number)
        node_od[n] = _pipe_od(big)
        nodes[n] = (x, y, node_z[n])
    # blend with the positions of the nodes each one connects to (chain neighbours)
    nb: dict[str, list[str]] = defaultdict(list)
    for ch in chains:
        if ch.end is not None:
            nb[ch.start.node].append(ch.end.node)
            nb[ch.end.node].append(ch.start.node)
    base = dict(nodes)
    for _ in range(3):
        new = {}
        for n, (x, y, z) in nodes.items():
            pts = []
            for m in nb.get(n, []):
                if m in pmap:
                    b = placement_box(pmap[m])
                    pts.append((b.center[0], b.center[1], b.z0 + 0.5 * (b.z1 - b.z0)))
                elif m in nodes:
                    pts.append(nodes[m])
            if pts:
                mx, my = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
                x, y = 0.5 * base[n][0] + 0.5 * mx, 0.5 * base[n][1] + 0.5 * my
                z = adapt_band(node_z[n], min(p[2] for p in pts), max(p[2] for p in pts))
            new[n] = (x, y, z)
        nodes = new

    nodes = _spread_nodes(ctx, chains, nodes, pmap)

    # --- terminals: nodes (hub bodies registered as obstacles), then nozzle approaches
    hub_size: dict[str, tuple[float, float, float]] = {}
    hub_pl: dict[str, Placement3D] = {}
    for n in sorted(nodes):
        inflate, margin = 0.25, 0.0
        if n in hubs:
            hub_size[n] = _inline_size(objs[n], node_od[n])
            inflate = max(0.25, max(hub_size[n]) / 2 + 0.05)
            margin = max(hub_size[n]) / 2 + 0.05
        elif not n.startswith("free:"):
            margin = node_od[n] / 2 + PIPE_CLEARANCE + 0.05
        ctx.terms[n] = _node_term(ctx, n, nodes[n], inflate, margin)
        nodes[n] = ctx.terms[n].point
        if n in hubs:
            o = objs[n]
            x, y, z = nodes[n]

            def make_hub(sz: tuple[float, float, float], o: PIDObject = o, x: float = x, y: float = y,
                         z: float = z) -> Placement3D:
                return Placement3D(object_id=o.id, cls=o.cls, position=Vec3(x=x, y=y, z=round(z - axis_height(o.cls, sz), 4)),
                                   size=Vec3(x=sz[0], y=sz[1], z=sz[2]), deck=room.deck_below_z(z),
                                   rationale="[router] branch item placed at its junction position")
            pl, box, clamped = _clamp_to_room(ctx, o, hub_size[n], make_hub)
            if clamped:
                notes.append(f"{n}: body reduced to fit inside the room")
            hub_pl[n] = pl
            grid.items.add(_box_arr(box)[0], _box_arr(box)[1], 0.0, (n, frozenset({n})))
    ctx.node_pos = nodes
    term_od: dict[str, float] = defaultdict(float)
    for ch in chains:
        od = _chain_medium_line(ctx, ch)[2]
        for e in (ch.start, ch.end):
            if e is not None and e.node in pmap:
                k = _term_key(e, pmap)
                term_od[k] = max(term_od[k], od)
    for k in sorted(term_od):
        oid, nz_key = k.split("\x00", 1)
        ctx.terms[k] = _nozzle_term(ctx, k, pmap[oid], nz_key, term_od[k])

    # --- route chains: priority lines, then largest OD first; stubs last
    prio = {ln.strip().upper(): i for i, ln in enumerate(ctx.routing.priority)}

    def chain_key(ch: _Chain):
        medium, line, od = _chain_medium_line(ctx, ch)
        lines = [pipes[p].line_number.strip().upper() for p in ch.pipes if pipes[p].line_number]
        pr = min((prio[ln] for ln in lines if ln in prio), default=len(prio))
        return (ch.end is None, pr, -od, -len(ch.pipes), ch.pipes[0])

    routes: dict[str, PipeRoute3D] = {}
    inline: list[Placement3D] = []
    for idx, ch in enumerate(sorted(chains, key=chain_key)):
        medium, line, od = _chain_medium_line(ctx, ch)
        zb = band_z(room, ctx.routing, medium, line)
        ta = ctx.terms[_term_key(ch.start, pmap)]
        tb = ctx.terms[_term_key(ch.end, pmap)] if ch.end is not None else None
        keys = frozenset({ta.key} | ({tb.key} if tb else set()))
        own = [c for t in ([ta] + ([tb] if tb else [])) for c in t.cells + t.ring]
        cc = _ChainCtx(idx, ch, frozenset({ch.start.node} | ({ch.end.node} if ch.end else set())), keys, own,
                       [ta] + ([tb] if tb else []))
        label = f"chain {', '.join(ch.pipes)}"
        a = ta.point
        mode = "clean"
        if tb is None:
            stats.stubs += 1
            sizes = [_inline_size(objs[i], od) for i, _, _ in ch.items]
            tip = max(sizes[-1]) / 2 if sizes and objs[ch.items[-1][0]].category == Category.INSTRUMENT else 0.0
            pts = _stub(ctx, cc, a, ta.normal, 0.3 + sum(s[0] for s in sizes) + 0.1 * len(sizes), od, tip)
        else:
            zb = adapt_band(zb, a[2], tb.point[2])
            if ta.key == tb.key:
                pts = [a, _add(a, (0, 0, 1), 0.5)]
            else:
                pts, mode = _route_between(ctx, cc, ta, tb, od, zb, label)
        if len(pts) < 2:
            pts = [pts[0], _add(pts[0], (0, 0, 1), 0.05)]
        m_end = tb.margin if tb else 0.0
        n_notes = len(notes)
        placed_items, spans, bodies, left = _items_along(ctx, cc, pts, od, line, tb is None, ta.margin, m_end)
        if left and tb is not None and ta.key != tb.key and time.time() < ctx.deadline:
            # rip up and retry: keep the items' bodies clear of everything, or give them a straight leg
            sizes = [_inline_size(objs[i], od) for i, _, _ in ch.items]
            r_item = max(_item_radius(objs[i], sz) for (i, _, _), sz in zip(ch.items, sizes))
            need = sum(sz[0] for sz in sizes) + 0.1 * len(sizes) + od + 0.1
            first_notes = notes[n_notes:]
            del notes[n_notes:]
            saved = (stats.astar, stats.widened)
            attempts: list[tuple[str, Callable[[], tuple[list[Pt], str]]]] = [
                (f"keeping {r_item:.2f} m around the pipe free",
                 lambda: _route_between(ctx, cc, ta, tb, max(od, 2 * r_item + 0.05), zb, label, MODES[:2]))]
            for at_start in sorted((True, False), key=lambda st: (ta if st else tb).normal is None):
                t_leg = ta if at_start else tb
                attempts.append((f"with a straight {need + t_leg.margin:.1f} m run out of {t_leg.key.split(chr(0))[0]}",
                                 lambda st=at_start, tl=t_leg: _route_with_leg(ctx, cc, ta, tb, od, zb,
                                                                               need + tl.margin, st)))
            best = None
            for how, fn in attempts:
                if time.time() >= ctx.deadline:
                    break
                pts2, mode2 = fn()
                res2 = _items_along(ctx, cc, pts2, od, line, False, ta.margin, m_end) if len(pts2) >= 2 else None
                extra = notes[n_notes:]
                del notes[n_notes:]
                if res2 is not None and res2[3] < (best[0] if best else left):
                    best = (res2[3], how, pts2, mode2, res2, extra)
                    if res2[3] == 0:
                        break
            stats.astar, stats.widened = saved  # still one search per chain in the statistics
            if best is not None:
                stats.retried += 1
                if mode == "relaxed":
                    stats.relaxed -= 1
                before = left
                n_left, how, pts, mode, (placed_items, spans, bodies, left), extra = best
                notes += extra + [f"{label}: re-routed {how} for its inline items (item clashes {before} -> {n_left})"]
            else:
                notes += first_notes
        if left:
            stats.item_clash_chains += 1
        inline.extend(placed_items)
        cum = _cum(pts)
        cuts = [0.0]
        for s_in, s_out in spans:
            cuts += [s_in, s_out]
        cuts.append(cum[-1])
        ctx.chain_keys.append(cc.keys)
        for k, pid in enumerate(ch.pipes):
            s0, s1 = cuts[2 * k], cuts[2 * k + 1]
            if s1 < s0:
                s1 = s0
            sub = _sub_polyline(pts, cum, s0, s1)
            if ch.rev[k]:
                sub = sub[::-1]
            p = pipes[pid]
            r = PipeRoute3D(pipe_id=pid, points=[Vec3(x=round(x, 4), y=round(y, 4), z=round(z, 4)) for x, y, z in sub],
                            diameter=round(_pipe_od(p), 4), medium=pipe_medium(p, default_medium),
                            line_number=p.line_number)
            routes[pid] = r
            grid.add_route([v.as_tuple() for v in r.points], r.diameter / 2, idx)
        for pl, box in zip(placed_items, bodies):
            grid.items.add(_box_arr(box)[0], _box_arr(box)[1], 0.0, (pl.object_id, frozenset()))

    # --- hubs (three-way valves etc.) sit at their node position; nozzles at the route ends
    for h in sorted(hubs):
        pl = hub_pl.get(h)
        if pl is None:
            continue
        for p, w in adj.get(h, []):
            r = routes.get(p.id)
            if r is not None:
                end = r.points[0] if w == 0 else r.points[-1]
                pl.nozzles[port_key(p.start if w == 0 else p.end, p.id)] = end
        inline.append(pl)

    # --- objects without process piping
    placed_ids = set(pmap) | {p.object_id for p in inline}
    for o in doc.objects:
        if o.id in placed_ids:
            continue
        pl = _place_isolated(ctx, o, {p.object_id: p for p in inline}, mapping)
        inline.append(pl)
        box = item_box(pl, o)
        grid.items.add(_box_arr(box)[0], _box_arr(box)[1], 0.0, (o.id, frozenset()))

    # --- audit
    ordered = [routes[p.id] for p in doc.pipes if p.id in routes]
    audit = audit_layout(doc, ordered, list(placements), inline, room)
    stats.audit = audit
    stats.collisions = audit["segments_through_equipment"]
    stats.length_m = audit["pipe_length_m"]
    stats.seconds = time.time() - t0
    notes.append(f"router: {stats.chains} chains ({stats.stubs} stubs), A* {stats.astar} "
                 f"(widened {stats.widened}, soft obstacles {stats.relaxed}, re-routed for inline items "
                 f"{stats.retried}), Manhattan fallback {stats.fallback}, "
                 f"segments through equipment {audit['segments_through_equipment']}, "
                 f"pipe/inline-item clashes {audit['pipe_item_clashes']}, "
                 f"pipe-pipe clashes {audit['pipe_pipe_clashes']} (pipes sharing no node), "
                 f"inline item overlaps {audit['item_overlaps']}, items outside the room {audit['items_outside_room']}, "
                 f"total pipe length {stats.length_m:.1f} m, grid {res} m, {stats.seconds:.2f} s")
    for kind in ("pipe_item", "pipe_pipe", "item_overlap"):
        ex = audit["examples"].get(kind)
        if ex:
            notes.append(f"router audit: {kind.replace('_', '/')} clashes e.g. " + "; ".join(ex[:5]))
    return ordered, inline, notes


def _chain_need(ctx: _Ctx, ch: _Chain) -> float:
    """Straight run a chain's inline items need (lengths, gaps, bend margins, junction margins)."""
    od = _chain_medium_line(ctx, ch)[2]
    items = [_inline_size(ctx.objs[i], od)[0] for i, _, _ in ch.items]
    return sum(items) + 0.1 * len(items) + od + 0.1 + 2 * (od / 2 + PIPE_CLEARANCE + 0.05)


def _spread_nodes(ctx: _Ctx, chains: Sequence[_Chain], nodes: dict[str, Pt], pmap: dict[str, Placement3D]
                  ) -> dict[str, Pt]:
    """Relax junction / hub / free-end positions before they are snapped: a node is pushed away
    (horizontally) from the other end of each chain whose inline items need a longer run than the
    Manhattan distance between its ends, and junctions closer than two grid cells are pushed
    apart. Otherwise junctions of parallel trains end up in neighbouring cells with no room for
    the valves between them."""
    room, res = ctx.room, ctx.grid.res
    pos = {n: list(p) for n, p in nodes.items()}
    pairs: list[tuple[str, str, Optional[Pt], float]] = []  # (node, other node, fixed point of the other, need)
    for ch in chains:
        if ch.end is None or not ch.items:
            continue
        need = _chain_need(ctx, ch)
        a, b = ch.start, ch.end
        for x, y in ((a, b), (b, a)):
            if x.node not in pos:
                continue
            if y.node in pos:
                pairs.append((x.node, y.node, None, need))
            elif y.node in pmap and y.key in pmap[y.node].nozzles:
                pairs.append((x.node, y.node, pmap[y.node].nozzles[y.key].as_tuple(), need / 2 if False else need))
    lo = (0.5, -room.breadth / 2 + 0.5)
    hi = (room.length - 0.5, room.breadth / 2 - 0.5)
    names = sorted(pos)
    min_gap = 2 * res
    for _ in range(12):
        moved = False
        delta: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for n, m, fixed, need in pairs:
            pa = pos[n]
            pb = fixed if fixed is not None else pos[m]
            d = sum(abs(pa[i] - pb[i]) for i in range(3))
            if d >= need - 1e-6:
                continue
            ax = 0 if abs(pa[0] - pb[0]) >= abs(pa[1] - pb[1]) else 1
            sgn = 1.0 if pa[ax] >= pb[ax] else -1.0
            share = 1.0 if fixed is not None else 0.5  # both ends movable: each moves half (once per side)
            delta[n][ax] += sgn * (need - d) * share / (1 if fixed is not None else 1)
        for i, n in enumerate(names):
            for m in names[i + 1:]:
                pa, pb = pos[n], pos[m]
                if max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1]), abs(pa[2] - pb[2])) >= min_gap:
                    continue
                if abs(pa[2] - pb[2]) >= min_gap:
                    continue
                ax = 0 if abs(pa[0] - pb[0]) >= abs(pa[1] - pb[1]) else 1
                gap = abs(pa[ax] - pb[ax])
                sgn = 1.0 if (pa[ax] > pb[ax] or (pa[ax] == pb[ax] and n < m)) else -1.0
                push = (min_gap - gap) / 2
                delta[n][ax] += sgn * push
                delta[m][ax] -= sgn * push
        for n, (dx, dy) in delta.items():
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
            step = 0.5  # damped
            nx = min(max(pos[n][0] + step * dx, lo[0]), hi[0])
            ny = min(max(pos[n][1] + step * dy, lo[1]), hi[1])
            if abs(nx - pos[n][0]) > 1e-4 or abs(ny - pos[n][1]) > 1e-4:
                pos[n][0], pos[n][1] = nx, ny
                moved = True
        if not moved:
            break
    return {n: (p[0], p[1], p[2]) for n, p in pos.items()}


def _place_isolated(ctx: _Ctx, o: PIDObject, inline: dict[str, Placement3D], mapping) -> Placement3D:
    """Object with no process connection: near a signal-linked object, else at its mapped drawing
    position on the floor; the first nearby spot clear of equipment, bodies and pipes is used."""
    doc, room, pmap = ctx.doc, ctx.room, ctx.placements
    anchor = None
    for p in doc.pipes:
        ids = {p.start.object_id, p.end.object_id}
        if o.id in ids:
            other = (ids - {o.id}).pop() if len(ids) > 1 else None
            if other and (other in pmap or other in inline):
                anchor = pmap.get(other) or inline.get(other)
                break
    size = TYPICAL_DIMENSIONS.get(o.cls, TYPICAL_DIMENSIONS[S.UNKNOWN])
    if anchor is not None:
        b = item_box(anchor, ctx.objs.get(anchor.object_id))
        target = (b.center[0], b.y1 + 0.3 + size[1] / 2, b.z1 + 0.2)
        why = f"no process piping; placed next to {anchor.object_id} it is signal-linked to"
    else:
        x, y = mapping(o.center.x, o.center.y)
        target = (x, y, room.deck_z("floor"))
        why = "no piping; placed at its drawing position on the floor deck"
    target = (target[0], target[1], min(target[2], room.height - size[2]))

    def make(c: Pt) -> Placement3D:
        return Placement3D(object_id=o.id, cls=o.cls, position=Vec3(x=round(c[0], 4), y=round(c[1], 4),
                                                                    z=round(c[2], 4)),
                           size=Vec3(x=size[0], y=size[1], z=size[2]), deck=room.deck_below_z(c[2]),
                           rationale=f"[router] {why}")

    best = None
    for r in np.arange(0.0, 8.0, 0.5):
        for ang in range(0, 360, 45 if r > 0 else 360):
            cx = min(max(target[0] + r * math.cos(math.radians(ang)), size[0] / 2), room.length - size[0] / 2)
            cy = min(max(target[1] + r * math.sin(math.radians(ang)), -room.breadth / 2 + size[1] / 2),
                     room.breadth / 2 - size[1] / 2)
            pl = make((cx, cy, target[2]))
            if _body_hits(ctx, item_box(pl, o).expand(0.1)) == 0:
                best = pl
                break
        if best is not None:
            break
    return best if best is not None else make(target)


# ---------------------------------------------------------------------------- audit


def audit_layout(doc: PIDDocument, routes: Sequence[PipeRoute3D], equipment: Sequence[Placement3D],
                 inline: Sequence[Placement3D], room: EngineRoom) -> dict[str, Any]:
    """Clash counts of a routed layout (bodies = rendered mesh bounds):

    * ``segments_through_equipment``: route segments through a solver equipment box;
    * ``pipe_item_clashes``: (pipe, inline / hub / isolated item) pairs whose pipe cylinder box
      overlaps the item body, the pipe's own end objects excepted;
    * ``pipe_pipe_clashes``: pairs of pipes sharing no node whose centre-lines come closer than
      the sum of their radii (``coincident_m``: length of identical centre-lines among them);
    * ``item_overlaps``: inline item vs inline item / equipment body overlaps, pairs joined by a pipe
      excepted; ``items_outside_room``.
    """
    objs = doc.object_by_id()
    pipes = {p.id: p for p in doc.pipes}
    eq_boxes = [(p.object_id, placement_box(p)) for p in equipment]
    it_boxes = [(p.object_id, item_box(p, objs.get(p.object_id))) for p in inline]
    linked: set[frozenset[str]] = set()
    for p in doc.pipes:
        a, b = p.start.object_id, p.end.object_id
        if a and b:
            linked.add(frozenset((a, b)))
    ex: dict[str, list[str]] = defaultdict(list)
    seg_eq = 0
    length = 0.0
    S_lo, S_hi, S_r, S_pipe = [], [], [], []
    for k, r in enumerate(routes):
        pts = [v.as_tuple() for v in r.points]
        for a, b in zip(pts, pts[1:]):
            length += _dist(a, b)
            seg_eq += sum(1 for _, bx in eq_boxes if segment_hits_box(a, b, bx))
            S_lo.append(np.minimum(a, b))
            S_hi.append(np.maximum(a, b))
            S_r.append(r.diameter / 2)
            S_pipe.append(k)
    S_lo_a = np.array(S_lo, dtype=float).reshape(-1, 3)
    S_hi_a = np.array(S_hi, dtype=float).reshape(-1, 3)
    S_r_a = np.array(S_r, dtype=float)
    S_pipe_a = np.array(S_pipe, dtype=int)
    # pipe vs inline bodies
    pipe_item: set[tuple[int, str]] = set()
    for oid, bx in it_boxes:
        lo, hi = _box_arr(bx)
        hit = box_overlaps(lo, hi, S_lo_a - S_r_a[:, None], S_hi_a + S_r_a[:, None])
        for i in np.nonzero(hit)[0]:
            k = int(S_pipe_a[i])
            p = pipes.get(routes[k].pipe_id)
            if p is not None and oid in (p.start.object_id, p.end.object_id):
                continue
            if (k, oid) not in pipe_item:
                pipe_item.add((k, oid))
                ex["pipe_item"].append(f"{routes[k].pipe_id} through {oid}")
    # pipe vs pipe (pipes sharing no node); box distance is exact for axis-aligned segments
    nodes = [({pipes[r.pipe_id].start.node_id, pipes[r.pipe_id].end.node_id} - {None}) if r.pipe_id in pipes
             else set() for r in routes]
    pairs: set[tuple[int, int]] = set()
    coincident = 0.0
    n = len(S_r_a)
    for i0 in range(0, n, 256):
        sl = slice(i0, min(n, i0 + 256))
        gap = np.maximum(0.0, np.maximum(S_lo_a[None, :, :] - S_hi_a[sl, None, :],
                                         S_lo_a[sl, None, :] - S_hi_a[None, :, :]))
        d = np.sqrt((gap ** 2).sum(axis=2))
        lim = S_r_a[sl, None] + S_r_a[None, :] - 1e-3
        for i, j in zip(*np.nonzero(d < lim)):
            a, b = int(S_pipe_a[i0 + i]), int(S_pipe_a[j])
            if a >= b or nodes[a] & nodes[b]:
                continue
            if (a, b) not in pairs:
                pairs.add((a, b))
                ex["pipe_pipe"].append(f"{routes[a].pipe_id} x {routes[b].pipe_id}")
            if d[i, j] < 1e-6:
                ov = np.minimum(S_hi_a[i0 + i], S_hi_a[j]) - np.maximum(S_lo_a[i0 + i], S_lo_a[j])
                coincident += float(max(0.0, ov.max())) if (ov > -1e-9).all() else 0.0
    # inline bodies vs inline bodies / equipment
    overlaps = 0
    rb = room.bounds
    outside = 0
    all_boxes = eq_boxes + it_boxes
    n_eq = len(eq_boxes)
    A_lo = np.array([[b.x0, b.y0, b.z0] for _, b in all_boxes], dtype=float).reshape(-1, 3)
    A_hi = np.array([[b.x1, b.y1, b.z1] for _, b in all_boxes], dtype=float).reshape(-1, 3)
    for i, (oid, bx) in enumerate(it_boxes):
        if bx.x0 < rb.x0 - 1e-3 or bx.y0 < rb.y0 - 1e-3 or bx.z0 < rb.z0 - 1e-3 or bx.x1 > rb.x1 + 1e-3 or \
                bx.y1 > rb.y1 + 1e-3 or bx.z1 > rb.z1 + 1e-3:
            outside += 1
            ex["outside"].append(oid)
        lo, hi = _box_arr(bx)
        for j in np.nonzero(box_overlaps(lo, hi, A_lo[:n_eq + i], A_hi[:n_eq + i]))[0]:
            other = all_boxes[j][0]
            if other == oid or frozenset((oid, other)) in linked:
                continue
            overlaps += 1
            ex["item_overlap"].append(f"{oid} / {other}")
    return {"segments_through_equipment": seg_eq, "pipe_item_clashes": len(pipe_item),
            "pipe_pipe_clashes": len(pairs), "coincident_m": round(coincident, 2), "item_overlaps": overlaps,
            "items_outside_room": outside, "pipe_length_m": length, "examples": dict(ex)}
