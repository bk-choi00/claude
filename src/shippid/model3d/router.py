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
   snapped to a free voxel.
3. Each chain is routed with orthogonal A* on a voxel grid (0.25-0.5 m): bend penalty, a cost
   for running away from the preferred elevation band of the medium / line, obstacles =
   equipment boxes inflated by OD/2 + clearance, routed cells become obstacles, search bounded
   to a box around the endpoints (widened once), largest pipe size first (plan priority lines
   before everything), global time budget. Failures fall back to a Manhattan path through the
   elevation band, flagged in the notes.
4. Inline items are placed along the routed polyline in drawing order at positions proportional
   to the drawn pipe lengths, on straight (preferably horizontal) segments, yaw following the
   segment; the polyline is then split back into one :class:`PipeRoute3D` per original pipe id.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..catalog import SYSTEMS, TYPICAL_DIMENSIONS, pipe_od_m
from ..model import (Category, Medium, PIDDocument, PIDObject, PipeKind, PipeRoute3D, PipeRun, Placement3D,
                     SymbolClass, Vec3)
from .heuristic import DEFAULT_ELEVATIONS, doc_system, normalized_position, pipe_medium
from .plan import RoutingPlan
from .solver import axis_height, nozzle_normal, placement_box, port_key, scale_for_od
from .zones import Box3, EngineRoom

S = SymbolClass
Pt = tuple[float, float, float]

PIPE_CLEARANCE = 0.1  # m between pipe wall and equipment
_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


# ---------------------------------------------------------------------------- small geometry


def _add(a: Pt, b: Pt, s: float = 1.0) -> Pt:
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


# ---------------------------------------------------------------------------- voxel grid


class VoxelGrid:
    """Occupancy grid over the room: equipment obstacles (per inflation), routed pipes, reserved cells."""

    def __init__(self, room: EngineRoom, res: float, boxes: Sequence[Box3]):
        self.room = room
        self.res = res
        b = room.bounds
        self.origin = (b.x0, b.y0, b.z0)
        self.shape = (max(1, int(math.ceil((b.x1 - b.x0) / res))), max(1, int(math.ceil((b.y1 - b.y0) / res))),
                      max(1, int(math.ceil((b.z1 - b.z0) / res))))
        self.boxes = list(boxes)
        self._blocked: dict[float, np.ndarray] = {}
        self.pipes = np.zeros(self.shape, dtype=bool)
        self.reserved = np.zeros(self.shape, dtype=bool)

    def center(self, c: tuple[int, int, int]) -> Pt:
        return tuple(self.origin[i] + (c[i] + 0.5) * self.res for i in range(3))  # type: ignore[return-value]

    def cell(self, p: Pt) -> tuple[int, int, int]:
        return tuple(min(self.shape[i] - 1, max(0, int(math.floor((p[i] - self.origin[i]) / self.res))))
                     for i in range(3))  # type: ignore[return-value]

    def blocked(self, inflate: float) -> np.ndarray:
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

    def free(self, c: tuple[int, int, int], inflate: float) -> bool:
        return not (self.blocked(inflate)[c] or self.pipes[c] or self.reserved[c])

    def nearest_free(self, c: tuple[int, int, int], inflate: float, radius: int = 6,
                     prefer: Optional[tuple[float, float, float]] = None, hard_only: bool = False
                     ) -> tuple[int, int, int]:
        """Nearest free cell to ``c`` (shell search, ties broken toward ``prefer``); with ``hard_only``
        only equipment obstacles count (pipes / reserved cells are soft for the router)."""
        blocked = self.blocked(inflate)
        if not blocked[c] and (hard_only or not (self.pipes[c] or self.reserved[c])):
            return c
        best: Optional[tuple[float, tuple[int, int, int]]] = None
        for r in range(1, radius + 1):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    for dk in range(-r, r + 1):
                        if max(abs(di), abs(dj), abs(dk)) != r:
                            continue
                        q = (c[0] + di, c[1] + dj, c[2] + dk)
                        if not all(0 <= q[i] < self.shape[i] for i in range(3)):
                            continue
                        if blocked[q] or (not hard_only and (self.pipes[q] or self.reserved[q])):
                            continue
                        d = abs(di) + abs(dj) + abs(dk)
                        if prefer is not None:
                            d -= 0.4 * (di * prefer[0] + dj * prefer[1] + dk * prefer[2])
                        if best is None or (d, q) < best:
                            best = (d, q)
            if best is not None:
                return best[1]
        return c


def astar(cost: np.ndarray, start: tuple[int, int, int], goal: tuple[int, int, int], band_k: float,
          start_dir: int = 6, bend: float = 2.0, band_w: float = 0.35, eps: float = 1.25,
          max_expansions: int = 120_000) -> Optional[list[tuple[int, int, int]]]:
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
    cells: list[tuple[int, int, int]] = []
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
    fallback: int = 0
    stubs: int = 0
    seconds: float = 0.0
    collisions: int = 0
    length_m: float = 0.0


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


# ---------------------------------------------------------------------------- router


@dataclass
class _Ctx:
    doc: PIDDocument
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
    reserved_by: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict)


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


def _endpoint(ctx: _Ctx, end: _End) -> tuple[Pt, Optional[tuple[float, float, float]]]:
    """World point and outward normal (None for junctions / hubs / free ends)."""
    pl = ctx.placements.get(end.node)
    if pl is not None and end.key is not None:
        nz = pl.nozzles.get(end.key)
        if nz is None:  # unexpected port: top centre of the box
            b = placement_box(pl)
            nz = Vec3(x=b.center[0], y=b.center[1], z=b.z1)
            pl.nozzles[end.key] = nz
        return nz.as_tuple(), nozzle_normal(pl, nz)
    return ctx.node_pos[end.node], None


def _sub_box(grid: VoxelGrid, a: tuple[int, int, int], b: tuple[int, int, int], band_k: int, margin_xy: int,
             margin_z: int) -> tuple[slice, slice, slice]:
    lo = [min(a[i], b[i]) for i in range(3)]
    hi = [max(a[i], b[i]) for i in range(3)]
    lo[2], hi[2] = min(lo[2], band_k), max(hi[2], band_k)
    m = (margin_xy, margin_xy, margin_z)
    return tuple(slice(max(0, lo[i] - m[i]), min(grid.shape[i], hi[i] + m[i] + 1)) for i in range(3))  # type: ignore


SOFT_PIPE = 6.0  # cost of entering a cell already used by another pipe or reserved for another terminal
SOFT_NEAR = 0.6  # cost of running close to equipment (outside the hard inflation)


def _search(ctx: _Ctx, ca: tuple[int, int, int], cb: tuple[int, int, int], inflate: float, zb: float,
            start_dir: int, wide: bool, own: Sequence[tuple[int, int, int]] = (), strict: bool = False
            ) -> Optional[tuple[list[Pt], list[tuple[int, int, int]]]]:
    """A* between two cells in a sub-box (or the whole grid); returns (points, global cells).

    Hard: equipment boxes inflated by OD/2 + clearance. Soft: other pipes and other terminals'
    reserved cells (SOFT_PIPE), cells close to equipment (SOFT_NEAR). The chain's own reserved
    cells are free and hard cells next to its terminals are softened, so a terminal squeezed
    between equipment or surrounded by earlier pipes never becomes an unreachable pocket.
    """
    g = ctx.grid
    band_k = int(round((zb - g.origin[2]) / g.res - 0.5))
    band_k = min(max(band_k, 0), g.shape[2] - 1)
    span = max(abs(ca[i] - cb[i]) for i in range(3))
    if wide:
        sl = (slice(0, g.shape[0]), slice(0, g.shape[1]), slice(0, g.shape[2]))
    else:
        sl = _sub_box(g, ca, cb, band_k, max(8, int(0.3 * span)), 4)
    hard = g.blocked(inflate)[sl]
    near = g.blocked(inflate + 0.3)[sl]
    cost = np.where(near, SOFT_NEAR, 0.0)
    cost[g.pipes[sl] | g.reserved[sl]] = SOFT_PIPE
    off = (sl[0].start, sl[1].start, sl[2].start)
    shape = cost.shape

    def local(c):
        q = (c[0] - off[0], c[1] - off[1], c[2] - off[2])
        return q if all(0 <= q[i] < shape[i] for i in range(3)) else None

    for c in own:
        q = local(c)
        if q is not None:
            cost[q] = 0.0
    cost[hard] = -1.0
    la, lb = local(ca), local(cb)
    for t in (la, lb):
        cost[t] = 0.0
        if strict:
            continue
        for di, dj, dk in _DIRS:
            q = (t[0] + di, t[1] + dj, t[2] + dk)
            if all(0 <= q[i] < shape[i] for i in range(3)) and cost[q] < 0:
                cost[q] = SOFT_PIPE
    cells = astar(cost, la, lb, band_k - off[2], start_dir=start_dir,
                  max_expansions=300_000 if wide else 100_000)
    if cells is None:
        return None
    glob = [(c[0] + off[0], c[1] + off[1], c[2] + off[2]) for c in cells]
    return [g.center(c) for c in glob], glob


def _dir_index(n: Optional[tuple[float, float, float]]) -> int:
    if n is None:
        return 6
    ax = max(range(3), key=lambda i: abs(n[i]))
    return ax * 2 + (0 if n[ax] > 0 else 1)


def _join(ctx: _Ctx, p: Pt, q: Pt, own: Optional[Box3]) -> list[Pt]:
    """Orthogonal connection p -> q choosing the axis order that avoids equipment best."""
    if _dist(p, q) < 1e-9:
        return [q]
    best, best_hits = None, None
    for order in ((2, 0, 1), (0, 1, 2), (1, 0, 2), (0, 2, 1), (1, 2, 0), (2, 1, 0)):
        pts = [p]
        cur = list(p)
        for ax in order:
            if abs(q[ax] - cur[ax]) > 1e-9:
                cur[ax] = q[ax]
                pts.append(tuple(cur))  # type: ignore[arg-type]
        hits = sum(1 for a, b in zip(pts, pts[1:]) for bx in ctx.grid.boxes if segment_hits_box(a, b, bx))
        if best_hits is None or hits < best_hits:
            best, best_hits = pts, hits
            if hits == 0:
                break
    return best[1:]  # type: ignore[index]


def _compose(ctx: _Ctx, a: Pt, na, sa: Pt, mid: list[Pt], sb: Pt, b: Pt, nb) -> list[Pt]:
    pts: list[Pt] = [a]
    if na is not None:
        pts.append(sa)
    pts += _join(ctx, pts[-1], mid[0], None)
    pts += mid[1:]
    tail = [sb, b] if nb is not None else [b]
    pts += _join(ctx, pts[-1], tail[0], None)
    pts += tail[1:]
    return _simplify(_orthogonalize(pts))


def _clashes(ctx: _Ctx, pts: Sequence[Pt]) -> int:
    return sum(1 for p, q in zip(pts, pts[1:]) for bx in ctx.grid.boxes if segment_hits_box(p, q, bx))


def _route_between(ctx: _Ctx, a: Pt, na, b: Pt, nb, od: float, zb: float, label: str,
                   own: Sequence[tuple[int, int, int]] = ()) -> tuple[list[Pt], bool]:
    """Polyline a -> b (orthogonal). Returns (points, used_fallback)."""
    g = ctx.grid
    inflate = od / 2 + PIPE_CLEARANCE
    stub = max(0.2, od / 2 + 0.15)
    sa = _add(a, na, stub) if na is not None else a
    sb = _add(b, nb, stub) if nb is not None else b
    ca = g.nearest_free(g.cell(sa), inflate, prefer=na, hard_only=True)
    cb = g.nearest_free(g.cell(sb), inflate, prefer=nb, hard_only=True)
    if ca == cb:
        return _compose(ctx, a, na, sa, [g.center(ca)], sb, b, nb), False
    res = None
    if time.time() < ctx.deadline:
        res = _search(ctx, ca, cb, inflate, zb, _dir_index(na), wide=False, own=own)
        if res is None and time.time() < ctx.deadline:
            ctx.stats.widened += 1
            res = _search(ctx, ca, cb, inflate, zb, _dir_index(na), wide=True, own=own)
    if res is not None:
        pts = _compose(ctx, a, na, sa, res[0], sb, b, nb)
        if _clashes(ctx, pts) and time.time() < ctx.deadline:
            strict = _search(ctx, ca, cb, inflate, zb, _dir_index(na), wide=False, own=own, strict=True)
            if strict is not None:
                spts = _compose(ctx, a, na, sa, strict[0], sb, b, nb)
                if _clashes(ctx, spts) < _clashes(ctx, pts):
                    res, pts = strict, spts
        ctx.stats.astar += 1
        _occupy(g, res[1], (ca, cb), od)
        return pts, False
    ctx.stats.fallback += 1
    ctx.notes.append(f"{label}: Manhattan fallback route (no path outside the inflated equipment "
                     f"within the search limits / time budget); may cross equipment")
    mid = [(sa[0], sa[1], zb), (sb[0], sa[1], zb), (sb[0], sb[1], zb)]
    return _compose(ctx, a, na, sa, mid, sb, b, nb), True


def _occupy(g: VoxelGrid, cells: Sequence[tuple[int, int, int]], terminals, od: float) -> None:
    """Mark routed cells as used (6-neighbourhood too for pipes larger than ~DN300)."""
    for c in cells:
        if c in terminals:
            continue
        g.pipes[c] = True
        if od > 0.35:
            for di, dj, dk in _DIRS:
                q = (c[0] + di, c[1] + dj, c[2] + dk)
                if all(0 <= q[i] < g.shape[i] for i in range(3)) and not g.reserved[q]:
                    g.pipes[q] = True


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
               od: float, gap: float = 0.05) -> Optional[list[tuple[float, int]]]:
    """Centre arc positions (and segment index) for items of ``lengths`` near ``fracs`` x total,
    on straight segments, in order, preferring horizontal segments."""
    total = cum[-1]
    segs = []
    for i in range(len(pts) - 1):
        s0, s1 = cum[i], cum[i + 1]
        if s1 - s0 < 1e-9:
            continue
        vertical = abs(pts[i + 1][2] - pts[i][2]) > 1e-9
        m0 = 0.0 if i == 0 else od / 2 + 0.02
        m1 = 0.0 if i == len(pts) - 2 else od / 2 + 0.02
        segs.append((s0 + m0, s1 - m1, vertical, i))
    out: list[tuple[float, int]] = []
    lb = gap
    for k, (L, f) in enumerate(zip(lengths, fracs)):
        target = f * total
        rest = sum(lengths[k + 1:]) + gap * (len(lengths) - k)
        best = None
        for a0, b0, vertical, i in segs:
            a = max(a0 + L / 2, lb + L / 2)
            b = min(b0 - L / 2, total - rest - L / 2 - gap)
            if a > b + 1e-9:
                continue
            c = min(max(target, a), b)
            cost = abs(c - target) + (1.5 if vertical else 0.0)
            if best is None or cost < best[0]:
                best = (cost, c, i)
        if best is None:
            return None
        out.append((best[1], best[2]))
        lb = best[1] + L / 2 + gap
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


def _items_along(ctx: _Ctx, chain: _Chain, pts: list[Pt], od: float, line: Optional[str], dead_end: bool
                 ) -> tuple[list[Placement3D], list[tuple[float, float]]]:
    """Place the chain's inline items along ``pts``; returns placements and their (s_in, s_out)."""
    objs = ctx.doc.object_by_id()
    cum = _cum(pts)
    total = cum[-1]
    n = len(chain.items)
    if n == 0:
        return [], []
    sizes = [_inline_size(objs[i], od) for i, _, _ in chain.items]
    drawn = [max(1e-3, ctx.pipes[p].length) for p in chain.pipes]
    acc, fracs = 0.0, []
    for k in range(n):
        acc += drawn[k]
        fracs.append(acc / sum(drawn) if not dead_end else (k + 1) / (n + 0.5))
    lengths = [s[0] for s in sizes]
    if dead_end and objs[chain.items[-1][0]].category == Category.INSTRUMENT:
        lengths[-1] = 0.0
    fit = None
    for shrink in (1.0, 0.7, 0.45):
        ls = [L * shrink for L in lengths]
        fit = _fit_items(pts, cum, ls, fracs, od) or _fit_items(pts, cum, ls, [(k + 1) / (n + 1) for k in range(n)], od)
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
    for k, ((oid, entry, exit_), (c, _), L) in enumerate(zip(chain.items, fit, lengths)):
        obj = objs[oid]
        size = sizes[k]
        tip_instrument = dead_end and k == n - 1 and obj.category == Category.INSTRUMENT
        if L < size[0] and not tip_instrument:
            size = (round(max(L, 0.02), 3), size[1], size[2])
        if tip_instrument:
            tip = pts[-1]
            prev = pts[-2] if len(pts) > 1 else pts[-1]
            pl = Placement3D(object_id=oid, cls=obj.cls, position=Vec3(x=round(tip[0], 4), y=round(tip[1], 4),
                                                                       z=round(tip[2], 4)),
                             yaw_deg=_yaw_of(prev, tip), size=Vec3(x=size[0], y=size[1], z=size[2]),
                             deck=ctx.room.deck_below_z(tip[2]),
                             nozzles={entry: Vec3(x=round(tip[0], 4), y=round(tip[1], 4), z=round(tip[2], 4))},
                             rationale=f"[router] instrument on a stub of {line or 'its line'}")
            out.append(pl)
            spans.append((total, total))
            continue
        s_in, s_out = c - L / 2, c + L / 2
        p_in, _ = _point_at(pts, cum, s_in)
        p_out, _ = _point_at(pts, cum, s_out)
        rev = entry in ("out", "discharge") or (exit_ in ("in", "suction"))
        out.append(_make_inline(ctx, obj, size, p_in, p_out, entry, exit_, rev,
                                f"inline on {line or 'unnumbered line'} (pipes {', '.join(chain.pipes)})"))
        spans.append((s_in, s_out))
    return out, spans


def _term_key(end: _End, placed: dict[str, Placement3D]) -> str:
    return f"{end.node}\x00{end.key}" if end.node in placed else end.node


def _stub(ctx: _Ctx, start: Pt, normal, length: float) -> list[Pt]:
    """Dead-end stub: out along the nozzle normal, then up (instrument taps, vents, drains)."""
    if normal is not None and abs(normal[2]) < 0.5:
        p1 = _add(start, normal, 0.3)
        return [start, p1, _add(p1, (0, 0, 1), max(0.2, length))]
    d = (0, 0, -1) if normal is not None and normal[2] < -0.5 else (0, 0, 1)
    return [start, _add(start, d, max(0.3, length))]


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
    ctx = _Ctx(doc, room, grid, pipes, pmap, {}, routing or RoutingPlan(), default_medium, t0 + time_budget,
               stats, notes)

    # --- positions of junctions, hubs and free ends
    mapping = _drawing_map(doc, pmap, room)
    jpos = {j.id: j.position for j in doc.junctions}
    nodes: dict[str, Pt] = {}
    node_z: dict[str, float] = {}
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
    hard = grid.blocked(0.25)
    for n in sorted(nodes):
        x, y, z = nodes[n]
        c = grid.nearest_free(grid.cell((x, y, z)), 0.25, radius=8)
        cells = [c] + [q for q in ((c[0] + di, c[1] + dj, c[2] + dk) for di, dj, dk in _DIRS)
                       if all(0 <= q[i] < grid.shape[i] for i in range(3)) and not hard[q]]
        for q in cells:
            grid.reserved[q] = True
        ctx.reserved_by[n] = cells
        nodes[n] = grid.center(c)
    ctx.node_pos = nodes

    # reserve nozzle stub cells of placed equipment
    for pl in placements:
        for name, nz in pl.nozzles.items():
            nrm = nozzle_normal(pl, nz)
            c = grid.cell(_add(nz.as_tuple(), nrm, 0.35))
            if not grid.blocked(0.1)[c]:
                grid.reserved[c] = True
                ctx.reserved_by[f"{pl.object_id}\x00{name}"] = [c]

    # --- route chains: priority lines, then largest OD first; stubs last
    prio = {ln.strip().upper(): i for i, ln in enumerate(ctx.routing.priority)}

    def chain_key(ch: _Chain):
        medium, line, od = _chain_medium_line(ctx, ch)
        lines = [pipes[p].line_number.strip().upper() for p in ch.pipes if pipes[p].line_number]
        pr = min((prio[ln] for ln in lines if ln in prio), default=len(prio))
        return (ch.end is None, pr, -od, -len(ch.pipes), ch.pipes[0])

    routes: dict[str, PipeRoute3D] = {}
    inline: list[Placement3D] = []
    for ch in sorted(chains, key=chain_key):
        medium, line, od = _chain_medium_line(ctx, ch)
        zb = band_z(room, ctx.routing, medium, line)
        a, na = _endpoint(ctx, ch.start)
        if ch.end is not None:
            zb = adapt_band(zb, a[2], _endpoint(ctx, ch.end)[0][2])
        label = f"chain {', '.join(ch.pipes)}"
        if ch.end is None:
            stats.stubs += 1
            sizes = [_inline_size(objs[i], od)[0] for i, _, _ in ch.items]
            pts = _stub(ctx, a, na, 0.3 + sum(sizes) + 0.1 * len(sizes))
        else:
            b, nb_ = _endpoint(ctx, ch.end)
            if ch.start.node == ch.end.node and ch.start.key == ch.end.key:
                pts = [a, _add(a, (0, 0, 1), 0.5)]
            else:
                own = ctx.reserved_by.get(_term_key(ch.start, pmap), []) + \
                    ctx.reserved_by.get(_term_key(ch.end, pmap), [])
                pts, _ = _route_between(ctx, a, na, b, nb_, od, zb, label, own)
        if len(pts) < 2:
            pts = [pts[0], _add(pts[0], (0, 0, 1), 0.05)]
        placed_items, spans = _items_along(ctx, ch, pts, od, line, ch.end is None)
        inline.extend(placed_items)
        cum = _cum(pts)
        cuts = [0.0]
        for s_in, s_out in spans:
            cuts += [s_in, s_out]
        cuts.append(cum[-1])
        for k, pid in enumerate(ch.pipes):
            s0, s1 = cuts[2 * k], cuts[2 * k + 1]
            if s1 < s0:
                s1 = s0
            sub = _sub_polyline(pts, cum, s0, s1)
            if ch.rev[k]:
                sub = sub[::-1]
            p = pipes[pid]
            routes[pid] = PipeRoute3D(pipe_id=pid, points=[Vec3(x=round(x, 4), y=round(y, 4), z=round(z, 4))
                                                           for x, y, z in sub],
                                      diameter=round(_pipe_od(p), 4), medium=pipe_medium(p, default_medium),
                                      line_number=p.line_number)

    # --- hubs (three-way valves etc.) sit at their node position
    for h in sorted(hubs):
        o = objs[h]
        lst = adj.get(h, [])
        od = max((_pipe_od(p) for p, _ in lst), default=0.1)
        size = _inline_size(o, od)
        x, y, z = nodes[h]
        noz = {}
        for p, w in lst:
            r = routes.get(p.id)
            if r is not None:
                end = r.points[0] if w == 0 else r.points[-1]
                noz[port_key(p.start if w == 0 else p.end, p.id)] = end
        inline.append(Placement3D(object_id=h, cls=o.cls,
                                  position=Vec3(x=x, y=y, z=round(z - axis_height(o.cls, size), 4)),
                                  size=Vec3(x=size[0], y=size[1], z=size[2]), deck=room.deck_below_z(z),
                                  nozzles=noz, rationale="[router] branch item placed at its junction position"))

    # --- objects without process piping
    placed_ids = set(pmap) | {p.object_id for p in inline}
    extra_boxes = [placement_box(p) for p in placements]
    for o in doc.objects:
        if o.id in placed_ids:
            continue
        inline.append(_place_isolated(doc, o, room, pmap, {p.object_id: p for p in inline}, mapping, extra_boxes))
        extra_boxes.append(placement_box(inline[-1]))

    # --- checks
    boxes = [(p.object_id, placement_box(p)) for p in placements]
    hits = 0
    for r in routes.values():
        pts = [v.as_tuple() for v in r.points]
        stats.length_m += sum(_dist(a, b) for a, b in zip(pts, pts[1:]))
        for a, b in zip(pts, pts[1:]):
            hits += sum(1 for _, bx in boxes if segment_hits_box(a, b, bx))
    stats.collisions = hits
    stats.seconds = time.time() - t0
    notes.append(f"router: {stats.chains} chains ({stats.stubs} stubs), A* {stats.astar} "
                 f"(widened {stats.widened}), Manhattan fallback {stats.fallback}, "
                 f"segments through equipment {hits}, total pipe length {stats.length_m:.1f} m, "
                 f"grid {res} m, {stats.seconds:.2f} s")
    ordered = [routes[p.id] for p in doc.pipes if p.id in routes]
    return ordered, inline, notes


def _place_isolated(doc: PIDDocument, o: PIDObject, room: EngineRoom, pmap: dict[str, Placement3D],
                    inline: dict[str, Placement3D], mapping, boxes: list[Box3]) -> Placement3D:
    """Object with no process connection: near a signal-linked object, else at its mapped drawing position."""
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
        b = placement_box(anchor)
        target = (b.center[0], b.y1 + 0.3 + size[1] / 2, b.z1 + 0.2)
        why = f"no process piping; placed next to {anchor.object_id} it is signal-linked to"
    else:
        x, y = mapping(o.center.x, o.center.y)
        target = (x, y, room.deck_z("floor"))
        why = "no piping; placed at its drawing position on the floor deck"
    best = target
    for r in np.arange(0.0, 8.0, 0.5):
        found = False
        for ang in range(0, 360, 45 if r > 0 else 360):
            cx = min(max(target[0] + r * math.cos(math.radians(ang)), size[0] / 2), room.length - size[0] / 2)
            cy = min(max(target[1] + r * math.sin(math.radians(ang)), -room.breadth / 2 + size[1] / 2),
                     room.breadth / 2 - size[1] / 2)
            bx = Box3(cx - size[0] / 2 - 0.1, cy - size[1] / 2 - 0.1, target[2], cx + size[0] / 2 + 0.1,
                      cy + size[1] / 2 + 0.1, target[2] + size[2])
            if not any(bx.x0 < b.x1 and b.x0 < bx.x1 and bx.y0 < b.y1 and b.y0 < bx.y1 and bx.z0 < b.z1 and
                       b.z0 < bx.z1 for b in boxes):
                best, found = (cx, cy, target[2]), True
                break
        if found:
            break
    return Placement3D(object_id=o.id, cls=o.cls, position=Vec3(x=round(best[0], 4), y=round(best[1], 4),
                                                                z=round(best[2], 4)),
                       size=Vec3(x=size[0], y=size[1], z=size[2]), deck=room.deck_below_z(best[2]),
                       rationale=f"[router] {why}")
