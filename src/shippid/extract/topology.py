"""Pipe network construction.

Pipe lines are cut where they pass through symbol bodies, their ends are
attached to symbols (ports) or to each other, and the result is traced into
runs between terminal nodes:

* line ends meeting at one point: two collinear ends continue the run, two
  perpendicular ends form a corner, three or more form a junction, except a
  plain four-way meeting of two straight lines without a dot (a crossing);
* a line end on the interior of another line is a tee (connected, dot or not);
* two lines crossing interior-to-interior are connected only with a dot;
* dashed (signal) and solid (process) networks never connect to each other.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

import shapely

from .pipes import Dot, MLine
from .primitives import DSU, Prim, point_prim_dist, point_seg_dist, seg_param

BBox4 = tuple[float, float, float, float]


@dataclass
class Body:
    """What topology needs to know about a detected symbol."""

    index: int
    bbox: BBox4  # full drawn bbox (incl. actuator)
    body: BBox4  # bbox of the symbol body (pipes passing through it are cut)
    prims: list[Prim]
    ports: list[tuple[str, tuple[float, float], tuple[float, float]]]  # named template ports (page)
    boundary_ports: bool = False
    is_valve: bool = False


@dataclass
class Piece:
    x0: float
    y0: float
    x1: float
    y1: float
    dashed: bool
    width: float
    color: object
    line_id: int
    crossings: list[tuple[float, float]] = field(default_factory=list)
    alive: bool = True

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    def point(self, t: float) -> tuple[float, float]:
        return (self.x0 + (self.x1 - self.x0) * t, self.y0 + (self.y1 - self.y0) * t)

    def end(self, k: int) -> tuple[float, float]:
        return (self.x0, self.y0) if k == 0 else (self.x1, self.y1)

    def direction_from(self, k: int) -> tuple[float, float]:
        """Unit vector pointing from end ``k`` into the piece."""
        (ax, ay), (bx, by) = (self.end(k), self.end(1 - k))
        n = math.hypot(bx - ax, by - ay) or 1.0
        return ((bx - ax) / n, (by - ay) / n)


@dataclass
class Node:
    id: int
    x: float
    y: float
    ends: list[tuple[int, int]] = field(default_factory=list)  # (piece index, end 0/1)
    body: Optional[int] = None  # attached symbol index
    port: Optional[str] = None
    dotted: bool = False
    cut: bool = False  # created by cutting at a body
    junction: Optional[int] = None
    passthrough: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    fixed_port: bool = False  # port name decided when the node was created (direct flange connection)
    direction: Optional[tuple[float, float]] = None


@dataclass
class Junc:
    id: int
    x: float
    y: float
    kind: str
    dotted: bool


@dataclass
class Run:
    points: list[tuple[float, float]]
    start: Node
    end: Node
    pieces: list[int]
    dashed: bool
    width: float
    color: object
    crossings: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Network:
    runs: list[Run]
    junctions: list[Junc]
    nodes: list[Node]
    ports: dict[int, list[tuple[str, tuple[float, float], tuple[float, float]]]]  # body index -> ports
    pieces: list[Piece]


def _clip_interval(p: Piece, b: BBox4, eps: float) -> Optional[tuple[float, float]]:
    """Parameter interval of the piece strictly inside box ``b`` shrunk by ``eps``."""
    x0, y0, x1, y1 = b[0] + eps, b[1] + eps, b[2] - eps, b[3] - eps
    if x0 >= x1 or y0 >= y1:
        return None
    dx, dy = p.x1 - p.x0, p.y1 - p.y0
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p.x0 - x0), (dx, x1 - p.x0), (-dy, p.y0 - y0), (dy, y1 - p.y0)):
        if abs(pp) < 1e-12:
            if qq < 0:
                return None
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t1 - t0 <= 1e-9:
        return None
    return (t0, t1)


def build_network(lines: Sequence[MLine], bodies: Sequence[Body], dots: Sequence[Dot], unit: float, tol: float) -> Network:
    node_tol = max(1.0, 0.5 * tol)
    tee_tol = max(1.0, min(tol, 0.12 * unit))
    attach_tol = max(tol, 0.15 * unit)

    pieces: list[Piece] = [Piece(m.x0, m.y0, m.x1, m.y1, m.dashed, m.width, m.color, m.id, list(m.crossings)) for m in lines]
    cut_ends: dict[tuple[int, int], int] = {}  # (piece, end) -> body index

    # ---- 1. cut pieces where they run through symbol bodies
    if bodies:
        btree = shapely.STRtree([shapely.box(*b.body) for b in bodies])
        k = 0
        while k < len(pieces):
            p = pieces[k]
            if not p.alive:
                k += 1
                continue
            hits = btree.query(shapely.LineString([(p.x0, p.y0), (p.x1, p.y1)]))
            best = None
            for bi in sorted(hits.tolist()):
                iv = _clip_interval(p, bodies[bi].body, 0.25)
                if iv is None or (iv[1] - iv[0]) * p.length < 0.3:
                    continue
                if best is None or iv[0] < best[1][0]:
                    best = (bi, iv)
            if best is None:
                k += 1
                continue
            bi, (t0, t1) = best
            p.alive = False
            ax, ay = p.point(t0)
            bx, by = p.point(t1)
            if t0 * p.length >= 0.3:
                q = Piece(p.x0, p.y0, ax, ay, p.dashed, p.width, p.color, p.line_id, _cross_in(p.crossings, p, 0, t0))
                pieces.append(q)
                cut_ends[(len(pieces) - 1, 1)] = bi
                for key, v in list(cut_ends.items()):
                    if key == (k, 0):
                        cut_ends[(len(pieces) - 1, 0)] = v
            if (1 - t1) * p.length >= 0.3:
                q = Piece(bx, by, p.x1, p.y1, p.dashed, p.width, p.color, p.line_id, _cross_in(p.crossings, p, t1, 1))
                pieces.append(q)
                cut_ends[(len(pieces) - 1, 0)] = bi
                for key, v in list(cut_ends.items()):
                    if key == (k, 1):
                        cut_ends[(len(pieces) - 1, 1)] = v
            k += 1
    alive = [i for i, p in enumerate(pieces) if p.alive]

    # ---- 2. attach ends to symbols
    attach: dict[tuple[int, int], tuple[int, bool]] = {}  # (piece, end) -> (body, on actuator)
    if bodies:
        bb_tree = shapely.STRtree([shapely.box(*b.bbox).buffer(attach_tol) for b in bodies])
    for i in alive:
        p = pieces[i]
        for e in (0, 1):
            if (i, e) in cut_ends:
                attach[(i, e)] = (cut_ends[(i, e)], False)
                continue
            if not bodies:
                continue
            x, y = p.end(e)
            best = None
            for bi in bb_tree.query(shapely.Point(x, y)).tolist():
                b = bodies[bi]
                d = min((point_prim_dist(x, y, q) for q in b.prims), default=1e9)
                inside = b.body[0] < x < b.body[2] and b.body[1] < y < b.body[3]
                if inside:
                    d = min(d, 0.0)
                if d <= attach_tol and (best is None or d < best[0]):
                    on_act = not (b.body[0] - attach_tol <= x <= b.body[2] + attach_tol and
                                  b.body[1] - attach_tol <= y <= b.body[3] + attach_tol)
                    best = (d, bi, on_act)
            if best is not None:
                # a line end attaches only if the line leaves the symbol (does not run along it)
                attach[(i, e)] = (best[1], best[2])

    # ---- 3. nodes from coincident ends (per network kind)
    dsu = DSU(2 * len(pieces))
    end_pts = [(i, e) for i in alive for e in (0, 1)]
    for dashed in (False, True):
        sel = [(i, e) for i, e in end_pts if pieces[i].dashed == dashed]
        if len(sel) < 2:
            continue
        pts = shapely.points([pieces[i].end(e) for i, e in sel])
        tree = shapely.STRtree(pts)
        a, b = tree.query(pts, predicate="dwithin", distance=node_tol)
        for ia, ib in zip(a.tolist(), b.tolist()):
            if ia < ib:
                ka, kb = sel[ia], sel[ib]
                ba, bb = attach.get(ka), attach.get(kb)
                if ba is not None and bb is not None and ba[0] != bb[0]:
                    continue  # two different symbols touching each other through a zero-length gap
                dsu.union(2 * ka[0] + ka[1], 2 * kb[0] + kb[1])
    nodes: list[Node] = []
    node_of: dict[tuple[int, int], int] = {}
    for root, members in dsu.groups([2 * i + e for i, e in end_pts]).items():
        ends = [(m // 2, m % 2) for m in members]
        xs = [pieces[i].end(e)[0] for i, e in ends]
        ys = [pieces[i].end(e)[1] for i, e in ends]
        n = Node(len(nodes), sum(xs) / len(xs), sum(ys) / len(ys), ends)
        for ie in ends:
            node_of[ie] = n.id
            if ie in attach:
                n.body = attach[ie][0]
                if attach[ie][1]:
                    n.port = "actuator"
            if ie in cut_ends:
                n.cut = True
        nodes.append(n)

    # ---- 4. dots: split lines passing through a dot, snap ends to the dot
    def split_piece(i: int, t: float, at_node: Node) -> None:
        p = pieces[i]
        x, y = p.point(t)
        p.alive = False
        q0 = Piece(p.x0, p.y0, x, y, p.dashed, p.width, p.color, p.line_id, _cross_in(p.crossings, p, 0, t))
        q1 = Piece(x, y, p.x1, p.y1, p.dashed, p.width, p.color, p.line_id, _cross_in(p.crossings, p, t, 1))
        pieces.extend([q0, q1])
        i0, i1 = len(pieces) - 2, len(pieces) - 1
        for old_e, new in ((0, (i0, 0)), (1, (i1, 1))):
            nid = node_of.pop((i, old_e))
            nd = nodes[nid]
            nd.ends = [new if ie == (i, old_e) else ie for ie in nd.ends]
            node_of[new] = nid
            if (i, old_e) in attach:
                attach[new] = attach.pop((i, old_e))
        at_node.ends.extend([(i0, 1), (i1, 0)])
        node_of[(i0, 1)] = at_node.id
        node_of[(i1, 0)] = at_node.id

    def live_pieces(dashed: bool) -> list[int]:
        return [i for i, p in enumerate(pieces) if p.alive and p.dashed == dashed]

    for dot in dots:
        target: Optional[Node] = None
        reach = dot.r + node_tol
        for n in nodes:
            if n.ends and math.hypot(n.x - dot.x, n.y - dot.y) <= reach and not pieces[n.ends[0][0]].dashed:
                if target is None:
                    target = n
                    n.x, n.y = dot.x, dot.y
                else:
                    for ie in n.ends:
                        node_of[ie] = target.id
                    target.ends.extend(n.ends)
                    n.ends = []
        if target is None:
            target = Node(len(nodes), dot.x, dot.y)
            nodes.append(target)
        target.dotted = True
        for i in live_pieces(False):
            p = pieces[i]
            if point_seg_dist(dot.x, dot.y, p.x0, p.y0, p.x1, p.y1) > reach:
                continue
            t = seg_param(dot.x, dot.y, p.x0, p.y0, p.x1, p.y1)
            if reach / max(p.length, 1e-9) < t < 1 - reach / max(p.length, 1e-9):
                split_piece(i, t, target)

    # ---- 5. tees: a free node lying on another piece's interior
    for dashed in (False, True):
        ids = live_pieces(dashed)
        if not ids:
            continue
        tree = shapely.STRtree([shapely.LineString([(pieces[i].x0, pieces[i].y0), (pieces[i].x1, pieces[i].y1)]) for i in ids])
        n_indexed = len(pieces)
        for n in list(nodes):
            if not n.ends or n.body is not None or pieces[n.ends[0][0]].dashed != dashed:
                continue
            own = {i for i, _ in n.ends}
            probe = shapely.Point(n.x, n.y).buffer(tee_tol)
            # pieces created by earlier splits are not in the tree: check them directly
            cand = [ids[k] for k in tree.query(probe).tolist()] + list(range(n_indexed, len(pieces)))
            best = None
            for i in cand:
                p = pieces[i]
                if i in own or not p.alive or p.dashed != dashed:
                    continue
                d = point_seg_dist(n.x, n.y, p.x0, p.y0, p.x1, p.y1)
                if d > tee_tol:
                    continue
                t = seg_param(n.x, n.y, p.x0, p.y0, p.x1, p.y1)
                m = node_tol / max(p.length, 1e-9)
                if not (m < t < 1 - m):
                    continue
                if best is None or d < best[0]:
                    best = (d, i, t)
            if best is not None:
                _, i, t = best
                x, y = pieces[i].point(t)
                n.x, n.y = x, y
                split_piece(i, t, n)
            if len(pieces) - n_indexed > 64:  # keep the linear tail short
                ids = live_pieces(dashed)
                tree = shapely.STRtree([shapely.LineString([(pieces[j].x0, pieces[j].y0), (pieces[j].x1, pieces[j].y1)])
                                        for j in ids])
                n_indexed = len(pieces)

    # ---- 6. classify nodes
    junctions: list[Junc] = []
    for n in nodes:
        deg = len(n.ends)
        if deg == 0:
            continue
        if n.body is not None:
            continue  # symbol port: terminal
        if deg == 1:
            continue  # free end
        if deg == 2 and not n.dotted:
            continue  # corner or straight continuation
        if deg == 4 and not n.dotted:
            pairs = _straight_pairs(n, pieces)
            if pairs is not None:
                n.passthrough = pairs
                continue
        kind = "tee" if deg == 3 else ("cross" if deg >= 4 else "corner")
        n.junction = len(junctions)
        junctions.append(Junc(len(junctions), n.x, n.y, kind, n.dotted))

    # ---- 7. trace runs between terminal nodes
    runs: list[Run] = []
    used: set[int] = set()
    runs.extend(_direct_links(bodies, nodes, node_tol, unit))

    def terminal(n: Node) -> bool:
        return n.body is not None or n.junction is not None or len(n.ends) == 1

    def walk(start: Node, first: tuple[int, int]) -> Run:
        pts = [(start.x, start.y)]
        pcs: list[int] = []
        cur_piece, cur_end = first
        node = start
        while True:
            used.add(cur_piece)
            pcs.append(cur_piece)
            other = (cur_piece, 1 - cur_end)
            node = nodes[node_of[other]]
            pts.append((node.x, node.y))
            if terminal(node):
                break
            if node.passthrough:
                nxt = node.passthrough.get(other)
            else:
                rest = [ie for ie in node.ends if ie != other]
                nxt = rest[0] if len(rest) == 1 else None
            if nxt is None or nxt[0] in used:
                break
            cur_piece, cur_end = nxt
        ps = [pieces[i] for i in pcs]
        crossings = [c for p in ps for c in p.crossings]
        L = sum(p.length for p in ps) or 1.0
        dashed = sum(p.length for p in ps if p.dashed) >= 0.5 * L
        width = max(p.width for p in ps)
        color = max(ps, key=lambda p: p.length).color
        return Run(_simplify(pts), start, node, pcs, dashed, width, color, crossings)

    for n in nodes:
        if n.ends and terminal(n):
            for ie in sorted(n.ends):
                if ie[0] not in used and pieces[ie[0]].alive:
                    runs.append(walk(n, ie))
    for i, p in enumerate(pieces):  # loops without terminals
        if p.alive and i not in used:
            n = nodes[node_of[(i, 0)]]
            runs.append(walk(n, (i, 0)))

    # ---- 8. port names per symbol
    ports: dict[int, list[tuple[str, tuple[float, float], tuple[float, float]]]] = defaultdict(list)
    by_body: dict[int, list[tuple[Node, tuple[float, float]]]] = defaultdict(list)
    for n in nodes:
        if n.body is None:
            continue
        if n.fixed_port:
            by_body[n.body].append((n, (0.0, 0.0)))
            continue
        if not n.ends:
            continue
        i, e = n.ends[0]
        d = pieces[i].direction_from(e)
        by_body[n.body].append((n, (-d[0], -d[1])))  # pointing from the pipe towards the symbol
    for bi, lst in by_body.items():
        b = bodies[bi]
        lst.sort(key=lambda t: (round(t[0].y, 1), round(t[0].x, 1)))
        taken: dict[str, tuple[float, float]] = {}
        counter = 0
        named_ports = [] if b.boundary_ports else list(b.ports)
        D = max(b.body[2] - b.body[0], b.body[3] - b.body[1], 1e-6)
        # greedy nearest assignment of attachment points to library ports
        cand = []
        for k, (n, _) in enumerate(lst):
            if n.port == "actuator" or n.fixed_port:
                continue
            for name, (px, py), _ in named_ports:
                cand.append((math.hypot(n.x - px, n.y - py), k, name))
        cand.sort()
        assigned: dict[int, str] = {}
        for d, k, name in cand:
            if k in assigned or name in assigned.values() or d > 0.45 * D:
                continue
            assigned[k] = name
        for k, (n, inward) in enumerate(lst):
            if n.port == "actuator":
                name = "actuator"
            elif n.fixed_port and n.port:
                name = n.port
            elif k in assigned:
                name = assigned[k]
            else:
                while f"n{counter}" in taken:
                    counter += 1
                name = f"n{counter}"
                counter += 1
            n.port = name
            outward = n.direction if n.direction is not None else _outward(b, n, inward)
            if name not in taken:
                taken[name] = (n.x, n.y)
                ports[bi].append((name, (n.x, n.y), outward))
    return Network(runs, junctions, nodes, dict(ports), pieces)


def _direct_links(bodies: Sequence[Body], nodes: list[Node], node_tol: float, unit: float) -> list[Run]:
    """Symbols bolted flange-to-flange: a named port of one symbol lying on the
    outline of another with no pipe attached there becomes a zero-length run."""
    out: list[Run] = []
    if len(bodies) < 2:
        return out
    tol = max(node_tol, 0.06 * unit)
    tree = shapely.STRtree([shapely.box(*b.bbox).buffer(tol) for b in bodies])
    attached = [(n.body, n.x, n.y) for n in nodes if n.body is not None]
    linked: set[tuple[int, int]] = set()

    def has_pipe(bi: int, x: float, y: float) -> bool:
        return any(b == bi and math.hypot(x - nx, y - ny) <= 2 * tol for b, nx, ny in attached)

    for a in bodies:
        for name, (px, py), d in a.ports:
            if has_pipe(a.index, px, py):
                continue
            for bi in tree.query(shapely.Point(px, py)).tolist():
                b = bodies[bi]
                if bi == a.index or (min(a.index, bi), max(a.index, bi)) in linked:
                    continue
                if min((point_prim_dist(px, py, q) for q in b.prims), default=1e9) > tol:
                    continue
                if has_pipe(bi, px, py):
                    continue
                other = None
                if b.ports and not b.boundary_ports:
                    nm, (qx, qy), qd = min(b.ports, key=lambda t: math.hypot(t[1][0] - px, t[1][1] - py))
                    if math.hypot(qx - px, qy - py) <= 3 * tol:
                        other = (nm, qd)
                na = Node(len(nodes), px, py, [], a.index, name, fixed_port=True, direction=d)
                nodes.append(na)
                nb = Node(len(nodes), px, py, [], bi, other[0] if other else None, fixed_port=other is not None,
                          direction=other[1] if other else (-d[0], -d[1]))
                nodes.append(nb)
                linked.add((min(a.index, bi), max(a.index, bi)))
                out.append(Run([(px, py), (px, py)], na, nb, [], False, 0.0, None))
                break
    return out


def _outward(b: Body, n: Node, toward_pipe: tuple[float, float]) -> tuple[float, float]:
    """Outward direction of a port: the direction in which the pipe leaves the symbol."""
    for name, (px, py), d in b.ports:
        if name == n.port:
            return d
    # direction of the pipe, pointing away from the symbol centre
    cx, cy = (b.body[0] + b.body[2]) / 2, (b.body[1] + b.body[3]) / 2
    dx, dy = -toward_pipe[0], -toward_pipe[1]
    if (n.x - cx) * dx + (n.y - cy) * dy < 0:
        dx, dy = -dx, -dy
    return (dx, dy)


def _straight_pairs(n: Node, pieces: list[Piece]) -> Optional[dict[tuple[int, int], tuple[int, int]]]:
    """For a 4-way node: pair opposite collinear arms, or None if not two straight lines."""
    dirs = [(ie, pieces[ie[0]].direction_from(ie[1])) for ie in n.ends]
    pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for a, da in dirs:
        best = None
        for b, db in dirs:
            if b == a:
                continue
            dot = da[0] * db[0] + da[1] * db[1]
            if dot < -0.97 and (best is None or dot < best[0]):
                best = (dot, b)
        if best is None:
            return None
        pairs[a] = best[1]
    if any(pairs[pairs[a]] != a for a in pairs):
        return None
    return pairs


def _cross_in(crossings, p: Piece, t0: float, t1: float) -> list[tuple[float, float]]:
    out = []
    for c in crossings:
        t = seg_param(c[0], c[1], p.x0, p.y0, p.x1, p.y1)
        if t0 - 1e-9 <= t <= t1 + 1e-9:
            out.append(c)
    return out


def _simplify(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop duplicate and collinear interior vertices."""
    out: list[tuple[float, float]] = []
    for p in pts:
        if out and math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) < 1e-6:
            continue
        out.append(p)
    k = 1
    while k < len(out) - 1:
        a, b, c = out[k - 1], out[k], out[k + 1]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
        if abs(cross) <= 1e-6 * max(1.0, math.hypot(c[0] - a[0], c[1] - a[1])) ** 2 and dot > 0:
            del out[k]
        else:
            k += 1
    if len(out) == 1:
        out.append(out[0])
    return out
