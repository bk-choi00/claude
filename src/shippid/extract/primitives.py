"""Geometric primitives used by the extractor and small geometry helpers.

:class:`PrimSet` converts a :class:`~shippid.pdf.reader.PageContent` into a flat
list of :class:`Prim` (straight segments and circles) plus the closed rings
(polygon outlines) they belong to, and offers the spatial queries the later
stages need (pairs within a distance, points inside shapes).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import shapely
from shapely.geometry import Polygon, box

from ..pdf.reader import Color, PageContent

AXIS_TOL_DEG = 1.0


class DSU:
    """Union-find over ``n`` integer items."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb

    def groups(self, items: Optional[Iterable[int]] = None) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for i in items if items is not None else range(len(self.parent)):
            out[self.find(i)].append(i)
        return dict(out)


@dataclass
class Prim:
    id: int
    kind: str  # 'seg' | 'circle'
    x0: float
    y0: float
    x1: float
    y1: float
    cx: float = 0.0
    cy: float = 0.0
    r: float = 0.0
    width: float = 0.0
    color: Color = None
    dashed: bool = False
    filled: bool = False
    closed: bool = False
    curve: bool = False
    ring: int = -1
    path: int = -1
    sub: int = 0
    layer: str = ""
    role: str = ""  # '' | excluded | frame | table | pipe | symbol | hop | dot | arrow | boundary | noise

    @property
    def is_seg(self) -> bool:
        return self.kind == "seg"

    @property
    def length(self) -> float:
        if self.kind == "circle":
            return 2 * math.pi * self.r
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        if self.kind == "circle":
            return (self.cx - self.r, self.cy - self.r, self.cx + self.r, self.cy + self.r)
        return (min(self.x0, self.x1), min(self.y0, self.y1), max(self.x0, self.x1), max(self.y0, self.y1))

    @property
    def axis(self) -> Optional[str]:
        """'h' / 'v' for axis-aligned straight segments, else None."""
        if self.kind != "seg":
            return None
        dx, dy = abs(self.x1 - self.x0), abs(self.y1 - self.y0)
        if dx == 0 and dy == 0:
            return None
        ang = math.degrees(math.atan2(dy, dx))
        if ang <= AXIS_TOL_DEG:
            return "h"
        if ang >= 90 - AXIS_TOL_DEG:
            return "v"
        return None

    @property
    def is_core(self) -> bool:
        """Strokes that only occur in symbols: closed / filled / curved / circular / diagonal."""
        if self.kind == "circle":
            return True
        return self.closed or self.filled or self.curve or self.axis is None

    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.x0, self.y0), (self.x1, self.y1)

    def sample(self, step: float) -> list[tuple[float, float]]:
        if self.kind == "circle":
            n = max(8, int(2 * math.pi * self.r / max(step, 1e-6)))
            return [(self.cx + self.r * math.cos(2 * math.pi * i / n), self.cy + self.r * math.sin(2 * math.pi * i / n)) for i in range(n)]
        n = max(1, int(self.length / max(step, 1e-6)))
        return [(self.x0 + (self.x1 - self.x0) * i / n, self.y0 + (self.y1 - self.y0) * i / n) for i in range(n + 1)]


@dataclass
class Ring:
    """A closed outline: an explicit closed subpath, a filled shape or a rectangle
    assembled from four loose segments."""

    id: int
    prim_ids: list[int]
    pts: list[tuple[float, float]]
    filled: bool = False
    dashed: bool = False
    synthetic: bool = False  # assembled from loose segments
    _poly: Optional[Polygon] = field(default=None, repr=False)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def polygon(self):
        """Area enclosed by the outline (a Polygon; a MultiPolygon for a self-intersecting outline
        such as a bow-tie drawn as one closed 4-point polyline: both lobes, where ``buffer(0)``
        would keep only one)."""
        if self._poly is None:
            poly = Polygon(self.pts) if len(self.pts) >= 3 else box(*self.bbox)
            if not poly.is_valid:
                fixed = shapely.make_valid(poly)
                parts = [g for g in getattr(fixed, "geoms", [fixed]) if g.geom_type in ("Polygon", "MultiPolygon")]
                poly = shapely.unary_union(parts) if parts else poly.buffer(0)
                if poly.is_empty:
                    poly = box(*self.bbox)
            self._poly = poly
        return self._poly

    @property
    def is_rect(self) -> bool:
        if len(self.pts) != 4:
            return False
        x0, y0, x1, y1 = self.bbox
        tol = 0.02 * max(x1 - x0, y1 - y0, 1e-6)
        return all((abs(x - x0) <= tol or abs(x - x1) <= tol) and (abs(y - y0) <= tol or abs(y - y1) <= tol) for x, y in self.pts)


class PrimSet:
    """All drawing primitives of a page with rings and spatial helpers."""

    def __init__(self, prims: list[Prim], rings: dict[int, Ring]):
        self.prims = prims
        self.rings = rings
        self._tree: Optional[shapely.STRtree] = None
        self._tree_ids: Optional[np.ndarray] = None

    # -- construction ----------------------------------------------------------
    @classmethod
    def from_page(cls, pc: PageContent) -> "PrimSet":
        prims: list[Prim] = []
        rings: dict[int, Ring] = {}
        ring_key: dict[tuple[int, int], int] = {}
        for s in pc.segments:
            p = Prim(len(prims), "seg", s.x0, s.y0, s.x1, s.y1, width=s.width, color=s.color, dashed=s.dashed,
                     filled=s.filled, closed=s.closed, curve=s.curve >= 0, path=s.path, sub=s.sub, layer=s.layer)
            if s.closed:
                key = (s.path, s.sub)
                rid = ring_key.get(key)
                if rid is None:
                    rid = len(rings)
                    ring_key[key] = rid
                    rings[rid] = Ring(rid, [], [], filled=s.filled, dashed=s.dashed)
                p.ring = rid
                rings[rid].prim_ids.append(p.id)
                rings[rid].pts.append((s.x0, s.y0))
            prims.append(p)
        for c in pc.circles:
            prims.append(Prim(len(prims), "circle", c.cx - c.r, c.cy - c.r, c.cx + c.r, c.cy + c.r, cx=c.cx, cy=c.cy, r=c.r,
                              width=c.width, color=c.color, dashed=c.dashed, filled=c.filled, closed=True,
                              path=c.path, sub=c.sub, layer=c.layer))
        # rings with fewer than 3 vertices are degenerate (e.g. a closed 2-point path)
        for rid in [r for r, ring in rings.items() if len(ring.pts) < 3]:
            for pid in rings[rid].prim_ids:
                prims[pid].ring = -1
                prims[pid].closed = False
            del rings[rid]
        return cls(*_dedupe(prims, rings))

    # -- spatial queries -------------------------------------------------------
    def bbox_array(self, ids: Sequence[int]) -> np.ndarray:
        return np.array([self.prims[i].bbox for i in ids], dtype=float).reshape(-1, 4)

    def pairs_within(self, ids: Sequence[int], tol: float) -> list[tuple[int, int]]:
        """Candidate pairs (i < j) whose bboxes are within ``tol`` (exact test left to the caller)."""
        ids = list(ids)
        if len(ids) < 2:
            return []
        bb = self.bbox_array(ids)
        geoms = shapely.box(bb[:, 0] - tol, bb[:, 1] - tol, bb[:, 2] + tol, bb[:, 3] + tol)
        tree = shapely.STRtree(geoms)
        a, b = tree.query(geoms, predicate="intersects")
        m = a < b
        arr = np.asarray(ids)
        return list(zip(arr[a[m]].tolist(), arr[b[m]].tolist()))

    def distance(self, i: int, j: int) -> float:
        return prim_distance(self.prims[i], self.prims[j])


DEDUPE_Q = 20.0  # coordinates are compared on a 0.05 pt grid


def _dedupe(prims: list[Prim], rings: dict[int, Ring]) -> tuple[list[Prim], dict[int, Ring]]:
    """Drop coincident duplicate primitives (CAD exports often draw an entity twice, or a
    filled shape and its outline as two paths): identical rings / circles merge their
    fill flags, open segments identical to another segment or to a ring edge go.
    Prim ids and ring ids are renumbered."""

    def q(v: float) -> int:
        return int(round(v * DEDUPE_Q))

    def seg_key(p: Prim) -> tuple:
        a, b = (q(p.x0), q(p.y0)), (q(p.x1), q(p.y1))
        return (a, b) if a <= b else (b, a)

    drop: set[int] = set()
    # rings with the same vertex set
    ring_by_key: dict[tuple, int] = {}
    for rid in sorted(rings):
        ring = rings[rid]
        key = (len(ring.pts), frozenset((q(x), q(y)) for x, y in ring.pts), ring.dashed)
        keep = ring_by_key.get(key)
        if keep is None:
            ring_by_key[key] = rid
            continue
        kr = rings[keep]
        if ring.filled and not kr.filled:
            kr.filled = True
            for pid in kr.prim_ids:
                prims[pid].filled = True
        for pid in ring.prim_ids:
            k_w = prims[kr.prim_ids[0]]
            k_w.width = max(k_w.width, prims[pid].width)
            drop.add(pid)
        del rings[rid]
    # circles
    circ_by_key: dict[tuple, int] = {}
    for p in prims:
        if p.kind != "circle":
            continue
        key = (q(p.cx), q(p.cy), q(p.r))
        keep = circ_by_key.get(key)
        if keep is None:
            circ_by_key[key] = p.id
            continue
        k = prims[keep]
        k.filled = k.filled or p.filled
        k.dashed = k.dashed and p.dashed
        k.width = max(k.width, p.width)
        drop.add(p.id)
    # open segments: duplicates of ring edges or of each other
    edges = {seg_key(p) for p in prims if p.kind == "seg" and p.ring >= 0 and p.id not in drop}
    seen: dict[tuple, int] = {}
    for p in prims:
        if p.kind != "seg" or p.ring >= 0 or p.id in drop:
            continue
        key = seg_key(p)
        if key[0] == key[1]:
            continue
        if key in edges:
            drop.add(p.id)
            continue
        keep = seen.get(key)
        if keep is None:
            seen[key] = p.id
            continue
        k = prims[keep]
        k.dashed = k.dashed and p.dashed
        k.width = max(k.width, p.width)
        drop.add(p.id)
    if not drop:
        return prims, rings
    new_id: dict[int, int] = {}
    out: list[Prim] = []
    for p in prims:
        if p.id in drop:
            continue
        new_id[p.id] = len(out)
        p.id = len(out)
        out.append(p)
    new_rings: dict[int, Ring] = {}
    for k, rid in enumerate(sorted(rings)):
        ring = rings[rid]
        ring.id = k
        ring.prim_ids = [new_id[i] for i in ring.prim_ids if i in new_id]
        for i in ring.prim_ids:
            out[i].ring = k
        new_rings[k] = ring
    return out, new_rings


# --------------------------------------------------------------------------- geometry helpers


def point_seg_dist(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 <= 1e-12:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / L2))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def seg_param(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """Projection parameter of a point on the infinite line through the segment."""
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 <= 1e-12:
        return 0.0
    return ((px - x0) * dx + (py - y0) * dy) / L2


def _segs_intersect(a: Prim, b: Prim) -> bool:
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    d1 = orient(b.x0, b.y0, b.x1, b.y1, a.x0, a.y0)
    d2 = orient(b.x0, b.y0, b.x1, b.y1, a.x1, a.y1)
    d3 = orient(a.x0, a.y0, a.x1, a.y1, b.x0, b.y0)
    d4 = orient(a.x0, a.y0, a.x1, a.y1, b.x1, b.y1)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0


def seg_seg_dist(a: Prim, b: Prim) -> float:
    if _segs_intersect(a, b):
        return 0.0
    return min(
        point_seg_dist(a.x0, a.y0, b.x0, b.y0, b.x1, b.y1),
        point_seg_dist(a.x1, a.y1, b.x0, b.y0, b.x1, b.y1),
        point_seg_dist(b.x0, b.y0, a.x0, a.y0, a.x1, a.y1),
        point_seg_dist(b.x1, b.y1, a.x0, a.y0, a.x1, a.y1),
    )


def circle_seg_dist(c: Prim, s: Prim) -> float:
    dmin = point_seg_dist(c.cx, c.cy, s.x0, s.y0, s.x1, s.y1)
    if c.filled:
        return max(0.0, dmin - c.r)
    dmax = max(math.hypot(s.x0 - c.cx, s.y0 - c.cy), math.hypot(s.x1 - c.cx, s.y1 - c.cy))
    if dmin <= c.r <= dmax:
        return 0.0
    return dmin - c.r if c.r < dmin else c.r - dmax


def circle_circle_dist(a: Prim, b: Prim) -> float:
    d = math.hypot(a.cx - b.cx, a.cy - b.cy)
    if d >= a.r + b.r:
        return d - a.r - b.r
    if d <= abs(a.r - b.r):  # nested
        outer = b if a.r < b.r else a
        if outer.filled:
            return 0.0
        return abs(a.r - b.r) - d
    return 0.0


def prim_distance(a: Prim, b: Prim) -> float:
    if a.kind == "seg" and b.kind == "seg":
        return seg_seg_dist(a, b)
    if a.kind == "circle" and b.kind == "circle":
        return circle_circle_dist(a, b)
    return circle_seg_dist(a, b) if a.kind == "circle" else circle_seg_dist(b, a)


def point_prim_dist(px: float, py: float, p: Prim) -> float:
    if p.kind == "circle":
        d = math.hypot(px - p.cx, py - p.cy)
        return max(0.0, d - p.r) if p.filled else abs(d - p.r)
    return point_seg_dist(px, py, p.x0, p.y0, p.x1, p.y1)


def bbox_union(boxes: Iterable[Sequence[float]]) -> tuple[float, float, float, float]:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def bbox_dist(a: Sequence[float], b: Sequence[float]) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def point_bbox_dist(x: float, y: float, b: Sequence[float]) -> float:
    dx = max(b[0] - x, 0.0, x - b[2])
    dy = max(b[1] - y, 0.0, y - b[3])
    return math.hypot(dx, dy)


class BoxGrid:
    """Uniform-grid index of items with bounding boxes (spatial hash, ``cell`` points per cell):
    :meth:`near` returns the items whose bbox is within ``reach`` of a box, in insertion order,
    without scanning every item. An item spanning more than :attr:`MAX_CELLS` cells (a hull or
    package outline) is kept in a short list that every query checks."""

    MAX_CELLS = 256

    def __init__(self, cell: float):
        self.cell = max(float(cell), 1e-3)
        self._cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._big: list[int] = []
        self._items: list = []
        self._boxes: list[tuple[float, float, float, float]] = []

    def _span(self, b: Sequence[float], pad: float = 0.0) -> tuple[int, int, int, int]:
        c = self.cell
        return (int(math.floor((b[0] - pad) / c)), int(math.floor((b[1] - pad) / c)),
                int(math.floor((b[2] + pad) / c)), int(math.floor((b[3] + pad) / c)))

    def add(self, item, bbox: Sequence[float]) -> None:
        k = len(self._items)
        self._items.append(item)
        self._boxes.append(tuple(bbox))
        i0, j0, i1, j1 = self._span(bbox)
        if (i1 - i0 + 1) * (j1 - j0 + 1) > self.MAX_CELLS:
            self._big.append(k)
            return
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self._cells[(i, j)].append(k)

    def near(self, bbox: Sequence[float], reach: float) -> list:
        i0, j0, i1, j1 = self._span(bbox, reach)
        found: set[int] = set(self._big)
        if (i1 - i0 + 1) * (j1 - j0 + 1) > 4 * self.MAX_CELLS:  # a huge query box: test every item
            found.update(range(len(self._items)))
        else:
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    found.update(self._cells.get((i, j), ()))
        return [self._items[k] for k in sorted(found) if bbox_dist(self._boxes[k], bbox) <= reach]

    def __len__(self) -> int:
        return len(self._items)


def bbox_inside(inner: Sequence[float], outer: Sequence[float], tol: float = 0.0) -> bool:
    return inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol


def find_loose_rects(prims: list[Prim], ids: Sequence[int], tol: float) -> list[list[int]]:
    """Axis-aligned rectangles assembled from four separate open segments whose
    four corners are clean L-corners (no third segment ends there)."""
    hs = [i for i in ids if prims[i].axis == "h"]
    vs = [i for i in ids if prims[i].axis == "v"]
    if not hs or not vs:
        return []
    q = max(tol, 1e-3)

    def key(x: float, y: float) -> tuple[int, int]:
        return (int(round(x / q)), int(round(y / q)))

    ends: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in hs + vs:
        p = prims[i]
        for x, y in p.endpoints():
            k = key(x, y)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    ends[(k[0] + dx, k[1] + dy)].append(i)

    def at(x: float, y: float) -> set[int]:
        return {i for i in ends.get(key(x, y), ()) if min(math.hypot(x - a, y - b) for a, b in prims[i].endpoints()) <= 1.5 * q}

    out: list[list[int]] = []
    used: set[int] = set()
    for h in hs:
        if h in used:
            continue
        p = prims[h]
        xa, xb = sorted((p.x0, p.x1))
        y = (p.y0 + p.y1) / 2
        left = [v for v in at(xa, y) if v in vs and v not in used]
        right = [v for v in at(xb, y) if v in vs and v not in used]
        for vl in left:
            pl = prims[vl]
            ya, yb = sorted((pl.y0, pl.y1))
            y_other = yb if abs(ya - y) <= 2 * tol else ya
            if abs(abs(y_other - y) - (yb - ya)) > 2 * tol:
                continue
            for vr in right:
                pr = prims[vr]
                ra, rb = sorted((pr.y0, pr.y1))
                if abs(ra - ya) > 2 * tol or abs(rb - yb) > 2 * tol:
                    continue
                tops = [t for t in at(xa, y_other) & at(xb, y_other) if t in hs and t != h and t not in used]
                tops = [t for t in tops if abs(min(prims[t].x0, prims[t].x1) - xa) <= 2 * tol and abs(max(prims[t].x0, prims[t].x1) - xb) <= 2 * tol]
                if not tops:
                    continue
                rect = [h, vl, vr, tops[0]]
                corners = [(xa, y), (xb, y), (xa, y_other), (xb, y_other)]
                if all(len({e for e in at(cx, cy) if e not in rect}) == 0 for cx, cy in corners):
                    out.append(rect)
                    used.update(rect)
                    break
            if h in used:
                break
    return out


def find_loose_polygons(prims: list[Prim], ids: Sequence[int], tol: float, max_size: float, max_edges: int = 64) -> list[list[int]]:
    """Small closed polygons assembled from separate straight segments (CAD exports
    that explode polylines: a valve triangle drawn as three lines)."""
    import networkx as nx

    q = max(tol, 1e-3)
    g = nx.MultiGraph()
    node_at: dict[tuple[int, int], list[tuple[int, float, float]]] = defaultdict(list)
    n_nodes = 0

    def node(x: float, y: float) -> int:
        nonlocal n_nodes
        k = (int(round(x / q)), int(round(y / q)))
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for n, nx_, ny_ in node_at.get((k[0] + dx, k[1] + dy), ()):
                    d = math.hypot(x - nx_, y - ny_)
                    if d <= q and (best is None or d < best[0]):
                        best = (d, n)
        if best is not None:
            return best[1]
        n = n_nodes
        n_nodes += 1
        node_at[k].append((n, x, y))
        return n

    for i in ids:
        p = prims[i]
        a, b = node(p.x0, p.y0), node(p.x1, p.y1)
        if a != b:
            g.add_edge(a, b, pid=i)
    out: list[list[int]] = []
    simple = nx.Graph()
    for a, b, d in g.edges(data=True):
        if not simple.has_edge(a, b):
            simple.add_edge(a, b, pid=d["pid"])
    for cyc in nx.cycle_basis(simple):
        if not (3 <= len(cyc) <= max_edges):
            continue
        pids = [simple.edges[cyc[k], cyc[(k + 1) % len(cyc)]]["pid"] for k in range(len(cyc))]
        bb = bbox_union(prims[i].bbox for i in pids)
        if max(bb[2] - bb[0], bb[3] - bb[1]) > max_size:
            continue
        out.append(pids)
    return out


A0_SHORT_SIDE = 2384.0  # points: pages larger than A0 are (almost always) drawings exported at a scale factor


def page_scale(page_size: Optional[tuple[float, float]]) -> float:
    """Size factor of a page beyond A0 (1.0 up to A0): the stroke-size windows that are absolute
    for ordinary sheets grow with it (a sheet exported at 5x or 6x of A1 / A2 size)."""
    if not page_size:
        return 1.0
    return max(1.0, min(page_size) / A0_SHORT_SIDE)


def estimate_unit(prims: list[Prim], ids: Sequence[int], tol: float, default: Optional[float] = None,
                  page_size: Optional[tuple[float, float]] = None) -> float:
    """Typical symbol size (points per symbol unit) from clusters of symbol-only strokes: the median
    size of the clusters between 5 and 80 points -- on a page larger than A0 between 5 and 80 times
    :func:`page_scale` -- or ``default`` (18 points times the page scale) with fewer than three."""
    s = page_scale(page_size)
    default = 18.0 * s if default is None else default
    core = [i for i in ids if prims[i].is_core and not prims[i].dashed]
    if not core:
        return default
    sub = PrimSet(prims, {})
    dsu = DSU(len(prims))
    for i, j in sub.pairs_within(core, tol):
        if prim_distance(prims[i], prims[j]) <= tol:
            dsu.union(i, j)
    sizes = []
    for members in dsu.groups(core).values():
        bb = bbox_union(prims[m].bbox for m in members)
        size = max(bb[2] - bb[0], bb[3] - bb[1])
        if all(prims[m].filled for m in members) and size < 6.0 * s:
            continue  # junction dots / arrowheads
        if 5.0 * s <= size <= 80.0 * s:
            sizes.append(size)
    if len(sizes) < 3:
        return default
    return float(np.median(sizes))
