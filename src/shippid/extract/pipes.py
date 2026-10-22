"""Pipe-line vs symbol-stroke separation.

Pipes are drawn as long, axis-aligned, open strokes (possibly chained, dashed for
signal lines, broken by a gap or bridged by a small arc where they cross another
line). Symbol strokes are short, diagonal, curved, closed or filled. The rules
below are deliberately geometric and relative to the drawing's symbol unit so
they hold across CAD styles:

1. open axis-aligned strokes lying inside a closed outline (inner lines of
   plate coolers, separators, instrument bubbles...) belong to that symbol;
2. collinear pieces are merged (touching pieces, crossing gaps, hop arcs,
   exploded dash patterns) into :class:`MLine` s;
3. merged lines longer than ``seed_units`` symbol units are pipes;
4. shorter lines that continue a pipe (and do not start inside a symbol body)
   are pipes, iterated to a fixed point (instrument taps, final legs);
5. short lines bridging two symbols outline-to-outline are pipe stubs;
6. small isolated filled dots on >= 3 pipe arms are junction dots, small filled
   triangles on a pipe are flow arrows.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import shapely

from .primitives import DSU, Prim, PrimSet, Ring, bbox_union, find_loose_rects, point_seg_dist, prim_distance, seg_param

BBox4 = tuple[float, float, float, float]


@dataclass
class MLine:
    """A maximal straight line assembled from collinear stroke pieces."""

    id: int
    axis: str  # 'h' | 'v' | 'd' (diagonal)
    x0: float
    y0: float
    x1: float
    y1: float
    prim_ids: list[int] = field(default_factory=list)
    dashed: bool = False
    width: float = 0.0
    color: Optional[tuple[float, float, float]] = None
    crossings: list[tuple[float, float]] = field(default_factory=list)  # gap / hop crossing points
    pipe: bool = False

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def bbox(self) -> BBox4:
        return (min(self.x0, self.x1), min(self.y0, self.y1), max(self.x0, self.x1), max(self.y0, self.y1))

    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.x0, self.y0), (self.x1, self.y1)

    def dist(self, x: float, y: float) -> float:
        return point_seg_dist(x, y, self.x0, self.y0, self.x1, self.y1)


@dataclass
class Dot:
    x: float
    y: float
    r: float
    prim_ids: list[int]


@dataclass
class Arrow:
    x: float
    y: float
    dx: float  # unit direction the arrow points to
    dy: float
    line_id: int
    prim_ids: list[int]


@dataclass
class Separation:
    lines: list[MLine]
    dots: list[Dot]
    arrows: list[Arrow]
    symbol_ids: list[int]  # prims left for symbol grouping
    core_boxes: list[BBox4]  # bboxes of touching clusters of symbol-only strokes
    unit: float

    @property
    def pipe_lines(self) -> list[MLine]:
        return [m for m in self.lines if m.pipe]


def _majority_color(prims: list[Prim]):
    cnt: dict = defaultdict(float)
    for p in prims:
        cnt[p.color] += p.length
    return max(cnt, key=cnt.get) if cnt else None


def separate(ps: PrimSet, unit: float, tol: float, seed_units: float = 1.2) -> Separation:
    prims = ps.prims
    touch = max(0.6, min(tol, 0.06 * unit))
    work = [p.id for p in prims if p.role == ""]

    # ---- dashed closed outlines are package / area boundaries, not symbols or pipes
    for ring in ps.rings.values():
        if ring.dashed:
            for pid in ring.prim_ids:
                if prims[pid].role == "":
                    prims[pid].role = "boundary"
    for p in prims:
        if p.role == "" and p.dashed and (p.kind == "circle" or p.curve):
            p.role = "boundary"

    # ---- rectangles assembled from four loose segments (tanks drawn line by line)
    loose = [i for i in work if prims[i].role == "" and prims[i].kind == "seg" and not prims[i].closed
             and not prims[i].curve and not prims[i].dashed and prims[i].axis is not None]
    for rect in find_loose_rects(prims, loose, touch):
        rid = max(ps.rings, default=-1) + 1
        pts = []
        xs = sorted({round(c, 6) for i in rect for c in (prims[i].x0, prims[i].x1)})
        ys = sorted({round(c, 6) for i in rect for c in (prims[i].y0, prims[i].y1)})
        x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        ps.rings[rid] = Ring(rid, list(rect), pts, synthetic=True)
        for i in rect:
            prims[i].closed = True
            prims[i].ring = rid

    # ---- containers: closed outlines and circles that can hold inner symbol lines
    cont_geoms = []
    cont_ids = []
    for ring in ps.rings.values():
        if ring.dashed:
            continue
        if any(prims[i].role not in ("",) for i in ring.prim_ids):
            continue
        cont_geoms.append(ring.polygon)
        cont_ids.append(("r", ring.id))
    for p in prims:
        if p.role == "" and p.kind == "circle" and p.r > 0.15 * unit:
            cont_geoms.append(shapely.Point(p.cx, p.cy).buffer(p.r, quad_segs=8))
            cont_ids.append(("c", p.id))
    open_axis = [i for i in work if prims[i].role == "" and prims[i].kind == "seg" and not prims[i].closed
                 and not prims[i].curve and prims[i].axis is not None]
    if cont_geoms and open_axis:
        tree = shapely.STRtree(cont_geoms)
        segs = shapely.linestrings([[(prims[i].x0, prims[i].y0), (prims[i].x1, prims[i].y1)] for i in open_axis])
        si, ci = tree.query(segs, predicate="dwithin", distance=touch)
        for s, c in zip(si.tolist(), ci.tolist()):
            p = prims[open_axis[s]]
            if p.role != "":
                continue
            g = cont_geoms[c]
            if shapely.distance(g, shapely.Point(p.x0, p.y0)) <= touch and shapely.distance(g, shapely.Point(p.x1, p.y1)) <= touch:
                p.role = "inner"

    # ---- core clusters (symbol-only strokes), used to know where symbol bodies are
    core = [i for i in work if prims[i].role == "" and prims[i].is_core and not prims[i].dashed]
    dsu = DSU(len(prims))
    for i, j in ps.pairs_within(core, touch):
        if prim_distance(prims[i], prims[j]) <= touch:
            dsu.union(i, j)
    core_boxes = [bbox_union(prims[m].bbox for m in g) for g in dsu.groups(core).values()]
    core_tree = shapely.STRtree([shapely.box(*b) for b in core_boxes]) if core_boxes else None

    def inside_core(x: float, y: float) -> bool:
        if core_tree is None:
            return False
        for k in core_tree.query(shapely.Point(x, y)).tolist():
            b = core_boxes[k]
            if b[0] + touch < x < b[2] - touch and b[1] + touch < y < b[3] - touch:
                return True
        return False

    # ---- hop arcs: open curve chains bridging two collinear line ends
    bridges: list[tuple[str, float, float, float, list[int], tuple[float, float]]] = []
    chains: list[list[int]] = []
    cur: list[int] = []
    for p in prims:
        if p.role == "" and p.kind == "seg" and p.curve and not p.closed:
            if cur and (prims[cur[-1]].path, prims[cur[-1]].sub) == (p.path, p.sub) and \
                    math.hypot(prims[cur[-1]].x1 - p.x0, prims[cur[-1]].y1 - p.y0) <= 1e-3:
                cur.append(p.id)
                continue
            if cur:
                chains.append(cur)
            cur = [p.id]
    if cur:
        chains.append(cur)
    line_ends: dict[tuple[int, int], list[int]] = defaultdict(list)
    q = max(touch, 1e-3)
    for i in open_axis:
        p = prims[i]
        if p.role != "":
            continue
        for x, y in p.endpoints():
            line_ends[(int(round(x / q)), int(round(y / q)))].append(i)

    def ends_at(x: float, y: float) -> list[int]:
        k = (int(round(x / q)), int(round(y / q)))
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out.extend(line_ends.get((k[0] + dx, k[1] + dy), ()))
        return [i for i in out if min(math.hypot(x - a, y - b) for a, b in prims[i].endpoints()) <= touch]

    for ch in chains:
        a = (prims[ch[0]].x0, prims[ch[0]].y0)
        b = (prims[ch[-1]].x1, prims[ch[-1]].y1)
        chord = math.dist(a, b)
        if not (0.05 * unit <= chord <= 3.0 * unit):
            continue
        dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
        axis = "h" if dy <= 0.05 * chord else ("v" if dx <= 0.05 * chord else None)
        if axis is None:
            continue
        la = [i for i in ends_at(*a) if prims[i].axis == axis]
        lb = [i for i in ends_at(*b) if prims[i].axis == axis]
        if not la or not lb:
            continue
        bulge = max(point_seg_dist(prims[k].x1, prims[k].y1, a[0], a[1], b[0], b[1]) for k in ch)
        if bulge > 1.2 * chord or bulge < 0.1 * chord:
            continue
        for k in ch:
            prims[k].role = "hop"
        coord = (a[1] + b[1]) / 2 if axis == "h" else (a[0] + b[0]) / 2
        lo, hi = (min(a[0], b[0]), max(a[0], b[0])) if axis == "h" else (min(a[1], b[1]), max(a[1], b[1]))
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        bridges.append((axis, coord, lo, hi, list(ch), mid))

    # ---- collinear merge of axis-aligned open strokes
    lines: list[MLine] = []
    merge_ids = [i for i in work if prims[i].role == "" and prims[i].kind == "seg" and not prims[i].closed
                 and not prims[i].curve and prims[i].axis is not None]
    for dashed in (False, True):
        for axis in ("h", "v"):
            pieces = []  # (coord, lo, hi, prim_ids, crossing)
            for i in merge_ids:
                p = prims[i]
                if p.axis != axis or p.dashed != dashed:
                    continue
                if axis == "h":
                    pieces.append(((p.y0 + p.y1) / 2, min(p.x0, p.x1), max(p.x0, p.x1), [i], None))
                else:
                    pieces.append(((p.x0 + p.x1) / 2, min(p.y0, p.y1), max(p.y0, p.y1), [i], None))
            if not dashed:
                for bax, coord, lo, hi, ch, mid in bridges:
                    if bax == axis:
                        pieces.append((coord, lo, hi, list(ch), mid))
            lines.extend(_merge_axis(pieces, axis, dashed, prims, unit, touch, len(lines)))
    # perpendicular crossing gaps need the other axis: done after all lines exist
    lines = _bridge_crossing_gaps(lines, prims, unit, tol, touch)
    for k, m in enumerate(lines):
        m.id = k

    # ---- seeds
    for m in lines:
        L = m.length
        if (m.dashed and L >= 0.6 * unit) or (not m.dashed and L >= seed_units * unit):
            m.pipe = True

    # ---- extensions (fixed point)
    def touching_pipe(x: float, y: float, me: MLine) -> bool:
        for o in lines:
            if o.pipe and o is not me and o.dashed == me.dashed:
                b = o.bbox
                if b[0] - touch <= x <= b[2] + touch and b[1] - touch <= y <= b[3] + touch and o.dist(x, y) <= touch:
                    return True
        return False

    changed = True
    while changed:
        changed = False
        for m in lines:
            if m.pipe:
                continue
            for x, y in m.endpoints():
                if not inside_core(x, y) and touching_pipe(x, y, m):
                    m.pipe = True
                    changed = True
                    break

    # ---- stubs between two symbol outlines
    others = [i for i in work if prims[i].role in ("", "inner")]
    other_tree = shapely.STRtree([shapely.box(*prims[i].bbox).buffer(touch) for i in others]) if others else None
    line_prims = {pid: m for m in lines for pid in m.prim_ids}

    def contacts(x: float, y: float, m: MLine) -> bool:
        if other_tree is None:
            return False
        own = set(m.prim_ids)
        for k in other_tree.query(shapely.Point(x, y)).tolist():
            pid = others[k]
            if pid in own:
                continue
            om = line_prims.get(pid)
            if om is not None and om.pipe and om.dashed != m.dashed:
                continue
            from .primitives import point_prim_dist

            if point_prim_dist(x, y, prims[pid]) <= touch:
                return True
        return False

    for m in lines:
        if m.pipe or m.dashed or m.length < max(1.5, 0.15 * unit):
            continue
        (xa, ya), (xb, yb) = m.endpoints()
        if inside_core(xa, ya) or inside_core(xb, yb):
            continue
        if contacts(xa, ya, m) and contacts(xb, yb, m):
            m.pipe = True

    pipe_lines = [m for m in lines if m.pipe]
    for m in lines:
        role = "pipe" if m.pipe else ""
        for pid in m.prim_ids:
            if prims[pid].role == "":
                prims[pid].role = role
            elif prims[pid].role == "hop" and not m.pipe:
                prims[pid].role = ""

    # ---- junction dots
    dots: list[Dot] = []
    sym_now = [p.id for p in prims if p.role in ("", "inner")]
    sym_tree = shapely.STRtree([shapely.box(*prims[i].bbox) for i in sym_now]) if sym_now else None
    filled_shapes: list[tuple[float, float, float, list[int]]] = []
    for p in prims:
        if p.role == "" and p.kind == "circle" and p.filled and p.r <= 0.3 * unit:
            filled_shapes.append((p.cx, p.cy, p.r, [p.id]))
    for ring in ps.rings.values():
        if ring.filled and all(prims[i].role == "" for i in ring.prim_ids):
            bb = ring.bbox
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            if max(w, h) <= 0.6 * unit and len(ring.pts) >= 6 and 0.75 <= (w / h if h else 0) <= 1.33:
                filled_shapes.append(((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, max(w, h) / 2, list(ring.prim_ids)))
    for cx, cy, r, pids in filled_shapes:
        arms = 0
        dirs = set()
        for m in pipe_lines:
            if m.dashed:
                continue
            d = m.dist(cx, cy)
            if d > max(touch, 0.5 * r):
                continue
            t = seg_param(cx, cy, m.x0, m.y0, m.x1, m.y1)
            L = m.length
            e = (r + touch) / max(L, 1e-6)
            if t > e:
                arms += 1
                dirs.add((m.axis, -1))
            if t < 1 - e:
                arms += 1
                dirs.add((m.axis, 1))
        if arms < 2:
            continue
        own = set(pids)
        isolated = True
        if sym_tree is not None:
            probe = shapely.Point(cx, cy).buffer(r + 0.15 * unit)
            for k in sym_tree.query(probe).tolist():
                pid = sym_now[k]
                if pid in own:
                    continue
                if prim_distance_point(prims[pid], cx, cy) <= r + 0.15 * unit:
                    isolated = False
                    break
        if not isolated:
            continue
        if arms >= 3 or len({a for a, _ in dirs}) == 2:
            dots.append(Dot(cx, cy, r, pids))
            for pid in pids:
                prims[pid].role = "dot"

    # ---- flow arrows: small filled triangles sitting on a pipe line
    arrows: list[Arrow] = []
    for ring in ps.rings.values():
        if not ring.filled or len(ring.pts) != 3 or any(prims[i].role != "" for i in ring.prim_ids):
            continue
        bb = ring.bbox
        size = max(bb[2] - bb[0], bb[3] - bb[1])
        if size > 0.6 * unit:
            continue
        pts = ring.pts
        sides = [math.dist(pts[i], pts[(i + 1) % 3]) for i in range(3)]
        k = int(np.argmin(sides))  # base = shortest side for a slim arrow; apex opposite
        base_mid = ((pts[k][0] + pts[(k + 1) % 3][0]) / 2, (pts[k][1] + pts[(k + 1) % 3][1]) / 2)
        apex = pts[(k + 2) % 3]
        dx, dy = apex[0] - base_mid[0], apex[1] - base_mid[1]
        n = math.hypot(dx, dy)
        if n <= 1e-6:
            continue
        dx, dy = dx / n, dy / n
        cxy = ((apex[0] + base_mid[0]) / 2, (apex[1] + base_mid[1]) / 2)
        host = None
        for m in pipe_lines:
            if m.dashed or m.dist(*cxy) > max(touch, 0.25 * size):
                continue
            ux, uy = (m.x1 - m.x0) / max(m.length, 1e-6), (m.y1 - m.y0) / max(m.length, 1e-6)
            if abs(ux * dx + uy * dy) >= math.cos(math.radians(20)):
                host = m
                break
        if host is None:
            continue
        own = set(ring.prim_ids)
        probe = shapely.box(*bb).buffer(0.15 * unit)
        if sym_tree is not None and any(sym_now[k2] not in own and prims[sym_now[k2]].role == "" and
                                        prim_distance(prims[sym_now[k2]], prims[ring.prim_ids[0]]) <= 0.15 * unit
                                        for k2 in sym_tree.query(probe).tolist()):
            continue
        arrows.append(Arrow(cxy[0], cxy[1], dx, dy, host.id, list(ring.prim_ids)))
        for pid in ring.prim_ids:
            prims[pid].role = "arrow"

    symbol_ids = []
    for p in prims:
        if p.role in ("", "inner"):
            p.role = "symbol"
            symbol_ids.append(p.id)
    return Separation(lines, dots, arrows, symbol_ids, core_boxes, unit)


def prim_distance_point(p: Prim, x: float, y: float) -> float:
    from .primitives import point_prim_dist

    return point_prim_dist(x, y, p)


def _merge_axis(pieces, axis: str, dashed: bool, prims: list[Prim], unit: float, touch: float, start_id: int) -> list[MLine]:
    """Merge collinear pieces of one axis: touching/overlapping pieces and
    exploded dash patterns (many short pieces with small regular gaps)."""
    if not pieces:
        return []
    pieces.sort(key=lambda p: p[0])
    clusters: list[list] = [[pieces[0]]]
    for pc in pieces[1:]:
        if pc[0] - clusters[-1][-1][0] <= min(0.6, touch):
            clusters[-1].append(pc)
        else:
            clusters.append([pc])
    out: list[MLine] = []
    for cl in clusters:
        coord = float(np.median([c[0] for c in cl]))
        cl.sort(key=lambda p: p[1])
        runs: list[list] = []  # [lo, hi, prim_ids, crossings, n_pieces, gaps]
        for _, lo, hi, ids, cross in cl:
            if runs and lo <= runs[-1][1] + max(0.3, 0.5 * touch):
                r = runs[-1]
                r[1] = max(r[1], hi)
                r[2].extend(ids)
                if cross is not None:
                    r[3].append(cross)
                r[4] += 1
            else:
                runs.append([lo, hi, list(ids), [cross] if cross is not None else [], 1, []])
        # exploded dashes: >= 4 consecutive short runs separated by small similar gaps
        merged: list[list] = []
        i = 0
        while i < len(runs):
            j = i
            gmax = min(0.4 * unit, 6.0)
            while j + 1 < len(runs) and 0 < runs[j + 1][0] - runs[j][1] <= gmax and \
                    (runs[j + 1][1] - runs[j + 1][0]) <= 1.5 * unit and (runs[j][1] - runs[j][0]) <= 1.5 * unit:
                j += 1
            if j - i + 1 >= 4:
                lens = [runs[k][1] - runs[k][0] for k in range(i, j + 1)]
                inner_lens = lens[1:-1] or lens
                if max(inner_lens) <= 3 * max(min(inner_lens), 1e-6):
                    r = [runs[i][0], runs[j][1], [pid for k in range(i, j + 1) for pid in runs[k][2]],
                         [c for k in range(i, j + 1) for c in runs[k][3]], j - i + 1, ["dash"]]
                    merged.append(r)
                    i = j + 1
                    continue
            merged.append(runs[i])
            i += 1
        for lo, hi, ids, cross, n, flag in merged:
            ps = [prims[k] for k in ids]
            m = MLine(start_id + len(out), axis, 0, 0, 0, 0, ids, dashed or bool(flag),
                      max((p.width for p in ps), default=0.0), _majority_color(ps), list(cross))
            if axis == "h":
                m.x0, m.y0, m.x1, m.y1 = lo, coord, hi, coord
            else:
                m.x0, m.y0, m.x1, m.y1 = coord, lo, coord, hi
            out.append(m)
    return out


def _bridge_crossing_gaps(lines: list[MLine], prims: list[Prim], unit: float, tol: float, touch: float) -> list[MLine]:
    """Join two collinear lines separated by a small gap through which a
    perpendicular line passes (the 'gap' convention for crossings)."""
    gap_max = max(3.0 * tol, 0.45 * unit)
    by_key: dict[tuple[str, bool], list[MLine]] = defaultdict(list)
    for m in lines:
        by_key[(m.axis, m.dashed)].append(m)
    result: list[MLine] = []
    for (axis, dashed), ms in by_key.items():
        perp = [o for o in lines if o.axis != axis and o.axis in ("h", "v")]
        coord_of = (lambda m: m.y0) if axis == "h" else (lambda m: m.x0)
        lo_of = (lambda m: min(m.x0, m.x1)) if axis == "h" else (lambda m: min(m.y0, m.y1))
        hi_of = (lambda m: max(m.x0, m.x1)) if axis == "h" else (lambda m: max(m.y0, m.y1))
        ms.sort(key=lambda m: (round(coord_of(m) / 0.6), lo_of(m)))
        out: list[MLine] = []
        for m in ms:
            if out:
                prev = out[-1]
                same = abs(coord_of(prev) - coord_of(m)) <= min(0.6, touch)
                gap = lo_of(m) - hi_of(prev)
                if same and 0 < gap <= gap_max:
                    c = coord_of(m)
                    g0, g1 = hi_of(prev), lo_of(m)
                    crossing = None
                    for o in perp:
                        oc = o.x0 if axis == "h" else o.y0
                        olo, ohi = (min(o.y0, o.y1), max(o.y0, o.y1)) if axis == "h" else (min(o.x0, o.x1), max(o.x0, o.x1))
                        if g0 + 0.15 * gap <= oc <= g1 - 0.15 * gap and olo < c - touch and ohi > c + touch:
                            crossing = (oc, c) if axis == "h" else (c, oc)
                            break
                    if crossing is not None:
                        if axis == "h":
                            prev.x0, prev.x1 = min(prev.x0, prev.x1), max(m.x0, m.x1)
                        else:
                            prev.y0, prev.y1 = min(prev.y0, prev.y1), max(m.y0, m.y1)
                        prev.prim_ids.extend(m.prim_ids)
                        prev.crossings.append(crossing)
                        prev.crossings.extend(m.crossings)
                        prev.width = max(prev.width, m.width)
                        continue
            out.append(m)
        result.extend(out)
    return result
