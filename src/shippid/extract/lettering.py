"""Lettering drawn as strokes with no text carrier at all: AutoCAD SHX text exported with
PDFSHX=0 (the glyphs are plain vector geometry, no hidden text and no annotation names the
string), or text exploded to lines by the CAD tool.

Left alone, every word becomes an unmatched symbol candidate (hundreds of ``UNKNOWN``
objects on a sheet), letters next to a pipe split it, and a tag written inside an instrument
bubble is grouped with the circle so the bubble no longer matches. This module finds rows of
small glyph-like stroke clusters of one height and gives their primitives the role
``lettering`` (out of symbol and pipe search, like the strokes of SHX text that has a carrier,
see :func:`shippid.extract.pipeline._mark_text_strokes`); rows of identical single straight
strokes (section hatching, tick fields) get the role ``hatch``. The string itself cannot be
read from geometry here; the rows are reported in the page meta (``stroke_lettering``) so they
can be read from an image (Gemma) later, and sheet-furniture detection counts them as text
lines (a title block drawn with stroke lettering is still a table).

Glyph cluster (one orientation; vertical text is handled by swapping the axes): strokes
touching each other (letters of a word may touch), not dashed, not filled, no circle; height
``H`` (across the reading direction) at most :data:`MAX_HEIGHT_UNITS` symbol units; every
stroke at most :data:`MAX_STROKE_H` x ``H``; not a round closed polyline (a tessellated
circle). Clusters on one band (same top and baseline, or a flat mark such as '-' inside the
band) with a gap of at most :data:`MAX_GAP_H` x ``H`` form a row; a row of at least three
glyphs (x-projection runs) is lettering, and shorter rows next to / above / below a lettering
row of the same height (or stacked in a column) are lettering too. Safeguards: strokes
connected to a pipe or a symbol outline are in that cluster and never qualify; rows inside a
closed outline less than :data:`OUTLINE_H` row heights across are a symbol's inside; and a page
must have at least :data:`MIN_PAGE_ROWS` rows (:data:`MIN_PAGE_GLYPHS` glyphs) -- a sheet whose
text is real text has none, except the insides of an unconnected symbol now and then.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .primitives import DSU, BoxGrid, Prim, PrimSet, bbox_inside, bbox_union, prim_distance

BBox4 = tuple[float, float, float, float]

MAX_HEIGHT_UNITS = 0.5  # glyph height limit (symbol units): text is 2.5-3.5 mm, symbols 6-10 mm
MAX_STROKE_H = 1.6  # longest stroke of a glyph / glyph height
MAX_GAP_H = 1.2  # letter and word spacing / glyph height
MIN_HEIGHT = 0.4  # points
MIN_GLYPHS = 3  # glyphs (x-projection runs) of a row that is lettering on its own
MIN_PAGE_ROWS = 8  # a page has stroke lettering when it has at least this many such rows ...
MIN_PAGE_GLYPHS = 40  # ... with this many glyphs in all
OUTLINE_H = 2.5  # a row inside a closed outline less than this many row heights across is a symbol's inside


@dataclass
class Lettering:
    """Stroke lettering found on a page (primitive ids, row bboxes in page space)."""

    text_ids: list[int] = field(default_factory=list)  # primitives of lettering rows (role 'lettering')
    hatch_ids: list[int] = field(default_factory=list)  # rows of identical single strokes (role 'hatch')
    rows: list[BBox4] = field(default_factory=list)
    row_rotation: list[int] = field(default_factory=list)  # 0 (horizontal) or 90 (vertical) per row
    row_height: list[float] = field(default_factory=list)  # glyph height per row
    hatch_rows: int = 0

    def __bool__(self) -> bool:
        return bool(self.text_ids or self.hatch_ids)

    def to_meta(self, max_boxes: int = 200) -> dict:
        return {"rows": len(self.rows), "strokes": len(self.text_ids), "hatch_rows": self.hatch_rows,
                "hatch_strokes": len(self.hatch_ids),
                "boxes": [[round(v, 2) for v in b] for b in self.rows[:max_boxes]],
                "boxes_truncated": len(self.rows) > max_boxes}


@dataclass
class _Cluster:
    k: int
    ids: list[int]
    bbox: BBox4
    single_dir: Optional[float]  # direction (degrees, mod 180) of a cluster that is one straight stroke
    single_len: float


@dataclass
class _Glyph:
    c: _Cluster
    lo: float  # along the reading direction
    hi: float
    top: float  # across (top < bottom in the rotated frame)
    bot: float
    runs: int  # glyphs: projection runs along the reading direction (0 for a mark)
    mark: bool = False  # flatter than MIN_HEIGHT ('-', '.', '_'): joins a row, never forms one

    @property
    def h(self) -> float:
        return self.bot - self.top


def _round_ring(prims: Sequence[Prim], bb: BBox4) -> bool:
    """A closed polyline whose vertices lie on a circle (a tessellated instrument bubble or disc)."""
    pts = [(p.x0, p.y0) for p in prims if p.closed and p.kind == "seg"]
    if len(pts) < 8:
        return False
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    rs = [math.hypot(x - cx, y - cy) for x, y in pts]
    mean = sum(rs) / len(rs)
    return mean > 0 and max(abs(r - mean) for r in rs) <= 0.1 * mean


def _clusters(ps: PrimSet, ids: Sequence[int], touch: float) -> list[_Cluster]:
    prims = ps.prims
    dsu = DSU(len(prims))
    for i, j in ps.pairs_within(ids, touch):
        if prim_distance(prims[i], prims[j]) <= touch:
            dsu.union(i, j)
    out = []
    for members in dsu.groups(ids).values():
        members.sort()
        bb = bbox_union(prims[m].bbox for m in members)
        p0 = prims[members[0]]
        single = len(members) == 1 and p0.kind == "seg" and not p0.curve
        d = math.degrees(math.atan2(p0.y1 - p0.y0, p0.x1 - p0.x0)) % 180.0 if single else None
        out.append(_Cluster(len(out), members, bb, d, p0.length if single else 0.0))
    return out


def _glyph(c: _Cluster, prims: Sequence[Prim], vertical: bool, max_h: float) -> Optional[_Glyph]:
    b = c.bbox
    lo, hi, top, bot = (b[1], b[3], b[0], b[2]) if vertical else (b[0], b[2], b[1], b[3])
    h = bot - top
    if h > max_h:
        return None
    ps = [prims[i] for i in c.ids]
    if any(p.kind == "circle" for p in ps):
        return None
    if h < MIN_HEIGHT:  # a dash, period or underscore
        if hi - lo > 0.5 * max_h:
            return None
        return _Glyph(c, lo, hi, top, bot, 0, mark=True)
    if any(p.length > MAX_STROKE_H * h for p in ps):
        return None
    if _round_ring(ps, b):
        return None
    iv = sorted(((p.y0, p.y1) if vertical else (p.x0, p.x1)) for p in ps)
    iv = [(min(a), max(a)) for a in iv]
    iv.sort()
    runs, end = 0, -math.inf
    for a, z in iv:
        if a > end + 0.06 * h:
            runs += 1
        end = max(end, z)
    if hi - lo > 1.6 * h * runs:
        return None
    return _Glyph(c, lo, hi, top, bot, runs)


def _linked(a: _Glyph, b: _Glyph) -> bool:
    if a.mark and b.mark:
        return False
    hm, hn = max(a.h, b.h), min(a.h, b.h)
    first, second = (a, b) if a.lo <= b.lo else (b, a)
    gap = second.lo - first.hi
    if gap < -0.15 * hm or gap > MAX_GAP_H * hm:
        return False
    if hn >= 0.6 * hm:
        return abs(a.top - b.top) <= 0.3 * hm and abs(a.bot - b.bot) <= 0.3 * hm
    small, big = (a, b) if a.h < b.h else (b, a)
    # a small mark inside the band of a full-height glyph: '-', '.', ',' ...
    return (small.hi - small.lo) <= 1.0 * big.h and small.top >= big.top - 0.1 * big.h and \
        small.bot <= big.bot + 0.1 * big.h


def _rows(glyphs: list[_Glyph]) -> list[list[_Glyph]]:
    if not glyphs:
        return []
    reach = MAX_GAP_H * max(g.h for g in glyphs)
    grid = BoxGrid(max(reach, 1.0))
    for k, g in enumerate(glyphs):
        grid.add(k, (g.lo, g.top, g.hi, g.bot))
    dsu = DSU(len(glyphs))
    for k, g in enumerate(glyphs):
        if g.mark:
            continue  # found from the glyphs next to it
        for j in grid.near((g.lo, g.top, g.hi, g.bot), MAX_GAP_H * g.h):
            if j != k and _linked(g, glyphs[j]):
                dsu.union(k, j)
    return [[glyphs[k] for k in sorted(m)] for m in dsu.groups().values()]


def _row_box(row: list[_Glyph]) -> tuple[float, float, float, float]:
    return (min(g.lo for g in row), min(g.top for g in row), max(g.hi for g in row), max(g.bot for g in row))


def _row_height(row: list[_Glyph]) -> float:
    hs = sorted(g.h for g in row if not g.mark) or [0.0]
    return hs[-1] if len(hs) < 3 else hs[len(hs) // 2]


def _is_hatch_row(row: list[_Glyph]) -> bool:
    """Identical single straight strokes side by side (section hatching, tick marks)."""
    cs = [g.c for g in row if not g.mark]
    if len(cs) < MIN_GLYPHS or any(c.single_dir is None for c in cs):
        return False
    d0, l0 = cs[0].single_dir, cs[0].single_len
    return all(min(abs(c.single_dir - d0), 180.0 - abs(c.single_dir - d0)) <= 5.0 and
               abs(c.single_len - l0) <= 0.15 * max(l0, 1e-6) for c in cs)


class LetteringFinder:
    """Stroke clusters of a page's free primitives, searched for lettering rows (:meth:`find`)."""

    def __init__(self, ps: PrimSet, tol: float):
        self.ps = ps
        ids = [p.id for p in ps.prims if p.role == "" and not p.dashed and not p.filled]
        self.clusters = _clusters(ps, ids, max(0.05, min(0.25 * tol, 0.6))) if len(ids) >= MIN_GLYPHS else []
        self._outlines: Optional[BoxGrid] = None

    def glyph_like_ids(self, max_height: float) -> list[int]:
        """Primitives of every cluster shaped like a glyph (in a row or not) up to ``max_height``."""
        prims = self.ps.prims
        return [i for c in self.clusters
                if _glyph(c, prims, False, max_height) is not None or _glyph(c, prims, True, max_height) is not None
                for i in c.ids]

    def _in_small_outline(self, row: list[_Glyph], vertical: bool) -> bool:
        """The row lies inside a closed outline (polygon, rectangle, circle) less than
        :data:`OUTLINE_H` row heights across: the inner strokes of a symbol body (the lamps of a BWTS
        unit, the bars of a filter) rather than a label written inside equipment or a bubble."""
        if self._outlines is None:
            self._outlines = BoxGrid(16.0)
            for ring in self.ps.rings.values():
                if ring.prim_ids and not ring.dashed:
                    self._outlines.add(ring.bbox, ring.bbox)
            for p in self.ps.prims:
                if p.kind == "circle" and not p.dashed:
                    self._outlines.add(p.bbox, p.bbox)
        lo, top, hi, bot = _row_box(row)
        box = (top, lo, bot, hi) if vertical else (lo, top, hi, bot)
        h = _row_height(row)
        area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
        for ob in self._outlines.near(box, 0.0):
            if not bbox_inside(box, ob, 0.5):
                continue
            if (ob[2] - ob[0]) * (ob[3] - ob[1]) <= 1.5 * area:
                continue  # the row's own glyph outline ('O', 'D')
            across = (ob[2] - ob[0]) if vertical else (ob[3] - ob[1])
            if across < OUTLINE_H * h:
                return True
        return False

    def find(self, max_height: float) -> Lettering:
        """Lettering rows whose glyphs are at most ``max_height`` points high."""
        prims, clusters = self.ps.prims, self.clusters
        out = Lettering()
        hatch: set[int] = set()  # cluster indices
        found = []
        for vertical in (False, True):
            glyphs = [g for g in (_glyph(c, prims, vertical, max_height) for c in clusters) if g is not None]
            strong, weak = [], []
            for row in _rows(glyphs):
                if sum(g.runs for g in row) < MIN_GLYPHS:
                    weak.append(row)
                elif self._in_small_outline(row, vertical):
                    continue
                elif _is_hatch_row(row):
                    hatch.update(g.c.k for g in row)
                    out.hatch_rows += 1
                else:
                    strong.append(row)
            found.append((vertical, strong, weak))
        # a page with stroke lettering has a lot of it (every tag, line number and note); a few isolated
        # "rows" on a sheet whose text is real text are symbol parts (legend glyphs, decorations)
        n_rows = sum(len(st) for _, st, _ in found)
        n_glyphs = sum(g.runs for _, st, _ in found for row in st for g in row)
        text: set[int] = set()
        if n_rows >= MIN_PAGE_ROWS and n_glyphs >= MIN_PAGE_GLYPHS:
            for vertical, strong, weak in found:
                extra = [r for r in _attached_lines(strong, weak) if not self._in_small_outline(r, vertical)]
                for row in strong + extra:
                    if any(g.c.k in hatch for g in row):
                        continue
                    text.update(g.c.k for g in row)
                    lo, top, hi, bot = _row_box(row)
                    out.rows.append((top, lo, bot, hi) if vertical else (lo, top, hi, bot))
                    out.row_rotation.append(90 if vertical else 0)
                    out.row_height.append(_row_height(row))
        for k in sorted(text):
            out.text_ids.extend(clusters[k].ids)
        for k in sorted(hatch - text):
            out.hatch_ids.extend(clusters[k].ids)
        return out


def _attached_lines(strong: list[list[_Glyph]], weak: list[list[_Glyph]]) -> list[list[_Glyph]]:
    """Short rows (one or two glyphs) that belong to the lettering: lines of a multi-line label
    right above / below a lettering row of the same height, items on the band of a lettering row
    (note numbers '1.', a unit after a value), and columns of three or more short rows of one
    height (table cells 'A', 'B', 'C'); transitively."""
    info = {id(r): (_row_height(r), _row_box(r)) for r in strong + weak}

    def same_h(h1: float, h2: float) -> bool:
        return 0.7 <= h1 / max(h2, 1e-6) <= 1.43

    def touches(row, s) -> bool:
        h, (lo, top, hi, bot) = info[id(row)]
        hs, (slo, stop, shi, sbot) = info[id(s)]
        if not same_h(h, hs):
            return False
        hm = max(h, hs)
        stacked = max(stop - bot, top - sbot) <= 1.0 * hm and min(hi, shi) - max(lo, slo) > -0.5 * hm
        on_band = abs(top - stop) <= 0.3 * hm and abs(bot - sbot) <= 0.3 * hm and max(slo - hi, lo - shi) <= 4.0 * hm
        return stacked or on_band

    out: list[list[_Glyph]] = []
    hmax = max((h for h, _ in info.values()), default=1.0)
    # columns of short rows: same height, left (or right) edges aligned, small gaps
    col = DSU(len(weak))
    wgrid = BoxGrid(max(3.6 * hmax, 1.0))
    for k, r in enumerate(weak):
        wgrid.add(k, info[id(r)][1])
    for a, ra_ in enumerate(weak):
        ha, (la, ta, ra, ba) = info[id(ra_)]
        for b in wgrid.near(info[id(ra_)][1], 2.5 * 1.43 * ha):
            if b <= a:
                continue
            hb, (lb, tb, rb, bb_) = info[id(weak[b])]
            hm = max(ha, hb)
            if same_h(ha, hb) and (abs(la - lb) <= 0.5 * hm or abs(ra - rb) <= 0.5 * hm) and \
                    max(tb - ba, ta - bb_) <= 2.5 * hm:
                col.union(a, b)
    taken: set[int] = set()
    for members in col.groups().values():
        if len(members) >= 3:
            for k in members:
                out.append(weak[k])
                taken.add(k)
    pending = [r for k, r in enumerate(weak) if k not in taken]
    agrid = BoxGrid(max(5.8 * hmax, 1.0))
    for r in strong + out:
        agrid.add(r, info[id(r)][1])
    changed = True
    while changed and pending:
        changed = False
        rest = []
        for row in pending:
            h, box = info[id(row)]
            if any(touches(row, s) for s in agrid.near(box, 4.0 * 1.43 * h)):
                agrid.add(row, box)
                out.append(row)
                changed = True
            else:
                rest.append(row)
        pending = rest
    return out


def find_stroke_lettering(ps: PrimSet, tol: float, max_height: float) -> Lettering:
    """Rows of stroke lettering among the free primitives (see the module docstring)."""
    return LetteringFinder(ps, tol).find(max_height)


def mark_lettering(ps: PrimSet, let: Lettering) -> None:
    """Give the primitives of ``let`` their roles (``lettering`` / ``hatch``)."""
    for i in let.text_ids:
        ps.prims[i].role = "lettering"
    for i in let.hatch_ids:
        ps.prims[i].role = "hatch"
