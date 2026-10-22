"""Lettering drawn as strokes with no text carrier at all: AutoCAD SHX text exported with
PDFSHX=0 (the glyphs are plain vector geometry, no hidden text and no annotation names the
string), or text exploded to lines by the CAD tool.

Left alone, every word becomes an unmatched symbol candidate (hundreds of ``UNKNOWN``
objects on a sheet), letters next to a pipe split it, and a tag written inside an instrument
bubble is grouped with the circle so the bubble no longer matches. This module finds rows of
small glyph-like stroke clusters of one height and marks their primitives ``text`` (like
:func:`shippid.extract.pipeline._mark_text_strokes` does for SHX text that has a carrier);
rows of identical single straight strokes (section hatching, tick fields) are marked
``hatch``. The string itself cannot be read from geometry here; the rows are reported in the
page meta (``stroke_lettering``) so they can be read from an image (Gemma) later.

Glyph cluster (one orientation; vertical text is handled by swapping the axes): strokes
touching each other (letters of a word may touch), not dashed, not filled, no circle; height
``H`` (across the reading direction) at most :data:`MAX_HEIGHT_UNITS` symbol units; every
stroke at most :data:`MAX_STROKE_H` x ``H``; not a round closed polyline (a tessellated
circle). Clusters on one band (same top and baseline, or a small mark such as '-' inside the
band) with a gap of at most :data:`MAX_GAP_H` x ``H`` form a row; a row of at least three
glyphs (x-projection runs) is lettering, and shorter rows directly above / below a lettering
row of the same height are its other text lines. Strokes connected to a pipe or a symbol are
in that cluster and never qualify.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .primitives import DSU, BoxGrid, Prim, PrimSet, bbox_union, prim_distance

BBox4 = tuple[float, float, float, float]

MAX_HEIGHT_UNITS = 0.5  # glyph height limit (symbol units): text is 2.5-3.5 mm, symbols 6-10 mm
MAX_STROKE_H = 1.6  # longest stroke of a glyph / glyph height
MAX_GAP_H = 1.2  # letter and word spacing / glyph height
MIN_HEIGHT = 0.4  # points
MIN_GLYPHS = 3  # glyphs (x-projection runs) of a row that is lettering on its own


@dataclass
class Lettering:
    """Stroke lettering found on a page (primitive ids, row bboxes in page space)."""

    text_ids: list[int] = field(default_factory=list)  # primitives of lettering rows (role 'text')
    hatch_ids: list[int] = field(default_factory=list)  # rows of identical single strokes (role 'hatch')
    rows: list[BBox4] = field(default_factory=list)
    hatch_rows: int = 0

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
    runs: int  # glyphs: projection runs along the reading direction

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
    if h < MIN_HEIGHT or h > max_h:
        return None
    ps = [prims[i] for i in c.ids]
    if any(p.kind == "circle" for p in ps) or any(p.length > MAX_STROKE_H * h for p in ps):
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
        for j in grid.near((g.lo, g.top, g.hi, g.bot), MAX_GAP_H * g.h):
            if j > k and _linked(g, glyphs[j]):
                dsu.union(k, j)
    return [[glyphs[k] for k in sorted(m)] for m in dsu.groups().values()]


def _row_box(row: list[_Glyph]) -> tuple[float, float, float, float]:
    return (min(g.lo for g in row), min(g.top for g in row), max(g.hi for g in row), max(g.bot for g in row))


def _row_height(row: list[_Glyph]) -> float:
    hs = sorted(g.h for g in row)
    return hs[-1] if len(hs) < 3 else hs[len(hs) // 2]


def _is_hatch_row(row: list[_Glyph]) -> bool:
    """Identical single straight strokes side by side (section hatching, tick marks)."""
    cs = [g.c for g in row]
    if len(cs) < MIN_GLYPHS or any(c.single_dir is None for c in cs):
        return False
    d0, l0 = cs[0].single_dir, cs[0].single_len
    return all(min(abs(c.single_dir - d0), 180.0 - abs(c.single_dir - d0)) <= 5.0 and
               abs(c.single_len - l0) <= 0.15 * max(l0, 1e-6) for c in cs)


def find_stroke_lettering(ps: PrimSet, tol: float, unit: float) -> Lettering:
    """Rows of stroke lettering among the free primitives (see the module docstring)."""
    prims = ps.prims
    ids = [p.id for p in prims if p.role == "" and not p.dashed and not p.filled]
    out = Lettering()
    if len(ids) < MIN_GLYPHS:
        return out
    touch = max(0.05, min(0.25 * tol, 0.6))
    clusters = _clusters(ps, ids, touch)
    max_h = MAX_HEIGHT_UNITS * unit
    text: set[int] = set()  # cluster indices
    hatch: set[int] = set()
    for vertical in (False, True):
        glyphs = [g for g in (_glyph(c, prims, vertical, max_h) for c in clusters) if g is not None]
        rows = _rows(glyphs)
        strong, weak = [], []
        for row in rows:
            if sum(g.runs for g in row) >= MIN_GLYPHS and sum(1 for g in row if g.h >= 0.6 * _row_height(row)) >= 1:
                if _is_hatch_row(row):
                    hatch.update(g.c.k for g in row)
                    out.hatch_rows += 1
                    continue
                strong.append(row)
            else:
                weak.append(row)
        # shorter lines of a multi-line label: right above / below a lettering row of the same height
        accepted = list(strong)
        changed = True
        while changed and weak:
            changed = False
            for row in list(weak):
                h, (lo, top, hi, bot) = _row_height(row), _row_box(row)
                for s in accepted:
                    hs, (slo, stop, shi, sbot) = _row_height(s), _row_box(s)
                    if not (0.7 <= h / max(hs, 1e-6) <= 1.43):
                        continue
                    gap = max(stop - bot, top - sbot)
                    if gap <= 1.0 * max(h, hs) and min(hi, shi) - max(lo, slo) > -0.5 * max(h, hs):
                        accepted.append(row)
                        weak.remove(row)
                        changed = True
                        break
        for row in accepted:
            if any(g.c.k in hatch for g in row):
                continue
            text.update(g.c.k for g in row)
            lo, top, hi, bot = _row_box(row)
            out.rows.append((top, lo, bot, hi) if vertical else (lo, top, hi, bot))
    for k in sorted(text):
        out.text_ids.extend(clusters[k].ids)
    for k in sorted(hatch - text):
        out.hatch_ids.extend(clusters[k].ids)
    return out


def mark_stroke_lettering(ps: PrimSet, tol: float, unit: float) -> Lettering:
    """:func:`find_stroke_lettering` and mark the primitives (``text`` / ``hatch``)."""
    let = find_stroke_lettering(ps, tol, unit)
    for i in let.text_ids:
        ps.prims[i].role = "text"
    for i in let.hatch_ids:
        ps.prims[i].role = "hatch"
    return let
