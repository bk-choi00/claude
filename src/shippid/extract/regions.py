"""Sheet-furniture detection: drawing frame, title block / revision table, notes
and legend areas. These regions are excluded from symbol and pipe search; the
text inside them keeps a matching :class:`~shippid.model.TextRole`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .primitives import DSU, Prim, PrimSet, bbox_inside, bbox_union, point_seg_dist, prim_distance
from .textlines import TextLine, line_height, median

BBox4 = tuple[float, float, float, float]

TITLE_KEYWORDS = (
    "DWG", "DRAWING", "TITLE", "HULL", "SCALE", "REV", "DATE", "DRAWN", "CHECKED", "APPROVED", "APP'D", "CHK'D",
    "SHEET", "PROJECT", "SHIP", "OWNER", "YARD", "DESIGNED", "SHIPBUILDING", "HEAVY INDUSTRIES", "CLASS",
)
NOTES_RE = re.compile(r"^\s*(GENERAL\s+)?NOTES?\b\s*[:.]?", re.I)
LEGEND_RE = re.compile(r"\bLEGENDS?\b|^\s*SYMBOLS?\b|SYMBOL\s+LIST|^\s*ABBREVIATIONS?\b", re.I)


@dataclass
class Regions:
    page_bbox: BBox4
    drawing_area: BBox4
    outer_frame: Optional[BBox4] = None
    title_blocks: list[BBox4] = field(default_factory=list)
    tables: list[BBox4] = field(default_factory=list)  # other text tables (treated like notes)
    notes: list[BBox4] = field(default_factory=list)
    legends: list[BBox4] = field(default_factory=list)

    def kind_at(self, bbox: Sequence[float], tol: float = 1.0) -> Optional[str]:
        """'title_block' | 'notes' | 'legend' | 'table' | 'margin' | None for a box."""
        for b in self.title_blocks:
            if bbox_inside(bbox, b, tol):
                return "title_block"
        for b in self.legends:
            if bbox_inside(bbox, b, tol):
                return "legend"
        for b in self.notes:
            if bbox_inside(bbox, b, tol):
                return "notes"
        for b in self.tables:
            if bbox_inside(bbox, b, tol):
                return "table"
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        da = self.drawing_area
        if not (da[0] - tol <= cx <= da[2] + tol and da[1] - tol <= cy <= da[3] + tol):
            return "margin"
        return None

    def excluded(self, bbox: Sequence[float], tol: float = 1.0) -> bool:
        return self.kind_at(bbox, tol) is not None


def _keyword_hits(lines: Sequence[TextLine]) -> int:
    hits = 0
    for t in lines:
        up = t.text.upper()
        hits += sum(1 for k in TITLE_KEYWORDS if k in up)
    return hits


def _lines_in(lines: Sequence[TextLine], bb: Sequence[float], tol: float = 1.0) -> list[TextLine]:
    return [t for t in lines if bbox_inside(t.bbox, bb, tol)]


def detect_regions(ps: PrimSet, lines: Sequence[TextLine], width: float, height: float, tol: float, unit: float) -> Regions:
    prims = ps.prims
    page = (0.0, 0.0, width, height)
    segs = [p for p in prims if p.kind == "seg" and not p.curve]

    # ---- frame: lines spanning most of the page close to the borders
    long_h = [p for p in segs if p.axis == "h" and abs(p.x1 - p.x0) >= 0.8 * width]
    long_v = [p for p in segs if p.axis == "v" and abs(p.y1 - p.y0) >= 0.8 * height]
    tops = [p.y0 for p in long_h if p.y0 < 0.2 * height]
    bottoms = [p.y0 for p in long_h if p.y0 > 0.8 * height]
    lefts = [p.x0 for p in long_v if p.x0 < 0.2 * width]
    rights = [p.x0 for p in long_v if p.x0 > 0.8 * width]
    frame_ids = {p.id for p in long_h if p.y0 < 0.2 * height or p.y0 > 0.8 * height}
    frame_ids |= {p.id for p in long_v if p.x0 < 0.2 * width or p.x0 > 0.8 * width}
    inner = (
        max(lefts) if lefts else 0.0,
        max(tops) if tops else 0.0,
        min(rights) if rights else width,
        min(bottoms) if bottoms else height,
    )
    outer = None
    if lefts or tops or rights or bottoms:
        outer = (min(lefts) if lefts else 0.0, min(tops) if tops else 0.0,
                 max(rights) if rights else width, max(bottoms) if bottoms else height)
    regions = Regions(page_bbox=page, drawing_area=inner, outer_frame=outer)
    for p in prims:
        if p.id in frame_ids:
            p.role = "frame"
        elif regions.kind_at(p.bbox, tol) == "margin":
            p.role = "frame"

    # ---- tables: connected networks of axis-aligned strokes forming cells
    cand = [p.id for p in segs if p.role == "" and p.axis is not None and not p.dashed]
    frame_lines = [prims[i] for i in frame_ids]
    dsu = DSU(len(prims))
    for i, j in ps.pairs_within(cand, tol):
        if prim_distance(prims[i], prims[j]) <= tol:
            dsu.union(i, j)
    comps = dsu.groups(cand)
    foreign = [p for p in prims if p.role == "" and (p.kind == "circle" or p.axis is None)]
    da_area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    for members in comps.values():
        if len(members) < 4:
            continue
        bb = bbox_union(prims[m].bbox for m in members)
        area = (bb[2] - bb[0]) * (bb[3] - bb[1])
        if area <= 0 or area > 0.4 * da_area or min(bb[2] - bb[0], bb[3] - bb[1]) < 2 * unit * 0.5:
            continue
        inside = _lines_in(lines, bb, tol)
        if len(inside) < 3:
            continue
        mset = set(members)
        # grid score: segment ends that land on another member (or the frame)
        n_end = n_touch = 0
        for m in members:
            p = prims[m]
            for x, y in p.endpoints():
                n_end += 1
                ok = any(point_seg_dist(x, y, f.x0, f.y0, f.x1, f.y1) <= tol for f in frame_lines)
                if not ok:
                    for o in members:
                        if o != m:
                            q = prims[o]
                            if point_seg_dist(x, y, q.x0, q.y0, q.x1, q.y1) <= tol:
                                ok = True
                                break
                n_touch += ok
        grid = n_touch / max(1, n_end)
        if grid < 0.85:
            continue
        n_foreign = 0
        for f in foreign:
            if bbox_inside(f.bbox, bb, tol) and any(prim_distance(f, prims[m]) <= tol for m in members):
                n_foreign += 1
        if n_foreign > 2:
            continue
        attached = any(
            point_seg_dist(x, y, f.x0, f.y0, f.x1, f.y1) <= tol
            for m in members for x, y in prims[m].endpoints() for f in frame_lines
        )
        near_corner = bb[2] >= inner[2] - 3 * unit and bb[3] >= inner[3] - 3 * unit
        kw = _keyword_hits(inside)
        full_h = sum(1 for m in mset if prims[m].axis == "h" and abs(prims[m].x1 - prims[m].x0) >= 0.9 * (bb[2] - bb[0]))
        full_v = sum(1 for m in mset if prims[m].axis == "v" and abs(prims[m].y1 - prims[m].y0) >= 0.9 * (bb[3] - bb[1]))
        is_title = kw >= 2 or ((attached or near_corner) and kw >= 1)
        is_table = full_h + full_v >= 3 and len(inside) >= 6
        if not (is_title or is_table):
            continue
        if is_title:
            regions.title_blocks.append(bb)
        else:
            regions.tables.append(bb)
        for m in members:
            prims[m].role = "table"

    # ---- notes and legend blocks headed by a keyword line
    used: set[int] = set()
    for t in lines:
        if t.rotation != 0 or regions.kind_at(t.bbox, tol) is not None:
            continue
        if NOTES_RE.match(t.text) and len(t.text) <= 30:
            bb = _grow_block(t, lines, regions, column=True)
            regions.notes.append(_enclosing_box(ps, bb, t, da_area, tol) or bb)
            used.add(t.order)
        elif LEGEND_RE.search(t.text) and len(t.text) <= 40:
            bb = _grow_block(t, lines, regions, column=False, unit=unit)
            box_bb = _enclosing_box(ps, bb, t, da_area, tol)
            if box_bb is None:
                bb = (bb[0] - 3.0 * unit, bb[1], bb[2], bb[3] + 0.5 * unit)
            regions.legends.append(box_bb or bb)
    for p in prims:
        if p.role == "" and regions.kind_at(p.bbox, tol) in ("title_block", "notes", "legend", "table"):
            p.role = "excluded"
    return regions


def _grow_block(head: TextLine, lines: Sequence[TextLine], regions: Regions, column: bool, unit: float = 18.0) -> BBox4:
    """Heading plus the lines below it that form one block of text."""
    h = max(line_height(head), 1.0)
    members = [head]
    pool = [t for t in lines if t is not head and t.rotation == 0 and t.bbox[1] >= head.bbox[1] - 0.2 * h
            and regions.kind_at(t.bbox) is None]
    if column:
        pool = [t for t in pool if head.bbox[0] - 2 * h <= t.bbox[0] <= head.bbox[0] + 8 * h]
    else:
        pool = [t for t in pool if head.bbox[0] - 6 * unit <= t.bbox[0] <= head.bbox[2] + 20 * unit]
    pool.sort(key=lambda t: t.bbox[1])
    pitch = 2.2 * h if column else 3.5 * max(h, 0.8 * unit)
    bottom = head.bbox[3]
    for t in pool:
        if t.bbox[1] - bottom > pitch:
            break
        members.append(t)
        bottom = max(bottom, t.bbox[3])
    bb = bbox_union(t.bbox for t in members)
    pad = 0.3 * h
    return (bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)


def _enclosing_box(ps: PrimSet, bb: BBox4, head: TextLine, da_area: float, tol: float) -> Optional[BBox4]:
    """Smallest closed rectangle that contains the heading (a boxed notes/legend area)."""
    best = None
    for ring in ps.rings.values():
        if not ring.is_rect or ring.dashed:
            continue
        rb = ring.bbox
        area = (rb[2] - rb[0]) * (rb[3] - rb[1])
        if area > 0.35 * da_area or not bbox_inside(head.bbox, rb, tol):
            continue
        hb = head.bbox
        if area < 6 * max(1.0, (hb[2] - hb[0]) * (hb[3] - hb[1])):
            continue
        if best is None or area < best[0]:
            best = (area, rb)
    return best[1] if best else None


def mark_roles(prims: Sequence[Prim]) -> None:  # pragma: no cover - debugging helper
    for p in prims:
        if not p.role:
            p.role = "free"
