"""Legend learning: a shipyard's own symbol conventions, read from the legend on the drawing.

Shipyards draw valves, pumps and instruments their own way and explain the convention in a
LEGEND block (on the sheet, or on a separate legend sheet of the document): one row per symbol,
a glyph and its description. This module turns those rows into templates for the matcher of
:mod:`shippid.extract.symbols_match`, for that document only:

1. :func:`find_legends` -- legend regions of a page (heading ``LEGEND`` / ``SYMBOL LEGEND`` /
   ``SYMBOLS`` / ``범례``, boxed, unboxed or a table cell) and, per row, the glyph primitives and
   the description text (glyph and description side by side, a description wrapped over several
   lines, two or more columns, gridded tables);
2. :func:`map_description` -- bilingual keyword matcher from the description to a
   :class:`~shippid.model.SymbolClass` and an :class:`~shippid.model.ActuatorType`
   (``CHECK VALVE`` / ``NON-RETURN`` / ``NRV`` / ``S.D.N.R`` / ``체크 밸브`` ...). Rows that map to
   no class are kept with their description (:attr:`LegendBook.unmapped`) for Gemma;
3. :meth:`LegendBook.learn` -- templates from the glyphs, in the local frame of
   :class:`shippid.symbols.Placement` (flow axis = local x through the glyph's outermost ends at
   the axis height, origin on the axis midway between them; scale such that the glyph has the
   size of the built-in symbol of its class; actuator decorations of the built-in library found
   on a valve glyph are taken off so every actuator can be matched on the learned body; pipe
   stubs drawn on both sides of an inline glyph are taken off; ports where the class's built-in
   ports are, on the glyph's ends, free stub ends or outermost ink). A glyph that is exactly the
   built-in symbol of its class only confirms it (no template, matches are labelled); a glyph
   that is a built-in symbol of *another* class or a new drawing becomes a template.

The extractor (:func:`shippid.extract.pipeline.extract_pdf`) matches every candidate against
the built-in library extended by the learned templates (:meth:`TemplateLibrary.extended`, the
learned variants first) and :func:`rerank` prefers a learned template over a built-in one that
is not better by more than :data:`LEGEND_TIE`. A learned template whose stub was stripped
because it merges into the pipe (a yard pump drawn as a circle with a tangential discharge
stub) only stands when that pipe is really there, so a plain circle with a radial instrument tap
stays an instrument. Matches record ``attributes["template"] = "legend:<description>"``; legend
regions are excluded from the objects.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
from scipy.spatial import cKDTree

from ..config import Config
from ..model import ActuatorType, Category, SymbolClass
from ..pdf.reader import PageContent, read_page
from ..symbols import ACTUATOR_PRIMS, Circle, Line, Poly, PortSpec, Primitive, Rect, SymbolSpec, get_symbol, sample_outline
from .lettering import Lettering
from .primitives import DSU, PrimSet, bbox_inside, bbox_union, prim_distance
from .regions import LEGEND_RE, NOTES_RE, Regions, _enclosing_box, _grow_block, detect_regions
from .symbols_match import (
    INK_TOL,
    N_SAMPLES,
    CandidateShape,
    MatchResult,
    TemplateLibrary,
    _fill_samples,
    get_library,
    match_candidate,
)
from .textlines import TextLine, build_lines, line_height

log = logging.getLogger(__name__)

BBox4 = tuple[float, float, float, float]
S = SymbolClass
A = ActuatorType

HEADING_RE = re.compile(LEGEND_RE.pattern + r"|범\s*례", re.I)
_PAGE_HINT_RE = re.compile(r"LEGEND|SYMBOL|범례", re.I)

LEGEND_TIE = 0.01  # a learned template within this score of the best built-in one is preferred
NO_STUB_FACTOR = 0.9  # score factor of a stripped-stub template whose stub is not drawn as pipe
EXACT_SCORE = 0.97  # a glyph matching a built-in symbol this well (no ink penalty) *is* that symbol
UNKNOWN_ACCEPT = 0.62  # unmatched objects resembling an unmapped legend glyph this well are annotated

# --------------------------------------------------------------------------- description mapping

# (class, keywords). English keywords match whole words of the normalised description (upper case,
# dots removed, other punctuation -> space); Hangul keywords match the description without spaces.
# The longest matching keyword wins (Hangul characters count double); ties go to the earlier class.
CLASS_KEYWORDS: tuple[tuple[SymbolClass, tuple[str, ...]], ...] = (
    (S.SDNR_VALVE, ("SCREW DOWN NON RETURN", "SCREW DOWN CHECK", "SCREW DOWN NRV", "SDNR", "SDNRV", "SDNRV",
                    "STOP CHECK", "STOP NON RETURN", "스크류다운체크", "스크류다운논리턴", "스톱체크")),
    (S.CHECK_VALVE, ("CHECK VALVE", "CHECK", "NON RETURN", "NRV", "SWING CHECK", "LIFT CHECK", "체크밸브", "체크",
                     "역지밸브", "역지", "논리턴")),
    (S.GATE_VALVE, ("GATE VALVE", "GATE", "SLUICE VALVE", "게이트밸브", "게이트", "슬루스밸브")),
    (S.GLOBE_VALVE, ("GLOBE VALVE", "GLOBE", "CONTROL VALVE", "글로브밸브", "글로브", "조절밸브", "제어밸브")),
    (S.BALL_VALVE, ("BALL VALVE", "BALL", "볼밸브")),
    (S.BUTTERFLY_VALVE, ("BUTTERFLY VALVE", "BUTTERFLY", "BFV", "BF VALVE", "버터플라이", "나비밸브")),
    (S.THREE_WAY_VALVE, ("THREE WAY", "3 WAY", "3WAY", "3방향", "삼방", "3방")),
    (S.SAFETY_VALVE, ("SAFETY VALVE", "RELIEF VALVE", "SAFETY", "RELIEF", "PSV", "안전밸브", "릴리프", "안전")),
    (S.PUMP_SCREW, ("SCREW PUMP", "GEAR PUMP", "SCREW GEAR PUMP", "POSITIVE DISPLACEMENT PUMP", "스크류펌프", "기어펌프")),
    (S.PUMP_CENTRIFUGAL, ("CENTRIFUGAL PUMP", "CENT PUMP", "PUMP", "원심펌프", "펌프")),
    (S.PURIFIER, ("PURIFIER", "SEPARATOR", "CENTRIFUGAL SEPARATOR", "청정기")),
    (S.COMPRESSOR, ("COMPRESSOR", "압축기")),
    (S.AIR_RECEIVER, ("AIR RECEIVER", "RECEIVER", "AIR BOTTLE", "공기저장조", "공기탱크", "저장조")),
    (S.OILY_WATER_SEPARATOR, ("OILY WATER SEPARATOR", "OWS", "BILGE SEPARATOR", "유수분리기")),
    (S.BWTS_UNIT, ("BALLAST WATER TREATMENT", "BWTS", "BWMS", "UV REACTOR", "평형수처리")),
    (S.ENGINE, ("ENGINE", "CONSUMER", "주기관", "엔진", "발전기관")),
    (S.FLOW_METER, ("FLOW METER", "FLOWMETER", "유량계")),
    (S.STRAINER, ("STRAINER", "MUD BOX", "MUDBOX", "스트레이너", "머드박스")),
    (S.FILTER, ("FILTER", "필터", "여과기")),
    (S.DUPLEX_FILTER, ("DUPLEX FILTER", "DUPLEX", "AUTO BACKFLUSH", "BACKFLUSH", "BACK FLUSH", "AUTO FILTER",
                       "복식필터", "복식", "자동역세")),
    (S.PLATE_HEAT_EXCHANGER, ("PLATE HEAT EXCHANGER", "PLATE COOLER", "CENTRAL COOLER", "PLATE TYPE",
                              "판형열교환기", "판형")),
    (S.HEAT_EXCHANGER, ("HEAT EXCHANGER", "SHELL TUBE", "SHELL AND TUBE", "COOLER", "열교환기", "냉각기")),
    (S.HEATER, ("HEATER", "히터", "가열기")),
    (S.SEA_CHEST, ("SEA CHEST", "씨체스트", "해수흡입함")),
    (S.OVERBOARD, ("OVERBOARD", "SHIP SIDE", "선외")),
    (S.TANK, ("TANK", "TK", "탱크")),
    (S.REDUCER, ("REDUCER", "레듀서", "리듀서")),
    (S.SPECTACLE_FLANGE, ("SPECTACLE", "SPECTACLE BLIND", "SPECTACLE FLANGE", "스펙터클")),
    (S.FLEXIBLE_HOSE, ("FLEXIBLE", "EXPANSION JOINT", "BELLOWS", "COMPENSATOR", "플렉시블", "신축이음")),
    (S.INSTRUMENT_LOCAL, ("LOCAL INSTRUMENT", "LOCALLY MOUNTED", "FIELD INSTRUMENT", "LOCAL GAUGE", "GAUGE",
                          "THERMOMETER", "현장계기")),
    (S.INSTRUMENT_PANEL, ("PANEL INSTRUMENT", "REMOTE INSTRUMENT", "PANEL MOUNTED", "BOARD MOUNTED", "원격계기", "패널")),
    (S.INSTRUMENT_SHARED, ("SHARED DISPLAY", "DCS", "AMS", "ALARM MONITORING", "공유표시")),
    (S.OFF_PAGE_CONNECTOR, ("OFF PAGE", "OFFPAGE", "CONTINUATION", "도면연결", "연결표시")),
)
# explicit actuator words, in priority order (the first group with a match wins)
ACTUATOR_KEYWORDS: tuple[tuple[ActuatorType, tuple[str, ...]], ...] = (
    (A.QUICK_CLOSING, ("QUICK CLOSING", "QUICK CLOSE", "QUICK ACTING", "QCV", "EMERGENCY SHUT", "급속차단", "긴급차단",
                       "속폐")),
    (A.HYDRAULIC, ("HYDRAULIC", "HYDRAULICALLY", "HYD", "HYDR", "유압")),
    (A.ELECTRIC, ("MOTOR OPERATED", "MOTOR", "ELECTRIC", "ELECTRICAL", "MOV", "SOLENOID", "전동", "전기", "모터")),
    (A.DIAPHRAGM, ("PNEUMATIC", "PNEU", "DIAPHRAGM", "AIR OPERATED", "CONTROL VALVE", "공압", "다이어프램", "조절밸브",
                   "제어밸브")),
    (A.HANDWHEEL, ("HAND WHEEL", "HANDWHEEL", "MANUAL", "HAND OPERATED", "수동")),
)
# remote-operated valves of ship systems (ballast, bilge, cargo) are hydraulic unless stated otherwise
REMOTE_KEYWORDS: tuple[str, ...] = ("REMOTE", "REMOTELY", "원격")

_HANGUL_RE = re.compile(r"[가-힣]")


def normalize_description(text: str) -> str:
    """Upper case, dots removed (``S.D.N.R.`` -> ``SDNR``, ``HYD.`` -> ``HYD``), any other
    punctuation -> one space; Hangul kept."""
    s = unicodedata.normalize("NFKC", text).upper().replace(".", "")
    s = re.sub(r"[^0-9A-Z가-힣]+", " ", s)
    return " ".join(s.split())


def _find(norm: str, keywords: Iterable[str]) -> Optional[tuple[str, int]]:
    """Longest keyword found in ``norm``: (keyword, weight)."""
    padded = f" {norm} "
    compact = norm.replace(" ", "")
    best: Optional[tuple[str, int]] = None
    for kw in keywords:
        if _HANGUL_RE.search(kw):
            hit = kw.replace(" ", "") in compact
            weight = 2 * len(kw.replace(" ", ""))
        else:
            hit = f" {kw} " in padded
            weight = len(kw)
        if hit and (best is None or weight > best[1]):
            best = (kw, weight)
    return best


@dataclass(frozen=True)
class DescriptionMatch:
    cls: SymbolClass = SymbolClass.UNKNOWN
    actuator: ActuatorType = ActuatorType.NONE
    keyword: Optional[str] = None  # the class keyword that matched
    actuator_keyword: Optional[str] = None


def map_description(text: str) -> DescriptionMatch:
    """Class and actuator stated by a legend description (English / Korean).

    ``"CHECK (NON-RETURN) VALVE"`` -> check valve; ``"S.D.N.R. VALVE"`` -> SDNR valve;
    ``"HYD. OPERATED BUTTERFLY VALVE"`` -> butterfly valve, hydraulic; ``"REMOTE BUTTERFLY
    VALVE"`` -> butterfly valve, hydraulic (remote-operated ship valves); ``"체크 밸브"`` -> check
    valve. A description naming no known symbol maps to ``SymbolClass.UNKNOWN``; actuators are
    only reported for valves."""
    norm = normalize_description(text)
    if not norm:
        return DescriptionMatch()
    best: Optional[tuple[SymbolClass, str, int]] = None
    for cls, kws in CLASS_KEYWORDS:
        hit = _find(norm, kws)
        if hit is not None and (best is None or hit[1] > best[2]):
            best = (cls, hit[0], hit[1])
    cls = best[0] if best else SymbolClass.UNKNOWN
    act, akw = ActuatorType.NONE, None
    if cls != SymbolClass.UNKNOWN and get_symbol(cls).category == Category.VALVE:
        for a, kws in ACTUATOR_KEYWORDS:
            hit = _find(norm, kws)
            if hit is not None:
                act, akw = a, hit[0]
                break
        if akw is None:
            hit = _find(norm, REMOTE_KEYWORDS)
            if hit is not None:
                act, akw = ActuatorType.HYDRAULIC, hit[0]
    return DescriptionMatch(cls, act, best[1] if best else None, akw)


# --------------------------------------------------------------------------- data


@dataclass
class LegendRow:
    """One legend entry: the glyph (page-space primitives) and its description."""

    page: int
    description: str
    glyph_bbox: BBox4
    text_bbox: BBox4
    prims: tuple = ()  # glyph geometry, :mod:`shippid.symbols` primitives in page coordinates
    cls: SymbolClass = SymbolClass.UNKNOWN  # stated by the description
    actuator: ActuatorType = ActuatorType.NONE  # stated by the description
    keyword: Optional[str] = None
    actuator_keyword: Optional[str] = None
    drawn_actuator: ActuatorType = ActuatorType.NONE  # actuator decoration found on the glyph
    # 'unmapped' | 'mapped' (not learned yet) | after LegendBook.learn: 'library' (the built-in symbol of the class)
    # | 'template' | 'remapped' (a built-in symbol of another class, learned as this class) | 'skipped: <reason>'
    # | 'conflict' (one glyph, several classes)
    status: str = "unmapped"
    same_as: Optional[str] = None  # built-in class the glyph is drawn exactly like

    @property
    def mapped(self) -> bool:
        return self.cls != SymbolClass.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page, "description": self.description,
            "glyph_bbox": [round(v, 2) for v in self.glyph_bbox], "text_bbox": [round(v, 2) for v in self.text_bbox],
            "cls": self.cls.value, "actuator": self.actuator.value, "keyword": self.keyword,
            "actuator_keyword": self.actuator_keyword, "drawn_actuator": self.drawn_actuator.value,
            "status": self.status, "same_as": self.same_as,
        }


@dataclass
class Legend:
    page: int
    bbox: BBox4
    heading: str
    rows: list[LegendRow] = field(default_factory=list)

    @property
    def unmapped(self) -> list[LegendRow]:
        return [r for r in self.rows if not r.mapped]


@dataclass(frozen=True)
class LegendSpec(SymbolSpec):
    """A symbol template learned from a legend row (the body; library actuator decorations are
    added by the matcher like for built-in valves)."""

    legend: str = ""  # description of the (first) row
    rows: tuple[tuple[str, str], ...] = ()  # (drawn actuator value, description) of every row with this body
    inherent_actuator: ActuatorType = ActuatorType.NONE  # stated by the description, not drawn on the glyph
    page: int = 0

    def template_name(self, actuator: ActuatorType = ActuatorType.NONE) -> str:
        for act, desc in self.rows:
            if act == actuator.value:
                return f"legend:{desc}"
        return f"legend:{self.legend}"


# --------------------------------------------------------------------------- legend regions and rows


@dataclass
class PageGeometry:
    """What legend finding needs of a page (the extractor's first steps)."""

    pc: PageContent
    ps: PrimSet
    lines: list[TextLine]
    regions: Regions
    unit: float
    tol: float
    lettering: Optional[Lettering] = None  # stroke lettering without a text carrier (marked 'text')


def page_lines(pc: PageContent) -> list[TextLine]:
    """The page's text lines, assembled exactly like the extractor does."""
    rules = [(s.x0, s.y0, s.x1, s.y1) for s in pc.segments if s.curve < 0 and s.length >= 2.0]
    return build_lines(pc.words, separators=rules)


def heading_lines(lines: Sequence[TextLine]) -> list[TextLine]:
    """Horizontal short lines that read as a legend heading (``LEGEND``, ``SYMBOL LEGEND``,
    ``SYMBOLS``, ``범례`` ...)."""
    return [t for t in lines if t.rotation == 0 and len(t.text) <= 40 and HEADING_RE.search(t.text)
            and not NOTES_RE.match(t.text)]


def prepare_page(pc: PageContent, config: Optional[Config] = None,
                 lines: Optional[list[TextLine]] = None) -> PageGeometry:
    """Primitives, text lines (SHX stroke text marked), symbol unit and sheet regions -- the same
    computation as the first steps of :func:`shippid.extract.pipeline.extract_page` (which can
    reuse the result)."""
    from .pipeline import page_text_strokes  # the extractor's own first step (imported late: no cycle)

    cfg = config or Config()
    tol = cfg.snap_tol
    ps = PrimSet.from_page(pc)
    if lines is None:
        lines = page_lines(pc)
    unit, lettering = page_text_strokes(ps, lines, tol, None, (pc.width, pc.height))
    regions = detect_regions(ps, lines, pc.width, pc.height, tol, unit)
    return PageGeometry(pc, ps, lines, regions, unit, tol, lettering)


def page_has_heading(pc_or_text: Union[PageContent, str]) -> bool:
    """Cheap test: the page's text (whitespace removed: also text written one character at a time)
    contains a legend heading word."""
    text = pc_or_text if isinstance(pc_or_text, str) else "".join(w.text for w in pc_or_text.words)
    return bool(_PAGE_HINT_RE.search(re.sub(r"\s+", "", text)))


def find_legends(page_or_content, page_index: int = 0, config: Optional[Config] = None,
                 geometry: Optional[PageGeometry] = None) -> list[Legend]:
    """Legend regions of a page (a PyMuPDF page or its :class:`PageContent`) with their rows
    (glyph primitives + description), rows mapped to a class / actuator by :func:`map_description`.
    Legends without any glyph row (an abbreviation list) are not returned."""
    if geometry is None:
        pc = page_or_content if isinstance(page_or_content, PageContent) else read_page(page_or_content)
        if not pc.readable or not page_has_heading(pc):
            return []
        lines = page_lines(pc)
        if not heading_lines(lines):
            return []
        geometry = prepare_page(pc, config, lines)
    g = geometry
    tol = g.tol
    heads = heading_lines(g.lines)
    out: list[Legend] = []
    seen: list[BBox4] = []
    for head in heads:
        box = _legend_box(g, head)
        if box is None or any(bbox_inside(box, b, tol) for b in seen):
            continue
        rows = _parse_rows(g, box, head, page_index)
        if not rows:
            wider = _table_around(g, head, box)
            if wider is not None:
                box = wider
                rows = _parse_rows(g, box, head, page_index)
        if not rows:
            continue
        seen.append(box)
        out.append(Legend(page_index, tuple(float(v) for v in box), head.text, rows))
    return out


def _legend_box(g: PageGeometry, head: TextLine) -> Optional[BBox4]:
    """The region of a legend heading: the extractor's legend region around it, else (headings the
    region finder does not know, e.g. Korean '범례') the same construction: an enclosing box, a
    table cell, or the text block below the heading with room for the glyphs on its left."""
    tol, unit, ps, reg = g.tol, g.unit, g.ps, g.regions
    inside = [b for b in reg.legends if bbox_inside(head.bbox, b, tol)]
    if inside:
        return min(inside, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    da = reg.drawing_area
    da_area = max(1.0, (da[2] - da[0]) * (da[3] - da[1]))
    where = reg.kind_at(head.bbox, tol)
    if where in ("title_block", "table"):
        box = _enclosing_box(ps, head.bbox, head, da_area, tol, unit)
        if box is not None:
            return box
        cell = reg.cell_at(head.cx, head.cy)
        hb = head.bbox
        if cell is not None and (cell[2] - cell[0]) * (cell[3] - cell[1]) > 6 * (hb[2] - hb[0]) * (hb[3] - hb[1]):
            return cell
        return None
    if where is not None:
        return None
    bb = _grow_block(head, g.lines, reg, column=False, unit=unit)
    box = _enclosing_box(ps, bb, head, da_area, tol, unit)
    return box or (bb[0] - 3.0 * unit, bb[1], bb[2], bb[3] + 0.5 * unit)


def _table_around(g: PageGeometry, head: TextLine, box: BBox4) -> Optional[BBox4]:
    """A heading in its own cell of a gridded legend table: the table holding it."""
    cand = [b for b in g.regions.tables if bbox_inside(head.bbox, b, g.tol) and not bbox_inside(b, box, g.tol)]
    return min(cand, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if cand else None


def _glyph_prim_ids(g: PageGeometry, box: BBox4) -> list[int]:
    """Drawing primitives inside a legend region that can belong to glyphs: not text strokes,
    frame or table rules, not the region's own outline and not long rules across it."""
    tol = g.tol
    bw, bh = box[2] - box[0], box[3] - box[1]
    pad = max(2 * tol, 0.01 * max(bw, bh))
    outline_rings = {rid for rid, r in g.ps.rings.items()
                     if all(abs(a - b) <= pad for a, b in zip(r.bbox, box))}
    ids = []
    for p in g.ps.prims:
        if p.role in ("text", "frame", "table") or not bbox_inside(p.bbox, box, tol) or p.ring in outline_rings:
            continue
        if p.kind == "seg" and p.axis == "h" and abs(p.x1 - p.x0) >= 0.45 * bw:
            continue
        if p.kind == "seg" and p.axis == "v" and abs(p.y1 - p.y0) >= 0.45 * bh:
            continue
        ids.append(p.id)
    return ids


def _parse_rows(g: PageGeometry, box: BBox4, head: TextLine, page_index: int) -> list[LegendRow]:
    tol = g.tol
    ps = g.ps
    texts = [t for t in g.lines if t is not head and t.rotation == 0 and t.text.strip()
             and bbox_inside(t.bbox, box, tol)]
    ids = _glyph_prim_ids(g, box)
    if not texts or not ids:
        return []
    touch = max(0.6, 0.5 * tol)
    dsu = DSU(len(ps.prims))
    for i, j in ps.pairs_within(ids, touch):
        if prim_distance(ps.prims[i], ps.prims[j]) <= touch:
            dsu.union(i, j)
    clusters = list(dsu.groups(ids).values())
    owner: dict[int, list[int]] = {}  # text index -> prim ids of its glyph
    for cl in clusters:
        bb = bbox_union(ps.prims[i].bbox for i in cl)
        k = _row_text(bb, texts)
        if k is not None:
            owner.setdefault(k, []).extend(cl)
    used = set(owner)
    rows: list[LegendRow] = []
    for k in sorted(owner, key=lambda k: texts[k].order):
        t = texts[k]
        parts = [t]
        # same baseline continuation (a description written as several text objects)
        while True:
            last = parts[-1]
            h = max(line_height(last), 1.0)
            nxt = [j for j, u in enumerate(texts) if j not in used and abs(u.cy - last.cy) <= 0.35 * h
                   and 0.0 <= u.bbox[0] - last.bbox[2] <= 3.0 * h]
            if not nxt:
                break
            j = min(nxt, key=lambda j: texts[j].bbox[0])
            used.add(j)
            parts.append(texts[j])
        desc = " ".join(p.text.strip() for p in parts)
        bottom, x0 = max(p.bbox[3] for p in parts), t.bbox[0]
        # wrapped lines below without a glyph of their own
        while True:
            h = max(line_height(t), 1.0)
            nxt = [j for j, u in enumerate(texts) if j not in used and abs(u.bbox[0] - x0) <= 1.5 * h
                   and -0.2 * h <= u.bbox[1] - bottom <= 0.9 * h]
            if not nxt:
                break
            j = min(nxt, key=lambda j: texts[j].bbox[1])
            used.add(j)
            parts.append(texts[j])
            desc += " " + texts[j].text.strip()
            bottom = texts[j].bbox[3]
        prim_ids = sorted(set(owner[k]))
        prims = _page_prims(ps, prim_ids)
        if not prims:
            continue
        gb = bbox_union(ps.prims[i].bbox for i in prim_ids)
        dm = map_description(desc)
        rows.append(LegendRow(page_index, " ".join(desc.split()), tuple(float(v) for v in gb),
                              tuple(float(v) for v in bbox_union(p.bbox for p in parts)), tuple(prims), dm.cls,
                              dm.actuator, dm.keyword, dm.actuator_keyword,
                              status="unmapped" if dm.cls == SymbolClass.UNKNOWN else "mapped"))
    return rows


def _row_text(bb: BBox4, texts: Sequence[TextLine]) -> Optional[int]:
    """Index of the description a glyph cluster belongs to: the nearest text on its row, to its
    right (descriptions on the left only when nothing is on the right)."""
    ch = bb[3] - bb[1]
    cy = (bb[1] + bb[3]) / 2
    size = max(bb[2] - bb[0], ch)
    for side in ("right", "left"):
        best = None
        for k, t in enumerate(texts):
            h = max(line_height(t), 1.0)
            dy = abs(cy - t.cy)
            if dy > max(0.5 * ch, 0.8 * h):
                continue
            gap = (t.bbox[0] - bb[2]) if side == "right" else (bb[0] - t.bbox[2])
            if gap < -0.5 * h or gap > max(15.0 * h, 3.0 * size):
                continue
            score = max(gap, 0.0) + 3.0 * dy
            if best is None or score < best[0]:
                best = (score, k)
        if best is not None:
            return best[1]
    return None


def _page_prims(ps: PrimSet, ids: Sequence[int]) -> list[Primitive]:
    """Extractor primitives -> :mod:`shippid.symbols` primitives (closed rings as polygons with
    their fill, circles, the other strokes as lines)."""
    idset = set(ids)
    done: set[int] = set()
    out: list[Primitive] = []
    for i in ids:
        p = ps.prims[i]
        if p.kind == "circle":
            out.append(Circle(p.cx, p.cy, p.r, fill=p.filled))
            continue
        if p.ring >= 0 and p.ring in ps.rings:
            ring = ps.rings[p.ring]
            if all(j in idset for j in ring.prim_ids):
                if p.ring not in done:
                    done.add(p.ring)
                    out.append(Poly(tuple(ring.pts), closed=True, fill=ring.filled))
                continue
        out.append(Line(p.x0, p.y0, p.x1, p.y1))
    return out


# --------------------------------------------------------------------------- geometry helpers


def _map(prims: Iterable[Primitive], f) -> list[Primitive]:
    """Apply the similarity ``f(x, y) -> (x', y')`` (uniform scale, no rotation) to primitives."""
    out: list[Primitive] = []
    for p in prims:
        if isinstance(p, Line):
            a, b = f(p.x0, p.y0), f(p.x1, p.y1)
            out.append(Line(a[0], a[1], b[0], b[1]))
        elif isinstance(p, Poly):
            out.append(Poly(tuple(f(x, y) for x, y in p.pts), closed=p.closed, fill=p.fill))
        elif isinstance(p, Circle):
            c = f(p.cx, p.cy)
            e = f(p.cx + p.r, p.cy)
            out.append(Circle(c[0], c[1], math.hypot(e[0] - c[0], e[1] - c[1]), fill=p.fill))
        elif isinstance(p, Rect):
            pts = ((p.x0, p.y0), (p.x1, p.y0), (p.x1, p.y1), (p.x0, p.y1))
            out.append(Poly(tuple(f(x, y) for x, y in pts), closed=True, fill=p.fill))
    return out


def _extent(prims: Sequence[Primitive]) -> BBox4:
    xs: list[float] = []
    ys: list[float] = []
    for p in prims:
        if isinstance(p, Line):
            xs += [p.x0, p.x1]
            ys += [p.y0, p.y1]
        elif isinstance(p, Poly):
            xs += [q[0] for q in p.pts]
            ys += [q[1] for q in p.pts]
        elif isinstance(p, Circle):
            xs += [p.cx - p.r, p.cx + p.r]
            ys += [p.cy - p.r, p.cy + p.r]
        elif isinstance(p, Rect):
            xs += [p.x0, p.x1]
            ys += [p.y0, p.y1]
    return (min(xs), min(ys), max(xs), max(ys))


def _samples(prims: Sequence[Primitive], step: float) -> np.ndarray:
    return np.asarray(sample_outline(prims, step), dtype=float).reshape(-1, 2)


def _ink_length(prims: Sequence[Primitive]) -> float:
    total = 0.0
    for p in prims:
        if isinstance(p, Line):
            total += math.hypot(p.x1 - p.x0, p.y1 - p.y0)
        elif isinstance(p, Poly):
            ring = list(p.pts) + ([p.pts[0]] if p.closed else [])
            total += sum(math.dist(a, b) for a, b in zip(ring, ring[1:]))
        elif isinstance(p, Circle):
            total += 2 * math.pi * p.r
        elif isinstance(p, Rect):
            total += 2 * (abs(p.x1 - p.x0) + abs(p.y1 - p.y0))
    return total


def shape_of(prims: Sequence[Primitive]) -> CandidateShape:
    """A :class:`CandidateShape` of :mod:`shippid.symbols` primitives (page coordinates), sampled
    like an extracted candidate."""
    bb = _extent(prims)
    D = max(bb[2] - bb[0], bb[3] - bb[1], 1e-6)
    step = D / N_SAMPLES
    pts = _samples(prims, step)
    fill = np.asarray(_fill_samples(prims, step), dtype=float).reshape(-1, 2)
    cs = CandidateShape(pts, fill, bb, sum(1 for p in prims if isinstance(p, Circle)))
    cs.tree = cKDTree(pts)
    cs.fill_tree = cKDTree(fill) if len(fill) else None
    return cs


def _inline_pair(spec: SymbolSpec) -> Optional[tuple[PortSpec, PortSpec]]:
    """The two opposite ports on the local x axis (in / out of an inline symbol)."""
    left = [p for p in spec.ports if abs(p.y) < 1e-9 and p.dx < -0.99 and abs(p.dy) < 1e-9]
    right = [p for p in spec.ports if abs(p.y) < 1e-9 and p.dx > 0.99 and abs(p.dy) < 1e-9]
    if left and right:
        return left[0], right[0]
    return None


def _axis_y(prims: Sequence[Primitive], bb: BBox4) -> float:
    """Height of the flow axis: midway between the vertical extents of the ink at the glyph's
    leftmost and at its rightmost end (the two outermost ends of an inline symbol)."""
    W = max(bb[2] - bb[0], 1e-6)
    pts = _samples(prims, 0.01 * W)
    mids = []
    for sel in (pts[:, 0] <= bb[0] + 0.02 * W, pts[:, 0] >= bb[2] - 0.02 * W):
        if sel.any():
            ys = pts[sel, 1]
            mids.append((ys.min() + ys.max()) / 2)
    return float(np.mean(mids)) if mids else (bb[1] + bb[3]) / 2


# --------------------------------------------------------------------------- templates

DECORATION_COVER = 0.04  # (template units) decoration ink must lie this close to glyph ink ...
DECORATION_SHARE = 0.97  # ... for this share of its samples
DECORATION_EXPLAIN = 0.025  # glyph strokes lying this close to the decoration are the actuator


def _actuator_on_glyph(prims: list[Primitive], spec: SymbolSpec) -> tuple[ActuatorType, list[Primitive]]:
    """A library actuator decoration drawn on an inline valve glyph (stem from the body centre on
    the flow axis, upwards): (actuator, body primitives without it)."""
    pair = _inline_pair(spec)
    if spec.category != Category.VALVE or pair is None or len(prims) < 2:
        return ActuatorType.NONE, prims
    bb = _extent(prims)
    W = bb[2] - bb[0]
    lib_w = pair[1].x - pair[0].x
    if W <= 1e-6 or lib_w <= 1e-9:
        return ActuatorType.NONE, prims
    s = W / lib_w
    ox, oy = (bb[0] + bb[2]) / 2, _axis_y(prims, bb)
    step = 0.01 * s
    tree = cKDTree(_samples(prims, step))
    best = None
    for act in (A.HYDRAULIC, A.ELECTRIC, A.QUICK_CLOSING, A.DIAPHRAGM, A.HANDWHEEL):
        dec = _map(ACTUATOR_PRIMS[act], lambda u, v: (ox + s * u, oy + s * v))
        dpts = _samples(dec, step)
        d, _ = tree.query(dpts)
        if (d <= DECORATION_COVER * s).mean() >= DECORATION_SHARE:
            ink = _ink_length(dec)
            if best is None or ink > best[0]:
                best = (ink, act, dpts)
    if best is None:
        return ActuatorType.NONE, prims
    _, act, dpts = best
    dtree = cKDTree(dpts)
    body = []
    for p in prims:
        d, _ = dtree.query(_samples([p], step))
        if not (d <= DECORATION_EXPLAIN * s).all():
            body.append(p)
    if len(body) == len(prims) or not body:
        return ActuatorType.NONE, prims
    nb = _extent(body)
    if abs((nb[2] - nb[0]) - W) > 0.02 * W:  # the decoration must not carry the body's ends
        return ActuatorType.NONE, prims
    return act, body


def _strip_pipe_stubs(prims: list[Primitive]) -> list[Primitive]:
    """Short horizontal pipe stubs drawn on both ends of an inline glyph (``-|>|<-``) are not part
    of the symbol: on the drawing the pipes end at the body."""
    bb = _extent(prims)
    W = bb[2] - bb[0]
    if W <= 1e-6:
        return prims

    def stub(side: int) -> Optional[int]:
        x_end = bb[0] if side == 0 else bb[2]
        found = []
        for k, p in enumerate(prims):
            if not isinstance(p, Line):
                continue
            L = math.hypot(p.x1 - p.x0, p.y1 - p.y0)
            if L <= 1e-9 or abs(p.y1 - p.y0) > 0.02 * L or L > 0.6 * W:
                continue
            if min(abs(p.x0 - x_end), abs(p.x1 - x_end)) <= 0.02 * W:
                found.append(k)
        return found[0] if len(found) == 1 else None

    a, b = stub(0), stub(1)
    if a is None or b is None or a == b:
        return prims
    ya, yb = prims[a].y0, prims[b].y0
    if abs(ya - yb) > 0.05 * W:
        return prims
    body = [p for k, p in enumerate(prims) if k not in (a, b)]
    return body if body else prims


def _snap_lines(prims: list[Primitive]) -> list[Primitive]:
    """Make nearly axis-aligned lines exactly axis-aligned and round coordinates (PDF float noise)."""
    out: list[Primitive] = []

    def r(v: float) -> float:
        v = round(v, 5)
        return 0.0 if abs(v) < 1e-9 else v

    for p in prims:
        if isinstance(p, Line):
            x0, y0, x1, y1 = p.x0, p.y0, p.x1, p.y1
            L = math.hypot(x1 - x0, y1 - y0)
            if L > 0 and abs(y1 - y0) <= 0.01 * L:
                y0 = y1 = (y0 + y1) / 2
            elif L > 0 and abs(x1 - x0) <= 0.01 * L:
                x0 = x1 = (x0 + x1) / 2
            out.append(Line(r(x0), r(y0), r(x1), r(y1)))
        elif isinstance(p, Poly):
            out.append(Poly(tuple((r(x), r(y)) for x, y in p.pts), closed=p.closed, fill=p.fill))
        elif isinstance(p, Circle):
            out.append(Circle(r(p.cx), r(p.cy), r(p.r), fill=p.fill))
        else:
            out.append(p)
    return out


def _free_ends(prims: Sequence[Primitive], tol: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """(point, outward unit direction) of open line ends touching no other primitive."""
    out = []
    for k, p in enumerate(prims):
        if not isinstance(p, Line):
            continue
        L = math.hypot(p.x1 - p.x0, p.y1 - p.y0)
        if L <= 1e-9:
            continue
        others = [q for j, q in enumerate(prims) if j != k]
        if not others:
            continue
        tree = cKDTree(_samples(others, tol / 2))
        for (x, y), (ax, ay) in (((p.x0, p.y0), (p.x1, p.y1)), ((p.x1, p.y1), (p.x0, p.y0))):
            d, _ = tree.query((x, y))
            if d > tol:
                out.append(((x, y), ((x - ax) / L, (y - ay) / L)))
    return out


def _infer_ports(prims: Sequence[Primitive], spec: SymbolSpec) -> tuple[PortSpec, ...]:
    """The class's built-in ports on the glyph (local frame): a free stub end pointing the port's
    way, else the outermost ink in the port direction, at the port's position across it."""
    if not spec.ports:
        return ()
    bb = _extent(prims)
    D = max(bb[2] - bb[0], bb[3] - bb[1], 1e-6)
    pts = _samples(prims, 0.01 * D)
    free = _free_ends(prims, 0.03 * D)
    used: set[int] = set()
    out = []
    for ps_ in spec.ports:
        n = math.hypot(ps_.dx, ps_.dy) or 1.0
        d = (ps_.dx / n, ps_.dy / n)
        cand = [(k, f) for k, f in enumerate(free) if k not in used and f[1][0] * d[0] + f[1][1] * d[1] >= 0.95]
        if cand:
            k, (pos, fd) = max(cand, key=lambda kf: kf[1][0][0] * d[0] + kf[1][0][1] * d[1])
            used.add(k)
            out.append(PortSpec(ps_.name, pos[0], pos[1], fd[0], fd[1]))
            continue
        proj = pts @ np.array(d)
        m = float(proj.max())
        near = pts[proj >= m - 0.03 * D]
        perp = np.array((-d[1], d[0]))
        pv = near @ perp
        t = min(max(ps_.x * perp[0] + ps_.y * perp[1], float(pv.min())), float(pv.max()))
        x, y = m * d[0] + t * perp[0], m * d[1] + t * perp[1]
        out.append(PortSpec(ps_.name, round(x, 5) + 0.0, round(y, 5) + 0.0, d[0], d[1]))
    return tuple(out)


def glyph_template(row: LegendRow, legend_scale: Optional[float] = None) -> Optional[LegendSpec]:
    """Template of a mapped row from its glyph (see the module docstring); None when the glyph has
    no usable geometry. Sets ``row.drawn_actuator``."""
    if not row.prims:
        return None
    if row.mapped:
        lib = get_symbol(row.cls)
    else:
        lib = SymbolSpec(SymbolClass.UNKNOWN, Category.EQUIPMENT, row.description, row.description, (), ())
    prims = list(row.prims)
    pair = _inline_pair(lib)
    if pair is not None:
        prims = _strip_pipe_stubs(prims)
    act, body = _actuator_on_glyph(prims, lib)
    row.drawn_actuator = act
    bb = _extent(body)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    if max(w, h) <= 1e-6:
        return None
    if pair is not None:
        s = w / (pair[1].x - pair[0].x)
        ox, oy = (bb[0] + bb[2]) / 2, _axis_y(body, bb)
    else:
        if row.mapped:
            le = lib.extent
            s = max(w, h) / max(le[2] - le[0], le[3] - le[1], 1e-6)
        else:
            s = legend_scale or max(w, h)
        ox, oy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    if s <= 1e-9:
        return None
    local = _snap_lines(_map(body, lambda x, y: ((x - ox) / s, (y - oy) / s)))
    ports = _infer_ports(local, lib) if row.mapped else ()
    inherent = row.actuator if (act == ActuatorType.NONE and lib.category == Category.VALVE) else ActuatorType.NONE
    return LegendSpec(
        lib.cls, lib.category, row.description, row.description, tuple(local), ports,
        boundary_ports=lib.boundary_ports, stretchable=False, holds_text=lib.holds_text, rotatable=lib.rotatable,
        description=f"legend: {row.description}", legend=row.description, rows=((act.value, row.description),),
        inherent_actuator=inherent, page=row.page,
    )


def _geometry_key(spec: SymbolSpec) -> tuple:
    items = []
    for p in spec.prims:
        if isinstance(p, Line):
            a, b = (round(p.x0, 2), round(p.y0, 2)), (round(p.x1, 2), round(p.y1, 2))
            items.append(("L",) + (a + b if a <= b else b + a))
        elif isinstance(p, Poly):
            items.append(("P", p.closed, p.fill) + tuple(sorted((round(x, 2), round(y, 2)) for x, y in p.pts)))
        elif isinstance(p, Circle):
            items.append(("C", round(p.cx, 2), round(p.cy, 2), round(p.r, 2), p.fill))
    return (spec.category.value, tuple(sorted(items, key=repr)))


_EQUIV_CACHE_MAX = 4096  # per built-in library: a yard's legend recurs on all of its drawings


def _normalised_key(prims: Sequence[Primitive]) -> tuple:
    """Glyph geometry relative to its bbox (max dimension 1), quantised (cache key)."""
    bb = _extent(prims)
    D = max(bb[2] - bb[0], bb[3] - bb[1], 1e-9)
    local = _map(prims, lambda x, y: ((x - bb[0]) / D, (y - bb[1]) / D))
    q = 400.0
    items = []
    for p in local:
        if isinstance(p, Line):
            a, b = (round(p.x0 * q), round(p.y0 * q)), (round(p.x1 * q), round(p.y1 * q))
            items.append(("L",) + (a + b if a <= b else b + a))
        elif isinstance(p, Poly):
            items.append(("P", p.closed, p.fill) + tuple((round(x * q), round(y * q)) for x, y in p.pts))
        elif isinstance(p, Circle):
            items.append(("C", round(p.cx * q), round(p.cy * q), round(p.r * q), p.fill))
    return tuple(sorted(items, key=repr))


def _library_equivalent(row: LegendRow) -> Optional[MatchResult]:
    """The built-in variant the glyph *is* (same drawing up to scale: no unexplained or missing ink),
    or None."""
    lib = get_library()
    cache: dict = lib.__dict__.setdefault("_legend_equivalents", {})  # lives and dies with the library
    key = _normalised_key(row.prims)
    if key in cache:
        return cache[key]
    res = match_candidate(shape_of(row.prims), lib, None, top=3)
    out = None
    if res:
        r = res[0]
        if r.score >= EXACT_SCORE and r.raw is None and r.unexplained <= INK_TOL and r.missing <= INK_TOL \
                and not r.variant.stripped:
            out = r
    if len(cache) >= _EQUIV_CACHE_MAX:
        cache.clear()
    cache[key] = out
    return out


def _stub_lengths(spec: SymbolSpec) -> dict[str, float]:
    """Port name -> length of the straight stub drawn from that port inwards (stripped by the
    matcher when the stub merges into the pipe)."""
    out: dict[str, float] = {}
    for p in spec.prims:
        if not isinstance(p, Line):
            continue
        for ps_ in spec.ports:
            for (ax, ay), (bx, by) in (((p.x0, p.y0), (p.x1, p.y1)), ((p.x1, p.y1), (p.x0, p.y0))):
                if abs(ax - ps_.x) < 1e-9 and abs(ay - ps_.y) < 1e-9:
                    dx, dy = bx - ax, by - ay
                    n = math.hypot(dx, dy) or 1.0
                    if abs(dx / n + ps_.dx) < 1e-6 and abs(dy / n + ps_.dy) < 1e-6:
                        out[ps_.name] = n
    return out


# --------------------------------------------------------------------------- the document's legend book


class LegendBook:
    """Everything learned from the legends of one document."""

    def __init__(self, legends: Sequence[Legend]):
        self.legends = list(legends)
        self.specs: list[LegendSpec] = []  # learned templates (added to the matcher)
        self.confirmed: dict[tuple[SymbolClass, ActuatorType], str] = {}  # built-in symbols the legend confirms
        self.unmapped_specs: list[LegendSpec] = []
        self._library: Optional[TemplateLibrary] = None
        self._unmapped_library: Optional[TemplateLibrary] = None
        self._stubs: dict[int, dict[str, float]] = {}

    # -- learning --------------------------------------------------------------
    @classmethod
    def learn(cls, legends: Sequence[Legend]) -> "LegendBook":
        """Templates of the mapped rows (see the module docstring)."""
        book = cls(legends)
        rows = [r for lg in book.legends for r in lg.rows]
        groups: dict[tuple, list[tuple[LegendRow, LegendSpec]]] = {}
        for row in rows:
            if not row.mapped:
                row.status = "unmapped"
                spec = glyph_template(row)
                if spec is not None:
                    book.unmapped_specs.append(spec)
                continue
            lib = get_symbol(row.cls)
            if lib.stretchable:
                row.status = "skipped: box-like equipment of any size (built-in template)"
                continue
            eq = _library_equivalent(row)
            if eq is not None:
                row.same_as = eq.variant.cls.value
                if eq.variant.cls == row.cls:
                    row.status = "library"
                    row.drawn_actuator = eq.variant.actuator
                    book.confirmed.setdefault((row.cls, eq.variant.actuator), row.description)
                    continue
            spec = glyph_template(row)
            if spec is None:
                row.status = "skipped: no glyph geometry"
                continue
            row.status = "remapped" if eq is not None else "template"
            groups.setdefault(_geometry_key(spec), []).append((row, spec))
        # one drawing, two meanings (compared without scale: each class normalises differently): undecided
        meanings: dict[tuple, set] = {}
        for members in groups.values():
            for r, sp in members:
                meanings.setdefault(_normalised_key(sp.prims), set()).add(r.cls)
        for key, members in groups.items():
            classes = set().union(*(meanings[_normalised_key(sp.prims)] for _, sp in members))
            if len(classes) > 1:
                for r, _ in members:
                    r.status = "conflict"
                log.info("legend glyph used for several classes %s; not learned", sorted(c.value for c in classes))
                continue
            row0, spec0 = members[0]
            names = []
            for r, sp in members:
                pair = (r.drawn_actuator.value, r.description)
                if pair not in names:
                    names.append(pair)
            inherent = next((sp.inherent_actuator for _, sp in members if sp.inherent_actuator != ActuatorType.NONE),
                            ActuatorType.NONE)
            book.specs.append(LegendSpec(
                spec0.cls, spec0.category, spec0.label_en, spec0.label_ko, spec0.prims, spec0.ports,
                boundary_ports=spec0.boundary_ports, stretchable=False, holds_text=spec0.holds_text,
                rotatable=spec0.rotatable, description=spec0.description, legend=row0.description,
                rows=tuple(names), inherent_actuator=inherent, page=row0.page))
        return book

    # -- matching --------------------------------------------------------------
    @property
    def library(self) -> Optional[TemplateLibrary]:
        """Built-in library extended by the learned templates (None: nothing learned)."""
        if self._library is None and self.specs:
            self._library = TemplateLibrary.extended(self.specs)
        return self._library

    def is_learned(self, variant) -> bool:
        return isinstance(variant.spec, LegendSpec)

    def template_for(self, variant) -> Optional[str]:
        """``"legend:<description>"`` of a matched variant: a learned template, or a built-in symbol
        the legend shows (same class and actuator, else the class drawn without actuator)."""
        spec = variant.spec
        if isinstance(spec, LegendSpec):
            return spec.template_name(variant.actuator)
        desc = self.confirmed.get((variant.cls, variant.actuator))
        if desc is None:
            desc = self.confirmed.get((variant.cls, ActuatorType.NONE))
        return f"legend:{desc}" if desc else None

    def actuator_for(self, variant, actuator: ActuatorType) -> ActuatorType:
        """Actuator of a match: the drawn decoration, else what the legend row states (a glyph
        whose drawing itself means remote / hydraulic operation)."""
        spec = variant.spec
        if actuator == ActuatorType.NONE and isinstance(spec, LegendSpec):
            return spec.inherent_actuator
        return actuator

    def regions(self, page: int) -> list[BBox4]:
        return [lg.bbox for lg in self.legends if lg.page == page]

    @property
    def unmapped(self) -> list[LegendRow]:
        """Rows whose description names no known symbol (for Gemma: description + glyph bbox)."""
        return [r for lg in self.legends for r in lg.rows if not r.mapped]

    def describe_unknown(self, shape: CandidateShape) -> Optional[dict[str, Any]]:
        """The unmapped legend row an unmatched symbol is drawn like (``{"description", "score",
        "page", "glyph_bbox"}``), or None."""
        if not self.unmapped_specs:
            return None
        if self._unmapped_library is None:
            self._unmapped_library = TemplateLibrary(self.unmapped_specs)
        res = match_candidate(shape, self._unmapped_library, None)
        if not res or res[0].score < UNKNOWN_ACCEPT:
            return None
        spec = res[0].variant.spec
        row = next((r for r in self.unmapped if r.description == getattr(spec, "legend", None)), None)
        out: dict[str, Any] = {"description": getattr(spec, "legend", spec.label_en), "score": round(res[0].score, 3)}
        if row is not None:
            out["page"] = row.page
            out["glyph_bbox"] = [round(v, 2) for v in row.glyph_bbox]
        return out

    def stub_lengths(self, spec: SymbolSpec) -> dict[str, float]:
        k = id(spec)
        if k not in self._stubs:
            self._stubs[k] = _stub_lengths(spec)
        return self._stubs[k]

    # -- reporting -------------------------------------------------------------
    def to_meta(self) -> dict[str, Any]:
        rows = [r for lg in self.legends for r in lg.rows]
        return {
            "legends": [{"page": lg.page, "bbox": [round(v, 2) for v in lg.bbox], "heading": lg.heading,
                         "rows": [r.to_dict() for r in lg.rows]} for lg in self.legends],
            "templates": [{"cls": s.cls.value, "legend": s.legend, "rows": [list(x) for x in s.rows],
                           "inherent_actuator": s.inherent_actuator.value, "page": s.page} for s in self.specs],
            "confirmed": sorted({f"{c.value}/{a.value}": d for (c, a), d in self.confirmed.items()}.items()),
            "unmapped": [{"page": r.page, "description": r.description,
                          "glyph_bbox": [round(v, 2) for v in r.glyph_bbox]} for r in rows if not r.mapped],
        }


def learn_legends(legends: Sequence[Legend]) -> LegendBook:
    """Shorthand for :meth:`LegendBook.learn`."""
    return LegendBook.learn(legends)


# --------------------------------------------------------------------------- matcher hook


def _points_to_segments(pts: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Distance of each point to the nearest segment ``(x0, y0, x1, y1)``."""
    a = segs[None, :, :2]
    b = segs[None, :, 2:]
    p = pts[:, None, :]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(-1), 1e-12)
    t = np.clip(((p - a) * ab).sum(-1) / L2, 0.0, 1.0)
    q = a + t[..., None] * ab
    return np.sqrt(((p - q) ** 2).sum(-1)).min(axis=1)


def stub_on_pipe(book: Optional[LegendBook], r: MatchResult, pipe_segs: Optional[np.ndarray], tol: float) -> bool:
    """For a match whose port stub was stripped: is the stub there, drawn as part of a pipe line
    (a pump's tangential discharge stub merged into the discharge pipe)?"""
    v = r.variant
    stubs = book.stub_lengths(v.spec) if book is not None else _stub_lengths(v.spec)
    if not stubs:
        return True
    if pipe_segs is None or not len(pipe_segs):
        return False
    for name, x, y, dx, dy in v.ports:
        L = stubs.get(name)
        if not L:
            continue
        a = np.array(r.to_page(x, y))
        b = np.array(r.to_page(x - dx * L, y - dy * L))
        pts = a[None, :] + np.linspace(0.0, 1.0, 5)[:, None] * (b - a)[None, :]
        lim = max(tol, 0.08 * float(np.hypot(*(b - a))))
        if (_points_to_segments(pts, pipe_segs) > lim).any():
            return False
    return True


def rerank(results: list[MatchResult], lib: TemplateLibrary, pipe_segs: Optional[np.ndarray], tol: float,
           book: Optional[LegendBook] = None) -> list[MatchResult]:
    """Matcher hook for a library extended by learned templates (``lib.n_extra`` leading variants):
    a learned template whose stub was stripped needs that stub on a pipe (else its score falls by
    :data:`NO_STUB_FACTOR`), and the best learned template is preferred over a built-in winner that
    is not better by more than :data:`LEGEND_TIE` (the legend states the document's convention)."""
    n = getattr(lib, "n_extra", 0)
    if not n or not results:
        return results
    changed = False
    for r in results:
        v = r.variant
        if v.idx < n and v.stripped and not stub_on_pipe(book, r, pipe_segs, tol):
            r.score *= NO_STUB_FACTOR
            if r.raw is not None:
                r.raw *= NO_STUB_FACTOR
            changed = True
    if changed:
        results.sort(key=lambda r: (-round(r.score, 4), r.variant.idx))
    best = results[0]
    if best.variant.idx >= n:
        learned = [r for r in results if r.variant.idx < n and r.score >= best.score - LEGEND_TIE]
        if learned:
            top = max(learned, key=lambda r: (r.score, -r.variant.idx))
            results.remove(top)
            results.insert(0, top)
    return results


def is_learned_variant(variant) -> bool:
    return isinstance(variant.spec, LegendSpec)


_AUTO_PORT_RE = re.compile(r"n\d+")


def name_stub_ports(net, matches: Iterable[tuple[int, MatchResult]], tol: float) -> int:
    """Port names of learned templates whose stub was stripped: the pipe runs over the stub and
    ends where the stub met the body (a pump's tangential discharge stub on the edge of its box),
    so the network attaches it there under an automatic name (``n0``); that attachment gets the
    stub's port name (``discharge``). ``net``: a :class:`shippid.extract.topology.Network`,
    ``matches``: (body index, best match) of the classified symbols. Returns the renamed count."""
    renamed = 0
    for k, r in matches:
        v = r.variant
        if not (v.stripped and is_learned_variant(v)):
            continue
        lst = net.ports.get(k)
        if not lst:
            continue
        stubs = _stub_lengths(v.spec)
        taken = {name for name, _, _ in lst}
        for name, x, y, dx, dy in v.ports:
            L = stubs.get(name)
            if not L or name in taken:
                continue
            q = np.array(r.to_page(x, y))
            t = np.array(r.to_page(x - dx * L, y - dy * L))
            seg = np.concatenate((t, q))[None, :]
            for i, (pn, pos, d) in enumerate(lst):
                if not _AUTO_PORT_RE.fullmatch(pn):
                    continue
                if float(_points_to_segments(np.array([pos], dtype=float), seg)[0]) <= tol:
                    lst[i] = (name, pos, d)
                    for n in net.nodes:
                        if n.body == k and n.port == pn:
                            n.port = name
                    taken.add(name)
                    renamed += 1
                    break
    return renamed
