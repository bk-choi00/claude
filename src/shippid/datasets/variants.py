"""Out-of-distribution export variants of the synthetic ship P&IDs (robustness suite).

The synthetic generator (:mod:`shippid.datasets.synthetic`) and the vector extractor share one
symbol library and one drawing style, so the extractor scores ~1.0 on the synthetic test split.
Real shipyard PDFs (AutoCAD / AVEVA / Cadmatic / Smart P&ID exports) differ in how the same
drawing is *written*: rotated pages, quads instead of triangles, pipes drawn through symbols,
tessellated curves, other stroke styles, fragmented and duplicated text, layers that are switched
off, unusual scales and yard-specific symbol conventions explained by a legend.

Every variant re-renders one generated drawing (ground truth :class:`~shippid.model.PIDDocument`
+ :class:`~shippid.datasets.pdf_writer.DrawingSpec`) with its own PDF writer and returns ground truth
that matches the new PDF exactly. Coordinates are always in the *displayed* page space (after
``/Rotate``), see :class:`~shippid.model.PageInfo`. Nothing is written into the PDF that leaks the
ground truth.

Public API:

* :data:`VARIANTS` -- ordered ``name -> description`` (``plain`` is a control: the drawing
  re-rendered by this module's writer without any change);
* :func:`make_variant` -- ``(seed, variant, system=None) -> (PIDDocument, pdf bytes)``,
  deterministic per ``(seed, variant, system)``;
* :func:`build_variant_suite` -- ``out_dir/<variant>/{<id>.pdf, <id>.json, manifest.jsonl}``
  (manifest rows in the format of :func:`shippid.benchmark.load_manifest`, split ``"test"``) and
  ``out_dir/summary.json``.

The ground-truth ``meta["variant"]`` records the variant, its components (``combined``) and the
parameters drawn for this file; for ``layers_off`` it lists the OCG states and the texts / symbols
that exist in the file only on layers that are OFF (they are *not* ground-truth items: an
extractor that reports them is wrong).
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pymupdf

from ..model import (
    ActuatorType,
    BBox,
    Category,
    PIDDocument,
    PipeKind,
    Point,
    SymbolClass,
    SystemType,
    TextRole,
)
from ..symbols import ACTUATOR_PRIMS, Circle, Line, Placement, Poly, Rect, get_symbol, placed_bbox
from .build import assign_systems, item_seed
from .pdf_writer import DrawingSpec
from .synthetic import generate, random_style
from .text import font_metrics, text_width

log = logging.getLogger(__name__)

S = SymbolClass
Color = tuple[float, float, float]
BLACK: Color = (0.0, 0.0, 0.0)
WHITE: Color = (1.0, 1.0, 1.0)

VARIANTS: dict[str, str] = {
    "plain": "control: the drawing re-rendered by the variant writer without any change",
    "rotated_page": "PDF /Rotate 90 or 270 with the content laid out so the displayed sheet is landscape "
                    "(CAD plotter output on portrait media)",
    "qu_bowties": "bow-tie valve bodies drawn as one closed self-intersecting 4-point polyline (PyMuPDF 'qu' item) "
                  "instead of two triangles; filled outlet triangles as a separate fill",
    "pipes_through": "process pipes drawn as continuous lines straight through inline valves / filters / flow meters / "
                     "reducers, the symbol on top (optionally over a white wipeout)",
    "polyline_curves": "every circle and arc (symbols, junction dots, crossing hops) tessellated into straight "
                       "segments (24-48 per circle)",
    "styled_lines": "pipe widths 0.25-2.0 pt, symbols sometimes thicker than pipes, all-black or colour-per-layer, "
                    "round / square caps, some entities drawn twice",
    "fragmented_text": "text emitted per character or per word with explicit positions, mixed Courier / Times / "
                       "Helvetica, Hangul labels in the 'korea' font, hidden render-mode-3 duplicates (PDFSHX=2)",
    "layers_off": "OCG layers; NOTES / DIMENSION / superseded-revision / construction layers start OFF and are "
                  "absent from the ground truth, the ON layers are present",
    "scaled": "whole sheet scaled 0.6x or 1.5x (an A1 sheet becomes A0), symbol and text sizes far from the defaults",
    "yard_symbols": "yard symbol convention for gate / check / SDNR / ball valves, centrifugal pumps and local "
                    "instruments, drawn consistently and explained in a LEGEND block; classes stay canonical",
    "combined": "a random mix of 2-4 of the variants above per drawing",
}

# variants that ``combined`` mixes, in the order their transformations are applied
COMBINABLE: tuple[str, ...] = ("yard_symbols", "fragmented_text", "pipes_through", "layers_off", "styled_lines",
                               "qu_bowties", "polyline_curves", "scaled", "rotated_page")

A0: tuple[float, float] = (3370.0, 2384.0)  # landscape, points
KOREA_FONT = "korea"
# extraction metrics of PyMuPDF's built-in 'korea' CJK font (rendered as Droid Sans Fallback, reported as
# "Dotum"): every character advances 1 em and the glyph box spans [-1.0, +0.2] em around the baseline
# (verified by tests/test_variants.py against get_text)
_KOREA_ASC, _KOREA_DESC = 1.0, -0.2
_FAMILIES = {"helv": ("helv", "hebo"), "tiro": ("tiro", "tibo"), "cour": ("cour", "cobo")}
_BOLD = {"hebo", "tibo", "cobo"}

# --------------------------------------------------------------------------- symbol geometry

_TRI_L = ((-0.5, -0.3), (-0.5, 0.3), (0.0, 0.0))
_TRI_R = ((0.5, -0.3), (0.5, 0.3), (0.0, 0.0))
_BOWTIE_QU = ((-0.5, -0.3), (0.5, 0.3), (0.5, -0.3), (-0.5, 0.3))  # self-intersecting closed polyline
_CY = math.sqrt(0.05)  # pump design A: circle through the suction port, tangent to the discharge stub


def _bowtie(fill_out: bool = False) -> tuple:
    return (Poly(_TRI_L), Poly(_TRI_R, fill=fill_out))


# Yard convention: (cls -> candidate designs in preference order). Ports stay the canonical ones; a
# design that leaves the canonical extent is used only when it collides with nothing in the file; the
# last design of each class stays inside the canonical extent.
YARD_DESIGNS: dict[SymbolClass, tuple[tuple, ...]] = {
    S.GATE_VALVE: (_bowtie() + (Line(0.0, -0.3, 0.0, 0.3),),),
    S.CHECK_VALVE: (_bowtie() + (Line(0.42, -0.25, 0.42, 0.25),),),
    S.SDNR_VALVE: (_bowtie() + (Line(0.42, -0.25, 0.42, 0.25), Circle(0.0, 0.0, 0.1, fill=True)),),
    S.BALL_VALVE: (_bowtie() + (Circle(0.0, 0.0, 0.17, fill=True),),),
    S.PUMP_CENTRIFUGAL: (
        (Circle(-0.3, _CY, 0.3), Line(0.0, _CY, 0.0, -0.5)),
        (Circle(-0.25, 0.0, 0.25), Line(0.0, 0.0, 0.0, -0.5)),
    ),
    S.INSTRUMENT_LOCAL: ((Circle(0.0, 0.0, 0.4), Circle(0.0, 0.0, 0.33)),),
}
YARD_DESCRIPTIONS: dict[SymbolClass, str] = {
    S.GATE_VALVE: "bow-tie with a vertical bar in the middle",
    S.CHECK_VALVE: "open bow-tie with a vertical bar at the outlet side (no fill)",
    S.SDNR_VALVE: "open bow-tie with an outlet-side bar and a filled centre dot",
    S.BALL_VALVE: "bow-tie with a large filled circle",
    S.PUMP_CENTRIFUGAL: "circle with a tangential discharge stub",
    S.INSTRUMENT_LOCAL: "double circle",
}

THROUGH_CLASSES = frozenset({
    S.GATE_VALVE, S.GLOBE_VALVE, S.BALL_VALVE, S.BUTTERFLY_VALVE, S.CHECK_VALVE, S.SDNR_VALVE, S.THREE_WAY_VALVE,
    S.FILTER, S.DUPLEX_FILTER, S.STRAINER, S.FLOW_METER, S.REDUCER, S.FLEXIBLE_HOSE, S.SPECTACLE_FLANGE,
})
BOWTIE_CLASSES = frozenset({S.GATE_VALVE, S.GLOBE_VALVE, S.BALL_VALVE, S.CHECK_VALVE, S.SDNR_VALVE,
                            S.THREE_WAY_VALVE})

# --------------------------------------------------------------------------- layer schemes

_LAYER_SCHEMES: tuple[dict[str, str], ...] = (
    {"pipe": "P-PIPE", "signal": "P-SIGNAL", "equipment": "P-EQPT", "valve": "P-VALV", "instrument": "P-INST",
     "fitting": "P-VALV", "connector": "P-ANNO", "text": "P-ANNO", "sheet": "BORDER", "notes": "NOTES",
     "legend": "LEGEND", "dim": "DIMENSION", "old": "REV-A", "construction": "DEFPOINTS"},
    {"pipe": "PROCESS_LINES", "signal": "INSTRUMENT_LINES", "equipment": "EQUIPMENT", "valve": "VALVES",
     "instrument": "INSTRUMENTS", "fitting": "FITTINGS", "connector": "CONNECTORS", "text": "ANNOTATION",
     "sheet": "DRAWING_FRAME", "notes": "GENERAL_NOTES", "legend": "SYMBOL_LEGEND", "dim": "DIMENSIONS",
     "old": "SUPERSEDED", "construction": "CONSTRUCTION"},
    {"pipe": "PIPING", "signal": "SIGNAL", "equipment": "EQUIP", "valve": "VALVE", "instrument": "INST",
     "fitting": "FITTING", "connector": "CONNECT", "text": "TEXT", "sheet": "TITLE", "notes": "NOTE",
     "legend": "LEGEND", "dim": "DIM", "old": "OLD-REV", "construction": "HIDDEN"},
)
# darkened AutoCAD index colours (readable on white paper)
_PALETTE: tuple[Color, ...] = ((0.85, 0.0, 0.0), (0.0, 0.55, 0.0), (0.0, 0.0, 0.85), (0.0, 0.55, 0.65),
                               (0.7, 0.0, 0.7), (0.7, 0.5, 0.0), (0.45, 0.45, 0.45), (0.55, 0.25, 0.0))

# --------------------------------------------------------------------------- Hangul labels

_KO_PHRASES: tuple[tuple[str, str], ...] = (
    ("OILY WATER SEPARATOR", "유수분리기"), ("BOILER FEED W.", "보일러 급수"), ("FIRE & G/S", "소화 겸 잡용"),
    ("MAIN AIR COMPRESSOR", "주 공기압축기"), ("MAIN AIR RECEIVER", "주 공기탱크"), ("AIR COMP.", "공기압축기"),
    ("HIGH SEA CHEST", "고위 씨체스트"), ("LOW SEA CHEST", "저위 씨체스트"), ("SEA CHEST", "씨체스트"),
    ("BILGE WELL", "빌지 웰"), ("MUD BOX", "머드 박스"), ("FLOW METER", "유량계"), ("MAIN ENGINE", "주기관"),
    ("AUX. BOILER", "보조 보일러"), ("HOT WELL", "핫 웰"), ("SEA W.", "해수"), ("UV REACTOR", "자외선 반응기"),
    ("BACK-FLUSHING", "역세"),
)
_KO_WORDS: dict[str, str] = {
    "PUMP": "펌프", "TK": "탱크", "W.B.TK": "평형수 탱크", "BALLAST": "평형수", "BILGE": "빌지", "M/E": "주기관",
    "H.F.O.": "중유", "F.O.": "연료유", "L.O.": "윤활유", "M.D.O.": "경유", "L.S.H.F.O.": "저유황중유", "STOR.": "저장",
    "SETT.": "침전", "SERV.": "서비스", "SUMP": "섬프", "C.F.W.": "청수냉각", "C.S.W.": "해수냉각", "CENTRAL": "중앙",
    "MAIN": "주", "EXPANSION": "팽창", "TRANSFER": "이송", "SUPPLY": "공급", "CIRC.": "순환", "HEATER": "히터",
    "COOLER": "냉각기", "PURIFIER": "청정기", "FILTER": "필터", "STRAINER": "스트레이너", "FIRE": "소화",
    "AIR": "공기", "COMPRESSOR": "압축기", "RECEIVER": "공기탱크", "OVERBOARD": "선외배출", "(P)": "(좌현)",
    "(S)": "(우현)", "AFT": "선미", "FWD": "선수", "E/R": "기관실", "S.W.": "해수", "F.W.": "청수", "AUTO.": "자동",
    "AUTO": "자동", "BWTS": "평형수처리", "B.W.T.": "평형수처리", "HEATING": "가열", "CALORIFIER": "온수기",
    "BOILER": "보일러", "FEED": "급수", "SEPARATOR": "분리기", "(OWS)": "(유수분리기)", "OWS": "유수분리기",
    "G/E": "발전기관", "UNIT": "장치", "STARTING": "시동", "INLET": "입구", "HOLDING": "저장", "F.P.TK": "선수탱크",
    "CASCADE": "캐스케이드", "WELL": "웰", "HIGH": "고위", "LOW": "저위",
}
_NO_RE = re.compile(r"^NO\.(\d+)$")


def korean_label(text: str) -> Optional[str]:
    """Korean rendering of an English equipment label (``NO.1 BALLAST PUMP`` -> ``1번 평형수 펌프``);
    ``None`` when a word is not in the small marine glossary."""
    s = f" {text.strip()} "
    for en, ko in _KO_PHRASES:
        s = s.replace(f" {en} ", f" \x00{ko.replace(' ', chr(1))} ")
    out = []
    for tok in s.split():
        if tok.startswith("\x00"):
            out.append(tok[1:].replace(chr(1), " "))
        elif _NO_RE.match(tok):
            out.append(f"{_NO_RE.match(tok).group(1)}번")
        elif tok in _KO_WORDS:
            out.append(_KO_WORDS[tok])
        else:
            return None
    res = " ".join(out)
    return res if res and any("가" <= c <= "힣" for c in res) else None


# --------------------------------------------------------------------------- text metrics


def _metrics(font: str) -> tuple[float, float]:
    return (_KOREA_ASC, _KOREA_DESC) if font == KOREA_FONT else font_metrics(font)


def _tw(text: str, font: str, size: float) -> float:
    if font == KOREA_FONT:
        return float(pymupdf.get_text_length(text, fontname=KOREA_FONT, fontsize=size))
    return text_width(text, font, size)


def _tbbox(text: str, x: float, y: float, size: float, font: str, rotate: int) -> tuple[float, float, float, float]:
    w = _tw(text, font, size)
    asc, desc = _metrics(font)
    if rotate == 0:
        return (x, y - asc * size, x + w, y - desc * size)
    return (x - asc * size, y - w, x - desc * size, y)


def _origin(text: str, x0: float, y0: float, size: float, font: str, rotate: int) -> tuple[float, float]:
    asc, _ = _metrics(font)
    if rotate == 0:
        return (x0, y0 + asc * size)
    return (x0 + asc * size, y0 + _tw(text, font, size))


def _r(v: float) -> float:
    return round(float(v), 4)


def _bb(b: Sequence[float]) -> BBox:
    return BBox(x0=_r(b[0]), y0=_r(b[1]), x1=_r(b[2]), y1=_r(b[3]))


# --------------------------------------------------------------------------- scene


@dataclass
class _Path:
    segs: list
    width: float
    color: Optional[Color]
    fill: Optional[Color] = None
    dashes: Optional[str] = None
    closed: bool = False
    layer: Optional[str] = None
    role: str = "sheet"  # sheet | notes | legend | pipe | signal | arrow | through | dim | old | construction
    pipe: Optional[int] = None  # ground-truth pipe index


@dataclass
class _Sym:
    cls: SymbolClass
    pl: Placement
    act: ActuatorType
    width: float
    color: Color
    layer: Optional[str]
    role: str  # "legend" | category value | "old"
    obj: Optional[int] = None  # ground-truth object index
    wipeout: bool = False  # white body fill under the symbol (CAD wipeout)


@dataclass
class _Circ:
    x: float
    y: float
    r: float
    width: float
    color: Color
    fill: Optional[Color]
    layer: Optional[str]
    role: str = "dot"


@dataclass
class _Text:
    text: str
    x: float  # baseline origin, displayed page space of the unscaled sheet
    y: float
    size: float
    font: str
    rotate: int
    color: Color
    layer: Optional[str]
    role: str  # TextRole value | "dim" | "old"
    gt: Optional[int] = None  # ground-truth text index
    mode: str = "whole"  # whole | word | char
    hidden_font: Optional[str] = None  # invisible (render mode 3) duplicate in this font


@dataclass
class _Opts:
    rotation: int = 0
    scale: float = 1.0
    offset: tuple[float, float] = (0.0, 0.0)
    page_size: Optional[tuple[float, float]] = None  # displayed size after scaling
    qu: bool = False
    curve_n: Optional[int] = None
    cap: int = 0
    join: int = 0
    yard: dict = field(default_factory=dict)  # SymbolClass -> local body primitives


@dataclass
class _Ctx:
    seed: int
    variant: str
    doc: PIDDocument
    spec: DrawingSpec
    ops: list
    opts: _Opts = field(default_factory=_Opts)
    layers: dict[str, bool] = field(default_factory=dict)  # OCG name -> initially on (insertion ordered)
    drop_texts: set[int] = field(default_factory=set)
    params: dict[str, Any] = field(default_factory=dict)
    off_texts: list[dict[str, Any]] = field(default_factory=list)
    off_objects: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text_ops(self) -> dict[int, _Text]:
        return {op.gt: op for op in self.ops if isinstance(op, _Text) and op.gt is not None}

    @property
    def obj_syms(self) -> dict[int, _Sym]:
        return {op.obj: op for op in self.ops if isinstance(op, _Sym) and op.obj is not None}

    @property
    def symbol_scale(self) -> float:
        return float(self.doc.meta.get("symbol_scale") or 18.0)


def _seg_points(segs: list) -> tuple[tuple[float, float], tuple[float, float]]:
    return (segs[0][1], segs[0][2]), (segs[-1][-2], segs[-1][-1])


def _reverse(segs: list) -> list:
    out = []
    for s in reversed(segs):
        if s[0] == "l":
            out.append(("l", s[3], s[4], s[1], s[2]))
        else:
            out.append(("c", s[7], s[8], s[5], s[6], s[3], s[4], s[1], s[2]))
    return out


def _classify_paths(doc: PIDDocument, spec: DrawingSpec) -> list[tuple[str, Optional[int]]]:
    """Role of every ``spec.paths`` entry: ("pipe"/"signal", gt pipe index) for pipe drawings (found by
    geometry: straight pieces lie on exactly one ground-truth polyline), ("arrow", None) for flow
    arrows, ("notes"/"legend", None) for the box around the notes / legend, ("sheet", None) otherwise."""
    A, B, owner = [], [], []
    for k, p in enumerate(doc.pipes):
        for a, b in zip(p.points, p.points[1:]):
            A.append((a.x, a.y))
            B.append((b.x, b.y))
            owner.append(k)
    A_ = np.array(A, dtype=float).reshape(-1, 2)
    B_ = np.array(B, dtype=float).reshape(-1, 2)
    own = np.array(owner, dtype=int)
    AB = B_ - A_
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)

    def on_pipes(q: tuple[float, float]) -> set[int]:
        if not len(own):
            return set()
        Q = np.array(q, dtype=float)
        t = np.clip(((Q - A_) * AB).sum(1) / L2, 0.0, 1.0)
        d = np.hypot(*(A_ + t[:, None] * AB - Q).T)
        return set(own[d < 0.01].tolist())

    def box_of(roles: set[TextRole]) -> Optional[tuple[float, float, float, float]]:
        bs = [t.bbox for t in doc.texts if t.role in roles]
        if not bs:
            return None
        return (min(b.x0 for b in bs), min(b.y0 for b in bs), max(b.x1 for b in bs), max(b.y1 for b in bs))

    notes_box, legend_box = box_of({TextRole.NOTE}), box_of({TextRole.LEGEND})
    out: list[tuple[str, Optional[int]]] = []
    for p in spec.paths:
        if p.closed and p.fill is not None and len(p.segs) == 3:
            out.append(("arrow", None))
            continue
        if p.closed and len(p.segs) == 4 and all(s[0] == "l" for s in p.segs):
            xs = [s[1] for s in p.segs]
            ys = [s[2] for s in p.segs]
            r = (min(xs), min(ys), max(xs), max(ys))
            hit = None
            for name, b in (("notes", notes_box), ("legend", legend_box)):
                if b and r[0] <= b[0] + 0.5 and r[1] <= b[1] + 0.5 and r[2] >= b[2] - 0.5 and r[3] >= b[3] - 0.5 \
                        and (r[2] - r[0]) * (r[3] - r[1]) < 0.25 * spec.width * spec.height:
                    hit = name
            if hit:
                out.append((hit, None))
                continue
        cand: Optional[set[int]] = None
        loose: set[int] = set()
        for s in p.segs:
            if s[0] == "l":
                m = on_pipes(((s[1] + s[3]) / 2, (s[2] + s[4]) / 2))
                cand = m if cand is None else (cand & m)
            else:
                loose |= on_pipes((s[1], s[2])) | on_pipes((s[7], s[8]))
        found = cand if cand is not None else loose
        if found and len(found) == 1:
            k = next(iter(found))
            out.append(("signal" if doc.pipes[k].kind == PipeKind.SIGNAL else "pipe", k))
        else:
            out.append(("sheet", None))
    return out


def _scene(doc: PIDDocument, spec: DrawingSpec, seed: int, variant: str) -> _Ctx:
    if len(spec.texts) != len(doc.texts) or any(a.text != b.text for a, b in zip(spec.texts, doc.texts)):
        raise RuntimeError("drawing spec texts do not line up with the ground truth")
    n_leg = len(spec.symbols) - len(doc.objects)
    if n_leg < 0 or any(spec.symbols[n_leg + i].cls != o.cls for i, o in enumerate(doc.objects)):
        raise RuntimeError("drawing spec symbols do not line up with the ground-truth objects")
    kinds = _classify_paths(doc, spec)
    paths = [_Path(list(p.segs), p.width, p.color, p.fill, p.dashes, p.closed, p.layer, role, k)
             for p, (role, k) in zip(spec.paths, kinds)]
    syms = []
    for i, sd in enumerate(spec.symbols):
        obj = i - n_leg if i >= n_leg else None
        role = "legend" if obj is None else get_symbol(sd.cls).category.value
        syms.append(_Sym(sd.cls, sd.placement, sd.actuator, sd.width, sd.color, sd.layer, role, obj))
    circles = [_Circ(c.x, c.y, c.r, c.width, c.color, c.fill, c.layer) for c in spec.circles]
    order = spec.order
    if order is None:
        order = ([("path", i) for i in range(len(paths))] + [("symbol", i) for i in range(len(syms))]
                 + [("circle", i) for i in range(len(circles))])
    pool = {"path": paths, "symbol": syms, "circle": circles}
    ops: list = [pool[k][i] for k, i in order]
    for i, (td, t) in enumerate(zip(spec.texts, doc.texts)):
        ops.append(_Text(td.text, td.x, td.y, td.size, td.font, td.rotate, td.color, td.layer, t.role.value, i))
    ctx = _Ctx(seed=seed, variant=variant, doc=doc.model_copy(deep=True), spec=spec, ops=ops)
    for name in spec.layers:
        ctx.layers[name] = True
    return ctx


# --------------------------------------------------------------------------- geometry helpers


def _xf(p, pl: Placement):
    """One local primitive placed on the page (same rules as :func:`shippid.symbols.placed_prims`)."""
    uniform = abs(pl.stretch[0] - pl.stretch[1]) < 1e-9
    if isinstance(p, Line):
        a, b = pl.to_page(p.x0, p.y0), pl.to_page(p.x1, p.y1)
        return Line(a[0], a[1], b[0], b[1])
    if isinstance(p, Poly):
        return Poly(tuple(pl.to_page(u, v) for u, v in p.pts), closed=p.closed, fill=p.fill)
    if isinstance(p, Circle):
        if uniform:
            c = pl.to_page(p.cx, p.cy)
            return Circle(c[0], c[1], p.r * pl.scale * pl.stretch[0], fill=p.fill)
        pts = [(p.cx + p.r * math.cos(2 * math.pi * k / 24), p.cy + p.r * math.sin(2 * math.pi * k / 24))
               for k in range(24)]
        return Poly(tuple(pl.to_page(u, v) for u, v in pts), closed=True, fill=p.fill)
    if isinstance(p, Rect):
        corners = [(p.x0, p.y0), (p.x1, p.y0), (p.x1, p.y1), (p.x0, p.y1)]
        return Poly(tuple(pl.to_page(u, v) for u, v in corners), closed=True, fill=p.fill)
    raise TypeError(p)


def _body_prims(cls: SymbolClass, opts: _Opts) -> tuple:
    return tuple(opts.yard.get(cls, get_symbol(cls).prims))


def _local_items(cls: SymbolClass, opts: _Opts) -> list[tuple[Any, frozenset]]:
    """Body primitives with drawing flags: ``qu`` (bow-tie drawn as one self-intersecting quad) and
    ``fill_only`` (a fill without outline, e.g. the hatch of a check valve's outlet triangle)."""
    prims = list(_body_prims(cls, opts))
    items: list[tuple[Any, frozenset]] = [(p, frozenset()) for p in prims]
    if not opts.qu:
        return items
    il = next((i for i, p in enumerate(prims) if isinstance(p, Poly) and p.closed and p.pts == _TRI_L), None)
    ir = next((i for i, p in enumerate(prims) if isinstance(p, Poly) and p.closed and p.pts == _TRI_R), None)
    if il is None or ir is None:
        return items
    out: list[tuple[Any, frozenset]] = []
    for i, p in enumerate(prims):
        if i == il:
            out.append((Poly(_BOWTIE_QU, closed=True), frozenset({"qu"})))
            for q in (prims[il], prims[ir]):
                if q.fill:
                    out.append((Poly(q.pts, closed=True, fill=True), frozenset({"fill_only"})))
        elif i != ir:
            out.append((p, frozenset()))
    return out


def _yard_bbox(cls: SymbolClass, prims: tuple, pl: Placement, act: ActuatorType) -> tuple[float, float, float, float]:
    spec = replace(get_symbol(cls), prims=tuple(prims))
    return placed_bbox(spec, pl, act if spec.category == Category.VALVE else ActuatorType.NONE)


def _rect_hits_seg(r: tuple[float, float, float, float], a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Axis-aligned rectangle vs segment (Liang-Barsky clipping)."""
    x0, y0, x1, y1 = r
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def _overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _expand(r: Sequence[float], d: float) -> tuple[float, float, float, float]:
    return (r[0] - d, r[1] - d, r[2] + d, r[3] + d)


# --------------------------------------------------------------------------- variant transformations


def _apply_yard(ctx: _Ctx, rng: random.Random) -> None:
    doc = ctx.doc
    syms = ctx.obj_syms
    chosen: dict[SymbolClass, tuple] = {}
    for cls, designs in YARD_DESIGNS.items():
        pick = designs[-1]
        for prims in designs[:-1]:
            if _design_fits(ctx, cls, prims, syms):
                pick = prims
                break
        chosen[cls] = pick
    ctx.opts.yard = chosen
    for i, o in enumerate(doc.objects):
        if o.cls in chosen:
            sd = syms[i]
            o.bbox = _bb(_yard_bbox(o.cls, chosen[o.cls], sd.pl, sd.act))
    # inside texts of double-circle bubbles: scale about the bubble centre to fit the inner circle
    inner = next(p.r for p in chosen[S.INSTRUMENT_LOCAL] if isinstance(p, Circle) and p.r < 0.4)
    tops = ctx.text_ops
    shrunk = 0
    for i, o in enumerate(doc.objects):
        if o.cls != S.INSTRUMENT_LOCAL:
            continue
        pl = syms[i].pl
        cx, cy = pl.cx, pl.cy
        own = [k for k, t in enumerate(doc.texts) if t.owner_id == o.id and t.role == TextRole.INSTRUMENT]
        if not own:
            continue
        far = max(math.hypot(x - cx, y - cy) for k in own for x, y in
                  ((doc.texts[k].bbox.x0, doc.texts[k].bbox.y0), (doc.texts[k].bbox.x1, doc.texts[k].bbox.y0),
                   (doc.texts[k].bbox.x0, doc.texts[k].bbox.y1), (doc.texts[k].bbox.x1, doc.texts[k].bbox.y1)))
        f = min(1.0, (inner * pl.scale - 0.6) / max(far, 1e-6))
        if f >= 0.999:
            continue
        shrunk += 1
        for k in own:
            t, op = doc.texts[k], tops[k]
            b = t.bbox
            nb = (cx + (b.x0 - cx) * f, cy + (b.y0 - cy) * f, cx + (b.x1 - cx) * f, cy + (b.y1 - cy) * f)
            _set_text(ctx, k, op.text, op.font, op.size * f, nb[0], nb[1])
    legend = sorted({op.cls.value for op in ctx.ops if isinstance(op, _Sym) and op.role == "legend"
                     and op.cls in chosen})
    ctx.params["yard_symbols"] = {
        "classes": {c.value: YARD_DESCRIPTIONS[c] for c in chosen},
        "pump_design": YARD_DESIGNS[S.PUMP_CENTRIFUGAL].index(chosen[S.PUMP_CENTRIFUGAL]),
        "legend_classes": legend, "instrument_texts_shrunk": shrunk,
    }


def _design_fits(ctx: _Ctx, cls: SymbolClass, prims: tuple, syms: dict[int, _Sym]) -> bool:
    """A design that leaves the canonical extent must not touch anything else on the sheet."""
    doc = ctx.doc
    area = doc.pages[0].drawing_area
    for i, o in enumerate(doc.objects):
        if o.cls != cls:
            continue
        sd = syms[i]
        yb = _yard_bbox(cls, prims, sd.pl, sd.act)
        cb = o.bbox
        if yb[0] >= cb.x0 - 1e-6 and yb[1] >= cb.y0 - 1e-6 and yb[2] <= cb.x1 + 1e-6 and yb[3] <= cb.y1 + 1e-6:
            continue
        if area and (yb[0] < area.x0 or yb[1] < area.y0 or yb[2] > area.x1 or yb[3] > area.y1):
            return False
        for j, q in enumerate(doc.objects):
            if j != i and _overlap(_expand(yb, 1.0), q.bbox.as_tuple()):
                return False
        for t in doc.texts:
            if _overlap(_expand(yb, 1.0), t.bbox.as_tuple()):
                return False
        own_ports = {(round(p.position.x, 2), round(p.position.y, 2)) for p in o.ports}
        for p in doc.pipes:
            mine = o.id in (p.start.object_id, p.end.object_id)
            for a, b in zip(p.points, p.points[1:]):
                pa, pb = (a.x, a.y), (b.x, b.y)
                if mine:
                    if (round(a.x, 2), round(a.y, 2)) in own_ports or (round(b.x, 2), round(b.y, 2)) in own_ports:
                        continue  # the segment ending at the port arrives along the port direction
                    if _rect_hits_seg(yb, pa, pb):
                        return False
                elif _rect_hits_seg(_expand(yb, 6.0), pa, pb):
                    return False
    return True


def _set_text(ctx: _Ctx, k: int, text: str, font: str, size: float, x0: float, y0: float) -> None:
    """Re-typeset ground-truth text ``k`` with its bbox's top-left corner at ``(x0, y0)``."""
    t = ctx.doc.texts[k]
    op = ctx.text_ops[k]
    ox, oy = _origin(text, x0, y0, size, font, t.rotation)
    bb = _tbbox(text, ox, oy, size, font, t.rotation)
    t.text, t.bbox, t.size = text, _bb(bb), round(size, 3)
    op.text, op.font, op.size, op.x, op.y = text, font, size, ox, oy


def _apply_text(ctx: _Ctx, rng: random.Random) -> None:
    doc = ctx.doc
    tops = ctx.text_ops
    groups = ("tag", "line_number", "instrument", "label", "connector", "title_block", "note", "legend")
    fam = {g: rng.choice(tuple(_FAMILIES)) for g in groups}
    p_mix = rng.uniform(0.1, 0.3)
    p_char = rng.uniform(0.3, 0.8)
    p_word = (1.0 - p_char) * rng.uniform(0.4, 0.9)
    p_hidden = rng.choice([rng.uniform(0.15, 0.5), 1.0])  # 1.0: every text has a PDFSHX=2 duplicate
    # Hangul labels: a share of the translatable equipment labels, typeset in the 'korea' CJK font
    labels = [k for k, t in enumerate(doc.texts) if t.role == TextRole.LABEL and korean_label(t.text)]
    n_ko = min(len(labels), max(1, int(round(len(labels) * rng.uniform(0.25, 0.6))))) if labels else 0
    korean = set(rng.sample(labels, n_ko)) if n_ko else set()
    n_ko_done = 0
    modes = Counter()
    fonts = Counter()
    for k, t in enumerate(doc.texts):
        op = tops[k]
        text = t.text
        b = t.bbox
        old_w = (b.x1 - b.x0) if t.rotation == 0 else (b.y1 - b.y0)
        old_h = (b.y1 - b.y0) if t.rotation == 0 else (b.x1 - b.x0)
        if k in korean:
            ko = korean_label(text)
            assert ko is not None
            font = KOREA_FONT
            new_text = ko
        else:
            family = fam.get(t.role.value, "helv")
            if rng.random() < p_mix:
                family = rng.choice(tuple(_FAMILIES))
            font = _FAMILIES[family][1 if op.font in _BOLD else 0]
            new_text = text
        asc, desc = _metrics(font)
        w = _tw(new_text, font, op.size)
        h = (asc - desc) * op.size
        f = min(1.0, old_w / max(w, 1e-6), old_h / max(h, 1e-6))
        if k in korean and f < 0.55:  # would be unreadably small: keep the English label
            font = _FAMILIES[fam.get("label", "helv")][0]
            new_text = text
            asc, desc = _metrics(font)
            w, h = _tw(text, font, op.size), (asc - desc) * op.size
            f = min(1.0, old_w / max(w, 1e-6), old_h / max(h, 1e-6))
        size = op.size * f
        nw, nh = w * f, h * f
        cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
        if t.rotation == 0:
            x0, y0 = cx - nw / 2, cy - nh / 2
        else:
            x0, y0 = cx - nh / 2, cy - nw / 2
        if new_text != text:
            n_ko_done += 1
            for o in doc.objects:
                if o.id == t.owner_id and o.attributes.get("name") == text:
                    o.attributes["name"] = new_text
                    o.attributes["name_en"] = text
        _set_text(ctx, k, new_text, font, size, x0, y0)
        u = rng.random()
        op.mode = "char" if u < p_char else ("word" if u < p_char + p_word else "whole")
        if rng.random() < p_hidden:  # PDFSHX=2 style invisible copy (a CJK font where Latin-1 cannot encode it)
            op.hidden_font = KOREA_FONT if any(ord(c) > 255 for c in new_text) else "helv"
        modes[op.mode] += 1
        fonts[font] += 1
    ctx.params["fragmented_text"] = {"families": fam, "p_mixed_font": round(p_mix, 3), "modes": dict(modes),
                                     "fonts": dict(sorted(fonts.items())), "hangul_labels": n_ko_done,
                                     "hidden_duplicates": sum(1 for op in tops.values() if op.hidden_font)}


def _apply_through(ctx: _Ctx, rng: random.Random) -> None:
    doc = ctx.doc
    wipeout = rng.random() < 0.5
    paths_of: dict[int, list[_Path]] = {}
    for op in ctx.ops:
        if isinstance(op, _Path) and op.role == "pipe" and op.pipe is not None:
            paths_of.setdefault(op.pipe, []).append(op)
    by_port: dict[tuple[str, str], int] = {}
    for k, p in enumerate(doc.pipes):
        for ep in (p.start, p.end):
            if ep.object_id:
                by_port[(ep.object_id, ep.port or "")] = k
    syms = ctx.obj_syms
    links = []  # (object index, pipe in, pipe out, port in, port out)
    for i, o in enumerate(doc.objects):
        if o.cls not in THROUGH_CLASSES:
            continue
        ports = {p.name: p for p in o.ports}
        if "in" not in ports or "out" not in ports:
            continue
        ka, kb = by_port.get((o.id, "in")), by_port.get((o.id, "out"))
        if ka is None or kb is None or ka == kb:
            continue
        pa, pb = doc.pipes[ka], doc.pipes[kb]
        if pa.kind != PipeKind.PROCESS or pb.kind != PipeKind.PROCESS or ka not in paths_of or kb not in paths_of:
            continue
        pin, pout = ports["in"].position, ports["out"].position
        ax = (pout.x - pin.x, pout.y - pin.y)
        n = math.hypot(*ax)
        if n < 1e-6 or min(abs(ax[0]), abs(ax[1])) > 1e-3 * n:
            continue

        def leaves(pipe, port: Point, sign: float) -> bool:
            pts = pipe.points if pipe.points[0].dist(port) < 1e-3 else list(reversed(pipe.points))
            if pts[0].dist(port) > 1e-3 or len(pts) < 2:
                return False
            d = (pts[1].x - pts[0].x, pts[1].y - pts[0].y)
            dn = math.hypot(*d)
            return dn > 1e-6 and abs(d[0] * ax[1] - d[1] * ax[0]) < 1e-3 * dn * n and sign * (d[0] * ax[0] + d[1] * ax[1]) > 0

        if not (leaves(pa, pin, -1.0) and leaves(pb, pout, 1.0)):
            continue
        links.append((i, ka, kb, (pin.x, pin.y), (pout.x, pout.y)))

    def end_at(path: _Path, q: tuple[float, float]) -> Optional[int]:
        a, b = _seg_points(path.segs)
        if math.dist(a, q) < 1e-2:
            return 0
        if math.dist(b, q) < 1e-2:
            return 1
        return None

    def endpoint(paths: list[_Path], q: tuple[float, float]) -> tuple[_Path, tuple[float, float]]:
        """The path of a pipe that ends at port ``q`` and its exact end point (``q`` if none does)."""
        for path in paths:
            e = end_at(path, q)
            if e is not None:
                return path, _seg_points(path.segs)[e]
        return paths[0], q

    def style(p: _Path) -> tuple:
        return (round(p.width, 6), p.color, p.dashes, p.layer)

    # chains of single-path pipes with equal style become one continuous polyline through the symbols
    merge: dict[tuple[int, int], tuple[int, int]] = {}  # (pipe, end) <-> (pipe, end)
    standalone: list[_Path] = []
    for i, ka, kb, pin, pout in links:
        pa_, pb_ = paths_of[ka], paths_of[kb]
        ea = end_at(pa_[0], pin) if len(pa_) == 1 else None
        eb = end_at(pb_[0], pout) if len(pb_) == 1 else None
        if ea is not None and eb is not None and style(pa_[0]) == style(pb_[0]) \
                and (ka, ea) not in merge and (kb, eb) not in merge:
            merge[(ka, ea)] = (kb, eb)
            merge[(kb, eb)] = (ka, ea)
            continue
        src, a = endpoint(pa_, pin)
        _, b = endpoint(pb_, pout)
        standalone.append(_Path([("l", a[0], a[1], b[0], b[1])], src.width, src.color, None, src.dashes, False,
                                src.layer, "through", None))
    merged_away: set[int] = set()
    n_chains = 0
    seen: set[int] = set()
    for k in sorted({key[0] for key in merge}):
        if k in seen:
            continue
        # walk backwards through end 0 to the chain's free end (or around a cycle)
        cur, exit_end = k, 0
        visited = {k}
        while (cur, exit_end) in merge:
            nxt, nxt_end = merge[(cur, exit_end)]
            if nxt in visited:
                break
            visited.add(nxt)
            cur, exit_end = nxt, 1 - nxt_end
        # forward walk: enter ``cur`` through that end; a pipe entered at its path end is reversed
        chain: list[tuple[int, bool]] = []
        entry = exit_end
        guard: set[int] = set()
        while True:
            guard.add(cur)
            out_end = 1 - entry
            chain.append((cur, entry == 1))
            nxt = merge.get((cur, out_end))
            if nxt is None:
                break
            if nxt[0] in guard:  # a closed loop: the last link stays a separate through segment
                a = _seg_points(paths_of[cur][0].segs)[out_end]
                b = _seg_points(paths_of[nxt[0]][0].segs)[nxt[1]]
                src = paths_of[cur][0]
                standalone.append(_Path([("l", a[0], a[1], b[0], b[1])], src.width, src.color, None, src.dashes,
                                        False, src.layer, "through", None))
                break
            cur, entry = nxt
        seen |= guard
        if len(chain) < 2:
            continue
        first = paths_of[chain[0][0]][0]
        segs: list = []
        for pk, rev in chain:
            ps = paths_of[pk][0].segs
            ps = _reverse(ps) if rev else list(ps)
            if segs:
                a = (segs[-1][-2], segs[-1][-1])
                b = (ps[0][1], ps[0][2])
                segs.append(("l", a[0], a[1], b[0], b[1]))  # straight through the symbol
            segs.extend(ps)
        first.segs = segs
        for pk, _ in chain[1:]:
            merged_away.add(id(paths_of[pk][0]))
        n_chains += 1
    through_objs = {lk[0] for lk in links}
    for i in through_objs:
        syms[i].wipeout = wipeout
    # graphics order: every line first, the symbols on top
    others = [op for op in ctx.ops if not isinstance(op, (_Sym, _Text)) and id(op) not in merged_away]
    ctx.ops = (others + standalone + [op for op in ctx.ops if isinstance(op, _Sym)]
               + [op for op in ctx.ops if isinstance(op, _Text)])
    ctx.params["pipes_through"] = {"symbols": len(through_objs), "merged_chains": n_chains,
                                   "standalone_segments": len(standalone), "wipeout": wipeout}


def _layer_key(op) -> str:
    if isinstance(op, _Path):
        if op.role in ("pipe", "through", "arrow"):
            return "pipe"
        return op.role if op.role in ("signal", "notes", "legend", "dim", "old", "construction") else "sheet"
    if isinstance(op, _Sym):
        return op.role  # "legend" | "old" | object category
    if isinstance(op, _Circ):
        return "pipe"
    role = op.role
    if role == TextRole.TITLE_BLOCK.value:
        return "sheet"
    if role == TextRole.NOTE.value:
        return "notes"
    if role == TextRole.LEGEND.value:
        return "legend"
    if role in ("dim", "old"):
        return role
    return "text"


def _assign_layers(ctx: _Ctx, scheme: dict[str, str], off: Iterable[str] = ()) -> None:
    off = set(off)
    used: dict[str, bool] = {}
    for op in ctx.ops:
        key = _layer_key(op)
        op.layer = scheme[key]
        used.setdefault(scheme[key], key not in off)
    order = list(scheme.values())
    ctx.layers = {name: used[name] for name in sorted(used, key=order.index)}


def _unique_text(ctx: _Ctx, s: str, bump) -> str:
    taken = {re.sub(r"\s+", "", t.text).upper() for t in ctx.doc.texts} | {re.sub(r"\s+", "", d["text"]).upper()
                                                                           for d in ctx.off_texts}
    while re.sub(r"\s+", "", s).upper() in taken:
        s = bump(s)
    return s


def _apply_layers_off(ctx: _Ctx, rng: random.Random) -> None:
    doc = ctx.doc
    scheme = dict(rng.choice(_LAYER_SCHEMES))
    s = ctx.symbol_scale
    off = {"dim"}
    notes_off = rng.random() < 0.5 and any(t.role == TextRole.NOTE for t in doc.texts)
    if notes_off:
        off.add("notes")
    with_old = rng.random() < 0.7
    with_construction = rng.random() < 0.5
    extra: list = []
    # dimensions between equipment (DIMENSION layer, OFF)
    equip = [o for o in doc.objects if o.category == Category.EQUIPMENT] or list(doc.objects)
    n_dim = 0
    for _ in range(rng.randint(2, 4)):
        if len(equip) < 2:
            break
        a, b = rng.sample(equip, 2)
        xa, xb = sorted((a.bbox.center.x, b.bbox.center.x))
        if xb - xa < 3 * s:
            continue
        y = min(a.bbox.y0, b.bbox.y0) - rng.uniform(8.0, 20.0)
        tick = 0.25 * s
        for x, top in ((xa, a.bbox.y0 if a.bbox.center.x == xa else b.bbox.y0),
                       (xb, b.bbox.y0 if b.bbox.center.x == xb else a.bbox.y0)):
            extra.append(_Path([("l", x, top - 2.0, x, y - 3.0)], 0.25, BLACK, role="dim"))
        extra.append(_Path([("l", xa, y, xb, y)], 0.25, BLACK, role="dim"))
        for x, d in ((xa, 1.0), (xb, -1.0)):
            extra.append(_Path([("l", x, y, x + d * tick, y - tick * 0.3), ("l", x + d * tick, y - tick * 0.3,
                                x + d * tick, y + tick * 0.3), ("l", x + d * tick, y + tick * 0.3, x, y)],
                               0.2, BLACK, fill=BLACK, closed=True, role="dim"))
        value = int(round((xb - xa) * rng.choice((8.0, 10.0, 12.5, 20.0)) / 10.0)) * 10
        txt = _unique_text(ctx, str(value), lambda v: str(int(v) + 10))
        size = 5.0
        w = text_width(txt, "helv", size)
        ox, oy = (xa + xb) / 2 - w / 2, y - 1.5
        extra.append(_Text(txt, ox, oy, size, "helv", 0, BLACK, None, "dim"))
        ctx.off_texts.append({"text": txt, "bbox": list(_tbbox(txt, ox, oy, size, "helv", 0)), "layer": "dim"})
        n_dim += 1
    # superseded revision: valves (and their tags) that were deleted from the line (OFF)
    n_old = 0
    if with_old:
        off.add("old")
        segs = []
        for p in doc.pipes:
            if p.kind != PipeKind.PROCESS:
                continue
            for a, b in zip(p.points, p.points[1:]):
                if a.dist(b) > 6 * s and (abs(a.x - b.x) < 1e-6 or abs(a.y - b.y) < 1e-6):
                    segs.append((a, b))
        for a, b in rng.sample(segs, min(len(segs), rng.randint(1, 3))):
            m = ((a.x + b.x) / 2, (a.y + b.y) / 2)
            horiz = abs(a.y - b.y) < 1e-6
            pl = Placement(m[0], m[1], scale=s, rotation=0 if horiz else 90)
            extra.append(_Sym(S.GATE_VALVE, pl, ActuatorType.HANDWHEEL, 0.6, BLACK, None, "old"))
            bb = placed_bbox(get_symbol(S.GATE_VALVE), pl, ActuatorType.HANDWHEEL)
            ctx.off_objects.append({"cls": S.GATE_VALVE.value, "bbox": list(bb), "layer": "old"})
            tag = _unique_text(ctx, f"V-9{rng.randint(10, 99)}", lambda v: f"V-9{(int(v[3:]) + 1) % 90 + 10}")
            size = 5.5
            w = text_width(tag, "helv", size)
            ox, oy = (bb[2] + 2.0, bb[1] - 1.0) if not horiz else (m[0] - w / 2, bb[1] - 2.0)
            extra.append(_Text(tag, ox, oy, size, "helv", 0, BLACK, None, "old"))
            ctx.off_texts.append({"text": tag, "bbox": list(_tbbox(tag, ox, oy, size, "helv", 0)), "layer": "old"})
            n_old += 1
    # construction lines (OFF)
    n_con = 0
    area = doc.pages[0].drawing_area
    if with_construction and area is not None:
        off.add("construction")
        for _ in range(rng.randint(4, 8)):
            if rng.random() < 0.5:
                y = rng.uniform(area.y0, area.y1)
                extra.append(_Path([("l", area.x0, y, area.x1, y)], 0.1, (0.5, 0.5, 0.5), role="construction"))
            else:
                x = rng.uniform(area.x0, area.x1)
                extra.append(_Path([("l", x, area.y0, x, area.y1)], 0.1, (0.5, 0.5, 0.5), role="construction"))
            n_con += 1
    texts = [op for op in ctx.ops if isinstance(op, _Text)]
    graphics = [op for op in ctx.ops if not isinstance(op, _Text)]
    ctx.ops = graphics + [op for op in extra if not isinstance(op, _Text)] + texts + \
        [op for op in extra if isinstance(op, _Text)]
    if notes_off:
        for k, t in enumerate(doc.texts):
            if t.role == TextRole.NOTE:
                ctx.drop_texts.add(k)
                ctx.off_texts.append({"text": t.text, "bbox": list(t.bbox.as_tuple()), "layer": "notes"})
    _assign_layers(ctx, scheme, off)
    for d in ctx.off_texts + ctx.off_objects:
        d["layer"] = scheme[d["layer"]]
    ctx.params["layers_off"] = {"scheme": scheme["pipe"], "off": sorted(scheme[k] for k in off),
                                "notes_off": notes_off, "dimensions": n_dim, "superseded_valves": n_old,
                                "construction_lines": n_con}


def _apply_styled(ctx: _Ctx, rng: random.Random) -> None:
    st = ctx.spec.style
    base_pw = float(getattr(st, "pipe_width", 1.0) or 1.0)
    pw = rng.uniform(0.25, 2.0)
    kp = pw / base_pw
    thick_symbols = rng.random() < 0.4
    sw = min(3.5, pw * rng.uniform(1.3, 2.2)) if thick_symbols else rng.uniform(0.25, 1.2)
    signal_w = rng.uniform(0.15, 0.6)
    mode = "black" if rng.random() < 0.5 else "layer"
    caps = rng.choice(("round", "round", "round", "square", "butt"))
    ctx.opts.cap, ctx.opts.join = {"round": (1, 1), "square": (2, 0), "butt": (0, 0)}[caps]
    if mode == "layer" and not any(isinstance(op, _Path) and op.layer for op in ctx.ops):
        _assign_layers(ctx, dict(rng.choice(_LAYER_SCHEMES)))
    colours: dict[Optional[str], Color] = {}
    if mode == "layer":
        palette = list(_PALETTE)
        rng.shuffle(palette)
        for i, name in enumerate(ctx.layers):
            colours[name] = palette[i % len(palette)]
        # sheet furniture usually stays black
        for op in ctx.ops:
            if _layer_key(op) == "sheet" and rng.random() < 0.7:
                colours[op.layer] = BLACK
                break
    for op in ctx.ops:
        if isinstance(op, _Path):
            if op.role in ("pipe", "through", "arrow"):
                op.width = max(0.1, op.width * kp) if op.role != "arrow" else op.width
            elif op.role == "signal":
                op.width = signal_w
        elif isinstance(op, _Sym) and op.role != "legend":
            op.width = sw
        elif isinstance(op, _Circ):
            op.r = max(op.r, 0.8 * pw)
        col = colours.get(op.layer, BLACK)
        if isinstance(op, _Text):
            op.color = col
        elif isinstance(op, _Path):
            op.color = col if op.color is not None else None
            if op.fill is not None:
                op.fill = col
        elif isinstance(op, (_Sym, _Circ)):
            op.color = col
            if isinstance(op, _Circ) and op.fill is not None:
                op.fill = col
    # duplicate overdraw: some entities written twice (immediately or at the end of the stream)
    rate = rng.uniform(0.03, 0.12)
    out: list = []
    tail: list = []
    n_dup = 0
    for op in ctx.ops:
        out.append(op)
        if not isinstance(op, _Circ) and rng.random() < rate:
            (out if rng.random() < 0.5 else tail).append(replace(op))
            n_dup += 1
    graphics = [op for op in out + tail if not isinstance(op, _Text)]
    ctx.ops = graphics + [op for op in out + tail if isinstance(op, _Text)]
    ctx.params["styled_lines"] = {"pipe_width": round(pw, 3), "symbol_width": round(sw, 3),
                                  "symbols_thicker": thick_symbols, "signal_width": round(signal_w, 3),
                                  "colour": mode, "caps": caps, "duplicates": n_dup}


def _apply_scaled(ctx: _Ctx, rng: random.Random) -> None:
    W, H = ctx.doc.pages[0].width, ctx.doc.pages[0].height
    f = rng.choice((0.6, 1.5))
    paper = ctx.spec.paper
    if f > 1.0 and paper == "A1":
        f = min(A0[0] / W, A0[1] / H)
        ctx.opts.page_size = A0
        ctx.opts.offset = ((A0[0] - W * f) / 2, (A0[1] - H * f) / 2)
        new_paper = "A0"
    else:
        ctx.opts.page_size = (round(W * f, 3), round(H * f, 3))
        new_paper = f"{paper}x{f:g}"
    ctx.opts.scale = f
    ctx.params["scaled"] = {"factor": round(f, 5), "paper": new_paper, "page": list(ctx.opts.page_size)}


def _apply_rotated(ctx: _Ctx, rng: random.Random) -> None:
    ctx.opts.rotation = rng.choice((90, 270))
    ctx.params["rotated_page"] = {"rotate": ctx.opts.rotation}


def _apply_qu(ctx: _Ctx, rng: random.Random) -> None:
    ctx.opts.qu = True
    ctx.params["qu_bowties"] = {"bowtie_symbols": sum(1 for o in ctx.doc.objects if o.cls in BOWTIE_CLASSES)}


def _apply_curves(ctx: _Ctx, rng: random.Random) -> None:
    ctx.opts.curve_n = rng.randint(24, 48)
    ctx.params["polyline_curves"] = {"segments_per_circle": ctx.opts.curve_n}


_APPLY = {
    "yard_symbols": _apply_yard, "fragmented_text": _apply_text, "pipes_through": _apply_through,
    "layers_off": _apply_layers_off, "styled_lines": _apply_styled, "qu_bowties": _apply_qu,
    "polyline_curves": _apply_curves, "scaled": _apply_scaled, "rotated_page": _apply_rotated,
}

# --------------------------------------------------------------------------- rendering


def _dashes(d: Optional[str], f: float) -> Optional[str]:
    if not d or abs(f - 1.0) < 1e-12:
        return d
    m = re.match(r"\s*\[([^\]]*)\]\s*([-\d.]+)?", d)
    if not m:
        return d
    nums = [float(x) * f for x in m.group(1).split()]
    phase = float(m.group(2) or 0.0) * f
    return "[" + " ".join(f"{x:g}" for x in nums) + f"] {phase:g}"


def _bezier(p0, p1, p2, p3, n: int) -> list[tuple[float, float]]:
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
                    u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1]))
    return out


def _pieces(text: str, mode: str, font: str, size: float) -> list[tuple[str, float]]:
    if mode == "whole":
        return [(text, 0.0)]
    out = []
    if mode == "char":
        for i, ch in enumerate(text):
            if not ch.isspace():
                out.append((ch, _tw(text[:i], font, size)))
        return out
    for m in re.finditer(r"\S+", text):
        out.append((m.group(0), _tw(text[:m.start()], font, size)))
    return out


def _render(ctx: _Ctx) -> bytes:
    o = ctx.opts
    W, H = ctx.spec.width, ctx.spec.height
    s = o.scale
    ox, oy = o.offset
    Wd, Hd = o.page_size or (W * s, H * s)
    rot = o.rotation % 360
    pdf = pymupdf.open()
    page = pdf.new_page(width=Hd, height=Wd) if rot in (90, 270) else pdf.new_page(width=Wd, height=Hd)
    ocg = {name: pdf.add_ocg(name, on=on) for name, on in ctx.layers.items()}

    def oc(layer: Optional[str]) -> int:
        return ocg.get(layer, 0) if layer else 0

    def P(x: float, y: float) -> pymupdf.Point:
        X, Y = x * s + ox, y * s + oy
        if rot == 90:
            return pymupdf.Point(Y, Wd - X)
        if rot == 270:
            return pymupdf.Point(Hd - Y, X)
        if rot == 180:
            return pymupdf.Point(Wd - X, Hd - Y)
        return pymupdf.Point(X, Y)

    sh = page.new_shape()
    cap, join = o.cap, o.join

    def circle(cx: float, cy: float, r: float) -> None:
        if o.curve_n:
            n = o.curve_n
            pts = [P(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
            sh.draw_polyline(pts + [pts[0]])
        else:
            sh.draw_circle(P(cx, cy), r * s)

    def path(p: _Path) -> None:
        for g in p.segs:
            if g[0] == "l":
                sh.draw_line(P(g[1], g[2]), P(g[3], g[4]))
            elif o.curve_n:
                pts = _bezier((g[1], g[2]), (g[3], g[4]), (g[5], g[6]), (g[7], g[8]), max(3, o.curve_n // 4))
                for a, b in zip(pts, pts[1:]):
                    sh.draw_line(P(*a), P(*b))
            else:
                sh.draw_bezier(P(g[1], g[2]), P(g[3], g[4]), P(g[5], g[6]), P(g[7], g[8]))
        sh.finish(width=p.width * s, color=p.color, fill=p.fill, dashes=_dashes(p.dashes, s), closePath=p.closed,
                  lineCap=cap, lineJoin=join, oc=oc(p.layer))

    def symbol(sd: _Sym) -> None:
        spec = get_symbol(sd.cls)
        items = _local_items(sd.cls, o)
        if sd.wipeout:
            for p, flags in items:
                if "fill_only" in flags:
                    continue
                q = _xf(p, sd.pl)
                if isinstance(q, Circle):
                    circle(q.cx, q.cy, q.r)
                elif isinstance(q, Poly) and q.closed:
                    sh.draw_polyline([P(x, y) for x, y in q.pts])
                else:
                    continue
                sh.finish(width=0, color=None, fill=WHITE, closePath=True, oc=oc(sd.layer))
        acts = [(p, frozenset()) for p in ACTUATOR_PRIMS.get(sd.act, ())] if spec.category == Category.VALVE else []
        for p, flags in items + acts:
            q = _xf(p, sd.pl)
            fill = None
            close = False
            color: Optional[Color] = sd.color
            if isinstance(q, Line):
                sh.draw_line(P(q.x0, q.y0), P(q.x1, q.y1))
            elif isinstance(q, Poly):
                pts = [P(x, y) for x, y in q.pts]
                if "qu" in flags:
                    sh.draw_polyline(pts + [pts[0]])  # explicitly closed: PyMuPDF reports one 'qu' item
                else:
                    sh.draw_polyline(pts)
                    close = q.closed
                fill = sd.color if q.fill else None
                if "fill_only" in flags:
                    color, close = None, True
            elif isinstance(q, Circle):
                circle(q.cx, q.cy, q.r)
                fill = sd.color if q.fill else None
                close = bool(o.curve_n)
            sh.finish(width=sd.width * s, color=color, fill=fill, closePath=close, lineCap=cap, lineJoin=join,
                      oc=oc(sd.layer))

    def dot(c: _Circ) -> None:
        circle(c.x, c.y, c.r)
        sh.finish(width=c.width * s, color=c.color, fill=c.fill, closePath=True, lineCap=cap, lineJoin=join,
                  oc=oc(c.layer))

    def text(t: _Text) -> None:
        d = (1.0, 0.0) if t.rotate == 0 else (0.0, -1.0)
        prot = (t.rotate + rot) % 360
        for piece, off in _pieces(t.text, t.mode, t.font, t.size):
            sh.insert_text(P(t.x + d[0] * off, t.y + d[1] * off), piece, fontsize=t.size * s, fontname=t.font,
                           rotate=prot, color=t.color, oc=oc(t.layer))
        if t.hidden_font:
            sh.insert_text(P(t.x, t.y), t.text, fontsize=t.size * s, fontname=t.hidden_font, rotate=prot,
                           color=t.color, render_mode=3, oc=oc(t.layer))

    for op in ctx.ops:
        if isinstance(op, _Path):
            path(op)
        elif isinstance(op, _Sym):
            symbol(op)
        elif isinstance(op, _Circ):
            dot(op)
        else:
            text(op)
    sh.commit()
    if rot:
        page.set_rotation(rot)
    meta = {k: "" for k in ("title", "author", "subject", "keywords", "creator", "producer")}
    meta.update({k: v for k, v in ctx.spec.metadata.items() if k in meta})
    meta.update({"creationDate": "", "modDate": ""})
    pdf.set_metadata(meta)
    try:
        return pdf.tobytes(garbage=3, deflate=True, no_new_id=True)
    finally:
        pdf.close()


# --------------------------------------------------------------------------- ground truth


def _finalize(ctx: _Ctx, components: list[str]) -> PIDDocument:
    doc = ctx.doc
    hidden = sorted({op.gt for op in ctx.ops if isinstance(op, _Text) and op.gt is not None and op.hidden_font
                     and op.gt not in ctx.drop_texts})
    hidden_ids = [doc.texts[k].id for k in hidden]
    if ctx.drop_texts:
        keep = [t for k, t in enumerate(doc.texts) if k not in ctx.drop_texts]
        idmap = {t.id: f"t{k + 1}" for k, t in enumerate(keep)}
        for t in keep:
            t.id = idmap[t.id]
        for o in doc.objects:
            o.text_ids = [idmap[x] for x in o.text_ids if x in idmap]
        for p in doc.pipes:
            p.text_ids = [idmap[x] for x in p.text_ids if x in idmap]
        doc.texts = keep
        hidden_ids = [idmap[x] for x in hidden_ids]
    if hidden_ids:
        ctx.params.setdefault("fragmented_text", {})["hidden_text_ids"] = hidden_ids
    o = ctx.opts
    f, (dx, dy) = o.scale, o.offset
    if abs(f - 1.0) > 1e-12 or dx or dy:
        def pt(p: Point) -> Point:
            return Point(x=_r(p.x * f + dx), y=_r(p.y * f + dy))

        def bx(b: BBox) -> BBox:
            return BBox(x0=_r(b.x0 * f + dx), y0=_r(b.y0 * f + dy), x1=_r(b.x1 * f + dx), y1=_r(b.y1 * f + dy))

        for ob in doc.objects:
            ob.bbox = bx(ob.bbox)
            for port in ob.ports:
                port.position = pt(port.position)
        for j in doc.junctions:
            j.position = pt(j.position)
        for p in doc.pipes:
            p.points = [pt(q) for q in p.points]
        for t in doc.texts:
            t.bbox = bx(t.bbox)
            t.size = round(t.size * f, 3)
        for d in ctx.off_texts + ctx.off_objects:
            b = d["bbox"]
            d["bbox"] = [b[0] * f + dx, b[1] * f + dy, b[2] * f + dx, b[3] * f + dy]
        page = doc.pages[0]
        page.width, page.height = o.page_size or (page.width * f, page.height * f)
        if page.drawing_area is not None:
            page.drawing_area = bx(page.drawing_area)
        doc.meta["symbol_scale"] = round(float(doc.meta.get("symbol_scale", 18.0)) * f, 3)
        doc.meta["paper"] = ctx.params.get("scaled", {}).get("paper", doc.meta.get("paper"))
        doc.info.extra["paper"] = doc.meta["paper"]
    doc.pages[0].rotation = o.rotation
    for d in ctx.off_texts + ctx.off_objects:
        d["bbox"] = [_r(v) for v in d["bbox"]]
    base_source = doc.source
    doc.source = f"{base_source}:{ctx.variant}"
    variant_meta: dict[str, Any] = {"name": ctx.variant, "components": components, "base_source": base_source,
                                    "params": ctx.params}
    if ctx.layers:
        variant_meta["layers"] = dict(ctx.layers)
    if ctx.off_texts or ctx.off_objects or "layers_off" in components:
        variant_meta["off_layer_texts"] = ctx.off_texts
        variant_meta["off_layer_objects"] = ctx.off_objects
    doc.meta["variant"] = variant_meta
    return doc


# --------------------------------------------------------------------------- public API


def _rng(seed: int, *parts: Any) -> random.Random:
    return random.Random(":".join(["shippid-variants", str(seed)] + [str(p) for p in parts]))


def _components(seed: int, variant: str) -> list[str]:
    if variant == "plain":
        return []
    if variant == "combined":
        rng = _rng(seed, "combined")
        picked = set(rng.sample(COMBINABLE, rng.choice((2, 3, 4))))
        return [v for v in COMBINABLE if v in picked]
    return [variant]


def _base(seed: int, system: Optional[SystemType], need_legend: bool) -> tuple[PIDDocument, DrawingSpec]:
    if not need_legend:
        return generate(seed, system)
    last: Optional[Exception] = None
    for k in range(12):
        style = random_style(_rng(seed, "legend-style", k))
        style.legend = True
        try:
            doc, spec = generate(seed, system, style)
        except RuntimeError as exc:  # a fixed style can fail where the random ones succeed
            last = exc
            continue
        n_leg = len(spec.symbols) - len(doc.objects)
        legend = {sd.cls for sd in spec.symbols[:n_leg]}
        needed = {o.cls for o in doc.objects if o.cls in YARD_DESIGNS}
        if n_leg > 0 and needed <= legend:
            return doc, spec
    raise RuntimeError(f"no drawing with a complete legend for seed={seed} system={system}: {last}")


def make_variant(seed: int, variant: str, system: Optional[SystemType | str] = None) -> tuple[PIDDocument, bytes]:
    """One out-of-distribution drawing: ``(ground truth, PDF bytes)``, deterministic per arguments.

    The base drawing is :func:`shippid.datasets.synthetic.generate` ``(seed, system)`` (for
    ``yard_symbols`` a seeded style with a legend block); see :data:`VARIANTS` for the variants."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose one of {list(VARIANTS)}")
    system = SystemType(system) if system is not None else None
    comps = _components(seed, variant)
    doc, spec = _base(seed, system, "yard_symbols" in comps)
    ctx = _scene(doc, spec, seed, variant)
    for name in COMBINABLE:
        if name in comps:
            _APPLY[name](ctx, _rng(seed, variant, name))
    pdf = _render(ctx)
    return _finalize(ctx, comps), pdf


def _variant_dirs(out: Path, variants: Sequence[str]) -> None:
    for v in variants:
        (out / v).mkdir(parents=True, exist_ok=True)


def _build_one(job: tuple[str, str, int, int, int, str]) -> dict[str, Any]:
    out_dir, variant, index, base_seed, seed, system = job
    iid = f"{variant}_{base_seed:04d}_{index:04d}"
    out = Path(out_dir) / variant
    t0 = time.perf_counter()
    try:
        doc, pdf = make_variant(seed, variant, SystemType(system))
        (out / f"{iid}.pdf").write_bytes(pdf)
        doc.source = f"{iid}.pdf"
        (out / f"{iid}.json").write_text(doc.to_json(), encoding="utf-8")
        page = doc.pages[0]
        vm = doc.meta["variant"]
        return {
            "id": iid, "index": index, "seed": seed, "split": "test", "pdf": f"{iid}.pdf", "gt": f"{iid}.json",
            "variant": variant, "components": vm["components"], "params": vm["params"],
            "system": doc.info.system.value, "ship_type": doc.info.ship_type, "hull_no": doc.info.hull_no,
            "drawing_no": doc.info.drawing_no, "paper": doc.meta.get("paper"), "width": page.width,
            "height": page.height, "rotation": page.rotation,
            "counts": {"objects": len(doc.objects), "pipes": len(doc.pipes), "junctions": len(doc.junctions),
                       "texts": len(doc.texts),
                       **dict(sorted(Counter(o.category.value for o in doc.objects).items()))},
            "classes": dict(sorted(Counter(o.cls.value for o in doc.objects).items())), "bytes": len(pdf),
            "seconds": round(time.perf_counter() - t0, 4),
        }
    except Exception as exc:  # recorded in the summary, never silently dropped
        log.exception("variant item %s failed", iid)
        return {"id": iid, "index": index, "seed": seed, "variant": variant, "system": system,
                "error": f"{type(exc).__name__}: {exc}"}


def build_variant_suite(out_dir: str | Path, per_variant: int = 20, seed: int = 1000,
                        variants: Optional[Iterable[str]] = None, workers: Optional[int] = None) -> dict[str, Any]:
    """Write ``per_variant`` drawings of every variant to ``out_dir/<variant>/`` (PDF, ground-truth JSON,
    ``manifest.jsonl`` with split ``"test"``) and ``out_dir/summary.json``; returns the summary.

    Item ``i`` of every variant starts from the same base drawing (seed ``build.item_seed(seed, i)``,
    systems balanced round-robin), so the variants are paired comparisons; ``yard_symbols`` (and any
    ``combined`` item containing it) needs a legend block and therefore starts from a sibling drawing
    of the same seed and system."""
    names = list(variants) if variants is not None else list(VARIANTS)
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    if per_variant < 0:
        raise ValueError("per_variant must be >= 0")
    out = Path(out_dir)
    _variant_dirs(out, names)
    systems = assign_systems(per_variant, seed)
    jobs = [(str(out), v, i, seed, item_seed(seed, i), systems[i].value) for v in names for i in range(per_variant)]
    workers = max(1, int(workers if workers is not None else (os.cpu_count() or 1)))
    t0 = time.perf_counter()
    if workers == 1 or len(jobs) <= 1:
        results = [_build_one(j) for j in jobs]
    else:
        with mp.get_context().Pool(processes=min(workers, len(jobs))) as pool:
            results = list(pool.imap_unordered(_build_one, jobs, chunksize=2))
    seconds = time.perf_counter() - t0
    summary: dict[str, Any] = {"out_dir": str(out), "per_variant": per_variant, "seed": seed, "workers": workers,
                               "seconds": round(seconds, 3), "variants": {}}
    for v in names:
        rows = sorted((r for r in results if r["variant"] == v), key=lambda r: r["index"])
        ok = [r for r in rows if "error" not in r]
        failed = [r for r in rows if "error" in r]
        with (out / v / "manifest.jsonl").open("w", encoding="utf-8") as fh:
            for r in ok:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        comp = Counter(c for r in ok for c in r["components"])
        summary["variants"][v] = {
            "description": VARIANTS[v], "dir": str(out / v), "manifest": str(out / v / "manifest.jsonl"),
            "ok": len(ok), "failed": len(failed), "objects": sum(r["counts"]["objects"] for r in ok),
            "by_system": dict(sorted(Counter(r["system"] for r in ok).items())),
            "by_paper": dict(sorted(Counter(str(r["paper"]) for r in ok).items())),
            "components": dict(sorted(comp.items())),
            "failures": [{"id": r["id"], "error": r["error"]} for r in failed],
        }
    summary["ok"] = sum(s["ok"] for s in summary["variants"].values())
    summary["failed"] = sum(s["failed"] for s in summary["variants"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary
