"""Test fixtures for the vector extractor: small P&ID sheets drawn directly with
:func:`shippid.symbols.draw_symbol` + PyMuPDF, with exact ground truth, and an
evaluator comparing an extracted :class:`PIDDocument` against it.

Independent of :mod:`shippid.datasets` on purpose (no ground-truth leakage from
the dataset generator into the extractor's validation).
"""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pymupdf
from scipy.spatial import cKDTree

from shippid.model import ActuatorType, BBox, Category, PIDDocument, SymbolClass, TextRole
from shippid.symbols import (
    Placement,
    draw_symbol,
    get_symbol,
    placed_bbox,
    placed_ports,
    placed_prims,
    sample_outline,
)

S = SymbolClass
A = ActuatorType
BLACK = (0.0, 0.0, 0.0)


@dataclass
class GTObject:
    key: str
    cls: SymbolClass
    placement: Placement
    actuator: ActuatorType
    bbox: tuple[float, float, float, float]
    ports: dict[str, tuple[float, float]]
    tag: Optional[str] = None
    name: Optional[str] = None


@dataclass
class GTText:
    text: str
    role: TextRole
    owner: Optional[str] = None  # object key, or "pipe:<edge index>"
    pos: tuple[float, float] = (0.0, 0.0)


@dataclass
class Style:
    scale: float = 18.0
    sym_width: float = 0.6
    pipe_width: float = 1.0
    font: float = 6.5
    through: bool = False  # draw process lines through inline symbols
    dot_tees: bool = True
    crossing: str = "plain"  # plain | gap | hop
    color: tuple[float, float, float] = BLACK
    # harder variations
    explode: bool = False  # every straight edge as its own path (CAD exports that explode polylines)
    eq_scale: float = 1.0  # equipment drawn at this multiple of the valve scale
    touching: float = 0.0  # probability that an inline symbol is bolted flange-to-flange to the previous one
    arrows: bool = False  # flow arrows on pipes
    vertical: bool = False  # add a vertical train (rotated symbols, rotated line numbers)
    dash_exploded: bool = False  # dashed signal lines drawn as separate dash segments
    dotted_cross: bool = False  # the crossing line gets a dot (connected 4-way cross)


class Sheet:
    """A drawing under construction with its ground truth."""

    def __init__(self, width: float = 1190.0, height: float = 842.0, style: Optional[Style] = None, frame: bool = True):
        self.style = style or Style()
        self.doc = pymupdf.open()
        self.page = self.doc.new_page(width=width, height=height)
        self.shape = self.page.new_shape()
        self.W, self.H = width, height
        self.objects: dict[str, GTObject] = {}
        self.edges: list[tuple[tuple[str, str], tuple[str, str]]] = []  # node = ("o", key) | ("j", key)
        self.texts: list[GTText] = []
        self.junctions: dict[str, tuple[float, float]] = {}
        self.signal_edges: list[tuple[tuple[str, str], tuple[str, str]]] = []
        self._pending_text: list[tuple] = []
        self._n = 0
        self.frame_box = None
        if frame:
            self.draw_frame()

    # -- primitives ------------------------------------------------------------
    def key(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def line(self, pts, width: Optional[float] = None, dashed: bool = False, color=None) -> None:
        width = width or self.style.pipe_width
        color = color or self.style.color
        if dashed and self.style.dash_exploded:
            for a, b in zip(pts, pts[1:]):
                L = math.dist(a, b)
                n = max(1, int(L / 6.0))
                for k in range(n):
                    t0, t1 = k / n, (k + 0.65) / n
                    self.shape.draw_line(pymupdf.Point(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                                         pymupdf.Point(a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1))
                    self.shape.finish(width=width, color=color, closePath=False)
            return
        if self.style.explode:
            for a, b in zip(pts, pts[1:]):
                self.shape.draw_line(pymupdf.Point(*a), pymupdf.Point(*b))
                self.shape.finish(width=width, color=color, closePath=False, dashes="[4 2] 0" if dashed else None)
            return
        self.shape.draw_polyline([pymupdf.Point(*p) for p in pts])
        self.shape.finish(width=width, color=color, closePath=False, dashes="[4 2] 0" if dashed else None)

    def arrow(self, x: float, y: float, dx: float, dy: float) -> None:
        s = 0.22 * self.style.scale
        nx, ny = -dy, dx
        tip = (x + dx * s, y + dy * s)
        b1 = (x - dx * s + nx * 0.45 * s, y - dy * s + ny * 0.45 * s)
        b2 = (x - dx * s - nx * 0.45 * s, y - dy * s - ny * 0.45 * s)
        self.shape.draw_polyline([pymupdf.Point(*tip), pymupdf.Point(*b1), pymupdf.Point(*b2)])
        self.shape.finish(width=0.3, color=self.style.color, fill=self.style.color, closePath=True)

    def text(self, x: float, y: float, s: str, role: TextRole, owner: Optional[str] = None, size: Optional[float] = None,
             rotate: int = 0) -> None:
        self._pending_text.append((x, y, s, size or self.style.font, rotate))
        self.texts.append(GTText(s, role, owner, (x, y)))

    def symbol(self, cls: SymbolClass, x: float, y: float, rotation: int = 0, mirrored: bool = False,
               actuator: ActuatorType = A.NONE, tag: Optional[str] = None, stretch=(1.0, 1.0), scale: Optional[float] = None,
               name: Optional[str] = None, tag_pos: Optional[str] = None) -> GTObject:
        spec = get_symbol(cls)
        if scale is None:
            scale = self.style.scale * (self.style.eq_scale if spec.category == Category.EQUIPMENT and not spec.stretchable else 1.0)
        pl = Placement(x, y, scale=scale, rotation=rotation, mirrored=mirrored, stretch=stretch)
        if self.style.explode:
            self._draw_exploded(spec, pl, actuator)
        else:
            draw_symbol(self.shape, spec, pl, actuator, width=self.style.sym_width)
        bb = placed_bbox(spec, pl, actuator)
        ports = {n: p for n, p, _ in placed_ports(spec, pl)}
        o = GTObject(self.key("o"), cls, pl, actuator, bb, ports, tag, name)
        self.objects[o.key] = o
        f = self.style.font
        if tag:
            if tag_pos == "right" or (tag_pos is None and rotation % 180 == 90):
                self.text(bb[2] + 0.25 * f, (bb[1] + bb[3]) / 2 + 0.35 * f, tag, TextRole.TAG, o.key)
            elif tag_pos == "below":
                self.text((bb[0] + bb[2]) / 2 - 0.3 * f * len(tag) / 1.0, bb[3] + 1.1 * f, tag, TextRole.TAG, o.key)
            else:
                self.text((bb[0] + bb[2]) / 2 - 0.27 * f * len(tag), bb[1] - 0.45 * f, tag, TextRole.TAG, o.key)
        return o

    def _draw_exploded(self, spec, pl: Placement, actuator: ActuatorType) -> None:
        from shippid.symbols import Circle, Line, Poly

        sh, w = self.shape, self.style.sym_width
        for p in placed_prims(spec, pl, actuator):
            if isinstance(p, Line):
                sh.draw_line(pymupdf.Point(p.x0, p.y0), pymupdf.Point(p.x1, p.y1))
                sh.finish(width=w, color=BLACK, closePath=False)
            elif isinstance(p, Poly) and not p.fill:
                ring = list(p.pts) + ([p.pts[0]] if p.closed else [])
                for a, b in zip(ring, ring[1:]):
                    sh.draw_line(pymupdf.Point(*a), pymupdf.Point(*b))
                    sh.finish(width=w, color=BLACK, closePath=False)
            elif isinstance(p, Poly):
                sh.draw_polyline([pymupdf.Point(*q) for q in p.pts])
                sh.finish(width=w, color=BLACK, fill=BLACK, closePath=True)
            elif isinstance(p, Circle):
                sh.draw_circle(pymupdf.Point(p.cx, p.cy), p.r)
                sh.finish(width=w, color=BLACK, fill=BLACK if p.fill else None)

    def port(self, o: GTObject, name: str) -> tuple[float, float]:
        return o.ports[name]

    def connect(self, a: tuple[str, str], b: tuple[str, str], pts, dashed: bool = False) -> None:
        self.line(pts, dashed=dashed)
        (self.signal_edges if dashed else self.edges).append((a, b))

    def junction(self, x: float, y: float, dot: Optional[bool] = None) -> str:
        k = self.key("j")
        self.junctions[k] = (x, y)
        if dot if dot is not None else self.style.dot_tees:
            self.shape.draw_circle(pymupdf.Point(x, y), max(1.6, 0.11 * self.style.scale))
            self.shape.finish(width=0.5, color=self.style.color, fill=self.style.color)
        return k

    def crossing_line(self, pts_before_after, axis_cross: tuple[float, float], horizontal: bool) -> None:
        """A straight line crossing another line at ``axis_cross`` without connecting."""
        (x0, y0), (x1, y1) = pts_before_after
        cx, cy = axis_cross
        g = 0.18 * self.style.scale
        if self.style.crossing == "plain":
            self.line([(x0, y0), (x1, y1)])
        elif self.style.crossing == "gap":
            if horizontal:
                self.line([(x0, y0), (cx - g, cy)])
                self.line([(cx + g, cy), (x1, y1)])
            else:
                self.line([(x0, y0), (cx, cy - g)])
                self.line([(cx, cy + g), (x1, y1)])
        else:  # hop: semicircle over the crossed line
            sh = self.shape
            if horizontal:
                self.line([(x0, y0), (cx - g, cy)])
                sh.draw_bezier(pymupdf.Point(cx - g, cy), pymupdf.Point(cx - g, cy - 1.33 * g), pymupdf.Point(cx + g, cy - 1.33 * g),
                               pymupdf.Point(cx + g, cy))
                sh.finish(width=self.style.pipe_width, color=self.style.color, closePath=False)
                self.line([(cx + g, cy), (x1, y1)])
            else:
                self.line([(x0, y0), (cx, cy - g)])
                sh.draw_bezier(pymupdf.Point(cx, cy - g), pymupdf.Point(cx + 1.33 * g, cy - g), pymupdf.Point(cx + 1.33 * g, cy + g),
                               pymupdf.Point(cx, cy + g))
                sh.finish(width=self.style.pipe_width, color=self.style.color, closePath=False)
                self.line([(cx, cy + g), (x1, y1)])

    # -- furniture -------------------------------------------------------------
    def draw_frame(self) -> None:
        m, mi = 14.0, 28.0
        sh = self.shape
        sh.draw_rect(pymupdf.Rect(m, m, self.W - m, self.H - m))
        sh.finish(width=0.8, color=BLACK)
        sh.draw_rect(pymupdf.Rect(mi, mi, self.W - mi, self.H - mi))
        sh.finish(width=1.2, color=BLACK)
        # zone markers
        for i in range(1, 8):
            x = mi + (self.W - 2 * mi) * i / 8
            sh.draw_line(pymupdf.Point(x, m), pymupdf.Point(x, mi))
            sh.finish(width=0.5, color=BLACK)
            self._pending_text.append((x - (self.W - 2 * mi) / 16 - 2, m + 10, str(i), 6, 0))
            self.texts.append(GTText(str(i), TextRole.TITLE_BLOCK))
        self.frame_box = (mi, mi, self.W - mi, self.H - mi)

    def title_block(self, title: str, drawing_no: str, hull: str, ship_type: str) -> tuple[float, float, float, float]:
        x1, y1 = self.frame_box[2], self.frame_box[3]
        w, h = 300.0, 110.0
        x0, y0 = x1 - w, y1 - h
        sh = self.shape
        sh.draw_rect(pymupdf.Rect(x0, y0, x1, y1))
        sh.finish(width=1.0, color=BLACK)
        rows = [y0 + 22, y0 + 44, y0 + 66, y0 + 88]
        for r in rows:
            sh.draw_line(pymupdf.Point(x0, r), pymupdf.Point(x1, r))
            sh.finish(width=0.5, color=BLACK)
        sh.draw_line(pymupdf.Point(x0 + 80, y0), pymupdf.Point(x0 + 80, y1))
        sh.finish(width=0.5, color=BLACK)
        cells = [("SHIPYARD", "SEAFORGE HEAVY INDUSTRIES"), ("SHIP TYPE", ship_type), ("HULL NO.", hull),
                 ("TITLE", title), ("DWG. NO.", drawing_no)]
        ys = [y0] + rows
        for (lab, val), ya in zip(cells, ys):
            self._pending_text.append((x0 + 4, ya + 14, lab, 6, 0))
            self._pending_text.append((x0 + 86, ya + 15, val, 8 if lab == "TITLE" else 7, 0))
            self.texts.append(GTText(lab, TextRole.TITLE_BLOCK))
            self.texts.append(GTText(val, TextRole.TITLE_BLOCK))
        return (x0, y0, x1, y1)

    def notes(self, x: float, y: float, items: list[str]) -> None:
        self._pending_text.append((x, y, "NOTES:", 7, 0))
        self.texts.append(GTText("NOTES:", TextRole.NOTE))
        for i, s in enumerate(items):
            line = f"{i + 1}. {s}"
            self._pending_text.append((x, y + 11 * (i + 1), line, 6, 0))
            self.texts.append(GTText(line, TextRole.NOTE))

    def legend(self, x: float, y: float, classes: list[SymbolClass]) -> None:
        sc = self.style.scale * 0.8
        h = 24 + 1.6 * sc * len(classes)
        sh = self.shape
        sh.draw_rect(pymupdf.Rect(x, y, x + 190, y + h))
        sh.finish(width=0.8, color=BLACK)
        self._pending_text.append((x + 6, y + 14, "LEGEND", 8, 0))
        self.texts.append(GTText("LEGEND", TextRole.LEGEND))
        for i, c in enumerate(classes):
            cy = y + 30 + 1.6 * sc * i
            draw_symbol(self.shape, get_symbol(c), Placement(x + 25, cy, scale=sc), width=self.style.sym_width)
            lab = get_symbol(c).label_en.upper()
            self._pending_text.append((x + 55, cy + 2.5, lab, 6, 0))
            self.texts.append(GTText(lab, TextRole.LEGEND))

    def finish(self) -> pymupdf.Page:
        self.shape.commit()
        for x, y, s, size, rot in self._pending_text:
            self.page.insert_text((x, y), s, fontsize=size, rotate=rot, color=BLACK)
        self._pending_text = []
        return self.page

    def save(self, path) -> None:
        self.doc.save(str(path))


# --------------------------------------------------------------------------- random sheets

INLINE = [
    (S.GATE_VALVE, [A.HANDWHEEL, A.NONE]), (S.GLOBE_VALVE, [A.HANDWHEEL]), (S.BALL_VALVE, [A.NONE, A.HANDWHEEL]),
    (S.BUTTERFLY_VALVE, [A.HYDRAULIC, A.HANDWHEEL, A.NONE]), (S.CHECK_VALVE, [A.NONE]), (S.SDNR_VALVE, [A.HANDWHEEL]),
    (S.GATE_VALVE, [A.ELECTRIC, A.QUICK_CLOSING]), (S.GLOBE_VALVE, [A.DIAPHRAGM]),
    (S.STRAINER, [A.NONE]), (S.FILTER, [A.NONE]), (S.DUPLEX_FILTER, [A.NONE]), (S.FLOW_METER, [A.NONE]),
    (S.REDUCER, [A.NONE]), (S.FLEXIBLE_HOSE, [A.NONE]), (S.SPECTACLE_FLANGE, [A.NONE]), (S.HEATER, [A.NONE]),
    (S.PUMP_SCREW, [A.NONE]), (S.PURIFIER, [A.NONE]), (S.COMPRESSOR, [A.NONE]), (S.BWTS_UNIT, [A.NONE]),
]
SOURCES = [S.TANK, S.SEA_CHEST, S.OFF_PAGE_CONNECTOR, S.AIR_RECEIVER]
SINKS = [S.TANK, S.OVERBOARD, S.ENGINE, S.OFF_PAGE_CONNECTOR, S.OILY_WATER_SEPARATOR]
SYS_CODES = ["BA", "BG", "FO", "LO", "SW", "CF"]
NAMES = {S.TANK: "NO.{n} W.B.TK", S.ENGINE: "M/E", S.OILY_WATER_SEPARATOR: "OILY WATER SEP.", S.AIR_RECEIVER: "AIR RCVR",
         S.SEA_CHEST: "SEA CHEST"}


def random_style(rng: random.Random, hard: bool = False) -> Style:
    st = Style(scale=rng.uniform(14, 22), sym_width=rng.uniform(0.3, 0.8), pipe_width=rng.uniform(0.5, 1.6),
               font=rng.uniform(5.5, 7.5), through=rng.random() < 0.4, dot_tees=rng.random() < 0.7,
               crossing=rng.choice(["plain", "gap", "hop"]))
    if hard:
        st.explode = rng.random() < 0.4
        st.eq_scale = rng.choice([1.0, 1.0, 1.3])
        st.touching = 0.2 if not st.through else 0.0
        st.arrows = rng.random() < 0.6
        st.vertical = rng.random() < 0.8
        st.dash_exploded = rng.random() < 0.4
        st.dotted_cross = rng.random() < 0.5
        if rng.random() < 0.5:
            from shippid.catalog import medium_rgb
            from shippid.model import Medium

            st.color = medium_rgb(rng.choice([Medium.SEA_WATER, Medium.FUEL_OIL, Medium.FRESH_WATER, Medium.FIRE_WATER]))
    return st


class _Counters:
    def __init__(self) -> None:
        self.valve = itertools.count(1)
        self.line = itertools.count(1)
        self.instr = itertools.count(101)


def random_sheet(seed: int, n_rows: int = 4, width: float = 1684.0, height: float = 1190.0,
                 style: Optional[Style] = None, hard: bool = False, n_el: tuple[int, int] = (4, 7),
                 furniture: bool = True) -> Sheet:
    """Rows of pump/valve trains from a source to a sink, with a header joining two
    trains (tees), instrument taps with bubbles, a dashed signal line to a control
    valve, a crossing line, line numbers, tags, a title block, notes and a legend.
    ``hard`` adds a vertical train, exploded geometry, flange-to-flange symbols,
    flow arrows, coloured pipes, exploded dashes and a dotted 4-way cross."""
    rng = random.Random(seed)
    st = style or random_style(rng, hard)
    sh = Sheet(width, height, st)
    u = st.scale
    code = rng.choice(SYS_CODES)
    cnt = _Counters()
    x_left = sh.frame_box[0] + 3 * u
    bottom_reserve = 260 if furniture else 120
    row_gap = (height - bottom_reserve) / max(1, n_rows)
    rows_y = [sh.frame_box[1] + 4 * u + i * row_gap for i in range(n_rows)]
    first = rng.randrange(1, n_rows - 1) if n_rows >= 3 else 0
    header_rows = [first, first + 1] if n_rows >= 3 else []  # adjacent rows: the header crosses nothing
    train_ends: dict[int, tuple[float, float, tuple[str, str]]] = {}
    cross = None
    row_right = 0.0

    for ri, y in enumerate(rows_y):
        x = x_left
        src = _place_endpoint(sh, rng, rng.choice(SOURCES), x, y, code, side="src")
        x = src.bbox[2] + 2.0 * u
        cur_node = ("o", src.key)
        cur_pt = _exit_point(src, side="right", y=y)
        if src.cls == S.SEA_CHEST:  # rise from the top outlet to the row
            sh.line([cur_pt, (cur_pt[0], y)])
            cur_pt = (cur_pt[0], y)
        through_run: list = []  # inline objects a single line is drawn through
        seg_start = cur_pt
        prev_inline = False
        for k in range(rng.randint(*n_el)):
            if rng.random() < 0.15:
                # centrifugal pump: suction from the left, discharge up and back down to the row
                pump = sh.symbol(S.PUMP_CENTRIFUGAL, x + 0.5 * u * st.eq_scale, y)
                if rng.random() < 0.5:
                    nm = f"NO.{rng.randint(1, 3)} BALLAST PUMP"
                    sh.text(pump.bbox[0] - 0.5 * u, pump.bbox[3] + 1.2 * st.font, nm, TextRole.LABEL, pump.key)
                    pump.name = nm
                sh.line([seg_start if through_run else cur_pt, pump.ports["suction"]])
                through_run = []
                sh.edges.append((cur_node, ("o", pump.key)))
                dx_, dy_ = pump.ports["discharge"]
                back = (dx_ + 1.8 * u, y)
                sh.line([(dx_, dy_), (dx_, y - 1.7 * u * st.eq_scale), (dx_ + 1.8 * u, y - 1.7 * u * st.eq_scale), back])
                cur_node, cur_pt, seg_start = ("o", pump.key), back, back
                x = back[0] + rng.uniform(1.4, 2.4) * u
                prev_inline = False
                continue
            cls, acts = rng.choice(INLINE)
            act = rng.choice(acts)
            spec = get_symbol(cls)
            mir = rng.random() < 0.3 and act != A.NONE
            tag = f"{code}-V{next(cnt.valve):03d}" if spec.category == Category.VALVE and rng.random() < 0.8 else None
            sc = u * (st.eq_scale if spec.category == Category.EQUIPMENT and not spec.stretchable else 1.0)
            half = (spec.extent[2] - spec.extent[0]) * sc / 2
            if cls == S.HEATER:
                half = 0.7 * sc
            if cls == S.SPECTACLE_FLANGE:
                half = 0.05 * sc
            touching = prev_inline and not through_run and rng.random() < st.touching and not tag
            if touching:
                x = cur_pt[0]
            o = sh.symbol(cls, x + half, y, mirrored=mir, actuator=act, tag=tag, tag_pos="below" if mir else None)
            p_in, p_out = o.ports["in"], o.ports["out"]
            gap = p_in[0] - cur_pt[0]
            if gap > 3.2 * u and rng.random() < 0.6:
                ln = _line_number(rng, code, next(cnt.line))
                mid = (cur_pt[0] + p_in[0]) / 2
                sh.text(mid - 0.27 * st.font * len(ln), y - 0.5 * st.font, ln, TextRole.LINE_NUMBER, f"pipe:{len(sh.edges)}")
            if st.arrows and gap > 2.5 * u and not through_run:
                sh.arrow(cur_pt[0] + 0.5 * gap, y, 1.0, 0.0)
            if touching:
                sh.edges.append((cur_node, ("o", o.key)))
            elif st.through and spec.category in (Category.VALVE, Category.FITTING):
                through_run.append(o)
                sh.edges.append((cur_node, ("o", o.key)))
            else:
                sh.line([seg_start if through_run else cur_pt, p_in])
                through_run = []
                sh.edges.append((cur_node, ("o", o.key)))
            cur_node = ("o", o.key)
            cur_pt = p_out
            if not through_run:
                seg_start = p_out
            prev_inline = True
            x = p_out[0] + rng.uniform(1.6, 3.2) * u
            # an instrument tap on the pipe after this element
            tx = cur_pt[0] + 1.0 * u
            clear = cross is None or abs(tx - cross[0]) > 3.0 * u  # keep the crossing column free
            if rng.random() < 0.25 and spec.category != Category.FITTING and clear:
                j = sh.junction(tx, y, dot=False)
                bub = _bubble(sh, rng, tx, y - 2.6 * u, cnt)
                sh.line([(tx, y), (tx, y - 2.6 * u + 0.4 * u)], width=st.pipe_width * 0.7)
                sh.edges.append((("j", j), ("o", bub.key)))
                sh.line([seg_start if through_run else cur_pt, (tx, y)])
                through_run = []
                sh.edges.append((cur_node, ("j", j)))
                cur_node, cur_pt, seg_start = ("j", j), (tx, y), (tx, y)
                x = max(x, tx + 1.6 * u)
                prev_inline = False
        if ri in header_rows:
            hx = x + 2 * u  # the train ends at a header (tee) instead of a sink
            sh.line([seg_start if through_run else cur_pt, (hx, y)])
            train_ends[ri] = (hx, y, cur_node)
            row_right = max(row_right, hx)
            continue
        snk = _place_endpoint(sh, rng, rng.choice(SINKS), x + 1.5 * u, y, code, side="snk")
        entry = _exit_point(snk, side="left", y=y)
        sh.line([seg_start if through_run else cur_pt, entry])
        if ri == 0:
            cross = ((cur_pt[0] + entry[0]) / 2, len(sh.edges), cur_node, snk.key)
        sh.edges.append((cur_node, ("o", snk.key)))
        row_right = max(row_right, snk.bbox[2])

    if len(header_rows) == 2:  # header joining two trains: vertical line with tees -> common sink
        (xa, ya, na), (xb, yb, nb) = (train_ends[header_rows[0]], train_ends[header_rows[1]])
        hx = max(xa, xb)
        top, bot = sorted([(ya, xa, na), (yb, xb, nb)])
        for (yy, xx, nn) in (top, bot):
            if xx < hx:
                sh.line([(xx, yy), (hx, yy)])
        ja = sh.junction(hx, top[0])
        jb = sh.junction(hx, bot[0])
        sh.edges.append((top[2], ("j", ja)))
        sh.edges.append((bot[2], ("j", jb)))
        sh.line([(hx, top[0]), (hx, bot[0])])
        sh.edges.append((("j", ja), ("j", jb)))
        snk = _place_endpoint(sh, rng, S.TANK, hx + 5 * u, bot[0] + 3.5 * u, code, side="snk")
        sh.line([(hx, bot[0]), (hx, bot[0] + 3.5 * u), _exit_point(snk, "left", bot[0] + 3.5 * u)])
        sh.edges.append((("j", jb), ("o", snk.key)))
        ln = _line_number(rng, code, next(cnt.line))
        sh.text(hx - 0.4 * u, (top[0] + bot[0]) / 2 + 0.25 * st.font * len(ln), ln, TextRole.LINE_NUMBER,
                f"pipe:{len(sh.edges) - 2}", rotate=90)
        row_right = max(row_right, snk.bbox[2])

    if n_rows >= 2 and cross is not None:  # a vertical line crossing the first row
        xx, edge_idx, before, after = cross
        y0 = rows_y[0] - 2 * u
        ymid = rows_y[0] + (rows_y[1] - rows_y[0]) / 2
        a = sh.symbol(S.OFF_PAGE_CONNECTOR, xx, y0 - 0.8 * u, rotation=270)
        b = sh.symbol(S.TANK, xx, ymid + 0.5 * u, stretch=(0.6, 0.5))
        if st.dotted_cross:
            sh.line([(xx, y0), (xx, ymid)])
            jx = sh.junction(xx, rows_y[0], dot=True)
            sh.edges[edge_idx] = (before, ("j", jx))
            sh.edges.extend([(("j", jx), ("o", after)), (("o", a.key), ("j", jx)), (("j", jx), ("o", b.key))])
        else:
            sh.crossing_line(((xx, y0), (xx, ymid)), (xx, rows_y[0]), horizontal=False)
            sh.edges.append((("o", a.key), ("o", b.key)))
        sh.text(xx + 0.6 * u, y0 - 0.8 * u, "TO SLUDGE TK", TextRole.CONNECTOR, a.key)
        a.name = "TO SLUDGE TK"

    if st.vertical:
        vx = max(row_right + 6 * u, width * 0.62)
        if vx < width - 260:
            _vertical_train(sh, rng, vx, rows_y[0] + 3 * u, height - (300 if furniture else 160), code, cnt)

    # control loop: transmitter -> dashed signal -> control valve
    cy = rows_y[-1] + 8.5 * u
    cx = x_left + 4 * u
    src = sh.symbol(S.TANK, cx, cy, stretch=(0.8, 0.8))
    cv = sh.symbol(S.GLOBE_VALVE, cx + 6 * u, cy, actuator=A.DIAPHRAGM)
    snk = sh.symbol(S.OFF_PAGE_CONNECTOR, cx + 12 * u, cy)
    sh.line([_exit_point(src, "right", cy), cv.ports["in"]])
    sh.edges.append((("o", src.key), ("o", cv.key)))
    sh.line([cv.ports["out"], snk.ports["in"]])
    sh.edges.append((("o", cv.key), ("o", snk.key)))
    sh.text(cx + 12 * u - 0.6 * u, cy + 0.2 * u, "TO F.O. SYS", TextRole.CONNECTOR, snk.key, size=st.font * 0.8)
    snk.name = "TO F.O. SYS"
    tt = sh.symbol(S.INSTRUMENT_SHARED, cx + 6 * u, cy - 4 * u)
    f = st.font * 0.9
    sh.text(cx + 6 * u - 0.3 * f * 3, cy - 4 * u - 0.15 * u, "TIC", TextRole.INSTRUMENT, tt.key, size=f)
    sh.text(cx + 6 * u - 0.28 * f * 3, cy - 4 * u + 0.32 * u, "201", TextRole.INSTRUMENT, tt.key, size=f)
    tt.tag = "TIC-201"
    sh.connect(("o", tt.key), ("o", cv.key), [(cx + 6 * u, cy - 4 * u + 0.4 * u), (cx + 6 * u, cv.bbox[1])], dashed=True)
    if furniture:
        sh.title_block(f"{code} SYSTEM DIAGRAM" if code != "BA" else "BALLAST WATER SYSTEM", f"H2310-{code}-001", "H.2310",
                       "300K VLCC")
        sh.notes(sh.frame_box[0] + 10, sh.frame_box[3] - 60, ["ALL VALVES ARE SHOWN IN NORMAL POSITION.",
                                                              "PIPE SIZES IN MM (JIS A)."])
        if rng.random() < 0.7:
            sh.legend(sh.frame_box[2] - 200, sh.frame_box[1] + 10, [S.GATE_VALVE, S.CHECK_VALVE, S.STRAINER])
    sh.finish()
    return sh


def _bubble(sh: Sheet, rng: random.Random, x: float, y: float, cnt: _Counters) -> GTObject:
    u = sh.style.scale
    bub = sh.symbol(rng.choice([S.INSTRUMENT_LOCAL, S.INSTRUMENT_LOCAL, S.INSTRUMENT_PANEL, S.INSTRUMENT_SHARED]), x, y)
    letters = rng.choice(["PI", "TI", "PT", "TT", "LS", "FI"])
    num = f"{next(cnt.instr)}"
    f = sh.style.font * 0.9
    sh.text(x - 0.3 * f * len(letters), y - 0.15 * u, letters, TextRole.INSTRUMENT, bub.key, size=f)
    sh.text(x - 0.28 * f * len(num), y + 0.32 * u, num, TextRole.INSTRUMENT, bub.key, size=f)
    bub.tag = f"{letters}-{num}"
    return bub


VERTICAL_INLINE = [
    (S.GATE_VALVE, [A.HANDWHEEL, A.NONE]), (S.GLOBE_VALVE, [A.HANDWHEEL, A.DIAPHRAGM]), (S.BALL_VALVE, [A.NONE]),
    (S.BUTTERFLY_VALVE, [A.HYDRAULIC, A.HANDWHEEL]), (S.CHECK_VALVE, [A.NONE]), (S.SDNR_VALVE, [A.HANDWHEEL]),
    (S.STRAINER, [A.NONE]), (S.FILTER, [A.NONE]), (S.REDUCER, [A.NONE]), (S.FLOW_METER, [A.NONE]),
]


def _vertical_train(sh: Sheet, rng: random.Random, x: float, y_top: float, y_max: float, code: str, cnt: _Counters) -> None:
    """A train drawn top-down: symbols rotated by 90 deg, tags to the right, rotated line numbers."""
    st = sh.style
    u = st.scale
    src = sh.symbol(S.TANK, x, y_top + 0.8 * u, stretch=(1.0, 0.8))
    name = f"NO.{rng.randint(1, 4)} F.O. TK"
    sh.text(src.bbox[0] + 0.15 * u, src.bbox[1] + 0.9 * u, name, TextRole.LABEL, src.key, size=st.font)
    src.name = name
    cur_pt = (x, src.bbox[3])
    cur_node = ("o", src.key)
    seg_start = cur_pt
    through_run: list = []
    y = cur_pt[1] + 2.2 * u
    while y < y_max - 6 * u:
        cls, acts = rng.choice(VERTICAL_INLINE)
        act = rng.choice(acts)
        spec = get_symbol(cls)
        rot = rng.choice([90, 270]) if cls not in (S.CHECK_VALVE, S.SDNR_VALVE) else 90
        mir = rng.random() < 0.4 and act != A.NONE
        half = (spec.extent[2] - spec.extent[0]) * u / 2
        tag = f"{code}-V{next(cnt.valve):03d}" if spec.category == Category.VALVE and rng.random() < 0.8 else None
        o = sh.symbol(cls, x, y + half, rotation=rot, mirrored=mir, actuator=act)
        p_in, p_out = (o.ports["in"], o.ports["out"]) if rot == 90 else (o.ports["out"], o.ports["in"])
        if tag:
            bb = o.bbox
            # tags go on the side away from the actuator
            act_right = act != A.NONE and bb[2] - x > x - bb[0] + 1e-6
            tx = bb[0] - 0.25 * st.font - 0.56 * st.font * len(tag) if act_right else bb[2] + 0.25 * st.font
            sh.text(tx, (bb[1] + bb[3]) / 2 + 0.35 * st.font, tag, TextRole.TAG, o.key)
            o.tag = tag
        gap = p_in[1] - cur_pt[1]
        if gap > 4.0 * u and rng.random() < 0.7:
            ln = _line_number(rng, code, next(cnt.line))
            L = 0.55 * st.font * len(ln)
            if L < gap - 0.5 * u:
                sh.text(x - 0.35 * u, cur_pt[1] + (gap + L) / 2, ln, TextRole.LINE_NUMBER, f"pipe:{len(sh.edges)}", rotate=90)
        if st.through and spec.category in (Category.VALVE, Category.FITTING):
            through_run.append(o)
        else:
            sh.line([seg_start if through_run else cur_pt, p_in])
            through_run = []
        sh.edges.append((cur_node, ("o", o.key)))
        cur_node, cur_pt = ("o", o.key), p_out
        if not through_run:
            seg_start = p_out
        y = p_out[1] + rng.uniform(2.6, 4.5) * u
    snk = sh.symbol(S.TANK, x, y + 0.6 * u + 0.8 * u, stretch=(1.0, 0.6))
    sh.line([seg_start if through_run else cur_pt, (x, snk.bbox[1])])
    sh.edges.append((cur_node, ("o", snk.key)))


def _line_number(rng: random.Random, code: str, n: int) -> str:
    size = rng.choice([50, 65, 80, 100, 125, 150, 200, 250])
    style = rng.randrange(3)
    if style == 0:
        return f"{size}A-{code}-{n:03d}-1S"
    if style == 1:
        return f"{code}-{n:03d}-{size}A"
    return f"{size}-{code}-{n:03d}"


def _place_endpoint(sh: Sheet, rng: random.Random, cls: SymbolClass, x: float, y: float, code: str, side: str) -> GTObject:
    u = sh.style.scale
    spec = get_symbol(cls)
    if cls == S.TANK:
        st = (rng.uniform(0.8, 1.4), rng.uniform(0.9, 1.6))
        o = sh.symbol(cls, x + 1.5 * u * st[0], y, stretch=st)
        name = NAMES[S.TANK].format(n=rng.randint(1, 6))
        f = sh.style.font
        sh.text(o.bbox[0] + 0.15 * u, (o.bbox[1] + o.bbox[3]) / 2, name, TextRole.LABEL, o.key, size=f)
        o.name = name
        return o
    if cls == S.SEA_CHEST:
        o = sh.symbol(cls, x + 0.8 * u, y + 0.4 * u + 0.8 * u, rotation=0)
        return o
    if cls == S.OFF_PAGE_CONNECTOR:
        rot = 180 if side == "src" else 0
        o = sh.symbol(cls, x + 0.8 * u, y, rotation=rot)
        txt = "FROM SEA" if side == "src" else "TO BILGE"
        f = sh.style.font * 0.75
        sh.text(o.bbox[0] + 0.15 * u + (0.25 * u if side == "src" else 0), y + 0.3 * f, txt, TextRole.CONNECTOR, o.key, size=f)
        o.name = txt
        return o
    if cls == S.AIR_RECEIVER:
        o = sh.symbol(cls, x + 0.4 * u, y, stretch=(1.0, 1.0))
        return o
    if cls == S.OVERBOARD:
        return sh.symbol(cls, x + 0.3 * u, y)
    if cls == S.ENGINE:
        o = sh.symbol(cls, x + 1.5 * u * 0.9, y, stretch=(0.9, 0.9))
        return o
    if cls == S.OILY_WATER_SEPARATOR:
        return sh.symbol(cls, x + 0.8 * u, y + 0.2 * u)
    raise ValueError(cls)


def _exit_point(o: GTObject, side: str, y: float) -> tuple[float, float]:
    """Where a horizontal pipe at height ``y`` meets the object's outline."""
    if "in" in o.ports and side == "left" and o.cls in (S.OFF_PAGE_CONNECTOR, S.OVERBOARD):
        return o.ports["in"]
    if o.cls == S.OFF_PAGE_CONNECTOR and side == "right":
        return (o.bbox[2], y)
    if o.cls == S.SEA_CHEST:
        # rise from the top outlet then go right: approximate with the top port
        return o.ports["out"]
    if o.cls == S.AIR_RECEIVER:
        return (o.bbox[2], y)
    return (o.bbox[2], y) if side == "right" else (o.bbox[0], y)


# --------------------------------------------------------------------------- evaluation


def _geom_signature(cls: SymbolClass, rot: int, mir: bool, act: ActuatorType) -> np.ndarray:
    spec = get_symbol(cls)
    prims = placed_prims(spec, Placement(0, 0, 1, rot, mir), act)
    pts = sample_outline(prims, 0.03)
    fills = [p for p in prims if getattr(p, "fill", False)]
    fpts = [(x + 100, y) for x, y in sample_outline(fills, 0.03)]
    return np.asarray(pts + fpts)


def same_orientation(cls: SymbolClass, a: tuple[int, bool], b: tuple[int, bool], act: ActuatorType) -> bool:
    if a == b:
        return True
    pa, pb = _geom_signature(cls, a[0], a[1], act), _geom_signature(cls, b[0], b[1], act)
    d1, _ = cKDTree(pb).query(pa)
    d2, _ = cKDTree(pa).query(pb)
    return max(d1.max(), d2.max()) < 0.05


@dataclass
class EvalResult:
    n_gt: int = 0
    n_pred: int = 0
    matched: int = 0
    correct_cls: int = 0
    per_class: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))  # cls -> [correct, total]
    orient_ok: int = 0
    orient_n: int = 0
    act_ok: int = 0
    act_n: int = 0
    tag_ok: int = 0
    tag_n: int = 0
    conn_tp: int = 0
    conn_fp: int = 0
    conn_fn: int = 0
    role_ok: int = 0
    role_n: int = 0
    errors: list = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.matched / max(1, self.n_gt)

    @property
    def precision(self) -> float:
        return self.matched / max(1, self.n_pred)

    @property
    def cls_acc(self) -> float:
        return self.correct_cls / max(1, self.matched)

    @property
    def conn_precision(self) -> float:
        return self.conn_tp / max(1, self.conn_tp + self.conn_fp)

    @property
    def conn_recall(self) -> float:
        return self.conn_tp / max(1, self.conn_tp + self.conn_fn)

    def add(self, o: "EvalResult") -> None:
        for k in ("n_gt", "n_pred", "matched", "correct_cls", "orient_ok", "orient_n", "act_ok", "act_n", "tag_ok",
                  "tag_n", "conn_tp", "conn_fp", "conn_fn", "role_ok", "role_n"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        for c, (a, b) in o.per_class.items():
            self.per_class[c][0] += a
            self.per_class[c][1] += b
        self.errors.extend(o.errors)

    def summary(self) -> dict:
        return {
            "objects_gt": self.n_gt, "objects_pred": self.n_pred, "recall@iou0.5": round(self.recall, 4),
            "precision@iou0.5": round(self.precision, 4), "class_acc": round(self.cls_acc, 4),
            "orientation_acc": round(self.orient_ok / max(1, self.orient_n), 4),
            "actuator_acc": round(self.act_ok / max(1, self.act_n), 4), "tag_acc": round(self.tag_ok / max(1, self.tag_n), 4),
            "conn_precision": round(self.conn_precision, 4), "conn_recall": round(self.conn_recall, 4),
            "text_role_acc": round(self.role_ok / max(1, self.role_n), 4),
        }


def _adjacency(edges, obj_nodes: set) -> set[frozenset]:
    """Object pairs connected through junctions only."""
    adj: dict = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    pairs: set[frozenset] = set()
    for o in obj_nodes:
        seen = {o}
        stack = [o]
        while stack:
            n = stack.pop()
            for m in adj[n]:
                if m in seen:
                    continue
                seen.add(m)
                if m[0] == "o":
                    if m != o:
                        pairs.add(frozenset((o[1], m[1])))
                else:
                    stack.append(m)
    return pairs


def evaluate(sheet: Sheet, doc: PIDDocument, iou: float = 0.5) -> EvalResult:
    r = EvalResult()
    gts = list(sheet.objects.values())
    preds = list(doc.objects)
    r.n_gt, r.n_pred = len(gts), len(preds)
    cand = []
    for gi, g in enumerate(gts):
        gb = BBox(x0=g.bbox[0], y0=g.bbox[1], x1=g.bbox[2], y1=g.bbox[3])
        for pi, p in enumerate(preds):
            v = gb.iou(p.bbox)
            if v >= iou:
                cand.append((v, gi, pi))
    cand.sort(reverse=True)
    g2p: dict[int, int] = {}
    used: set[int] = set()
    for v, gi, pi in cand:
        if gi in g2p or pi in used:
            continue
        g2p[gi] = pi
        used.add(pi)
    r.matched = len(g2p)
    key2pid: dict[str, str] = {}
    for gi, g in enumerate(gts):
        r.per_class[g.cls.value][1] += 1
        if gi not in g2p:
            r.errors.append(("missed", g.cls.value, g.key, tuple(round(v) for v in g.bbox)))
            continue
        p = preds[g2p[gi]]
        key2pid[g.key] = p.id
        if p.cls == g.cls:
            r.correct_cls += 1
            r.per_class[g.cls.value][0] += 1
            r.orient_n += 1
            if same_orientation(g.cls, (g.placement.rotation, g.placement.mirrored), (p.rotation, p.mirrored), g.actuator):
                r.orient_ok += 1
            else:
                r.errors.append(("orientation", g.cls.value, g.key, (g.placement.rotation, g.placement.mirrored), (p.rotation, p.mirrored)))
            if get_symbol(g.cls).category == Category.VALVE:
                r.act_n += 1
                if p.actuator == g.actuator:
                    r.act_ok += 1
                else:
                    r.errors.append(("actuator", g.cls.value, g.key, g.actuator.value, p.actuator.value))
        else:
            r.errors.append(("class", g.cls.value, p.cls.value, g.key, p.attributes.get("candidates")))
        if g.tag:
            r.tag_n += 1
            if (p.tag or "").replace(" ", "") == g.tag.replace(" ", ""):
                r.tag_ok += 1
            else:
                r.errors.append(("tag", g.cls.value, g.key, g.tag, p.tag))
    for pi, p in enumerate(preds):
        if pi not in used:
            r.errors.append(("false_positive", p.cls.value, p.id, tuple(round(v) for v in p.bbox.as_tuple())))
    # connectivity: object adjacency through junctions (process + signal)
    gt_pairs = _adjacency(sheet.edges + sheet.signal_edges, {("o", k) for k in sheet.objects})
    gt_pairs = {frozenset(key2pid.get(k, "gt:" + k) for k in pr) for pr in gt_pairs}
    pred_edges = []
    for p in doc.pipes:
        a = ("o", p.start.object_id) if p.start.object_id else (("j", p.start.junction_id) if p.start.junction_id else ("f", p.id + "s"))
        b = ("o", p.end.object_id) if p.end.object_id else (("j", p.end.junction_id) if p.end.junction_id else ("f", p.id + "e"))
        pred_edges.append((a, b))
    pred_pairs = _adjacency(pred_edges, {("o", o.id) for o in preds})
    r.conn_tp = len(gt_pairs & pred_pairs)
    r.conn_fp = len(pred_pairs - gt_pairs)
    r.conn_fn = len(gt_pairs - pred_pairs)
    for pr in sorted(pred_pairs - gt_pairs, key=sorted):
        r.errors.append(("conn_fp", tuple(sorted(pr))))
    for pr in sorted(gt_pairs - pred_pairs, key=sorted):
        r.errors.append(("conn_fn", tuple(sorted(pr))))
    # text roles (match by text content, nearest position)
    pool = defaultdict(list)
    for t in doc.texts:
        pool[t.text.strip().upper()].append(t)
    for gt in sheet.texts:
        cands = pool.get(gt.text.strip().upper(), [])
        if not cands:
            continue
        t = min(cands, key=lambda t: math.hypot(t.bbox.x0 - gt.pos[0], t.bbox.y1 - gt.pos[1]))
        r.role_n += 1
        if t.role == gt.role:
            r.role_ok += 1
        else:
            r.errors.append(("role", gt.text, gt.role.value, t.role.value))
    return r
