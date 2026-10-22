"""Synthetic ship P&ID generator: blueprint topology -> layout -> exact ground truth + drawing spec.

``generate(seed, system=None, style=None) -> (PIDDocument, DrawingSpec)``

Pipeline (all deterministic for a given seed):

1. :func:`shippid.datasets.blueprints.build_topology` gives the stages / trains of a ship system.
2. Every train is laid out in its own local frame (flow along +u): inline symbols are
   oriented so that their main-flow ports face the pipe, pumps get discharge risers,
   instruments hang off tap tees, relief loops / bypasses / cooler secondary connections
   and attachments (level transmitters, receiver safety valves) are added around it.
3. Stages are placed in columns, trains stacked as rows, joined by vertical headers
   (one per header group; positions shifted until no 4-way junctions or near-coincident
   tees remain). The last stage may be drawn as vertical branches off a horizontal main
   (``fan-out``). Spacing is stretched to fill the chosen sheet (A3/A2/A1). A sheet forced with
   ``DrawingStyle.paper`` may shrink the symbol scale to ``FORCED_MIN_SCALE``, switch the title
   block to the other side and lower the topology complexity on retries before failing with an
   error that names the requested sheet.
4. Return lines are routed through free lanes back to source tanks (crossing other lines).
5. The pipe network (edges between ports / free nodes) is merged into ``PipeRun``s between
   ports and tees; tees become ``Junction``s; crossings get hops or gaps in the drawing.
6. Tags, names, connector texts and line numbers are placed collision-free and unambiguously
   nearest to their owner (box gap and text centre, as the extractor associates texts); every
   text gets a font-metric bbox, role and owner. Ground truth reflects the page: an object's
   ``tag`` / ``attributes["name"]`` / ``["rating"]`` only when printed (blueprint values that the
   placer had to drop are kept as ``name_undrawn`` / ``rating_undrawn``); off-page connectors
   carry ``direction`` ("in" | "out") and ``arrow_along_flow`` (see :func:`_connector_directions`).
7. Validation (no overlapping symbols/texts, no pipe through a symbol, no collinear pipe
   overlaps); on failure the next attempt uses a fresh sub-seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Iterable, Optional, Sequence

from ..catalog import medium_rgb
from ..model import (
    ActuatorType,
    BBox,
    Category,
    Endpoint,
    Junction,
    Medium,
    PageInfo,
    PIDDocument,
    PIDObject,
    PipeKind,
    PipeRun,
    Point,
    Port,
    SymbolClass,
    SystemType,
    TextItem,
    TextRole,
)
from ..symbols import Placement, get_symbol, placed_bbox
from .blueprints import Branch, Item, SystemTopology, Tap, Train, build_topology
from .pdf_writer import CircleDraw, DrawingSpec, PathDraw, SymbolDraw, TextDraw
from .sheet import PAPER_ORDER, Sheet, SheetStyle, build_sheet
from .text import font_metrics, line_height, origin_for_box, text_bbox, text_width

S = SymbolClass
A = ActuatorType
Pt = tuple[float, float]
Rect = tuple[float, float, float, float]

ALL_SYSTEMS: tuple[SystemType, ...] = tuple(s for s in SystemType if s != SystemType.UNKNOWN)
MAX_ATTEMPTS = 8
MIN_SCALE, MAX_SCALE = 14.0, 24.0  # symbol scale range (points per symbol unit)
FORCED_MIN_SCALE = 10.0  # lowest symbol scale when DrawingStyle.paper forces a (small) sheet
# a label must be nearest to its owner: other objects / pipes at least this much farther by box gap (pt) ...
NEAR_GAP_MARGIN = 0.5
NEAR_CENTRE_TOL = 1.0  # ... and not more than this much nearer to the text centre (pt)


class LayoutError(RuntimeError):
    """The random layout violated a drawing rule; the generator retries with a new sub-seed."""


# --------------------------------------------------------------------------- style


@dataclass
class DrawingStyle:
    scale: float = 18.0  # symbol scale, points per symbol unit
    inst_scale: float = 1.45  # instrument bubble scale relative to ``scale``
    conn_scale: float = 1.25  # off-page connector scale relative to ``scale``
    symbol_width: float = 0.6
    pipe_width: float = 1.0
    graded_widths: bool = True  # branch / small lines thinner than mains
    tap_width: float = 0.5
    signal_width: float = 0.5
    signal_dashes: str = "[3 2] 0"
    font: str = "helv"
    tag_size: float = 6.0
    label_size: float = 6.5
    lineno_size: float = 5.5
    junction_dots: bool = True
    dot_taps: bool = True
    crossing: str = "hop"  # "hop" | "gap"
    hop_on: str = "h"  # the line that yields at a crossing: "h" | "v"
    hop_radius: float = 3.5
    color_pipes: bool = False  # ISO 14726 colours for process lines
    layers: tuple[str, ...] = ()  # (pipe, symbol, text, frame) OCG names; empty -> no layers
    legend: bool = False
    sheet_layout: str = "right"  # "right" | "bottom"
    notes_box: bool = True
    fanout: str = "down"  # "none" | "down" | "up" (used when the last stage allows it)
    valve_mirror: float = 0.0  # probability that a valve is drawn with its actuator below the pipe
    flow_arrows: bool = False
    spacing: float = 1.0  # base spacing multiplier
    split_pipes: bool = False  # one PDF path per straight segment instead of one polyline per pipe
    interleave: bool = False  # graphics emitted in mixed order instead of pipes-then-symbols
    paper: Optional[str] = None  # force "A1" | "A2" | "A3"; default: smallest sheet that fits. A forced sheet
    # may shrink the symbol scale down to FORCED_MIN_SCALE, use the other sheet_layout and lower the
    # topology complexity (retries) to fit; the layout used is doc.meta["sheet_layout"].


LAYER_SETS = (("PIPE", "SYMBOL", "TEXT", "FRAME"), ("M-PIPE", "M-SYMB", "M-TEXT", "M-BORDER"),
              ("0-PIPING", "0-EQUIP", "0-ANNO", "0-TITLE"))


def random_style(rng: random.Random) -> DrawingStyle:
    """Random drawing style so the extractor cannot overfit one CAD look."""
    s = rng.uniform(MIN_SCALE, MAX_SCALE)
    k = (s / 18.0) ** 0.5
    pw = rng.uniform(0.7, 1.6)
    return DrawingStyle(
        scale=s,
        inst_scale=rng.uniform(1.3, 1.6),
        conn_scale=rng.uniform(1.15, 1.4),
        symbol_width=rng.uniform(0.4, 0.9),
        pipe_width=pw,
        graded_widths=rng.random() < 0.5,
        tap_width=max(0.3, pw * rng.uniform(0.4, 0.7)),
        signal_width=rng.uniform(0.3, 0.6),
        signal_dashes=rng.choice(["[3 2] 0", "[4 2] 0", "[2 2] 0", "[6 2 1 2] 0"]),
        font=rng.choice(["helv", "helv", "helv", "tiro", "cour"]),
        tag_size=round(rng.uniform(5.2, 7.2) * k, 2),
        label_size=round(rng.uniform(5.6, 7.8) * k, 2),
        lineno_size=round(rng.uniform(4.8, 6.4) * k, 2),
        junction_dots=rng.random() < 0.6,
        dot_taps=rng.random() < 0.6,
        crossing=rng.choice(["hop", "hop", "gap"]),
        hop_on=rng.choice(["h", "v"]),
        hop_radius=round(rng.uniform(2.5, 4.5) * k, 2),
        color_pipes=rng.random() < 0.3,
        layers=rng.choice(LAYER_SETS) if rng.random() < 0.4 else (),
        legend=rng.random() < 0.35,
        sheet_layout=rng.choice(["right", "right", "bottom"]),
        notes_box=rng.random() < 0.7,
        fanout=rng.choice(["none", "down", "down", "up"]),
        valve_mirror=rng.choice([0.0, 0.0, 0.08]),
        flow_arrows=rng.random() < 0.5,
        spacing=rng.uniform(0.9, 1.25),
        split_pipes=rng.random() < 0.3,
        interleave=rng.random() < 0.3,
    )


# --------------------------------------------------------------------------- symbol helpers

_SCALE_K = {S.PUMP_CENTRIFUGAL: 1.35, S.PUMP_SCREW: 1.35, S.COMPRESSOR: 1.35, S.PURIFIER: 1.4, S.HEATER: 1.3,
            S.HEAT_EXCHANGER: 1.25, S.PLATE_HEAT_EXCHANGER: 1.3, S.FILTER: 1.2, S.DUPLEX_FILTER: 1.25,
            S.FLOW_METER: 1.2, S.OVERBOARD: 1.2, S.STRAINER: 1.1, S.BWTS_UNIT: 1.0}
_BOUNDARY = {S.TANK, S.ENGINE, S.OILY_WATER_SEPARATOR, S.AIR_RECEIVER}
_ACT_TOP = {A.HANDWHEEL: 0.45, A.DIAPHRAGM: 0.69, A.HYDRAULIC: 0.75, A.ELECTRIC: 0.75, A.QUICK_CLOSING: 0.72}
_ORIENTS = ((0, False), (0, True), (90, False), (90, True), (180, False), (180, True), (270, False), (270, True))
_CLR = {"valve": 0.55, "fitting": 0.5, "equipment": 0.8, "tap": 0.75, "branch": 0.6, "connector": 0.7,
        "source": 0.7, "header": 0.9, "sink": 0.8, "end": 0.6, "instrument": 0.7}


def _rdir(d: Pt) -> Pt:
    return (float(round(d[0])), float(round(d[1])))


@lru_cache(maxsize=None)
def _orientations(cls: SymbolClass, constraints: tuple[tuple[str, Pt], ...], mirror_ok: bool) -> tuple[tuple[int, bool], ...]:
    """Orientations (rotation, mirrored) whose ports face the requested directions, best first.

    Valves prefer the actuator above the pipe (page -y) and rotation 0."""
    spec = get_symbol(cls)
    ports = {p.name: p for p in spec.ports}
    out = []
    for rot, mir in _ORIENTS:
        if mir and not mirror_ok:
            continue
        pl = Placement(0.0, 0.0, 1.0, rot, mir)
        ok = all(_rdir(pl.dir_to_page(ports[n].dx, ports[n].dy)) == d for n, d in constraints)
        if not ok:
            continue
        score = 0.0 if rot == 0 else 0.1
        if spec.category == Category.VALVE:
            score += 0.0 if _rdir(pl.dir_to_page(0, -1)) == (0.0, -1.0) else 1.0
        out.append((score, rot, mir))
    out.sort()
    return tuple((r, m) for _, r, m in out)


@lru_cache(maxsize=None)
def _inline_orientations(cls: SymbolClass, pin: str, pout: str) -> tuple[tuple[float, int, bool], ...]:
    """Orientations for an inline item, scored: inlet facing left/down/up, outlet facing right/up/down."""
    spec = get_symbol(cls)
    ports = {p.name: p for p in spec.ports}
    res = []
    for rot, mir in _ORIENTS:
        pl = Placement(0.0, 0.0, 1.0, rot, mir)
        a, b = ports[pin], ports[pout]
        ia, oa = pl.to_page(a.x, a.y), pl.to_page(b.x, b.y)
        di, do = _rdir(pl.dir_to_page(a.dx, a.dy)), _rdir(pl.dir_to_page(b.dx, b.dy))
        si = {(-1.0, 0.0): 0.0, (0.0, 1.0): 3.0, (0.0, -1.0): 3.2}.get(di)
        so = {(1.0, 0.0): 0.0, (0.0, -1.0): 1.0, (0.0, 1.0): 2.0}.get(do)
        if si is None or so is None or oa[0] <= ia[0] + 1e-9:
            continue
        score = si + so + (0.0 if rot == 0 else 0.1) + (0.05 if mir else 0.0)
        if spec.category == Category.VALVE and _rdir(pl.dir_to_page(0, -1)) != (0.0, -1.0):
            score += 1.0
        res.append((score, rot, mir))
    res.sort()
    return tuple(res)


def _default_ports(it: Item) -> tuple[str, str]:
    if it.in_port and it.out_port:
        return it.in_port, it.out_port
    names = [p.name for p in get_symbol(it.cls).ports]
    if "suction" in names:
        return "suction", "discharge"
    return it.in_port or "in", it.out_port or "out"


# --------------------------------------------------------------------------- scene graph


@dataclass
class _Obj:
    key: str
    item: Item
    pl: Placement
    boundary: bool = False
    box: Optional[Rect] = None  # boundary objects: outline box
    ports: dict[str, Pt] = field(default_factory=dict)  # port name -> position
    pdirs: dict[str, Pt] = field(default_factory=dict)
    role: str = "inline"  # source | sink | inline | instrument | deco
    axis: str = "h"  # page axis of the main flow through the object (label preferences)
    stage: int = 0
    nb: int = 0  # boundary ports allocated

    @property
    def spec(self):
        return get_symbol(self.item.cls)

    def bbox(self) -> Rect:
        cached = self.__dict__.get("_bb")
        if cached is not None and cached[0] is self.pl:
            return cached[1]
        act = self.item.actuator if self.spec.category == Category.VALVE else A.NONE
        bb = placed_bbox(self.spec, self.pl, act)
        self.__dict__["_bb"] = (self.pl, bb)
        return bb


@dataclass
class _Edge:
    a: str
    b: str
    kind: PipeKind = PipeKind.PROCESS
    line: Optional[str] = None  # logical line id (one line number per id)
    size: str = "main"  # size class
    medium: Optional[Medium] = None
    code: Optional[str] = None
    flow: Optional[bool] = True  # True: flow from a to b


class _Net:
    """Objects, nodes (free points and ports) and straight axis-aligned edges in one frame."""

    def __init__(self) -> None:
        self.objs: dict[str, _Obj] = {}
        self.pos: dict[str, Pt] = {}
        self.port_of: dict[str, tuple[str, str]] = {}
        self.edges: list[_Edge] = []
        self.forced: set[str] = set()  # nodes that must become junctions (tap tees)
        self.branch: dict[str, str] = {}  # Branch key -> node
        self.reserve: list[Rect] = []  # label reservations (extent only)
        self._n = 0

    def node(self, p: Pt, prefix: str = "n") -> str:
        self._n += 1
        k = f"{prefix}{self._n}"
        self.pos[k] = (float(p[0]), float(p[1]))
        return k

    def add(self, o: _Obj) -> _Obj:
        self.objs[o.key] = o
        if not o.boundary:
            spec = o.spec
            for ps in spec.ports:
                o.ports[ps.name] = o.pl.to_page(ps.x, ps.y)
                o.pdirs[ps.name] = _rdir(o.pl.dir_to_page(ps.dx, ps.dy))
        return o

    def port(self, okey: str, name: str) -> str:
        k = f"{okey}:{name}"
        if k not in self.pos:
            self.pos[k] = self.objs[okey].ports[name]
            self.port_of[k] = (okey, name)
        return k

    def bport(self, okey: str, p: Pt, d: Pt) -> str:
        """New boundary port n<k> on a boundary / instrument object."""
        o = self.objs[okey]
        name = f"n{o.nb}"
        o.nb += 1
        o.ports[name] = (float(p[0]), float(p[1]))
        o.pdirs[name] = d
        return self.port(okey, name)

    def edge(self, a: str, b: str, **kw) -> None:
        pa, pb = self.pos[a], self.pos[b]
        if abs(pa[0] - pb[0]) > 1e-6 and abs(pa[1] - pb[1]) > 1e-6:
            raise LayoutError(f"non-orthogonal edge {a}->{b} {pa} {pb}")
        if abs(pa[0] - pb[0]) < 1e-9 and abs(pa[1] - pb[1]) < 1e-9:
            raise LayoutError(f"zero-length edge {a}->{b}")
        self.edges.append(_Edge(a, b, **kw))

    def path(self, nodes: Sequence[str], **kw) -> None:
        for a, b in zip(nodes, nodes[1:]):
            self.edge(a, b, **kw)

    def extent(self) -> Rect:
        xs0, ys0, xs1, ys1 = [], [], [], []
        for o in self.objs.values():
            b = o.bbox()
            xs0.append(b[0]); ys0.append(b[1]); xs1.append(b[2]); ys1.append(b[3])
        for x, y in self.pos.values():
            xs0.append(x); ys0.append(y); xs1.append(x); ys1.append(y)
        for r in self.reserve:
            xs0.append(r[0]); ys0.append(r[1]); xs1.append(r[2]); ys1.append(r[3])
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    def merge(self, other: "_Net", T, rot: int, prefix: str, alias: dict[str, str]) -> None:
        """Copy ``other`` (a local train frame) into this net through transform ``T`` (point -> point)
        with frame rotation ``rot``; free nodes are renamed with ``prefix`` unless aliased."""
        for o in other.objs.values():
            self.objs[o.key] = _transform_obj(o, T, rot)
        rename: dict[str, str] = {}
        for k, p in other.pos.items():
            if k in other.port_of:
                rename[k] = k
                self.pos[k] = self.objs[other.port_of[k][0]].ports[other.port_of[k][1]]
                self.port_of[k] = other.port_of[k]
            elif k in alias:
                rename[k] = alias[k]
            else:
                nk = f"{prefix}/{k}"
                rename[k] = nk
                self.pos[nk] = T(p)
        for k in other.forced:
            self.forced.add(rename[k])
        for bk, nk in other.branch.items():
            self.branch[bk] = rename[nk]
        for e in other.edges:
            self.edges.append(replace(e, a=rename[e.a], b=rename[e.b]))
        for r in other.reserve:
            a, b = T((r[0], r[1])), T((r[2], r[3]))
            self.reserve.append((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])))


def _rot_vec(d: Pt, rot: int) -> Pt:
    r = rot % 360
    if r == 0:
        return d
    if r == 90:
        return (-d[1], d[0])
    if r == 180:
        return (-d[0], -d[1])
    return (d[1], -d[0])


def _transform_obj(o: _Obj, T, rot: int) -> _Obj:
    spec = o.spec
    if o.boundary or not spec.rotatable:
        if o.box is not None:
            a, b = T((o.box[0], o.box[1])), T((o.box[2], o.box[3]))
            box = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
            pl = _box_placement(o.item.cls, box, o.pl.scale)
        else:
            c = T((o.pl.cx, o.pl.cy))
            box = None
            pl = Placement(c[0], c[1], o.pl.scale, 0, False, o.pl.stretch)
        n = _Obj(o.key, o.item, pl, o.boundary, box, {}, {}, o.role, o.axis, o.stage, o.nb)
        for name, p in o.ports.items():
            n.ports[name] = T(p)
            n.pdirs[name] = _rot_vec(o.pdirs[name], rot)
        return n
    c = T((o.pl.cx, o.pl.cy))
    pl = Placement(c[0], c[1], o.pl.scale, (o.pl.rotation + rot) % 360, o.pl.mirrored, o.pl.stretch)
    n = _Obj(o.key, o.item, pl, False, None, {}, {}, o.role, o.axis, o.stage, o.nb)
    for ps in spec.ports:
        n.ports[ps.name] = pl.to_page(ps.x, ps.y)
        n.pdirs[ps.name] = _rdir(pl.dir_to_page(ps.dx, ps.dy))
    for name, p in o.ports.items():  # extra ports (signal "actuator")
        if name not in n.ports:
            n.ports[name] = T(p)
            n.pdirs[name] = _rot_vec(o.pdirs[name], rot)
    if rot in (90, 270):
        n.axis = "v" if o.axis == "h" else "h"
    return n


def _box_placement(cls: SymbolClass, box: Rect, scale: float) -> Placement:
    u0, v0, u1, v1 = get_symbol(cls).extent
    sx = (box[2] - box[0]) / ((u1 - u0) * scale)
    sy = (box[3] - box[1]) / ((v1 - v0) * scale)
    return Placement((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, scale, 0, False, (sx, sy))


# --------------------------------------------------------------------------- generator


@dataclass
class _Train:
    tr: Train
    net: _Net
    start: str  # start node (open start) or source port
    end: Optional[str]  # open end node (None if the train ends at a sink)
    end_level: float
    sink_u: Optional[float]
    open_start: bool


class _Gen:
    def __init__(self, topo: SystemTopology, style: DrawingStyle, rng: random.Random, seed: int):
        self.topo = topo
        self.st = style
        self.rng = rng
        self.seed = seed
        self.font = style.font
        self._gap_cache: dict[str, list[float]] = {}
        self._tpl_cache: dict[str, dict[str, float]] = {}
        self._mirror: dict[str, bool] = {}
        self.stage_index = {id(stg): k for k, stg in enumerate(topo.stages)}
        self.drain_stages = {self.stage_index[id(d.stage)] for d in topo.drains if id(d.stage) in self.stage_index}
        sizes = topo.sizes or {"main": 100}
        sample = topo.namer.line_style.format(size=max(sizes.values()), c=topo.code, n=999, spec=topo.namer.spec)
        self.slot = text_width(sample, self.font, style.lineno_size) + 10.0
        self.ret_targets = {r.target for r in topo.returns}
        self.side_ports: dict[str, list[str]] = {}
        for sl in topo.side_links:
            self.side_ports.setdefault(sl.item, []).append(sl.port)
        self.set_scale(style.scale)
        self.lh_label = line_height(self.font, style.label_size)
        self.fanout_stage = self._fanout_stage()

    # -- random but template-stable spacing ------------------------------------
    def _extra(self, tpl: str, i: int) -> float:
        lst = self._gap_cache.setdefault(tpl, [])
        while len(lst) <= i:
            lst.append(self.rng.uniform(0.0, 0.6))
        return lst[i]

    def _tpl(self, tpl: str, name: str, lo: float, hi: float) -> float:
        d = self._tpl_cache.setdefault(tpl, {})
        if name not in d:
            d[name] = self.rng.uniform(lo, hi)
        return d[name]

    def _mirrored(self, key: str) -> bool:
        if key not in self._mirror:
            self._mirror[key] = self.rng.random() < self.st.valve_mirror
        return self._mirror[key]

    def set_scale(self, s: float) -> None:
        """Symbol scale (points per symbol unit); instrument / connector scales follow."""
        self.s = s
        self.si = s * self.st.inst_scale
        self.sc = s * self.st.conn_scale
        self.inst_fs = self._instrument_font()

    def _instrument_font(self) -> float:
        """Font size for bubble texts; grows the bubble scale if the longest text does not fit."""
        size = max(4.0, min(6.5, self.si * 0.3))
        r = 0.4 * self.si
        need = r
        for it in self.topo.items():
            if it.category != Category.INSTRUMENT:
                continue
            for t in it.inside:
                w = text_width(t, self.font, size)
                h = line_height(self.font, size)
                need = max(need, math.hypot(w / 2 / 0.92, 0.6 + h))
        if need > r:
            self.si = need / 0.4
        return size

    def _fanout_stage(self) -> Optional[int]:
        if self.st.fanout == "none" or len(self.topo.stages) < 2:
            return None
        k = len(self.topo.stages) - 1
        st = self.topo.stages[k]
        if not st.fanout or len(st.trains) < 2:
            return None
        if any(t.sink is None or t.source is not None or t.end_return for t in st.trains):
            return None
        if len({t.group for t in st.trains}) != 1:
            return None
        return k

    # -- item geometry -----------------------------------------------------------
    def _scale_of(self, it: Item) -> float:
        if it.category == Category.INSTRUMENT:
            return self.si
        if it.cls == S.OFF_PAGE_CONNECTOR:
            return self.sc
        return self.s * _SCALE_K.get(it.cls, 1.0)

    def _box_size(self, it: Item, frot: int = 0) -> tuple[float, float]:
        """Outline size in the train's local frame (u along the flow, v across) for boundary equipment.

        ``it.size`` is the page size (width, height) in symbol units; in a rotated (fan-out)
        frame the local axes are swapped so the page shape stays the same."""
        s = self.s
        if it.size:
            w, h = it.size
        else:
            u0, v0, u1, v1 = get_symbol(it.cls).extent
            w, h = u1 - u0, v1 - v0
        w, h = w * s, h * s
        if it.name_inside and it.cls == S.TANK and it.name:
            pad = 4.0
            tw = max(text_width(n, self.font, self.st.label_size) for n in it.name)
            w = max(w, tw + 2 * pad)
            h = max(h, len(it.name) * self.lh_label * 1.15 + 2 * pad)
        if it.key in self.ret_targets:
            h = max(h, 2.8 * s)
        return (h, w) if frot in (90, 270) else (w, h)

    def _stretch(self, it: Item, sc: float) -> tuple[float, float]:
        if not it.size or not get_symbol(it.cls).stretchable:
            return (1.0, 1.0)
        u0, v0, u1, v1 = get_symbol(it.cls).extent
        return (it.size[0] / (u1 - u0) * self.s / sc, it.size[1] / (v1 - v0) * self.s / sc)

    def _kind(self, el) -> str:
        if isinstance(el, Tap):
            return "tap"
        if isinstance(el, Branch):
            return "branch"
        cat = el.category
        return {Category.EQUIPMENT: "equipment", Category.VALVE: "valve", Category.FITTING: "fitting",
                Category.CONNECTOR: "connector", Category.INSTRUMENT: "instrument"}[cat]

    # -- train layout (local frame: flow +u, v down) -------------------------------
    def build_train(self, tr: Train, fu: float, sink_u: Optional[float] = None, frot: int = 0) -> _Train:
        net = _Net()
        s = self.s
        tpl = tr.template
        gi = [0]

        def gap(a: str, b: str, slot: bool = False) -> float:
            g = (_CLR[a] + _CLR[b] + self._extra(tpl, gi[0])) * s * fu * self.st.spacing
            gi[0] += 1
            return g + (self.slot if slot else 0.0)

        line = [0]

        def lid() -> str:
            return f"{tr.key}#{line[0]}"

        size = tr.size_class
        open_start = tr.source is None
        if open_start:
            cur = net.node((0.0, 0.0), "s")
            u, lvl = 0.0, 0.0
            prev = "header"
        else:
            cur, u, lvl = self._place_source(net, tr, tr.source)
            prev = "source"
        start = cur
        elems = list(tr.elements)
        # one long straight run per line segment for its line number: segment 0 right after the
        # start, later segments (after each equipment) right before their end
        slotted = False
        for i, el in enumerate(elems):
            kind = self._kind(el)
            want = (i == 0) or (not slotted and isinstance(el, Item) and el.category == Category.EQUIPMENT)
            g = gap(prev, kind, slot=want)
            slotted = slotted or want
            if isinstance(el, Item):
                cur, u, lvl = self._place_inline(net, tr, el, cur, u, lvl, g, lid(), size)
                if el.category == Category.EQUIPMENT:
                    line[0] += 1
                    slotted = False
            elif isinstance(el, Tap):
                j = net.node((u + g, lvl), "j")
                net.edge(cur, j, line=lid(), size=size)
                net.forced.add(j)
                self._place_tap(net, tr, el, j)
                cur, u = j, u + g
            else:
                j = net.node((u + g, lvl), "b")
                net.edge(cur, j, line=lid(), size=size)
                net.branch[el.key] = j
                cur, u = j, u + g
            prev = kind
        end: Optional[str] = None
        if tr.sink is not None:
            g = gap(prev, "sink", slot=not slotted)
            us = max(u + g, sink_u if sink_u is not None else -1e18)
            self._place_sink(net, tr, tr.sink, cur, us, lvl, lid(), size, frot)
            sink_at = us
        else:
            g = gap(prev, "end", slot=not slotted)
            end = net.node((u + g, lvl), "e")
            net.edge(cur, end, line=lid(), size=size)
            if tr.end_return:
                net.reserve.append((u + g, lvl - s, u + g + 2.2 * s, lvl + s))
            sink_at = None
        for sl in self.topo.side_links:
            if sl.item in net.objs:
                self._place_side(net, tr, sl)
        for rel in self.topo.reliefs:
            if rel.inlet in net.branch and rel.outlet in net.branch:
                self._place_relief(net, tr, rel)
        for bp in self.topo.bypasses:
            if bp.start in net.branch and (bp.end in net.branch or bp.end in net.objs):
                self._place_bypass(net, tr, bp)
        for at in self.topo.attachments:
            if at.item in net.objs:
                self._place_attachment(net, tr, at)
        # label reservation bands above / below the train (beside it in a rotated frame,
        # where tags of vertical valves sit next to the pipe)
        x0, y0, x1, y1 = net.extent()
        m = 2.0 * self.lh_label + 0.25 * s
        if frot:
            tags = [text_width(o.item.tag, self.font, self.st.tag_size) for o in net.objs.values() if o.item.tag]
            m = max(m, max(tags, default=0.0) + 6.0)
        net.reserve.append((x0, y0 - m, x1, y1 + m))
        lvl_end = net.pos[end][1] if end else lvl
        return _Train(tr, net, start, end, lvl_end, sink_at, open_start)

    def _place_source(self, net: _Net, tr: Train, it: Item) -> tuple[str, float, float]:
        s = self.s
        if it.cls in _BOUNDARY:
            w, h = self._box_size(it)
            frac = 0.5 if it.key in self.ret_targets else self._tpl(tr.template, "port_frac", 0.3, 0.7)
            a = h * frac
            o = net.add(_Obj(it.key, it, _box_placement(it.cls, (-w, -a, 0, h - a), s), True, (-w, -a, 0.0, h - a),
                             role="source"))
            p = net.bport(o.key, (0.0, 0.0), (1.0, 0.0))
            return p, 0.0, 0.0
        if it.cls == S.SEA_CHEST:
            sc = self._scale_of(it)
            st = self._stretch(it, sc)
            u0, v0, u1, v1 = get_symbol(it.cls).extent
            w, h = (u1 - u0) * st[0] * sc, (v1 - v0) * st[1] * sc
            rise = self._tpl(tr.template, "rise", 1.0, 1.6) * s
            pl = Placement(-w / 2, rise + h / 2, sc, 0, False, st)
            o = net.add(_Obj(it.key, it, pl, role="source"))
            pout = net.port(o.key, "out")
            c = net.node((-w / 2, 0.0))
            net.edge(pout, c, line=f"{tr.key}#0", size=tr.size_class)
            return c, -w / 2, 0.0
        sc = self._scale_of(it)
        if it.cls == S.OFF_PAGE_CONNECTOR:
            rot, mir = _orientations(it.cls, (("in", (1.0, 0.0)),), False)[0]
            pname = "in"
        else:
            pin, pout = _default_ports(it)
            rot, mir = _orientations(it.cls, ((pout, (1.0, 0.0)),), False)[0]
            pname = pout
        pl0 = Placement(0, 0, sc, rot, mir)
        ps = {p.name: p for p in get_symbol(it.cls).ports}[pname]
        off = pl0.to_page(ps.x, ps.y)
        o = net.add(_Obj(it.key, it, Placement(-off[0], -off[1], sc, rot, mir), role="source"))
        if it.cls == S.OFF_PAGE_CONNECTOR:
            lw = max(text_width(n, self.font, self.st.label_size) for n in it.name) if it.name else 0.0
            b = o.bbox()
            net.reserve.append((b[0] - lw - 6, b[1], b[0], b[3]))
        return net.port(o.key, pname), 0.0, 0.0

    def _place_sink(self, net: _Net, tr: Train, it: Item, cur: str, us: float, lvl: float, lid: str,
                    size: str, frot: int = 0) -> None:
        s = self.s
        if it.cls in _BOUNDARY:
            w, h = self._box_size(it, frot)
            a = h * self._tpl(tr.template, "sink_frac", 0.3, 0.7)
            box = (us, lvl - a, us + w, lvl - a + h)
            o = net.add(_Obj(it.key, it, _box_placement(it.cls, box, s), True, box, role="sink"))
            p = net.bport(o.key, (us, lvl), (-1.0, 0.0))
        else:
            sc = self._scale_of(it)
            pin, _ = _default_ports(it)
            rot, mir = _orientations(it.cls, ((pin, (-1.0, 0.0)),), False)[0]
            pl0 = Placement(0, 0, sc, rot, mir)
            ps = {p.name: p for p in get_symbol(it.cls).ports}[pin]
            off = pl0.to_page(ps.x, ps.y)
            o = net.add(_Obj(it.key, it, Placement(us - off[0], lvl - off[1], sc, rot, mir), role="sink"))
            p = net.port(o.key, pin)
            if it.cls == S.OFF_PAGE_CONNECTOR and it.name:
                lw = max(text_width(n, self.font, self.st.label_size) for n in it.name) + 6
                b = o.bbox()
                if frot == 0:
                    net.reserve.append((b[2], b[1], b[2] + lw, b[3]))
                else:  # label beside the top of a vertical connector (page right)
                    lh2 = 2 * self.lh_label
                    if frot == 90:
                        net.reserve.append((b[0], b[1] - lw, b[0] + lh2, b[1]))
                    else:
                        net.reserve.append((b[2] - lh2, b[3], b[2], b[3] + lw))
        net.edge(cur, p, line=lid, size=size)

    def _place_inline(self, net: _Net, tr: Train, it: Item, cur: str, u: float, lvl: float, g: float, lid: str,
                      size: str) -> tuple[str, float, float]:
        s = self.s
        if it.cls in _BOUNDARY:
            w, h = self._box_size(it)
            if it.cls == S.AIR_RECEIVER:
                frac = self._tpl(tr.template, "rx_frac", 0.4, 0.6)
            else:
                frac = self._tpl(tr.template, "in_frac", 0.35, 0.65)
            a = h * frac
            box = (u + g, lvl - a, u + g + w, lvl - a + h)
            o = net.add(_Obj(it.key, it, _box_placement(it.cls, box, s), True, box))
            pi = net.bport(o.key, (u + g, lvl), (-1.0, 0.0))
            po = net.bport(o.key, (u + g + w, lvl), (1.0, 0.0))
            net.edge(cur, pi, line=lid, size=size)
            return po, u + g + w, lvl
        sc = self._scale_of(it)
        pin, pout = _default_ports(it)
        opts = list(_inline_orientations(it.cls, pin, pout))
        if not opts:
            raise LayoutError(f"no inline orientation for {it.cls}")
        if it.cls == S.THREE_WAY_VALVE:
            want = (0.0, float(it.attrs.get("branch_side", 1)))
            opts = [o for o in opts if _rdir(Placement(0, 0, 1, o[1], o[2]).dir_to_page(0, 1)) == want] or opts
        side_ports = self.side_ports.get(it.key)
        if side_ports:  # secondary connections (cooler shell / plate side) preferably upwards
            pmap = {p.name: p for p in get_symbol(it.cls).ports}

            def faces(o, d):
                pl_ = Placement(0, 0, 1, o[1], o[2])
                return all(_rdir(pl_.dir_to_page(pmap[n].dx, pmap[n].dy)) == d for n in side_ports)

            opts = [o for o in opts if faces(o, (0.0, -1.0))] or [o for o in opts if faces(o, (0.0, 1.0))] or opts
        _, rot, mir = opts[0]
        if it.category == Category.VALVE and it.cls != S.THREE_WAY_VALVE and self._mirrored(it.key):
            alt = [o for o in opts if o[2] != mir and o[0] - opts[0][0] < 1.2]
            if alt:
                _, rot, mir = alt[0]
        st = self._stretch(it, sc)
        pl0 = Placement(0, 0, sc, rot, mir, st)
        ports = {p.name: p for p in get_symbol(it.cls).ports}
        a, b = ports[pin], ports[pout]
        ia = pl0.to_page(a.x, a.y)
        di = _rdir(pl0.dir_to_page(a.dx, a.dy))
        rise = self._tpl(tr.template, "rise", 0.9, 1.3) * s
        corner = None
        if di == (-1.0, 0.0):
            target = (u + g, lvl)
        elif di == (0.0, 1.0):
            target, corner = (u + g, lvl - rise), (u + g, lvl)
        else:
            target, corner = (u + g, lvl + rise), (u + g, lvl)
        pl = Placement(target[0] - ia[0], target[1] - ia[1], sc, rot, mir, st)
        o = net.add(_Obj(it.key, it, pl))
        p_in = net.port(o.key, pin)
        if corner:
            c = net.node(corner)
            net.path([cur, c, p_in], line=lid, size=size)
        else:
            net.edge(cur, p_in, line=lid, size=size)
        p_out = net.port(o.key, pout)
        ox, oy = o.ports[pout]
        do = o.pdirs[pout]
        if do == (1.0, 0.0):
            return p_out, ox, oy
        if do == (0.0, -1.0):
            lift = self._tpl(tr.template, "lift", 0.9, 1.6) * s
            ny = oy - lift
        else:
            ny = max(oy + rise * 0.8, lvl) if corner else oy + rise
        c = net.node((ox, ny))
        net.edge(p_out, c, line=lid, size=size)
        return c, ox, ny

    def _place_tap(self, net: _Net, tr: Train, tap: Tap, j: str) -> None:
        s, r = self.s, 0.4 * self.si
        ju, jv = net.pos[j]
        side = tap.side
        L = self._tpl(tr.template, "tap", 0.8, 1.3) * s
        valve = net.objs.get(tap.signal_to) if tap.signal_to else None
        act_pt = None
        if valve is not None:
            h_act = _ACT_TOP.get(valve.item.actuator, 0.5)
            act_pt = valve.pl.to_page(0.0, -h_act)
            side = -1 if act_pt[1] < valve.pl.cy else 1
            need = abs(jv - act_pt[1]) + 0.8 * s
            L = max(L, need)
        c = (ju, jv + side * (L + r))
        it = tap.instr
        o = net.add(_Obj(it.key, it, Placement(c[0], c[1], self.si), role="instrument"))
        p0 = net.bport(o.key, (ju, jv + side * L), (0.0, float(-side)))
        net.edge(j, p0, kind=PipeKind.INSTRUMENT_TAP, line=None, size="tap", flow=None)
        if valve is not None and act_pt is not None:
            dx = -1.0 if act_pt[0] < c[0] else 1.0
            p1 = net.bport(o.key, (c[0] + dx * r, c[1]), (dx, 0.0))
            valve.ports["actuator"] = act_pt
            valve.pdirs["actuator"] = (0.0, -1.0 if act_pt[1] < valve.pl.cy else 1.0)
            pa = net.port(valve.key, "actuator")
            corner = net.node((act_pt[0], c[1]))
            net.path([p1, corner, pa], kind=PipeKind.SIGNAL, line=None, size="signal")

    def _objs_between(self, net: _Net, u0: float, u1: float) -> list[Rect]:
        out = []
        for o in net.objs.values():
            b = o.bbox()
            if b[2] > u0 + 1e-6 and b[0] < u1 - 1e-6:
                out.append(b)
        return out

    def _place_relief(self, net: _Net, tr: Train, rel) -> None:
        s = self.s
        jo, ji = net.branch[rel.inlet], net.branch[rel.outlet]
        (uo, vo), (ui, vi) = net.pos[jo], net.pos[ji]
        boxes = self._objs_between(net, ui, uo)
        top = min([b[1] for b in boxes] + [vo, vi])
        it = rel.valve
        sc = self._scale_of(it)
        rot, mir = _orientations(it.cls, (("in", (0.0, 1.0)), ("out", (-1.0, 0.0))), True)[0]
        pl0 = Placement(0, 0, sc, rot, mir)
        ports = {p.name: p for p in get_symbol(it.cls).ports}
        ia = pl0.to_page(ports["in"].x, ports["in"].y)
        vin = top - 0.9 * s
        pl = Placement(uo - ia[0], vin - ia[1], sc, rot, mir)
        o = net.add(_Obj(it.key, it, pl, role="deco"))
        pin, pout = net.port(o.key, "in"), net.port(o.key, "out")
        ox, oy = o.ports["out"]
        if ox - ui < 0.6 * s:
            raise LayoutError("relief loop too narrow")
        lid = f"{tr.key}/rv"
        net.edge(jo, pin, line=lid, size="small")
        c = net.node((ui, oy))
        net.path([pout, c, ji], line=lid, size="small")

    def _place_bypass(self, net: _Net, tr: Train, bp) -> None:
        s = self.s
        js = net.branch[bp.start]
        us, vs = net.pos[js]
        if bp.end_port:
            ue, ve = net.objs[bp.end].ports[bp.end_port]
            je = net.port(bp.end, bp.end_port)
        else:
            je = net.branch[bp.end]
            ue, ve = net.pos[je]
        boxes = self._objs_between(net, us + 1e-3, ue - 1e-3)
        it = bp.valve
        sc = self._scale_of(it)
        side = bp.side
        rot, mir = _orientations(it.cls, (("in", (-1.0, 0.0)), ("out", (1.0, 0.0))), False)[0]
        pl0 = Placement(0, 0, sc, rot, mir)
        vb = placed_bbox(get_symbol(it.cls), pl0, it.actuator)
        if side > 0:
            bottom = max([b[3] for b in boxes] + [vs, ve])
            v = bottom + 1.0 * s + self.lh_label - vb[1]
        else:
            top = min([b[1] for b in boxes] + [vs, ve])
            v = top - 1.0 * s - self.lh_label - vb[3]
        uc = (us + ue) / 2
        if ue - us < (vb[2] - vb[0]) + 1.2 * s:
            raise LayoutError("bypass too short")
        o = net.add(_Obj(it.key, it, Placement(uc, v, sc, rot, mir), role="deco"))
        pin, pout = net.port(o.key, "in"), net.port(o.key, "out")
        lid = f"{tr.key}/bp{bp.start}"
        c0, c1 = net.node((us, v)), net.node((ue, v))
        net.path([js, c0, pin], line=lid, size=tr.size_class)
        net.path([pout, c1, je], line=lid, size=tr.size_class)

    def _place_side(self, net: _Net, tr: Train, sl) -> None:
        s = self.s
        unit = net.objs[sl.item]
        p, d = unit.ports[sl.port], unit.pdirs[sl.port]
        it = sl.connector
        sc = self._scale_of(it)
        rise = self._tpl(tr.template, "side_rise", 0.8, 1.2) * s
        want = (-d[0], -d[1])
        rot, mir = _orientations(it.cls, (("in", want),), False)[0]
        pl0 = Placement(0, 0, sc, rot, mir)
        ps = get_symbol(it.cls).ports[0]
        off = pl0.to_page(ps.x, ps.y)
        tgt = (p[0] + d[0] * rise, p[1] + d[1] * rise)
        o = net.add(_Obj(it.key, it, Placement(tgt[0] - off[0], tgt[1] - off[1], sc, rot, mir), role="deco"))
        pc = net.port(o.key, "in")
        pu = net.port(unit.key, sl.port)
        lid = f"{tr.key}/sl{sl.port}"
        kw = dict(line=lid, size=sl.size_class, medium=sl.medium, code=sl.code)
        if sl.outgoing:
            net.edge(pu, pc, **kw)
        else:
            net.edge(pc, pu, **kw)
        if it.name:
            lw = max(text_width(n, self.font, self.st.label_size) for n in it.name)
            b = o.bbox()
            net.reserve.append((b[2], b[1], b[2] + lw + 6, b[1] + self.lh_label * 2))

    def _place_attachment(self, net: _Net, tr: Train, at) -> None:
        s = self.s
        o = net.objs[at.item]
        if not o.boundary or o.box is None:
            return
        x0, y0, x1, y1 = o.box
        if at.side == "top":
            side = "top"
        elif o.role == "source":
            side = "left"
        elif o.role == "sink":
            side = "right"
        else:
            side = "bottom"
        if side == "left":
            p, n = (x0, (y0 + y1) / 2), (-1.0, 0.0)
        elif side == "right":
            p, n = (x1, (y0 + y1) / 2), (1.0, 0.0)
        elif side == "top":
            p, n = ((x0 + x1) / 2, y0), (0.0, -1.0)
        else:
            p, n = ((x0 + x1) / 2, y1), (0.0, 1.0)
        for q in o.ports.values():  # keep away from the process connections on the same edge
            if math.hypot(q[0] - p[0], q[1] - p[1]) < 0.6 * s:
                if n[0]:
                    p = (p[0], y0 + (y1 - y0) * (0.25 if q[1] > p[1] else 0.75))
                else:
                    p = (x0 + (x1 - x0) * (0.25 if q[0] > p[0] else 0.75), p[1])
        if o.item.cls == S.AIR_RECEIVER and n[0] != 0:  # capsule: only the straight sides are vertical
            p = (p[0], (y0 + y1) / 2)
        it = at.obj
        po = net.bport(o.key, p, n)
        if at.kind == "tap":
            r = 0.4 * self.si
            L = 0.9 * s
            c = (p[0] + n[0] * (L + r), p[1] + n[1] * (L + r))
            oi = net.add(_Obj(it.key, it, Placement(c[0], c[1], self.si), role="instrument"))
            pi = net.bport(oi.key, (p[0] + n[0] * L, p[1] + n[1] * L), (-n[0], -n[1]))
            net.edge(po, pi, kind=PipeKind.INSTRUMENT_TAP, line=None, size="tap", flow=None)
            return
        sc = self._scale_of(it)
        rot, mir = _orientations(it.cls, (("in", (-n[0], -n[1])),), True)[0]
        pl0 = Placement(0, 0, sc, rot, mir)
        ps = {q.name: q for q in get_symbol(it.cls).ports}["in"]
        off = pl0.to_page(ps.x, ps.y)
        tgt = (p[0] + n[0] * 0.9 * s, p[1] + n[1] * 0.9 * s)
        ov = net.add(_Obj(it.key, it, Placement(tgt[0] - off[0], tgt[1] - off[1], sc, rot, mir), role="deco"))
        net.edge(po, net.port(ov.key, "in"), line=f"{tr.key}/att{it.key}", size="small")

    # -- stage layout --------------------------------------------------------------
    def layout(self, fu: float, fv: float, split: Optional[int] = None) -> "_Scene":
        """Place all trains. ``split``: first stage of the second band (two-band layout) or None."""
        s = self.s
        stages = self.topo.stages
        fo = self.fanout_stage
        frot_fo = 270 if (self.st.fanout == "up" and split is None) else 90
        built: list[list[_Train]] = []
        for k, stg in enumerate(stages):
            frot = frot_fo if k == fo else 0
            lts = [self.build_train(tr, fu, frot=frot) for tr in stg.trains]
            sinks = [i for i, t in enumerate(lts) if t.sink_u is not None]
            if (len(sinks) > 1 and k != fo) or (k == fo and sinks):
                us = max(lts[i].sink_u for i in sinks)
                for i in sinks:
                    if lts[i].sink_u < us - 1e-6:
                        lts[i] = self.build_train(stg.trains[i], fu, sink_u=us, frot=frot)
            built.append(lts)
        scene = _Scene()

        def row_gap(k: int, i: int) -> float:
            return (0.45 + self._extra(f"rows{k}", i)) * s * fv * self.st.spacing

        # vertical stacking per stage (offsets of each train's start level), centred on y = 0
        ys: list[list[float]] = []
        for k, lts in enumerate(built):
            if k == fo:
                ys.append([0.0] * len(lts))
                continue
            yy: list[float] = []
            for i, t in enumerate(lts):
                e = t.net.extent()
                if i == 0:
                    yy.append(-e[1])
                else:
                    pe = lts[i - 1].net.extent()
                    yy.append(yy[-1] + pe[3] + row_gap(k, i) - e[1])
            top = yy[0] + lts[0].net.extent()[1]
            bot = yy[-1] + lts[-1].net.extent()[3]
            ys.append([y - (top + bot) / 2 for y in yy])
        # align single straight connections / avoid header conflicts
        for k in range(1, len(built)):
            if k == fo or k == split:
                continue
            ys[k] = self._resolve_shift(built[k - 1], ys[k - 1], built[k], ys[k])
        y_t = 0.0
        if split is not None:  # second band below the first one
            rows = lambda ks: [(k, i) for k in ks if k != fo for i in range(len(built[k]))]  # noqa: E731
            a_bot = max(ys[k][i] + built[k][i].net.extent()[3] for k, i in rows(range(split)))
            b_top = min(ys[k][i] + built[k][i].net.extent()[1] for k, i in rows(range(split, len(built))))
            gap = (2.4 + self._extra("band", 0)) * s * fv + 2 * self.lh_label
            if any(k < split for k in self.drain_stages):
                gap += self._drain_depth()
            dy = a_bot + gap - b_top
            for k in range(split, len(built)):
                ys[k] = [y + dy for y in ys[k]]
            y_t = a_bot + gap / 2
        # x positions
        x = 0.0
        prev_hdr: dict[int, float] = {}
        for k, lts in enumerate(built):
            if k == fo:
                self._place_fanout(scene, k, lts, prev_hdr, fu, frot_fo)
                break
            if split is not None and k == split:
                xb = scene.net.extent()[0] + (0.6 + self._extra("band", 1)) * s * fu
                scene.breaks[k - 1] = (xb, y_t)
                x = xb
            X = x
            u1 = max(t.net.extent()[2] for t in lts)
            for i, t in enumerate(lts):
                ox, oy = X, ys[k][i]
                T = (lambda ox, oy: (lambda p: (p[0] + ox, p[1] + oy)))(ox, oy)
                alias = {}
                if t.open_start and k > 0:
                    alias[t.start] = f"H{k - 1}g{t.tr.group}:{t.tr.key}"
                scene.net.merge(t.net, T, 0, t.tr.key, alias)
                for okey in t.net.objs:
                    scene.net.objs[okey].stage = k
                if t.open_start and k > 0:
                    scene.net.pos[alias[t.start]] = (ox, oy)
                    scene.starts.append((k, t.tr.group, t.tr.key, alias[t.start], (ox, oy)))
                if t.end is not None:
                    scene.ends.append((k, t.tr, f"{t.tr.key}/{t.end}", (ox + t.net.pos[t.end][0], oy + t.end_level)))
            scene.stage_x.append((X + min(t.net.extent()[0] for t in lts), X + u1))
            if k + 1 < len(built):
                xh = X + u1 + (0.9 + self._extra("hdr", k)) * s * fu * self.st.spacing
                hdr = self._headers(scene, k, xh, fu)
                prev_hdr = hdr
                x = max(hdr.values()) if hdr else xh
        scene.connect_headers(self, fu)
        for k in sorted(self.drain_stages):  # room for drain lines below their stage
            if k == fo or k >= len(scene.stage_x):
                continue
            objs = [o.bbox() for o in scene.net.objs.values() if o.stage == k]
            if objs:
                bot = max(b[3] for b in objs)
                scene.net.reserve.append((scene.stage_x[k][0], bot, scene.stage_x[k][1], bot + self._drain_depth()))
        if self.topo.returns:  # room for the return-line lanes above and below
            ex = scene.extent()
            lane = 2.0 * s + 2 * self.lh_label
            scene.net.reserve.append((ex[0], ex[1] - lane, ex[2], ex[3] + lane))
        return scene

    def _drain_depth(self) -> float:
        return 2.6 * self.s + 1.7 * self.sc + 2 * self.lh_label

    def band_split(self, stage_x: list[tuple[float, float]]) -> Optional[int]:
        """Best stage index to start a second band (balanced widths), or None if not possible."""
        stages = self.topo.stages
        n = len(stages)
        if n < 3 or self.topo.returns or len(stage_x) != n:
            return None
        best, best_w = None, None
        for m in range(1, n):
            if m == self.fanout_stage or any(t.source is not None for t in stages[m].trains):
                continue
            groups = {t.group for t in stages[m - 1].trains if t.sink is None and not t.end_return}
            groups |= {t.group for t in stages[m].trains}
            if len(groups) != 1:
                continue
            w = max(stage_x[m - 1][1] - stage_x[0][0], stage_x[-1][1] - stage_x[m][0])
            if best_w is None or w < best_w:
                best, best_w = m, w
        return best

    def _attach_ys(self, prev: list[_Train], prev_y: list[float], nxt: list[_Train], nxt_y: list[float]):
        groups: dict[int, tuple[list[float], list[float]]] = {}
        for t, y in zip(prev, prev_y):
            if t.end is not None and not t.tr.end_return:
                groups.setdefault(t.tr.group, ([], []))[0].append(y + t.end_level)
        for t, y in zip(nxt, nxt_y):
            if t.open_start:
                groups.setdefault(t.tr.group, ([], []))[1].append(y)
        return groups

    def _conflicts(self, groups) -> float:
        """Penalty for header layouts with 4-way junctions or near-coincident tees (0 = fine)."""
        s = self.s
        bad = 0.0
        spans = []
        for g, (L, R) in sorted(groups.items()):
            allv = sorted(L + R)
            if not allv:
                continue
            lo, hi = allv[0], allv[-1]
            spans.append((lo, hi, g, L, R))
            for yl in L:
                for yr in R:
                    if abs(yl - yr) < 1e-6 and lo + 1e-6 < yl < hi - 1e-6:
                        bad += 10
            for a, b in zip(allv, allv[1:]):
                if 1e-6 < b - a < 0.9 * s:
                    bad += 5
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                a, b = spans[i], spans[j]
                if a[0] - 0.9 * s < b[1] and b[0] - 0.9 * s < a[1]:
                    bad += 1  # overlapping groups need separate header columns
                    for y in a[3] + a[4]:
                        for yb in b[3] + b[4]:
                            if abs(y - yb) < 0.9 * s:
                                bad += 5
        return bad

    def _resolve_shift(self, prev: list[_Train], prev_y: list[float], nxt: list[_Train], nxt_y: list[float]) -> list[float]:
        s = self.s
        groups = self._attach_ys(prev, prev_y, nxt, [y for y in nxt_y])
        cands = [0.0]
        single = [(L, R) for L, R in groups.values() if len(L) == 1 and len(R) == 1]
        if len(groups) == 1 and single:
            L, R = single[0]
            cands.insert(0, L[0] - R[0])
        step = 0.25 * s
        for i in range(1, 25):
            cands += [i * step, -i * step]
        best, best_bad = nxt_y, None
        for d in cands:
            yy = [y + d for y in nxt_y]
            bad = self._conflicts(self._attach_ys(prev, prev_y, nxt, yy)) + abs(d) / (100 * s)
            if best_bad is None or bad < best_bad:
                best, best_bad = yy, bad
            if bad < 1.0:
                return yy
        return best

    def _headers(self, scene: "_Scene", k: int, xh: float, fu: float) -> dict[int, float]:
        """Header x per group between stage k and k+1. Each group gets its own column, so headers of
        groups whose vertical spans overlap never share a line (their pipes cross instead)."""
        ends = [e for e in scene.ends if e[0] == k and not e[1].end_return]
        groups = sorted({e[1].group for e in ends} | {t.group for t in self.topo.stages[k + 1].trains
                                                       if t.source is None})
        hdr = {g: xh + i * 0.8 * self.s * fu for i, g in enumerate(groups)}
        scene.headers[k] = hdr
        return hdr

    def _place_fanout(self, scene: "_Scene", k: int, lts: list[_Train], prev_hdr: dict[int, float], fu: float,
                      rot: int) -> None:
        s = self.s
        down = rot == 90
        ends = [e for e in scene.ends if e[0] == k - 1 and not e[1].end_return]
        g = lts[0].tr.group
        ys_l = sorted(e[3][1] for e in ends if e[1].group == g)
        if not ys_l:
            raise LayoutError("fan-out stage without feed")
        y_main = ys_l[0] if down else ys_l[-1]
        xh = prev_hdr.get(g, scene.stage_x[-1][1] + s)
        x = xh + (1.2 + self._extra("fan", 0)) * s * fu
        gap_b = (0.5 + self._extra("fan", 1)) * s * fu * self.st.spacing
        main_nodes: list[tuple[str, float]] = []
        prev_edge = x
        for i, t in enumerate(lts):
            e = t.net.extent()  # local: the v range becomes the lateral page extent
            lat0, lat1 = (-e[3], -e[1]) if down else (e[1], e[3])
            if i == 0:
                xi = x - lat0
            else:
                xi = max(prev_edge + gap_b - lat0, main_nodes[-1][1] + 1.0 * s)
            prev_edge = xi + lat1
            if down:
                T = (lambda xi: (lambda p: (xi - p[1], y_main + p[0])))(xi)
            else:
                T = (lambda xi: (lambda p: (xi + p[1], y_main - p[0])))(xi)
            node = f"F{k}:{i}"
            scene.net.pos[node] = (xi, y_main)
            scene.net.merge(t.net, T, rot, t.tr.key, {t.start: node})
            for okey in t.net.objs:
                scene.net.objs[okey].stage = k
            main_nodes.append((node, xi))
        hn = f"H{k - 1}g{g}:fan"
        scene.net.pos[hn] = (xh, y_main)
        scene.starts.append((k, g, "fan", hn, (xh, y_main)))
        scene.net.path([hn] + [n for n, _ in main_nodes], line=f"fan{k}", size="main")
        scene.stage_x.append((xh, prev_edge))

    # -- page placement --------------------------------------------------------------
    def compose(self) -> tuple["_Scene", Sheet]:
        """Choose layout variant (one or two bands), symbol scale and sheet; stretch the spacing
        to fill the drawing area and centre the drawing in it."""
        st = self.st
        sheets: dict[tuple[str, str], Sheet] = {}

        def sheet_for(paper: str, layout: str) -> Sheet:
            if (paper, layout) not in sheets:
                srng = random.Random(f"{self.seed}:sheet:{paper}")
                sst = SheetStyle(font=self.font, frame_width=srng.uniform(0.9, 1.6),
                                 line_width=srng.uniform(0.35, 0.6), layout=layout, legend=st.legend,
                                 notes_box=st.notes_box, layer=st.layers[3] if st.layers else None)
                sheets[(paper, layout)] = build_sheet(paper, sst, srng, self.topo)
            return sheets[(paper, layout)]

        papers = [st.paper] if st.paper else list(PAPER_ORDER)
        floor = min(FORCED_MIN_SCALE, st.scale) if st.paper else min(MIN_SCALE, st.scale)

        def search(layout: str):
            """Largest symbol scale (from ``st.scale`` down to ``floor``) at which the drawing fits one
            of ``papers``; returns ((paper, pick), None) or (None, failure message)."""
            scale = st.scale
            while True:
                self.set_scale(scale)
                single = self.layout(1.0, 1.0)
                options = [(None, _size(single.extent()))]
                m = self.band_split(single.stage_x)
                if m is not None:
                    options.append((m, _size(self.layout(1.0, 1.0, m).extent())))
                for paper in papers:
                    area = sheet_for(paper, layout).drawing_area
                    aw, ah = area.width - 30, area.height - 30
                    fits = [(min(w / aw, h / ah), split, w, h) for split, (w, h) in options if w <= aw and h <= ah]
                    if fits:
                        fits.sort(key=lambda f: -f[0])
                        pick = fits[0]
                        if len(fits) > 1 and self.rng.random() < 0.25:
                            pick = fits[1]
                        return (paper, pick), ""
                if scale <= floor:
                    w, h = min((sz for _, sz in options), key=lambda sz: max(sz[0] / aw, sz[1] / ah))
                    return None, (f"drawing does not fit {papers[-1]} ({w:.0f} x {h:.0f} pt at symbol scale "
                                  f"{scale:.1f}; drawing area {aw:.0f} x {ah:.0f} pt, title block {layout})")
                scale = max(floor, scale * 0.88)

        layout = st.sheet_layout
        chosen, msg = search(layout)
        if chosen is None and st.paper:  # forced sheet: try the other title-block arrangement before failing
            layout = "bottom" if st.sheet_layout == "right" else "right"
            chosen, msg2 = search(layout)
            msg = f"{msg}; {msg2}"
        if chosen is None:
            raise LayoutError(msg)
        self.sheet_layout = layout
        paper, (_, split, bw, bh) = chosen
        if not st.paper and paper != "A1" and self.rng.random() < 0.15:
            paper = PAPER_ORDER[PAPER_ORDER.index(paper) + 1]
        sheet = sheet_for(paper, layout)
        area = sheet.drawing_area
        aw, ah = area.width - 30, area.height - 30
        tw_, th_ = _size(self.layout(2.0, 2.0, split).extent())
        fill = self.rng.uniform(0.82, 0.95)

        def solve(b: float, t: float, avail: float) -> float:
            if t - b < 1e-6:
                return 1.0
            return max(1.0, min(2.6, 1.0 + (avail * fill - b) / (t - b)))

        fu, fv = solve(bw, tw_, aw), solve(bh, th_, ah)
        scene = None
        for _ in range(10):
            scene = self.layout(fu, fv, split)
            w, h = _size(scene.extent())
            if w <= aw and h <= ah:
                break
            fu = max(1.0, fu * (0.9 if w > aw else 1.0))
            fv = max(1.0, fv * (0.9 if h > ah else 1.0))
        else:
            scene = self.layout(1.0, 1.0, split)
        ex = scene.extent()
        cx = (area.x0 + area.x1) / 2 + self.rng.uniform(-0.04, 0.04) * (aw - (ex[2] - ex[0]))
        cy = (area.y0 + area.y1) / 2 + self.rng.uniform(-0.04, 0.04) * (ah - (ex[3] - ex[1]))
        scene.translate(cx - (ex[0] + ex[2]) / 2, cy - (ex[1] + ex[3]) / 2)
        scene.fu, scene.fv, scene.split = fu, fv, split
        return scene, sheet


def _size(r: Rect) -> tuple[float, float]:
    return r[2] - r[0], r[3] - r[1]


# --------------------------------------------------------------------------- scene


class _Scene:
    def __init__(self) -> None:
        self.net = _Net()
        self.ends: list[tuple] = []  # (stage, Train, node, (x, y))
        self.starts: list[tuple] = []  # (stage, group, train key | "fan", node, (x, y))
        self.headers: dict[int, dict[int, float]] = {}
        self.stage_x: list[tuple[float, float]] = []
        self.breaks: dict[int, tuple[float, float]] = {}  # stage boundary -> (x of second header, transfer y)
        self.fu = self.fv = 1.0
        self.split: Optional[int] = None

    def extent(self) -> Rect:
        return self.net.extent()

    def connect_headers(self, gen: _Gen, fu: float) -> None:
        """Vertical header per (stage boundary, group) joining open ends to open starts; at a band
        break the two headers are joined by a transfer line between the bands."""
        net = self.net
        for k, hdr in sorted(self.headers.items()):
            for g, xh in sorted(hdr.items()):
                L = [(y, node, "L") for (sk, tr, node, (x, y)) in self.ends
                     if sk == k and tr.group == g and not tr.end_return]
                R = [(y, node, "R") for (sk, gg, tkey, node, (x, y)) in self.starts if sk == k + 1 and gg == g]
                if not L and not R:
                    continue
                if not L or not R:
                    raise LayoutError(f"header group {k}/{g} has no inlet or no outlet")
                hid = f"H{k}g{g}"
                if k in self.breaks:
                    xb, yt = self.breaks[k]
                    na = _header(net, hid + "a", hid, xh, L + [(yt, "", "X")])
                    nb = _header(net, hid + "b", hid, xb, [(yt, "", "X")] + R)
                    net.edge(na[round(yt, 6)], nb[round(yt, 6)], line=hid, size="main")
                else:
                    _header(net, hid, hid, xh, L + R)

    def translate(self, dx: float, dy: float) -> None:
        net = self.net
        for o in net.objs.values():
            o.pl = replace(o.pl, cx=o.pl.cx + dx, cy=o.pl.cy + dy)
            if o.box:
                o.box = (o.box[0] + dx, o.box[1] + dy, o.box[2] + dx, o.box[3] + dy)
            o.ports = {k: (p[0] + dx, p[1] + dy) for k, p in o.ports.items()}
        net.pos = {k: (p[0] + dx, p[1] + dy) for k, p in net.pos.items()}
        net.reserve = [(r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy) for r in net.reserve]
        self.ends = [(a, b, c, (p[0] + dx, p[1] + dy)) for a, b, c, p in self.ends]
        self.stage_x = [(a + dx, b + dx) for a, b in self.stage_x]


def _header(net: _Net, hid: str, line: str, xh: float, pts: list[tuple[float, str, str]]) -> dict[float, str]:
    """Vertical header at ``xh`` through attachment points (y, node, side): 'L' open ends arriving from
    the left, 'R' open starts leaving to the right, 'X' a bare node (band transfer corner)."""
    by_y: dict[float, list[tuple[str, str]]] = {}
    exact: dict[float, float] = {}
    for y, node, side in pts:
        key = round(y, 6)
        by_y.setdefault(key, []).append((node, side))
        exact.setdefault(key, y)
    out: dict[float, str] = {}
    hnodes = []
    for key in sorted(by_y):
        y = exact[key]
        hn = f"{hid}@{y:.4f}"
        net.pos[hn] = (xh, y)
        hnodes.append(hn)
        out[key] = hn
        for node, side in by_y[key]:
            if side == "L":
                if abs(net.pos[node][0] - xh) < 1e-9:
                    raise LayoutError("open end on header")
                net.edge(node, hn, line=_line_of(net, node), size=_size_of(net, node))
            elif side == "R":
                if abs(net.pos[node][0] - xh) > 1e-9:
                    net.edge(hn, node, line=f"{hid}:{node}", size="main")
                else:
                    _rename_node(net, node, hn)
    for a, b in zip(hnodes, hnodes[1:]):
        net.edge(a, b, line=line, size="main", flow=None)
    return out


def _line_of(net: _Net, node: str) -> Optional[str]:
    for e in net.edges:
        if e.b == node or e.a == node:
            return e.line
    return None


def _size_of(net: _Net, node: str) -> str:
    for e in net.edges:
        if e.b == node or e.a == node:
            return e.size
    return "main"


def _rename_node(net: _Net, old: str, new: str) -> None:
    for e in net.edges:
        if e.a == old:
            e.a = new
        if e.b == old:
            e.b = new
    net.pos.pop(old, None)
    if old in net.forced:
        net.forced.discard(old)
        net.forced.add(new)


# --------------------------------------------------------------------------- spatial index


class _Index:
    """Uniform-grid index of axis-aligned rectangles."""

    def __init__(self, cell: float = 32.0):
        self.cell = cell
        self.grid: dict[tuple[int, int], list[int]] = {}
        self.rects: list[Rect] = []
        self.tags: list[object] = []

    def _cells(self, r: Rect) -> Iterable[tuple[int, int]]:
        c = self.cell
        for i in range(int(math.floor(r[0] / c)), int(math.floor(r[2] / c)) + 1):
            for j in range(int(math.floor(r[1] / c)), int(math.floor(r[3] / c)) + 1):
                yield (i, j)

    def add(self, r: Rect, tag: object = None) -> None:
        idx = len(self.rects)
        self.rects.append(r)
        self.tags.append(tag)
        for cell in self._cells(r):
            self.grid.setdefault(cell, []).append(idx)

    def hits(self, r: Rect, ignore: Sequence[object] = ()) -> list[int]:
        seen: set[int] = set()
        out = []
        for cell in self._cells(r):
            for i in self.grid.get(cell, ()):
                if i in seen:
                    continue
                seen.add(i)
                q = self.rects[i]
                if q[0] < r[2] and r[0] < q[2] and q[1] < r[3] and r[1] < q[3] and self.tags[i] not in ignore:
                    out.append(i)
        return out


def _seg_rect(a: Pt, b: Pt, pad: float) -> Rect:
    return (min(a[0], b[0]) - pad, min(a[1], b[1]) - pad, max(a[0], b[0]) + pad, max(a[1], b[1]) + pad)


def _overlap(a: Rect, b: Rect, tol: float = 0.0) -> bool:
    return a[0] < b[2] - tol and b[0] < a[2] - tol and a[1] < b[3] - tol and b[1] < a[3] - tol


# --------------------------------------------------------------------------- finishing


@dataclass
class _Run:
    nodes: list[str]
    edges: list[_Edge]
    kind: PipeKind
    line: Optional[str]
    size: str
    medium: Optional[Medium]
    code: Optional[str]
    flow: Optional[bool]


@dataclass
class _Text:
    text: str
    x: float
    y: float
    size: float
    font: str
    rotate: int
    role: TextRole
    owner: Optional[str]  # object key or run index ("p:<i>")
    bbox: Rect
    layer: str = "text"


class _Finisher:
    """Return lines, network merge, labels, validation and output for a composed scene."""

    def __init__(self, gen: _Gen, scene: _Scene, sheet: Sheet):
        self.g = gen
        self.sc = scene
        self.sheet = sheet
        self.net = scene.net
        self.s = gen.s
        self.st = gen.st
        self.area = sheet.drawing_area
        self.font = gen.font
        self.texts: list[_Text] = []
        self.dropped = 0
        self.dropped_texts: list[str] = []

    # -- return lines ------------------------------------------------------------
    def route_returns(self) -> None:
        for rl in self.g.topo.returns:
            self._route_return(rl)

    def route_drains(self) -> None:
        """Optional drain lines; a drain without a clean route is left out of the drawing."""
        self.skipped_drains = 0
        for dr in self.g.topo.drains:
            k = self.g.stage_index.get(id(dr.stage))
            if k is None or k == self.g.fanout_stage or not self._route_drain(dr, k):
                self.skipped_drains += 1

    def _route_drain(self, dr, k: int) -> bool:
        net, s = self.net, self.s
        trains = self.g.topo.stages[k].trains
        prefix = f"{trains[0].key}#"
        stage_boxes = [o.bbox() for o in net.objs.values() if o.stage == k]
        if not stage_boxes:
            return False
        bottom = max(b[3] for b in stage_boxes)
        keys = tuple(f"{t.key}#" for t in trains)
        for e in net.edges:
            if e.line and e.line.startswith(keys):
                bottom = max(bottom, net.pos[e.a][1], net.pos[e.b][1])
        obstacles = [(o.key, o.bbox()) for o in net.objs.values()]
        segs = self._segments()
        cands = []
        for i, e in enumerate(net.edges):
            if e.kind != PipeKind.PROCESS or not e.line or not e.line.startswith(prefix):
                continue
            a, b = net.pos[e.a], net.pos[e.b]
            if abs(a[1] - b[1]) > 1e-9 or abs(a[0] - b[0]) < 2.4 * s:
                continue
            x0, x1 = sorted((a[0], b[0]))
            n = int((x1 - x0 - 2.0 * s) / (0.3 * s)) + 1
            cands += [(i, x0 + 1.0 * s + j * 0.3 * s, a[1]) for j in range(n)]
        if not cands:
            return False
        rng = random.Random(f"{self.g.seed}:drain:{k}")
        rng.shuffle(cands)
        vit, cit = dr.valve, dr.connector
        vrot = _orientations(vit.cls, (("in", (0.0, -1.0)), ("out", (0.0, 1.0))), False)[0] if vit else None
        crot = _orientations(cit.cls, (("in", (0.0, -1.0)),), False)[0]
        vsc, csc = (self.g._scale_of(vit) if vit else 0.0), self.g._scale_of(cit)
        clear = 0.35 * s
        area = self.area
        for i, x, y in cands[:60]:
            new_objs = []
            pts1_end = (x, y + 1.1 * s)
            if vit:
                pl0 = Placement(0, 0, vsc, vrot[0], vrot[1])
                ps = {q.name: q for q in get_symbol(vit.cls).ports}
                ia, oa = pl0.to_page(ps["in"].x, ps["in"].y), pl0.to_page(ps["out"].x, ps["out"].y)
                vpl = Placement(x - ia[0], pts1_end[1] - ia[1], vsc, vrot[0], vrot[1])
                vo = _Obj(vit.key, vit, vpl, role="deco", axis="v", stage=k)
                new_objs.append(vo)
                start2 = (x + oa[0] - ia[0], pts1_end[1] + oa[1] - ia[1])
            else:
                start2 = pts1_end
            y_end = max(bottom + 1.2 * s, start2[1] + 1.0 * s)
            cpl0 = Placement(0, 0, csc, crot[0], crot[1])
            cp = cpl0.to_page(get_symbol(cit.cls).ports[0].x, get_symbol(cit.cls).ports[0].y)
            co = _Obj(cit.key, cit, Placement(x - cp[0], y_end - cp[1], csc, crot[0], crot[1]), role="deco",
                      axis="v", stage=k)
            new_objs.append(co)
            ok = True
            for o in new_objs:
                b = o.bbox()
                if b[0] < area.x0 + 3 or b[2] > area.x1 - 3 or b[3] > area.y1 - 3 - 2 * self.g.lh_label:
                    ok = False
                    break
                if any(_overlap(_expand(b, clear), ob) for _, ob in obstacles):
                    ok = False
                    break
                if any(_overlap(_seg_rect(c, d, 0.5), _expand(b, 0.3)) for c, d in segs):
                    ok = False
                    break
            if not ok:
                continue
            path1 = [(x, y), pts1_end] if vit else [(x, y), (x, y_end)]
            if not self._path_ok(path1, obstacles, segs, set()):
                continue
            if vit and not self._path_ok([start2, (x, y_end)], obstacles, segs, set()):
                continue
            # commit: split the train pipe at the tee and add the drain
            e = net.edges[i]
            j = net.node((x, y), "d")
            net.edges[i] = replace(e, b=j)
            net.edges.append(replace(e, a=j))
            lid = f"drain{k}"
            for o in new_objs:
                net.add(o)
            pc = net.port(cit.key, "in")
            if vit:
                net.edge(j, net.port(vit.key, "in"), line=lid, size="small")
                net.edge(net.port(vit.key, "out"), pc, line=lid, size="small")
            else:
                net.edge(j, pc, line=lid, size="small")
            return True
        return False

    def _segments(self) -> list[tuple[Pt, Pt]]:
        return [(self.net.pos[e.a], self.net.pos[e.b]) for e in self.net.edges]

    def _route_return(self, rl) -> None:
        net, s = self.net, self.s
        tank = net.objs.get(rl.target)
        if tank is None or tank.box is None:
            raise LayoutError("return target missing")
        if rl.train:
            ends = [e for e in self.sc.ends if e[1].key == rl.train]
            if not ends:
                raise LayoutError("return train missing")
            start = ends[0][2]
            sx, sy = net.pos[start]
            start_dir = (1.0, 0.0)
            start_obj = None
        else:
            o = net.objs[rl.item]
            start = net.port(o.key, rl.port)
            sx, sy = o.ports[rl.port]
            start_dir = o.pdirs[rl.port]
            start_obj = o.key
        tx0, ty0, tx1, ty1 = tank.box
        out_ports = [p for p in tank.ports.values() if abs(p[0] - tx1) < 1e-6]
        # channel between the source column and the first stage-0 items
        src_right = max(o.bbox()[2] for o in net.objs.values() if o.role == "source" and o.stage == 0)
        firsts = [o.bbox()[0] for o in net.objs.values() if o.stage == 0 and o.role not in ("source",)
                  and o.bbox()[0] > src_right - 1e-6]
        first_x = min(firsts) if firsts else src_right + 3 * s
        obstacles = [(o.key, o.bbox()) for o in net.objs.values()]
        segs = self._segments()
        best = None
        for lane in ("top", "bottom"):
            for fx in (0.5, 0.35, 0.65):
                xc = src_right + (first_x - src_right) * fx
                for frac in (0.25, 0.75, 0.15, 0.85):
                    yt = ty0 + (ty1 - ty0) * frac
                    if any(abs(yt - p[1]) < 0.6 * s for p in out_ports):
                        continue
                    if (lane == "top") != (yt < min((p[1] for p in out_ports), default=yt + 1)):
                        continue
                    for d in (1.2, 1.8, 0.8):
                        cand = self._return_candidate(lane, (sx, sy), start_dir, xc, yt, tx1, d, obstacles, segs,
                                                      tank.key, start_obj)
                        if cand is not None and (best is None or cand[0] < best[0]):
                            best = cand
                    if best is not None:
                        break
                if best is not None:
                    break
        if best is None:
            raise LayoutError("no route for return line")
        _, pts = best
        lid = f"ret:{rl.target}:{start}"
        nodes = [start] + [net.node(p, "r") for p in pts[1:-1]]
        end = net.bport(tank.key, pts[-1], (1.0, 0.0))
        nodes.append(end)
        size = "branch"
        if rl.valve is not None:
            # put the valve on the longest horizontal leg
            best_i, best_len = None, 0.0
            for i in range(len(nodes) - 1):
                a, b = net.pos[nodes[i]], net.pos[nodes[i + 1]]
                if abs(a[1] - b[1]) < 1e-9 and abs(a[0] - b[0]) > best_len:
                    best_i, best_len = i, abs(a[0] - b[0])
            if best_i is not None and best_len > 3 * s:
                a, b = net.pos[nodes[best_i]], net.pos[nodes[best_i + 1]]
                it = rl.valve
                dirx = 1.0 if b[0] > a[0] else -1.0
                rot, mir = _orientations(it.cls, (("in", (-dirx, 0.0)), ("out", (dirx, 0.0))), True)[0]
                pl = Placement((a[0] + b[0]) / 2, a[1], self.g._scale_of(it), rot, mir)
                o = net.add(_Obj(it.key, it, pl, role="deco"))
                if any(_overlap(_expand(o.bbox(), 2), r) for _, r in obstacles):
                    del net.objs[it.key]
                else:
                    pin, pout = net.port(it.key, "in"), net.port(it.key, "out")
                    net.path(nodes[: best_i + 1] + [pin], line=lid, size=size)
                    net.path([pout] + nodes[best_i + 1:], line=lid, size=size)
                    return
        net.path(nodes, line=lid, size=size)

    def _return_candidate(self, lane, start, sdir, xc, yt, tx1, d, obstacles, segs, tank_key, start_obj):
        s = self.s
        sx, sy = start
        if sdir == (1.0, 0.0):
            x1 = sx + d * s
            head = [(sx, sy), (x1, sy)]
            xa = x1
        else:
            head = [(sx, sy)]
            xa = sx
            if (lane == "top" and sdir != (0.0, -1.0)) or (lane == "bottom" and sdir != (0.0, 1.0)):
                return None
        lo_x, hi_x = min(xc, xa), max(xc, xa)
        ys = []
        for key, b in obstacles:
            if b[2] > lo_x - s and b[0] < hi_x + s:
                ys.append(b)
        for a, b in segs:
            if max(a[0], b[0]) > lo_x - s and min(a[0], b[0]) < hi_x + s:
                ys.append((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])))
        if not ys:
            return None
        if lane == "top":
            yl = min(r[1] for r in ys) - 1.4 * s - self.g.lh_label
            if yl < self.area.y0 + 10:
                return None
        else:
            yl = max(r[3] for r in ys) + 1.4 * s + self.g.lh_label
            if yl > self.area.y1 - 10:
                return None
        pts = head + [(xa, yl), (xc, yl), (xc, yt), (tx1, yt)]
        # drop zero-length / collinear points
        clean = [pts[0]]
        for p in pts[1:]:
            if abs(p[0] - clean[-1][0]) < 1e-9 and abs(p[1] - clean[-1][1]) < 1e-9:
                continue
            clean.append(p)
        pts = _simplify(clean)
        if not self._path_ok(pts, obstacles, segs, {tank_key, start_obj}):
            return None
        length = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(pts, pts[1:]))
        return (length, pts)

    def _path_ok(self, pts: list[Pt], obstacles, segs, own: set) -> bool:
        s = self.s
        clear = 0.35 * s
        n = len(pts) - 1
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            r = _seg_rect(a, b, 0.0)
            for key, ob in obstacles:
                if key in own:
                    shrink = _expand(ob, -0.5)
                    if _overlap(r, shrink):
                        return False
                    continue
                if _overlap(_seg_rect(a, b, clear), ob):
                    return False
            horiz = abs(a[1] - b[1]) < 1e-9
            for c, d in segs:
                ch = abs(c[1] - d[1]) < 1e-9
                if horiz == ch:  # parallel: must keep a distance where they overlap
                    if horiz:
                        if abs(a[1] - c[1]) < clear and min(a[0], b[0]) < max(c[0], d[0]) - 1e-6 and \
                                min(c[0], d[0]) < max(a[0], b[0]) - 1e-6:
                            if not (i == 0 and _touches(a, c, d)):
                                return False
                    else:
                        if abs(a[0] - c[0]) < clear and min(a[1], b[1]) < max(c[1], d[1]) - 1e-6 and \
                                min(c[1], d[1]) < max(a[1], b[1]) - 1e-6:
                            if not (i == 0 and _touches(a, c, d)):
                                return False
                else:  # perpendicular: crossing allowed away from the other segment's ends
                    if horiz:
                        x, y = c[0], a[1]
                        inside = min(a[0], b[0]) - 1e-6 <= x <= max(a[0], b[0]) + 1e-6 and \
                            min(c[1], d[1]) - 1e-6 <= y <= max(c[1], d[1]) + 1e-6
                    else:
                        x, y = a[0], c[1]
                        inside = min(c[0], d[0]) - 1e-6 <= x <= max(c[0], d[0]) + 1e-6 and \
                            min(a[1], b[1]) - 1e-6 <= y <= max(a[1], b[1]) + 1e-6
                    if inside:
                        if i == 0 and abs(x - a[0]) < 1e-6 and abs(y - a[1]) < 1e-6:
                            continue  # own start point
                        near = min(math.hypot(x - p[0], y - p[1]) for p in (a, b, c, d))
                        if near < clear + self.st.hop_radius:
                            return False
        return True

    # -- network -> runs ---------------------------------------------------------
    def merge_runs(self) -> list[_Run]:
        net = self.net
        adj: dict[str, list[int]] = {}
        for i, e in enumerate(net.edges):
            adj.setdefault(e.a, []).append(i)
            adj.setdefault(e.b, []).append(i)
        for n, lst in adj.items():
            if n not in net.port_of and len(lst) == 1:
                raise LayoutError(f"dangling pipe end at {n}")
        stop = {n for n, lst in adj.items() if n in net.port_of or len(lst) != 2 or n in net.forced}
        used = [False] * len(net.edges)
        runs: list[_Run] = []
        for i, e in enumerate(net.edges):
            if used[i]:
                continue
            # walk back to a stop node
            chain_edges = [i]
            used[i] = True
            # extend forward from e.b and backward from e.a
            fwd_nodes = [e.a, e.b]
            cur = e.b
            while cur not in stop:
                nxt = [j for j in adj[cur] if not used[j]]
                if not nxt:
                    break
                j = nxt[0]
                used[j] = True
                chain_edges.append(j)
                f = net.edges[j]
                cur = f.b if f.a == cur else f.a
                fwd_nodes.append(cur)
            cur = e.a
            back_nodes: list[str] = []
            back_edges: list[int] = []
            while cur not in stop:
                nxt = [j for j in adj[cur] if not used[j]]
                if not nxt:
                    break
                j = nxt[0]
                used[j] = True
                back_edges.append(j)
                f = net.edges[j]
                cur = f.b if f.a == cur else f.a
                back_nodes.append(cur)
            nodes = list(reversed(back_nodes)) + fwd_nodes
            edges = [net.edges[j] for j in list(reversed(back_edges)) + chain_edges]
            kinds = {x.kind for x in edges}
            if len(kinds) != 1:
                raise LayoutError("mixed pipe kinds in one run")
            # orientation by flow
            fw = bw = 0
            for k, ed in enumerate(edges):
                a, b = nodes[k], nodes[k + 1]
                if ed.flow is None:
                    continue
                if (ed.a, ed.b) == (a, b):
                    fw += 1
                else:
                    bw += 1
            flow: Optional[bool] = True if fw or bw else None
            if bw > fw:
                nodes.reverse()
                edges.reverse()
            main = max(edges, key=lambda ed: _elen(net, ed))
            lines = sorted({ed.line for ed in edges if ed.line})
            # a run bending around a header corner belongs to the train line, not to the header
            line = max(lines, key=lambda ln: (_line_rank(ln), sum(_elen(net, ed) for ed in edges if ed.line == ln))) \
                if lines else None
            med = next((ed.medium for ed in edges if ed.medium), None)
            code = next((ed.code for ed in edges if ed.code), None)
            runs.append(_Run(nodes, edges, main.kind, line, main.size, med, code,
                             flow if main.kind != PipeKind.INSTRUMENT_TAP else None))
        return runs

    # -- labels ------------------------------------------------------------------
    def build_index(self, runs: list[_Run], run_pts: list[list[Pt]]) -> _Index:
        idx = _Index(cell=max(24.0, self.s * 1.5))
        for key, o in self.net.objs.items():
            idx.add(_expand(o.bbox(), 1.2), ("obj", key))
        for i, pts in enumerate(run_pts):
            w = self._pipe_width(runs[i]) / 2 + 1.0
            for a, b in zip(pts, pts[1:]):
                idx.add(_seg_rect(a, b, w), ("pipe", i))
        for r in self.sheet.occupied:
            idx.add(r, ("sheet", 0))
        return idx

    def _free(self, idx: _Index, r: Rect, ignore=()) -> bool:
        a = self.area
        if r[0] < a.x0 + 3 or r[1] < a.y0 + 3 or r[2] > a.x1 - 3 or r[3] > a.y1 - 3:
            return False
        return not idx.hits(r, ignore)

    def _add_text(self, idx: Optional[_Index], text: str, x0: float, y0: float, size: float, font: str,
                  role: TextRole, owner: Optional[str], rotate: int = 0) -> _Text:
        ox, oy = origin_for_box(text, x0, y0, size, font, rotate)
        bb = text_bbox(text, ox, oy, size, font, rotate)
        t = _Text(text, ox, oy, size, font, rotate, role, owner, bb)
        self.texts.append(t)
        if idx is not None:
            idx.add(_expand(bb, 0.8), ("text", len(self.texts) - 1))
        return t

    def inside_texts(self, idx: _Index) -> None:
        """Instrument letters / loop numbers, connector references and tank names (fixed positions)."""
        for key, o in self.net.objs.items():
            it = o.item
            if it.category == Category.INSTRUMENT and it.inside:
                fs = self.g.inst_fs
                h = line_height(self.font, fs)
                gap = 0.6 if it.cls == S.INSTRUMENT_PANEL else 0.4
                top, bot = it.inside[0], it.inside[1] if len(it.inside) > 1 else ""
                cx, cy = o.pl.cx, o.pl.cy
                w = text_width(top, self.font, fs)
                self._add_text(idx, top, cx - w / 2, cy - gap - h, fs, self.font, TextRole.INSTRUMENT, key)
                if bot:
                    w = text_width(bot, self.font, fs)
                    self._add_text(idx, bot, cx - w / 2, cy + gap, fs, self.font, TextRole.INSTRUMENT, key)
            elif it.cls == S.OFF_PAGE_CONNECTOR and it.inside:
                self._connector_text(idx, o)
            elif it.cls == S.TANK and it.name_inside and it.name and o.box:
                fs = self.st.label_size
                lh = self.g.lh_label * 1.15
                bx = o.box
                h = lh * len(it.name)
                y = (bx[1] + bx[3]) / 2 - h / 2
                for ln in it.name:
                    w = text_width(ln, self.font, fs)
                    self._add_text(idx, ln, (bx[0] + bx[2]) / 2 - w / 2, y + (lh - self.g.lh_label) / 2, fs,
                                   self.font, TextRole.LABEL, key)
                    y += lh

    def _connector_text(self, idx: _Index, o: _Obj) -> None:
        ref = o.item.inside[0]
        sc = o.pl.scale
        vertical = o.pl.rotation in (90, 270)
        # usable part of the pentagon: u in [-0.75, 0.45], v in [-0.2, 0.2]
        a = o.pl.to_page(-0.75, 0.0)
        b = o.pl.to_page(0.45, 0.0)
        cxy = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        length = 1.2 * sc
        thick = 0.4 * sc
        fs = min(self.st.tag_size, thick / (font_metrics(self.font)[0] - font_metrics(self.font)[1]) * 0.95)
        while fs > 3.0 and text_width(ref, self.font, fs) > length * 0.92:
            fs -= 0.2
        w = text_width(ref, self.font, fs)
        h = line_height(self.font, fs)
        if vertical:
            self._add_text(idx, ref, cxy[0] - h / 2, cxy[1] - w / 2, fs, self.font, TextRole.CONNECTOR, o.key, 90)
        else:
            self._add_text(idx, ref, cxy[0] - w / 2, cxy[1] - h / 2, fs, self.font, TextRole.CONNECTOR, o.key)

    def _nearest_owner(self, idx: _Index, rects: Sequence[Rect], owner: str) -> bool:
        """True when every text box in ``rects`` is unambiguously nearest to object ``owner``: by box gap
        (how the extractor associates texts; another object must be ``NEAR_GAP_MARGIN`` farther) and
        from the text centre (no other object more than ``NEAR_CENTRE_TOL`` closer)."""
        ob = self.net.objs[owner].bbox()
        for r in rects:
            g0, c0 = _gap(r, ob), _cdist(r, ob)
            for i in idx.hits(_expand(r, max(g0 + NEAR_GAP_MARGIN, c0) + 0.5)):
                tag = idx.tags[i]
                if not isinstance(tag, tuple) or tag[0] != "obj" or tag[1] == owner:
                    continue
                fb = self.net.objs[tag[1]].bbox()
                if _gap(r, fb) < g0 + NEAR_GAP_MARGIN or _cdist(r, fb) < c0 - NEAR_CENTRE_TOL:
                    return False
        return True

    def _nearest_pipe(self, idx: _Index, r: Rect, i: int, run_pts: list[list[Pt]]) -> bool:
        """True when the text box ``r`` is closer to pipe run ``i`` than to any other run."""
        def dists(k: int) -> tuple[float, float]:
            segs = [_seg_rect(a, b, 0.0) for a, b in zip(run_pts[k], run_pts[k][1:])]
            return min(_gap(r, q) for q in segs), min(_cdist(r, q) for q in segs)

        g0, c0 = dists(i)
        seen = {i}
        for j in idx.hits(_expand(r, max(g0 + NEAR_GAP_MARGIN, c0) + 0.5)):
            tag = idx.tags[j]
            if not isinstance(tag, tuple) or tag[0] != "pipe" or tag[1] in seen:
                continue
            seen.add(tag[1])
            g, c = dists(tag[1])
            if g < g0 + NEAR_GAP_MARGIN or c < c0 - NEAR_CENTRE_TOL:
                return False
        return True

    def _block(self, idx: _Index, lines: list[str], size: float, font: str, box: Rect, prefs: Sequence[str],
               role: TextRole, owner: str, roles: Optional[list[TextRole]] = None, gap: float = 2.0) -> bool:
        """Place a block of label lines beside ``box`` at the first free position (in ``prefs`` order,
        widening the gap) that is unambiguously nearest to its owner; False if there is none."""
        lines = [ln for ln in lines if ln]
        if not lines:
            return True
        lh = line_height(font, size) * 1.12
        lh0 = line_height(font, size)
        ws = [text_width(ln, font, size) for ln in lines]
        W, H = max(ws), lh * len(lines) - (lh - line_height(font, size))
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for mult in (1.0, 2.2, 4.0):
            gg = gap * mult
            for p in prefs:
                if p == "above":
                    bx, by, align = cx - W / 2, y0 - gg - H, "c"
                elif p == "below":
                    bx, by, align = cx - W / 2, y1 + gg, "c"
                elif p == "right":
                    bx, by, align = x1 + gg, cy - H / 2, "l"
                elif p == "left":
                    bx, by, align = x0 - gg - W, cy - H / 2, "r"
                elif p == "above_right":
                    bx, by, align = x1 + gg * 0.5, y0 - gg - H, "l"
                elif p == "below_right":
                    bx, by, align = x1 + gg * 0.5, y1 + gg, "l"
                elif p == "above_left":
                    bx, by, align = x0 - gg * 0.5 - W, y0 - gg - H, "r"
                elif p == "below_left":
                    bx, by, align = x0 - gg * 0.5 - W, y1 + gg, "r"
                elif p == "right_top":
                    bx, by, align = x1 + gg, y0, "l"
                elif p == "left_top":
                    bx, by, align = x0 - gg - W, y0, "r"
                elif p == "right_bottom":
                    bx, by, align = x1 + gg, y1 - H, "l"
                elif p == "left_bottom":
                    bx, by, align = x0 - gg - W, y1 - H, "r"
                else:
                    continue
                r = (bx, by, bx + W, by + H)
                if not self._free(idx, _expand(r, 0.8)):
                    continue
                xs = [bx + (W - w) / 2 if align == "c" else (bx if align == "l" else bx + W - w) for w in ws]
                line_rects = [(xs[i], by + i * lh, xs[i] + ws[i], by + i * lh + lh0) for i in range(len(lines))]
                if not self._nearest_owner(idx, [r] + line_rects, owner):
                    continue
                for i, ln in enumerate(lines):
                    self._add_text(idx, ln, xs[i], by + i * lh, size, font, (roles[i] if roles else role), owner)
                return True
        return False

    def object_labels(self, idx: _Index) -> dict[str, bool]:
        """Names, tags and connector labels; returns which object tags were drawn."""
        drawn: dict[str, bool] = {}
        order = sorted(self.net.objs.values(), key=lambda o: (_label_rank(o), o.key))
        fs_t, fs_l = self.st.tag_size, self.st.label_size
        for o in order:
            it = o.item
            cat = it.category
            box = o.bbox()
            if cat == Category.INSTRUMENT:
                drawn[o.key] = True
                continue
            if it.cls == S.OFF_PAGE_CONNECTOR:
                if it.name:
                    tip = o.pl.to_page(0.8, 0)
                    d = (tip[0] - o.pl.cx, tip[1] - o.pl.cy)
                    if abs(d[0]) > abs(d[1]):
                        prefs = ["right", "right_top", "above", "below"] if d[0] > 0 else \
                            ["left", "left_top", "above", "below"]
                    else:
                        prefs = ["right_top", "left_top", "right", "left", "above_right", "below_right"]
                    self._block(idx, list(it.name), fs_l, self.font, box, prefs, TextRole.CONNECTOR, o.key)
                continue
            if cat == Category.VALVE or cat == Category.FITTING:
                if it.tag:
                    if o.axis == "h":
                        prefs = ["above", "below", "above_right", "below_right", "above_left", "below_left"]
                    else:
                        prefs = ["right", "left", "right_top", "left_top", "above_right", "below_right"]
                    if o.axis == "h" and _rdir(o.pl.dir_to_page(0, -1)) == (0.0, 1.0):
                        prefs = ["below", "above"] + prefs[2:]
                    drawn[o.key] = self._block(idx, [it.tag], fs_t, self.font, box, prefs, TextRole.TAG, o.key)
                continue
            lines: list[str] = []
            roles: list[TextRole] = []
            if it.tag:
                lines.append(it.tag)
                roles.append(TextRole.TAG)
            if it.name and not (it.cls == S.TANK and it.name_inside):
                lines += list(it.name)
                roles += [TextRole.LABEL] * len(it.name)
            if not lines:
                continue
            if o.boundary or it.cls in (S.SEA_CHEST, S.BWTS_UNIT):
                prefs = ["above", "below", "right", "left", "above_right", "below_right", "above_left", "below_left"]
            elif it.cls in (S.PUMP_CENTRIFUGAL, S.PUMP_SCREW, S.COMPRESSOR):
                prefs = ["below", "below_right", "below_left", "above", "right", "left", "above_right", "above_left"]
            elif o.axis == "v":
                prefs = ["right", "left", "right_top", "left_top", "above", "below"]
            else:
                prefs = ["above", "below", "above_right", "below_right", "above_left", "below_left", "right", "left"]
            prefs += [p for p in _SIDE_LABEL_PREFS if p not in prefs]  # beside the body, clear of attachments
            ok = self._block(idx, lines, fs_l, self.font, box, prefs, TextRole.LABEL, o.key, roles=roles)
            if not ok and len(lines) > 1:  # drop the capacity line and retry
                ok = self._block(idx, lines[:-1], fs_l, self.font, box, prefs, TextRole.LABEL, o.key,
                                 roles=roles[:-1])
            if not ok:
                self.dropped += 1
                self.dropped_texts.append(" / ".join(lines))
            drawn[o.key] = ok and bool(it.tag)
        return drawn

    def line_numbers(self, idx: _Index, runs: list[_Run], run_pts: list[list[Pt]], numbers: dict[str, str]) -> set[str]:
        """Place one line number per logical line on its longest free straight segment; returns the
        line ids whose number was drawn."""
        drawn: set[str] = set()
        fs = self.st.lineno_size
        font = self.font
        by_line: dict[str, list[int]] = {}
        for i, r in enumerate(runs):
            if r.kind == PipeKind.PROCESS and r.line:
                by_line.setdefault(r.line, []).append(i)
        for line, idxs in by_line.items():
            text = numbers[line]
            w = text_width(text, font, fs)
            h = line_height(font, fs)
            segs = []
            for i in idxs:
                pts = run_pts[i]
                for a, b in zip(pts, pts[1:]):
                    segs.append((abs(a[0] - b[0]) + abs(a[1] - b[1]), i, a, b))
            segs.sort(key=lambda t: -t[0])
            placed = False
            for length, i, a, b in segs:
                if length < w + 8:
                    break
                horiz = abs(a[1] - b[1]) < 1e-9
                off = self._pipe_width(runs[i]) / 2 + 2.4  # clear of the pipe's padded index box
                for t in (0.5, 0.3, 0.7, 0.18, 0.82):
                    px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                    if horiz:
                        lo, hi = min(a[0], b[0]) + 3, max(a[0], b[0]) - 3
                        x0 = min(max(px - w / 2, lo), hi - w)
                        cands = [(x0, py - off - h, 0), (x0, py + off, 0)]
                    else:
                        lo, hi = min(a[1], b[1]) + 3, max(a[1], b[1]) - 3
                        y0 = min(max(py - w / 2, lo), hi - w)
                        cands = [(px - off - h, y0, 90), (px + off, y0, 90)]
                    for x0_, y0_, rot in cands:
                        r = (x0_, y0_, x0_ + (w if rot == 0 else h), y0_ + (h if rot == 0 else w))
                        if self._free(idx, _expand(r, 0.6), ignore=()) and self._nearest_pipe(idx, r, i, run_pts):
                            self._add_text(idx, text, x0_, y0_, fs, font, TextRole.LINE_NUMBER, f"p:{i}", rot)
                            placed = True
                            drawn.add(line)
                            break
                    if placed:
                        break
                if placed:
                    break
            if not placed:
                self.dropped += 1
                self.dropped_texts.append(text)
        return drawn

    def _pipe_width(self, r: _Run) -> float:
        st = self.st
        if r.kind == PipeKind.SIGNAL:
            return st.signal_width
        if r.kind == PipeKind.INSTRUMENT_TAP:
            return st.tap_width
        if st.graded_widths and r.size in ("branch", "small"):
            return st.pipe_width * (0.8 if r.size == "branch" else 0.65)
        return st.pipe_width


_SIDE_LABEL_PREFS = ("right_top", "left_top", "right_bottom", "left_bottom", "above_right", "below_right",
                     "above_left", "below_left", "right", "left")


def _line_rank(line: str) -> int:
    if "#" in line:
        return 3
    if line.startswith("H") or line.startswith("fan"):
        return 1
    return 2


def _label_rank(o: _Obj) -> int:
    cat = o.item.category
    if cat == Category.EQUIPMENT:
        return 0 if o.boundary else 1
    if o.item.cls == S.OFF_PAGE_CONNECTOR:
        return 2
    if cat == Category.VALVE:
        return 3
    return 4


def _expand(r: Rect, d: float) -> Rect:
    return (r[0] - d, r[1] - d, r[2] + d, r[3] + d)


def _gap(a: Rect, b: Rect) -> float:
    """Euclidean gap between two boxes (0 when they touch or overlap)."""
    return math.hypot(max(0.0, a[0] - b[2], b[0] - a[2]), max(0.0, a[1] - b[3], b[1] - a[3]))


def _cdist(a: Rect, b: Rect) -> float:
    """Distance from the centre of box ``a`` to box ``b``."""
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return math.hypot(max(b[0] - cx, 0.0, cx - b[2]), max(b[1] - cy, 0.0, cy - b[3]))


def _touches(p: Pt, c: Pt, d: Pt) -> bool:
    return (abs(p[0] - c[0]) < 1e-6 and abs(p[1] - c[1]) < 1e-6) or (abs(p[0] - d[0]) < 1e-6 and abs(p[1] - d[1]) < 1e-6)


def _elen(net: _Net, e: _Edge) -> float:
    a, b = net.pos[e.a], net.pos[e.b]
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _simplify(pts: list[Pt]) -> list[Pt]:
    """Drop duplicate and collinear intermediate points of an orthogonal polyline."""
    out: list[Pt] = []
    for p in pts:
        if out and abs(p[0] - out[-1][0]) < 1e-9 and abs(p[1] - out[-1][1]) < 1e-9:
            continue
        if len(out) >= 2:
            a, b = out[-2], out[-1]
            if (abs(a[0] - b[0]) < 1e-9 and abs(b[0] - p[0]) < 1e-9) or (abs(a[1] - b[1]) < 1e-9 and abs(b[1] - p[1]) < 1e-9):
                out[-1] = p
                continue
        out.append(p)
    return out


# --------------------------------------------------------------------------- crossings & drawing


def _crossings(run_pts: list[list[Pt]], runs: list[_Run]) -> list[tuple[int, int, int, int, Pt]]:
    """(h_run, h_seg, v_run, v_seg, point) for every perpendicular crossing of different runs."""
    H, V = [], []
    for i, pts in enumerate(run_pts):
        for j, (a, b) in enumerate(zip(pts, pts[1:])):
            if abs(a[1] - b[1]) < 1e-9:
                H.append((a[1], min(a[0], b[0]), max(a[0], b[0]), i, j))
            elif abs(a[0] - b[0]) < 1e-9:
                V.append((a[0], min(a[1], b[1]), max(a[1], b[1]), i, j))
    out = []
    eps = 1e-6
    for y, hx0, hx1, hi, hj in H:
        for x, vy0, vy1, vi, vj in V:
            if vi == hi:
                continue
            if hx0 + eps < x < hx1 - eps and vy0 + eps < y < vy1 - eps:
                out.append((hi, hj, vi, vj, (x, y)))
    return out


def _pipe_segs(pts: list[Pt], cuts: dict[int, list[Pt]], style: DrawingStyle) -> list[tuple]:
    """Drawing segments of a polyline with hops / gaps at the given crossing points (per segment)."""
    segs: list[tuple] = []
    r = style.hop_radius
    for j, (a, b) in enumerate(zip(pts, pts[1:])):
        pts_c = cuts.get(j, [])
        if not pts_c:
            segs.append(("l", a[0], a[1], b[0], b[1]))
            continue
        horiz = abs(a[1] - b[1]) < 1e-9
        sgn = 1.0 if (b[0] - a[0] if horiz else b[1] - a[1]) > 0 else -1.0
        key = (lambda p: p[0]) if horiz else (lambda p: p[1])
        pts_c = sorted(pts_c, key=lambda p: sgn * key(p))
        cur = a
        for c in pts_c:
            if horiz:
                p0, p1 = (c[0] - sgn * r, c[1]), (c[0] + sgn * r, c[1])
            else:
                p0, p1 = (c[0], c[1] - sgn * r), (c[0], c[1] + sgn * r)
            if math.hypot(p0[0] - cur[0], p0[1] - cur[1]) > 1e-6:
                segs.append(("l", cur[0], cur[1], p0[0], p0[1]))
            if style.crossing == "hop":
                k = 0.5523 * r
                if horiz:  # bump upwards (page -y)
                    m = (c[0], c[1] - r)
                    segs.append(("c", p0[0], p0[1], p0[0], p0[1] - k, m[0] - sgn * k, m[1], m[0], m[1]))
                    segs.append(("c", m[0], m[1], m[0] + sgn * k, m[1], p1[0], p1[1] - k, p1[0], p1[1]))
                else:  # bump to the right (page +x)
                    m = (c[0] + r, c[1])
                    segs.append(("c", p0[0], p0[1], p0[0] + k, p0[1], m[0], m[1] - sgn * k, m[0], m[1]))
                    segs.append(("c", m[0], m[1], m[0], m[1] + sgn * k, p1[0] + k, p1[1], p1[0], p1[1]))
            cur = p1
        if math.hypot(b[0] - cur[0], b[1] - cur[1]) > 1e-6:
            segs.append(("l", cur[0], cur[1], b[0], b[1]))
    return segs


# --------------------------------------------------------------------------- public API


def generate(seed: int, system: Optional[SystemType] = None, style: Optional[DrawingStyle] = None) -> tuple[PIDDocument, DrawingSpec]:
    """Generate one synthetic ship P&ID: exact ground truth and the drawing spec for :mod:`pdf_writer`."""
    forced = style.paper if style is not None else None
    if forced is not None and forced not in PAPER_ORDER:
        raise ValueError(f"DrawingStyle.paper must be one of {PAPER_ORDER} or None, not {forced!r}")
    rng0 = random.Random(seed)
    if system is None or system == SystemType.UNKNOWN:
        system = rng0.choice(ALL_SYSTEMS)
    # a forced (possibly small) sheet may take the topology down to the blueprint minimum
    c_floor, c_step = (0.0, 0.15) if forced else (0.3, 0.12)
    errors = []
    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random(f"{seed}:{attempt}")
        topo = build_topology(system, rng, complexity=max(c_floor, 1.0 - c_step * attempt))
        st = style if style is not None else random_style(rng)
        try:
            return _generate(topo, st, rng, seed, attempt)
        except LayoutError as exc:
            errors.append(str(exc))
    hint = (f" (paper={forced!r} is forced: use a larger sheet or paper=None to pick the smallest sheet that fits)"
            if forced and any("does not fit" in e for e in errors) else "")
    raise RuntimeError(f"synthetic generation failed for seed={seed} system={system.value}{hint}: {errors}")


def _generate(topo: SystemTopology, st: DrawingStyle, rng: random.Random, seed: int,
              attempt: int) -> tuple[PIDDocument, DrawingSpec]:
    gen = _Gen(topo, st, rng, seed)
    scene, sheet = gen.compose()
    fin = _Finisher(gen, scene, sheet)
    fin.route_returns()
    fin.route_drains()
    _check_objects(scene.net, sheet)
    runs = fin.merge_runs()
    net = scene.net
    run_pts = [_simplify([net.pos[n] for n in r.nodes]) for r in runs]
    _check_pipes(net, runs, run_pts, gen.s)

    # ---- ids
    obj_ids = {k: f"o{i + 1}" for i, k in enumerate(net.objs)}
    junction_nodes: list[str] = []
    deg: dict[str, int] = {}
    for e in net.edges:
        deg[e.a] = deg.get(e.a, 0) + 1
        deg[e.b] = deg.get(e.b, 0) + 1
    for r in runs:
        for n in (r.nodes[0], r.nodes[-1]):
            if n not in net.port_of and n not in junction_nodes:
                junction_nodes.append(n)
    jids = {n: f"j{i + 1}" for i, n in enumerate(junction_nodes)}

    # ---- line numbers
    namer = topo.namer
    numbers: dict[str, str] = {}
    line_size: dict[str, int] = {}
    for r in runs:
        if r.kind == PipeKind.PROCESS and r.line and r.line not in numbers:
            size_mm = (topo.sizes or {}).get(r.size, (topo.sizes or {}).get("main", 100))
            if r.code and r.code != topo.code:
                size_mm = (topo.sizes or {}).get("branch", size_mm)
            numbers[r.line] = namer.line_number(size_mm, r.code)
            line_size[r.line] = size_mm

    # ---- labels
    idx = fin.build_index(runs, run_pts)
    crossings = _crossings(run_pts, runs)
    for _, _, _, _, p in crossings:
        idx.add(_expand((p[0], p[1], p[0], p[1]), st.hop_radius + 1.5), ("hop", 0))
    dot_r = max(1.1, 1.5 * st.pipe_width)
    for r in runs:
        for n in (r.nodes[0], r.nodes[-1]):
            if n not in net.port_of:
                x, y = net.pos[n]
                idx.add((x - dot_r - 0.5, y - dot_r - 0.5, x + dot_r + 0.5, y + dot_r + 0.5), ("dot", 0))
    fin.inside_texts(idx)
    tag_drawn = fin.object_labels(idx)
    drawn_lines = fin.line_numbers(idx, runs, run_pts, numbers)
    _check_texts(fin)

    # ---- ground truth
    page_w, page_h = sheet.width, sheet.height
    printed: dict[str, set[str]] = {}
    for t in fin.texts:
        if t.owner and not t.owner.startswith("p:"):
            printed.setdefault(t.owner, set()).add(t.text)
    objects: list[PIDObject] = []
    for k, o in net.objs.items():
        it = o.item
        spec = o.spec
        act = it.actuator if spec.category == Category.VALVE else A.NONE
        bb = placed_bbox(spec, o.pl, act)
        ports = [Port(name=n, position=Point(x=_r(p[0]), y=_r(p[1])),
                      direction=Point(x=o.pdirs[n][0], y=o.pdirs[n][1]) if n in o.pdirs else None)
                 for n, p in o.ports.items()]
        if it.category == Category.INSTRUMENT:
            tag = it.tag
        else:
            tag = it.tag if tag_drawn.get(k) else None
        attrs: dict = {}
        # name / rating only as far as they are printed on the sheet (the label placer may drop them);
        # the blueprint values that did not make it onto the page are kept under separate keys
        for key, value in zip(("name", "rating"), it.name):
            attrs[key if value in printed.get(k, ()) else f"{key}_undrawn"] = value
        for key in ("service", "letters", "loop"):
            if it.attrs.get(key):
                attrs[key] = it.attrs[key]
        if it.inside and it.cls == S.OFF_PAGE_CONNECTOR:
            attrs["ref"] = it.inside[0]
        if o.role in ("source", "sink"):
            attrs["role"] = o.role
        objects.append(PIDObject(
            id=obj_ids[k], cls=it.cls, category=spec.category, bbox=BBox(x0=_r(bb[0]), y0=_r(bb[1]), x1=_r(bb[2]),
                                                                         y1=_r(bb[3])),
            rotation=o.pl.rotation, mirrored=o.pl.mirrored, tag=tag, actuator=act, ports=ports, attributes=attrs))

    def endpoint(n: str) -> Endpoint:
        if n in net.port_of:
            okey, pname = net.port_of[n]
            return Endpoint(object_id=obj_ids[okey], port=pname)
        return Endpoint(junction_id=jids[n])

    pipes: list[PipeRun] = []
    for i, r in enumerate(runs):
        pts = run_pts[i]
        line_no = numbers.get(r.line) if r.kind == PipeKind.PROCESS and r.line in drawn_lines else None
        medium = r.medium or (topo.medium if r.kind != PipeKind.SIGNAL else Medium.UNKNOWN)
        size = f"{line_size[r.line]}A" if line_no else None
        pipes.append(PipeRun(id=f"p{i + 1}", points=[Point(x=_r(x), y=_r(y)) for x, y in pts],
                             start=endpoint(r.nodes[0]), end=endpoint(r.nodes[-1]), kind=r.kind, line_number=line_no,
                             size=size, medium=medium, flow_direction=1 if r.flow else None))
    _connector_directions(objects, pipes)
    junctions = []
    for n in junction_nodes:
        p = net.pos[n]
        d = deg.get(n, 0)
        tap = any(r.kind == PipeKind.INSTRUMENT_TAP and n in (r.nodes[0], r.nodes[-1]) for r in runs)
        dotted = st.junction_dots and (st.dot_taps or not tap)
        junctions.append(Junction(id=jids[n], position=Point(x=_r(p[0]), y=_r(p[1])),
                                  kind="tee" if d == 3 else ("cross" if d == 4 else "tee"), dotted=dotted))

    # ---- texts (drawing + sheet)
    texts: list[TextItem] = []
    draw_texts: list[TextDraw] = []
    lay = st.layers
    for t in fin.texts:
        tid = f"t{len(texts) + 1}"
        owner = None
        if t.owner:
            owner = f"p{int(t.owner[2:]) + 1}" if t.owner.startswith("p:") else obj_ids[t.owner]
        texts.append(TextItem(id=tid, text=t.text, bbox=BBox(x0=_r(t.bbox[0]), y0=_r(t.bbox[1]), x1=_r(t.bbox[2]),
                                                             y1=_r(t.bbox[3])),
                              rotation=t.rotate, size=round(t.size, 3), role=t.role, owner_id=owner))
        draw_texts.append(TextDraw(t.text, t.x, t.y, t.size, t.font, t.rotate, layer=lay[2] if lay else None))
        if owner and owner.startswith("o"):
            next(o for o in objects if o.id == owner).text_ids.append(tid)
        elif owner:
            pipes[int(owner[1:]) - 1].text_ids.append(tid)
    for td, role in sheet.texts:
        tid = f"t{len(texts) + 1}"
        bb = text_bbox(td.text, td.x, td.y, td.size, td.font, td.rotate)
        texts.append(TextItem(id=tid, text=td.text, bbox=BBox(x0=_r(bb[0]), y0=_r(bb[1]), x1=_r(bb[2]), y1=_r(bb[3])),
                              rotation=td.rotate, size=round(td.size, 3), role=role))
        draw_texts.append(td)

    info = sheet.info
    doc = PIDDocument(
        source=f"synthetic:{seed}",
        pages=[PageInfo(index=0, width=page_w, height=page_h, drawing_area=sheet.drawing_area)],
        info=info, objects=objects, junctions=junctions, pipes=pipes, texts=texts,
        meta={"generator": "shippid.datasets.synthetic", "seed": seed, "attempt": attempt, "paper": sheet.paper,
              "symbol_scale": round(gen.s, 3), "crossings": len(crossings), "labels_dropped": fin.dropped,
              "drains_skipped": fin.skipped_drains, "bands": scene.split is not None,
              "sheet_layout": gen.sheet_layout,
              "fanout": gen.fanout_stage is not None},
    )

    # ---- drawing spec
    spec = DrawingSpec(paper=sheet.paper, width=page_w, height=page_h, seed=seed, system=topo.system,
                       layers=list(lay), style=st,
                       metadata={"title": f"{info.drawing_no} {info.title}", "creator": "shippid.datasets",
                                 "subject": "P&ID"})
    spec.paths.extend(sheet.paths)
    spec.symbols.extend(sheet.symbols)
    cuts: dict[int, dict[int, list[Pt]]] = {}
    for hi, hj, vi, vj, p in crossings:
        kinds = (runs[hi].kind, runs[vi].kind)
        if PipeKind.SIGNAL in kinds:
            continue
        yi, yj = (hi, hj) if st.hop_on == "h" else (vi, vj)
        cuts.setdefault(yi, {}).setdefault(yj, []).append(p)
    pipe_layer = lay[0] if lay else None
    n_sheet_paths = len(spec.paths)
    for i, r in enumerate(runs):
        pts = run_pts[i]
        color = (0.0, 0.0, 0.0)
        if st.color_pipes and r.kind != PipeKind.SIGNAL:
            color = medium_rgb(pipes[i].medium)
        segs = _pipe_segs(pts, cuts.get(i, {}), st)
        kw = dict(width=fin._pipe_width(r), color=color,
                  dashes=st.signal_dashes if r.kind == PipeKind.SIGNAL else None, layer=pipe_layer)
        if st.split_pipes and r.kind != PipeKind.SIGNAL:
            spec.paths.extend(PathDraw([sg], **kw) for sg in segs)
        else:
            spec.paths.append(PathDraw(segs, **kw))
    if st.junction_dots:
        for j in junctions:
            if j.dotted:
                spec.circles.append(CircleDraw(j.position.x, j.position.y, dot_r, width=0.3, layer=pipe_layer))
    if st.flow_arrows:
        spec.paths.extend(_flow_arrows(fin, runs, run_pts, idx, pipe_layer))
    sym_layer = lay[1] if lay else None
    for k, o in net.objs.items():
        act = o.item.actuator if o.spec.category == Category.VALVE else A.NONE
        spec.symbols.append(SymbolDraw(o.item.cls, o.pl, act, width=st.symbol_width, layer=sym_layer))
    spec.texts = draw_texts
    if st.interleave:  # sheet furniture first, then the drawing graphics in a seeded random order
        items = ([("path", i) for i in range(n_sheet_paths, len(spec.paths))]
                 + [("symbol", i) for i in range(len(sheet.symbols), len(spec.symbols))]
                 + [("circle", i) for i in range(len(spec.circles))])
        random.Random(f"{seed}:order").shuffle(items)
        spec.order = ([("path", i) for i in range(n_sheet_paths)] + [("symbol", i) for i in range(len(sheet.symbols))]
                      + items)
    return doc, spec


def _connector_directions(objects: list[PIDObject], pipes: list[PipeRun]) -> None:
    """Record on each off-page connector whether its line enters (``direction="in"``, "FROM ...")
    or leaves (``"out"``, "TO ...") the drawing.

    Contract limitation: :data:`symbols.OFF_PAGE_CONNECTOR` has a single port at the flat end, so
    the arrow tip always points away from the pipe -- along the flow for outgoing lines, against it
    for incoming ones. ``arrow_along_flow`` records this so downstream stages do not learn flow
    direction from the arrow shape of these drawings.
    """
    ends: dict[str, set[str]] = {}
    for p in pipes:
        if p.flow_direction is None:
            continue
        for ep, d in ((p.start, "in"), (p.end, "out")):  # flow leaves the start object, enters the end one
            if ep.object_id:
                ends.setdefault(ep.object_id, set()).add(d)
    for o in objects:
        if o.cls != S.OFF_PAGE_CONNECTOR:
            continue
        seen = ends.get(o.id, set())
        if len(seen) == 1:
            direction = next(iter(seen))
        else:  # no directed line: fall back to the layout role / label wording
            label = str(o.attributes.get("name") or o.attributes.get("name_undrawn") or "")
            direction = "in" if o.attributes.get("role") == "source" or label.startswith("FROM") else "out"
        o.attributes["direction"] = direction
        o.attributes["arrow_along_flow"] = direction == "out"


def _flow_arrows(fin: _Finisher, runs: list[_Run], run_pts: list[list[Pt]], idx: _Index,
                 layer: Optional[str]) -> list[PathDraw]:
    """Small filled flow arrows on long process runs (decoration, kept clear of texts)."""
    out = []
    s = fin.s
    for i, r in enumerate(runs):
        if r.kind != PipeKind.PROCESS or not r.flow:
            continue
        pts = run_pts[i]
        best = max(zip(pts, pts[1:]), key=lambda ab: abs(ab[0][0] - ab[1][0]) + abs(ab[0][1] - ab[1][1]))
        a, b = best
        L = abs(a[0] - b[0]) + abs(a[1] - b[1])
        if L < 4 * s:
            continue
        d = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
        sz = 0.32 * s
        for t in (0.25, 0.75, 0.4, 0.6):
            c = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            tip = (c[0] + d[0] * sz, c[1] + d[1] * sz)
            n = (-d[1], d[0])
            b0 = (c[0] - d[0] * sz + n[0] * sz * 0.45, c[1] - d[1] * sz + n[1] * sz * 0.45)
            b1 = (c[0] - d[0] * sz - n[0] * sz * 0.45, c[1] - d[1] * sz - n[1] * sz * 0.45)
            r_ = (min(tip[0], b0[0], b1[0]), min(tip[1], b0[1], b1[1]), max(tip[0], b0[0], b1[0]),
                  max(tip[1], b0[1], b1[1]))
            if idx.hits(_expand(r_, 1.0), ignore=(("pipe", i),)):
                continue
            idx.add(r_, ("arrow", i))
            out.append(PathDraw.polyline([tip, b0, b1], closed=True, width=0.3, fill=(0.0, 0.0, 0.0), layer=layer))
            break
    return out


def _r(v: float) -> float:
    return round(float(v), 4)


# --------------------------------------------------------------------------- validation


def _check_objects(net: _Net, sheet: Sheet) -> None:
    a = sheet.drawing_area
    boxes = [(k, o.bbox()) for k, o in net.objs.items()]
    idx = _Index(64.0)
    for k, b in boxes:
        if b[0] < a.x0 or b[1] < a.y0 or b[2] > a.x1 or b[3] > a.y1:
            raise LayoutError(f"object {k} outside the drawing area")
        for i in idx.hits(_expand(b, 0.5)):
            raise LayoutError(f"symbols overlap: {k} / {idx.tags[i]}")
        idx.add(b, k)


def _check_pipes(net: _Net, runs: list[_Run], run_pts: list[list[Pt]], s: float) -> None:
    idx = _Index(64.0)
    for k, o in net.objs.items():
        idx.add(_expand(o.bbox(), -0.4), k)
    H, V = [], []
    for i, pts in enumerate(run_pts):
        # a pipe may enter the outline of the symbols it connects to (e.g. the plates of a spectacle
        # flange sit inside the bbox of its ball), never any other symbol
        own = tuple(net.port_of[n][0] for n in (runs[i].nodes[0], runs[i].nodes[-1]) if n in net.port_of)
        for a, b in zip(pts, pts[1:]):
            for hit in idx.hits(_seg_rect(a, b, 0.0), ignore=own):
                raise LayoutError(f"pipe {i} passes through symbol {idx.tags[hit]}")
            if abs(a[1] - b[1]) < 1e-9:
                H.append((a[1], min(a[0], b[0]), max(a[0], b[0]), i))
            else:
                V.append((a[0], min(a[1], b[1]), max(a[1], b[1]), i))
    for lst in (H, V):
        by_c: dict[float, list] = {}
        for c, lo, hi, i in lst:
            by_c.setdefault(round(c, 3), []).append((lo, hi, i))
        cs = sorted(by_c)
        for ci, c in enumerate(cs):
            group = by_c[c]
            for cj in range(ci, len(cs)):
                c2 = cs[cj]
                if c2 - c > 0.25 * s:
                    break
                for lo1, hi1, i1 in group:
                    for lo2, hi2, i2 in by_c[c2]:
                        if i1 == i2:
                            continue
                        if lo1 < hi2 - 1e-6 and lo2 < hi1 - 1e-6:
                            raise LayoutError("parallel pipes overlap or run too close")
    # a node of one run must not lie on another run's segment interior
    ends = {}
    for i, r in enumerate(run_pts):
        for p in (r[0], r[-1]):
            ends.setdefault((round(p[0], 3), round(p[1], 3)), set()).add(i)
    for i, pts in enumerate(run_pts):
        for a, b in zip(pts, pts[1:]):
            for (x, y), owners in ends.items():
                if i in owners:
                    continue
                if abs(a[1] - b[1]) < 1e-9 and abs(y - a[1]) < 1e-6 and min(a[0], b[0]) + 1e-6 < x < max(a[0], b[0]) - 1e-6:
                    raise LayoutError("pipe passes through a junction / port")
                if abs(a[0] - b[0]) < 1e-9 and abs(x - a[0]) < 1e-6 and min(a[1], b[1]) + 1e-6 < y < max(a[1], b[1]) - 1e-6:
                    raise LayoutError("pipe passes through a junction / port")


def _check_texts(fin: _Finisher) -> None:
    idx = _Index(48.0)
    for i, t in enumerate(fin.texts):
        for j in idx.hits(_expand(t.bbox, 0.2)):
            raise LayoutError(f"texts overlap: {t.text!r} / {fin.texts[idx.tags[j]].text!r}")
        idx.add(t.bbox, i)
