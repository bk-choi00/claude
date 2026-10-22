"""Symbol library: canonical geometry, ports and labels for every :class:`SymbolClass`.

Each symbol is defined in a *local* frame measured in symbol units: the local
x-axis is the main flow axis, the local y-axis points "down" (same handedness as
PDF page space), and the symbol is centred on the origin. A placement maps the
local frame to the page::

    page = center + scale * R(rotation) @ S(stretch) @ M(mirrored) @ (u, v)

* ``scale``      points per symbol unit (typical 14-24 pt on an A1/A2 sheet)
* ``rotation``   0/90/180/270; 90 turns local +x to page +y (visually clockwise)
* ``mirrored``   flips the local y axis (actuator drawn below the pipe, etc.)
* ``stretch``    (sx, sy) for box-like equipment drawn at arbitrary size

The library is used by the synthetic PDF writer (to draw), by the vector
extractor (to match templates and locate ports) and by the 3D builder (for
nozzle names). Symbol styles differ between shipyards; :func:`register_symbol`
lets a project override or extend the library (e.g. from a legend sheet).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

from .model import ActuatorType, Category, SymbolClass

# --------------------------------------------------------------------------- primitives


@dataclass(frozen=True)
class Line:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class Poly:
    pts: tuple[tuple[float, float], ...]
    closed: bool = True
    fill: bool = False


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float
    fill: bool = False


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float
    fill: bool = False


Primitive = Union[Line, Poly, Circle, Rect]


@dataclass(frozen=True)
class PortSpec:
    name: str
    x: float
    y: float
    dx: float
    dy: float


@dataclass(frozen=True)
class SymbolSpec:
    cls: SymbolClass
    category: Category
    label_en: str
    label_ko: str
    prims: tuple[Primitive, ...]
    ports: tuple[PortSpec, ...]
    boundary_ports: bool = False  # pipes may attach anywhere on the outline (tanks, engines)
    stretchable: bool = False  # drawn at arbitrary size via ``stretch``
    holds_text: bool = False  # text is written inside (instrument bubbles, connectors)
    rotatable: bool = True  # instrument bubbles are never rotated
    description: str = ""  # short visual description, used in Gemma prompts

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Local bounding box (u0, v0, u1, v1) over all primitives."""
        pts = list(_prim_extent_points(self.prims))
        us = [p[0] for p in pts]
        vs = [p[1] for p in pts]
        return (min(us), min(vs), max(us), max(vs))


def _prim_extent_points(prims: Iterable[Primitive]) -> Iterable[tuple[float, float]]:
    for p in prims:
        if isinstance(p, Line):
            yield (p.x0, p.y0)
            yield (p.x1, p.y1)
        elif isinstance(p, Poly):
            yield from p.pts
        elif isinstance(p, Circle):
            yield (p.cx - p.r, p.cy - p.r)
            yield (p.cx + p.r, p.cy + p.r)
        elif isinstance(p, Rect):
            yield (p.x0, p.y0)
            yield (p.x1, p.y1)


# --------------------------------------------------------------------------- helpers


def _tri_left(h: float = 0.3) -> Poly:
    return Poly(((-0.5, -h), (-0.5, h), (0.0, 0.0)))


def _tri_right(h: float = 0.3, fill: bool = False) -> Poly:
    return Poly(((0.5, -h), (0.5, h), (0.0, 0.0)), fill=fill)


def _inline_ports(half: float) -> tuple[PortSpec, ...]:
    return (PortSpec("in", -half, 0.0, -1.0, 0.0), PortSpec("out", half, 0.0, 1.0, 0.0))


def _zigzag(x0: float, x1: float, amp: float, n: int, y: float = 0.0) -> Poly:
    pts = [(x0, y)]
    for i in range(1, n):
        x = x0 + (x1 - x0) * i / n
        pts.append((x, y + (amp if i % 2 else -amp)))
    pts.append((x1, y))
    return Poly(tuple(pts), closed=False)


def _wave(x0: float, x1: float, amp: float, y: float, n: int = 24, periods: float = 2.0) -> Poly:
    pts = []
    for i in range(n + 1):
        t = i / n
        pts.append((x0 + (x1 - x0) * t, y + amp * math.sin(2 * math.pi * periods * t)))
    return Poly(tuple(pts), closed=False)


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 12) -> list[tuple[float, float]]:
    return [
        (cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)), cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
        for i in range(n + 1)
    ]


def _capsule(w: float, h: float) -> Poly:
    r = w / 2
    top = _arc(0.0, -h / 2 + r, r, 180, 360)
    bottom = _arc(0.0, h / 2 - r, r, 0, 180)
    return Poly(tuple(top + bottom), closed=True)


BOWTIE = (_tri_left(), _tri_right())

# --------------------------------------------------------------------------- library

_LIB: dict[SymbolClass, SymbolSpec] = {}


def register_symbol(spec: SymbolSpec) -> None:
    """Add or replace a symbol definition (project-specific symbol styles)."""
    _LIB[spec.cls] = spec


def _reg(cls, cat, en, ko, prims, ports, desc, **kw) -> None:
    register_symbol(SymbolSpec(cls, cat, en, ko, tuple(prims), tuple(ports), description=desc, **kw))


E, V, I, F, C = Category.EQUIPMENT, Category.VALVE, Category.INSTRUMENT, Category.FITTING, Category.CONNECTOR
S = SymbolClass

# ---- equipment
_reg(S.TANK, E, "Tank", "탱크", [Rect(-1.5, -1.0, 1.5, 1.0)], [], "plain rectangle (large)",
     boundary_ports=True, stretchable=True)
_reg(S.PUMP_CENTRIFUGAL, E, "Centrifugal pump", "원심 펌프",
     [Circle(0, 0, 0.5), Poly(((-0.32, 0.32), (0.32, 0.32), (0.0, -0.5)), closed=True)],
     [PortSpec("suction", -0.5, 0.0, -1, 0), PortSpec("discharge", 0.0, -0.5, 0, -1)],
     "circle with an inscribed triangle whose apex touches the top (discharge)")
_reg(S.PUMP_SCREW, E, "Screw / gear pump", "스크류(기어) 펌프",
     [Circle(0, 0, 0.5), Circle(-0.18, 0, 0.13), Circle(0.18, 0, 0.13)],
     _inline_ports(0.5), "circle containing two small circles side by side")
_reg(S.HEAT_EXCHANGER, E, "Shell & tube heat exchanger", "쉘앤튜브 열교환기",
     [Rect(-1.0, -0.4, 1.0, 0.4), _zigzag(-0.8, 0.8, 0.22, 8)],
     [PortSpec("tube_in", -1.0, 0, -1, 0), PortSpec("tube_out", 1.0, 0, 1, 0),
      PortSpec("shell_in", -0.6, -0.4, 0, -1), PortSpec("shell_out", 0.6, 0.4, 0, 1)],
     "horizontal rectangle with a zigzag line through it")
_reg(S.PLATE_HEAT_EXCHANGER, E, "Plate heat exchanger (central cooler)", "판형 열교환기",
     [Rect(-0.4, -0.8, 0.4, 0.8)] + [Line(x, -0.7, x, 0.7) for x in (-0.24, -0.08, 0.08, 0.24)],
     [PortSpec("a_in", -0.4, -0.6, -1, 0), PortSpec("a_out", -0.4, 0.6, -1, 0),
      PortSpec("b_in", 0.4, 0.6, 1, 0), PortSpec("b_out", 0.4, -0.6, 1, 0)],
     "tall rectangle filled with four vertical plate lines")
_reg(S.HEATER, E, "Heater", "히터",
     [Rect(-0.7, -0.35, 0.7, 0.35), _wave(-0.55, 0.55, 0.18, 0.0, n=20, periods=3)],
     _inline_ports(0.7), "rectangle with a sine-wave coil inside")
_reg(S.FILTER, E, "Filter", "필터",
     [Poly(((-0.4, 0), (0, -0.4), (0.4, 0), (0, 0.4))), Line(0, -0.4, 0, 0.4)],
     _inline_ports(0.4), "diamond with a vertical line")
_reg(S.DUPLEX_FILTER, E, "Duplex / auto-backflush filter", "복식(자동역세) 필터",
     [Poly(((-0.4, 0), (0, -0.4), (0.4, 0), (0, 0.4))), Line(-0.12, -0.28, -0.12, 0.28), Line(0.12, -0.28, 0.12, 0.28)],
     _inline_ports(0.4), "diamond with two vertical lines")
_reg(S.STRAINER, E, "Strainer / mud box", "스트레이너(머드박스)",
     [Rect(-0.35, -0.3, 0.35, 0.3), Line(-0.35, 0.3, 0.35, -0.3)],
     _inline_ports(0.35), "small square with one diagonal")
_reg(S.SEA_CHEST, E, "Sea chest", "씨체스트",
     [Rect(-0.8, -0.4, 0.8, 0.4), _wave(-0.65, 0.65, 0.08, 0.15, n=24, periods=3)],
     [PortSpec("out", 0.0, -0.4, 0, -1)], "wide rectangle with a wavy water line inside", stretchable=True)
_reg(S.OVERBOARD, E, "Overboard discharge", "선외 배출구",
     [Line(-0.3, 0, 0, 0), Line(0, -0.4, 0, 0.4), Line(0, -0.3, 0.18, -0.4), Line(0, 0, 0.18, -0.1),
      Line(0, 0.3, 0.18, 0.2)],
     [PortSpec("in", -0.3, 0, -1, 0)], "vertical ship-side line with three short hatch strokes")
_reg(S.PURIFIER, E, "Centrifugal purifier", "청정기",
     [Circle(0, 0, 0.5), Poly(((-0.3, -0.25), (0.3, -0.25), (0, 0.35)))],
     _inline_ports(0.5), "circle with an inverted triangle (bowl) inside")
_reg(S.COMPRESSOR, E, "Air compressor", "공기 압축기",
     [Circle(0, 0, 0.5), Line(-0.35, -0.35, 0.45, -0.15), Line(-0.35, 0.35, 0.45, 0.15)],
     _inline_ports(0.5), "circle with two lines converging to the right")
_reg(S.AIR_RECEIVER, E, "Air receiver", "공기 저장조",
     [_capsule(0.8, 1.6)], [], "vertical capsule (vessel with rounded ends)",
     boundary_ports=True, stretchable=True)
_reg(S.OILY_WATER_SEPARATOR, E, "Oily water separator", "유수분리기",
     [Rect(-0.8, -0.6, 0.8, 0.6), Line(-0.8, -0.2, 0.8, -0.2), Line(-0.8, 0.2, 0.8, 0.2)],
     [], "rectangle split by two horizontal lines", boundary_ports=True, stretchable=True)
_reg(S.BWTS_UNIT, E, "Ballast water treatment unit (UV)", "평형수 처리장치",
     [Rect(-0.9, -0.4, 0.9, 0.4), Line(-0.4, -0.25, -0.4, 0.25), Line(0, -0.25, 0, 0.25), Line(0.4, -0.25, 0.4, 0.25)],
     _inline_ports(0.9), "rectangle with three short vertical lamps inside", stretchable=True)
_reg(S.ENGINE, E, "Engine / consumer", "엔진(소비처)",
     [Rect(-1.5, -1.0, 1.5, 1.0), Line(-1.5, -1.0, 1.5, 1.0), Line(-1.5, 1.0, 1.5, -1.0)],
     [], "large rectangle with both diagonals", boundary_ports=True, stretchable=True)
_reg(S.FLOW_METER, E, "Flow meter", "유량계",
     [Rect(-0.35, -0.25, 0.35, 0.25), Circle(0, 0, 0.17)],
     _inline_ports(0.35), "small rectangle with a circle inside")

# ---- valves
_reg(S.GATE_VALVE, V, "Gate valve", "게이트 밸브", list(BOWTIE), _inline_ports(0.5),
     "bow-tie: two open triangles meeting at their tips")
_reg(S.GLOBE_VALVE, V, "Globe valve", "글로브 밸브", list(BOWTIE) + [Circle(0, 0, 0.1, fill=True)],
     _inline_ports(0.5), "bow-tie with a filled dot at the centre")
_reg(S.BALL_VALVE, V, "Ball valve", "볼 밸브", list(BOWTIE) + [Circle(0, 0, 0.16)],
     _inline_ports(0.5), "bow-tie with an open circle at the centre")
_reg(S.BUTTERFLY_VALVE, V, "Butterfly valve", "버터플라이 밸브",
     [Line(-0.25, -0.35, -0.25, 0.35), Line(0.25, -0.35, 0.25, 0.35), Line(-0.17, 0.3, 0.17, -0.3),
      Circle(0, 0, 0.06, fill=True)],
     _inline_ports(0.25), "two parallel bars with a diagonal disc and a centre dot")
_reg(S.CHECK_VALVE, V, "Check (non-return) valve", "체크 밸브", [_tri_left(), _tri_right(fill=True)],
     _inline_ports(0.5), "bow-tie whose outlet-side triangle is filled (flow towards the filled side)")
_reg(S.SDNR_VALVE, V, "Screw-down non-return valve", "스크류다운 체크 밸브",
     [_tri_left(), _tri_right(fill=True), Circle(0, 0, 0.1, fill=True)],
     _inline_ports(0.5), "bow-tie with filled outlet triangle and a filled centre dot")
_reg(S.THREE_WAY_VALVE, V, "Three-way valve", "3방향 밸브",
     list(BOWTIE) + [Poly(((0, 0), (-0.3, 0.5), (0.3, 0.5)))],
     _inline_ports(0.5) + (PortSpec("branch", 0.0, 0.5, 0, 1),), "three open triangles meeting at a point")
_reg(S.SAFETY_VALVE, V, "Safety / relief valve", "안전(릴리프) 밸브",
     [Poly(((-0.3, 0.5), (0.3, 0.5), (0, 0))), _tri_right(),
      Poly(((0, 0), (0, -0.15), (-0.15, -0.22), (0.15, -0.32), (-0.15, -0.42), (0.15, -0.52), (0, -0.6)), closed=False)],
     [PortSpec("in", 0.0, 0.5, 0, 1), PortSpec("out", 0.5, 0.0, 1, 0)],
     "angle valve (triangle from below and triangle to the right) with a spring zigzag on top")

# ---- instruments (never rotated; text inside)
_reg(S.INSTRUMENT_LOCAL, I, "Local instrument", "현장 계기", [Circle(0, 0, 0.4)], [],
     "plain circle with the instrument letters and number inside", boundary_ports=True, holds_text=True,
     rotatable=False)
_reg(S.INSTRUMENT_PANEL, I, "Panel / remote instrument", "원격(패널) 계기",
     [Circle(0, 0, 0.4), Line(-0.4, 0, 0.4, 0)], [], "circle with a horizontal line through the middle",
     boundary_ports=True, holds_text=True, rotatable=False)
_reg(S.INSTRUMENT_SHARED, I, "Shared display (DCS/AMS)", "공유 표시(AMS)",
     [Rect(-0.4, -0.4, 0.4, 0.4), Circle(0, 0, 0.4)], [], "circle inscribed in a square",
     boundary_ports=True, holds_text=True, rotatable=False)

# ---- fittings
_reg(S.REDUCER, F, "Reducer", "레듀서", [Poly(((-0.2, -0.25), (0.2, -0.13), (0.2, 0.13), (-0.2, 0.25)))],
     _inline_ports(0.2), "small trapezoid")
_reg(S.SPECTACLE_FLANGE, F, "Spectacle flange", "스펙터클 플랜지",
     [Line(-0.05, -0.3, -0.05, 0.3), Line(0.05, -0.3, 0.05, 0.3), Line(0, -0.3, 0, -0.33),
      Circle(0, -0.45, 0.12, fill=True)],
     _inline_ports(0.05), "two close parallel bars with a filled ball above")
_reg(S.FLEXIBLE_HOSE, F, "Flexible hose / expansion joint", "플렉시블 호스",
     [Line(-0.4, -0.2, -0.4, 0.2), Line(0.4, -0.2, 0.4, 0.2), _wave(-0.4, 0.4, 0.1, 0.0, n=16, periods=2)],
     _inline_ports(0.4), "wavy line between two short bars")

# ---- connectors
_reg(S.OFF_PAGE_CONNECTOR, C, "Off-page connector", "도면 연결 표시",
     [Poly(((-0.8, -0.25), (0.5, -0.25), (0.8, 0.0), (0.5, 0.25), (-0.8, 0.25)))],
     [PortSpec("in", -0.8, 0.0, -1, 0)], "arrow-shaped pentagon with the destination written inside",
     holds_text=True)

del E, V, I, F, C, S

# --------------------------------------------------------------------------- actuators
# Drawn on the local -y side of a valve (above the pipe unless mirrored).

ACTUATOR_PRIMS: dict[ActuatorType, tuple[Primitive, ...]] = {
    ActuatorType.NONE: (),
    ActuatorType.HANDWHEEL: (Line(0, 0, 0, -0.45), Line(-0.18, -0.45, 0.18, -0.45)),
    ActuatorType.DIAPHRAGM: (
        Line(0, 0, 0, -0.45),
        Poly(tuple(_arc(0, -0.45, 0.24, 180, 360)), closed=True),
    ),
    ActuatorType.HYDRAULIC: (Line(0, 0, 0, -0.45), Rect(-0.15, -0.75, 0.15, -0.45)),
    ActuatorType.ELECTRIC: (Line(0, 0, 0, -0.45), Circle(0, -0.6, 0.15)),
    ActuatorType.QUICK_CLOSING: (Line(0, 0, 0, -0.45), Poly(((-0.16, -0.72), (0.16, -0.72), (0, -0.45)), fill=True)),
}

ACTUATOR_DESCRIPTIONS: dict[ActuatorType, str] = {
    ActuatorType.NONE: "no stem drawn",
    ActuatorType.HANDWHEEL: "stem with a T bar (manual handwheel)",
    ActuatorType.DIAPHRAGM: "stem with a dome (pneumatic diaphragm, control valve)",
    ActuatorType.HYDRAULIC: "stem with a small open square (hydraulic remote operated)",
    ActuatorType.ELECTRIC: "stem with a small open circle (electric motor operated)",
    ActuatorType.QUICK_CLOSING: "stem with a filled downward triangle (quick-closing, remote trip)",
}

# --------------------------------------------------------------------------- api


def get_symbol(cls: SymbolClass) -> SymbolSpec:
    return _LIB[cls]


def all_symbols() -> list[SymbolSpec]:
    return list(_LIB.values())


def category_of(cls: SymbolClass) -> Category:
    if cls in _LIB:
        return _LIB[cls].category
    return Category.EQUIPMENT


@dataclass(frozen=True)
class Placement:
    """Where/how a symbol is drawn on the page."""

    cx: float
    cy: float
    scale: float = 18.0
    rotation: int = 0
    mirrored: bool = False
    stretch: tuple[float, float] = (1.0, 1.0)

    def to_page(self, u: float, v: float) -> tuple[float, float]:
        sx, sy = self.stretch
        u, v = u * sx, (-v if self.mirrored else v) * sy
        r = self.rotation % 360
        if r == 0:
            x, y = u, v
        elif r == 90:
            x, y = -v, u
        elif r == 180:
            x, y = -u, -v
        elif r == 270:
            x, y = v, -u
        else:
            a = math.radians(r)
            x, y = u * math.cos(a) - v * math.sin(a), u * math.sin(a) + v * math.cos(a)
        return (self.cx + self.scale * x, self.cy + self.scale * y)

    def dir_to_page(self, du: float, dv: float) -> tuple[float, float]:
        x0, y0 = self.to_page(0, 0)
        x1, y1 = self.to_page(du, dv)
        n = math.hypot(x1 - x0, y1 - y0) or 1.0
        return ((x1 - x0) / n, (y1 - y0) / n)


def placed_prims(spec: SymbolSpec, pl: Placement, actuator: ActuatorType = ActuatorType.NONE) -> list[Primitive]:
    """Primitives of a symbol (plus actuator decoration) transformed to page space.

    Circles stay circles only for uniform stretch; otherwise they are polygonised.
    """
    out: list[Primitive] = []
    prims: Sequence[Primitive] = spec.prims + (ACTUATOR_PRIMS.get(actuator, ()) if spec.category == Category.VALVE else ())
    uniform = abs(pl.stretch[0] - pl.stretch[1]) < 1e-9
    for p in prims:
        if isinstance(p, Line):
            a, b = pl.to_page(p.x0, p.y0), pl.to_page(p.x1, p.y1)
            out.append(Line(a[0], a[1], b[0], b[1]))
        elif isinstance(p, Poly):
            out.append(Poly(tuple(pl.to_page(u, v) for u, v in p.pts), closed=p.closed, fill=p.fill))
        elif isinstance(p, Circle):
            if uniform:
                c = pl.to_page(p.cx, p.cy)
                out.append(Circle(c[0], c[1], p.r * pl.scale * pl.stretch[0], fill=p.fill))
            else:
                pts = _arc(p.cx, p.cy, p.r, 0, 360, n=24)[:-1]
                out.append(Poly(tuple(pl.to_page(u, v) for u, v in pts), closed=True, fill=p.fill))
        elif isinstance(p, Rect):
            corners = [(p.x0, p.y0), (p.x1, p.y0), (p.x1, p.y1), (p.x0, p.y1)]
            out.append(Poly(tuple(pl.to_page(u, v) for u, v in corners), closed=True, fill=p.fill))
    return out


def placed_ports(spec: SymbolSpec, pl: Placement) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    """(name, page position, outward page direction) for each fixed port."""
    return [(ps.name, pl.to_page(ps.x, ps.y), pl.dir_to_page(ps.dx, ps.dy)) for ps in spec.ports]


def placed_bbox(spec: SymbolSpec, pl: Placement, actuator: ActuatorType = ActuatorType.NONE) -> tuple[float, float, float, float]:
    pts = list(_prim_extent_points(placed_prims(spec, pl, actuator)))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def sample_outline(prims: Sequence[Primitive], step: float) -> list[tuple[float, float]]:
    """Points sampled every ``step`` along all primitive outlines (for template matching)."""
    pts: list[tuple[float, float]] = []

    def seg(a, b):
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        for i in range(n + 1):
            t = i / n
            pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))

    for p in prims:
        if isinstance(p, Line):
            seg((p.x0, p.y0), (p.x1, p.y1))
        elif isinstance(p, Poly):
            ring = list(p.pts) + ([p.pts[0]] if p.closed else [])
            for a, b in zip(ring, ring[1:]):
                seg(a, b)
        elif isinstance(p, Circle):
            n = max(8, int(2 * math.pi * p.r / step))
            pts.extend((p.cx + p.r * math.cos(2 * math.pi * i / n), p.cy + p.r * math.sin(2 * math.pi * i / n)) for i in range(n))
        elif isinstance(p, Rect):
            ring = [(p.x0, p.y0), (p.x1, p.y0), (p.x1, p.y1), (p.x0, p.y1), (p.x0, p.y0)]
            for a, b in zip(ring, ring[1:]):
                seg(a, b)
    return pts


def draw_symbol(
    shape,  # pymupdf.Shape
    spec: SymbolSpec,
    pl: Placement,
    actuator: ActuatorType = ActuatorType.NONE,
    width: float = 0.6,
    color: tuple[float, float, float] = (0, 0, 0),
    oc: int = 0,
) -> None:
    """Draw a symbol into a PyMuPDF ``Shape`` as plain vector paths (no metadata),
    the way CAD tools flatten blocks when exporting P&IDs to PDF."""
    import pymupdf

    for p in placed_prims(spec, pl, actuator):
        fill = None
        close = False
        if isinstance(p, Line):
            shape.draw_line(pymupdf.Point(p.x0, p.y0), pymupdf.Point(p.x1, p.y1))
        elif isinstance(p, Poly):
            shape.draw_polyline([pymupdf.Point(x, y) for x, y in p.pts])
            close = p.closed
            fill = color if p.fill else None
        elif isinstance(p, Circle):
            shape.draw_circle(pymupdf.Point(p.cx, p.cy), p.r)
            fill = color if p.fill else None
        elif isinstance(p, Rect):  # pragma: no cover - rects are polygonised by placed_prims
            shape.draw_rect(pymupdf.Rect(p.x0, p.y0, p.x1, p.y1))
        shape.finish(width=width, color=color, fill=fill, closePath=close, oc=oc)


def legend_text() -> str:
    """Compact textual legend of the library, used to ground Gemma prompts."""
    lines = []
    for spec in all_symbols():
        ports = ", ".join(p.name for p in spec.ports) or ("anywhere on outline" if spec.boundary_ports else "-")
        lines.append(f"- {spec.cls.value} ({spec.category.value}; {spec.label_en}): {spec.description}; ports: {ports}")
    lines.append("Valve actuators are drawn on a stem perpendicular to the pipe:")
    for act, d in ACTUATOR_DESCRIPTIONS.items():
        if act != ActuatorType.NONE:
            lines.append(f"- actuator {act.value}: {d}")
    return "\n".join(lines)


@dataclass
class SymbolInstance:
    """Convenience bundle used by the synthetic generator."""

    cls: SymbolClass
    placement: Placement
    actuator: ActuatorType = ActuatorType.NONE
    tag: Optional[str] = None
    extra: dict = field(default_factory=dict)
