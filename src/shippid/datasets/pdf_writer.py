"""Render a :class:`DrawingSpec` to a readable vector PDF with PyMuPDF.

The spec is a flat list of drawing instructions produced by
:mod:`shippid.datasets.synthetic`: paths (pipes, frame, title-block grid, hops),
symbols (drawn with :func:`shippid.symbols.draw_symbol`, i.e. flattened vector
paths like a CAD export), filled circles (junction dots) and real text in
base-14 fonts. Optional content groups (OCG layers) mimic CAD layer export.
Nothing else is written: no custom metadata keys and no embedded ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pymupdf

from ..model import ActuatorType, SymbolClass, SystemType
from ..symbols import Placement, draw_symbol, get_symbol

Color = tuple[float, float, float]
BLACK: Color = (0.0, 0.0, 0.0)


@dataclass
class TextDraw:
    text: str
    x: float  # baseline origin
    y: float
    size: float
    font: str = "helv"
    rotate: int = 0
    color: Color = BLACK
    layer: Optional[str] = None


@dataclass
class PathDraw:
    """Line / cubic Bezier segments; consecutive segments sharing end points form one sub-path.

    ``segs`` items: ``("l", x0, y0, x1, y1)`` or ``("c", x0, y0, x1, y1, x2, y2, x3, y3)``.
    """

    segs: list[tuple]
    width: float = 0.8
    color: Color = BLACK
    fill: Optional[Color] = None
    dashes: Optional[str] = None
    closed: bool = False
    layer: Optional[str] = None

    @classmethod
    def polyline(cls, pts: list[tuple[float, float]], closed: bool = False, **kw: Any) -> "PathDraw":
        ring = list(pts) + ([pts[0]] if closed else [])
        segs = [("l", a[0], a[1], b[0], b[1]) for a, b in zip(ring, ring[1:])]
        return cls(segs, closed=closed, **kw)

    @classmethod
    def rect(cls, x0: float, y0: float, x1: float, y1: float, **kw: Any) -> "PathDraw":
        return cls.polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True, **kw)


@dataclass
class CircleDraw:
    x: float
    y: float
    r: float
    width: float = 0.5
    color: Color = BLACK
    fill: Optional[Color] = BLACK
    layer: Optional[str] = None


@dataclass
class SymbolDraw:
    cls: SymbolClass
    placement: Placement
    actuator: ActuatorType = ActuatorType.NONE
    width: float = 0.6
    color: Color = BLACK
    layer: Optional[str] = None


@dataclass
class DrawingSpec:
    paper: str
    width: float
    height: float
    seed: int
    system: SystemType
    layers: list[str] = field(default_factory=list)  # OCG names, empty for none
    paths: list[PathDraw] = field(default_factory=list)
    symbols: list[SymbolDraw] = field(default_factory=list)
    circles: list[CircleDraw] = field(default_factory=list)
    texts: list[TextDraw] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)  # standard PDF info keys only
    style: Any = None  # synthetic.DrawingStyle (kept for previews / statistics)


_INFO_KEYS = ("title", "author", "subject", "keywords", "creator", "producer")


def render(spec: DrawingSpec) -> pymupdf.Document:
    """Build the PyMuPDF document for ``spec`` (one page)."""
    doc = pymupdf.open()
    page = doc.new_page(width=spec.width, height=spec.height)
    ocg = {name: doc.add_ocg(name, on=True) for name in spec.layers}

    def oc(layer: Optional[str]) -> int:
        return ocg.get(layer, 0) if layer else 0

    shape = page.new_shape()
    for p in spec.paths:
        for s in p.segs:
            if s[0] == "l":
                shape.draw_line((s[1], s[2]), (s[3], s[4]))
            else:
                shape.draw_bezier((s[1], s[2]), (s[3], s[4]), (s[5], s[6]), (s[7], s[8]))
        shape.finish(width=p.width, color=p.color, fill=p.fill, dashes=p.dashes, closePath=p.closed,
                     oc=oc(p.layer))
    for sd in spec.symbols:
        draw_symbol(shape, get_symbol(sd.cls), sd.placement, sd.actuator, width=sd.width, color=sd.color,
                    oc=oc(sd.layer))
    for c in spec.circles:
        shape.draw_circle((c.x, c.y), c.r)
        shape.finish(width=c.width, color=c.color, fill=c.fill, oc=oc(c.layer))
    for t in spec.texts:
        shape.insert_text((t.x, t.y), t.text, fontsize=t.size, fontname=t.font, rotate=t.rotate, color=t.color,
                          oc=oc(t.layer))
    shape.commit()
    meta = {k: "" for k in _INFO_KEYS}
    meta.update({k: v for k, v in spec.metadata.items() if k in _INFO_KEYS})
    meta.update({"creationDate": "", "modDate": ""})
    doc.set_metadata(meta)
    return doc


def to_bytes(spec: DrawingSpec) -> bytes:
    """Deterministic PDF bytes (no random /ID, no timestamps)."""
    doc = render(spec)
    try:
        return doc.tobytes(garbage=3, deflate=True, no_new_id=True)
    finally:
        doc.close()


def write_pdf(spec: DrawingSpec, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_bytes(spec))
    return path


def render_png(pdf: str | Path | bytes, out: str | Path, dpi: int = 60) -> Path:
    """Rasterise page 0 of a PDF (path or bytes) to PNG."""
    doc = pymupdf.open("pdf", pdf) if isinstance(pdf, (bytes, bytearray)) else pymupdf.open(str(pdf))
    try:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        doc[0].get_pixmap(dpi=dpi).save(str(out))
        return out
    finally:
        doc.close()
