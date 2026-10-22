"""Legend learning (:mod:`shippid.extract.legend`): description mapping, legend regions and rows,
templates learned from glyphs and their use by :func:`shippid.extract.pipeline.extract_pdf`.

The sheets are drawn here directly with PyMuPDF (independent of the dataset generator); the
glyphs of the hand-built legends are deliberately *not* in the built-in symbol library.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import pymupdf
import pytest

from shippid.evaluate import match_boxes
from shippid.extract.legend import (
    LEGEND_TIE,
    LegendBook,
    LegendSpec,
    find_legends,
    glyph_template,
    map_description,
    normalize_description,
)
from shippid.extract.pipeline import ExtractOptions, extract_pdf
from shippid.extract.symbols_match import TemplateLibrary, get_library
from shippid.model import ActuatorType, BBox, Category, PIDDocument, SymbolClass, TextRole
from shippid.symbols import (
    Circle,
    Line,
    Placement,
    Poly,
    SymbolSpec,
    all_symbols,
    draw_symbol,
    get_symbol,
    placed_bbox,
)

S = SymbolClass
A = ActuatorType
BLACK = (0.0, 0.0, 0.0)

# ----------------------------------------------------------------------------- glyphs not in the library

_C45 = math.cos(math.radians(45))
REMOTE_BFV = (Circle(0.0, 0.0, 0.5), Line(-0.5 * _C45, -0.5 * _C45, 0.5 * _C45, 0.5 * _C45),
              Line(-0.5 * _C45, 0.5 * _C45, 0.5 * _C45, -0.5 * _C45))  # circle with an inscribed X
_TRI_L = Poly(((-0.5, -0.3), (-0.5, 0.3), (0.0, 0.0)))
_TRI_R = Poly(((0.5, -0.3), (0.5, 0.3), (0.0, 0.0)))
YARD_CHECK = (_TRI_L, _TRI_R, Line(0.42, -0.25, 0.42, 0.25))  # open bow-tie + outlet bar
YARD_GATE = (_TRI_L, _TRI_R, Line(0.0, -0.3, 0.0, 0.3))  # bow-tie + middle bar
_CY = math.sqrt(0.05)
YARD_PUMP = (Circle(-0.3, _CY, 0.3), Line(0.0, _CY, 0.0, -0.5))  # circle + tangential discharge stub
DOUBLE_BUBBLE = (Circle(0.0, 0.0, 0.4), Circle(0.0, 0.0, 0.33))
WIDGET = (Poly(tuple((0.45 * math.cos(math.radians(60 * k)), 0.45 * math.sin(math.radians(60 * k))) for k in range(6))),
          Line(0.0, -0.39, 0.0, 0.39))  # a hexagon with a bar


def _spec(cls: SymbolClass, prims: Sequence, ports=None) -> SymbolSpec:
    base = get_symbol(cls) if cls != S.UNKNOWN else None
    return SymbolSpec(cls, base.category if base else Category.EQUIPMENT, "x", "x", tuple(prims),
                      tuple(base.ports) if (base and ports is None) else tuple(ports or ()),
                      rotatable=base.rotatable if base else True)


class Sheet:
    """A small A3 drawing with exact symbol boxes (independent of the dataset generator)."""

    def __init__(self, pdf: Optional[pymupdf.Document] = None, frame: bool = True):
        self.doc = pdf if pdf is not None else pymupdf.open()
        self.page = self.doc.new_page(width=1190, height=842)
        self.sh = self.page.new_shape()
        self.texts: list[tuple[float, float, str, float, str]] = []
        self.objects: list[tuple[SymbolClass, ActuatorType, tuple[float, float, float, float]]] = []
        if frame:
            self.sh.draw_rect(pymupdf.Rect(20, 20, 1170, 822))
            self.sh.finish(width=1.2, color=BLACK)

    def glyph(self, cls: SymbolClass, prims: Sequence, pl: Placement, act: ActuatorType = A.NONE,
              record: bool = True, width: float = 0.6) -> tuple[float, float, float, float]:
        spec = _spec(cls if cls != S.UNKNOWN else S.TANK, prims) if prims else get_symbol(cls)
        if cls == S.UNKNOWN:
            spec = SymbolSpec(S.UNKNOWN, Category.EQUIPMENT, "x", "x", tuple(prims), ())
        draw_symbol(self.sh, spec, pl, act, width=width)
        bb = placed_bbox(spec, pl, act)
        if record:
            self.objects.append((cls, act, bb))
        return bb

    def line(self, pts, width: float = 1.0) -> None:
        self.sh.draw_polyline([pymupdf.Point(*p) for p in pts])
        self.sh.finish(width=width, color=BLACK, closePath=False)

    def text(self, x: float, y: float, s: str, size: float = 7.0, font: str = "helv") -> None:
        self.texts.append((x, y, s, size, font))

    def legend(self, x0: float, y0: float, heading: str, rows: list[tuple[SymbolClass, Sequence, ActuatorType, str]],
               font: str = "helv", sc: float = 10.0, pitch: float = 22.0, two_columns: bool = False) -> tuple:
        n = len(rows)
        per_col = -(-n // 2) if two_columns else n
        w = 520.0 if two_columns else 280.0
        h = 36 + pitch * per_col
        self.sh.draw_rect(pymupdf.Rect(x0, y0, x0 + w, y0 + h))
        self.sh.finish(width=0.8, color=BLACK)
        self.text(x0 + 6, y0 + 14, heading, 9.0, "hebo" if font == "helv" else font)
        for i, (cls, prims, act, desc) in enumerate(rows):
            c, r = divmod(i, per_col)
            cx = x0 + 30 + c * 260
            cy = y0 + 36 + pitch * r
            self.glyph(cls, prims, Placement(cx, cy, scale=sc), act, record=False, width=0.5)
            self.text(cx + 30, cy + 2.5, desc, 6.5, font)
        return (x0, y0, x0 + w, y0 + h)

    def save(self, path) -> str:
        self.sh.commit()
        for x, y, s, size, font in self.texts:
            self.page.insert_text((x, y), s, fontsize=size, fontname=font, color=BLACK)
        self.doc.save(str(path))
        return str(path)


def _inline(sheet: Sheet, cls: SymbolClass, prims: Sequence, x: float, y: float, act: ActuatorType = A.NONE,
            scale: float = 18.0, half: float = 0.5) -> tuple[float, float]:
    """An inline symbol on a horizontal pipe at (x, y); returns its port x positions."""
    sheet.glyph(cls, prims, Placement(x, y, scale=scale), act)
    return (x - half * scale, x + half * scale)


def _objects_by_box(doc: PIDDocument, sheet: Sheet):
    """(gt cls, gt actuator, predicted object or None) per drawn object (IoU >= 0.5)."""
    gt = [BBox(x0=b[0], y0=b[1], x1=b[2], y1=b[3]) for _, _, b in sheet.objects]
    m = {j: i for i, j, _ in match_boxes([o.bbox for o in doc.objects], gt, 0.5)}
    return [(c, a, doc.objects[m[j]] if j in m else None) for j, (c, a, _) in enumerate(sheet.objects)]


# ----------------------------------------------------------------------------- description mapping


def test_library_labels_map_to_their_class():
    for spec in all_symbols():
        assert map_description(spec.label_en.upper()).cls == spec.cls, spec.label_en
        assert map_description(spec.label_en).cls == spec.cls, spec.label_en
        assert map_description(spec.label_ko).cls == spec.cls, spec.label_ko


@pytest.mark.parametrize("text, cls, act", [
    ("CHECK VALVE", S.CHECK_VALVE, A.NONE),
    ("NON-RETURN VALVE", S.CHECK_VALVE, A.NONE),
    ("NRV", S.CHECK_VALVE, A.NONE),
    ("S.D.N.R. VALVE", S.SDNR_VALVE, A.NONE),
    ("SCREW DOWN NON RETURN VALVE", S.SDNR_VALVE, A.NONE),
    ("SCREW-DOWN NON-RETURN VALVE", S.SDNR_VALVE, A.NONE),
    ("GATE VALVE", S.GATE_VALVE, A.NONE),
    ("GLOBE VALVE", S.GLOBE_VALVE, A.NONE),
    ("BALL VALVE", S.BALL_VALVE, A.NONE),
    ("BUTTERFLY VALVE", S.BUTTERFLY_VALVE, A.NONE),
    ("B.F.V.", S.BUTTERFLY_VALVE, A.NONE),
    ("SAFETY VALVE", S.SAFETY_VALVE, A.NONE),
    ("RELIEF VALVE", S.SAFETY_VALVE, A.NONE),
    ("3-WAY VALVE", S.THREE_WAY_VALVE, A.NONE),
    ("CENTRIFUGAL PUMP", S.PUMP_CENTRIFUGAL, A.NONE),
    ("SCREW PUMP", S.PUMP_SCREW, A.NONE),
    ("GEAR PUMP", S.PUMP_SCREW, A.NONE),
    ("STRAINER", S.STRAINER, A.NONE),
    ("MUD BOX", S.STRAINER, A.NONE),
    ("FILTER", S.FILTER, A.NONE),
    ("FLOW METER", S.FLOW_METER, A.NONE),
    ("REDUCER", S.REDUCER, A.NONE),
    ("SPECTACLE BLIND", S.SPECTACLE_FLANGE, A.NONE),
    ("FLEXIBLE HOSE", S.FLEXIBLE_HOSE, A.NONE),
    ("LOCAL INSTRUMENT", S.INSTRUMENT_LOCAL, A.NONE),
    ("체크 밸브", S.CHECK_VALVE, A.NONE),
    ("체크밸브", S.CHECK_VALVE, A.NONE),
    ("게이트 밸브", S.GATE_VALVE, A.NONE),
    ("글로브 밸브", S.GLOBE_VALVE, A.NONE),
    ("볼 밸브", S.BALL_VALVE, A.NONE),
    ("버터플라이 밸브", S.BUTTERFLY_VALVE, A.NONE),
    ("안전 밸브", S.SAFETY_VALVE, A.NONE),
    ("원심 펌프", S.PUMP_CENTRIFUGAL, A.NONE),
    ("스크류다운 체크 밸브", S.SDNR_VALVE, A.NONE),
    ("HYD. OPERATED BUTTERFLY VALVE", S.BUTTERFLY_VALVE, A.HYDRAULIC),
    ("REMOTE BUTTERFLY VALVE", S.BUTTERFLY_VALVE, A.HYDRAULIC),
    ("유압 버터플라이 밸브", S.BUTTERFLY_VALVE, A.HYDRAULIC),
    ("PNEU. CONTROL GLOBE VALVE", S.GLOBE_VALVE, A.DIAPHRAGM),
    ("TEMP. CONTROL VALVE", S.GLOBE_VALVE, A.DIAPHRAGM),
    ("MOTOR OPERATED GATE VALVE", S.GATE_VALVE, A.ELECTRIC),
    ("QUICK-CLOSING GLOBE VALVE", S.GLOBE_VALVE, A.QUICK_CLOSING),
    ("REMOTE QUICK CLOSING GATE VALVE", S.GATE_VALVE, A.QUICK_CLOSING),
    ("PANEL / REMOTE INSTRUMENT", S.INSTRUMENT_PANEL, A.NONE),  # 'remote' is no actuator of an instrument
    ("CENTRIFUGAL PURIFIER", S.PURIFIER, A.NONE),
    ("BALLAST WATER TREATMENT UNIT", S.BWTS_UNIT, A.NONE),  # 'BALL' only as a whole word
    ("OILY WATER SEPARATOR", S.OILY_WATER_SEPARATOR, A.NONE),
    ("SIGHT GLASS", S.UNKNOWN, A.NONE),
    ("REMOTE OPERATED", S.UNKNOWN, A.NONE),
    ("", S.UNKNOWN, A.NONE),
])
def test_map_description(text, cls, act):
    m = map_description(text)
    assert (m.cls, m.actuator) == (cls, act), (text, m)
    if cls != S.UNKNOWN:
        assert m.keyword


def test_page_heading_gate():
    from shippid.extract.legend import page_has_heading

    assert page_has_heading("GENERAL NOTES\n1. ...\nSYMBOL LEGEND")
    assert page_has_heading("L\nE\nG\nE\nN\nD")  # text written one character at a time
    assert page_has_heading("범 례")
    assert not page_has_heading("GENERAL NOTES\nBALLAST SYSTEM")


def test_normalize_description():
    assert normalize_description(" S.D.N.R.  valve ") == "SDNR VALVE"
    assert normalize_description("CHECK (NON-RETURN) VALVE") == "CHECK NON RETURN VALVE"
    assert normalize_description("체크(역지) 밸브") == "체크 역지 밸브"


# ----------------------------------------------------------------------------- the hand-built legend sheet

LEGEND_ROWS = [
    (S.BUTTERFLY_VALVE, REMOTE_BFV, A.NONE, "REMOTE BUTTERFLY VALVE"),
    (S.CHECK_VALVE, YARD_CHECK, A.NONE, "CHECK VALVE (NON-RETURN)"),
    (S.GATE_VALVE, YARD_GATE, A.HANDWHEEL, "GATE VALVE"),
    (S.GLOBE_VALVE, (), A.NONE, "GLOBE VALVE"),  # the built-in drawing: confirms the library symbol
    (S.UNKNOWN, WIDGET, A.NONE, "SPECIAL WIDGET"),  # names no known symbol: kept for Gemma
    (S.TANK, (), A.NONE, "TANK"),  # box-like equipment of any size: the built-in template is used
]


def _remote_valve_sheet(tmp_path, with_legend: bool = True, name: str = "remote.pdf"):
    sh = Sheet()
    box = sh.legend(860, 40, "LEGEND", LEGEND_ROWS) if with_legend else None
    y = 400.0
    xs = [(S.BUTTERFLY_VALVE, REMOTE_BFV, 200.0, A.NONE), (S.CHECK_VALVE, YARD_CHECK, 320.0, A.NONE),
          (S.GATE_VALVE, YARD_GATE, 440.0, A.HYDRAULIC), (S.GLOBE_VALVE, (), 560.0, A.HANDWHEEL),
          (S.BUTTERFLY_VALVE, REMOTE_BFV, 680.0, A.NONE)]
    x_prev = 100.0
    for cls, prims, x, act in xs:
        a, b = _inline(sh, cls, prims, x, y, act, scale=18.0)
        sh.line([(x_prev, y), (a, y)])
        x_prev = b
    sh.line([(x_prev, y), (800.0, y), (800.0, 520.0)])
    sh.glyph(S.UNKNOWN, WIDGET, Placement(420.0, 620.0, scale=22.0))
    sh.text(110, 392, "150A-BA-001", 6.5)
    return sh.save(tmp_path / name), sh, box


def test_find_legends_rows_and_mapping(tmp_path):
    path, sh, box = _remote_valve_sheet(tmp_path)
    with pymupdf.open(path) as pdf:
        legends = find_legends(pdf[0], 0)
    assert len(legends) == 1
    lg = legends[0]
    assert lg.heading == "LEGEND"
    assert all(abs(a - b) <= 1.0 for a, b in zip(lg.bbox, box))
    got = {r.description: r for r in lg.rows}
    assert set(got) == {d for *_, d in LEGEND_ROWS}
    assert got["REMOTE BUTTERFLY VALVE"].cls == S.BUTTERFLY_VALVE
    assert got["REMOTE BUTTERFLY VALVE"].actuator == A.HYDRAULIC
    assert got["CHECK VALVE (NON-RETURN)"].cls == S.CHECK_VALVE
    assert got["SPECIAL WIDGET"].cls == S.UNKNOWN and [r.description for r in lg.unmapped] == ["SPECIAL WIDGET"]
    # each glyph sits left of its description, on its row
    for r in lg.rows:
        assert r.glyph_bbox[2] < r.text_bbox[0]
        assert abs((r.glyph_bbox[1] + r.glyph_bbox[3]) / 2 - (r.text_bbox[1] + r.text_bbox[3]) / 2) < 6.0
    # a page without a legend
    empty = Sheet()
    p2 = empty.save(tmp_path / "empty.pdf")
    with pymupdf.open(p2) as pdf:
        assert find_legends(pdf[0]) == []


def test_learned_templates(tmp_path):
    path, _, _ = _remote_valve_sheet(tmp_path)
    with pymupdf.open(path) as pdf:
        book = LegendBook.learn(find_legends(pdf[0], 0))
    status = {r.description: r.status for lg in book.legends for r in lg.rows}
    assert status == {"REMOTE BUTTERFLY VALVE": "template", "CHECK VALVE (NON-RETURN)": "template",
                      "GATE VALVE": "template", "GLOBE VALVE": "library", "SPECIAL WIDGET": "unmapped",
                      "TANK": "skipped: box-like equipment of any size (built-in template)"}
    specs = {s.legend: s for s in book.specs}
    assert set(specs) == {"REMOTE BUTTERFLY VALVE", "CHECK VALVE (NON-RETURN)", "GATE VALVE"}
    bfv = specs["REMOTE BUTTERFLY VALVE"]
    assert isinstance(bfv, LegendSpec) and bfv.cls == S.BUTTERFLY_VALVE and bfv.category == Category.VALVE
    assert bfv.inherent_actuator == A.HYDRAULIC  # stated by the description, the glyph draws no actuator
    # normalised frame: flow axis = local x through the circle's outermost ends, port distance as built in
    circle = next(p for p in bfv.prims if isinstance(p, Circle))
    lib = get_symbol(S.BUTTERFLY_VALVE)
    assert abs(circle.cx) < 1e-3 and abs(circle.cy) < 1e-3
    assert abs(2 * circle.r - (lib.ports[1].x - lib.ports[0].x)) < 1e-3
    ports = {p.name: p for p in bfv.ports}
    assert set(ports) == {"in", "out"}
    assert ports["in"].x == pytest.approx(-circle.r, abs=1e-3) and ports["in"].y == pytest.approx(0, abs=1e-3)
    assert (ports["in"].dx, ports["in"].dy) == (-1.0, 0.0) and (ports["out"].dx, ports["out"].dy) == (1.0, 0.0)
    # the gate's handwheel is a library decoration: taken off the learned body, recorded as drawn
    gate = specs["GATE VALVE"]
    assert gate.rows == (("handwheel", "GATE VALVE"),) and gate.inherent_actuator == A.NONE
    ext = gate.extent
    assert ext[1] >= -0.31 and ext[3] <= 0.31 and ext[0] == pytest.approx(-0.5, abs=1e-3)  # no stem above the body
    assert book.confirmed == {(S.GLOBE_VALVE, A.NONE): "GLOBE VALVE"}
    meta = book.to_meta()
    assert [u["description"] for u in meta["unmapped"]] == ["SPECIAL WIDGET"]
    assert len(meta["templates"]) == 3


def test_extract_pdf_uses_the_legend(tmp_path):
    path, sh, box = _remote_valve_sheet(tmp_path)
    doc = extract_pdf(path)
    res = _objects_by_box(doc, sh)
    for cls, act, o in res:
        assert o is not None, (cls, act)
    remote = [o for c, _, o in res if c == S.BUTTERFLY_VALVE]
    assert len(remote) == 2
    for o in remote:
        assert o.cls == S.BUTTERFLY_VALVE and o.actuator == A.HYDRAULIC
        assert o.attributes["template"] == "legend:REMOTE BUTTERFLY VALVE"
        assert o.confidence >= 0.9
        assert sorted(p.name for p in o.ports) == ["in", "out"]
        assert sum(1 for p in doc.pipes for e in (p.start, p.end) if e.object_id == o.id) == 2
    by_cls = {c: o for c, _, o in res}
    assert by_cls[S.CHECK_VALVE].cls == S.CHECK_VALVE
    assert by_cls[S.CHECK_VALVE].attributes["template"] == "legend:CHECK VALVE (NON-RETURN)"
    gate = by_cls[S.GATE_VALVE]
    assert gate.cls == S.GATE_VALVE and gate.actuator == A.HYDRAULIC  # learned body + library actuator decoration
    assert gate.attributes["template"].startswith("legend:GATE VALVE")
    globe = by_cls[S.GLOBE_VALVE]
    assert globe.cls == S.GLOBE_VALVE and globe.actuator == A.HANDWHEEL
    assert globe.attributes["template"] == "legend:GLOBE VALVE"  # built-in symbol confirmed by the legend
    widget = by_cls[S.UNKNOWN]
    assert widget.cls == S.UNKNOWN and widget.attributes["legend"]["description"] == "SPECIAL WIDGET"
    # the legend itself is no part of the drawing
    for o in doc.objects:
        assert not (o.bbox.x0 >= box[0] - 1 and o.bbox.x1 <= box[2] + 1 and o.bbox.y0 >= box[1] - 1
                    and o.bbox.y1 <= box[3] + 1), o
    legend_texts = [t for t in doc.texts if t.role == TextRole.LEGEND]
    assert {t.text for t in legend_texts} >= {"LEGEND", "REMOTE BUTTERFLY VALVE", "SPECIAL WIDGET"}
    meta = doc.meta["legend"]
    assert len(meta["legends"]) == 1 and len(meta["legends"][0]["rows"]) == len(LEGEND_ROWS)
    assert doc.pages[0].index == 0 and doc.meta["pages"]["0"]["regions"]["legends"]


def test_without_learning_the_glyph_is_not_a_butterfly_valve(tmp_path):
    path, sh, _ = _remote_valve_sheet(tmp_path)
    doc = extract_pdf(path, options=ExtractOptions(learn_legends=False))
    assert "legend" not in doc.meta
    res = _objects_by_box(doc, sh)
    remote = [o for c, _, o in res if c == S.BUTTERFLY_VALVE]
    assert remote and all(o is None or o.cls != S.BUTTERFLY_VALVE for o in remote)
    assert all(o is None or "template" not in o.attributes for _, _, o in res)
    check = next(o for c, _, o in res if c == S.CHECK_VALVE)
    assert check is None or check.cls != S.CHECK_VALVE or check.confidence < 0.75


def test_separate_legend_sheet(tmp_path):
    """Legend on sheet 1, drawing on sheet 2: extracting only the drawing still learns the legend."""
    pdf = pymupdf.open()
    leg = Sheet(pdf)
    leg.legend(100, 100, "SYMBOL LEGEND", LEGEND_ROWS[:3], two_columns=True)
    leg.sh.commit()
    for x, y, s, size, font in leg.texts:
        leg.page.insert_text((x, y), s, fontsize=size, fontname=font, color=BLACK)
    dr = Sheet(pdf)
    y, x_prev = 300.0, 100.0
    for k, x in enumerate((220.0, 380.0, 540.0)):
        a, b = _inline(dr, S.BUTTERFLY_VALVE, REMOTE_BFV, x, y) if k != 1 else _inline(dr, S.CHECK_VALVE, YARD_CHECK, x, y)
        dr.line([(x_prev, y), (a, y)])
        x_prev = b
    dr.line([(x_prev, y), (700.0, y)])
    path = dr.save(tmp_path / "two_sheets.pdf")
    doc = extract_pdf(path, pages=[1])
    assert [p.index for p in doc.pages] == [1]
    assert doc.meta["legend"]["legends"][0]["page"] == 0
    res = _objects_by_box(doc, dr)
    assert [o.cls if o else None for _, _, o in res] == [S.BUTTERFLY_VALVE, S.CHECK_VALVE, S.BUTTERFLY_VALVE]
    assert all(o.attributes["template"].startswith("legend:") for _, _, o in res)


YARD_SDNR = YARD_CHECK + (Circle(0.0, 0.0, 0.1, fill=True),)


def test_table_legend_with_wrapped_description(tmp_path):
    """A gridded legend table: heading cell, SYMBOL | DESCRIPTION header row, one description on two
    lines; the grid rules are no glyphs."""
    sh = Sheet()
    x0, y0, w, xs = 100.0, 100.0, 320.0, 170.0
    rows_y = [y0, y0 + 20, y0 + 40, y0 + 70, y0 + 100, y0 + 130]  # heading, header, 4 data rows (one tall)
    for y in rows_y + [y0 + 160]:
        sh.line([(x0, y), (x0 + w, y)], width=0.5)
    sh.line([(x0, y0), (x0, y0 + 160)], width=0.5)
    sh.line([(x0 + w, y0), (x0 + w, y0 + 160)], width=0.5)
    sh.line([(xs, y0 + 20), (xs, y0 + 160)], width=0.5)
    sh.text(x0 + 6, y0 + 14, "SYMBOL LEGEND", 8.0, "hebo")
    sh.text(x0 + 15, y0 + 33, "SYMBOL", 6.0)
    sh.text(xs + 8, y0 + 33, "DESCRIPTION", 6.0)
    entries = [(S.CHECK_VALVE, YARD_CHECK, ["NON-RETURN VALVE"]),
               (S.SDNR_VALVE, YARD_SDNR, ["SCREW DOWN NON RETURN", "VALVE (S.D.N.R.)"]),
               (S.STRAINER, (), ["STRAINER"]),
               (S.UNKNOWN, WIDGET, ["DECK WATER SEAL"])]
    for (cls, prims, texts), ya, yb in zip(entries, rows_y[2:], rows_y[3:] + [y0 + 160]):
        cy = (ya + yb) / 2
        sh.glyph(cls, prims, Placement((x0 + xs) / 2, cy, scale=12.0), record=False, width=0.5)
        for k, t in enumerate(texts):
            sh.text(xs + 8, cy + 2.5 - 4.0 * (len(texts) - 1) + 8.0 * k, t, 6.0)
    path = sh.save(tmp_path / "table_legend.pdf")
    with pymupdf.open(path) as pdf:
        legends = find_legends(pdf[0], 0)
    assert len(legends) == 1 and legends[0].heading == "SYMBOL LEGEND"
    got = [(r.description, r.cls) for r in legends[0].rows]
    assert got == [("NON-RETURN VALVE", S.CHECK_VALVE), ("SCREW DOWN NON RETURN VALVE (S.D.N.R.)", S.SDNR_VALVE),
                   ("STRAINER", S.STRAINER), ("DECK WATER SEAL", S.UNKNOWN)]
    for r in legends[0].rows:  # glyph geometry only: no grid rules
        assert r.glyph_bbox[2] - r.glyph_bbox[0] < 20 and r.glyph_bbox[3] - r.glyph_bbox[1] < 20
    book = LegendBook.learn(legends)
    assert {s.legend: s.cls for s in book.specs} == {"NON-RETURN VALVE": S.CHECK_VALVE,
                                                    "SCREW DOWN NON RETURN VALVE (S.D.N.R.)": S.SDNR_VALVE}
    assert [r.description for r in book.unmapped] == ["DECK WATER SEAL"]
    assert book.confirmed == {(S.STRAINER, A.NONE): "STRAINER"}


def test_legend_relabels_a_builtin_glyph(tmp_path):
    """A yard whose legend calls the plain bow-tie a non-return valve: on this drawing every plain
    bow-tie is a check valve (the legend states the convention; the built-in gate ties and loses)."""
    sh = Sheet()
    sh.legend(860, 40, "LEGEND", [(S.GATE_VALVE, (), A.NONE, "NON-RETURN VALVE"),
                                 (S.GLOBE_VALVE, (), A.NONE, "GLOBE VALVE")])
    y, x_prev = 400.0, 100.0
    for x, cls in ((220.0, S.GATE_VALVE), (380.0, S.GLOBE_VALVE), (540.0, S.GATE_VALVE)):
        a, b = _inline(sh, cls, (), x, y)
        sh.line([(x_prev, y), (a, y)])
        x_prev = b
    sh.line([(x_prev, y), (700.0, y)])
    path = sh.save(tmp_path / "relabel.pdf")
    doc = extract_pdf(path)
    rows = {r["description"]: r for r in doc.meta["legend"]["legends"][0]["rows"]}
    assert rows["NON-RETURN VALVE"]["status"] == "remapped" and rows["NON-RETURN VALVE"]["same_as"] == "gate_valve"
    assert rows["GLOBE VALVE"]["status"] == "library"
    got = [(o.cls, o.attributes.get("template")) for _, _, o in _objects_by_box(doc, sh)]
    assert got == [(S.CHECK_VALVE, "legend:NON-RETURN VALVE"), (S.GLOBE_VALVE, "legend:GLOBE VALVE"),
                   (S.CHECK_VALVE, "legend:NON-RETURN VALVE")]
    plain = extract_pdf(path, options=ExtractOptions(learn_legends=False))
    assert [o.cls for _, _, o in _objects_by_box(plain, sh)] == [S.GATE_VALVE, S.GLOBE_VALVE, S.GATE_VALVE]


def test_one_glyph_two_meanings_is_not_learned():
    from shippid.extract.legend import Legend, LegendRow

    def row(desc: str, x: float) -> LegendRow:
        prims = tuple(_map_prims(REMOTE_BFV, x, 50.0, 10.0))
        m = map_description(desc)
        return LegendRow(0, desc, (x - 5, 45.0, x + 5, 55.0), (x + 20, 47.0, x + 80, 53.0), prims, m.cls, m.actuator)

    book = LegendBook.learn([Legend(0, (0.0, 0.0, 400.0, 100.0), "LEGEND",
                                    [row("REMOTE BUTTERFLY VALVE", 50.0), row("BALL VALVE", 250.0)])])
    assert book.specs == [] and book.library is None
    assert {r.status for lg in book.legends for r in lg.rows} == {"conflict"}


def _map_prims(prims, cx: float, cy: float, s: float):
    from shippid.extract.legend import _map

    return _map(prims, lambda u, v: (cx + s * u, cy + s * v))


# ----------------------------------------------------------------------------- Korean legend, stripped stubs


def test_korean_legend_pump_stub_and_instrument(tmp_path):
    """'범례' legend with a yard pump (circle + tangential discharge stub that merges into the
    discharge pipe) and a double-circle local instrument. A plain circle with a radial tap stays
    an instrument: the pump template's stub is only taken as stripped when the pipe runs there."""
    sh = Sheet()
    box = sh.legend(860, 40, "범례", [(S.PUMP_CENTRIFUGAL, YARD_PUMP, A.NONE, "원심 펌프"),
                                     (S.INSTRUMENT_LOCAL, DOUBLE_BUBBLE, A.NONE, "현장 계기"),
                                     (S.CHECK_VALVE, YARD_CHECK, A.NONE, "체크 밸브")], font="korea")
    s = 22.0
    px, py = 300.0, 500.0
    sh.glyph(S.PUMP_CENTRIFUGAL, YARD_PUMP, Placement(px, py, scale=s))
    sh.line([(120.0, py), (px - 0.5 * s, py)])  # suction to the port on the circle
    top = py - 0.5 * s
    sh.line([(px, top), (px, 300.0), (420.0, 300.0)])  # discharge continues the stub
    a, b = _inline(sh, S.CHECK_VALVE, YARD_CHECK, 440.0, 300.0)
    sh.line([(420.0, 300.0), (a, 300.0)])
    sh.line([(b, 300.0), (700.0, 300.0)])
    # instruments on the discharge line: a double-circle bubble and a plain circle, both on radial taps
    for x, prims in ((560.0, DOUBLE_BUBBLE), (640.0, (Circle(0.0, 0.0, 0.4),))):
        cls = S.INSTRUMENT_LOCAL
        sh.glyph(cls, prims, Placement(x, 240.0, scale=s))
        sh.line([(x, 300.0), (x, 240.0 + 0.4 * s)], width=0.6)
        sh.text(x - 4.5, 242.0, "PI", 5.0)
    path = sh.save(tmp_path / "korean.pdf")
    doc = extract_pdf(path)
    meta = doc.meta["legend"]
    rows = {r["description"]: r for r in meta["legends"][0]["rows"]}
    assert meta["legends"][0]["heading"] == "범례"
    assert {d: r["cls"] for d, r in rows.items()} == {"원심 펌프": "pump_centrifugal", "현장 계기": "instrument_local",
                                                      "체크 밸브": "check_valve"}
    res = _objects_by_box(doc, sh)
    got = [(c, o.cls if o else None) for c, _, o in res]
    assert got == [(S.PUMP_CENTRIFUGAL, S.PUMP_CENTRIFUGAL), (S.CHECK_VALVE, S.CHECK_VALVE),
                   (S.INSTRUMENT_LOCAL, S.INSTRUMENT_LOCAL), (S.INSTRUMENT_LOCAL, S.INSTRUMENT_LOCAL)], got
    pump = res[0][2]
    assert pump.attributes["template"] == "legend:원심 펌프"
    assert {p.name for p in pump.ports} == {"suction", "discharge"}
    plain = res[3][2]
    assert plain.attributes.get("template") is None  # the built-in plain bubble, not the stub-less pump
    # the Korean legend region is excluded from the objects and its texts are legend texts
    assert doc.meta["pages"]["0"]["regions"]["legends"]
    for o in doc.objects:
        assert not (o.bbox.x0 >= box[0] - 1 and o.bbox.x1 <= box[2] + 1 and o.bbox.y0 >= box[1] - 1
                    and o.bbox.y1 <= box[3] + 1)
    assert any(t.text == "원심 펌프" and t.role == TextRole.LEGEND for t in doc.texts)


# ----------------------------------------------------------------------------- matcher plumbing


def test_extended_library_puts_learned_variants_first():
    base = get_library()
    spec = LegendSpec(S.CHECK_VALVE, Category.VALVE, "x", "x", YARD_CHECK, get_symbol(S.CHECK_VALVE).ports,
                      legend="CHECK")
    lib = TemplateLibrary.extended([spec])
    assert lib.n_extra > 0 and all(isinstance(v.spec, LegendSpec) for v in lib.variants[:lib.n_extra])
    assert not any(isinstance(v.spec, LegendSpec) for v in lib.variants[lib.n_extra:])
    assert len(lib) == lib.n_extra + len(base)  # nothing of the base is a duplicate of the yard check valve
    assert [v.idx for v in lib.variants] == list(range(len(lib)))
    assert [v.idx for v in base.variants] == list(range(len(base)))  # the cached library is untouched
    assert base.n_extra == 0
    # identical geometry and class as a built-in variant: the built-in duplicate is dropped
    dup = LegendSpec(S.GATE_VALVE, Category.VALVE, "x", "x", get_symbol(S.GATE_VALVE).prims,
                     get_symbol(S.GATE_VALVE).ports, legend="GATE")
    lib2 = TemplateLibrary.extended([dup])
    assert len(lib2) == len(base)
    assert sum(1 for v in lib2.variants if v.cls == S.GATE_VALVE and not isinstance(v.spec, LegendSpec)) == 0


def test_glyph_template_strips_pipe_stubs():
    """A legend that draws pipe stubs on both sides of a glyph (-|><|-): the stubs are not part of
    the symbol, the ports are where the body ends."""
    from shippid.extract.legend import LegendRow

    s = 10.0
    body = [Poly(tuple((100 + s * u, 50 + s * v) for u, v in t.pts)) for t in (_TRI_L, _TRI_R)]
    stubs = [Line(100 - 0.5 * s - 6, 50, 100 - 0.5 * s, 50), Line(100 + 0.5 * s, 50, 100 + 0.5 * s + 6, 50)]
    row = LegendRow(0, "SLUICE VALVE", (89.0, 47.0, 111.0, 53.0), (120.0, 47.0, 160.0, 53.0), tuple(body + stubs),
                    S.GATE_VALVE)
    spec = glyph_template(row)
    assert spec is not None and len(spec.prims) == 2
    assert spec.extent == pytest.approx((-0.5, -0.3, 0.5, 0.3), abs=1e-3)
    assert {p.name: (round(p.x, 3), round(p.y, 3)) for p in spec.ports} == {"in": (-0.5, 0.0), "out": (0.5, 0.0)}
    # stubs and a drawn electric actuator together: both come off, the actuator is recorded
    from shippid.symbols import ACTUATOR_PRIMS
    from shippid.extract.legend import _map

    act = _map(ACTUATOR_PRIMS[A.ELECTRIC], lambda u, v: (100 + s * u, 50 + s * v))
    row2 = LegendRow(0, "MOTOR OPERATED SLUICE VALVE", (89.0, 41.0, 111.0, 53.0), (120.0, 47.0, 160.0, 53.0),
                     tuple(body + stubs + act), S.GATE_VALVE, A.ELECTRIC)
    spec2 = glyph_template(row2)
    assert row2.drawn_actuator == A.ELECTRIC and spec2.inherent_actuator == A.NONE
    assert spec2.extent == pytest.approx((-0.5, -0.3, 0.5, 0.3), abs=1e-3) and len(spec2.prims) == 2


def test_legend_tie_constant_is_small():
    assert 0 < LEGEND_TIE <= 0.02


# ----------------------------------------------------------------------------- the yard_symbols variant


@pytest.mark.parametrize("seed", [1000, 1007])
def test_yard_variant_classes_from_its_legend(tmp_path, seed):
    from shippid.datasets.variants import make_variant

    gt, pdf = make_variant(seed, "yard_symbols")
    path = tmp_path / f"yard_{seed}.pdf"
    path.write_bytes(pdf)
    doc = extract_pdf(path)
    assert doc.meta["legend"]["templates"]
    m = match_boxes([o.bbox for o in doc.objects], [o.bbox for o in gt.objects], 0.5)
    assert len(m) == len(gt.objects) == len(doc.objects)
    wrong = [(gt.objects[j].cls.value, doc.objects[i].cls.value) for i, j, _ in m
             if doc.objects[i].cls != gt.objects[j].cls or doc.objects[i].actuator != gt.objects[j].actuator]
    assert wrong == []
    learned = [o for o in doc.objects if str(o.attributes.get("template", "")).startswith("legend:")]
    assert learned
