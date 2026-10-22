"""Regression tests for the extractor review findings: degenerate pipe runs, outlines that
contain other drawing content (package boundaries, hull outlines, tank internals), note
callouts, equipment boxes with text, invisible / clipped / cropped content, scanned and
blank pages, duplicated CAD entities, implausibly small "equipment" and hatching.

Fixtures are drawn directly with PyMuPDF (``extract_fixtures.py``); nothing here depends
on :mod:`shippid.datasets`.
"""

from __future__ import annotations

import math

import pymupdf
import pytest

from extract_fixtures import A, S, Sheet, Style, evaluate
from shippid.extract.pipeline import _flow_directions, _propagate_line_numbers, extract_page, extract_pdf
from shippid.model import (
    BBox,
    Category,
    Endpoint,
    PIDObject,
    PipeKind,
    PipeRun,
    Point,
    SymbolClass,
    TextRole,
)
from shippid.pdf.reader import read_page

U = 18.0


def _reopen(sh: Sheet) -> pymupdf.Page:
    """Round-trip through PDF bytes (what a reader of a saved file sees)."""
    return pymupdf.open("pdf", sh.doc.tobytes())[0]


def _train(sh: Sheet, x0: float = 150, y: float = 300, k: float = 1.0) -> dict:
    """tank -> gate valve -> screw pump -> check valve -> tank, connected by pipes
    (``k`` scales the spacing; k = 1 spans about 580 pt)."""
    o = sh.symbol
    t1 = o(S.TANK, x0, y, stretch=(0.8, 1.0))
    v1 = o(S.GATE_VALVE, x0 + 180 * k, y, actuator=A.HANDWHEEL, tag="FO-V001")
    p1 = o(S.PUMP_SCREW, x0 + 270 * k, y)
    v2 = o(S.CHECK_VALVE, x0 + 360 * k, y)
    t2 = o(S.TANK, x0 + 550 * k, y)
    sh.connect(("o", t1.key), ("o", v1.key), [(t1.bbox[2], y), v1.ports["in"]])
    sh.connect(("o", v1.key), ("o", p1.key), [v1.ports["out"], p1.ports["in"]])
    sh.connect(("o", p1.key), ("o", v2.key), [p1.ports["out"], v2.ports["in"]])
    sh.connect(("o", v2.key), ("o", t2.key), [v2.ports["out"], (t2.bbox[0], y)])
    return dict(t1=t1, v1=v1, p1=p1, v2=v2, t2=t2)


def _summary(sh: Sheet, doc) -> dict:
    return evaluate(sh, doc).summary()


# --------------------------------------------------------------------------- degenerate runs


@pytest.mark.parametrize("overshoot", [0.0, 1.0, 2.0, 4.0])
def test_line_overshooting_an_end_valve(overshoot, tmp_path):
    """A line drawn through an end-of-line valve that stops 1-2 pt past its outlet left a
    stub run with both ends on the valve and crashed line-number propagation."""
    sh = Sheet(1190, 842, Style(scale=U))
    t1 = sh.symbol(S.TANK, 150, 300, stretch=(0.8, 1.0))
    v1 = sh.symbol(S.GATE_VALVE, 330, 300, actuator=A.HANDWHEEL, tag="FO-V001")
    sh.line([(t1.bbox[2], 300), (v1.ports["out"][0] + overshoot, 300)])
    sh.text(200, 296, "25A-FO-001", TextRole.LINE_NUMBER, size=5)
    sh.finish()
    path = tmp_path / "overshoot.pdf"
    sh.save(path)
    doc = extract_pdf(path)  # a crash here used to abort the whole multi-page document
    valve = next(o for o in doc.objects if o.cls == S.GATE_VALVE)
    assert "in" in {p.name for p in valve.ports}
    for p in doc.pipes:  # no run starts and ends on the same object
        assert not (p.start.object_id and p.start.object_id == p.end.object_id), p
    main = next(p for p in doc.pipes if {p.start.object_id, p.end.object_id} == {doc.objects[0].id, valve.id})
    assert main.line_number == "25A-FO-001"


def test_loop_run_on_one_valve_is_ignored_by_propagation():
    """Defence in depth: a run whose both ends sit on one valve (a loop) must not break
    line-number propagation or flow directions."""
    v = PIDObject(id="o1", cls=SymbolClass.GATE_VALVE, category=Category.VALVE, bbox=BBox(x0=0, y0=-5, x1=18, y1=5))
    t = PIDObject(id="o2", cls=SymbolClass.TANK, category=Category.EQUIPMENT, bbox=BBox(x0=-80, y0=-20, x1=-30, y1=20))
    pipes = [
        PipeRun(id="p1", points=[Point(x=-30, y=0), Point(x=0, y=0)], start=Endpoint(object_id="o2", port="n0"),
                end=Endpoint(object_id="o1", port="in"), line_number="25A-FO-001"),
        PipeRun(id="p2", points=[Point(x=18, y=0), Point(x=18, y=20), Point(x=0, y=20), Point(x=0, y=1)],
                start=Endpoint(object_id="o1", port="out"), end=Endpoint(object_id="o1", port="n1")),
    ]
    _propagate_line_numbers(pipes, [v, t], {0: (-15.0, 0.0)})

    class _Net:
        pieces: list = []

    class _Sep:
        arrows: list = []

    _flow_directions(pipes, [], _Net(), _Sep(), [v, t])
    assert pipes[0].line_number == "25A-FO-001"


# --------------------------------------------------------------------------- outlines around content


@pytest.mark.parametrize("dashes", [None, "[6 2 1 2] 0"])
def test_package_boundary_does_not_swallow_symbols(dashes):
    sh = Sheet(1190, 842, Style(scale=U))
    _train(sh)
    sh.shape.draw_rect(pymupdf.Rect(290, 250, 550, 345))  # skid / package boundary (solid or dash-dot)
    sh.shape.finish(width=0.5, color=(0, 0, 0), dashes=dashes)
    sh.text(295, 260, "F.O. TRANSFER PUMP UNIT", TextRole.LABEL, None, size=6)
    sh.finish()
    doc = extract_page(_reopen(sh), 0)
    s = _summary(sh, doc)
    assert s["recall@iou0.5"] == 1.0 and s["precision@iou0.5"] == 1.0 and s["class_acc"] == 1.0, s
    assert s["conn_precision"] == 1.0 and s["conn_recall"] == 1.0, s


def test_hull_outline_behind_the_diagram():
    """A closed ship-plan outline (lines + Beziers) around the whole diagram is a boundary."""
    sh = Sheet(1190, 842, Style(scale=U))
    _train(sh, y=300)
    _train(sh, y=450)
    x0, x1, y0, y1 = 90, 760, 240, 510
    ym = (y0 + y1) / 2
    shp = sh.shape
    shp.draw_line((x0 + 60, y0), (x1 - 120, y0))
    shp.draw_bezier((x1 - 120, y0), (x1 - 20, y0), (x1, ym - 40), (x1 + 10, ym))
    shp.draw_bezier((x1 + 10, ym), (x1, ym + 40), (x1 - 20, y1), (x1 - 120, y1))
    shp.draw_line((x1 - 120, y1), (x0 + 60, y1))
    shp.draw_bezier((x0 + 60, y1), (x0, y1), (x0, y1 - 30), (x0, ym))
    shp.draw_bezier((x0, ym), (x0, y0 + 30), (x0, y0), (x0 + 60, y0))
    shp.finish(width=0.35, color=(0, 0, 0), closePath=False)
    sh.finish()
    doc = extract_page(_reopen(sh), 0)
    s = _summary(sh, doc)
    assert s["recall@iou0.5"] == 1.0 and s["precision@iou0.5"] == 1.0 and s["class_acc"] == 1.0, s
    assert s["conn_recall"] == 1.0 and s["conn_precision"] == 1.0, s


def _tank_with(inside: str) -> tuple[Sheet, object]:
    sh = Sheet(1190, 842, Style(scale=U))
    t1 = sh.symbol(S.TANK, 200, 300, stretch=(1.4, 1.4))
    sh.text(t1.bbox[0] + 4, t1.bbox[1] + 10, "NO.1 H.F.O. STOR. TK", TextRole.LABEL, t1.key, size=6)
    v1 = sh.symbol(S.GATE_VALVE, 330, 310, actuator=A.HANDWHEEL)
    t2 = sh.symbol(S.TANK, 480, 310)
    sh.connect(("o", t1.key), ("o", v1.key), [(t1.bbox[2], 310), v1.ports["in"]])
    sh.connect(("o", v1.key), ("o", t2.key), [v1.ports["out"], (t2.bbox[0], 310)])
    if inside == "coil":  # steam heating coil, fed through the tank top
        x0, x1, y = t1.bbox[0] + 10, t1.bbox[2] - 10, t1.bbox[3] - 8
        pts = [(x0, y - 18)] + [(x0 + (x1 - x0) * i / 8, y - (6 if i % 2 else 0)) for i in range(9)]
        sh.line([(x0, t1.bbox[1] - 30), (x0, y - 18)], width=0.6)
        sh.line(pts, width=0.6)
        sh.line([(x1, y), (x1, t1.bbox[1] - 30)], width=0.6)
    elif inside == "level":  # level switch bubble inside the tank
        sh.symbol(S.INSTRUMENT_LOCAL, t1.bbox[2] - 14, t1.bbox[1] + 18)
    sh.finish()
    return sh, t1


@pytest.mark.parametrize("inside", ["coil", "level"])
def test_tank_internals_are_not_merged_into_the_tank(inside):
    sh, t1 = _tank_with(inside)
    doc = extract_page(_reopen(sh), 0)
    s = _summary(sh, doc)
    assert s["recall@iou0.5"] == 1.0 and s["class_acc"] == 1.0 and s["conn_recall"] == 1.0, s
    tank = max((o for o in doc.objects if o.cls == S.TANK), key=lambda o: o.bbox.area)
    if inside == "coil":
        assert tank.attributes.get("internals", 0) >= 1  # the coil is kept as the tank's internal detail
        assert len(doc.objects) == 3
    else:
        assert any(o.cls == S.INSTRUMENT_LOCAL for o in doc.objects)


@pytest.mark.parametrize("stretch", [(1.6, 1.4), (1.2, 1.4)])  # above / below the containment size limit
def test_valve_drawn_inside_a_tank_is_connected(stretch):
    """A ballast suction valve drawn inside its tank: the pipe leaving the tank runs to the
    valve, it is not cut at the tank outline."""
    sh = Sheet(1190, 842, Style(scale=U))
    sh.symbol(S.TANK, 200, 300, stretch=stretch)
    v1 = sh.symbol(S.BUTTERFLY_VALVE, 205, 300, actuator=A.HYDRAULIC)
    v2 = sh.symbol(S.GATE_VALVE, 400, 300, actuator=A.HANDWHEEL)
    t2 = sh.symbol(S.TANK, 560, 300)
    sh.connect(("o", v1.key), ("o", v2.key), [v1.ports["out"], v2.ports["in"]])
    sh.connect(("o", v2.key), ("o", t2.key), [v2.ports["out"], (t2.bbox[0], 300)])
    sh.finish()
    doc = extract_page(sh.page, 0)
    s = _summary(sh, doc)
    assert s["recall@iou0.5"] == 1.0 and s["class_acc"] == 1.0, s
    assert s["conn_precision"] == 1.0 and s["conn_recall"] == 1.0, s


def test_level_switch_in_a_tank_keeps_its_signal_line():
    sh = Sheet(1190, 842, Style(scale=U))
    t1 = sh.symbol(S.TANK, 200, 300, stretch=(1.4, 1.4))
    ls = sh.symbol(S.INSTRUMENT_LOCAL, t1.bbox[2] - 16, t1.bbox[1] + 16)
    pnl = sh.symbol(S.INSTRUMENT_PANEL, 420, t1.bbox[1] + 16)
    sh.connect(("o", ls.key), ("o", pnl.key), [(ls.bbox[2], t1.bbox[1] + 16), (pnl.bbox[0], t1.bbox[1] + 16)], dashed=True)
    v = sh.symbol(S.GATE_VALVE, 330, 320, actuator=A.HANDWHEEL)
    sh.connect(("o", t1.key), ("o", v.key), [(t1.bbox[2], 320), v.ports["in"]])
    sh.finish()
    doc = extract_page(sh.page, 0)
    s = _summary(sh, doc)
    assert s["recall@iou0.5"] == 1.0 and s["precision@iou0.5"] == 1.0 and s["class_acc"] == 1.0, s
    assert s["conn_precision"] == 1.0 and s["conn_recall"] == 1.0, s
    sig = [p for p in doc.pipes if p.kind == PipeKind.SIGNAL]
    assert len(sig) == 1 and {doc.object_by_id()[e.object_id].cls for e in (sig[0].start, sig[0].end)} == \
        {S.INSTRUMENT_LOCAL, S.INSTRUMENT_PANEL}


def test_large_sea_chest_keeps_its_insert():
    """A sea chest drawn larger than the containment limit is still a sea chest: the
    rectangle and its wave insert are re-joined when together they match the library."""
    sh = Sheet(1190, 842, Style(scale=U))
    sc = sh.symbol(S.SEA_CHEST, 200, 400, stretch=(3.5, 3.0))  # ~ 5.6 x 2.4 symbol units
    v1 = sh.symbol(S.GATE_VALVE, 200, 300, rotation=270, actuator=A.HANDWHEEL)
    v2 = sh.symbol(S.GATE_VALVE, 400, 250, actuator=A.HANDWHEEL)
    v3 = sh.symbol(S.GLOBE_VALVE, 550, 250, actuator=A.HANDWHEEL)
    sh.connect(("o", sc.key), ("o", v1.key), [sc.ports["out"], v1.ports["in"]])
    sh.connect(("o", v1.key), ("o", v2.key), [v1.ports["out"], (200, 250), v2.ports["in"]])
    sh.connect(("o", v2.key), ("o", v3.key), [v2.ports["out"], v3.ports["in"]])
    sh.finish()
    doc = extract_page(_reopen(sh), 0)
    s = _summary(sh, doc)
    assert s["recall@iou0.5"] == 1.0 and s["class_acc"] == 1.0, (s, [o.cls for o in doc.objects])


def test_hatched_area_is_not_a_symbol():
    sh = Sheet(1190, 842, Style(scale=U))
    _train(sh)
    x0, y0, w, h = 300, 450, 300, 80
    c = 0.0
    while c < w + h:  # 45 degree hatching clipped to the rectangle
        xa, ya = (x0 + c, y0) if c <= w else (x0 + w, y0 + c - w)
        xb, yb = (x0, y0 + c) if c <= h else (x0 + c - h, y0 + h)
        sh.shape.draw_line((xa, ya), (xb, yb))
        sh.shape.finish(width=0.25, color=(0, 0, 0))
        c += 3.0
    sh.shape.draw_rect(pymupdf.Rect(x0, y0, x0 + w, y0 + h))
    sh.shape.finish(width=0.6, color=(0, 0, 0))
    sh.finish()
    doc = extract_page(_reopen(sh), 0)
    s = _summary(sh, doc)
    assert s["precision@iou0.5"] == 1.0 and s["recall@iou0.5"] == 1.0, s
    assert not [o for o in doc.objects if o.bbox.y0 >= y0 - 5]


def test_small_box_between_valves_is_not_equipment():
    """Stretchable equipment has a size prior: a 14 x 12 pt box between 18 pt valves is no
    tank and no main engine."""
    sh = Sheet(1190, 842, Style(scale=U))
    t1 = sh.symbol(S.TANK, 150, 300, stretch=(0.8, 1.0))
    v1 = sh.symbol(S.GATE_VALVE, 300, 300, actuator=A.HANDWHEEL)
    v2 = sh.symbol(S.GLOBE_VALVE, 560, 300, actuator=A.HANDWHEEL)
    t2 = sh.symbol(S.TANK, 720, 300)
    for x0, diag in ((380, True), (460, False)):
        sh.shape.draw_rect(pymupdf.Rect(x0, 294, x0 + 14, 306))
        sh.shape.finish(width=0.6, color=(0, 0, 0))
        if diag:
            sh.shape.draw_line((x0, 294), (x0 + 14, 306))
            sh.shape.draw_line((x0, 306), (x0 + 14, 294))
            sh.shape.finish(width=0.6, color=(0, 0, 0))
    sh.line([(t1.bbox[2], 300), v1.ports["in"]])
    sh.line([v1.ports["out"], (380, 300)])
    sh.line([(394, 300), (460, 300)])
    sh.line([(474, 300), v2.ports["in"]])
    sh.line([v2.ports["out"], (t2.bbox[0], 300)])
    sh.finish()
    doc = extract_page(_reopen(sh), 0)
    small = [o for o in doc.objects if o.bbox.width < 20 and o.bbox.y0 > 290 and 370 < o.bbox.x0 < 480]
    assert len(small) == 2
    assert all(o.cls not in (S.TANK, S.ENGINE, S.BWTS_UNIT, S.SEA_CHEST, S.OILY_WATER_SEPARATOR) for o in small)
    plain = next(o for o in small if o.bbox.x0 > 450)
    assert plain.cls == S.UNKNOWN and plain.confidence < 0.62  # a plain box is no library symbol
    assert {o.cls for o in doc.objects if o.bbox.width > 30} == {S.TANK}


def test_tanks_alone_on_a_page_are_still_tanks():
    """Without valves the symbol unit is estimated from the tanks themselves: the size prior
    must not apply then."""
    sh = Sheet(500, 400, Style(scale=U), frame=False)
    a = sh.symbol(S.TANK, 80, 200, stretch=(0.5, 0.5))
    b = sh.symbol(S.TANK, 420, 200, stretch=(0.5, 0.5))
    sh.line([(a.bbox[2], 200), (b.bbox[0], 200)])
    sh.finish()
    doc = extract_page(sh.page, 0)
    assert [o.cls for o in doc.objects] == [S.TANK, S.TANK]
    assert all(o.confidence > 0.9 for o in doc.objects)
    assert doc.meta["pages"]["0"]["unit_anchored"] is False


# --------------------------------------------------------------------------- notes and tables


@pytest.mark.parametrize("where", ["inside_tank", "above_valve"])
def test_note_callout_is_not_a_notes_heading(where):
    sh = Sheet(1190, 842, Style(scale=U))
    t1 = sh.symbol(S.TANK, 150, 300, stretch=(1.0, 1.2))
    sh.text(t1.bbox[0] + 4, 290, "NO.1 F.O. TK", TextRole.LABEL, t1.key, size=6)
    if where == "inside_tank":
        sh.text(t1.bbox[0] + 4, 300, "NOTE 2", TextRole.NOTE, None, size=6)
    v1 = sh.symbol(S.GATE_VALVE, 300, 300, actuator=A.HANDWHEEL, tag="FO-V001")
    if where == "above_valve":
        sh.text(285, 278, "NOTE 1", TextRole.NOTE, None, size=6)
    p1 = sh.symbol(S.PUMP_SCREW, 420, 300)
    v2 = sh.symbol(S.GLOBE_VALVE, 540, 300, actuator=A.HANDWHEEL, tag="FO-V002")
    t2 = sh.symbol(S.TANK, 700, 300)
    sh.connect(("o", t1.key), ("o", v1.key), [(t1.bbox[2], 300), v1.ports["in"]])
    sh.connect(("o", v1.key), ("o", p1.key), [v1.ports["out"], p1.ports["in"]])
    sh.connect(("o", p1.key), ("o", v2.key), [p1.ports["out"], v2.ports["in"]])
    sh.connect(("o", v2.key), ("o", t2.key), [v2.ports["out"], (t2.bbox[0], 300)])
    sh.title_block("F.O. TRANSFER SYSTEM", "H1-FO-001", "H.1", "VLCC")
    sh.notes(40, 700, ["SEE SPEC.", "ALL VALVES NORMALLY OPEN."])
    sh.finish()
    doc = extract_page(sh.page, 0)
    s = _summary(sh, doc)
    assert s["recall@iou0.5"] == 1.0 and s["tag_acc"] == 1.0 and s["conn_recall"] == 1.0, s
    assert s["text_role_acc"] == 1.0, [e for e in evaluate(sh, doc).errors if e[0] == "role"]
    regions = doc.meta["pages"]["0"]["regions"]
    assert len(regions["notes"]) == 1 and regions["notes"][0][1] > 600  # only the real notes block
    callout = next(t for t in doc.texts if t.text.startswith("NOTE "))
    assert callout.role == TextRole.NOTE and callout.owner_id is None
    host = next(o for o in doc.objects if o.cls == (S.TANK if where == "inside_tank" else S.GATE_VALVE)
                and o.bbox.x0 < 320)
    assert host.attributes.get("notes") == callout.text


@pytest.mark.parametrize("cls", [S.TANK, S.ENGINE])
def test_equipment_box_with_data_lines_is_not_a_table(cls):
    lines = ["NO.1 H.F.O. SETT. TK", "CAP. 50 M3", "FR. 45-52 (P)", "TEMP. 90 C", "HTG. COIL 30 M2", "INSUL. 75T"]
    sh = Sheet(1190, 842, Style(scale=U))
    t1 = sh.symbol(cls, 250, 300, stretch=(1.4, 1.6))
    for i, s in enumerate(lines):
        sh.text(t1.bbox[0] + 4, t1.bbox[1] + 10 + 8 * i, s, TextRole.LABEL, t1.key if i == 0 else None, size=6)
    sh.title_block("F.O. TRANSFER SYSTEM", "H1-FO-001", "H.1", "VLCC")
    sh.finish()
    doc = extract_page(sh.page, 0)
    assert [o.cls for o in doc.objects] == [cls]
    assert all(t.role == TextRole.LABEL for t in doc.texts if t.text in lines)
    assert doc.objects[0].attributes["name"] == "NO.1 H.F.O. SETT. TK"


def test_tanks_sharing_a_bulkhead_are_not_a_table():
    sh = Sheet(1190, 842, Style(scale=U))
    t1 = sh.symbol(S.TANK, 200, 300, stretch=(1.2, 1.4))
    w = t1.bbox[2] - t1.bbox[0]
    t2 = sh.symbol(S.TANK, 200 + w, 300, stretch=(1.2, 1.4))
    for t, nm in ((t1, "H.F.O. SETT. TK"), (t2, "H.F.O. SERV. TK")):
        sh.text(t.bbox[0] + 4, t.bbox[1] + 10, nm, TextRole.LABEL, t.key, size=6)
        sh.text(t.bbox[0] + 4, t.bbox[1] + 18, "(50 M3)", TextRole.LABEL, t.key, size=6)
        sh.text(t.bbox[0] + 4, t.bbox[1] + 26, "FR.45-52", TextRole.LABEL, t.key, size=6)
    sh.finish()
    doc = extract_page(sh.page, 0)
    assert sorted(o.cls.value for o in doc.objects) == ["tank", "tank"]
    assert {t.role for t in doc.texts if t.bbox.y0 > 200 and t.bbox.y1 < 400} == {TextRole.LABEL}


def test_ruled_text_table_is_still_a_table():
    """A real table (rows and columns of cells) keeps its NOTE role and blocks symbol search."""
    sh = Sheet(1190, 842, Style(scale=U))
    _train(sh)
    x0, y0, cw, rh = 600, 500, 90, 14
    shp = sh.shape
    for r in range(6):
        shp.draw_line((x0, y0 + r * rh), (x0 + 3 * cw, y0 + r * rh))
    for c in range(4):
        shp.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + 5 * rh))
    shp.finish(width=0.5, color=(0, 0, 0))
    for r in range(5):
        for c in range(3):
            sh.text(x0 + c * cw + 3, y0 + r * rh + 10, f"ITEM {r}{c}", TextRole.NOTE, None, size=5)
    sh.finish()
    doc = extract_page(sh.page, 0)
    assert all(t.role == TextRole.NOTE for t in doc.texts if t.text.startswith("ITEM"))
    assert _summary(sh, doc)["precision@iou0.5"] == 1.0


# --------------------------------------------------------------------------- reader: visibility


def _clipped_sheet() -> tuple[Sheet, pymupdf.Page]:
    """The sheet shown through a clipping window (CAD viewport): only the left half is visible."""
    sh = Sheet(1190, 842, Style(scale=U))
    _train(sh, x0=100, y=200, k=0.75)  # left half
    _train(sh, x0=700, y=600, k=0.75)  # right half: hidden by the clip
    sh.text(900, 560, "HIDDEN-V001", TextRole.TAG)
    sh.finish()
    out = pymupdf.open()
    pg = out.new_page(width=1190, height=842)
    pg.show_pdf_page(pymupdf.Rect(0, 0, 595, 842), pymupdf.open("pdf", sh.doc.tobytes()), 0,
                     clip=pymupdf.Rect(0, 0, 595, 842))
    return sh, pymupdf.open("pdf", out.tobytes())[0]


def test_reader_honours_clip_paths():
    sh, page = _clipped_sheet()
    pc = read_page(page)
    assert pc.n_clipped_paths > 0
    assert all(min(s.x0, s.x1) <= 595.5 for s in pc.segments)
    assert all(c.cx - c.r <= 595.5 for c in pc.circles)
    assert not [w for w in pc.words if w.bbox[0] > 595.5]
    doc = extract_page(page, 0)
    assert len(doc.objects) == 5 and all(o.bbox.x1 <= 596 for o in doc.objects)
    assert all(q.x <= 596 for p in doc.pipes for q in p.points)
    assert "HIDDEN-V001" not in {t.text for t in doc.texts}


def test_reader_honours_the_cropbox():
    sh = Sheet(1190, 842, Style(scale=U))
    _train(sh, x0=100, y=200, k=0.75)
    _train(sh, x0=700, y=600, k=0.75)
    sh.finish()
    src = pymupdf.open("pdf", sh.doc.tobytes())
    src[0].set_cropbox(pymupdf.Rect(0, 0, 595, 842))
    page = pymupdf.open("pdf", src.tobytes())[0]
    doc = extract_page(page, 0)
    assert doc.pages[0].width == 595
    assert len(doc.objects) == 5
    assert all(o.bbox.x1 <= page.rect.width + 1 for o in doc.objects)
    assert all(q.x <= page.rect.width + 1 for p in doc.pipes for q in p.points)


def test_scanned_page_with_ocr_layer_is_unreadable():
    sh = Sheet(1190, 842, Style(scale=U))
    _train(sh)
    sh.text(330, 280, "FO-V001", TextRole.TAG)
    sh.finish()
    pix = sh.page.get_pixmap(dpi=72)
    doc = pymupdf.open()
    page = doc.new_page(width=1190, height=842)
    page.insert_image(page.rect, pixmap=pix)
    for w in sh.page.get_text("words"):
        page.insert_text((w[0], w[3]), w[4], fontsize=6, render_mode=3)  # invisible OCR text
    pc = read_page(page)
    assert pc.n_hidden_chars > 0 and pc.n_chars == 0 and pc.scanned and not pc.readable
    assert all(w.hidden for w in pc.words)
    out = extract_page(page, 0)
    assert out.meta["unreadable_pages"] == [0] and not out.objects


def test_blank_page_is_empty_not_scanned(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    p2 = doc.new_page()
    shp = p2.new_shape()
    shp.draw_line((50, 50), (400, 50))
    shp.finish(width=1, color=(0, 0, 0))
    shp.commit()
    path = tmp_path / "blank.pdf"
    doc.save(path)
    out = extract_pdf(path)
    assert out.meta["empty_pages"] == [0]
    assert "unreadable_pages" not in out.meta
    assert out.meta["pages"]["0"]["empty"] and not out.meta["pages"]["1"]["empty"]


# --------------------------------------------------------------------------- duplicated entities


def _dotted_header(dup: str) -> Sheet:
    st = Style(scale=U, dot_tees=True)
    sh = Sheet(1190, 842, st)
    src = sh.symbol(S.TANK, 120, 300, stretch=(0.8, 1.0))
    xs = [300, 420, 540]
    sh.line([(src.bbox[2], 300), (640, 300)])
    sinks = []
    for x in xs:
        j = sh.junction(x, 300)
        if dup in ("dots", "all"):
            sh.junction(x, 300)  # the same dot drawn twice (duplicate CAD entity)
            sh.junctions.pop(list(sh.junctions)[-1])
        if dup == "ring":  # a dot with a separate, slightly larger outline circle
            sh.shape.draw_circle(pymupdf.Point(x, 300), max(1.6, 0.11 * U) + 0.4)
            sh.shape.finish(width=0.3, color=(0, 0, 0))
        v = sh.symbol(S.GATE_VALVE, x, 380, rotation=90, actuator=A.NONE)
        sh.line([(x, 300), v.ports["in"]])
        sh.edges.append((("j", j), ("o", v.key)))
        sinks.append(j)
    t2 = sh.symbol(S.TANK, 700, 300)
    sh.line([(640, 300), (t2.bbox[0], 300)])
    sh.edges += [(("o", src.key), ("j", sinks[0])), (("j", sinks[0]), ("j", sinks[1])), (("j", sinks[1]), ("j", sinks[2])),
                 (("j", sinks[2]), ("o", t2.key))]
    if dup == "all":  # every path drawn twice
        sh.finish()
        page = sh.page
        page.show_pdf_page(page.rect, pymupdf.open("pdf", sh.doc.tobytes()), 0)
        return sh
    sh.finish()
    return sh


@pytest.mark.parametrize("dup", ["none", "dots", "ring", "all"])
def test_duplicated_junction_dots(dup):
    sh = _dotted_header(dup)
    doc = extract_page(_reopen(sh), 0)
    s = _summary(sh, doc)
    assert s["recall@iou0.5"] == 1.0 and s["precision@iou0.5"] == 1.0, (s, [o.cls for o in doc.objects])
    assert s["conn_precision"] == 1.0 and s["conn_recall"] == 1.0, s
    assert sum(j.dotted for j in doc.junctions) == 3
    assert doc.meta["pages"]["0"]["counts"]["dots"] == 3


def test_primset_dedupes_coincident_primitives():
    from shippid.extract.primitives import PrimSet

    sh = Sheet(400, 300, Style(scale=U), frame=False)
    for _ in range(2):
        sh.symbol(S.CHECK_VALVE, 100, 100)
        sh.line([(10, 200), (300, 200)])
        sh.junction(150, 200)
    sh.finish()
    once = Sheet(400, 300, Style(scale=U), frame=False)
    once.symbol(S.CHECK_VALVE, 100, 100)
    once.line([(10, 200), (300, 200)])
    once.junction(150, 200)
    once.finish()
    a, b = PrimSet.from_page(read_page(sh.page)), PrimSet.from_page(read_page(once.page))
    assert len(a.prims) == len(b.prims) and len(a.rings) == len(b.rings)
    assert [p.id for p in a.prims] == list(range(len(a.prims)))
    assert all(a.prims[i].ring == rid for rid, r in a.rings.items() for i in r.prim_ids)
    assert sum(1 for r in a.rings.values() if r.filled) == 1  # the check valve's filled triangle


def test_duplicated_text_gives_one_tag():
    sh = Sheet(600, 300, Style(scale=U), frame=False)
    t = sh.symbol(S.TANK, 80, 150, stretch=(0.6, 0.8))
    v = sh.symbol(S.GATE_VALVE, 250, 150, actuator=A.HANDWHEEL, tag="BA-V001")
    sh.line([(t.bbox[2], 150), v.ports["in"]])
    pos = next(x.pos for x in sh.texts if x.text == "BA-V001")
    sh.finish()
    sh.page.insert_text(pos, "BA-V001", fontsize=sh.style.font)  # the tag text entity drawn twice
    doc = extract_page(_reopen(sh), 0)
    assert [x.text for x in doc.texts].count("BA-V001") == 1
    assert next(o for o in doc.objects if o.cls == S.GATE_VALVE).tag == "BA-V001"


def test_page_rotation_is_recorded():
    sh = Sheet(600, 300, Style(scale=U), frame=False)
    sh.symbol(S.GATE_VALVE, 250, 150, actuator=A.HANDWHEEL)
    sh.finish()
    sh.page.set_rotation(90)
    doc = extract_page(sh.page, 0)
    assert doc.pages[0].rotation == 90 and (doc.pages[0].width, doc.pages[0].height) == (300, 600)
    assert math.isclose(doc.objects[0].bbox.center.x, 300 - 150, abs_tol=2)
