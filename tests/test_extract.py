"""Tests for shippid.pdf.reader and shippid.extract (deterministic vector extraction).

Fixtures are drawn directly with :func:`shippid.symbols.draw_symbol`, PyMuPDF
shapes and real text (see ``extract_fixtures.py``); nothing here depends on
:mod:`shippid.datasets`.
"""

from __future__ import annotations

import math
import time

import pymupdf
import pytest

from extract_fixtures import A, S, Sheet, Style, evaluate, random_sheet, same_orientation
from shippid.config import Config
from shippid.extract.pipeline import ExtractOptions, extract_page, extract_pdf
from shippid.extract.symbols_match import get_library
from shippid.model import ActuatorType, Category, Medium, PIDDocument, PipeKind, SystemType, TextRole
from shippid.pdf.reader import read_page
from shippid.symbols import Placement, draw_symbol, get_symbol

U = 18.0


# --------------------------------------------------------------------------- reader


def _page(width: float = 400, height: float = 300):
    doc = pymupdf.open()
    return doc, doc.new_page(width=width, height=height)


def test_reader_segments_circles_text():
    doc, page = _page()
    sh = page.new_shape()
    draw_symbol(sh, get_symbol(S.GLOBE_VALVE), Placement(100, 100, scale=20), ActuatorType.ELECTRIC)
    sh.draw_polyline([pymupdf.Point(10, 200), pymupdf.Point(150, 200)])
    sh.finish(width=1.0, color=(0, 0, 0), closePath=False, dashes="[3 2] 0")
    sh.commit()
    page.insert_text((300, 250), "250A-BA-012", fontsize=7, rotate=90)
    page.insert_text((300, 100), "PI 1012", fontsize=7)
    pc = read_page(page)
    # filled centre dot and electric actuator circle come back as circles, not segments
    assert sorted((round(c.r, 1), c.filled) for c in pc.circles) == [(2.0, True), (3.0, False)]
    closed = [s for s in pc.segments if s.closed]
    assert len(closed) == 6  # two triangles (closing edge included)
    assert any(s.dashed and s.length > 100 for s in pc.segments)
    words = {w.text: w for w in pc.words}
    assert words["250A-BA-012"].rotation == 90
    assert words["PI"].rotation == 0 and words["1012"].bbox[0] > words["PI"].bbox[2]
    assert pc.readable and pc.image_coverage == 0.0


def test_reader_rotated_page_uses_visible_coordinates():
    doc, page = _page(400, 300)
    sh = page.new_shape()
    sh.draw_line(pymupdf.Point(10, 20), pymupdf.Point(110, 20))
    sh.finish(width=1, color=(0, 0, 0))
    sh.commit()
    page.set_rotation(90)
    pc = read_page(page)
    assert (pc.width, pc.height) == (300, 400)
    seg = pc.segments[0]
    # unrotated (10,20)-(110,20) is drawn vertically at x = 300 - 20 on the visible page
    assert {round(seg.x0), round(seg.x1)} == {280}
    assert sorted((round(seg.y0), round(seg.y1))) == [10, 110]


def test_reader_scanned_page_is_unreadable():
    doc, page = _page(300, 300)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), False)
    pix.clear_with(200)
    page.insert_image(page.rect, pixmap=pix)
    pc = read_page(page)
    assert pc.image_coverage > 0.9 and not pc.readable
    out = extract_page(page, 0)
    assert out.meta["unreadable_pages"] == [0] and not out.objects


# --------------------------------------------------------------------------- golden sheet


def _golden() -> Sheet:
    st = Style(scale=U, sym_width=0.6, pipe_width=1.0, font=6.0, dot_tees=True)
    sh = Sheet(1190, 842, st)
    o = sh.symbol
    t1 = o(S.TANK, 123, 200, stretch=(0.6, 1.0))
    t1.name = "NO.1 W.B.TK"
    sh.text(t1.bbox[0] + 2, 202, "NO.1 W.B.TK", TextRole.LABEL, t1.key, size=4.5)
    v1 = o(S.GATE_VALVE, 200, 200, actuator=A.HANDWHEEL, tag="BA-V001")
    s1 = o(S.STRAINER, 260, 200)
    p1 = o(S.PUMP_CENTRIFUGAL, 330, 200)
    cv1 = o(S.CHECK_VALVE, 330, 160, rotation=270)
    # one process line drawn straight through the gate valve and the strainer
    sh.line([(t1.bbox[2], 200), p1.ports["suction"]])
    sh.edges += [(("o", t1.key), ("o", v1.key)), (("o", v1.key), ("o", s1.key)), (("o", s1.key), ("o", p1.key))]
    sh.text(145, 196, "150A-BA-001", TextRole.LINE_NUMBER, size=5)
    sh.connect(("o", p1.key), ("o", cv1.key), [p1.ports["discharge"], cv1.ports["in"]])
    # discharge header with an instrument tap (plain tee) and a dotted tee
    v4 = o(S.GLOBE_VALVE, 480, 120, actuator=A.DIAPHRAGM, tag="TCV-102", tag_pos="below")
    jt = sh.junction(380, 120, dot=False)
    j1 = sh.junction(420, 120, dot=True)
    sh.line([cv1.ports["out"], (330, 120), (380, 120)])
    sh.line([(380, 120), (420, 120)])
    sh.line([(420, 120), v4.ports["in"]])
    sh.edges += [(("o", cv1.key), ("j", jt)), (("j", jt), ("j", j1)), (("j", j1), ("o", v4.key))]
    pi = o(S.INSTRUMENT_LOCAL, 380, 80)
    pi.tag = "PI-101"
    sh.text(375, 79, "PI", TextRole.INSTRUMENT, pi.key, size=5)
    sh.text(373.5, 86, "101", TextRole.INSTRUMENT, pi.key, size=5)
    sh.connect(("j", jt), ("o", pi.key), [(380, 120), (380, 80 + 0.4 * U)])
    v2 = o(S.BUTTERFLY_VALVE, 560, 120, mirrored=True, actuator=A.HYDRAULIC, tag="BA-V002")
    t2 = o(S.TANK, 640, 120, stretch=(0.8, 0.8))
    sh.connect(("o", v4.key), ("o", v2.key), [v4.ports["out"], v2.ports["in"]])
    sh.connect(("o", v2.key), ("o", t2.key), [v2.ports["out"], (t2.bbox[0], 120)])
    # control loop: shared display -> dashed signal -> diaphragm actuator
    tic = o(S.INSTRUMENT_SHARED, 480, 60)
    tic.tag = "TIC-102"
    sh.text(474, 59, "TIC", TextRole.INSTRUMENT, tic.key, size=5)
    sh.text(473.5, 66, "102", TextRole.INSTRUMENT, tic.key, size=5)
    sh.connect(("o", tic.key), ("o", v4.key), [(480, 60 + 0.4 * U), (480, v4.bbox[1])], dashed=True)
    # vertical branch from the dotted tee: ball valve (rotated) to an off-page connector
    v3 = o(S.BALL_VALVE, 420, 300, rotation=90, tag="BA-V003")
    c1 = o(S.OFF_PAGE_CONNECTOR, 420, 364.4, rotation=90)
    c1.name = "TO BILGE TK"
    sh.connect(("j", j1), ("o", v3.key), [(420, 120), v3.ports["in"]])
    sh.connect(("o", v3.key), ("o", c1.key), [v3.ports["out"], c1.ports["in"]])
    sh.text(416, 262, "100A-BA-003-1S", TextRole.LINE_NUMBER, rotate=90, size=5)
    sh.text(424, 362, "B12", TextRole.CONNECTOR, c1.key, size=4.0, rotate=90)
    sh.text(430, 366, "TO BILGE TK", TextRole.CONNECTOR, c1.key, size=5)
    # a plain crossing (no dot): T3 -> engine passes the vertical branch without connecting
    t3 = o(S.TANK, 300, 250, stretch=(0.6, 0.5))
    e1 = o(S.ENGINE, 520, 250, stretch=(0.5, 0.5))
    sh.connect(("o", t3.key), ("o", e1.key), [(t3.bbox[2], 250), (e1.bbox[0], 250)])
    sh.title_block("BALLAST WATER SYSTEM", "H2310-BA-001", "H.2310", "300K VLCC")
    sh.notes(40, 760, ["ALL VALVES ARE SHOWN IN NORMAL POSITION.", "PIPE SIZES IN MM (JIS A)."])
    sh.finish()
    sh.keys = {k: v.key for k, v in dict(t1=t1, v1=v1, s1=s1, p1=p1, cv1=cv1, v4=v4, pi=pi, v2=v2, t2=t2, tic=tic,
                                         v3=v3, c1=c1, t3=t3, e1=e1).items()}
    return sh


@pytest.fixture(scope="module")
def golden(tmp_path_factory):
    sh = _golden()
    path = tmp_path_factory.mktemp("golden") / "golden.pdf"
    sh.save(path)
    doc = extract_pdf(path)
    # GT key -> extracted object (best IoU)
    byk = {}
    for k, key in sh.keys.items():
        g = sh.objects[key]
        best = max(doc.objects, key=lambda o: _iou(o.bbox.as_tuple(), g.bbox))
        assert _iou(best.bbox.as_tuple(), g.bbox) >= 0.6, k
        byk[k] = best
    return sh, doc, byk


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def test_golden_objects(golden):
    sh, doc, o = golden
    assert len(doc.objects) == len(sh.objects)
    expect = {
        "t1": (S.TANK, None), "v1": (S.GATE_VALVE, "BA-V001"), "s1": (S.STRAINER, None),
        "p1": (S.PUMP_CENTRIFUGAL, None), "cv1": (S.CHECK_VALVE, None), "v4": (S.GLOBE_VALVE, "TCV-102"),
        "pi": (S.INSTRUMENT_LOCAL, "PI-101"), "v2": (S.BUTTERFLY_VALVE, "BA-V002"), "t2": (S.TANK, None),
        "tic": (S.INSTRUMENT_SHARED, "TIC-102"), "v3": (S.BALL_VALVE, "BA-V003"), "c1": (S.OFF_PAGE_CONNECTOR, None),
        "t3": (S.TANK, None), "e1": (S.ENGINE, None),
    }
    for k, (cls, tag) in expect.items():
        assert o[k].cls == cls, k
        assert o[k].tag == tag, (k, o[k].tag)
        assert o[k].source == "vector" and 0.6 <= o[k].confidence <= 1.0
        assert o[k].attributes["candidates"][0]["cls"] == cls.value
        g = sh.objects[sh.keys[k]]
        assert same_orientation(cls, (g.placement.rotation, g.placement.mirrored), (o[k].rotation, o[k].mirrored), g.actuator), k
    assert o["cv1"].rotation == 270  # check valve: flow direction is visible, rotation unique
    assert o["v1"].actuator == ActuatorType.HANDWHEEL
    assert o["v4"].actuator == ActuatorType.DIAPHRAGM
    assert o["v2"].actuator == ActuatorType.HYDRAULIC and o["v2"].mirrored
    assert o["t1"].attributes["name"] == "NO.1 W.B.TK"
    assert o["c1"].attributes == {**o["c1"].attributes, "name": "TO BILGE TK", "ref": "B12"}
    assert {p.name for p in o["p1"].ports} == {"suction", "discharge"}
    assert {p.name for p in o["cv1"].ports} == {"in", "out"}
    assert {p.name for p in o["v4"].ports} >= {"in", "out", "actuator"}


def _neighbours(doc: PIDDocument, oid: str, kinds=(PipeKind.PROCESS, PipeKind.INSTRUMENT_TAP)) -> set[str]:
    """Objects reachable from ``oid`` through pipes of the given kinds and junctions only."""
    adj: dict[str, set[str]] = {}
    for p in doc.pipes:
        if p.kind not in kinds:
            continue
        a, b = p.start.node_id or p.id + "s", p.end.node_id or p.id + "e"
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    objs = {x.id for x in doc.objects}
    seen, stack, out = {oid}, [oid], set()
    while stack:
        n = stack.pop()
        for m in adj.get(n, ()):
            if m in seen:
                continue
            seen.add(m)
            if m in objs:
                out.add(m)
            else:
                stack.append(m)
    return out


def test_golden_connectivity(golden):
    sh, doc, o = golden
    ids = {k: v.id for k, v in o.items()}
    assert _neighbours(doc, ids["v1"]) == {ids["t1"], ids["s1"]}  # line drawn through the valve is cut
    assert _neighbours(doc, ids["s1"]) == {ids["v1"], ids["p1"]}
    assert _neighbours(doc, ids["cv1"]) == {ids["p1"], ids["pi"], ids["v4"], ids["v3"]}  # tap tee + dotted tee
    assert _neighbours(doc, ids["v3"]) == {ids["cv1"], ids["pi"], ids["v4"], ids["c1"]}
    assert _neighbours(doc, ids["t3"]) == {ids["e1"]}  # plain crossing: not connected to the branch
    assert _neighbours(doc, ids["v2"]) == {ids["v4"], ids["t2"]}
    sig = [p for p in doc.pipes if p.kind == PipeKind.SIGNAL]
    assert len(sig) == 1
    ends = {(e.object_id, e.port) for e in (sig[0].start, sig[0].end)}
    assert (ids["v4"], "actuator") in ends and any(e[0] == ids["tic"] for e in ends)
    taps = [p for p in doc.pipes if p.kind == PipeKind.INSTRUMENT_TAP]
    assert len(taps) == 1 and ids["pi"] in {taps[0].start.object_id, taps[0].end.object_id}
    dotted = [j for j in doc.junctions if j.dotted]
    assert len(dotted) == 1 and abs(dotted[0].position.x - 420) < 1 and abs(dotted[0].position.y - 120) < 1
    assert len(doc.junctions) == 2  # dotted tee + tap tee; the plain crossing creates no junction
    # the same through the generic evaluator (object adjacency P/R, text roles)
    r = evaluate(sh, doc)
    assert r.conn_precision == 1.0 and r.conn_recall == 1.0, r.errors


def test_golden_flow_and_ports(golden):
    sh, doc, o = golden
    ids = {k: v.id for k, v in o.items()}
    run = next(p for p in doc.pipes if {p.start.object_id, p.end.object_id} == {ids["p1"], ids["cv1"]})
    first = run.points[0] if run.flow_direction == 1 else run.points[-1]
    last = run.points[-1] if run.flow_direction == 1 else run.points[0]
    assert run.flow_direction is not None and last.y < first.y  # pump discharges upwards into the check valve
    suction = next(p for p in o["p1"].ports if p.name == "suction")
    assert suction.position.x == pytest.approx(330 - 9, abs=0.6) and suction.direction.x < -0.9


def test_golden_texts_and_info(golden):
    sh, doc, o = golden
    by_text = {t.text: t for t in doc.texts}
    pipes = {p.id: p for p in doc.pipes}
    ids = {k: v.id for k, v in o.items()}
    assert by_text["BA-V001"].role == TextRole.TAG and by_text["BA-V001"].owner_id == ids["v1"]
    assert by_text["BA-V002"].owner_id == ids["v2"] and by_text["TCV-102"].owner_id == ids["v4"]
    assert by_text["PI"].role == TextRole.INSTRUMENT and by_text["PI"].owner_id == ids["pi"]
    assert by_text["B12"].role == TextRole.CONNECTOR and by_text["TO BILGE TK"].role == TextRole.CONNECTOR
    ln = by_text["150A-BA-001"]
    assert ln.role == TextRole.LINE_NUMBER
    owner = pipes[ln.owner_id]
    assert {owner.start.object_id, owner.end.object_id} == {ids["t1"], ids["v1"]}
    assert owner.size == "150A" and owner.medium == Medium.BALLAST_WATER
    vln = by_text["100A-BA-003-1S"]
    assert vln.rotation == 90 and vln.role == TextRole.LINE_NUMBER
    vowner = pipes[vln.owner_id]
    assert all(abs(p.x - 420) < 1 for p in vowner.points)  # the vertical pipe beside the rotated text
    # propagated across the inline ball valve to the run towards the connector
    after = next(p for p in doc.pipes if {p.start.object_id, p.end.object_id} == {ids["v3"], ids["c1"]})
    assert after.line_number == "100A-BA-003-1S" and not after.text_ids
    assert by_text["NOTES:"].role == TextRole.NOTE
    assert by_text["H2310-BA-001"].role == TextRole.TITLE_BLOCK
    assert by_text["1"].role == TextRole.TITLE_BLOCK  # zone marker in the frame margin
    info = doc.info
    assert info.drawing_no == "H2310-BA-001" and info.hull_no == "H.2310" and info.ship_type == "300K VLCC"
    assert info.title == "BALLAST WATER SYSTEM" and info.system == SystemType.BALLAST
    assert doc.pages[0].drawing_area is not None
    r = evaluate(sh, doc)
    assert r.role_ok == r.role_n, [e for e in r.errors if e[0] == "role"]


def test_golden_document_contract(golden, tmp_path):
    sh, doc, o = golden
    assert doc.meta["extractor"] == "vector" and doc.source.endswith("golden.pdf")
    assert [p.index for p in doc.pages] == [0]
    assert [x.id for x in doc.objects] == [f"o{i + 1}" for i in range(len(doc.objects))]
    assert [x.id for x in doc.texts] == [f"t{i + 1}" for i in range(len(doc.texts))]
    known = {x.id for x in doc.objects} | {x.id for x in doc.pipes}
    assert all(t.owner_id in known for t in doc.texts if t.owner_id)
    junctions = {j.id for j in doc.junctions}
    for p in doc.pipes:
        for e in (p.start, p.end):
            assert not (e.object_id and e.junction_id)
            assert e.junction_id is None or e.junction_id in junctions
            assert (e.object_id is None) == (e.port is None)
    assert PIDDocument.load(doc.save(tmp_path / "x.json")) == doc


# --------------------------------------------------------------------------- crossings and junctions


def _cross_page(style: str, dot: bool) -> tuple[Sheet, PIDDocument]:
    st = Style(scale=U, crossing=style)
    sh = Sheet(500, 400, st, frame=False)
    a = sh.symbol(S.TANK, 80, 200, stretch=(0.5, 0.5))
    b = sh.symbol(S.TANK, 420, 200, stretch=(0.5, 0.5))
    c = sh.symbol(S.TANK, 250, 60, stretch=(0.5, 0.5))
    d = sh.symbol(S.TANK, 250, 340, stretch=(0.5, 0.5))
    sh.line([(a.bbox[2], 200), (b.bbox[0], 200)])
    if dot:
        sh.line([(250, c.bbox[3]), (250, d.bbox[1])])
        sh.junction(250, 200, dot=True)
    else:
        sh.crossing_line(((250, c.bbox[3]), (250, d.bbox[1])), (250, 200), horizontal=False)
    sh.finish()
    return sh, extract_page(sh.page, 0)


@pytest.mark.parametrize("style", ["plain", "gap", "hop"])
def test_crossing_without_dot_is_not_connected(style):
    sh, doc = _cross_page(style, dot=False)
    assert len(doc.objects) == 4 and all(o.cls == S.TANK for o in doc.objects)
    assert not doc.junctions
    procs = [p for p in doc.pipes if p.kind == PipeKind.PROCESS]
    assert len(procs) == 2 and all(p.start.object_id and p.end.object_id for p in procs)
    if style != "plain":
        assert any(p.length > 300 for p in procs)  # the broken / hopping line is one run again


def test_crossing_with_dot_is_connected():
    sh, doc = _cross_page("plain", dot=True)
    assert len(doc.junctions) == 1 and doc.junctions[0].kind == "cross" and doc.junctions[0].dotted
    assert len(doc.pipes) == 4 and all(p.end.junction_id or p.start.junction_id for p in doc.pipes)


# --------------------------------------------------------------------------- robustness


def test_exploded_geometry_and_flange_to_flange():
    """CAD exports that explode polylines (every edge its own path) and symbols bolted
    directly to each other without a pipe in between."""
    st = Style(scale=U, explode=True)
    sh = Sheet(600, 300, st, frame=False)
    t = sh.symbol(S.TANK, 70, 150, stretch=(0.6, 0.8))
    v = sh.symbol(S.GATE_VALVE, 170, 150, actuator=A.HANDWHEEL)
    s = sh.symbol(S.STRAINER, v.ports["out"][0] + 0.35 * U, 150)  # flange to flange with the valve
    f = sh.symbol(S.FILTER, s.ports["out"][0] + 0.4 * U, 150)  # and the filter with the strainer
    c = sh.symbol(S.CHECK_VALVE, 400, 150)
    sh.line([(t.bbox[2], 150), v.ports["in"]])
    sh.line([f.ports["out"], c.ports["in"]])
    sh.finish()
    doc = extract_page(sh.page, 0)
    assert sorted(o.cls.value for o in doc.objects) == sorted(
        ["tank", "gate_valve", "strainer", "filter", "check_valve"])
    ids = {o.cls: o.id for o in doc.objects}
    # direct (zero-length) connections between bolted symbols
    direct = [p for p in doc.pipes if p.length == 0]
    pairs = {frozenset((p.start.object_id, p.end.object_id)) for p in direct}
    assert frozenset((ids[S.GATE_VALVE], ids[S.STRAINER])) in pairs
    assert frozenset((ids[S.STRAINER], ids[S.FILTER])) in pairs
    gv = next(o for o in doc.objects if o.cls == S.GATE_VALVE)
    assert gv.actuator == ActuatorType.HANDWHEEL


def test_unknown_symbol_keeps_candidates():
    sh = Sheet(300, 200, Style(scale=U), frame=False)
    shape = sh.shape
    shape.draw_polyline([pymupdf.Point(130, 90), pymupdf.Point(150, 80), pymupdf.Point(170, 90),
                         pymupdf.Point(150, 120), pymupdf.Point(130, 90)])
    shape.finish(width=0.6, color=(0, 0, 0), closePath=True)
    sh.line([(40, 100), (130, 100)])
    sh.finish()
    doc = extract_page(sh.page, 0)
    unk = [o for o in doc.objects if o.cls == S.UNKNOWN]
    assert len(unk) == 1
    assert unk[0].confidence < ExtractOptions().accept_score
    assert 1 <= len(unk[0].attributes["candidates"]) <= 3
    assert unk[0].category in set(Category)


def test_multi_page_ids_continue(tmp_path):
    a, b = random_sheet(3, n_rows=3), random_sheet(4, n_rows=3)
    pdf = pymupdf.open()
    pdf.insert_pdf(a.doc)
    pdf.insert_pdf(b.doc)
    path = tmp_path / "two.pdf"
    pdf.save(path)
    doc = extract_pdf(path)
    assert [p.index for p in doc.pages] == [0, 1]
    assert {o.page for o in doc.objects} == {0, 1}
    ids = [o.id for o in doc.objects]
    assert len(ids) == len(set(ids)) == len(a.objects) + len(b.objects)
    only_second = extract_pdf(path, pages=[1])
    assert {o.page for o in only_second.objects} == {1}
    with pytest.raises(IndexError):
        extract_pdf(path, pages=[5])


# --------------------------------------------------------------------------- measured quality


@pytest.mark.parametrize("hard", [False, True])
def test_quality_random_sheets(hard):
    """Per-class accuracy, connectivity and text roles on randomised sheets (normal and
    hard: vertical trains, exploded geometry, flange-to-flange symbols, coloured pipes,
    flow arrows, exploded dashes, dotted crosses)."""
    from extract_fixtures import EvalResult

    tot = EvalResult()
    for seed in range(6):
        sh = random_sheet(seed, hard=hard)
        tot.add(evaluate(sh, extract_page(sh.page, 0)))
    s = tot.summary()
    assert s["recall@iou0.5"] >= 0.98 and s["precision@iou0.5"] >= 0.98, s
    assert s["class_acc"] >= 0.98 and s["orientation_acc"] >= 0.97 and s["actuator_acc"] >= 0.97, s
    assert s["tag_acc"] >= 0.97 and s["text_role_acc"] >= 0.97, s
    assert s["conn_precision"] >= 0.97 and s["conn_recall"] >= 0.97, s


def test_distorted_symbol_proportions():
    """Yard symbol styles differ in proportions: +-10 % per axis must still classify."""
    import random

    from extract_fixtures import EvalResult, random_style

    tot = EvalResult()
    for seed in range(3):
        st = random_style(random.Random(seed))
        st.distort = 0.1
        sh = random_sheet(seed, style=st)
        tot.add(evaluate(sh, extract_page(sh.page, 0)))
    assert tot.cls_acc >= 0.95, tot.summary()


def test_performance_a1_sheet():
    """An A1 sheet with ~350 symbols and ~500 pipe segments extracts in < 5 s."""
    get_library()  # one-off template build (~0.5 s), not part of the per-page budget
    sh = random_sheet(2, n_rows=14, width=2384, height=1684, hard=True, n_el=(16, 20))
    best = math.inf
    for _ in range(2):  # the second attempt only guards against a busy machine
        t = time.perf_counter()
        doc = extract_page(sh.page, 0, Config())
        best = min(best, time.perf_counter() - t)
        if best < 5.0:
            break
    segs = sum(len(p.points) - 1 for p in doc.pipes)
    assert len(sh.objects) >= 300 and segs >= 450
    assert best < 5.0, f"{best:.2f} s"
    r = evaluate(sh, doc)
    assert r.recall >= 0.98 and r.cls_acc >= 0.98
