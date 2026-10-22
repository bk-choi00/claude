"""Extractor hardening for CAD export conventions and unseen symbol styles: fragmented and
invisible text, AutoCAD SHX text (stroke glyphs with PDFSHX=1 annotations or PDFSHX=2
invisible text), tessellated circles and arcs, and confidence calibration (unexplained
ink, off-radius pipe attachments, double-circle bubbles).

Fixtures are drawn directly with PyMuPDF (``extract_fixtures.py``); nothing here depends
on :mod:`shippid.datasets`.
"""

from __future__ import annotations

import math

import pymupdf
import pytest

from extract_fixtures import A, S, Sheet, Style
from shippid.analysis.gemma import GemmaAnalyzer
from shippid.extract.pipeline import extract_page
from shippid.extract.symbols_match import REVIEW_BAND, ink_penalty, review_band
from shippid.extract.textlines import build_lines
from shippid.llm.base import LLMClient, MockBackend
from shippid.model import Category, PipeKind, TextRole
from shippid.pdf.reader import _arc_runs, read_page
from shippid.symbols import Placement

U = 18.0
BLACK = (0.0, 0.0, 0.0)


def _reopen(doc: pymupdf.Document) -> pymupdf.Page:
    return pymupdf.open("pdf", doc.tobytes())[0]


def _analyzer() -> GemmaAnalyzer:
    return GemmaAnalyzer(LLMClient(MockBackend(lambda *a: {})))


def _texts(doc) -> list[str]:
    return sorted(t.text for t in doc.texts)


# --------------------------------------------------------------------------- reader: invisible text


def test_reader_invisible_text_flags():
    d = pymupdf.open()
    p = d.new_page(width=400, height=300)
    p.insert_text((50, 50), "VIS-1", fontsize=10)
    p.insert_text((50, 80), "HID-2", fontsize=10, render_mode=3)  # neither filled nor stroked
    p.insert_text((50, 110), "STR-3", fontsize=10, render_mode=1)  # stroked outline text is visible
    p.insert_text((50, 140), "ALPHA-4", fontsize=10, fill_opacity=0)
    pc = read_page(_reopen(d))
    hidden = {w.text: w.hidden for w in pc.words}
    assert hidden == {"VIS-1": False, "HID-2": True, "STR-3": False, "ALPHA-4": True}
    assert pc.n_chars == 10 and pc.n_hidden_chars == 12


def test_reader_drops_invisible_text_on_layers_that_are_off():
    """MuPDF drops visible OFF-layer text from get_text but still reports invisible text
    there; the reader must not pick it up."""
    d = pymupdf.open()
    p = d.new_page(width=400, height=300)
    off, on = d.add_ocg("DIM", on=False), d.add_ocg("TEXT", on=True)
    sh = p.new_shape()
    sh.insert_text((50, 50), "4700", fontsize=8, render_mode=3, oc=off)
    sh.insert_text((50, 80), "V-101", fontsize=8, render_mode=3, oc=on)
    sh.insert_text((50, 110), "4800", fontsize=8, oc=off)
    sh.commit()
    pc = read_page(_reopen(d))
    assert [w.text for w in pc.words] == ["V-101"]


# --------------------------------------------------------------------------- reader: SHX annotations


@pytest.mark.parametrize("page_rot", [0, 90])
def test_reader_shx_annotations(page_rot):
    d = pymupdf.open()
    p = d.new_page(width=600, height=400)
    a = p.add_freetext_annot(pymupdf.Rect(100, 50, 140, 59), "V-101", fontsize=7, rotate=0)
    b = p.add_freetext_annot(pymupdf.Rect(200, 50, 209, 110), "250A-FO-001", fontsize=7, rotate=90)
    c = p.add_text_annot((300, 200), "TK-1")
    other = p.add_freetext_annot(pymupdf.Rect(400, 300, 440, 310), "REVIEW ME", fontsize=7)
    off_ocg = d.add_ocg("OLD", on=False)
    e = p.add_freetext_annot(pymupdf.Rect(100, 300, 140, 309), "V-999", fontsize=7)
    for an, subj in ((a, "AutoCAD SHX Text"), (b, "AutoCAD SHX Text"), (c, "AutoCAD SHX Text"), (other, "Comment"),
                     (e, "AutoCAD SHX Text")):
        an.set_info(subject=subj)
        an.set_flags(pymupdf.PDF_ANNOT_IS_HIDDEN)
        an.update()
    e.set_oc(off_ocg)
    e.update()
    d.xref_set_key(c.xref, "Rect", "[300 190 330 199]")  # PDF space (y up): page y 201 .. 210
    if page_rot:
        p.set_rotation(page_rot)
    page = _reopen(d)
    pc = read_page(page)
    words = {w.text: w for w in pc.words}
    assert set(words) == {"V-101", "250A-FO-001", "TK-1"}  # not the plain comment, not the OFF layer
    assert all(w.annotation and not w.hidden for w in words.values())
    assert pc.n_annot_chars == len("V-101") + len("250A-FO-001") + len("TK-1")
    m = page.rotation_matrix
    for text, rect in (("V-101", (100, 50, 140, 59)), ("250A-FO-001", (200, 50, 209, 110)), ("TK-1", (300, 201, 330, 210))):
        r = pymupdf.Rect(rect) * m
        r.normalize()
        assert words[text].bbox == pytest.approx((r.x0, r.y0, r.x1, r.y1), abs=0.01), text
    if page_rot == 0:
        assert words["V-101"].rotation == 0 and words["250A-FO-001"].rotation == 90
        assert words["V-101"].size == pytest.approx(7.0)
    else:  # horizontal annotation text reads downwards on the rotated page
        assert words["V-101"].rotation in (90, 270) and words["250A-FO-001"].rotation == 0
    lines = build_lines(pc.words)
    assert {t.source for t in lines} == {"annotation"}


# --------------------------------------------------------------------------- reader: tessellated curves


def _arc_pts(cx, cy, r, a0, a1, n):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)), cy - r * math.sin(math.radians(a0 + (a1 - a0) * k / n)))
            for k in range(n + 1)]


def test_arc_runs_find_arcs_not_waves_or_zigzags():
    def items(pts):
        return [("l", [a, b], -1) for a, b in zip(pts, pts[1:])]

    hop = [(0.0, 100.0)] + _arc_pts(50, 100, 4, 180, 0, 12) + [(100.0, 100.0)]  # pipe, 12-piece hop, pipe
    runs = _arc_runs(items(hop))
    assert len(runs) == 1 and runs[0][:2] == (1, 13) and runs[0][2] == pytest.approx(180, abs=1)
    zig = [(x * 3.0, 100 + (2.0 if x % 2 else -2.0)) for x in range(12)]
    assert _arc_runs(items(zig)) == []
    wave = [(x * 2.0, 100 + 3 * math.sin(x * 2 * math.pi / 8)) for x in range(25)]  # 3 periods, 8 points each
    assert _arc_runs(items(wave)) == []
    square = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    assert _arc_runs(items(square)) == []


def test_reader_tessellated_circles_and_hops():
    d = pymupdf.open()
    p = d.new_page(width=400, height=300)
    sh = p.new_shape()
    # closed 24-gon, an unclosed 36-piece circle, a hop arc inside a pipe polyline, a sine wave
    sh.draw_polyline(_arc_pts(100, 100, 7, 0, 360, 24))
    sh.finish(width=0.5, color=BLACK, closePath=False)
    sh.draw_polyline(_arc_pts(200, 100, 7, 0, 350, 35))
    sh.finish(width=0.5, color=BLACK, closePath=False)
    sh.draw_polyline([(20, 200), (146, 200)] + _arc_pts(150, 200, 4, 180, 0, 10) + [(300, 200)])
    sh.finish(width=1.0, color=BLACK, closePath=False)
    sh.draw_polyline([(250 + x * 2.0, 100 + 3 * math.sin(x * 2 * math.pi / 8)) for x in range(25)])
    sh.finish(width=0.5, color=BLACK, closePath=False)
    sh.commit()
    pc = read_page(_reopen(d))
    assert sorted((round(c.cx), round(c.cy), round(c.r)) for c in pc.circles) == [(100, 100, 7), (200, 100, 7)]
    hop = [s for s in pc.segments if 140 < (s.x0 + s.x1) / 2 < 160 and s.y0 < 201]
    assert len(hop) == 10 and len({s.curve for s in hop}) == 1 and hop[0].curve >= 0
    straight = [s for s in pc.segments if s.length > 50]
    assert len(straight) == 2 and all(s.curve < 0 for s in straight)
    assert not [s for s in pc.segments if s.x0 >= 250 and s.curve >= 0]  # the wave stays straight pieces
    assert pc.n_arcs == 2


def test_tessellated_hop_is_a_crossing_not_a_symbol():
    sh = Sheet(700, 500, Style(scale=U), frame=False)
    v1 = sh.symbol(S.GATE_VALVE, 150, 200, actuator=A.HANDWHEEL)
    v2 = sh.symbol(S.GLOBE_VALVE, 500, 200, actuator=A.HANDWHEEL)
    v3 = sh.symbol(S.GATE_VALVE, 330, 380, rotation=90, actuator=A.HANDWHEEL)
    t1 = sh.symbol(S.TANK, 40, 200, stretch=(0.5, 0.6))
    t2 = sh.symbol(S.TANK, 640, 200, stretch=(0.5, 0.6))
    sh.line([(t1.bbox[2], 200), v1.ports["in"]])
    g = 4.0
    pts = [v1.ports["out"], (330 - g, 200)] + _arc_pts(330, 200, g, 180, 0, 12)[1:-1] + [(330 + g, 200), v2.ports["in"]]
    sh.line(pts)
    sh.line([v2.ports["out"], (t2.bbox[0], 200)])
    sh.line([(330, 60), v3.ports["out"]])  # vertical line crossing under the hop
    sh.finish()
    doc = extract_page(_reopen(sh.doc), 0)
    assert {o.cls for o in doc.objects} == {S.TANK, S.GATE_VALVE, S.GLOBE_VALVE}
    assert not doc.junctions  # a hop is a crossing without connection
    assert doc.meta["pages"]["0"]["tessellated_arcs"] == 1


# --------------------------------------------------------------------------- text lines


def _fragments(page, x, y, text, font="helv", size=7.0, mode="char", rotate=0):
    """Insert ``text`` per character or per word at explicit positions (CAD text fragments)."""
    if mode == "char":
        parts = [(i, ch) for i, ch in enumerate(text) if not ch.isspace()]
    else:
        parts, i = [], 0
        for w in text.split(" "):
            parts.append((i, w))
            i += len(w) + 1
    for i, piece in parts:
        off = pymupdf.get_text_length(text[:i], fontname=font, fontsize=size)
        pt = (x + off, y) if rotate == 0 else (x, y - off)
        page.insert_text(pt, piece, fontsize=size, fontname=font, rotate=rotate)


def test_fragmented_text_becomes_logical_tokens():
    d = pymupdf.open()
    p = d.new_page(width=500, height=400)
    _fragments(p, 50, 50, "FO-101V", mode="char")
    _fragments(p, 50, 80, "NO.1 BALLAST PUMP", font="cour", mode="word")  # Courier space = 0.6 em
    _fragments(p, 50, 110, "250A-BA-012", font="tiro", mode="char", rotate=90)
    _fragments(p, 200, 50, "V-101", mode="word")
    _fragments(p, 200 + pymupdf.get_text_length("V-101", fontsize=7) + 14, 50, "V-102", mode="char")  # 2 em apart
    _fragments(p, 50, 200, "1번 평형수 펌프", font="korea", mode="word")  # full-width spaces
    lines = build_lines(read_page(_reopen(d)).words)
    assert sorted(t.text for t in lines) == sorted(["FO-101V", "NO.1 BALLAST PUMP", "250A-BA-012", "V-101", "V-102",
                                                    "1번 평형수 펌프"])
    assert next(t for t in lines if t.text == "250A-BA-012").rotation == 90


def test_table_rule_keeps_cell_texts_apart():
    d = pymupdf.open()
    p = d.new_page(width=400, height=300)
    w = pymupdf.get_text_length("DATE", fontsize=7)
    _fragments(p, 50, 50, "DATE", mode="char")
    _fragments(p, 50 + w + 3.5, 50, "2026", mode="char")  # 0.5 em gap, a cell border in between
    sh = p.new_shape()
    sh.draw_line((50 + w + 1.75, 38), (50 + w + 1.75, 56))
    sh.finish(width=0.5, color=BLACK)
    sh.commit()
    page = _reopen(d)
    pc = read_page(page)
    rules = [(s.x0, s.y0, s.x1, s.y1) for s in pc.segments]
    assert sorted(t.text for t in build_lines(pc.words, separators=rules)) == ["2026", "DATE"]
    assert [t.text for t in build_lines(pc.words)] == ["DATE 2026"]  # without the rule: one token


def test_invisible_duplicates_dropped_invisible_only_used():
    d = pymupdf.open()
    p = d.new_page(width=400, height=300)
    _fragments(p, 50, 50, "NO.2 F.O. PUMP", font="tiro", mode="word")
    p.insert_text((50, 50), "NO.2 F.O. PUMP", fontsize=7, render_mode=3)  # PDFSHX=2 style copy
    _fragments(p, 50, 90, "AB C", font="cour", mode="word")
    p.insert_text((50, 90), "AB C", fontsize=7, fontname="helv", render_mode=3)
    p.insert_text((200, 150), "PI-101", fontsize=7, render_mode=3)  # nothing visible there: the text layer
    lines = build_lines(read_page(_reopen(d)).words)
    assert sorted((t.text, t.source) for t in lines) == [("AB C", "text"), ("NO.2 F.O. PUMP", "text"),
                                                         ("PI-101", "hidden")]
    assert next(t for t in lines if t.text == "PI-101").geometric


# --------------------------------------------------------------------------- SHX stroke text

_GLYPHS = {  # tiny stroke font (x 0..1, y 0..1 up from the baseline)
    "O": [[(0.5 + 0.5 * math.cos(math.radians(a)), 0.5 + 0.5 * math.sin(math.radians(a))) for a in range(90, 451, 24)]],
    "0": [[(0.5 + 0.4 * math.cos(math.radians(a)), 0.5 + 0.5 * math.sin(math.radians(a))) for a in range(90, 451, 24)]],
    "I": [[(0.5, 0.0), (0.5, 1.0)]],
    "1": [[(0.2, 0.8), (0.55, 1.0), (0.55, 0.0)]],
    "-": [[(0.1, 0.5), (0.9, 0.5)]],
    "L": [[(0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]],
    "T": [[(0.0, 1.0), (1.0, 1.0)], [(0.5, 1.0), (0.5, 0.0)]],
    "P": [[(0.0, 0.0), (0.0, 1.0), (0.75, 1.0), (1.0, 0.85), (1.0, 0.65), (0.75, 0.5), (0.0, 0.5)]],
    "V": [[(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)]],
    "2": [[(0.0, 0.8), (0.2, 1.0), (0.8, 1.0), (1.0, 0.8), (1.0, 0.6), (0.0, 0.0), (1.0, 0.0)]],
}


def _shx(sh: Sheet, x: float, y: float, text: str, size: float, carrier: str, rotate: int = 0) -> None:
    """Text drawn as strokes in the cells of Helvetica advances, with its PDFSHX carrier."""
    shape = sh.shape
    d, up = ((1.0, 0.0), (0.0, -1.0)) if rotate == 0 else ((0.0, -1.0), (-1.0, 0.0))
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        u0 = pymupdf.get_text_length(text[:i], fontsize=size)
        adv = pymupdf.get_text_length(ch, fontsize=size)
        for line in _GLYPHS[ch]:
            pts = [(u0 + 0.14 * adv + gx * 0.72 * adv, gy * 0.7 * size) for gx, gy in line]
            shape.draw_polyline([pymupdf.Point(x + u * d[0] + v * up[0], y + u * d[1] + v * up[1]) for u, v in pts])
    shape.finish(width=0.25, color=BLACK, closePath=False)
    if carrier == "hidden":
        shape.insert_text((x, y), text, fontsize=size, rotate=rotate, render_mode=3)
    else:
        w = pymupdf.get_text_length(text, fontsize=size)
        r = pymupdf.Rect(x, y - 1.075 * size, x + w, y + 0.299 * size) if rotate == 0 else \
            pymupdf.Rect(x - 1.075 * size, y - w, x + 0.299 * size, y)
        sh.annots.append((r, text, size, rotate))


@pytest.mark.parametrize("carrier", ["hidden", "annotation"])
def test_shx_stroke_text_is_text_not_symbols_or_pipes(carrier):
    sh = Sheet(700, 500, Style(scale=U), frame=False)
    sh.annots = []
    t1 = sh.symbol(S.TANK, 60, 200, stretch=(0.6, 0.8))
    v1 = sh.symbol(S.GATE_VALVE, 220, 200, actuator=A.HANDWHEEL)
    v2 = sh.symbol(S.GLOBE_VALVE, 420, 200, actuator=A.HANDWHEEL)
    t2 = sh.symbol(S.TANK, 620, 200, stretch=(0.6, 0.8))
    pi = sh.symbol(S.INSTRUMENT_LOCAL, 320, 130)
    sh.line([(t1.bbox[2], 200), v1.ports["in"]])
    sh.line([v1.ports["out"], v2.ports["in"]])
    sh.line([v2.ports["out"], (t2.bbox[0], 200)])
    sh.line([(320, 200), (320, pi.bbox[3])])  # instrument tap
    _shx(sh, 205, 182, "V-101", 6.0, carrier)
    _shx(sh, 405, 182, "V-I02", 6.0, carrier)
    _shx(sh, 317.6, 129, "PI", 5.0, carrier)
    _shx(sh, 315.8, 135.5, "100", 5.0, carrier)
    _shx(sh, 250, 214, "OIL-TOP", 6.0, carrier)  # 'O' and '0' would pass for bubbles, 'I' / 'L' / '-' for pipes
    _shx(sh, 520, 260, "100-OIL-01", 6.0, carrier, rotate=90)
    _shx(sh, 40, 250, "T-1", 7.0, carrier)
    sh.finish()
    page = sh.page
    for r, text, size, rot in sh.annots:
        a = page.add_freetext_annot(r, text, fontsize=size, rotate=rot)
        a.set_info(subject="AutoCAD SHX Text")
        a.set_flags(pymupdf.PDF_ANNOT_IS_HIDDEN)
        a.update()
    doc = extract_page(_reopen(sh.doc), 0)
    meta = doc.meta["pages"]["0"]
    assert meta["chars"] == 0 and meta["counts"]["text_strokes"] > 40
    assert sorted(o.cls for o in doc.objects) == sorted([S.TANK, S.TANK, S.GATE_VALVE, S.GLOBE_VALVE,
                                                         S.INSTRUMENT_LOCAL])
    assert _texts(doc) == sorted(["V-101", "V-I02", "PI", "100", "OIL-TOP", "100-OIL-01", "T-1"])
    process = [p for p in doc.pipes if p.kind != PipeKind.SIGNAL]
    assert len(process) == 5  # tank-V-101, V-101-tee, tee-V-I02, V-I02-tank, tee-PI (tap)
    assert all((p.start.object_id or p.start.junction_id) and (p.end.object_id or p.end.junction_id) for p in process)
    inst = next(o for o in doc.objects if o.cls == S.INSTRUMENT_LOCAL)
    assert inst.tag == "PI-100" and inst.confidence > 0.9
    tags = {o.cls: o.tag for o in doc.objects if o.category == Category.VALVE}
    assert tags == {S.GATE_VALVE: "V-101", S.GLOBE_VALVE: "V-I02"}
    roles = {t.text: t.role for t in doc.texts}
    assert roles["PI"] == TextRole.INSTRUMENT and roles["T-1"] == TextRole.TAG


# --------------------------------------------------------------------------- calibration


def test_ink_penalty_and_review_band():
    assert ink_penalty(0.95, 0.0, 0.0) == 0.95
    assert ink_penalty(0.95, 0.015, 0.015) == 0.95  # within tolerance
    lo, hi = REVIEW_BAND
    a, b = ink_penalty(0.99, 0.06, 0.0), ink_penalty(0.93, 0.06, 0.0)
    assert lo <= b < a <= hi < 0.75  # order kept, inside the review band
    assert ink_penalty(0.5, 0.2, 0.0) < 0.5
    assert review_band(1.0) == pytest.approx(hi) and review_band(lo) == pytest.approx(lo)


def _valve_line(sh: Sheet, xs, y: float):
    t1 = sh.symbol(S.TANK, 60, y, stretch=(0.5, 0.6))
    vals = []
    prev = (t1.bbox[2], y)
    for x, cls, act in xs:
        v = sh.symbol(cls, x, y, actuator=act)
        sh.line([prev, v.ports["in"]])
        prev = v.ports["out"]
        vals.append(v)
    t2 = sh.symbol(S.TANK, xs[-1][0] + 140, y, stretch=(0.5, 0.6))
    sh.line([prev, (t2.bbox[0], y)])
    return vals


def test_canonical_symbols_are_confident_and_not_flagged():
    sh = Sheet(1190, 842, Style(scale=U))
    _valve_line(sh, [(200, S.GATE_VALVE, A.HANDWHEEL), (300, S.BALL_VALVE, A.NONE), (400, S.CHECK_VALVE, A.NONE),
                     (500, S.SDNR_VALVE, A.HANDWHEEL), (600, S.GLOBE_VALVE, A.HANDWHEEL), (700, S.GATE_VALVE, A.NONE)],
                300)
    sh.finish()
    doc = extract_page(_reopen(sh.doc), 0)
    an = _analyzer()
    valves = [o for o in doc.objects if o.category == Category.VALVE]
    assert len(valves) == 6
    for o in valves:
        assert o.confidence >= 0.9 and not an.needs_review(o), (o.cls, o.attributes["candidates"])
        assert "calibration" not in o.attributes


def test_unseen_symbol_styles_go_to_review():
    """A yard check valve (bow-tie + outlet bar, no fill) matches the gate template except for
    the bar; a yard pump (circle + tangential discharge stub) looks like a bubble whose pipes
    arrive off its radius; a double-circle bubble is one instrument, not two. None of them may
    leave the extractor with high confidence."""
    sh = Sheet(1190, 842, Style(scale=U))
    vals = _valve_line(sh, [(250, S.GATE_VALVE, A.HANDWHEEL), (420, S.GATE_VALVE, A.NONE)], 250)
    yard = vals[1].placement  # add the outlet-side bar of the yard check valve
    a, b = yard.to_page(0.42, -0.25), yard.to_page(0.42, 0.25)
    sh.shape.draw_line(pymupdf.Point(*a), pymupdf.Point(*b))
    sh.shape.finish(width=sh.style.sym_width, color=BLACK)
    # yard pump: circle through the suction port, tangent to the discharge stub
    cy = math.sqrt(0.05)
    pl = Placement(600, 500, scale=U)
    c = pl.to_page(-0.3, cy)
    sh.shape.draw_circle(pymupdf.Point(*c), 0.3 * U)
    sh.shape.finish(width=sh.style.sym_width, color=BLACK)
    suction, top = pl.to_page(-0.5, 0.0), pl.to_page(0.0, cy)
    tk = sh.symbol(S.TANK, 420, 500, stretch=(0.5, 0.6))
    sh.line([(tk.bbox[2], suction[1]), suction])
    sh.line([top, (top[0], 420), (900, 420)])  # stub + discharge pipe drawn as one line
    v = sh.symbol(S.GATE_VALVE, 760, 420, actuator=A.HANDWHEEL)
    # double-circle local instrument on a tap
    sh.shape.draw_circle(pymupdf.Point(330, 160), 0.4 * U)
    sh.shape.draw_circle(pymupdf.Point(330, 160), 0.33 * U)
    sh.shape.finish(width=sh.style.sym_width, color=BLACK)
    sh.line([(330, 250), (330, 160 + 0.4 * U)])
    sh.text(324, 159, "PI", TextRole.INSTRUMENT)
    sh.text(322, 166, "101", TextRole.INSTRUMENT)
    sh.finish()
    doc = extract_page(_reopen(sh.doc), 0)
    an = _analyzer()

    def at(x, y):
        return [o for o in doc.objects if o.bbox.x0 - 2 <= x <= o.bbox.x1 + 2 and o.bbox.y0 - 2 <= y <= o.bbox.y1 + 2]

    gate, check = at(*vals[0].ports["out"]), at(*vals[1].ports["out"])
    assert len(gate) == 1 and gate[0].confidence >= 0.9 and not an.needs_review(gate[0])
    assert len(check) == 1 and check[0].confidence < 0.75 and an.needs_review(check[0])
    assert check[0].cls != S.UNKNOWN and check[0].attributes["calibration"]
    pump = at(*c)
    assert len(pump) == 1 and pump[0].confidence < 0.75 and an.needs_review(pump[0])
    bubble = at(330, 160)
    assert len(bubble) == 1 and bubble[0].cls == S.INSTRUMENT_LOCAL and bubble[0].tag == "PI-101"
    assert bubble[0].confidence < 0.75 and an.needs_review(bubble[0])
    assert v and at(760, 420)[0].confidence >= 0.9
