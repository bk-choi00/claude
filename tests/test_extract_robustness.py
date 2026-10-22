"""Robustness of the extractor against input the review found expensive or misleading:
reduced copies (frame band thinner than the snap tolerance), drawings exported far larger than
A0, damaged (repaired) PDF files, fields of loose strokes (hatching, tick marks) and lettering
drawn as strokes without any text carrier (AutoCAD PDFSHX=0).

Fixtures are drawn directly with PyMuPDF (``extract_fixtures.py``); the tests that use
:mod:`shippid.datasets` skip when it is not available.
"""

from __future__ import annotations

import math
import random

import pymupdf
import pytest

from extract_fixtures import A, S, Sheet, Style, evaluate
from shippid.extract import pipeline as P
from shippid.extract.pipeline import extract_pdf
from shippid.extract.primitives import BoxGrid, Prim, bbox_dist, estimate_unit, page_scale
from shippid.extract.regions import Regions
from shippid.model import SymbolClass

U = 18.0


def _train(sh: Sheet, x0: float = 150, y: float = 300, tags: bool = True) -> dict:
    """tank -> gate valve -> screw pump -> check valve -> tank, connected by pipes."""
    o = sh.symbol
    t1 = o(S.TANK, x0, y, stretch=(0.8, 1.0))
    v1 = o(S.GATE_VALVE, x0 + 180, y, actuator=A.HANDWHEEL, tag="FO-V001" if tags else None)
    p1 = o(S.PUMP_SCREW, x0 + 270, y)
    v2 = o(S.CHECK_VALVE, x0 + 360, y)
    t2 = o(S.TANK, x0 + 550, y)
    sh.connect(("o", t1.key), ("o", v1.key), [(t1.bbox[2], y), v1.ports["in"]])
    sh.connect(("o", v1.key), ("o", p1.key), [v1.ports["out"], p1.ports["in"]])
    sh.connect(("o", p1.key), ("o", v2.key), [p1.ports["out"], v2.ports["in"]])
    sh.connect(("o", v2.key), ("o", t2.key), [v2.ports["out"], (t2.bbox[0], y)])
    return dict(t1=t1, v1=v1, p1=p1, v2=v2, t2=t2)


def _placed(src: pymupdf.Document, k: float, page_w: float, page_h: float) -> pymupdf.Document:
    """Page 0 of ``src`` scaled by ``k`` onto a new page (a reduced print or an oversized export)."""
    W, H = src[0].rect.width, src[0].rect.height
    out = pymupdf.open()
    pg = out.new_page(width=page_w, height=page_h)
    dx, dy = (page_w - W * k) / 2, (page_h - H * k) / 2
    pg.show_pdf_page(pymupdf.Rect(dx, dy, dx + W * k, dy + H * k), src, 0)
    return out


def _classes(doc) -> list[str]:
    return sorted(o.cls.value for o in doc.objects)


# --------------------------------------------------------------------------- spatial index


def test_box_grid_matches_brute_force():
    rng = random.Random(7)
    boxes = []
    for _ in range(400):
        x, y = rng.uniform(0, 1000), rng.uniform(0, 700)
        lo, hi = rng.choice(((0.0, 6.0), (5.0, 60.0), (300.0, 900.0)))
        w, h = rng.uniform(lo, hi), rng.uniform(lo, hi)
        boxes.append((x, y, x + w, y + h))
    g = BoxGrid(18.0)
    for k, b in enumerate(boxes):
        g.add(k, b)
    for q in boxes[:80] + [(0, 0, 2000, 2000)]:
        for reach in (0.0, 7.2, 40.0):
            want = [k for k, b in enumerate(boxes) if bbox_dist(b, q) <= reach]
            assert g.near(q, reach) == want


# --------------------------------------------------------------------------- reduced copies (frame band)


def test_margin_tolerance_follows_the_frame_band():
    reg = Regions(page_bbox=(0, 0, 842, 595), drawing_area=(14.9, 15.82, 827.1, 579.2), frame_gap=4.94)
    tick = (83.4, 10.88, 83.4, 15.82)  # a zone-marker tick between the outer and the inner frame line
    assert reg.margin_tol(2.5) == pytest.approx(0.25 * 4.94)
    assert reg.kind_at(tick, 2.5) == "margin"
    assert reg.kind_at((100.0, 16.5, 110.0, 22.0), 2.5) is None  # a symbol drawn right at the inner frame
    wide = Regions(page_bbox=(0, 0, 842, 595), drawing_area=(14.9, 15.82, 827.1, 579.2))
    assert wide.kind_at(tick, 2.5) is None  # (single frame: the absolute tolerance, as before)


def test_reduced_copy_frame_ticks_are_not_symbols():
    """A sheet printed reduced (zone ticks 14 pt -> 4.2 pt, shorter than twice the snap tolerance):
    the ticks between the outer and the inner frame line used to become UNKNOWN symbols."""
    sh = Sheet(1190, 842, Style(scale=U))
    _train(sh)
    sh.finish()
    full = extract_pdf_doc(sh.doc)
    small = extract_pdf_doc(_placed(sh.doc, 0.3, 1190 * 0.3 + 20, 842 * 0.3 + 20))
    assert not [o for o in small.objects if o.cls == SymbolClass.UNKNOWN]
    assert _classes(small) == _classes(full)


def extract_pdf_doc(doc: pymupdf.Document, **kw):
    """extract_pdf on an in-memory document (saved to a temporary file)."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sheet.pdf"
        doc.save(str(path))
        return extract_pdf(path, **kw)


def test_generated_a1_sheet_reduced_to_a4(tmp_path):
    """Reviewer case: A1 sheets of the generator placed on an A4 page (k = 0.353)."""
    synthetic = pytest.importorskip("shippid.datasets.synthetic")
    writer = pytest.importorskip("shippid.datasets.pdf_writer")
    from shippid.evaluate import evaluate_document

    gt, spec = synthetic.generate(500)
    src = writer.write_pdf(spec, tmp_path / "a1.pdf")
    a1 = pymupdf.open(src)
    W, H = a1[0].rect.width, a1[0].rect.height
    k = min(842 / W, 595 / H)
    out = _placed(a1, k, 842, 595)
    path = tmp_path / "a4.pdf"
    out.save(str(path))
    doc = extract_pdf(path)
    assert not [o for o in doc.objects if o.cls == SymbolClass.UNKNOWN]
    dx, dy = (842 - W * k) / 2, (595 - H * k) / 2
    g = gt.model_copy(deep=True)
    for o in g.objects:
        b = o.bbox
        o.bbox = type(b)(x0=b.x0 * k + dx, y0=b.y0 * k + dy, x1=b.x1 * k + dx, y1=b.y1 * k + dy)
    r = evaluate_document(doc, g)
    assert r["detection"]["f1"] == 1.0 and r["detection_class_aware"]["f1"] == 1.0, r["detection"]


# --------------------------------------------------------------------------- oversized pages


def test_estimate_unit_window_follows_the_page_size():
    def bowties(k: float) -> list[Prim]:
        prims = []
        for i in range(12):
            x, y, s = 100.0 * k + i * 60.0 * k, 200.0 * k, 18.0 * k
            pts = [(x - s / 2, y - 0.3 * s), (x - s / 2, y + 0.3 * s), (x + s / 2, y - 0.3 * s), (x + s / 2, y + 0.3 * s)]
            for (a, b) in ((pts[0], pts[3]), (pts[1], pts[2]), (pts[0], pts[1]), (pts[2], pts[3])):
                prims.append(Prim(len(prims), "seg", a[0], a[1], b[0], b[1]))
        return prims

    big = bowties(6.0)  # an A2 sheet exported at 6x: 108 pt valves on a 10104 x 7146 pt page
    ids = list(range(len(big)))
    assert estimate_unit(big, ids, 1.25) == 18.0  # the absolute 5-80 pt window misses them all
    assert estimate_unit(big, ids, 1.25, page_size=(10104.0, 7146.0)) == pytest.approx(108.0, rel=0.02)
    assert page_scale((2384.0, 3370.0)) == 1.0 and page_scale((14400.0, 10000.0)) == pytest.approx(10000 / 2384)
    small = bowties(1.0)
    assert estimate_unit(small, ids, 1.25, page_size=(1190.0, 842.0)) == estimate_unit(small, ids, 1.25)


@pytest.mark.parametrize("k", [6.0, 12.0])
def test_oversized_export_keeps_its_symbols(k):
    """A drawing exported at k x its size (up to the 14400 pt PDF limit): the unit estimate and the
    rigid-symbol size limit follow the page scale."""
    sh = Sheet(1190, 842, Style(scale=U))
    _train(sh)
    sh.finish()
    full = extract_pdf_doc(sh.doc)
    big = extract_pdf_doc(_placed(sh.doc, k, 1190 * k, 842 * k))
    assert _classes(big) == _classes(full)
    assert len(big.pipes) == len(full.pipes)
    assert big.meta["pages"]["0"]["unit"] == pytest.approx(k * full.meta["pages"]["0"]["unit"], rel=0.05)


# --------------------------------------------------------------------------- damaged files


def test_damaged_pdf_is_reported(tmp_path, caplog):
    sh = Sheet(1190, 842, Style(scale=U))
    _train(sh)
    sh.finish()
    good = tmp_path / "good.pdf"
    sh.doc.save(str(good), deflate=True)
    intact = extract_pdf(good)
    assert "pdf" not in intact.meta and "damaged" not in intact.meta
    data = good.read_bytes()
    bad = tmp_path / "truncated.pdf"
    for frac in (0.5, 0.55, 0.6, 0.45, 0.65, 0.7):  # an interrupted download MuPDF can still repair
        bad.write_bytes(data[: int(len(data) * frac)])
        try:
            pymupdf.open(str(bad)).close()
            break
        except Exception:
            continue
    with caplog.at_level("WARNING"):
        doc = extract_pdf(bad)
    assert doc.meta["damaged"] is True
    assert doc.meta["pdf"]["repaired"] is True and doc.meta["pdf"]["n_warnings"] >= 1
    assert any("repaired" in r.getMessage() for r in caplog.records)
    again = extract_pdf(good)  # MuPDF's warning store does not leak into the next document
    assert "pdf" not in again.meta


# --------------------------------------------------------------------------- loose strokes and fields


def _count_calls(monkeypatch) -> dict[str, int]:
    """Count template matches (``_Matcher.detect``) and the pipeline's bbox-distance tests."""
    calls = {"detect": 0, "bbox_dist": 0}
    detect, dist = P._Matcher.detect, P.bbox_dist

    def counted_detect(self, prim_ids):
        calls["detect"] += 1
        return detect(self, prim_ids)

    def counted_dist(a, b):
        calls["bbox_dist"] += 1
        return dist(a, b)

    monkeypatch.setattr(P._Matcher, "detect", counted_detect)
    monkeypatch.setattr(P, "bbox_dist", counted_dist)
    return calls


def test_loose_strokes_are_noise_and_matching_stays_linear(monkeypatch):
    """1500 isolated short hatch strokes beside a train: no object each (they were UNKNOWN
    'equipment' objects), and the fragment absorption no longer tries every pair of them."""
    sh = Sheet(1190, 842, Style(scale=U))
    objs = _train(sh, y=250)
    s = sh.shape
    rng = random.Random(3)
    n = 0
    for r in range(30):
        for c in range(50):  # short strokes of random length and direction (no two alike)
            x, y = 80 + c * 20.0 + rng.uniform(-2, 2), 420 + r * 12.0 + rng.uniform(-1, 1)
            a, L = math.radians(rng.uniform(15, 75)), rng.uniform(5, 9)
            s.draw_line(pymupdf.Point(x, y), pymupdf.Point(x + L * math.cos(a), y - L * math.sin(a)))
            n += 1
    s.finish(width=0.25, color=(0, 0, 0))
    sh.finish()
    calls = _count_calls(monkeypatch)
    doc = extract_pdf_doc(sh.doc)
    pm = doc.meta["pages"]["0"]
    assert _classes(doc) == sorted(o.cls.value for o in objs.values())
    let = pm.get("stroke_lettering") or {}
    gone = pm["clutter"]["noise"] + pm["clutter"]["hatch_field"] + let.get("strokes", 0) + let.get("hatch_strokes", 0)
    assert gone >= 0.95 * n, (pm["clutter"], let)
    # the fragment absorption used to test every detection against every other (n^2 per round, 23 M
    # bbox_dist calls for 10k strokes); now a grid index hands it the neighbours
    n_c = pm["counts"]["candidates"]
    assert calls["bbox_dist"] <= 20 * n_c, (calls, n_c)
    # template matches: one per candidate and classification pass (the unit is re-anchored once or twice)
    assert calls["detect"] <= 6 * n_c + pm["decompose_budget_used"] + 100, (calls, n_c)


def test_field_of_identical_marks_is_hatch_but_a_piped_row_is_kept():
    sh = Sheet(1190, 842, Style(scale=U))
    objs = _train(sh, y=200)
    s = sh.shape
    for r in range(8):  # 64 crosses of 12 pt: too large for lettering, not single strokes
        for c in range(8):
            x, y = 120 + c * 22.0, 420 + r * 22.0
            s.draw_line(pymupdf.Point(x, y), pymupdf.Point(x + 12, y + 12))
            s.draw_line(pymupdf.Point(x, y + 12), pymupdf.Point(x + 12, y))
    s.finish(width=0.35, color=(0, 0, 0))
    # six identical drawings no library symbol explains, inline on one pipe 2 units apart
    y = 700.0
    xs = [500 + 40.0 * i for i in range(6)]
    sh.line([(440, y), (xs[0] - 9, y)])
    for i, x in enumerate(xs):
        pts = [(x - 9, y), (x - 3, y - 7), (x + 3, y + 7), (x + 9, y)]
        s.draw_polyline([pymupdf.Point(*p) for p in pts])
        s.finish(width=0.6, color=(0, 0, 0), closePath=False)
        s.draw_line(pymupdf.Point(x - 3, y - 7), pymupdf.Point(x + 3, y - 7))
        s.finish(width=0.6, color=(0, 0, 0))
        if i + 1 < len(xs):
            sh.line([(x + 9, y), (xs[i + 1] - 9, y)])
    sh.line([(xs[-1] + 9, y), (800, y)])
    sh.finish()
    doc = extract_pdf_doc(sh.doc)
    pm = doc.meta["pages"]["0"]
    assert pm["clutter"]["hatch_field"] >= 60
    in_field = [o for o in doc.objects if 110 <= o.bbox.x0 and o.bbox.x1 <= 300 and 410 <= o.bbox.y0 and o.bbox.y1 <= 600]
    assert not in_field
    row = [o for o in doc.objects if abs((o.bbox.y0 + o.bbox.y1) / 2 - y) < 10 and 480 <= o.bbox.x0]
    assert len(row) == 6 and all(o.ports for o in row)  # the piped row stays (for review / Gemma)
    assert len([o for o in doc.objects if o.cls != SymbolClass.UNKNOWN]) == len(objs)


# --------------------------------------------------------------------------- stroke lettering (PDFSHX=0)

_GLYPHS = [
    [[(0, 0), (0.5, 1), (1, 0)], [(0.25, 0.5), (0.75, 0.5)]],  # A
    [[(1, 0), (0, 0), (0, 1), (1, 1)], [(0, 0.5), (0.7, 0.5)]],  # E
    [[(0.2, 0), (0, 0.3), (0, 0.7), (0.2, 1), (0.8, 1), (1, 0.7), (1, 0.3), (0.8, 0), (0.2, 0)]],  # O
    [[(0, 0), (0, 1), (1, 0), (1, 1)]],  # N
    [[(0, 1), (0, 0), (1, 0)]],  # L
    [[(0, 1), (1, 1), (0.3, 0)]],  # 7
    [[(0, 0), (0, 1), (0.8, 1), (1, 0.8), (1, 0.6), (0.8, 0.5), (0, 0.5)]],  # P
]


def stroke_text(sh: Sheet, x: float, y: float, text: str, size: float = 5.0, vertical: bool = False) -> None:
    """``text`` as stroke glyphs (no text object): baseline at ``y``, cap height 0.7 x size; vertical
    text reads bottom-up from (x, y)."""
    s = sh.shape
    adv, w, h = 0.62 * size, 0.45 * size, 0.7 * size
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        if ch == "-":
            lines = [[(0.1, 0.5), (0.9, 0.5)]]
        elif ch == ".":
            lines = [[(0.45, 0.0), (0.55, 0.0)]]
        else:
            lines = _GLYPHS[ord(ch) % len(_GLYPHS)]
        for ln in lines:
            if vertical:
                pts = [pymupdf.Point(x - v * h, y - i * adv - u * w) for u, v in ln]
            else:
                pts = [pymupdf.Point(x + i * adv + u * w, y - v * h) for u, v in ln]
            s.draw_polyline(pts)
            s.finish(width=0.25, color=(0, 0, 0), closePath=False)


def _lettered_sheet() -> tuple[Sheet, dict]:
    sh = Sheet(1190, 842, Style(scale=U))
    o, y, x0 = sh.symbol, 300.0, 150.0
    t1 = o(S.TANK, x0, y, stretch=(0.8, 1.0))
    v1 = o(S.GATE_VALVE, x0 + 180, y, actuator=A.HANDWHEEL)
    p1 = o(S.PUMP_SCREW, x0 + 270, y)
    v2 = o(S.CHECK_VALVE, x0 + 360, y)
    t2 = o(S.TANK, x0 + 550, y)
    # an instrument on a tap off the pipe between the pump and the check valve, its tag inside the bubble
    xb, yb = 470.0, 250.0
    bub = o(S.INSTRUMENT_LOCAL, xb, yb)
    j = sh.junction(xb, y)
    sh.connect(("o", t1.key), ("o", v1.key), [(t1.bbox[2], y), v1.ports["in"]])
    sh.connect(("o", v1.key), ("o", p1.key), [v1.ports["out"], p1.ports["in"]])
    sh.connect(("o", p1.key), ("j", j), [p1.ports["out"], (xb, y)])
    sh.connect(("j", j), ("o", v2.key), [(xb, y), v2.ports["in"]])
    sh.connect(("o", v2.key), ("o", t2.key), [v2.ports["out"], (t2.bbox[0], y)])
    sh.connect(("o", bub.key), ("j", j), [(xb, yb + 0.4 * U), (xb, y)])
    objs = dict(t1=t1, v1=v1, p1=p1, v2=v2, t2=t2, bub=bub)
    stroke_text(sh, xb - 2.1, yb - 1.0, "PI", size=4.0)
    stroke_text(sh, xb - 3.4, yb + 4.0, "101", size=4.0)
    # tags above the valves, line numbers above the pipes, a vertical line number, notes, a title
    stroke_text(sh, v1.bbox[0] - 6, v1.bbox[1] - 3, "FO-V001")
    stroke_text(sh, v2.bbox[0] - 6, v2.bbox[1] - 3, "FO-V002")
    stroke_text(sh, t1.bbox[2] + 20, 296, "25A-FO-001")
    stroke_text(sh, 560, 296, "25A-FO-002")
    stroke_text(sh, 120, 330, "NO.1 F.O. TANK")
    stroke_text(sh, 700, 520, "40A-FO-003", vertical=True)
    stroke_text(sh, 60, 640, "NOTES", size=6.0)
    for k, line in enumerate(["1. ALL VALVES SHOWN OPEN", "2. PIPE SIZES ARE NOMINAL",
                              "3. FOR SYMBOLS SEE LEGEND", "4. LINE NUMBERS PER SPEC"]):
        stroke_text(sh, 60, 655 + 11 * k, line)
    stroke_text(sh, 820, 760, "FUEL OIL TRANSFER SYSTEM", size=7.0)
    sh.finish()
    return sh, objs


def test_stroke_lettering_without_text_carrier_is_not_symbols():
    """SHX text exported as plain geometry: every word used to become an UNKNOWN object, the
    bubble with its tag inside was UNKNOWN too, and letters next to pipes split them."""
    sh, objs = _lettered_sheet()
    doc = extract_pdf_doc(sh.doc)
    pm = doc.meta["pages"]["0"]
    assert not [o for o in doc.objects if o.cls == SymbolClass.UNKNOWN], [(o.id, o.bbox) for o in doc.objects]
    r = evaluate(sh, doc)
    assert r.recall == 1.0 and r.precision == 1.0 and r.cls_acc == 1.0, r.errors
    assert r.conn_recall == 1.0 and r.conn_precision == 1.0, r.summary()
    let = pm["stroke_lettering"]
    assert let["rows"] >= 10 and let["strokes"] > 100
    assert any(b[3] - b[1] > b[2] - b[0] for b in let["boxes"])  # the vertical line number is a row too


def test_stroke_lettering_leaves_symbol_insides_alone():
    """A sheet with real text: unconnected symbols with small inner strokes (a BWTS unit's lamps, a
    plate heat exchanger's plates, legend glyphs) are not lettering."""
    sh = Sheet(1190, 842, Style(scale=U))
    objs = _train(sh, y=300)
    bw = sh.symbol(S.BWTS_UNIT, 300, 500)
    hx = sh.symbol(S.PLATE_HEAT_EXCHANGER, 500, 500)
    sh.line([(200, 500), bw.ports["in"]])
    sh.line([bw.ports["out"], (400, 500)])
    sh.legend(900, 60, [S.BWTS_UNIT, S.PLATE_HEAT_EXCHANGER, S.GATE_VALVE, S.DUPLEX_FILTER, S.FLOW_METER])
    sh.finish()
    doc = extract_pdf_doc(sh.doc)
    pm = doc.meta["pages"]["0"]
    assert "stroke_lettering" not in pm
    got = {o.cls for o in doc.objects}
    assert {S.BWTS_UNIT, S.PLATE_HEAT_EXCHANGER} <= got
    assert len(doc.objects) == len(objs) + 2


def test_pdfshx0_variant_drawings(tmp_path):
    """The shx_text variant with its text carriers removed (hidden text and SHX annotations): what
    AutoCAD writes with PDFSHX=0. Base code: detection F1 0.04-0.39 and 30-70 s per sheet."""
    variants = pytest.importorskip("shippid.datasets.variants")
    build = pytest.importorskip("shippid.datasets.build")
    from shippid.evaluate import evaluate_document

    gt, pdf = variants.make_variant(build.item_seed(1000, 0), "shx_text")
    d = pymupdf.open("pdf", pdf)
    p = d[0]
    for a in list(p.annots()):
        p.delete_annot(a)
    p.add_redact_annot(p.rect)
    p.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                       text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    path = tmp_path / "shx0.pdf"
    d.save(str(path))
    doc = extract_pdf(path)
    assert not doc.texts  # nothing to read: the strings exist only as strokes
    r = evaluate_document(doc, gt)
    assert r["detection"]["f1"] >= 0.9 and r["connections"]["f1"] >= 0.95, (r["detection"], r["connections"])
    assert sum(o.cls == SymbolClass.UNKNOWN for o in doc.objects) <= 0.1 * len(gt.objects)
    assert doc.meta["pages"]["0"]["stroke_lettering"]["rows"] >= 50
