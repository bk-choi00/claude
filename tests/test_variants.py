"""Tests for shippid.datasets.variants: out-of-distribution export variants with exact ground truth."""

import json
import math
import re
from collections import Counter

import pymupdf
import pytest

from shippid.benchmark import load_manifest
from shippid.datasets import generate, to_bytes
from shippid.datasets.synthetic import ALL_SYSTEMS
from shippid.datasets.variants import (
    A0,
    BOWTIE_CLASSES,
    COMBINABLE,
    THROUGH_CLASSES,
    VARIANTS,
    YARD_DESIGNS,
    _components,
    build_variant_suite,
    korean_label,
    make_variant,
)
from shippid.model import BBox, Category, PIDDocument, PipeKind, SymbolClass, TextRole

REQUIRED = ("rotated_page", "qu_bowties", "pipes_through", "polyline_curves", "styled_lines", "fragmented_text",
            "layers_off", "scaled", "yard_symbols", "combined")
# seed 1: A0 / layer colours / notes OFF / /Rotate 270; seed 3: thick symbols / 1.5x / /Rotate 270;
# seed 9: wipeouts / superseded valves / /Rotate 90
SEEDS = [(1, ALL_SYSTEMS[1]), (3, ALL_SYSTEMS[3]), (9, ALL_SYSTEMS[9])]


@pytest.fixture(scope="module")
def made():
    out = {}
    for v in VARIANTS:
        for seed, system in SEEDS:
            out[(v, seed)] = make_variant(seed, v, system)
    return out


def _page(pdf: bytes) -> pymupdf.Page:
    return pymupdf.open("pdf", pdf)[0]


def _chars(page: pymupdf.Page) -> list[tuple[str, pymupdf.Rect, bool]]:
    """(char, bbox in displayed coordinates, hidden) for every non-space character."""
    m = page.rotation_matrix
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for line in b.get("lines", []):
            for span in line["spans"]:
                hidden = span.get("alpha", 255) == 0
                for c in span["chars"]:
                    if not c["c"].isspace():
                        out.append((c["c"], pymupdf.Rect(c["bbox"]) * m, hidden))
    return out


def _items(page: pymupdf.Page) -> list[tuple[str, list[pymupdf.Point], dict]]:
    """Every get_drawings() item as (kind, points in displayed coordinates, drawing)."""
    m = page.rotation_matrix
    out = []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "re":
                r = it[1]
                pts = [r.tl, r.tr, r.br, r.bl]
            elif it[0] == "qu":
                q = it[1]
                pts = [q.ul, q.ur, q.lr, q.ll]
            else:
                pts = [p for p in it[1:] if isinstance(p, pymupdf.Point)]
            out.append((it[0], [p * m for p in pts], d))
    return out


def _inside(pts, b: BBox, tol: float) -> bool:
    return all(b.x0 - tol <= p.x <= b.x1 + tol and b.y0 - tol <= p.y <= b.y1 + tol for p in pts)


def _text_of(chars, t, tol=1.0) -> tuple[str, list]:
    b = t.bbox
    sel = {}
    for c, r, hidden in chars:
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if not hidden and b.x0 - tol <= cx <= b.x1 + tol and b.y0 - tol <= cy <= b.y1 + tol:
            sel[(c, round(r.x0, 1), round(r.y0, 1))] = r  # duplicate overdraw counts once
    keys = sorted(sel, key=(lambda k: k[1]) if t.rotation == 0 else (lambda k: -sel[k].y1))
    return "".join(k[0] for k in keys), [sel[k] for k in keys]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


# --------------------------------------------------------------------------- registry / api


def test_registry_and_unknown_variant():
    assert set(REQUIRED) <= set(VARIANTS)
    assert list(VARIANTS)[0] == "plain"
    assert all(isinstance(d, str) and len(d) > 20 for d in VARIANTS.values())
    assert set(COMBINABLE) == set(REQUIRED) - {"combined"}
    with pytest.raises(ValueError):
        make_variant(1, "no_such_variant")


def test_plain_is_byte_identical_to_the_base_writer():
    for seed, system in SEEDS[:2]:
        doc, spec = generate(seed, system)
        gt, pdf = make_variant(seed, "plain", system)
        assert pdf == to_bytes(spec)
        assert gt.model_dump(exclude={"meta", "source"}) == doc.model_dump(exclude={"meta", "source"})


def test_determinism():
    for v in ("combined", "fragmented_text", "layers_off", "yard_symbols", "pipes_through"):
        a_gt, a_pdf = make_variant(5, v)
        b_gt, b_pdf = make_variant(5, v)
        assert a_pdf == b_pdf, v
        assert a_gt.to_json() == b_gt.to_json(), v


# --------------------------------------------------------------------------- every variant


def test_pdf_opens_and_page_geometry(made):
    for (v, seed), (gt, pdf) in made.items():
        d = pymupdf.open("pdf", pdf)
        assert d.page_count == 1 and d.embfile_count() == 0
        page = d[0]
        info = gt.pages[0]
        assert page.rect.width == pytest.approx(info.width, abs=0.01), (v, seed)
        assert page.rect.height == pytest.approx(info.height, abs=0.01), (v, seed)
        assert page.rotation == info.rotation
        assert page.rect.width > page.rect.height  # landscape as displayed
        assert gt.meta["variant"]["name"] == v
        assert all(o.cls != SymbolClass.UNKNOWN for o in gt.objects)
        PIDDocument.model_validate_json(gt.to_json())
        blobs = b"\n".join(d.xref_object(x).encode() + ((d.xref_stream(x) or b"") if d.xref_is_stream(x) else b"")
                           for x in range(1, d.xref_length()))
        for marker in (b'"owner_id"', b'"text_ids"', b"synthetic:", b"off_layer", b'"variant"'):
            assert marker not in blobs, (v, seed, marker)


def test_visible_text_matches_ground_truth(made):
    n = 0
    for (v, seed), (gt, pdf) in made.items():
        chars = _chars(_page(pdf))
        for t in gt.texts:
            got, rects = _text_of(chars, t)
            assert got == t.text.replace(" ", ""), (v, seed, t.id, t.text, got)
            u = BBox(x0=min(r.x0 for r in rects), y0=min(r.y0 for r in rects), x1=max(r.x1 for r in rects),
                     y1=max(r.y1 for r in rects))
            assert u.iou(t.bbox) >= 0.6, (v, seed, t.text, u, t.bbox)
            n += 1
        # no visible character outside the ground-truth texts (e.g. content of an OFF layer)
        boxes = [t.bbox for t in gt.texts]
        for c, r, hidden in chars:
            if not hidden:
                cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
                assert any(b.contains_point(cx, cy, tol=1.0) for b in boxes), (v, seed, c, r)
    assert n > 5000


def test_object_bboxes_contain_their_drawn_paths(made):
    checked = 0
    for (v, seed), (gt, pdf) in made.items():
        items = _items(_page(pdf))
        for o in gt.objects:
            b = o.bbox
            tol = 0.05 * max(1.0, (gt.meta.get("symbol_scale") or 18.0) / 18.0)
            inside = [pts for kind, pts, _ in items if _inside(pts, b, tol)]
            assert inside, (v, seed, o.id, o.cls)
            xs = [p.x for pts in inside for p in pts]
            ys = [p.y for pts in inside for p in pts]
            # the drawn glyph spans its box (tessellated circles are inscribed: allow 2 %)
            slack_x = 0.6 + 0.02 * b.width
            slack_y = 0.6 + 0.02 * b.height
            assert min(xs) <= b.x0 + slack_x and max(xs) >= b.x1 - slack_x, (v, seed, o.id, o.cls, b)
            assert min(ys) <= b.y0 + slack_y and max(ys) >= b.y1 - slack_y, (v, seed, o.id, o.cls, b)
            checked += 1
    assert checked > 1000


def test_topology_is_unchanged_by_export_variants(made):
    for seed, _ in SEEDS:
        base = made[("plain", seed)][0]
        for v in ("rotated_page", "qu_bowties", "pipes_through", "polyline_curves", "styled_lines", "layers_off"):
            gt = made[(v, seed)][0]
            assert [o.model_dump() for o in gt.objects] == [o.model_dump() for o in base.objects], (v, seed)
            assert [p.model_dump() for p in gt.pipes] == [p.model_dump() for p in base.pipes], (v, seed)
            assert gt.junctions == base.junctions


# --------------------------------------------------------------------------- specific variants


def test_rotated_pages(made):
    rotations = set()
    for seed, _ in SEEDS:
        gt, pdf = made[("rotated_page", seed)]
        page = _page(pdf)
        assert page.rotation in (90, 270) and page.rotation == gt.pages[0].rotation
        rotations.add(page.rotation)
        assert page.rect.width > page.rect.height
        assert page.mediabox.width < page.mediabox.height  # portrait media, landscape display
        base = made[("plain", seed)][0]
        assert (gt.pages[0].width, gt.pages[0].height) == (base.pages[0].width, base.pages[0].height)
        # reading direction of horizontal labels is left-to-right on the displayed page
        line = next(ln for b in page.get_text("dict")["blocks"] for ln in b.get("lines", []))
        d = pymupdf.Point(line["dir"]) * pymupdf.Matrix(page.rotation_matrix.a, page.rotation_matrix.b,
                                                        page.rotation_matrix.c, page.rotation_matrix.d, 0, 0)
        assert round(abs(d.x) + abs(d.y)) == 1
    assert rotations == {90, 270}


def test_qu_bowties(made):
    for seed, _ in SEEDS:
        for v, expect in (("qu_bowties", True), ("plain", False)):
            gt, pdf = made[(v, seed)]
            items = _items(_page(pdf))
            valves = [o for o in gt.objects if o.cls in BOWTIE_CLASSES]
            assert valves
            for o in valves:
                n_qu = sum(1 for kind, pts, _ in items if kind == "qu" and _inside(pts, o.bbox, 0.1))
                assert (n_qu >= 1) == expect, (v, seed, o.id, o.cls)
                if expect and o.cls in (SymbolClass.CHECK_VALVE, SymbolClass.SDNR_VALVE):
                    fills = [d for kind, pts, d in items if _inside(pts, o.bbox, 0.1) and d.get("fill")]
                    assert fills  # outlet triangle hatch kept as a separate fill


def test_polyline_curves(made):
    for seed, _ in SEEDS:
        gt, pdf = made[("polyline_curves", seed)]
        items = _items(_page(pdf))
        assert not [k for k, _, _ in items if k == "c"]
        n = gt.meta["variant"]["params"]["polyline_curves"]["segments_per_circle"]
        assert 24 <= n <= 48
        circled = [o for o in gt.objects if o.category == Category.INSTRUMENT]
        for o in circled:
            short = sum(1 for k, pts, _ in items if k == "l" and _inside(pts, o.bbox, 0.1))
            assert short >= n, (seed, o.id)
        # the base drawing does have curves (circles, hops)
        assert [k for k, _, _ in _items(_page(made[("plain", seed)][1])) if k == "c"]


def test_pipes_through(made):
    wipeouts = set()
    for seed, _ in SEEDS:
        gt, pdf = made[("pipes_through", seed)]
        params = gt.meta["variant"]["params"]["pipes_through"]
        wipeouts.add(params["wipeout"])
        items = _items(_page(pdf))
        lines = [pts for k, pts, _ in items if k == "l"]
        attached = Counter()
        for p in gt.pipes:
            if p.kind == PipeKind.PROCESS:
                for ep in (p.start, p.end):
                    if ep.object_id:
                        attached[(ep.object_id, ep.port)] += 1
        n = 0
        for o in gt.objects:
            ports = {pt.name: pt.position for pt in o.ports}
            if o.cls not in THROUGH_CLASSES or not ({"in", "out"} <= set(ports)):
                continue
            if not (attached[(o.id, "in")] and attached[(o.id, "out")]):
                continue
            a, b = ports["in"], ports["out"]
            through = [pts for pts in lines if
                       {(round(pts[0].x, 1), round(pts[0].y, 1)), (round(pts[1].x, 1), round(pts[1].y, 1))}
                       == {(round(a.x, 1), round(a.y, 1)), (round(b.x, 1), round(b.y, 1))}]
            if through:
                n += 1
            # the plain drawing has no stroke across the symbol
            plain_lines = [pts for k, pts, _ in _items(_page(made[("plain", seed)][1])) if k == "l"]
            assert not [pts for pts in plain_lines if
                        {(round(pts[0].x, 1), round(pts[0].y, 1)), (round(pts[1].x, 1), round(pts[1].y, 1))}
                        == {(round(a.x, 1), round(a.y, 1)), (round(b.x, 1), round(b.y, 1))}]
        assert n == params["symbols"] and n >= 5, (seed, n, params)
        if params["wipeout"]:
            whites = [d for _, _, d in items if d.get("fill") == (1.0, 1.0, 1.0)]
            assert len(whites) >= n
    assert wipeouts == {True, False}


def test_styled_lines(made):
    colours = set()
    for seed, _ in SEEDS:
        gt, pdf = made[("styled_lines", seed)]
        p = gt.meta["variant"]["params"]["styled_lines"]
        assert 0.25 <= p["pipe_width"] <= 2.0
        drawings = _page(pdf).get_drawings()
        widths = Counter(round(d.get("width") or 0, 2) for d in drawings if d.get("color"))
        assert round(p["symbol_width"], 2) in widths
        if p["symbols_thicker"]:
            assert p["symbol_width"] > p["pipe_width"]
        caps = {"round": 1, "square": 2, "butt": 0}[p["caps"]]
        assert any(d.get("lineCap") and d["lineCap"][0] == caps for d in drawings if d.get("color"))
        keys = Counter((d["type"], d.get("width"), d.get("color"), d.get("fill"),
                        tuple((it[0],) + tuple(str(x) for x in it[1:]) for it in d["items"])) for d in drawings)
        assert p["duplicates"] > 0 and max(keys.values()) >= 2  # same path drawn twice
        cols = {d.get("color") for d in drawings if d.get("color")}
        if p["colour"] == "black":
            assert cols == {(0.0, 0.0, 0.0)}
        else:
            assert len(cols) >= 3
            assert gt.meta["variant"]["layers"]
        colours.add(p["colour"])
    assert colours == {"black", "layer"}
    assert any(made[("styled_lines", s)][0].meta["variant"]["params"]["styled_lines"]["symbols_thicker"]
               for s, _ in SEEDS)


def test_fragmented_text(made):
    hangul = 0
    for seed, _ in SEEDS:
        gt, pdf = made[("fragmented_text", seed)]
        page = _page(pdf)
        p = gt.meta["variant"]["params"]["fragmented_text"]
        spans = [s for b in page.get_text("rawdict")["blocks"] for ln in b.get("lines", []) for s in ln["spans"]]
        fonts = {s["font"] for s in spans if s.get("alpha", 255)}
        assert len({"Courier", "Times-Roman", "Helvetica"} & {f.split("-Bold")[0] for f in fonts}) >= 2, fonts
        # fragmentation: many more text objects than logical strings
        n_bt = len(re.findall(rb"\bBT\b", page.read_contents()))
        assert n_bt > 2 * len(gt.texts)
        # hidden PDFSHX=2 duplicates carry the logical strings
        hidden = Counter(c["c"] for s in spans if s.get("alpha", 255) == 0 for c in s["chars"] if not c["c"].isspace())
        dup = {t.id: t for t in gt.texts if t.id in set(p["hidden_text_ids"])}
        assert len(dup) == p["hidden_duplicates"] > 0
        expect = Counter(c for t in dup.values() for c in t.text if not c.isspace())
        # MuPDF folds a glyph repeated at the same spot in the same font ("fake bold"), so an invisible copy
        # exactly over its Helvetica original may vanish from get_text; nothing else is hidden
        assert not hidden - expect
        assert sum(hidden.values()) >= 0.6 * sum(expect.values())
        ko = [t for t in gt.texts if any("가" <= c <= "힣" for c in t.text)]
        assert len(ko) == p["hangul_labels"]
        hangul += len(ko)
        chars = _chars(page)
        for t in ko:
            assert t.role == TextRole.LABEL
            got, rects = _text_of(chars, t)
            assert got == t.text.replace(" ", "")
            u = BBox(x0=min(r.x0 for r in rects), y0=min(r.y0 for r in rects), x1=max(r.x1 for r in rects),
                     y1=max(r.y1 for r in rects))
            assert u.iou(t.bbox) >= 0.95, (t.text, u, t.bbox)  # 'korea' font metrics are exact
            owner = next(o for o in gt.objects if o.id == t.owner_id)
            if "name_en" in owner.attributes:
                assert owner.attributes["name"] == t.text
    assert hangul >= 3


def test_korean_labels():
    assert korean_label("NO.1 BALLAST PUMP") == "1번 평형수 펌프"
    assert korean_label("NO.2 W.B.TK (P)") == "2번 평형수 탱크 (좌현)"
    assert korean_label("HIGH SEA CHEST") == "고위 씨체스트"
    assert korean_label("6G70ME-C10.5") is None
    assert korean_label("80 M3") is None


def test_layers_off(made):
    notes_off = False
    for v in ("layers_off", "combined"):
        for seed, _ in SEEDS:
            gt, pdf = made[(v, seed)]
            vm = gt.meta["variant"]
            if "layers_off" not in vm["components"]:
                continue
            layers = vm["layers"]
            off = {name for name, on in layers.items() if not on}
            on = {name for name, on in layers.items() if on}
            assert off and on
            d = pymupdf.open("pdf", pdf)
            states = {o["name"]: o["on"] for o in d.get_ocgs().values()}
            assert states == layers
            offs = vm["off_layer_texts"]
            assert offs and all(t["layer"] in off for t in offs)
            assert all(o["layer"] in off for o in vm["off_layer_objects"])
            chars = _chars(d[0])
            gt_strings = {_norm(t.text) for t in gt.texts}
            for t in offs:
                b = BBox(x0=t["bbox"][0], y0=t["bbox"][1], x1=t["bbox"][2], y1=t["bbox"][3])
                seen = "".join(c for c, r, _ in chars if b.contains_point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
                assert _norm(t["text"]) not in _norm(seen), (seed, t)
                if _norm(t["text"]) not in gt_strings:
                    assert t["text"] not in {w[4] for w in d[0].get_text("words")}
            notes_off |= vm["params"]["layers_off"]["notes_off"]
            if vm["params"]["layers_off"]["notes_off"]:
                assert not [t for t in gt.texts if t.role == TextRole.NOTE]
                ids = {t.id for t in gt.texts}
                assert ids == {f"t{i + 1}" for i in range(len(gt.texts))}
                assert all(x in ids for o in gt.objects for x in o.text_ids)
                assert all(x in ids for p in gt.pipes for x in p.text_ids)
            # the OFF content is really in the file: switching every layer on reveals it
            d.set_layer(-1, on=list(d.get_ocgs()), off=[])
            words = {w[4] for w in pymupdf.open("pdf", d.tobytes())[0].get_text("words")}
            single = [t["text"] for t in offs if " " not in t["text"]]
            assert single and all(s in words for s in single)
            if vm["off_layer_objects"]:
                n_on = len(pymupdf.open("pdf", d.tobytes())[0].get_drawings())
                assert n_on > len(d[0].get_drawings())
    assert notes_off


def test_scaled(made):
    papers = set()
    cases = [(seed, made[("scaled", seed)], made[("plain", seed)][0]) for seed, _ in SEEDS]
    cases.append((4, make_variant(4, "scaled", ALL_SYSTEMS[4]), make_variant(4, "plain", ALL_SYSTEMS[4])[0]))
    for seed, (gt, pdf), base in cases:
        p = gt.meta["variant"]["params"]["scaled"]
        f = p["factor"]
        papers.add("A0" if p["paper"] == "A0" else ("down" if f < 1 else "up"))
        page = _page(pdf)
        if p["paper"] == "A0":
            assert (page.rect.width, page.rect.height) == pytest.approx(A0)
        else:
            assert f in (0.6, 1.5)
            assert page.rect.width == pytest.approx(base.pages[0].width * f, abs=0.01)
        for o, ob in zip(gt.objects, base.objects):
            assert o.bbox.width == pytest.approx(ob.bbox.width * f, abs=2e-3)
            assert o.bbox.height == pytest.approx(ob.bbox.height * f, abs=2e-3)
        for t, tb in zip(gt.texts, base.texts):
            assert t.size == pytest.approx(tb.size * f, abs=2e-3)
        assert gt.meta["symbol_scale"] == pytest.approx(base.meta["symbol_scale"] * f, abs=2e-3)
    assert papers == {"A0", "down", "up"}


def test_yard_symbols(made):
    seen_classes = set()
    for seed, _ in SEEDS:
        gt, pdf = made[("yard_symbols", seed)]
        p = gt.meta["variant"]["params"]["yard_symbols"]
        items = _items(_page(pdf))
        used = {o.cls for o in gt.objects if o.cls in YARD_DESIGNS}
        assert used and {c.value for c in used} <= set(p["legend_classes"])
        legend = [t.text for t in gt.texts if t.role == TextRole.LEGEND]
        for cls in used:
            label = {SymbolClass.CHECK_VALVE: "CHECK", SymbolClass.GATE_VALVE: "GATE VALVE",
                     SymbolClass.BALL_VALVE: "BALL VALVE", SymbolClass.PUMP_CENTRIFUGAL: "CENTRIFUGAL PUMP",
                     SymbolClass.SDNR_VALVE: "SCREW-DOWN", SymbolClass.INSTRUMENT_LOCAL: "LOCAL INSTRUMENT"}[cls]
            assert any(label in s for s in legend), (seed, cls, legend)
        for o in gt.objects:
            inside = [(k, pts, d) for k, pts, d in items if _inside(pts, o.bbox, 0.1)]
            if o.cls == SymbolClass.CHECK_VALVE:
                assert not [d for _, _, d in inside if d.get("fill")], (seed, o.id)  # yard check: no fill
            if o.cls == SymbolClass.GATE_VALVE:
                ports = {pt.name: pt.position for pt in o.ports}
                cx, cy = (ports["in"].x + ports["out"].x) / 2, (ports["in"].y + ports["out"].y) / 2
                bars = [pts for k, pts, _ in inside if k == "l" and
                        math.dist(((pts[0].x + pts[1].x) / 2, (pts[0].y + pts[1].y) / 2), (cx, cy)) < 0.05]
                assert bars, (seed, o.id)  # bar through the middle of the bow-tie
            if o.cls == SymbolClass.INSTRUMENT_LOCAL:
                assert sum(1 for k, _, _ in inside if k == "c") >= 8  # two circles
            seen_classes.add(o.cls)
        # ground truth stays canonical
        assert all(isinstance(o.cls, SymbolClass) and o.cls != SymbolClass.UNKNOWN for o in gt.objects)
    assert {SymbolClass.CHECK_VALVE, SymbolClass.GATE_VALVE, SymbolClass.PUMP_CENTRIFUGAL,
            SymbolClass.INSTRUMENT_LOCAL} <= seen_classes


def test_combined_mixes_two_to_four(made):
    seen = set()
    for seed in range(12):
        comps = _components(seed, "combined")
        assert 2 <= len(comps) <= 4 and set(comps) <= set(COMBINABLE)
        seen |= set(comps)
    assert len(seen) >= 7
    for seed, _ in SEEDS:
        gt = made[("combined", seed)][0]
        assert gt.meta["variant"]["params"].keys() == set(gt.meta["variant"]["components"])


def test_build_variant_suite(tmp_path):
    summary = build_variant_suite(tmp_path, per_variant=2, seed=7, variants=["plain", "rotated_page"], workers=2)
    assert summary["ok"] == 4 and summary["failed"] == 0
    assert json.loads((tmp_path / "summary.json").read_text())["ok"] == 4
    for v in ("plain", "rotated_page"):
        rows = load_manifest(tmp_path / v, split="test")
        assert len(rows) == 2
        for r in rows:
            assert r["variant"] == v and r["split"] == "test"
            assert (tmp_path / v / r["pdf"]).is_file()
            gt = PIDDocument.load(tmp_path / v / r["gt"])
            assert gt.source == r["pdf"] and gt.pages[0].rotation == r["rotation"]
    # items are paired across variants: same base drawing
    a = load_manifest(tmp_path / "plain", split="test")
    b = load_manifest(tmp_path / "rotated_page", split="test")
    assert [r["seed"] for r in a] == [r["seed"] for r in b]
    assert [r["counts"] for r in a] == [r["counts"] for r in b]
    with pytest.raises(ValueError):
        build_variant_suite(tmp_path, per_variant=1, variants=["nope"])
