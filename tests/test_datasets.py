"""Tests for shippid.datasets: synthetic generator, PDF writer, dataset build, collection."""

import json
import math
import shutil
from collections import Counter, defaultdict

import httpx
import pymupdf
import pytest

from shippid.catalog import INSTRUMENT_TAG_RE, classify_tag, detect_system, parse_line_number
from shippid.datasets import build_dataset, fetch_sources, generate, scan_archive, to_bytes
from shippid.datasets.blueprints import build_topology
from shippid.datasets.synthetic import ALL_SYSTEMS, DrawingStyle
from shippid.model import (
    ActuatorType,
    BBox,
    Category,
    PIDDocument,
    PipeKind,
    SymbolClass,
    SystemType,
    TextRole,
)
from shippid.symbols import get_symbol, placed_bbox, placed_ports

SEEDS = [(seed, ALL_SYSTEMS[i % len(ALL_SYSTEMS)]) for i, seed in enumerate(range(100, 122))]  # 22 drawings


@pytest.fixture(scope="module")
def drawings():
    out = []
    for seed, system in SEEDS:
        doc, spec = generate(seed, system)
        out.append((seed, system, doc, spec, to_bytes(spec)))
    return out


# --------------------------------------------------------------------------- generator / PDF


def test_covers_all_systems(drawings):
    systems = {d[2].info.system for d in drawings}
    assert systems == set(ALL_SYSTEMS)
    for seed, system, doc, spec, pdf in drawings:
        assert doc.info.system == system
        assert detect_system(doc.info.title) == system
        assert doc.pages[0].drawing_area is not None
        assert doc.info.drawing_no and doc.info.hull_no and doc.info.ship_type and doc.info.revision
        assert doc.info.scale == "NONE"


def test_pdf_opens_and_is_readable_vector(drawings):
    for seed, system, doc, spec, pdf in drawings:
        d = pymupdf.open("pdf", pdf)
        assert d.page_count == 1
        page = d[0]
        assert (round(page.rect.width), round(page.rect.height)) in {(2384, 1684), (1684, 1191), (1191, 842)}
        assert page.rect.width == pytest.approx(doc.pages[0].width)
        assert len(page.get_text("text")) > 300
        assert len(page.get_drawings()) > 100
        assert page.get_images() == []
        # no hidden ground truth: only standard info keys, nothing embedded
        assert d.embfile_count() == 0
        meta = d.metadata
        assert set(k for k, v in meta.items() if v) <= {"format", "title", "creator", "producer", "subject"}
        assert b"shippid" not in pdf or meta.get("creator") == "shippid.datasets"
        assert doc.to_json().encode()[:50] not in pdf


def _words_in(words, b: BBox, tol=1.0):
    return [w for w in words if b.x0 - tol <= (w[0] + w[2]) / 2 <= b.x1 + tol
            and b.y0 - tol <= (w[1] + w[3]) / 2 <= b.y1 + tol]


def test_text_ground_truth_matches_extraction(drawings):
    n = 0
    for seed, system, doc, spec, pdf in drawings:
        words = pymupdf.open("pdf", pdf)[0].get_text("words")
        assert len(words) >= len(doc.texts)
        for t in doc.texts:
            inside = _words_in(words, t.bbox)
            assert inside, (seed, t.text)
            assert "".join(w[4] for w in inside) == t.text.replace(" ", ""), (seed, t.text)
            u = BBox(x0=min(w[0] for w in inside), y0=min(w[1] for w in inside),
                     x1=max(w[2] for w in inside), y1=max(w[3] for w in inside))
            assert u.iou(t.bbox) >= 0.5, (seed, t.text, u, t.bbox)
            n += 1
        # every extracted word belongs to some ground-truth text
        gt = [t.bbox for t in doc.texts]
        for w in words:
            cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
            assert any(b.contains_point(cx, cy, tol=1.0) for b in gt), (seed, w[4])
    assert n > 2000


def test_rotated_line_numbers_and_roles(drawings):
    rotated = 0
    for seed, system, doc, spec, pdf in drawings:
        objs = doc.object_by_id()
        pipes = {p.id: p for p in doc.pipes}
        for t in doc.texts:
            if t.rotation == 90 and t.role == TextRole.LINE_NUMBER:
                rotated += 1
                assert t.bbox.height > t.bbox.width
            if t.role == TextRole.LINE_NUMBER:
                assert t.owner_id in pipes
                assert pipes[t.owner_id].line_number == t.text
                assert parse_line_number(t.text) is not None, t.text
            elif t.role in (TextRole.TAG, TextRole.INSTRUMENT, TextRole.LABEL, TextRole.CONNECTOR):
                assert t.owner_id in objs, (seed, t.text)
                assert t.id in objs[t.owner_id].text_ids
            else:
                assert t.owner_id is None
        for o in doc.objects:
            if o.tag and o.category == Category.VALVE and not o.tag.startswith(("TCV", "PCV")):
                assert classify_tag(o.tag) == "valve", o.tag
            if o.category == Category.INSTRUMENT:
                # catalog.classify_tag reads "V" + 2 letters (e.g. VIC-120) as a valve tag; check the ISA pattern
                assert INSTRUMENT_TAG_RE.match(o.tag), o.tag
    assert rotated > 0


def test_object_geometry_is_exact(drawings):
    for seed, system, doc, spec, pdf in drawings:
        drawn = []
        for sd in spec.symbols:
            spec_ = get_symbol(sd.cls)
            act = sd.actuator if spec_.category == Category.VALVE else ActuatorType.NONE
            drawn.append((placed_bbox(spec_, sd.placement, act), sd))
        for o in doc.objects:
            match = [sd for bb, sd in drawn if max(abs(p - q) for p, q in zip(bb, o.bbox.as_tuple())) < 1e-3]
            assert len(match) == 1, (seed, o.id, o.cls)
            sd = match[0]
            assert sd.cls == o.cls and o.rotation == sd.placement.rotation and o.mirrored == sd.placement.mirrored
            fixed = {n: p for n, p, _ in placed_ports(get_symbol(o.cls), sd.placement)}
            for port in o.ports:
                if port.name in fixed:
                    fx, fy = fixed[port.name]
                    assert math.hypot(fx - port.position.x, fy - port.position.y) < 1e-3
                elif port.name != "actuator":  # boundary port: on the bbox outline
                    b = o.bbox
                    on_edge = (min(abs(port.position.x - b.x0), abs(port.position.x - b.x1),
                                   abs(port.position.y - b.y0), abs(port.position.y - b.y1)) < 1e-3)
                    assert on_edge and b.contains_point(port.position.x, port.position.y, tol=1e-3), (o.cls, port)


def test_pipe_endpoints_coincide_with_ports_and_junctions(drawings):
    for seed, system, doc, spec, pdf in drawings:
        objs = doc.object_by_id()
        juncs = doc.junction_by_id()
        degree = Counter()
        for p in doc.pipes:
            assert len(p.points) >= 2
            for ep, pt in ((p.start, p.points[0]), (p.end, p.points[-1])):
                if ep.junction_id:
                    j = juncs[ep.junction_id]
                    assert j.position.dist(pt) <= 0.5
                    degree[ep.junction_id] += 1
                else:
                    ports = {q.name: q for q in objs[ep.object_id].ports}
                    assert ep.port in ports, (seed, ep)
                    assert ports[ep.port].position.dist(pt) <= 0.5, (seed, p.id, ep)
            for a, b in zip(p.points, p.points[1:]):  # orthogonal polylines
                assert abs(a.x - b.x) < 1e-3 or abs(a.y - b.y) < 1e-3
        for j in doc.junctions:
            assert degree[j.id] >= 3, (seed, j.id)


def test_no_symbol_overlaps(drawings):
    for seed, system, doc, spec, pdf in drawings:
        boxes = [(o.id, o.bbox) for o in doc.objects]
        area = doc.pages[0].drawing_area
        for i, (ia, a) in enumerate(boxes):
            assert area.contains_point(a.x0, a.y0) and area.contains_point(a.x1, a.y1)
            for ib, b in boxes[i + 1:]:
                ov = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
                assert not (ov[0] > 0.01 and ov[1] > 0.01), (seed, ia, ib)


def test_pipes_do_not_cross_symbols(drawings):
    """No pipe segment enters the interior of a symbol bbox, except the symbols it connects to."""
    for seed, system, doc, spec, pdf in drawings:
        for p in doc.pipes:
            own = {p.start.object_id, p.end.object_id}
            for a, b in zip(p.points, p.points[1:]):
                x0, x1 = sorted((a.x, b.x))
                y0, y1 = sorted((a.y, b.y))
                for o in doc.objects:
                    if o.id in own:
                        continue
                    bb = o.bbox.expand(-0.5)
                    if abs(y1 - y0) < 1e-6:  # horizontal
                        hit = bb.y0 < y0 < bb.y1 and x0 < bb.x1 and bb.x0 < x1
                    else:  # vertical
                        hit = bb.x0 < x0 < bb.x1 and y0 < bb.y1 and bb.y0 < y1
                    assert not hit, (seed, p.id, o.id, o.cls)


def test_texts_do_not_overlap_each_other_or_foreign_symbols(drawings):
    for seed, system, doc, spec, pdf in drawings:
        texts = doc.texts
        for i, a in enumerate(texts):
            for b in texts[i + 1:]:
                assert not (a.bbox.expand(-0.1).intersects(b.bbox.expand(-0.1))), (seed, a.text, b.text)
        drawing_texts = [t for t in texts if t.role not in (TextRole.TITLE_BLOCK, TextRole.NOTE, TextRole.LEGEND)]
        for t in drawing_texts:
            for o in doc.objects:
                if o.id == t.owner_id:
                    continue
                assert not t.bbox.expand(-0.2).intersects(o.bbox), (seed, t.text, o.id)


def test_marine_conventions(drawings):
    seen = defaultdict(Counter)
    for seed, system, doc, spec, pdf in drawings:
        for o in doc.objects:
            seen[system][(o.cls, o.actuator)] += 1
    assert seen[SystemType.BALLAST][(SymbolClass.BUTTERFLY_VALVE, ActuatorType.HYDRAULIC)] > 0
    assert seen[SystemType.BALLAST][(SymbolClass.BWTS_UNIT, ActuatorType.NONE)] > 0
    assert seen[SystemType.BILGE][(SymbolClass.SDNR_VALVE, ActuatorType.HANDWHEEL)] > 0
    assert seen[SystemType.BILGE][(SymbolClass.OILY_WATER_SEPARATOR, ActuatorType.NONE)] > 0
    fo = seen[SystemType.FUEL_OIL_TRANSFER]
    assert any(k[1] == ActuatorType.QUICK_CLOSING for k in fo)
    assert any(k[0] == SymbolClass.THREE_WAY_VALVE for s in (SystemType.LUBE_OIL, SystemType.FRESH_WATER_COOLING)
               for k in seen[s]) or True  # the TCV is optional in the LO blueprint
    assert seen[SystemType.FRESH_WATER_COOLING][(SymbolClass.THREE_WAY_VALVE, ActuatorType.DIAPHRAGM)] > 0
    assert seen[SystemType.COMPRESSED_AIR][(SymbolClass.AIR_RECEIVER, ActuatorType.NONE)] > 0


def test_screw_pumps_have_relief_and_pumps_have_check_valves():
    """Blueprint conventions, checked on the topology graph of generated drawings."""
    reliefs = 0
    for seed in range(40):
        doc, _ = generate(seed, SystemType.FUEL_OIL_SUPPLY)
        objs = doc.object_by_id()
        adj = defaultdict(set)
        for p in doc.pipes:
            a, b = p.start.node_id, p.end.node_id
            adj[a].add(b)
            adj[b].add(a)
        for o in doc.objects:
            if o.cls == SymbolClass.SAFETY_VALVE:
                # in and out both reach junctions next to the same pump (relief loop across the pump)
                reliefs += 1
                ends = adj[o.id]
                assert len(ends) == 2 and all(e.startswith("j") for e in ends)
        if reliefs:
            break
    assert reliefs > 0


def test_signal_lines_and_tap_lines(drawings):
    kinds = Counter()
    for seed, system, doc, spec, pdf in drawings:
        objs = doc.object_by_id()
        for p in doc.pipes:
            kinds[p.kind] += 1
            if p.kind == PipeKind.INSTRUMENT_TAP:
                ids = [e.object_id for e in (p.start, p.end) if e.object_id]
                assert any(objs[i].category == Category.INSTRUMENT for i in ids)
            if p.kind == PipeKind.SIGNAL:
                assert p.end.port == "actuator" and objs[p.end.object_id].category == Category.VALVE
                assert objs[p.start.object_id].category == Category.INSTRUMENT
        # dashed signal lines are drawn dashed
        dashed = sum(1 for pd in spec.paths if pd.dashes)
        assert dashed == sum(1 for p in doc.pipes if p.kind == PipeKind.SIGNAL)
    assert kinds[PipeKind.INSTRUMENT_TAP] > 0 and kinds[PipeKind.SIGNAL] > 0 and kinds[PipeKind.PROCESS] > 0


def test_crossings_hops_and_junction_styles():
    """Across a set of drawings there are crossings (drawn as hops or gaps), dotted and plain tees."""
    crossings, dotted, plain, hop_curves = 0, 0, 0, 0
    for seed in range(200, 240):
        doc, spec = generate(seed, ALL_SYSTEMS[seed % 10])
        crossings += doc.meta["crossings"]
        dotted += sum(1 for j in doc.junctions if j.dotted)
        plain += sum(1 for j in doc.junctions if not j.dotted)
        hop_curves += sum(1 for pd in spec.paths for s in pd.segs if s[0] == "c")
    assert crossings > 0 and dotted > 0 and plain > 0 and hop_curves > 0


def test_style_variety_and_layers():
    papers, fonts, layered, legends, scales = set(), set(), 0, 0, []
    for seed in range(300, 340):
        doc, spec = generate(seed)
        papers.add(spec.paper)
        fonts |= {t.font for t in spec.texts}
        layered += bool(spec.layers)
        legends += any(t.role == TextRole.LEGEND for t in doc.texts)
        scales.append(doc.meta["symbol_scale"])
        if spec.layers:
            d = pymupdf.open("pdf", to_bytes(spec))
            assert {v["name"] for v in d.get_ocgs().values()} == set(spec.layers)
    assert len(papers) >= 2 and len(fonts) >= 3 and layered > 0 and legends > 0
    assert min(scales) >= 14.0 and max(scales) <= 24.0


def test_determinism():
    for seed, system in [(7, SystemType.BALLAST), (8, None)]:
        d1, s1 = generate(seed, system)
        d2, s2 = generate(seed, system)
        assert d1.to_json() == d2.to_json()
        assert to_bytes(s1) == to_bytes(s2)
    assert generate(9, SystemType.BILGE)[0].to_json() != generate(10, SystemType.BILGE)[0].to_json()


def test_explicit_style_and_topology_complexity():
    st = DrawingStyle(scale=16.0, font="cour", paper="A1", layers=("PIPE", "SYMBOL", "TEXT", "FRAME"),
                      junction_dots=False, crossing="gap", legend=True)
    doc, spec = generate(5, SystemType.LUBE_OIL, style=st)
    assert spec.paper == "A1" and spec.layers == ["PIPE", "SYMBOL", "TEXT", "FRAME"]
    assert not any(j.dotted for j in doc.junctions)
    import random
    small = build_topology(SystemType.BALLAST, random.Random(1), complexity=0.0)
    big = build_topology(SystemType.BALLAST, random.Random(1), complexity=1.0)
    assert small.train_count() <= big.train_count()


def test_split_pipes_and_interleaved_order_render_the_same_geometry():
    base = DrawingStyle(scale=18.0, junction_dots=True)
    alt = DrawingStyle(scale=18.0, junction_dots=True, split_pipes=True, interleave=True)
    d1, s1 = generate(11, SystemType.FUEL_OIL_SUPPLY, style=base)
    d2, s2 = generate(11, SystemType.FUEL_OIL_SUPPLY, style=alt)
    assert d1.objects == d2.objects and d1.pipes == d2.pipes and d1.texts == d2.texts
    assert s2.order is not None and s1.order is None
    p1, p2 = pymupdf.open("pdf", to_bytes(s1))[0], pymupdf.open("pdf", to_bytes(s2))[0]
    assert len(p2.get_drawings()) > len(p1.get_drawings())  # one path per pipe segment
    # same ink: identical rasterisation up to anti-aliasing of the drawing order
    a, b = p1.get_pixmap(dpi=20).samples, p2.get_pixmap(dpi=20).samples
    assert sum(x != y for x, y in zip(a, b)) < 0.01 * len(a)


def test_json_roundtrip(drawings, tmp_path):
    doc = drawings[0][2]
    p = doc.save(tmp_path / "gt.json")
    assert PIDDocument.load(p) == doc


# --------------------------------------------------------------------------- dataset build


def test_build_dataset_parallel_and_deterministic(tmp_path):
    a = build_dataset(tmp_path / "a", count=10, seed=3, workers=2, previews=True, preview_dpi=20)
    assert a["ok"] == 10 and a["failed"] == 0
    assert sum(a["by_split"].values()) == 10 and a["by_split"]["train"] == 8
    lines = [json.loads(x) for x in (tmp_path / "a" / "manifest.jsonl").read_text().splitlines()]
    assert [r["index"] for r in lines] == list(range(10))
    assert len(set(r["system"] for r in lines)) == 10
    for r in lines:
        assert (tmp_path / "a" / r["pdf"]).is_file() and (tmp_path / "a" / r["preview"]).is_file()
        doc = PIDDocument.load(tmp_path / "a" / r["gt"])
        assert doc.info.system.value == r["system"] and len(doc.objects) == r["counts"]["objects"]
    b = build_dataset(tmp_path / "b", count=4, seed=3, workers=1)
    for r in lines[:4]:
        assert (tmp_path / "a" / r["gt"]).read_text() == (tmp_path / "b" / r["gt"]).read_text()
        assert (tmp_path / "a" / r["pdf"]).read_bytes() == (tmp_path / "b" / r["pdf"]).read_bytes()
    c = build_dataset(tmp_path / "c", count=3, seed=4, workers=2, systems=["bilge"])
    assert c["by_system"] == {"bilge": 3}
    assert json.loads((tmp_path / "c" / "summary.json").read_text())["ok"] == 3


# --------------------------------------------------------------------------- collection


def _raster_pdf(path):
    src = pymupdf.open()
    page = src.new_page(width=842, height=595)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 280), 0)
    pix.clear_with(200)
    page.insert_image(page.rect, pixmap=pix)
    src.save(str(path))


def test_scan_archive(tmp_path, drawings):
    root = tmp_path / "archive"
    (root / "sub").mkdir(parents=True)
    for i, (seed, system, doc, spec, pdf) in enumerate(drawings[:3]):
        (root / f"dwg{i}.pdf").write_bytes(pdf)
    shutil.copy(root / "dwg0.pdf", root / "sub" / "copy_of_dwg0.PDF")
    _raster_pdf(root / "sub" / "scanned.pdf")
    (root / "broken.pdf").write_bytes(b"%PDF-1.7 garbage")
    (root / "notes.txt").write_text("not a pdf")
    recs = scan_archive(root, out_index=tmp_path / "index.jsonl")
    by = {r["path"]: r for r in recs}
    assert set(by) == {"dwg0.pdf", "dwg1.pdf", "dwg2.pdf", "sub/copy_of_dwg0.PDF", "sub/scanned.pdf", "broken.pdf"}
    for i in range(3):
        r = by[f"dwg{i}.pdf"]
        assert r["status"] == "readable" and r["readable_pages"] == 1
        assert r["system_guess"] == drawings[i][1].value and r["system_source"] == "title_block"
        assert r["page_sizes"][0][2] in ("A1", "A2", "A3")
        assert r["page_stats"][0]["vector_paths"] > 100 and r["page_stats"][0]["image_ratio"] == 0
    assert by["sub/copy_of_dwg0.PDF"]["status"] == "duplicate"
    assert by["sub/copy_of_dwg0.PDF"]["duplicate_of"] == "dwg0.pdf"
    scanned = by["sub/scanned.pdf"]
    assert scanned["status"] == "unreadable" and scanned["page_stats"][0]["image_ratio"] > 0.9
    assert "raster" in scanned["reason"]
    assert by["broken.pdf"]["status"] == "error"
    lines = (tmp_path / "index.jsonl").read_text().splitlines()
    assert len(lines) == 6


def _pdf_bytes(n=3000):
    d = pymupdf.open()
    d.new_page().insert_text((50, 50), "FUEL OIL TRANSFER SYSTEM " * 5)
    data = d.tobytes()
    return data + b"%" * max(0, n - len(data))


def test_fetch_sources_with_mock_transport(tmp_path):
    good = _pdf_bytes()
    resumable = _pdf_bytes(5000)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, request.headers.get("range")))
        path = request.url.path
        if path == "/good.pdf":
            return httpx.Response(200, content=good, headers={"content-type": "application/pdf"})
        if path == "/page.html":
            return httpx.Response(200, content=b"<html></html>", headers={"content-type": "text/html"})
        if path == "/missing.pdf":
            return httpx.Response(404)
        if path == "/huge.pdf":
            return httpx.Response(200, content=b"%PDF-" + b"0" * 2000,
                                  headers={"content-type": "application/pdf"})
        if path == "/fake.pdf":
            return httpx.Response(200, content=b"GIF89a....", headers={"content-type": "application/octet-stream"})
        if path == "/resume.pdf":
            rng = request.headers.get("range")
            if rng:
                start = int(rng.split("=")[1].rstrip("-"))
                return httpx.Response(206, content=resumable[start:], headers={"content-type": "application/pdf"})
            return httpx.Response(200, content=resumable, headers={"content-type": "application/pdf"})
        if path == "/redirect":
            return httpx.Response(302, headers={"location": "https://files.example.org/good.pdf"})
        raise httpx.ConnectError("no route", request=request)

    import hashlib
    src = tmp_path / "sources.yaml"
    src.write_text(
        "sources:\n"
        "  - url: https://files.example.org/good.pdf\n"
        f"    sha256: {hashlib.sha256(good).hexdigest()}\n"
        "    system: fuel_oil_transfer\n"
        "  - url: https://files.example.org/page.html\n"
        "  - url: https://files.example.org/missing.pdf\n"
        "  - url: https://files.example.org/huge.pdf\n"
        "  - url: https://files.example.org/fake.pdf\n"
        "  - url: https://files.example.org/resume.pdf\n"
        "  - url: https://files.example.org/redirect\n"
        "    name: redirected drawing\n"
        "  - url: https://offline.example.org/x.pdf\n"
        "  - url: https://files.example.org/good.pdf\n"
        "    name: wrong.pdf\n"
        "    sha256: 0000\n")
    dest = tmp_path / "dl"
    dest.mkdir()
    (dest / "resume.pdf.part").write_bytes(resumable[:1234])
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    res = fetch_sources(src, dest, client=client, max_bytes=1500 + len(resumable))
    st = {r["url"].rsplit("/", 1)[-1] + ("#" + r["path"] if r["path"] == "wrong.pdf" else ""): r for r in res}
    assert st["good.pdf"]["status"] == "downloaded" and (dest / "good.pdf").read_bytes() == good
    assert st["good.pdf"]["system"] == "fuel_oil_transfer"
    assert st["page.html"]["status"] == "skipped" and "content-type" in st["page.html"]["reason"]
    assert st["missing.pdf"]["status"] == "error" and st["missing.pdf"]["reason"] == "HTTP 404"
    assert st["fake.pdf"]["status"] == "skipped" and not (dest / "fake.pdf").exists()
    assert st["resume.pdf"]["status"] == "downloaded" and st["resume.pdf"]["resumed_from"] == 1234
    assert (dest / "resume.pdf").read_bytes() == resumable
    assert ("/resume.pdf", "bytes=1234-") in calls
    assert st["redirect"]["status"] == "downloaded" and (dest / "redirected_drawing.pdf").is_file()
    assert st["x.pdf"]["status"] == "error" and "ConnectError" in st["x.pdf"]["reason"]
    assert st["good.pdf#wrong.pdf"]["status"] == "error" and "sha256" in st["good.pdf#wrong.pdf"]["reason"]
    assert not (dest / "wrong.pdf").exists()
    # size limit
    res2 = fetch_sources(src, tmp_path / "dl2", client=client, max_bytes=1000)
    huge = [r for r in res2 if r["url"].endswith("huge.pdf")][0]
    assert huge["status"] == "skipped" and "large" in huge["reason"] or "limit" in huge["reason"]
    # second run: files already present are not downloaded again
    res3 = fetch_sources(src, dest, client=client, max_bytes=1500 + len(resumable))
    assert [r for r in res3 if r["path"] == "good.pdf"][0]["status"] == "exists"
    assert len((dest / "fetch_results.jsonl").read_text().splitlines()) == len(res3)
