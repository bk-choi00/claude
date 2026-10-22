"""Tests for shippid.datasets: synthetic generator, PDF writer, dataset build, collection."""

import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import pymupdf
import pytest

from shippid.catalog import INSTRUMENT_TAG_RE, classify_tag, detect_system, parse_line_number
from shippid.datasets import build_dataset, fetch_sources, generate, scan_archive, to_bytes
from shippid.datasets import synthetic
from shippid.datasets.blueprints import build_topology
from shippid.datasets.collect import SIDECAR_SUFFIX, part_path, target_name
from shippid.datasets.synthetic import ALL_SYSTEMS, FORCED_MIN_SCALE, DrawingStyle
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
        # no ground truth in any (decompressed) object or stream: JSON keys, ids, generator markers
        blobs = [d.xref_object(x).encode() + (d.xref_stream(x) or b"" if d.xref_is_stream(x) else b"")
                 for x in range(1, d.xref_length())]
        payload = b"\n".join(blobs)
        for marker in (b'"owner_id"', b'"text_ids"', b'"attributes"', b'"flow_direction"', b"synthetic:",
                       b"shippid.datasets.synthetic", doc.objects[0].id.encode() + b'"', b"labels_dropped"):
            assert marker not in payload, (seed, marker)
        assert b"BT" in payload and b" Tf" in payload  # the scan sees inflated content streams


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
    # the cooler TCV is mandatory on central F.W. cooling (optional in the L.O. blueprint)
    assert seen[SystemType.FRESH_WATER_COOLING][(SymbolClass.THREE_WAY_VALVE, ActuatorType.DIAPHRAGM)] > 0
    for seed, system, doc, spec, pdf in drawings:
        if system == SystemType.FRESH_WATER_COOLING:
            tcvs = [o for o in doc.objects if o.cls == SymbolClass.THREE_WAY_VALVE]
            assert len(tcvs) == 1 and tcvs[0].tag.startswith("TCV-"), seed
    assert seen[SystemType.COMPRESSED_AIR][(SymbolClass.AIR_RECEIVER, ActuatorType.NONE)] > 0


def _flow_graph(doc):
    """Directed process pipes: (object id | junction id) -> list of (pipe, start port)."""
    out = defaultdict(list)
    for p in doc.pipes:
        if p.kind == PipeKind.PROCESS and p.flow_direction == 1:
            out[p.start.node_id].append(p)
    return out


def _walk(doc, pipes, out):
    """Node ids (objects and junctions) passed going downstream from ``pipes`` along the straight run
    of a train (instrument taps, relief and drain branches ignored) up to a header (merge, or a fork
    without a straight continuation) or a dead end."""
    objs = doc.object_by_id()

    def heading(a, b):
        return (b.x > a.x) - (b.x < a.x), (b.y > a.y) - (b.y < a.y)

    incoming = Counter(p.end.junction_id for ps in out.values() for p in ps if p.end.junction_id)
    seen = []
    while len(pipes) == 1:
        p = pipes[0]
        nxt = p.end.node_id
        if nxt in seen:
            break
        seen.append(nxt)
        if p.end.object_id:
            pipes = list(out[nxt])
            continue
        if incoming[nxt] > 1:
            break  # header merge
        cands = [q for q in out[nxt] if not (q.end.object_id and objs[q.end.object_id].cls == SymbolClass.SAFETY_VALVE)]
        d = heading(p.points[-2], p.points[-1])
        pipes = cands if len(cands) == 1 else [q for q in cands if heading(q.points[0], q.points[1]) == d]
    return seen


def _downstream_train(doc, obj_id, port, out):
    objs = doc.object_by_id()
    return [objs[n] for n in _walk(doc, [p for p in out[obj_id] if p.start.port == port], out) if n in objs]


def _until_equipment(doc, nodes):
    objs = doc.object_by_id()
    return nodes[:next((i for i, n in enumerate(nodes) if n in objs and objs[n].category == Category.EQUIPMENT),
                       len(nodes)) + 1]


PUMPS = (SymbolClass.PUMP_CENTRIFUGAL, SymbolClass.PUMP_SCREW)
DISCHARGE = {SymbolClass.PUMP_CENTRIFUGAL: "discharge", SymbolClass.PUMP_SCREW: "out"}
SUCTION = {SymbolClass.PUMP_CENTRIFUGAL: "suction", SymbolClass.PUMP_SCREW: "in"}


def test_screw_pumps_have_relief_and_pumps_have_check_valves():
    """Blueprint conventions, checked on the pipe graph of generated drawings: every relief valve spans
    its own screw pump (inlet tee on the discharge side, outlet tee on the suction side) and every
    pump of a pump stage that requires a non-return valve passes one before its discharge header."""
    reliefs = pumps = 0
    for seed in range(400, 430):
        system = (SystemType.FUEL_OIL_SUPPLY, SystemType.FUEL_OIL_TRANSFER, SystemType.BILGE,
                  SystemType.FIRE_MAIN, SystemType.STEAM, SystemType.LUBE_OIL)[seed % 6]
        doc, _ = generate(seed, system)
        objs = doc.object_by_id()
        out = _flow_graph(doc)
        by_port = {}
        for p in doc.pipes:
            for ep, other in ((p.start, p.end), (p.end, p.start)):
                if ep.object_id:
                    by_port[(ep.object_id, ep.port)] = other
        for o in doc.objects:
            if o.cls == SymbolClass.SAFETY_VALVE and system in (SystemType.FUEL_OIL_SUPPLY,
                                                                  SystemType.FUEL_OIL_TRANSFER):
                reliefs += 1
                j_in, j_out = by_port[(o.id, "in")], by_port[(o.id, "out")]
                assert j_in.junction_id and j_out.junction_id, (seed, o.id)
                # the suction-side tee leads (along the train) into a screw pump, and that pump's
                # outlet reaches the discharge-side tee before any other equipment
                ahead = _until_equipment(doc, _walk(doc, list(out[j_out.junction_id]), out))
                pump = objs[ahead[-1]]
                assert pump.cls == SymbolClass.PUMP_SCREW, (seed, o.id, ahead)
                assert next(p for p in doc.pipes if p.end.object_id == pump.id).end.port == "in"
                after = _until_equipment(doc, _walk(doc, [p for p in out[pump.id] if p.start.port == "out"], out))
                assert j_in.junction_id in after, (seed, o.id, pump.id, after)
            if o.cls in PUMPS:
                chain = _downstream_train(doc, o.id, DISCHARGE[o.cls], out)
                if system == SystemType.LUBE_OIL and any(c.cls == SymbolClass.PURIFIER for c in chain):
                    continue  # purifier feed pump: no non-return valve in the blueprint
                pumps += 1
                # on the pump's own discharge line: before the train reaches any other equipment
                own = chain[:next((i for i, c in enumerate(chain) if c.category == Category.EQUIPMENT), len(chain))]
                checks = [i for i, c in enumerate(own) if c.cls in (SymbolClass.CHECK_VALVE, SymbolClass.SDNR_VALVE)]
                assert checks, (seed, system, o.id, [c.cls.value for c in chain])
                # ... followed by the discharge isolation valve
                assert any(c.category == Category.VALVE for c in own[checks[0] + 1:]), (seed, o.id)
    assert reliefs >= 5 and pumps >= 30, (reliefs, pumps)


def test_control_valves_share_the_loop_number_of_their_controller():
    """ISA 5.1: TIC-120 drives TCV-120 (one loop number per control loop)."""
    loops = Counter()
    for seed in range(500, 530):
        system = (SystemType.FRESH_WATER_COOLING, SystemType.LUBE_OIL, SystemType.COMPRESSED_AIR)[seed % 3]
        doc, _ = generate(seed, system)
        objs = doc.object_by_id()
        for p in doc.pipes:
            if p.kind != PipeKind.SIGNAL:
                continue
            ctrl, valve = objs[p.start.object_id], objs[p.end.object_id]
            c_letters, c_num = ctrl.tag.split("-")
            v_letters, v_num = valve.tag.split("-")
            assert c_num == v_num, (seed, ctrl.tag, valve.tag)
            assert c_letters[0] == v_letters[0] and c_letters.endswith("C") and v_letters.endswith("CV")
            assert ctrl.attributes["loop"] == valve.attributes["loop"] == c_num
            loops[v_letters] += 1
    assert loops["TCV"] >= 10 and loops["PCV"] >= 3, loops


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


def _gap(a: BBox, b: BBox) -> float:
    return math.hypot(max(0.0, a.x0 - b.x1, b.x0 - a.x1), max(0.0, a.y0 - b.y1, b.y0 - a.y1))


def test_names_and_ratings_in_ground_truth_are_printed(drawings):
    """GT reflects the page: ``name`` / ``rating`` only when that text is drawn and owned by the object."""
    undrawn = 0
    for seed, system, doc, spec, pdf in drawings:
        tx = doc.text_by_id()
        for o in doc.objects:
            owned = {tx[i].text for i in o.text_ids}
            for key in ("name", "rating"):
                if key in o.attributes:
                    assert o.attributes[key] in owned, (seed, o.id, key, o.attributes[key], owned)
                if f"{key}_undrawn" in o.attributes:
                    undrawn += 1
                    assert o.attributes[f"{key}_undrawn"] not in owned
                    assert key not in o.attributes
    assert sum(1 for d in drawings for o in d[2].objects if "name" in o.attributes) > 200
    # the generator records what the label placer could not print (it does drop some)
    assert undrawn == sum(1 for d in drawings for o in d[2].objects for k in o.attributes if k.endswith("_undrawn"))


def test_texts_are_nearest_to_their_owner(drawings):
    """Tags / labels / connector labels sit nearer (by box gap) to their owner than to any other object,
    and line numbers nearer to their own pipe than to any other pipe -- as the extractor associates them."""
    n_obj = n_line = 0
    for seed, system, doc, spec, pdf in drawings:
        objs = doc.object_by_id()
        for t in doc.texts:
            if t.role in (TextRole.TAG, TextRole.LABEL, TextRole.CONNECTOR) and t.owner_id in objs:
                own = objs[t.owner_id]
                g0 = _gap(t.bbox, own.bbox)
                if g0 == 0.0:
                    continue  # written inside its owner (tank names, connector references)
                n_obj += 1
                near = min((_gap(t.bbox, o.bbox), o.id) for o in doc.objects if o.id != own.id)
                assert near[0] > g0 + 0.25, (seed, t.text, own.id, round(g0, 2), near)
            elif t.role == TextRole.LINE_NUMBER:
                n_line += 1

                def pgap(p):
                    return min(_gap(t.bbox, BBox(x0=min(a.x, b.x), y0=min(a.y, b.y), x1=max(a.x, b.x),
                                                 y1=max(a.y, b.y))) for a, b in zip(p.points, p.points[1:]))
                own = pgap(next(p for p in doc.pipes if p.id == t.owner_id))
                other = min(pgap(p) for p in doc.pipes if p.id != t.owner_id)
                assert other > own + 0.25, (seed, t.text, own, other)
    assert n_obj > 500 and n_line > 300


def test_offpage_connector_direction_is_recorded(drawings):
    """``symbols.OFF_PAGE_CONNECTOR`` has one port at the flat end, so the arrow always points away
    from the pipe; GT says which lines enter the drawing and that their arrow points against the flow."""
    seen = Counter()
    for seed, system, doc, spec, pdf in drawings:
        for o in doc.objects:
            if o.cls != SymbolClass.OFF_PAGE_CONNECTOR:
                continue
            p = next(p for p in doc.pipes if o.id in (p.start.object_id, p.end.object_id))
            assert p.flow_direction == 1
            entering = p.start.object_id == o.id
            assert o.attributes["direction"] == ("in" if entering else "out"), (seed, o.id)
            d = o.ports[0].direction
            flow = (d.x, d.y) if entering else (-d.x, -d.y)
            tip = (-d.x, -d.y)
            assert o.attributes["arrow_along_flow"] == (flow == tip)
            label = o.attributes.get("name") or o.attributes.get("name_undrawn", "")
            if label.startswith("FROM"):
                assert o.attributes["direction"] == "in"
            seen[o.attributes["direction"]] += 1
    assert seen["in"] > 0 and seen["out"] > 0


def test_forced_paper_fits_or_fails_clearly(monkeypatch):
    # systems that are too wide for A3 at the free-choice minimum scale still come out on A3
    for seed, system in [(0, SystemType.BALLAST), (1, SystemType.BALLAST), (0, SystemType.BILGE),
                         (2, SystemType.BILGE), (4, SystemType.FUEL_OIL_SUPPLY), (0, SystemType.FIRE_MAIN)]:
        doc, spec = generate(seed, system, style=DrawingStyle(paper="A3"))
        assert spec.paper == "A3" and doc.meta["paper"] == "A3"
        assert (round(spec.width), round(spec.height)) == (1191, 842)
        assert FORCED_MIN_SCALE <= doc.meta["symbol_scale"] <= 18.0
        assert doc.meta["sheet_layout"] in ("right", "bottom")
    doc, spec = generate(3, SystemType.BALLAST, style=DrawingStyle(paper="A2", scale=24.0))
    assert spec.paper == "A2"
    with pytest.raises(ValueError, match="paper"):
        generate(1, SystemType.BILGE, style=DrawingStyle(paper="A4"))
    # no shrinking allowed -> the error names the requested sheet, not A1
    monkeypatch.setattr(synthetic, "FORCED_MIN_SCALE", 40.0)
    with pytest.raises(RuntimeError) as exc:
        generate(0, SystemType.BALLAST, style=DrawingStyle(paper="A3", scale=40.0))
    msg = str(exc.value)
    assert "does not fit A3" in msg and "does not fit A1" not in msg and "paper='A3' is forced" in msg


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


def _rewrap_rotated(pdf: bytes, rot: int) -> bytes:
    """The landscape sheet placed on portrait media with /Rotate (as CAD plots often are)."""
    src = pymupdf.open("pdf", pdf)
    w, h = src[0].rect.width, src[0].rect.height
    out = pymupdf.open()
    page = out.new_page(width=h, height=w)
    page.show_pdf_page(page.rect, src, 0, rotate=rot)
    page.set_rotation(rot)
    return out.tobytes()


def test_scan_archive_rotated_pages(tmp_path, drawings):
    root = tmp_path / "rot"
    root.mkdir()
    seed, system, doc, spec, pdf = drawings[1]
    for rot in (90, 270):
        data = _rewrap_rotated(pdf, rot)
        page = pymupdf.open("pdf", data)[0]
        assert page.rotation == rot and page.rect.width > page.rect.height  # displayed landscape
        (root / f"rot{rot}.pdf").write_bytes(data)
    # full-page scan on rotated portrait media
    src = pymupdf.open()
    page = src.new_page(width=595, height=842)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 283), 0)
    pix.clear_with(180)
    page.insert_image(page.rect, pixmap=pix, keep_proportion=False)
    page.set_rotation(90)
    src.save(str(root / "scan_rot.pdf"))
    by = {r["path"]: r for r in scan_archive(root)}
    for rot in (90, 270):
        r = by[f"rot{rot}.pdf"]
        assert r["status"] == "readable"
        assert r["system_guess"] == system.value and r["system_source"] == "title_block", r["title_text"][:200]
        assert doc.info.drawing_no in r["title_text"]
    scan = by["scan_rot.pdf"]
    assert scan["status"] == "unreadable" and scan["page_stats"][0]["image_ratio"] == pytest.approx(1.0, abs=1e-3)


def _pdf_bytes(n=3000):
    d = pymupdf.open()
    d.new_page().insert_text((50, 50), "FUEL OIL TRANSFER SYSTEM " * 5)
    data = d.tobytes()
    return data + b"%" * max(0, n - len(data))


def _public(host, port):
    """Test resolver (no DNS in tests): every host name is a public address."""
    return ["93.184.215.14"]


def _yaml(path, entries):
    lines = ["sources:"]
    for e in entries:
        lines.append(f"  - url: '{e['url']}'")
        for k, v in e.items():
            if k != "url":
                lines.append(f"    {k}: '{v}'")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_fetch_sources_with_mock_transport(tmp_path):
    import hashlib

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
                return httpx.Response(206, content=resumable[start:], headers={
                    "content-type": "application/pdf",
                    "content-range": f"bytes {start}-{len(resumable) - 1}/{len(resumable)}"})
            return httpx.Response(200, content=resumable, headers={"content-type": "application/pdf"})
        if path == "/redirect":
            return httpx.Response(302, headers={"location": "https://files.example.org/good.pdf"})
        raise httpx.ConnectError("no route", request=request)

    base = "https://files.example.org"
    entries = [
        {"url": f"{base}/good.pdf", "sha256": hashlib.sha256(good).hexdigest(), "system": "fuel_oil_transfer"},
        {"url": f"{base}/page.html"},
        {"url": f"{base}/missing.pdf"},
        {"url": f"{base}/huge.pdf"},
        {"url": f"{base}/fake.pdf"},
        {"url": f"{base}/resume.pdf"},
        {"url": f"{base}/redirect", "name": "redirected drawing"},
        {"url": "https://offline.example.org/x.pdf"},
        {"url": f"{base}/good.pdf?copy=2", "name": "wrong.pdf", "sha256": "0000"},
    ]
    src = _yaml(tmp_path / "sources.yaml", entries)
    names = [target_name(e) for e in entries]
    assert names[0].startswith("good_") and names[0].endswith(".pdf") and names[6] == "redirected_drawing.pdf"
    dest = tmp_path / "dl"
    dest.mkdir()
    part_path(dest / names[5], entries[5]["url"]).write_bytes(resumable[:1234])
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    res = fetch_sources(src, dest, client=client, resolver=_public, max_bytes=1500 + len(resumable))
    st = {r["url"].rsplit("/", 1)[-1]: r for r in res}
    assert [r["path"] for r in res] == names
    assert st["good.pdf"]["status"] == "downloaded" and (dest / names[0]).read_bytes() == good
    assert st["good.pdf"]["system"] == "fuel_oil_transfer"
    assert json.loads((dest / (names[0] + SIDECAR_SUFFIX)).read_text())["url"] == entries[0]["url"]
    assert st["page.html"]["status"] == "skipped" and "content-type" in st["page.html"]["reason"]
    assert st["missing.pdf"]["status"] == "error" and st["missing.pdf"]["reason"] == "HTTP 404"
    assert st["fake.pdf"]["status"] == "skipped" and not (dest / names[4]).exists()
    assert st["resume.pdf"]["status"] == "downloaded" and st["resume.pdf"]["resumed_from"] == 1234
    assert (dest / names[5]).read_bytes() == resumable
    assert ("/resume.pdf", "bytes=1234-") in calls
    assert st["redirect"]["status"] == "downloaded" and (dest / "redirected_drawing.pdf").is_file()
    assert st["x.pdf"]["status"] == "error" and "ConnectError" in st["x.pdf"]["reason"]
    assert st["good.pdf?copy=2"]["status"] == "error" and "sha256" in st["good.pdf?copy=2"]["reason"]
    assert not (dest / "wrong.pdf").exists()
    assert not list(dest.glob("*.part"))  # completed / rejected downloads leave no partial files
    # size limit (declared length -> "too large"; streamed overflow -> "size limit")
    res2 = fetch_sources(src, tmp_path / "dl2", client=client, resolver=_public, max_bytes=1000)
    huge = [r for r in res2 if r["url"].endswith("huge.pdf")][0]
    assert huge["status"] == "skipped" and huge["reason"].startswith("too large"), huge
    assert not list((tmp_path / "dl2").glob(names[3] + "*"))
    # second run: files already fetched from the same URL are not downloaded again
    assert st["huge.pdf"]["status"] == "downloaded"
    n_calls = len(calls)
    res3 = fetch_sources(src, dest, client=client, resolver=_public, max_bytes=1500 + len(resumable))
    assert [r["status"] for r in res3 if r["path"] in (names[0], names[3], names[5], names[6])] == ["exists"] * 4
    # only the sources without a verified local copy are requested again (good.pdf?copy=2 is wrong.pdf)
    assert sorted(path for path, _ in calls[n_calls:]) == sorted(
        ["/page.html", "/missing.pdf", "/fake.pdf", "/x.pdf", "/good.pdf"])
    lines = [json.loads(x) for x in (dest / "fetch_results.jsonl").read_text().splitlines()]
    assert lines == res3


def test_fetch_sources_keeps_sources_apart(tmp_path):
    """Same basename / same path with different queries -> different files; provenance is checked."""
    import hashlib

    def pdf(text):
        d = pymupdf.open()
        d.new_page().insert_text((50, 50), text)
        return d.tobytes()

    bodies = {"/a/pid.pdf": pdf("DRAWING A ballast"), "/b/pid.pdf": pdf("DRAWING B bilge"),
              "/download.php?id=1": pdf("DRAWING C"), "/download.php?id=2": pdf("DRAWING D")}

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.path + (f"?{request.url.query.decode()}" if request.url.query else "")
        if key in bodies:
            return httpx.Response(200, content=bodies[key], headers={"content-type": "application/pdf"})
        if request.url.path == "/badlen.pdf":
            return httpx.Response(200, content=bodies["/a/pid.pdf"],
                                  headers={"content-type": "application/pdf", "content-length": "abc"})
        if request.url.path == "/chunked.pdf":  # no content-length: the limit is enforced while streaming
            return httpx.Response(200, content=iter([b"%PDF-" + b"0" * 600] * 3),
                                  headers={"content-type": "application/pdf"})
        if request.url.path == "/shifted.pdf":  # ignores the requested range start
            return httpx.Response(206, content=bodies["/a/pid.pdf"][10:], headers={
                "content-type": "application/pdf", "content-range": "bytes 10-99/100"})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    keys = list(bodies)
    entries = [{"url": f"https://x.org{k}"} for k in keys]
    entries[1]["sha256"] = hashlib.sha256(bodies[keys[1]]).hexdigest()
    entries.append({"url": "https://x.org/a/pid.pdf"})  # the same URL again
    entries.append({"url": "not a url"})  # unusable URLs are recorded, not fatal (ValueError / InvalidURL)
    entries.append({"url": "https://x.org/badlen.pdf"})  # malformed content-length
    dest = tmp_path / "dl"
    src = _yaml(tmp_path / "s.yaml", entries)
    src.write_text(src.read_text() + '  - url: "http://exa\\0mple.org/x.pdf"\n')
    res = fetch_sources(src, dest, client=client, resolver=_public)
    assert [r["status"] for r in res] == ["downloaded"] * 4 + ["duplicate", "error", "downloaded", "error"]
    assert res[5]["reason"].startswith(("ValueError", "UnsupportedProtocol"))
    assert res[7]["url"] == "http://exa\x00mple.org/x.pdf" and res[7]["reason"].startswith("InvalidURL")
    assert len({r["path"].lower() for r in res[:4]}) == 4
    for k, r in zip(keys, res):
        assert (dest / r["path"]).read_bytes() == bodies[k]
        assert r["sha256"] == hashlib.sha256(bodies[k]).hexdigest()
    assert res[4]["duplicate_of"] == 0 and res[4]["sha256"] == res[0]["sha256"]
    # explicit names that collide (case-insensitively) with another source are a configuration error
    bad = _yaml(tmp_path / "bad.yaml", [{"url": "https://x.org/a/pid.pdf", "name": "PID.pdf"},
                                        {"url": "https://x.org/b/pid.pdf", "name": "pid"}])
    with pytest.raises(ValueError, match="share the file name"):
        fetch_sources(bad, tmp_path / "bad", client=client, resolver=_public)
    assert not (tmp_path / "bad" / "pid.pdf").exists()
    bad2 = _yaml(tmp_path / "bad2.yaml", [{"url": "https://x.org/a/pid.pdf", "sha256": "aa"},
                                         {"url": "https://x.org/a/pid.pdf", "sha256": "bb"}])
    with pytest.raises(ValueError, match="different sha256"):
        fetch_sources(bad2, tmp_path / "bad2", client=client, resolver=_public)
    # a file under the target name that was not fetched from this URL is not reported as "exists"
    named = _yaml(tmp_path / "named.yaml", [{"url": "https://x.org/b/pid.pdf", "name": "drawing.pdf"}])
    (dest / "drawing.pdf").write_bytes(bodies[keys[0]])  # stray file, no provenance
    r = fetch_sources(named, dest, client=client, resolver=_public)[0]
    assert r["status"] == "downloaded" and (dest / "drawing.pdf").read_bytes() == bodies[keys[1]]
    assert fetch_sources(named, dest, client=client, resolver=_public)[0]["status"] == "exists"
    moved = _yaml(tmp_path / "moved.yaml", [{"url": "https://x.org/a/pid.pdf", "name": "drawing.pdf"}])
    r = fetch_sources(moved, dest, client=client, resolver=_public)[0]  # same name, different URL now
    assert r["status"] == "downloaded" and (dest / "drawing.pdf").read_bytes() == bodies[keys[0]]
    # a partial file of another URL under the same name is never resumed
    part_path(dest / "drawing.pdf", "https://x.org/b/pid.pdf").write_bytes(bodies[keys[1]][:100])
    (dest / "drawing.pdf").unlink()
    r = fetch_sources(moved, dest, client=client, resolver=_public)[0]
    assert r["status"] == "downloaded" and r["resumed_from"] == 0
    assert (dest / "drawing.pdf").read_bytes() == bodies[keys[0]] and not list(dest.glob("drawing.pdf.*.part"))
    # a server resuming at the wrong offset does not corrupt the file
    shifted = _yaml(tmp_path / "shifted.yaml", [{"url": "https://x.org/shifted.pdf", "name": "shifted.pdf"}])
    part_path(dest / "shifted.pdf", "https://x.org/shifted.pdf").write_bytes(b"%PDF-1.7 partial")
    r = fetch_sources(shifted, dest, client=client, resolver=_public)[0]
    assert r["status"] == "error" and "instead of" in r["reason"] and not (dest / "shifted.pdf").exists()
    chunked = _yaml(tmp_path / "chunked.yaml", [{"url": "https://x.org/chunked.pdf", "name": "chunked.pdf"}])
    r = fetch_sources(chunked, dest, client=client, resolver=_public, max_bytes=1000)[0]
    assert r["status"] == "skipped" and r["reason"].startswith("exceeded size limit")
    assert not list(dest.glob("chunked.pdf*"))


# --------------------------------------------------------------------------- collection hardening


def _rmtree_iterative(top: Path) -> None:
    import os

    stack, dirs = [str(top)], []
    while stack:
        d = stack.pop()
        dirs.append(d)
        with os.scandir(d) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                else:
                    os.unlink(e.path)
    for d in reversed(dirs):
        os.rmdir(d)


def test_scan_archive_stays_inside_root_and_survives_deep_trees(tmp_path, drawings):
    import os

    root = tmp_path / "archive"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    pdf = drawings[0][4]
    (outside / "private.pdf").write_bytes(pdf)
    (outside / "nested.pdf").write_bytes(drawings[1][4])
    (root / "real.pdf").write_bytes(pdf)
    (root / "innocent.pdf").symlink_to("../outside/private.pdf")
    (root / "linked_dir").symlink_to("../outside", target_is_directory=True)
    (root / "dangling.pdf").symlink_to("../outside/missing.pdf")
    os.mkfifo(root / "pipe.pdf")  # reading a FIFO would block the scan forever
    deep = str(root)
    for _ in range(80):  # deeper than max_depth (and, at 1100 levels, than Python's recursion limit)
        deep += "/d"
        os.mkdir(deep)
    (Path(deep) / "too_deep.pdf").write_bytes(drawings[2][4])
    recs = scan_archive(root, out_index=tmp_path / "index.jsonl", max_depth=64)
    by = {r["path"]: r for r in recs}
    assert by["real.pdf"]["status"] == "readable"
    for link in ("innocent.pdf", "linked_dir", "dangling.pdf"):
        assert by[link]["status"] == "skipped" and "symbolic link" in by[link]["reason"], by[link]
        assert by[link]["sha256"] == "" and by[link]["title_text"] == ""  # the target was never read
    assert by["pipe.pdf"]["status"] == "skipped" and "regular file" in by["pipe.pdf"]["reason"]
    over = [r for r in recs if "max_depth" in r["reason"]]
    assert len(over) == 1 and over[0]["status"] == "error" and over[0]["path"].count("/") == 64
    assert not any(r["path"].endswith("too_deep.pdf") or "nested" in r["path"] for r in recs)
    assert len((tmp_path / "index.jsonl").read_text().splitlines()) == len(recs)
    # a very deep tree (beyond the recursion limit) no longer aborts the scan
    try:
        for _ in range(1100):
            deep += "/d"
            os.mkdir(deep)
        recs = scan_archive(root, max_depth=5000)
        assert {r["path"] for r in recs if r["status"] == "readable"} == {"real.pdf",
                                                                           "/".join(["d"] * 80) + "/too_deep.pdf"}
    finally:  # shutil.rmtree (pytest's tmp_path cleanup) recurses and cannot delete a tree this deep
        _rmtree_iterative(root / "d")
    # size limit
    small = scan_archive(root, max_file_bytes=10, max_depth=1)
    assert {r["path"]: r["status"] for r in small}["real.pdf"] == "skipped"


def test_is_public_address():
    from shippid.datasets.collect import is_public_address

    for a in ("93.184.215.14", "8.8.8.8", "2606:4700:4700::1111"):
        assert is_public_address(a), a
    for a in ("127.0.0.1", "127.0.0.2", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "100.64.0.1",
              "0.0.0.0", "224.0.0.1", "255.255.255.255", "::1", "fe80::1", "fd00::1", "::ffff:10.0.0.1",
              "::ffff:127.0.0.1", "2002:0a00:0001::1", "64:ff9b::a00:1", "not an ip", ""):
        assert not is_public_address(a), a


def test_fetch_sources_refuses_private_hosts_and_bounds_downloads(tmp_path):
    import time

    good = _pdf_bytes()
    hops = []

    def handler(request: httpx.Request) -> httpx.Response:
        hops.append(str(request.url))
        host, path = request.url.host, request.url.path
        if path == "/to-loopback.pdf":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:8080/internal.pdf"})
        if path == "/to-metadata":
            return httpx.Response(301, headers={"location": "http://169.254.169.254/latest/meta-data/"})
        if path == "/to-intranet.pdf":
            return httpx.Response(307, headers={"location": "https://intranet.corp/drawing.pdf"})
        if path == "/to-file":
            return httpx.Response(302, headers={"location": "file:///etc/passwd"})
        if path == "/loop":
            return httpx.Response(302, headers={"location": "/loop"})
        if path == "/relative":
            return httpx.Response(302, headers={"location": "/good.pdf"})
        if path in ("/good.pdf", "/drawing.pdf", "/internal.pdf"):
            return httpx.Response(200, content=good, headers={"content-type": "application/pdf"})
        if path == "/drip.pdf":
            def drip():
                yield b"%PDF-"
                for _ in range(20):
                    time.sleep(0.05)
                    yield b"x"
            return httpx.Response(200, content=drip(), headers={"content-type": "application/pdf"})
        return httpx.Response(404)

    def resolver(host, port):
        return {"intranet.corp": ["10.0.0.5"], "dual.example.org": ["93.184.215.14", "192.168.0.9"]}.get(
            host, ["93.184.215.14"])

    base = "https://files.example.org"
    entries = [{"url": f"{base}/to-loopback.pdf", "name": "a"}, {"url": f"{base}/to-metadata", "name": "b"},
               {"url": f"{base}/to-intranet.pdf", "name": "c"}, {"url": f"{base}/to-file", "name": "d"},
               {"url": f"{base}/loop", "name": "e"}, {"url": f"{base}/relative", "name": "f"},
               {"url": "http://10.1.2.3/direct.pdf", "name": "g"}, {"url": "https://dual.example.org/x.pdf", "name": "h"},
               {"url": "ftp://files.example.org/good.pdf", "name": "i"}]
    src = _yaml(tmp_path / "s.yaml", entries)
    # the client follows redirects itself: fetch_sources must not let it (it checks every hop)
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    res = {r["path"][0]: r for r in fetch_sources(src, tmp_path / "dl", client=client, resolver=resolver)}
    for k, needle in (("a", "127.0.0.1"), ("b", "169.254.169.254"), ("c", "intranet.corp"), ("g", "10.1.2.3"),
                      ("h", "192.168.0.9")):
        assert res[k]["status"] == "skipped" and res[k]["reason"].startswith("refused: ") and needle in res[k]["reason"]
    assert {httpx.URL(h).host for h in hops} == {"files.example.org"}, hops  # no request reached a private host
    assert "content_type" not in res["b"]  # nothing of the internal answer leaks into the record
    assert res["d"]["status"] == "error" and "scheme 'file'" in res["d"]["reason"]
    assert res["e"]["status"] == "error" and res["e"]["reason"].startswith("TooManyRedirects")
    assert hops.count(f"{base}/loop") == 6  # the first request + 5 redirects
    assert res["f"]["status"] == "downloaded" and res["f"]["final_url"] == f"{base}/good.pdf"
    side = json.loads((tmp_path / "dl" / ("f.pdf" + SIDECAR_SUFFIX)).read_text())
    assert side["url"] == f"{base}/relative" and side["final_url"] == f"{base}/good.pdf"
    assert res["i"]["status"] == "error" and "scheme 'ftp'" in res["i"]["reason"]
    assert not list((tmp_path / "dl").glob("[abcdeghi].pdf"))
    # intranet sources are allowed explicitly
    res = {r["path"][0]: r for r in fetch_sources(src, tmp_path / "dl2", client=client, resolver=resolver,
                                                    allow_private=True)}
    assert res["a"]["status"] == "downloaded" and res["a"]["final_url"] == "http://127.0.0.1:8080/internal.pdf"
    assert res["c"]["status"] == "downloaded"
    # total deadline and minimum rate: a dripping server cannot hold a download open
    drip = _yaml(tmp_path / "drip.yaml", [{"url": f"{base}/drip.pdf", "name": "drip"}])
    t0 = time.monotonic()
    r = fetch_sources(drip, tmp_path / "dl3", client=client, resolver=resolver, max_seconds=0.3)[0]
    assert time.monotonic() - t0 < 0.9
    assert r["status"] == "error" and "exceeded 0.3 s" in r["reason"]
    assert list((tmp_path / "dl3").glob("drip.pdf.*.part"))  # kept for a later resume
    r = fetch_sources(drip, tmp_path / "dl4", client=client, resolver=resolver, min_rate=1e6, rate_grace=0.1)[0]
    assert r["status"] == "error" and "slower than" in r["reason"]


def test_multisheet_sets_keep_one_paper_size():
    from shippid.datasets.multisheet import generate_multisheet

    # seeds whose later sheet did not fit the first sheet's paper used to come out mixed
    # (26: A3 + A1, 81: A3 + A2 + A3); the whole set now moves to the larger paper
    for seed in (26, 81):
        gt, pdf = generate_multisheet(seed)
        assert len(set(gt.meta["paper"])) == 1, gt.meta["paper"]
        assert len({(p.width, p.height) for p in gt.pages}) == 1
        with pymupdf.open("pdf", pdf) as d:
            assert len({(round(pg.rect.width), round(pg.rect.height)) for pg in d}) == 1
        assert gt.meta["sheet_links"] and gt.meta["multisheet"]["sheets_requested"] == gt.meta["multisheet"]["sheets"]
    with pytest.raises(ValueError, match="sheets must be 2..3"):
        generate_multisheet(22, sheets=4)
    with pytest.raises(ValueError):
        generate_multisheet(22, sheets=1)
