"""Tests for multi-sheet P&ID sets: generation (shippid.datasets.multisheet), cross-sheet connector
linking (shippid.analysis.sheets) and the ``sheet_link`` edges of shippid.analysis.graph."""

from __future__ import annotations

import re
from collections import Counter

import pymupdf
import pytest

from shippid.analysis.analyzer import analyze
from shippid.analysis.graph import (
    SHEET_LINK,
    FlowNet,
    build_graph,
    components,
    flow_paths,
    isolation_analysis,
    isolation_valves,
    object_adjacency,
    sheet_link_pairs,
    terminal_role,
)
from shippid.analysis.rules import run_rules
from shippid.analysis.sheets import (
    ambiguous_links,
    apply_link_decisions,
    candidate_pairs,
    connector_keys,
    link_sheets,
    linked_pairs,
    name_tokens,
    names_match,
    page_sheet_numbers,
)
from shippid.datasets.multisheet import YARD_STYLES, _merge, generate_multisheet, group_name, pdf_bytes
from shippid.datasets.synthetic import generate
from shippid.evaluate import match_boxes
from shippid.extract.pipeline import extract_pdf
from shippid.model import (
    ActuatorType,
    PageInfo,
    PIDDocument,
    SymbolClass,
    SystemType,
    TextRole,
    Verdict,
)

from analysis_builder import DocBuilder

S = SymbolClass
A = ActuatorType
ST = SystemType

# (system, style, sheets, seed): every style, 2 and 3 sheets, cut return lines (bilge OWS three-way valve ->
# holding tank, condensate -> boiler), several header groups across one cut (bilge)
CASES = [(ST.BILGE, "sheet", 3, 5), (ST.BALLAST, "mark", 2, 4), (ST.STEAM, "dwg", 3, 2)]


@pytest.fixture(scope="module")
def made():
    return {c[1]: generate_multisheet(c[3], c[0], sheets=c[2], style=c[1]) for c in CASES}


@pytest.fixture(scope="module")
def extracted(made, tmp_path_factory):
    out = {}
    d = tmp_path_factory.mktemp("multisheet")
    for style, (gt, pdf) in made.items():
        path = d / f"{style}.pdf"
        path.write_bytes(pdf)
        out[style] = extract_pdf(path)
    return out


def _gt_pairs(doc: PIDDocument) -> set[frozenset[str]]:
    return {frozenset((lk["a"], lk["b"])) for lk in doc.meta["sheet_links"]}


def _unlinked(doc: PIDDocument) -> PIDDocument:
    out = doc.model_copy(deep=True)
    for k in ("sheet_links", "sheet_links_ambiguous", "sheet_link_stats"):
        out.meta.pop(k, None)
    return out


def _obj_map(pred: PIDDocument, gt: PIDDocument) -> dict[str, str]:
    m = {}
    for page in {o.page for o in gt.objects}:
        P = [o for o in pred.objects if o.page == page]
        G = [o for o in gt.objects if o.page == page]
        for pi, gi, _ in match_boxes([o.bbox for o in P], [o.bbox for o in G], 0.5):
            m[P[pi].id] = G[gi].id
    return m


def _cross(doc: PIDDocument, G=None) -> list:
    page = {o.id: o.page for o in doc.objects}
    return [fp for fp in flow_paths(doc, G) if len({page[n] for n in fp.node_ids if n in page}) > 1]


# --------------------------------------------------------------------------- generator


def test_generator_pages_ids_and_links(made):
    for (system, style, n, seed) in CASES:
        gt, pdf = made[style]
        assert [p.index for p in gt.pages] == list(range(n))
        with pymupdf.open("pdf", pdf) as d:
            assert d.page_count == n
        for prefix, items in (("o", gt.objects), ("j", gt.junctions), ("p", gt.pipes), ("t", gt.texts)):
            ids = [x.id for x in items]
            assert ids == [f"{prefix}{i + 1}" for i in range(len(ids))]  # unique, page-ordered numbering
            pages = [x.page for x in items]
            assert pages == sorted(pages) and set(pages) <= set(range(n))
        objs = gt.object_by_id()
        js = gt.junction_by_id()
        for p in gt.pipes:  # pipes stay on their page
            for ep in (p.start, p.end):
                node = objs.get(ep.object_id) or js.get(ep.junction_id)
                assert node is not None and node.page == p.page
        for t in gt.texts:
            if t.owner_id:
                owner = objs.get(t.owner_id) or next(p for p in gt.pipes if p.id == t.owner_id)
                assert owner.page == t.page
        links = gt.meta["sheet_links"]
        assert links and gt.meta["multisheet"]["style"] == style and gt.meta["multisheet"]["sheets"] == n
        used = Counter()
        for lk in links:
            a, b = objs[lk["a"]], objs[lk["b"]]
            assert a.cls == b.cls == S.OFF_PAGE_CONNECTOR and a.page != b.page
            assert a.attributes["direction"] == "out" and b.attributes["direction"] == "in"
            assert lk["sheets"] == [a.page + 1, b.page + 1] and lk["directed"] and lk["source"] == "gt"
            assert "service" not in a.attributes and "service" not in b.attributes  # temporary marker removed
            used.update((a.id, b.id))
        assert max(used.values()) == 1
        # consecutive sheets are joined; every sheet takes part
        assert {p for lk in links for p in lk["sheets"]} == set(range(1, n + 1))


def test_generator_cut_returns(made):
    gt, _ = made["sheet"]  # bilge: the OWS three-way valve recirculates to the holding tank on sheet 1
    ret = [lk for lk in gt.meta["sheet_links"] if lk["kind"] == "return"]
    assert len(ret) == 1 and ret[0]["sheets"] == [3, 1]
    objs = gt.object_by_id()
    tank_in = objs[ret[0]["b"]]
    pipe = next(p for p in gt.pipes if tank_in.id in (p.start.object_id, p.end.object_id))
    assert pipe.start.object_id == tank_in.id and pipe.flow_direction == 1  # flows connector -> tank
    assert objs[pipe.end.object_id].cls == S.TANK
    assert len([lk for lk in gt.meta["sheet_links"] if lk["sheets"] == [1, 2]]) == 2  # two header groups cut


def test_generator_title_blocks_and_texts(made):
    for (system, style, n, seed) in CASES:
        gt, pdf = made[style]
        dwg = gt.meta["multisheet"]["drawing_no"]
        assert gt.info.drawing_no == dwg
        assert [r["sheet"] for r in gt.meta["sheets"]] == list(range(1, n + 1))
        with pymupdf.open("pdf", pdf) as d:
            for k in range(n):
                title = [t.text for t in gt.texts if t.page == k and t.role == TextRole.TITLE_BLOCK]
                assert dwg in title
                assert f"{k + 1} OF {n}" in title or f"{k + 1}/{n}" in title
                words = d[k].get_text()
                assert dwg in words and (f"{k + 1} OF {n}" in words or f"{k + 1}/{n}" in words)
                for t in gt.texts:  # every ground-truth text of the page is printed on it
                    if t.page == k and t.role == TextRole.CONNECTOR:
                        assert t.text in words
        objs = gt.object_by_id()
        for lk in gt.meta["sheet_links"]:
            ta = [t.text for t in gt.texts if t.owner_id == lk["a"]]
            tb = [t.text for t in gt.texts if t.owner_id == lk["b"]]
            i, j = lk["sheets"]
            ra, rb = objs[lk["a"]].attributes.get("ref"), objs[lk["b"]].attributes.get("ref")
            if style == "sheet":
                assert ra == f"SH.{j}" and rb == f"SH.{i}"
            elif style == "mark":
                assert ra == rb and lk["key"] == f"mark:{ra}"
                assert any(t.endswith(f"/ SH.{j}") for t in ta) and any(t.endswith(f"/ SH.{i}") for t in tb)
            else:
                assert ra == str(j) and rb == str(i)
                assert f"CONT. ON DWG {dwg} SH.{j}" in ta and f"CONT. FROM DWG {dwg} SH.{i}" in tb


def test_generator_deterministic_and_unique_refs():
    a = generate_multisheet(11, ST.FRESH_WATER_COOLING, sheets=2, style="mark")
    b = generate_multisheet(11, ST.FRESH_WATER_COOLING, sheets=2, style="mark")
    assert a[1] == b[1] and a[0].to_json() == b[0].to_json()
    gt = a[0]
    refs = Counter(re.sub(r"[^0-9A-Z]", "", o.attributes["ref"]) for o in gt.objects
                   if o.cls == S.OFF_PAGE_CONNECTOR and o.attributes.get("ref"))
    paired = {o for lk in gt.meta["sheet_links"] for o in (lk["a"], lk["b"])}
    shared = {r for r, c in refs.items() if c > 1}
    # only continuation pairs share a reference (a shared mark means a continuation)
    assert all({o.id for o in gt.objects if o.attributes.get("ref") and
                re.sub(r"[^0-9A-Z]", "", o.attributes["ref"]) == r} <= paired for r in shared)
    with pytest.raises(ValueError):
        generate_multisheet(1, sheets=1)
    with pytest.raises(ValueError):
        generate_multisheet(1, style="fancy")
    assert set(YARD_STYLES) == {"sheet", "mark", "dwg"}


def test_group_name():
    assert group_name(["NO.1 BALLAST PUMP", "NO.2 BALLAST PUMP"]) == "BALLAST PUMPS"
    assert group_name(["NO.1 W.B.TK (P)", "NO.1 W.B.TK (S)", "OVERBOARD"]) == "W.B.TKS & OVERBOARD"
    assert group_name(["M/E"]) == "M/E" and group_name([None, ""]) is None


# --------------------------------------------------------------------------- linking (ground truth input)


def test_link_sheets_on_ground_truth(made):
    for style, (gt, _) in made.items():
        doc = _unlinked(gt)
        links = link_sheets(doc)
        assert {frozenset((lk["a"], lk["b"])) for lk in links} == _gt_pairs(gt), style
        assert doc.meta["sheet_links_ambiguous"] == []
        for lk in links:  # orientation and metadata
            assert lk["source"] == "rule" and lk["directed"] and 0.5 < lk["confidence"] <= 0.99
            assert gt.object_by_id()[lk["a"]].attributes["direction"] == "out"
        stats = doc.meta["sheet_link_stats"]
        assert stats["linked"] == len(links) and not stats["sheet_numbers_inferred"]
        expect = {"sheet": "sheet:", "mark": "mark:", "dwg": "dwg"}[style]
        assert all(expect in lk["key"] for lk in links)


def test_page_sheet_numbers(made):
    gt, _ = made["dwg"]
    sheets = page_sheet_numbers(gt)
    dwg = re.sub(r"[^0-9A-Z]", "", gt.meta["multisheet"]["drawing_no"])
    assert {p: (v["sheet"], v["of"], v["drawing_no"]) for p, v in sheets.items()} == \
        {0: (1, 3, dwg), 1: (2, 3, dwg), 2: (3, 3, dwg)}


def test_unrelated_drawings_are_not_linked():
    """Two separate single-sheet drawings in one PDF: different drawing numbers, every sheet 1 OF 1,
    connectors to other systems with random references -> nothing is paired."""
    for seeds in ((10_000, 10_001), (10_084, 10_085, 10_086), (10_112, 10_113)):
        parts = [generate(s) for s in seeds]
        doc = _merge(parts)
        assert link_sheets(doc) == [] and doc.meta["sheet_links_ambiguous"] == []


# --------------------------------------------------------------------------- linking (extracted PDFs)


def test_link_sheets_on_extracted_pdfs(made, extracted):
    for style, (gt, _) in made.items():
        pred = extracted[style]
        links = link_sheets(pred)
        m = _obj_map(pred, gt)
        got = {frozenset((m.get(lk["a"]), m.get(lk["b"]))) for lk in links}
        assert got == _gt_pairs(gt), style
        assert pred.meta["sheet_links_ambiguous"] == []
        # flow paths now cross sheets as in the ground truth; without the links none do
        gt_cross = {(fp.source_id, fp.target_id) for fp in _cross(gt)}
        pred_cross = {(m.get(fp.source_id), m.get(fp.target_id)) for fp in _cross(pred)}
        assert gt_cross and pred_cross <= gt_cross and len(pred_cross) >= 0.9 * len(gt_cross)
        assert _cross(_unlinked(pred), build_graph(pred, with_sheet_links=False)) == []


def test_analyze_links_multipage_documents(made, extracted):
    pred = extracted["mark"].model_copy(deep=True)
    for k in ("sheet_links", "sheet_links_ambiguous", "sheet_link_stats"):
        pred.meta.pop(k, None)
    rep = analyze(pred)
    assert rep.meta["sheet_links"] == len(made["mark"][0].meta["sheet_links"]) == len(pred.meta["sheet_links"])
    assert rep.meta["sheet_links_ambiguous"] == 0
    assert "cross-sheet connector links" in rep.summary_en and "도면 간 연결" in rep.summary_ko
    page = {o.id: o.page for o in pred.objects}
    assert any(len({page[n] for n in fp.node_ids if n in page}) > 1 for fp in rep.flow_paths)
    assert any(fp.pipe_ids and any(p.startswith(f"{SHEET_LINK}:") for p in fp.pipe_ids) for fp in rep.flow_paths)
    # a single-page document is analysed exactly as before
    single, _ = generate(3)
    rep1 = analyze(single)
    assert "sheet_links" not in rep1.meta and "sheet_links" not in single.meta and "Sheets:" not in rep1.summary_en


# --------------------------------------------------------------------------- graph semantics (hand-built)


def _two_sheets(*, out_texts=("TO BWTS FILTER",), out_ref="SH.2", in_texts=("FROM BALLAST PUMP",), in_ref="SH.1",
                nrv_on_sheet2: bool = True, title=True) -> tuple[PIDDocument, dict[str, str]]:
    """Sheet 1: tank -> valve -> pump -> valve -> connector C1; sheet 2: connector C2 -> [check valve] ->
    filter -> valve -> tank."""
    b = DocBuilder(ST.BALLAST, title="BALLAST SYSTEM")
    ids = {}
    ids["t1"] = b.obj(S.TANK, 60, 300, name="NO.1 W.B.TK (P)")
    ids["v1"] = b.obj(S.GATE_VALVE, 160, 300, tag="V-001", actuator=A.HANDWHEEL)
    ids["pump"] = b.obj(S.PUMP_CENTRIFUGAL, 300, 300, name="NO.1 BALLAST PUMP", tag="P-01")
    ids["v2"] = b.obj(S.GATE_VALVE, 440, 300, tag="V-002", actuator=A.HANDWHEEL)
    ids["c1"] = b.obj(S.OFF_PAGE_CONNECTOR, 560, 300, attrs={"ref": out_ref} if out_ref else None)
    b.chain(ids["t1"], ids["v1"], ids["pump"], ids["v2"], ids["c1"], flow=1)
    ids["c2"] = b.obj(S.OFF_PAGE_CONNECTOR, 60, 300, rotation=180, page=1,
                      attrs={"ref": in_ref} if in_ref else None)
    chain = [ids["c2"]]
    if nrv_on_sheet2:
        ids["nrv"] = b.obj(S.CHECK_VALVE, 160, 300, tag="V-003", page=1)
        chain.append(ids["nrv"])
    ids["filter"] = b.obj(S.FILTER, 280, 300, name="BWTS FILTER", page=1)
    ids["v4"] = b.obj(S.GATE_VALVE, 400, 300, tag="V-004", actuator=A.HANDWHEEL, page=1)
    ids["t2"] = b.obj(S.TANK, 540, 300, name="NO.2 W.B.TK (P)", page=1)
    chain += [ids["filter"], ids["v4"], ids["t2"]]
    b.chain(*chain, flow=1, page=1)
    for t in out_texts:
        b.text(t, 600, 290, role=TextRole.CONNECTOR, owner=ids["c1"])
    tid = len(b.texts)
    for t in in_texts:
        b.text(t, 0, 290, role=TextRole.CONNECTOR, owner=ids["c2"])
    if title:
        for k in range(2):
            b.text("SHEET", 900, 700, role=TextRole.TITLE_BLOCK)
            b.text(f"{k + 1} OF 2", 900, 710, role=TextRole.TITLE_BLOCK)
            b.text("DWG. NO.", 960, 700, role=TextRole.TITLE_BLOCK)
            b.text("4446-PD-BA-12", 960, 710, role=TextRole.TITLE_BLOCK)
    doc = b.build()
    doc.pages.append(PageInfo(index=1, width=1200, height=800))
    title_k = 0
    for i, t in enumerate(doc.texts):
        if i >= tid and i < tid + len(in_texts):
            t.page = 1
        if t.role == TextRole.TITLE_BLOCK:
            t.page = title_k // 4
            title_k += 1
    return doc, ids


def test_sheet_link_edge_semantics():
    doc, ids = _two_sheets()
    links = link_sheets(doc)
    assert [(lk["a"], lk["b"]) for lk in links] == [(ids["c1"], ids["c2"])]
    assert set(links[0]["key"].split("+")) >= {"sheet:1>2", "name"} and links[0]["directed"]
    G = build_graph(doc)
    edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["kind"] == SHEET_LINK]
    assert len(edges) == 1 and edges[0][2]["flow_direction"] == 1 and edges[0][2]["length"] == 0.0
    assert edges[0][2]["pipe_id"] == f"sheet_link:{ids['c1']}:{ids['c2']}"
    assert terminal_role(G, ids["c1"]) == (False, False) == terminal_role(G, ids["c2"])
    assert FlowNet(doc, G).passable(ids["c1"])
    paths = flow_paths(doc, G)
    assert [(fp.source_id, fp.target_id) for fp in paths] == [(ids["t1"], ids["t2"])]
    assert ids["c1"] in paths[0].node_ids and ids["c2"] in paths[0].node_ids
    assert frozenset((ids["c1"], ids["c2"])) in object_adjacency(doc, G)
    assert len(components(doc, G)) == 1
    # isolation passes the continuation: the filter on sheet 2 is isolated upstream by V-002 on sheet 1
    # (through the check valve and both connectors)
    assert set(isolation_valves(doc, G, ids["filter"])) == {ids["v2"], ids["v4"]}
    branches = isolation_analysis(doc, G, ids["filter"])
    assert all(br.isolated for br in branches)
    # per-sheet graph (and documents without links) unchanged: the incoming connector is an open source
    G0 = build_graph(doc, with_sheet_links=False)
    assert not any(d["kind"] == SHEET_LINK for *_, d in G0.edges(data=True))
    assert len(components(doc, G0)) == 2 and terminal_role(G0, ids["c2"]) == (True, False)
    assert any(ids["c2"] in br.open_to for br in isolation_analysis(doc, G0, ids["filter"]))
    assert linked_pairs(doc) == {frozenset((ids["c1"], ids["c2"]))}


def test_rules_see_the_continuation():
    """The pump's discharge non-return valve is drawn on sheet 2: not found per sheet, found once linked."""
    doc, ids = _two_sheets()
    unlinked = run_rules(doc, rule_ids=["R_PUMP_DISCHARGE_NRV"])
    link_sheets(doc)
    linked = run_rules(doc, rule_ids=["R_PUMP_DISCHARGE_NRV"])
    assert [f.verdict for f in unlinked] == [Verdict.NOT_APPLICABLE] and ids["nrv"] not in unlinked[0].object_ids
    assert [f.verdict for f in linked] == [Verdict.PASS] and ids["nrv"] in linked[0].object_ids


def test_sheet_link_pairs_validation():
    doc, ids = _two_sheets()
    doc.meta["sheet_links"] = [{"a": ids["c1"], "b": "o999"}, {"a": ids["c1"], "b": ids["c1"]}, "junk",
                               {"a": ids["c1"], "b": ids["c2"]}, {"a": ids["c2"], "b": ids["pump"]}]
    assert [(a, b) for a, b, _ in sheet_link_pairs(doc)] == [(ids["c1"], ids["c2"])]
    G = build_graph(doc)
    assert G.edges[ids["c1"], ids["c2"], f"sheet_link:{ids['c1']}:{ids['c2']}"]["flow_direction"] is None


# --------------------------------------------------------------------------- linker rules (hand-built)


def test_continuation_keys_and_rejections():
    # shared mark only (no sheet refs, no names): paired
    doc, ids = _two_sheets(out_texts=(), out_ref="A1", in_texts=(), in_ref="A-1")
    assert [(lk["a"], lk["b"]) for lk in link_sheets(doc)] == [(ids["c1"], ids["c2"])]
    # different marks: never
    doc, _ = _two_sheets(out_texts=("TO BWTS FILTER / SH.2",), out_ref="A1", in_texts=("FROM BALLAST PUMP / SH.1",),
                         in_ref="A2")
    assert link_sheets(doc) == []
    # both outgoing: never
    doc, _ = _two_sheets(in_texts=("TO BALLAST PUMP",))
    assert link_sheets(doc) == []
    # the arrow names another sheet
    doc, _ = _two_sheets(out_ref="SH.3")
    assert link_sheets(doc) == []
    # a reference to a drawing that is not in the document (external system)
    doc, _ = _two_sheets(out_texts=("TO BWTS FILTER", "CONT. ON DWG 4446-PD-FO-03 SH.2"), out_ref="")
    assert link_sheets(doc) == []
    # the same drawing's reference: paired, with the drawing key
    doc, ids = _two_sheets(out_texts=("TO BWTS FILTER", "CONT. ON DWG 4446-PD-BA-12 SH.2"), out_ref="2",
                           in_texts=("FROM BALLAST PUMP", "CONT. FROM DWG 4446-PD-BA-12 SH.1"), in_ref="1")
    links = link_sheets(doc)
    assert len(links) == 1 and "dwg" in links[0]["key"] and "sheet:1>2" in links[0]["key"]
    # one side's text says nothing about the other: no pair
    doc, _ = _two_sheets(out_texts=("TO SLUDGE TK",), out_ref="")
    assert link_sheets(doc) == []


def test_contradicting_names_go_to_review():
    """Sheet references agree but both destination names contradict the equipment beside the partner:
    no automatic link, the pair is listed for review (Gemma) instead."""
    doc, ids = _two_sheets(out_texts=("TO OVERBOARD",), in_texts=("FROM SLUDGE PUMP",))
    assert link_sheets(doc) == []
    amb = {row["id"]: [c["id"] for c in row["candidates"]] for row in doc.meta["sheet_links_ambiguous"]}
    assert amb == {ids["c1"]: [ids["c2"]], ids["c2"]: [ids["c1"]]}
    # one contradicting name only costs score: still linked
    doc, ids = _two_sheets(out_texts=("TO OVERBOARD",))
    links = link_sheets(doc)
    assert [(lk["a"], lk["b"]) for lk in links] == [(ids["c1"], ids["c2"])]
    assert any("not the equipment beside" in e for e in links[0]["evidence"])


def test_sheet_number_parsing_prefers_the_sheet_field():
    doc, _ = _two_sheets()
    for t in doc.texts:
        if t.role == TextRole.TITLE_BLOCK and t.text == "SHEET":
            t.text = "SCALE"  # no SHEET label: a scale "1/100" must not be read as the sheet
    doc.texts[-1].text = "1/100"
    sheets = page_sheet_numbers(doc)
    assert sheets[0]["sheet"] == 1 and sheets[1]["sheet"] == 2 and sheets[1]["of"] == 2


def test_different_drawings_need_a_drawing_reference():
    doc, ids = _two_sheets(out_texts=(), out_ref="A1", in_texts=(), in_ref="A1")
    title = [t for t in doc.texts if t.role == TextRole.TITLE_BLOCK and t.page == 1 and t.text == "4446-PD-BA-12"]
    title[0].text = "4446-PD-BA-13"  # sheet 2 belongs to another drawing: marks are per drawing
    assert link_sheets(doc) == []


def test_same_page_continuations():
    b = DocBuilder(ST.COMPRESSED_AIR)
    rec = b.obj(S.AIR_RECEIVER, 60, 300, name="NO.1 MAIN AIR RECEIVER")
    c1 = b.obj(S.OFF_PAGE_CONNECTOR, 200, 300, attrs={"ref": "S.3"})
    c2 = b.obj(S.OFF_PAGE_CONNECTOR, 200, 500, rotation=180, attrs={"ref": "S.3"})
    eng = b.obj(S.ENGINE, 400, 500, name="M/E")
    b.pipe(rec, c1, flow=1)
    b.pipe(c2, eng, flow=1)
    b.text("TO C.F.W. SYSTEM", 230, 290, role=TextRole.CONNECTOR, owner=c1)
    b.text("FROM C.F.W. SYSTEM", 100, 490, role=TextRole.CONNECTOR, owner=c2)
    doc = b.build()
    # one reference for both arrows to the same other system: not a continuation
    assert link_sheets(doc) == []
    # bare shared marks on one sheet continue the line
    for t in doc.texts:
        t.owner_id = None
    assert [(lk["a"], lk["b"]) for lk in link_sheets(doc)] == [(c1, c2)]
    # names that point at each other also do
    doc.texts[0].owner_id, doc.texts[0].text = c1, "TO M/E"
    doc.texts[1].owner_id, doc.texts[1].text = c2, "FROM MAIN AIR RECEIVER"
    assert [(lk["a"], lk["b"]) for lk in link_sheets(doc)] == [(c1, c2)]


def _ambiguous_doc() -> tuple[PIDDocument, list[str], list[str]]:
    """Two outgoing arrows on sheet 1 and two incoming ones on sheet 2, sheet references only."""
    b = DocBuilder(ST.BILGE)
    outs, ins = [], []
    for i in range(2):
        y = 200 + 200 * i
        t = b.obj(S.TANK, 60, y, name=f"BILGE WELL {i}")
        c = b.obj(S.OFF_PAGE_CONNECTOR, 300, y, attrs={"ref": "SH.2"})
        b.pipe(t, c, flow=1)
        b.text("TO SH.2", 330, y - 10, role=TextRole.CONNECTOR, owner=c)
        outs.append(c)
    for i in range(2):
        y = 200 + 200 * i
        c = b.obj(S.OFF_PAGE_CONNECTOR, 60, y, rotation=180, attrs={"ref": "SH.1"}, page=1)
        p = b.obj(S.OVERBOARD, 300, y, name=f"OVERBOARD {i}", page=1)
        b.pipe(c, p, flow=1, page=1)
        ins.append(c)
    doc = b.build()
    doc.pages.append(PageInfo(index=1, width=1200, height=800))
    return doc, outs, ins


def test_ambiguous_links_and_review_decisions():
    doc, outs, ins = _ambiguous_doc()
    assert link_sheets(doc) == []
    amb = doc.meta["sheet_links_ambiguous"]
    assert {row["id"] for row in amb} == set(outs + ins)
    assert all({c["id"] for c in row["candidates"]} == (set(ins) if row["id"] in outs else set(outs)) for row in amb)
    view = ambiguous_links(doc)
    assert len(view) == 4 and view[0]["connector"]["sheet_refs"] == [2]
    assert {c["id"] for c in view[0]["candidates"]} == set(ins) and "texts" in view[0]["candidates"][0]
    assert len(candidate_pairs(doc)) == 4
    # a decision must be one of the listed candidates; each connector is linked once
    added = apply_link_decisions(doc, [{"a": outs[0], "b": "o999"}, {"a": outs[0], "b": outs[1]},
                                       {"a": outs[0], "b": ins[1], "confidence": 0.8, "reason": "same header"},
                                       {"a": outs[1], "b": ins[1]}])
    assert [(lk["a"], lk["b"], lk["source"]) for lk in added] == [(outs[0], ins[1], "gemma")]
    assert {row["id"] for row in doc.meta["sheet_links_ambiguous"]} == {outs[1], ins[0]}
    # the rule pass keeps reviewer decisions and resolves the rest by elimination
    links = link_sheets(doc)
    assert {(lk["a"], lk["b"], lk["source"]) for lk in links} == {(outs[0], ins[1], "gemma"), (outs[1], ins[0], "rule")}
    assert doc.meta["sheet_links_ambiguous"] == []


def test_name_matching():
    assert name_tokens("NO.1 & 2 BALLAST PUMPS (P)") == {"BALLAST", "PUMP"}
    assert names_match("BALLAST PUMPS", name_tokens("NO.2 BALLAST PUMP"))
    assert names_match("W.B.TKS & F.P.TK", name_tokens("NO.1 W.B.TK (S)") | name_tokens("F.P.TK"))
    assert not names_match("OWS FEED PUMP", name_tokens("NO.1 BILGE, FIRE & G/S PUMP"))
    assert not names_match("PUMPS", name_tokens("NO.1 BILGE PUMP"))  # generic words alone do not identify


def test_connector_keys_read_texts(made):
    gt, _ = made["dwg"]
    keys = connector_keys(_unlinked(gt))
    lk = gt.meta["sheet_links"][0]
    a, b = keys[lk["a"]], keys[lk["b"]]
    assert a.direction == "out" and b.direction == "in" and a.direction_hard
    assert a.sheet_refs == {lk["sheets"][1]} and b.sheet_refs == {lk["sheets"][0]}
    dwg = re.sub(r"[^0-9A-Z]", "", gt.meta["multisheet"]["drawing_no"])
    assert a.dwg_refs == {dwg} == b.dwg_refs and a.mark == str(lk["sheets"][1])
    assert a.names and b.names and a.near and b.near


def test_multipage_pdf_writer_is_deterministic():
    parts = [generate(1), generate(2)]
    one, two = pdf_bytes([s for _, s in parts]), pdf_bytes([s for _, s in parts])
    assert one == two
    with pymupdf.open("pdf", one) as d:
        assert d.page_count == 2 and d.metadata["title"] == parts[0][1].metadata["title"]
