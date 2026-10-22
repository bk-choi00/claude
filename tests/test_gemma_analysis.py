"""GemmaAnalyzer with MockBackend responders (and one OpenAI-compatible HTTP emulation).

The fixture is a small readable vector P&ID drawn with ``symbols.draw_symbol``
plus a hand-built deterministic extraction result with low-confidence symbols,
ambiguous tags, a line number without owner and a title block.
"""

import json
from typing import Protocol, runtime_checkable

import httpx
import pymupdf
import pytest

from shippid.analysis.gemma import GemmaAnalyzer, _adjacency, _vector_candidates, make_analyzer
from shippid.config import LLMConfig
from shippid.llm.base import ImagePart, LLMClient, LLMError, MockBackend
from shippid.llm.openai_compat import OpenAICompatBackend
from shippid.llm.render import png_size
from shippid.model import (
    ActuatorType, AnalysisReport, BBox, Category, DrawingInfo, Endpoint, Finding, FlowPath, Junction, PageInfo,
    PIDDocument, PIDObject, PipeRun, Point, Severity, SymbolClass, SystemType, TextItem, TextRole, Verdict,
)
from shippid.symbols import Placement, category_of, draw_symbol, get_symbol, placed_bbox

W, H = 842.0, 595.0


def _text(page, tid, text, x, y, size, role, owner=None, rotate=0):
    page.insert_text((x, y), text, fontsize=size, fontname="helv")
    w = pymupdf.get_text_length(text, fontname="helv", fontsize=size)
    return TextItem(id=tid, text=text, bbox=BBox(x0=x, y0=y - size * 0.8, x1=x + w, y1=y + size * 0.2),
                    size=size, role=role, owner_id=owner, rotation=rotate)


@pytest.fixture()
def pid(tmp_path):
    """(pdf path, deterministic PIDDocument) of a small ballast P&ID."""
    pdf = pymupdf.open()
    page = pdf.new_page(width=W, height=H)
    shape = page.new_shape()
    drawn = {  # id: (true class, actuator, placement)
        "o1": (SymbolClass.TANK, ActuatorType.NONE, Placement(100, 200, scale=30)),
        "o2": (SymbolClass.GLOBE_VALVE, ActuatorType.HANDWHEEL, Placement(220, 200, scale=18)),
        "o3": (SymbolClass.PUMP_CENTRIFUGAL, ActuatorType.NONE, Placement(340, 200, scale=20)),
        "o4": (SymbolClass.CHECK_VALVE, ActuatorType.NONE, Placement(460, 200, scale=18)),
        "o5": (SymbolClass.BALL_VALVE, ActuatorType.NONE, Placement(220, 262, scale=18)),
        "o6": (SymbolClass.INSTRUMENT_LOCAL, ActuatorType.NONE, Placement(340, 110, scale=24)),
    }
    for cls, act, pl in drawn.values():
        draw_symbol(shape, get_symbol(cls), pl, act)
    pipes_xy = {
        "p1": [(145, 200), (211, 200)], "p2": [(229, 200), (330, 200)],
        "p3": [(340, 190), (340, 170), (400, 170), (400, 200), (451, 200)],
        "p4": [(469, 200), (560, 200)], "p5": [(160, 262), (211, 262)],
    }
    for pts in pipes_xy.values():
        shape.draw_polyline([pymupdf.Point(*p) for p in pts])
        shape.finish(width=1.0, color=(0, 0, 0))
    shape.draw_rect(pymupdf.Rect(600, 480, 830, 585))
    shape.finish(width=1.0, color=(0, 0, 0))
    shape.commit()
    # vector result: o2 mis-classified with low confidence, o3 ambiguous, o4 unknown
    vector = {
        "o1": (SymbolClass.TANK, ActuatorType.NONE, 0.95, None),
        "o2": (SymbolClass.GATE_VALVE, ActuatorType.HANDWHEEL, 0.55,
               [{"cls": "gate_valve", "score": 0.55}, {"cls": "globe_valve", "score": 0.52}, {"cls": "ball_valve", "score": 0.4}]),
        "o3": (SymbolClass.PUMP_CENTRIFUGAL, ActuatorType.NONE, 0.9, [["pump_centrifugal", 0.9], ["purifier", 0.86]]),
        "o4": (SymbolClass.UNKNOWN, ActuatorType.NONE, 0.3, {"check_valve": 0.3, "sdnr_valve": 0.28}),
        "o5": (SymbolClass.BALL_VALVE, ActuatorType.NONE, 0.95, None),
        "o6": (SymbolClass.INSTRUMENT_LOCAL, ActuatorType.NONE, 0.97, None),
    }
    objects = []
    for oid, (cls, act, conf, cands) in vector.items():
        tcls, tact, pl = drawn[oid]
        attrs = {"candidates": cands} if cands is not None else {}
        if oid == "o4":
            attrs["primitives"] = {"lines": 3, "filled_polygons": 1}
        objects.append(PIDObject(id=oid, cls=cls, category=category_of(cls) if cls != SymbolClass.UNKNOWN else Category.EQUIPMENT,
                                 bbox=BBox.from_points([placed_bbox(get_symbol(tcls), pl, tact)[:2],
                                                        placed_bbox(get_symbol(tcls), pl, tact)[2:]]),
                                 actuator=act, confidence=conf, source="vector", attributes=attrs))
    ends = {"p1": (Endpoint(object_id="o1"), Endpoint(object_id="o2", port="in")),
            "p2": (Endpoint(object_id="o2", port="out"), Endpoint(object_id="o3", port="suction")),
            "p3": (Endpoint(object_id="o3", port="discharge"), Endpoint(object_id="o4", port="in")),
            "p4": (Endpoint(object_id="o4", port="out"), Endpoint(junction_id="j1")),
            "p5": (Endpoint(), Endpoint(object_id="o5", port="in"))}
    pipes = [PipeRun(id=pid_, points=[Point(x=x, y=y) for x, y in pts], start=ends[pid_][0], end=ends[pid_][1],
                     source="vector") for pid_, pts in pipes_xy.items()]
    texts = [
        _text(page, "t1", "BA-V001", 232, 236, 7, TextRole.TAG, owner="o5"),  # between o2 and o5 -> ambiguous
        _text(page, "t2", "150A-BA-001", 255, 195, 6, TextRole.LINE_NUMBER),  # on p2, no owner
        _text(page, "t3", "PI 101", 330, 113, 6, TextRole.INSTRUMENT, owner="o6"),  # inside its bubble only
        _text(page, "t4", "NO.1 BALLAST PUMP", 310, 240, 7, TextRole.LABEL, owner="o3"),
        _text(page, "t10", "BALLAST WATER SYSTEM", 610, 500, 10, TextRole.TITLE_BLOCK),
        _text(page, "t11", "DWG. NO.", 610, 520, 6, TextRole.TITLE_BLOCK),
        _text(page, "t12", "1234-BA-001", 660, 520, 8, TextRole.TITLE_BLOCK),
        _text(page, "t13", "HULL NO.", 610, 540, 6, TextRole.TITLE_BLOCK),
        _text(page, "t14", "S1234", 660, 540, 8, TextRole.TITLE_BLOCK),
        _text(page, "t15", "REV.", 740, 520, 6, TextRole.TITLE_BLOCK),
        _text(page, "t16", "A", 770, 520, 8, TextRole.TITLE_BLOCK),
        _text(page, "t17", "174K LNG CARRIER", 610, 565, 8, TextRole.TITLE_BLOCK),
        _text(page, "t18", "1/1", 722, 548, 5, TextRole.TITLE_BLOCK),  # isolated value inside the block
        # extractors also label frame zone markers and the legend panel as title block
        _text(page, "t20", "1", 100, 12, 7, TextRole.TITLE_BLOCK),
        _text(page, "t21", "2", 400, 12, 7, TextRole.TITLE_BLOCK),
        _text(page, "t22", "3", 700, 12, 7, TextRole.TITLE_BLOCK),
        _text(page, "t23", "SYMBOL LEGEND", 650, 60, 7, TextRole.TITLE_BLOCK),
        _text(page, "t24", "GATE VALVE", 650, 75, 6, TextRole.TITLE_BLOCK),
        _text(page, "t25", "GLOBE VALVE", 650, 88, 6, TextRole.TITLE_BLOCK),
    ]
    path = tmp_path / "ballast.pdf"
    pdf.save(path)
    doc = PIDDocument(source=str(path), pages=[PageInfo(index=0, width=W, height=H)], objects=objects,
                      junctions=[Junction(id="j1", position=Point(x=560, y=200), kind="end")], pipes=pipes,
                      texts=texts, info=DrawingInfo(title="BALLAST WATER SYSTEM"))
    return path, doc


def make(responder, **kw):
    backend = MockBackend(responder)
    return GemmaAnalyzer(LLMClient(backend, max_attempts=kw.pop("max_attempts", 3)), **kw), backend


def user_text(msgs):
    return "\n".join(m.text() for m in msgs if m.role == "user")


def digest_of(msgs, marker):
    text = user_text(msgs)
    start = text.index(marker) + len(marker)
    return json.loads(text[start:].split("\n", 1)[0])


SYMBOLS_OK = {"items": [
    {"id": "o2", "reason": "bow-tie with filled centre dot, T-bar stem", "cls": "globe_valve", "actuator": "handwheel", "confidence": 0.9},
    {"id": "o3", "reason": "circle with inscribed triangle", "cls": "pump_centrifugal", "actuator": "none", "confidence": 0.8},
    {"id": "o4", "reason": "bow-tie with filled outlet triangle", "cls": "check_valve", "actuator": "handwheel", "confidence": 0.85},
]}


# --------------------------------------------------------------------------- symbols


def test_vector_candidate_formats(pid):
    _, doc = pid
    objs = doc.object_by_id()
    assert _vector_candidates(objs["o2"])[0] == ("gate_valve", 0.55)
    assert _vector_candidates(objs["o3"]) == [("pump_centrifugal", 0.9), ("purifier", 0.86)]
    assert _vector_candidates(objs["o4"])[0] == ("check_valve", 0.3)
    assert _vector_candidates(objs["o1"]) == []


def test_refine_symbols_merges_and_records_provenance(pid):
    path, doc = pid
    analyzer, backend = make(lambda m, s, n: SYMBOLS_OK)
    original = doc.model_copy(deep=True)
    out = analyzer.refine_symbols(doc, path)
    assert doc == original  # input untouched
    msgs, schema, name = backend.calls[0]
    assert len(backend.calls) == 1 and name == "SymbolBatch" and schema is not None
    assert "gate_valve" in msgs[0].text() and "quick_closing" in msgs[0].text()  # legend in the system prompt
    user = msgs[1]
    assert isinstance(user.parts()[0], ImagePart)  # contact sheet first, then the digest
    w, h = png_size(user.images()[0].data)
    assert w % 48 == 0 and h % 48 == 0 and max(w, h) <= 1536
    digest = digest_of(msgs, "Candidates:\n")
    assert [d["id"] for d in digest] == ["o2", "o3", "o4"] and [d["n"] for d in digest] == [1, 2, 3]
    d2 = digest[0]
    assert d2["vector_class"] == "gate_valve" and len(d2["vector_candidates"]) == 3
    assert d2["connected_pipes"] == 2 and any(t["text"] == "BA-V001" for t in d2["nearby_text"])
    assert digest[2]["primitives"] == {"lines": 3, "filled_polygons": 1}
    objs = out.object_by_id()
    o2, o3, o4 = objs["o2"], objs["o3"], objs["o4"]
    assert (o2.cls, o2.actuator, o2.confidence, o2.source) == (SymbolClass.GLOBE_VALVE, ActuatorType.HANDWHEEL, 0.9, "mock")
    assert o2.attributes["vector_cls"] == "gate_valve" and o2.attributes["vector_confidence"] == 0.55
    assert o2.attributes["refine_decision"] == "adopted"
    assert (o3.cls, o3.source, o3.confidence) == (SymbolClass.PUMP_CENTRIFUGAL, "merged", 0.9)
    assert (o4.cls, o4.category, o4.source) == (SymbolClass.CHECK_VALVE, Category.VALVE, "mock")
    assert o4.actuator == ActuatorType.HANDWHEEL and o4.attributes["vector_cls"] == "unknown"
    assert objs["o1"].source == "vector" and "gemma_cls" not in objs["o1"].attributes
    stats = out.meta["gemma"]["symbols"]
    assert stats == {"source": "mock", "candidates": 3, "batches": 1, "failed_batches": 0, "adopted": 2,
                     "confirmed": 1, "kept_vector": 0, "images": True}
    assert out.meta["llm_usage"]["calls"] == 1 and out.meta["llm_usage"]["backend"] == "mock"
    # refined objects are not sent again
    again = analyzer.refine_symbols(out, path)
    assert len(backend.calls) == 1 and again.meta["gemma"]["symbols"]["candidates"] == 0


def test_refine_symbols_keeps_more_confident_vector_result(pid):
    path, doc = pid
    answer = {"items": [dict(SYMBOLS_OK["items"][0], confidence=0.4), SYMBOLS_OK["items"][1],
                        dict(SYMBOLS_OK["items"][2], cls="unknown", confidence=0.2)]}
    analyzer, _ = make(lambda m, s, n: answer)
    objs = analyzer.refine_symbols(doc, path).object_by_id()
    assert objs["o2"].cls == SymbolClass.GATE_VALVE and objs["o2"].source == "vector"
    assert objs["o2"].attributes["gemma_cls"] == "globe_valve" and objs["o2"].attributes["refine_decision"] == "kept_vector"
    assert objs["o4"].cls == SymbolClass.UNKNOWN and objs["o4"].source == "merged"  # both say unknown


def test_refine_symbols_repairs_invalid_ids(pid):
    path, doc = pid
    bad = {"items": [dict(SYMBOLS_OK["items"][0], id="o99"), SYMBOLS_OK["items"][1]]}
    answers = iter([bad, SYMBOLS_OK])
    analyzer, backend = make(lambda m, s, n: next(answers))
    out = analyzer.refine_symbols(doc, path)
    assert len(backend.calls) == 2
    repair = backend.calls[1][0][-1].text()
    assert "unknown ids ['o99']" in repair and "missing ids ['o2', 'o4']" in repair
    assert out.object_by_id()["o2"].cls == SymbolClass.GLOBE_VALVE
    assert analyzer.usage.errors and "SymbolBatch" in analyzer.usage.errors[0]


def test_refine_symbols_llm_error_keeps_deterministic(pid):
    path, doc = pid
    analyzer, backend = make(lambda m, s, n: "I cannot see the image.", max_attempts=2)
    out = analyzer.refine_symbols(doc, path)
    assert len(backend.calls) == 2
    assert [o.model_dump() for o in out.objects] == [o.model_dump() for o in doc.objects]
    assert out.meta["gemma"]["symbols"]["failed_batches"] == 1
    assert out.meta["llm_errors"][0].startswith("refine_symbols[o2..o4]")


def test_refine_symbols_batches_and_digest_only(pid):
    path, doc = pid

    def responder(m, s, n):
        ids = [d["id"] for d in digest_of(m, "Candidates:\n")]
        return {"items": [i for i in SYMBOLS_OK["items"] if i["id"] in ids]}

    analyzer, backend = make(responder, symbol_batch=2, concurrency=2)
    out = analyzer.refine_symbols(doc, path)
    assert len(backend.calls) == 2 and out.meta["gemma"]["symbols"]["batches"] == 2
    assert out.object_by_id()["o4"].cls == SymbolClass.CHECK_VALVE
    analyzer, backend = make(responder)
    out = analyzer.refine_symbols(doc, None)  # no PDF: digest-only prompt, recorded
    assert backend.calls[0][0][1].images() == [] and out.meta["gemma"]["symbols"]["images"] is False
    assert "digest-only" in out.meta["llm_notes"][0]


# --------------------------------------------------------------------------- drawing info


INFO_OK = {"drawing_no": "1234-BA-001", "title": "BALLAST WATER SYSTEM", "system": "ballast", "systems": ["ballast"],
           "hull_no": "S1234", "ship_type": "174K LNG CARRIER", "revision": "A", "sheet": None, "scale": None,
           "evidence_text_ids": ["t10", "t12", "t14", "t16", "t17"]}


def test_read_drawing_info(pid):
    path, doc = pid
    analyzer, backend = make(lambda m, s, n: INFO_OK)
    out = analyzer.read_drawing_info(doc, path)
    msgs = backend.calls[0][0]
    assert backend.calls[0][2] == "DrawingInfoOut" and isinstance(msgs[1].parts()[0], ImagePart)
    items = digest_of(msgs, "Title-block texts:\n")
    # zone markers and the legend cluster are left out; the render covers the title block only
    assert {i["id"] for i in items} == {"t10", "t11", "t12", "t13", "t14", "t15", "t16", "t17", "t18"}
    w, h = png_size(msgs[1].images()[0].data)
    assert w == 1536 and h < 1000
    hints = digest_of(msgs, "Hints:\n")
    assert hints["keyword_system"] == "ballast"
    info = out.info
    assert (info.drawing_no, info.hull_no, info.revision, info.system) == ("1234-BA-001", "S1234", "A", SystemType.BALLAST)
    assert info.ship_type == "174K LNG CARRIER" and info.systems == [SystemType.BALLAST]
    assert info.extra["source"] == "mock" and info.extra["field_sources"]["drawing_no"] == "mock"
    assert info.extra["vector_info"] == {"title": "BALLAST WATER SYSTEM"}
    assert doc.info.drawing_no is None
    # an all-null (schema-valid) answer keeps the deterministic values and falls back to the keyword system
    analyzer, _ = make(lambda m, s, n: {})
    empty = analyzer.read_drawing_info(doc.model_copy(update={"info": DrawingInfo(title="BALLAST WATER SYSTEM")}), path)
    assert empty.info.system == SystemType.BALLAST and empty.info.title == "BALLAST WATER SYSTEM"
    assert empty.info.extra["field_sources"] == {"title": "vector", "system": "keyword"}
    assert empty.info.extra["source"] == "vector"


def test_read_drawing_info_rejects_values_not_in_texts(pid):
    path, doc = pid
    answers = iter([dict(INFO_OK, drawing_no="1234-BA-999", evidence_text_ids=["t99"]), INFO_OK])
    analyzer, backend = make(lambda m, s, n: next(answers))
    out = analyzer.read_drawing_info(doc, path)
    repair = backend.calls[1][0][-1].text()
    assert "t99" in repair and "1234-BA-999" in repair
    assert out.info.drawing_no == "1234-BA-001"


def test_read_drawing_info_llm_error_keeps_keyword_system(pid):
    path, doc = pid
    analyzer, _ = make(lambda m, s, n: "no JSON here", max_attempts=1)
    out = analyzer.read_drawing_info(doc, path)
    assert out.info.system == SystemType.BALLAST and out.info.extra["field_sources"] == {"system": "keyword"}
    assert out.info.extra["source"] == "vector" and out.meta["gemma"]["drawing_info"]["source"] == "vector"
    assert out.info.drawing_no is None and out.meta["llm_errors"][0].startswith("read_drawing_info")


# --------------------------------------------------------------------------- text ownership


TEXTS_OK = {"assignments": [
    {"text_id": "t1", "reason": "valve tag right of the globe valve", "owner_id": "o2", "role": "tag"},
    {"text_id": "t2", "reason": "parallel to p2", "owner_id": "p2", "role": "line_number"},
]}


def test_ambiguous_texts_selection(pid):
    _, doc = pid
    analyzer, _ = make(lambda m, s, n: {})
    found = {t.id: [c["id"] for c in cands] for t, cands in analyzer.ambiguous_texts(doc)}
    assert set(found) == {"t1", "t2"}  # t3 sits alone inside its bubble
    assert set(found["t1"][:2]) == {"o2", "o5"} and found["t2"][0] == "p2"


def test_resolve_texts_applies_assignments(pid):
    _, doc = pid
    analyzer, backend = make(lambda m, s, n: TEXTS_OK)
    out = analyzer.resolve_texts(doc)
    digest = digest_of(backend.calls[0][0], "Texts:\n")
    assert [d["text_id"] for d in digest] == ["t1", "t2"]
    assert digest[0]["current_owner"] == "o5" and {c["id"] for c in digest[0]["candidates"]} >= {"o2", "o5"}
    texts, objs = out.text_by_id(), out.object_by_id()
    pipes = {p.id: p for p in out.pipes}
    assert texts["t1"].owner_id == "o2" and objs["o2"].tag == "BA-V001" and "t1" in objs["o2"].text_ids
    assert texts["t2"].owner_id == "p2"
    assert (pipes["p2"].line_number, pipes["p2"].size, pipes["p2"].medium.value) == ("150A-BA-001", "150A", "ballast_water")
    assert out.meta["gemma"]["texts"]["changed"] == 2
    assert {r["text_id"] for r in out.meta["text_resolution"]} == {"t1", "t2"}
    assert doc.text_by_id()["t1"].owner_id == "o5"


def test_resolve_texts_rejects_owner_outside_candidates(pid):
    _, doc = pid
    bad = {"assignments": [dict(TEXTS_OK["assignments"][0], owner_id="o6"), TEXTS_OK["assignments"][1]]}
    answers = iter([bad, TEXTS_OK])
    analyzer, backend = make(lambda m, s, n: next(answers))
    out = analyzer.resolve_texts(doc)
    assert "t1->o6" in backend.calls[1][0][-1].text()
    assert out.text_by_id()["t1"].owner_id == "o2"


def test_resolve_texts_llm_error(pid):
    _, doc = pid

    def boom(m, s, n):
        raise LLMError("HTTP 503 from server")

    analyzer, _ = make(boom)
    out = analyzer.resolve_texts(doc)
    assert out.texts == doc.texts and out.meta["llm_errors"][0].startswith("resolve_texts[t1..t2]: HTTP 503")


# --------------------------------------------------------------------------- review


def _report(doc):
    return AnalysisReport(
        document=doc, counts={"equipment": 3, "valve": 3, "instrument": 1},
        flow_paths=[FlowPath(source_id="o1", target_id="j1", node_ids=["o1", "o2", "o3", "o4", "j1"],
                             pipe_ids=["p1", "p2", "p3", "p4"], description="tank to overboard")],
        findings=[Finding(id="f1", rule_id="OVB-NRV", verdict=Verdict.PASS, severity=Severity.INFO,
                          title="Overboard non-return valve present", description="o4", object_ids=["o4"],
                          reference="ICLL Reg. 22")],
        summary_en="Deterministic summary.")


REVIEW_OK = {"findings": [
    {"rule_id": "GEMMA-PUMP-ISO", "verdict": "fail", "severity": "warning", "title": "No suction isolation for pump",
     "description": "NO.1 BALLAST PUMP (o3) has no isolation valve on its suction side.", "object_ids": ["o3", "o3", "p2"],
     "reference": None, "confidence": 0.7},
    {"rule_id": "tag missing", "verdict": "insufficient_data", "severity": "info", "title": "Untagged check valve",
     "description": "o4 has no tag.", "object_ids": ["o4"], "reference": None, "confidence": 1.4},
], "summary_en": "Ballast transfer from NO.1 tank via pump o3.", "summary_ko": "1번 탱크에서 펌프(o3)를 통해 평형수를 이송한다."}


@runtime_checkable
class _ReportReviewer(Protocol):
    def review(self, report: AnalysisReport) -> AnalysisReport: ...


def test_review_merges_findings_and_usage(pid):
    _, doc = pid
    doc = doc.model_copy(update={"meta": {"llm_usage": {"backend": "mock", "model": "mock", "calls": 2,
                                                        "prompt_tokens": 100, "completion_tokens": 10}}})
    analyzer, backend = make(lambda m, s, n: REVIEW_OK)
    assert isinstance(analyzer, _ReportReviewer)
    report = _report(doc)
    out = analyzer.review(report)
    msgs = backend.calls[0][0]
    assert "SOLAS II-2/4.2.2.3.4" in msgs[0].text() and "summary_ko" in msgs[0].text()
    digest = digest_of(msgs, "P&ID digest:\n")
    assert digest["rule_checks"][0]["rule_id"] == "OVB-NRV" and digest["flow_paths"][0]["via"] == ["o1", "o2", "o3", "o4"]
    assert {"id": "o3", "to": ["o2", "o4"]} in digest["adjacency"]
    assert any(e.get("name") == "NO.1 BALLAST PUMP" for e in digest["equipment"])
    assert digest["process_pipes_without_line_number"] == 5
    new = [f for f in out.findings if f.source == "mock"]
    assert [f.rule_id for f in new] == ["MOCK-PUMP-ISO", "MOCK-TAG-MISSING"]
    assert new[0].object_ids == ["o3", "p2"] and new[1].confidence == 1.0 and new[0].id == "mock-001"
    assert out.findings[0] == report.findings[0] and len(report.findings) == 1
    assert out.summary_ko.startswith("1번 탱크") and out.summary_en.startswith("Ballast")
    assert out.meta["summary_deterministic"]["en"] == "Deterministic summary."
    assert out.llm.backend == "mock" and out.llm.calls == 3  # 2 refine calls (doc meta) + 1 review call
    assert out.llm.prompt_tokens > 100
    assert analyzer.review(out).llm.calls == 4  # refine usage folded only once


def test_review_repair_and_failure(pid):
    _, doc = pid
    answers = iter([dict(REVIEW_OK, summary_ko="English only"),
                    dict(REVIEW_OK, findings=[dict(REVIEW_OK["findings"][0], object_ids=["o77"])]), REVIEW_OK])
    analyzer, backend = make(lambda m, s, n: next(answers))
    out = analyzer.review(_report(doc))
    assert len(backend.calls) == 3
    assert "Korean" in backend.calls[1][0][-1].text() and "o77" in backend.calls[2][0][-1].text()
    assert len(out.findings) == 3 and len(out.llm.errors) == 2
    analyzer, _ = make(lambda m, s, n: "not json", max_attempts=1)
    report = _report(doc)
    failed = analyzer.review(report)
    assert failed.findings == report.findings and failed.summary_en == "Deterministic summary."
    assert failed.summary_ko == "" and failed.llm.errors[-1].startswith("review:")
    assert failed.meta["llm_errors"][0].startswith("review:")


def test_review_through_openai_compatible_server(pid):
    """Real backend code path against an emulated vLLM server: thinking on, 'gemma' provenance."""
    _, doc = pid
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        content = json.dumps(REVIEW_OK, ensure_ascii=False)
        return httpx.Response(200, json={
            "model": "google/gemma-4-31B-it",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content, "reasoning": "Check pumps first."}}],
            "usage": {"prompt_tokens": 3000, "completion_tokens": 400}})

    backend = OpenAICompatBackend("google/gemma-4-31B-it", "http://localhost:8000/v1",
                                  transport=httpx.MockTransport(handler))
    analyzer = GemmaAnalyzer(LLMClient(backend))
    out = analyzer.review(_report(doc))
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert bodies[0]["response_format"]["json_schema"]["name"] == "ReviewOut"
    assert [f.rule_id for f in out.findings if f.source == "gemma"] == ["GEMMA-PUMP-ISO", "GEMMA-TAG-MISSING"]
    assert out.meta["summary_source"] == "gemma"
    assert (out.llm.backend, out.llm.model, out.llm.calls, out.llm.prompt_tokens, out.llm.completion_tokens) == (
        "openai", "google/gemma-4-31B-it", 1, 3000, 400)
    assert analyzer.usage.calls == 1


# --------------------------------------------------------------------------- whole document


def test_refine_document_tolerates_failing_step(pid):
    path, doc = pid

    def responder(m, s, n):
        if n == "DrawingInfoOut":
            raise LLMError("HTTP 500: CUDA out of memory")
        return {"SymbolBatch": SYMBOLS_OK, "TextAssignments": TEXTS_OK}[n]

    analyzer, backend = make(responder)
    out = analyzer.refine_document(doc, path)
    assert [c[2] for c in backend.calls] == ["DrawingInfoOut", "SymbolBatch", "TextAssignments"]
    assert out.info.system == SystemType.BALLAST and out.info.extra["field_sources"]["system"] == "keyword"
    assert out.object_by_id()["o2"].cls == SymbolClass.GLOBE_VALVE
    assert out.text_by_id()["t1"].owner_id == "o2"
    assert len(out.meta["llm_errors"]) == 1 and "CUDA out of memory" in out.meta["llm_errors"][0]
    assert out.meta["refined_by"] == {"label": "mock", "backend": "mock", "model": "mock"}
    assert out.meta["llm_usage"]["calls"] == 2
    assert doc.meta == {} and doc.object_by_id()["o2"].cls == SymbolClass.GATE_VALVE


def test_adjacency_collapses_junctions(pid):
    _, doc = pid
    doc = doc.model_copy(deep=True)
    doc.pipes.append(PipeRun(id="p9", points=[Point(x=560, y=200), Point(x=600, y=200)],
                             start=Endpoint(junction_id="j1"), end=Endpoint(object_id="o5")))
    adj = _adjacency(doc)
    assert adj["o4"] == ["o3", "o5"] and adj["o1"] == ["o2"]


def test_make_analyzer_from_config():
    assert make_analyzer(LLMConfig(backend="none")) is None
    an = make_analyzer(LLMConfig(backend="vllm", max_image_side=1152, concurrency=3), symbol_batch=12)
    assert an.crop_max_side == 1152 and an.concurrency == 3 and an.symbol_batch == 12 and an.label == "gemma"
    mock = make_analyzer(LLMConfig(backend="mock"), responder=lambda m, s, n: {})
    assert mock.label == "mock"
