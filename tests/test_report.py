"""Tests for shippid.report (HTML report and CSV/JSON export)."""

from __future__ import annotations

import base64
import csv
import re
from html.parser import HTMLParser
from pathlib import Path

import pymupdf
import pytest

from analysis_builder import DocBuilder, pump_station
from shippid.analysis.analyzer import analyze
from shippid.catalog import medium_rgb
from shippid.model import (
    ActuatorType,
    AnalysisReport,
    Deck,
    Finding,
    Layout3D,
    LLMUsage,
    Medium,
    PipeKind,
    Severity,
    SymbolClass,
    SystemType,
    Verdict,
)
from shippid.report.export import export_tables, write_csv
from shippid.report.html import write_html_report

S = SymbolClass
XSS = '<script>alert("x")</script>'


class _Collector(HTMLParser):
    """Collects start tags with attributes and the text of <script> / <td> elements."""

    def __init__(self):
        super().__init__()
        self.tags: list[tuple[str, dict[str, str]]] = []
        self.scripts: list[str] = []
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        self._in_script = tag == "script"

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, data):
        if self._in_script:
            self.scripts.append(data)

    def find(self, tag: str, **attrs: str) -> list[dict[str, str]]:
        out = []
        for t, a in self.tags:
            if t == tag and all(re.search(v, a.get(k, "") or "") for k, v in attrs.items()):
                out.append(a)
        return out


def _parse(path: Path) -> tuple[str, _Collector]:
    src = path.read_text(encoding="utf-8")
    c = _Collector()
    c.feed(src)
    return src, c


def _report(evil: bool = False) -> AnalysisReport:
    b, ids = pump_station(SystemType.BALLAST, source_name="NO.1 W.B.TK (P)", sink_name="NO.2 W.B.TK (S)",
                          nrv=(True, False))
    i = b.obj(S.INSTRUMENT_LOCAL, 520, 230, tag="PI-101")
    b.pipe(i, ids["pumps"][0], kind=PipeKind.INSTRUMENT_TAP)
    doc = b.build()
    for p in doc.pipes[:3]:
        p.medium = Medium.BALLAST_WATER
        p.line_number = "300A-BA-001"
    if evil:
        doc.objects[0].tag = XSS
        doc.objects[0].attributes["name"] = 'Tank "A" & <b>B</b>'
        doc.info.title = "<img src=x onerror=alert(1)>"
    rep = analyze(doc)
    if evil:
        rep.findings.append(Finding(id="g-1", rule_id="GEMMA-X", verdict=Verdict.FAIL, severity=Severity.WARNING,
                                    title=XSS, description="</td><script>bad()</script>", object_ids=[doc.objects[0].id],
                                    reference="<ref>", source="gemma"))
        rep.summary_en = XSS
    return rep


def test_html_overlay_without_pdf(tmp_path):
    rep = _report()
    out = write_html_report(rep, tmp_path / "r.html")
    src, c = _parse(out)
    doc = rep.document
    assert src.startswith("<!DOCTYPE html>")
    assert len(c.find("svg", **{"class": "drawing"})) == 1
    rects = c.find("rect", **{"class": r"\bobj\b"})
    assert sorted(a["data-id"] for a in rects) == sorted(o.id for o in doc.objects)
    assert {a["class"].split()[1] for a in rects} >= {"cat-equipment", "cat-valve", "cat-instrument"}
    polylines = c.find("polyline")
    assert sorted(a["data-id"] for a in polylines) == sorted(p.id for p in doc.pipes)
    ballast = [a for a in polylines if a["data-id"] in {p.id for p in doc.pipes[:3]}]
    expected = "#" + "".join(f"{round(v * 255):02x}" for v in medium_rgb(Medium.BALLAST_WATER))
    assert ballast and all(a["stroke"] == expected for a in ballast)
    assert any("kind-instrument_tap" in a["class"] for a in polylines)
    assert len(c.find("circle", **{"class": "jn"})) == len(doc.junctions)
    assert not c.find("image")  # no PDF: overlay only
    assert "overlay only" in src
    # every overlay element carries a tooltip
    assert all(a.get("data-tip") for a in rects + polylines)
    # tables: one clickable row per equipment / valve / instrument / line
    rows = c.find("tr", **{"data-ids": r"."})
    ids = {a["data-ids"] for a in rows}
    assert {r["id"] for r in rep.equipment_list + rep.valve_list + rep.instrument_list} <= ids
    assert {r["pipe_ids"] for r in rep.line_list} <= ids
    assert "설비 목록 (Equipment list)" in src and "밸브 목록 (Valve list)" in src and "계기 목록" in src
    assert "결정론적 템플릿" in src and "LLM 미사용" in src


def test_html_findings_sorted_with_badges(tmp_path):
    rep = _report()
    rep.findings.append(Finding(id="g-1", rule_id="GEMMA-NOTE", verdict=Verdict.INSUFFICIENT_DATA,
                                severity=Severity.INFO, title="t", description="d", source="gemma"))
    out = write_html_report(rep, tmp_path / "r.html")
    src, c = _parse(out)
    rows = c.find("tr", **{"data-verdict": r"."})
    seq = [a["data-verdict"] for a in rows]
    rank = {"fail": 0, "insufficient_data": 1, "pass": 2, "not_applicable": 3}
    assert seq == sorted(seq, key=rank.get) and seq[0] == "fail" and "insufficient_data" in seq
    assert len(rows) == len(rep.findings)
    assert c.find("span", **{"class": "badge v-fail"}) and c.find("span", **{"class": "src-gemma"})
    assert c.find("span", **{"class": "src-rule"})
    assert "부적합 FAIL" in src and "R_PUMP_DISCHARGE_NRV" in src


def test_html_escapes_untrusted_text(tmp_path):
    rep = _report(evil=True)
    out = write_html_report(rep, tmp_path / "r.html", viewer_href="javascript:alert(1)")
    src, c = _parse(out)
    assert XSS not in src and "<img src=x" not in src and "</td><script>bad()" not in src
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in src
    assert "Tank &quot;A&quot; &amp; &lt;b&gt;B&lt;/b&gt;" in src
    # the only <script> element is the report's own inline script, and no handler attributes exist
    assert len(c.find("script")) == 1 and "alert" not in "".join(c.scripts)
    assert not any(k.startswith("on") for _, a in c.tags for k in a)
    assert "javascript:" not in src and "unsafe viewer link omitted" in src


def test_html_has_no_external_resources(tmp_path):
    rep = _report()
    out = write_html_report(rep, tmp_path / "r.html", viewer_href="model3d.html")
    src, c = _parse(out)
    assert not c.find("link") and not c.find("script", src=".")
    for tag, a in c.tags:
        for k in ("src", "href"):
            v = a.get(k)
            if v and tag != "svg":
                assert not re.match(r"https?:", v), (tag, v)
    assert not re.search(r"url\(\s*['\"]?https?:", src)
    assert c.find("iframe", src="^model3d.html$") and c.find("a", href="^model3d.html$")


def _pdf(tmp_path: Path, w: float = 600, h: float = 400) -> Path:
    pdf = pymupdf.open()
    page = pdf.new_page(width=w, height=h)
    page.draw_rect(pymupdf.Rect(40, 40, 120, 90), color=(0, 0, 0))
    page.insert_text((50, 70), "NO.1 TK", fontsize=8)
    path = tmp_path / "d.pdf"
    pdf.save(path)
    return path


def test_html_embeds_page_render(tmp_path):
    rep = _report()
    rep.document.pages[0].width, rep.document.pages[0].height = 600, 400
    out = write_html_report(rep, tmp_path / "r.html", pdf_path=_pdf(tmp_path))
    src, c = _parse(out)
    imgs = c.find("image", **{"class": "page-image"})
    assert len(imgs) == 1 and imgs[0]["width"] == "600.0" and imgs[0]["height"] == "400.0"
    data = imgs[0]["href"]
    assert data.startswith("data:image/png;base64,")
    png = base64.b64decode(data.split(",", 1)[1])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pix = pymupdf.Pixmap(png)
    assert pix.width >= 600 and abs(pix.width / pix.height - 1.5) < 0.01  # >= 72 dpi, aspect preserved
    assert c.find("svg", viewbox="^0 0 600.0 400.0$")  # HTMLParser lower-cases attribute names


def test_html_missing_pdf_falls_back_to_overlay(tmp_path):
    out = write_html_report(_report(), tmp_path / "r.html", pdf_path=tmp_path / "missing.pdf")
    src, c = _parse(out)
    assert not c.find("image") and "overlay only" in src


def test_html_layout_viewer_and_llm_sections(tmp_path):
    rep = _report()
    rep.llm = LLMUsage(backend="mock", model="mock", calls=3, prompt_tokens=100, completion_tokens=20,
                       errors=["review: <boom>"])
    layout = Layout3D(planner="gemma", decks=[Deck(name="floor", z=4.5)], notes=["placed <3> objects"])
    out = write_html_report(rep, tmp_path / "r.html", viewer_href="scene.html", layout=layout.model_dump())
    src, c = _parse(out)
    assert "Gemma 4 31B 배치 계획" in src and "floor 4.5 m" in src and "placed &lt;3&gt; objects" in src
    assert "Gemma 출력이 아닙니다" in src  # mock output is flagged as such
    assert "review: &lt;boom&gt;" in src
    assert c.find("iframe", src="^scene.html$")


def test_html_multi_page_and_no_page_info(tmp_path):
    b = DocBuilder()
    v1 = b.obj(S.GATE_VALVE, 100, 100)
    v2 = b.obj(S.GATE_VALVE, 200, 100)
    b.pipe(v1, v2)
    b.obj(S.GATE_VALVE, 150, 150, page=1)
    doc = b.build()
    doc.pages = []
    out = write_html_report(analyze(doc), tmp_path / "r.html")
    src, c = _parse(out)
    svgs = c.find("svg", **{"class": "drawing"})
    assert [a["data-page"] for a in svgs] == ["0", "1"]
    assert "페이지 Page 2" in src


def test_export_tables_csv_and_json(tmp_path):
    rep = _report()
    rep.valve_list[0]["tag"] = "=HYPERLINK(\"http://x\")"
    rep.valve_list[0]["service"] = "평형수 펌프 토출"
    paths = export_tables(rep, tmp_path / "out")
    assert set(paths) == {"equipment", "valves", "instruments", "lines", "findings", "flow_paths", "report_json"}
    raw = paths["valves"].read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM for Excel
    with paths["valves"].open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(rep.valve_list)
    assert rows[0]["tag"].startswith("'=")  # formula injection neutralised
    assert rows[0]["service"] == "평형수 펌프 토출"
    with paths["findings"].open(encoding="utf-8-sig", newline="") as fh:
        frows = list(csv.DictReader(fh))
    assert len(frows) == len(rep.findings)
    assert {"rule_id", "verdict", "severity", "reference", "source"} <= set(frows[0])
    with paths["lines"].open(encoding="utf-8-sig", newline="") as fh:
        lrows = list(csv.DictReader(fh))
    assert lrows[0]["line_number"] == "300A-BA-001"
    loaded = AnalysisReport.load(paths["report_json"])
    assert loaded.model_dump() == rep.model_dump()


def test_write_csv_values(tmp_path):
    p = write_csv([{"a": 1, "b": [1, 2], "c": None, "d": True, "e": "-x", "f": "-1.5"}, {"g": {"k": "v"}}],
                  tmp_path / "x.csv")
    with p.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["a", "b", "c", "d", "e", "f", "g"]
    assert rows[1] == ["1", "1; 2", "", "true", "'-x", "-1.5", ""]
    assert rows[2][-1] == '{"k": "v"}'


def test_export_handles_empty_report(tmp_path):
    from shippid.model import PIDDocument

    paths = export_tables(analyze(PIDDocument()), tmp_path)
    assert paths["equipment"].read_bytes() == b"\xef\xbb\xbf\r\n"
    assert AnalysisReport.load(paths["report_json"]).counts["objects"] == 0


@pytest.mark.parametrize("actuator", [ActuatorType.QUICK_CLOSING, ActuatorType.HYDRAULIC])
def test_valve_tooltip_mentions_actuator(tmp_path, actuator):
    b = DocBuilder()
    b.obj(S.GLOBE_VALVE, 100, 100, actuator=actuator, tag="FO-V001")
    out = write_html_report(analyze(b.build()), tmp_path / "r.html")
    _, c = _parse(out)
    tip = c.find("rect", **{"data-id": "^o1$"})[0]["data-tip"]
    assert f"actuator: {actuator.value}" in tip and "FO-V001" in tip


# --------------------------------------------------------------------------- pages not analysed / cross-sheet paths


def test_report_warns_about_pages_not_analysed(tmp_path):
    from shippid.model import PageInfo, PIDDocument

    scanned = PIDDocument(source="scanned.pdf", pages=[PageInfo(index=0, width=600, height=400)],
                          meta={"unreadable_pages": [0]})
    src, c = _parse(write_html_report(analyze(scanned), tmp_path / "scan.html"))
    assert len(c.find("div", id="pages-not-analysed")) == 2  # summary and drawing sections
    assert "scanned / raster-only (unreadable): 페이지 page 1" in src
    assert "scanned / raster-only page, not analysed" in src  # page heading badge
    assert "No objects were extracted" in src and "적합 PASS" in src
    # a partly readable set lists both kinds; the analysed page keeps its overlay
    b, _ = pump_station(SystemType.BALLAST)
    doc = b.build()
    doc.pages += [PageInfo(index=1, width=600, height=400), PageInfo(index=2, width=600, height=400)]
    doc.meta.update({"unreadable_pages": [1], "empty_pages": [2]})
    src, c = _parse(write_html_report(analyze(doc), tmp_path / "part.html"))
    assert "page 2</li>" in src and "empty: 페이지 page 3" in src and "empty page, not analysed" in src
    assert c.find("rect", **{"class": r"\bobj\b"})
    # nothing skipped: no warning
    src, c = _parse(write_html_report(_report(), tmp_path / "ok.html"))
    assert not c.find("div", id="pages-not-analysed")


def _two_sheet_report() -> tuple[AnalysisReport, str, str]:
    from shippid.model import PageInfo

    b = DocBuilder(SystemType.BALLAST)
    t1 = b.obj(S.TANK, 60, 300, name="NO.1 W.B.TK (P)")
    p = b.obj(S.PUMP_CENTRIFUGAL, 300, 300, name="NO.1 BALLAST PUMP")
    c1 = b.obj(S.OFF_PAGE_CONNECTOR, 560, 300)
    b.chain(t1, p, c1, flow=1)
    c2 = b.obj(S.OFF_PAGE_CONNECTOR, 60, 300, rotation=180, page=1)
    t2 = b.obj(S.TANK, 400, 300, name="NO.2 W.B.TK (P)", page=1)
    b.chain(c2, t2, flow=1, page=1)
    doc = b.build()
    doc.pages.append(PageInfo(index=1, width=1200, height=800))
    doc.meta["sheet_links"] = [{"a": c1, "b": c2, "directed": True, "source": "manual"}]
    return analyze(doc), c1, c2


def test_flow_path_rows_reference_drawn_elements_and_export_sheet_links(tmp_path):
    rep, c1, c2 = _two_sheet_report()
    assert len(rep.flow_paths) == 1 and rep.meta["flow_path_sheet_links"] == [{"path": 0, "links": [[c1, c2]]}]
    fp = rep.flow_paths[0]
    assert set(fp.pipe_ids) <= {p.id for p in rep.document.pipes}
    src, c = _parse(write_html_report(rep, tmp_path / "r.html"))
    drawn = {a["data-id"] for _, a in c.tags if "data-id" in a}
    flows = src.split('id="flows"')[1].split("</section>")[0]
    row_ids = [i for ids in re.findall(r'<tr data-ids="([^"]*)"', flows) for i in ids.split()]
    assert row_ids and set(row_ids) <= drawn  # every id highlights something on the drawing
    assert "도면 간 연결 Sheet links" in flows and "(p.1) -&gt; " in flows and "(p.2)" in flows
    paths = export_tables(rep, tmp_path / "t")
    with paths["flow_paths"].open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["sheet_links"] == f"{c1}>{c2}" and "sheet_link:" not in rows[0]["pipe_ids"]
    assert rows[0]["pipe_ids"] == "; ".join(fp.pipe_ids)
    # a report written before the fix (pseudo pipe id) is rendered / exported without it
    old = rep.model_copy(deep=True)
    old.flow_paths[0].pipe_ids.insert(2, f"sheet_link:{c1}:{c2}")
    src, _ = _parse(write_html_report(old, tmp_path / "old.html"))
    assert f"sheet_link:{c1}:{c2}" not in src.split('id="flows"')[1].split("</section>")[0]
    paths = export_tables(old, tmp_path / "old")
    with paths["flow_paths"].open(encoding="utf-8-sig", newline="") as fh:
        assert "sheet_link:" not in list(csv.DictReader(fh))[0]["pipe_ids"]
