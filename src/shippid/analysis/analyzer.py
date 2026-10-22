"""Deterministic analysis of a P&ID document -> :class:`~shippid.model.AnalysisReport`.

:func:`analyze` builds the pipe graph, flow paths, lists and rule findings and writes
deterministic template summaries (English / Korean). An optional reviewer (e.g. the Gemma 4
31B ``GemmaAnalyzer``) then gets a *copy* of the report and may add findings / rewrite the
summaries; if it raises, the deterministic report is kept and the error is recorded in
``report.llm.errors`` and ``report.meta["review_status"]``. A reviewer that handles its own
failure and returns the report unchanged is reported with its own status (see
:func:`review_status`), never as ``"ok"``.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional, Protocol, runtime_checkable

from ..catalog import SYSTEMS
from ..model import AnalysisReport, Category, Finding, FlowPath, PIDDocument, PipeKind, SymbolClass, Verdict
from .graph import FlowNet, build_graph, components, flow_paths, node_name, sources_and_sinks
from .lists import equipment_list, instrument_list, line_list, valve_list
from .rules import RULES, RuleContext, run_rules
from .sheets import link_sheets

log = logging.getLogger(__name__)

ANALYZER_VERSION = "1.0"
_VERDICT_ORDER = {Verdict.FAIL: 0, Verdict.INSUFFICIENT_DATA: 1, Verdict.PASS: 2, Verdict.NOT_APPLICABLE: 3}


@runtime_checkable
class ReportReviewer(Protocol):
    """Anything that can review a report (Gemma engineering review, human-in-the-loop, ...)."""

    def review(self, report: AnalysisReport) -> AnalysisReport: ...


def analyze(doc: PIDDocument, reviewer: Optional[ReportReviewer] = None, *, max_paths: int = 200) -> AnalysisReport:
    """Graph + flow paths + lists + rule findings + template summaries, then optional review.

    A document with several pages and no ``meta["sheet_links"]`` yet gets its off-page connectors
    paired first (:func:`shippid.analysis.sheets.link_sheets`, stored in ``doc.meta``), so the
    graph, flow paths and rules see one system across the sheets."""
    if _pages(doc) > 1 and "sheet_links" not in doc.meta:
        link_sheets(doc)
    G = build_graph(doc)
    net = FlowNet(doc, G)
    ctx = RuleContext(doc, G, net)
    all_paths = flow_paths(doc, G, max_paths=10 ** 6, net=net)
    paths = all_paths[: max(0, max_paths)]
    findings = run_rules(doc, G, context=ctx)
    sources, sinks = sources_and_sinks(doc, G)
    comps = components(doc, G)

    report = AnalysisReport(
        document=doc,
        counts=_counts(doc, paths, findings, len(sources), len(sinks), len(comps)),
        equipment_list=equipment_list(doc, G),
        valve_list=valve_list(doc, G),
        instrument_list=instrument_list(doc, G),
        line_list=line_list(doc, G, paths),
        flow_paths=paths,
        findings=findings,
    )
    report.meta.update({
        "analysis": "deterministic",
        "analyzer_version": ANALYZER_VERSION,
        "systems": [s.value for s in ctx.systems],
        "system_basis": ctx.system_basis,
        "summary_source": "template",
        "flow_paths_total": len(all_paths),
        "flow_paths_truncated": len(all_paths) > len(paths),
        "rules": list(RULES),
        "components": len(comps),
    })
    if _pages(doc) > 1 or doc.meta.get("sheet_links"):
        report.meta["sheet_links"] = sum(1 for *_, linked in G.edges(data="sheet_link") if linked)
        report.meta["sheet_links_ambiguous"] = len(doc.meta.get("sheet_links_ambiguous") or [])
    report.summary_en = summary_en(report, ctx)
    report.summary_ko = summary_ko(report, ctx)

    if reviewer is None:
        report.meta["review_status"] = "none"
        return report
    name = type(reviewer).__name__
    n_errors = len(report.llm.errors)
    try:
        reviewed = reviewer.review(report.model_copy(deep=True))
        if not isinstance(reviewed, AnalysisReport):
            raise TypeError(f"reviewer returned {type(reviewed).__name__}, expected AnalysisReport")
    except Exception as exc:  # keep the deterministic report on any reviewer failure
        log.warning("report review by %s failed: %s", name, exc)
        report.llm.errors.append(f"review ({name}): {type(exc).__name__}: {exc}"[:1000])
        report.meta["review_status"] = "error"
        report.meta["reviewer"] = name
        return report
    reviewed.meta.setdefault("analysis", "deterministic")
    reviewed.meta["review_status"] = review_status(reviewed, n_errors)
    reviewed.meta["reviewer"] = name
    return reviewed


def _pages(doc: PIDDocument) -> int:
    return len({p.index for p in doc.pages} | {o.page for o in doc.objects})


def review_status(reviewed: AnalysisReport, n_errors_before: int = 0) -> str:
    """Outcome of a review that returned normally: the reviewer's own ``meta["review"]["status"]``
    when it records one (a reviewer such as ``GemmaAnalyzer`` returns the unchanged copy with
    status ``"error"`` when its LLM call fails), ``"error"`` when it reported nothing but added
    errors to ``llm.errors``, else ``"ok"``."""
    own = reviewed.meta.get("review")
    status = own.get("status") if isinstance(own, dict) else None
    if isinstance(status, str) and status.strip():
        return status.strip().lower()
    return "error" if len(reviewed.llm.errors) > n_errors_before else "ok"


def _counts(doc: PIDDocument, paths: list[FlowPath], findings: list[Finding], n_src: int, n_snk: int,
            n_comp: int) -> dict[str, int]:
    cats = Counter(o.category for o in doc.objects)
    kinds = Counter(p.kind for p in doc.pipes)
    verdicts = Counter(f.verdict for f in findings)
    counts = {
        "objects": len(doc.objects),
        "equipment": cats[Category.EQUIPMENT],
        "valves": cats[Category.VALVE],
        "instruments": cats[Category.INSTRUMENT],
        "fittings": cats[Category.FITTING],
        "connectors": cats[Category.CONNECTOR],
        "unknown": sum(1 for o in doc.objects if o.cls == SymbolClass.UNKNOWN),
        "junctions": len(doc.junctions),
        "pipes": len(doc.pipes),
        "process_pipes": kinds[PipeKind.PROCESS],
        "instrument_taps": kinds[PipeKind.INSTRUMENT_TAP],
        "signal_lines": kinds[PipeKind.SIGNAL],
        "line_numbers": len({p.line_number.strip().upper() for p in doc.pipes if p.line_number}),
        "texts": len(doc.texts),
        "flow_paths": len(paths),
        "sources": n_src,
        "sinks": n_snk,
        "components": n_comp,
        "findings": len(findings),
    }
    for v in Verdict:
        counts[f"findings_{v.value}"] = verdicts[v]
    for cls, n in sorted(Counter(o.cls.value for o in doc.objects).items()):
        counts[f"class:{cls}"] = n
    return counts


def sorted_findings(findings: list[Finding]) -> list[Finding]:
    """FAIL -> INSUFFICIENT_DATA -> PASS -> NOT_APPLICABLE, then severity, then rule id."""
    sev = {"error": 0, "warning": 1, "info": 2}
    return sorted(findings, key=lambda f: (_VERDICT_ORDER.get(f.verdict, 9), sev.get(f.severity.value, 9),
                                           f.rule_id, f.id))


def _system_names(ctx: RuleContext, ko: bool) -> str:
    if not ctx.systems:
        return "계통 미상" if ko else "system unknown"
    return ", ".join(f"{SYSTEMS[s].name_ko if ko else SYSTEMS[s].name_en} ({SYSTEMS[s].code})"
                     for s in ctx.systems if s in SYSTEMS)


_BASIS_KO = {"drawing_info": "도면 정보", "title": "도면 제목", "line_numbers": "라인 번호", "title_block": "표제란",
             "unknown": "판단 불가"}


def _fail_lines(report: AnalysisReport, ctx: RuleContext, ko: bool, limit: int = 8) -> list[str]:
    fails = [f for f in sorted_findings(report.findings) if f.verdict == Verdict.FAIL]
    out = []
    for f in fails[:limit]:
        title = RULES[f.rule_id].title_ko if (ko and f.rule_id in RULES) else f.title
        objs = ", ".join(node_name(ctx.G, i, with_tag=False) for i in f.object_ids[:3] if i in ctx.G)
        out.append(f"- [{f.rule_id}] {title}" + (f" - {objs}" if objs else ""))
    if len(fails) > limit:
        out.append(f"- ... 외 {len(fails) - limit}건" if ko else f"- ... and {len(fails) - limit} more")
    return out


def summary_en(report: AnalysisReport, ctx: RuleContext) -> str:
    """Deterministic English summary (template text, no LLM)."""
    c, info = report.counts, report.document.info
    lines = [
        "[Deterministic summary: template text generated by the shippid rule engine; no language model was used.]",
        f"Drawing {info.drawing_no or '-'} \"{info.title or '-'}\" - {_system_names(ctx, False)} "
        f"(basis: {ctx.system_basis.replace('_', ' ')}).",
        f"Objects: {c['objects']} ({c['equipment']} equipment, {c['valves']} valves, {c['instruments']} instruments, "
        f"{c['fittings']} fittings, {c['connectors']} off-page connectors; {c['unknown']} unclassified). "
        f"Pipes: {c['pipes']} runs ({c['process_pipes']} process, {c['instrument_taps']} instrument taps, "
        f"{c['signal_lines']} signal lines), {c['line_numbers']} line numbers, {c['junctions']} junctions, "
        f"{c['components']} connected groups.",
    ]
    if "sheet_links" in report.meta:
        lines.append(f"Sheets: {_pages(report.document)} pages joined by {report.meta['sheet_links']} cross-sheet "
                     f"connector links ({report.meta.get('sheet_links_ambiguous', 0)} connectors left ambiguous).")
    total = report.meta.get("flow_paths_total", c["flow_paths"])
    fp = f"Flow paths: {total} driven paths from {c['sources']} sources to {c['sinks']} sinks"
    if report.flow_paths:
        fp += f", e.g. {report.flow_paths[0].description}"
    if report.meta.get("flow_paths_truncated"):
        fp += f" (first {c['flow_paths']} kept)"
    lines.append(fp + ".")
    lines.append(f"Rule checks ({len(RULES)} rules, {c['findings']} findings): {c['findings_fail']} fail, "
                 f"{c['findings_insufficient_data']} insufficient data, {c['findings_pass']} pass, "
                 f"{c['findings_not_applicable']} not applicable.")
    fails = _fail_lines(report, ctx, False)
    if fails:
        lines.append("Failed checks:")
        lines += fails
    else:
        lines.append("No rule check failed.")
    return "\n".join(lines)


def summary_ko(report: AnalysisReport, ctx: RuleContext) -> str:
    """Deterministic Korean summary (template text, no LLM)."""
    c, info = report.counts, report.document.info
    lines = [
        "[결정론적 요약: shippid 규칙 엔진이 생성한 템플릿 문장이며 언어 모델을 사용하지 않았습니다.]",
        f"도면 {info.drawing_no or '-'} \"{info.title or '-'}\" - {_system_names(ctx, True)} "
        f"(판단 근거: {_BASIS_KO.get(ctx.system_basis, ctx.system_basis)}).",
        f"객체 {c['objects']}개 (설비 {c['equipment']}, 밸브 {c['valves']}, 계기 {c['instruments']}, "
        f"배관 부속 {c['fittings']}, 도면 연결 표시 {c['connectors']}; 미분류 {c['unknown']}). "
        f"배관 {c['pipes']}개 (공정 배관 {c['process_pipes']}, 계기 연결관 {c['instrument_taps']}, "
        f"신호선 {c['signal_lines']}), 라인 번호 {c['line_numbers']}개, 분기점 {c['junctions']}개, "
        f"연결 그룹 {c['components']}개.",
    ]
    if "sheet_links" in report.meta:
        lines.append(f"도면 {_pages(report.document)}매, 도면 간 연결 표시 쌍 {report.meta['sheet_links']}개 "
                     f"(미결 연결 표시 {report.meta.get('sheet_links_ambiguous', 0)}개).")
    total = report.meta.get("flow_paths_total", c["flow_paths"])
    fp = f"유동 경로: 출발점 {c['sources']}곳에서 도착점 {c['sinks']}곳까지 펌프 구동 경로 {total}개"
    if report.flow_paths:
        fp += f" (예: {report.flow_paths[0].description})"
    if report.meta.get("flow_paths_truncated"):
        fp += f", 상위 {c['flow_paths']}개만 수록"
    lines.append(fp + ".")
    lines.append(f"규칙 검토 ({len(RULES)}개 규칙, {c['findings']}건): 부적합 {c['findings_fail']}건, "
                 f"자료 부족 {c['findings_insufficient_data']}건, 적합 {c['findings_pass']}건, "
                 f"해당 없음 {c['findings_not_applicable']}건.")
    fails = _fail_lines(report, ctx, True)
    if fails:
        lines.append("부적합 항목:")
        lines += fails
    else:
        lines.append("부적합 항목이 없습니다.")
    return "\n".join(lines)
