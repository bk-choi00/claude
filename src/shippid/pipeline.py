"""End-to-end pipeline: readable PDF -> objects -> Gemma 4 31B analysis -> 3D model -> reports.

Every stage writes its artefact into the output directory so a run can be
inspected (and resumed) stage by stage::

    out/
      extracted.json    deterministic vector extraction (PIDDocument, source="vector")
      document.json     after Gemma refinement (symbols, drawing info, text ownership)
      analysis.json     AnalysisReport (lists, flow paths, rule verdicts, Gemma review)
      tables/*.csv      equipment / valve / instrument / line lists, findings
      layout3d.json     Layout3D (Gemma-planned or heuristic arrangement + routed pipes)
      model.glb         3D model (glTF binary)
      viewer.html       interactive 3D viewer
      report.html       analysis report with the drawing overlay

A multi-page document gets its off-page connectors paired across the sheets
(:func:`shippid.analysis.sheets.link_sheets`, ``meta["sheet_links"]``) before it is saved, so
``extracted.json``, ``document.json`` and the document inside ``analysis.json`` carry the same
cross-sheet links (with Gemma, ``document.json`` also holds its decisions on ambiguous pairs).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

from .config import Config, load_config
from .model import AnalysisReport, Layout3D, PIDDocument

if TYPE_CHECKING:  # pragma: no cover
    from .llm.base import LLMClient

log = logging.getLogger(__name__)

PLANNERS = ("auto", "gemma", "heuristic")
VIEWERS = ("threejs", "plotly", "both")


@dataclass
class RunResult:
    document: PIDDocument
    report: Optional[AnalysisReport] = None
    layout: Optional[Layout3D] = None
    paths: dict[str, Path] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    llm_backend: str = "none"


def _client(cfg: Config):
    if not cfg.llm.enabled:
        return None
    from .llm.factory import make_client

    return make_client(cfg.llm)


def llm_label(client: Optional["LLMClient"]) -> str:
    """``backend:model`` of the client that produced an output (``"none"`` without one): a mock
    backend is recorded as ``mock:...``, never as Gemma."""
    return f"{client.backend.name}:{client.model}" if client is not None else "none"


def check_planner(planner: str, client: Optional["LLMClient"]) -> None:
    """Fail early (before any stage runs) when the 3D planner cannot be honoured."""
    if planner not in PLANNERS:
        raise ValueError(f"planner must be one of {PLANNERS}, not {planner!r}")
    if planner == "gemma" and client is None:
        raise ValueError("planner='gemma' needs an LLM backend (set --llm / --base-url or SHIPPID_LLM_*, "
                         "see `shippid llm-check`), or use --planner auto|heuristic")


def make_gemma_analyzer(client: "LLMClient", cfg: Config):
    """GemmaAnalyzer for an existing client, wired like :func:`shippid.analysis.gemma.make_analyzer`
    (``max_image_side`` -> ``crop_max_side``, ``concurrency``)."""
    from .analysis.gemma import GemmaAnalyzer

    return GemmaAnalyzer(client, crop_max_side=cfg.llm.max_image_side, concurrency=max(1, int(cfg.llm.concurrency)))


def page_count(doc: PIDDocument) -> int:
    """Pages of a document, counted as :func:`shippid.analysis.analyzer.analyze` counts them."""
    return len({p.index for p in doc.pages} | {o.page for o in doc.objects})


def link_multisheet(doc: PIDDocument) -> PIDDocument:
    """Pair the off-page connectors of a multi-page document in place, as ``analyze()`` does
    (:func:`shippid.analysis.sheets.link_sheets`). No-op for a single page or when
    ``doc.meta["sheet_links"]`` already exists (e.g. written by the Gemma refinement)."""
    if page_count(doc) > 1 and "sheet_links" not in doc.meta:
        from .analysis.sheets import link_sheets

        link_sheets(doc)
    return doc


def extract(pdf_path: str | Path, cfg: Optional[Config] = None, pages: Optional[Iterable[int]] = None) -> PIDDocument:
    from .extract.pipeline import extract_pdf

    return extract_pdf(pdf_path, cfg or load_config(), pages=pages)


def run_pipeline(
    pdf_path: str | Path,
    out_dir: str | Path,
    cfg: Optional[Config] = None,
    *,
    pages: Optional[Iterable[int]] = None,
    planner: str = "auto",  # auto | gemma | heuristic
    viewer: str = "threejs",  # threejs | plotly | both
    stages: Iterable[str] = ("extract", "refine", "analyze", "model3d", "report"),
    client: Optional["LLMClient"] = None,
) -> RunResult:
    """Run the selected stages on one PDF. With ``cfg.llm.backend == 'none'`` every
    Gemma step is skipped and the deterministic path is used (recorded in the outputs).
    ``client`` overrides the client built from ``cfg`` (tests, custom backends).

    Raises ``ValueError`` before any stage runs when ``planner='gemma'`` is asked for the 3D
    stage without an LLM backend, or ``planner`` / ``viewer`` is unknown."""
    cfg = cfg or load_config()
    stages = set(stages)
    if viewer not in VIEWERS:
        raise ValueError(f"viewer must be one of {VIEWERS}, not {viewer!r}")
    if planner not in PLANNERS:
        raise ValueError(f"planner must be one of {PLANNERS}, not {planner!r}")
    client = client if client is not None else _client(cfg)
    if "model3d" in stages:
        check_planner(planner, client)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    paths: dict[str, Path] = {}

    analyzer = make_gemma_analyzer(client, cfg) if client is not None else None
    refine = analyzer is not None and "refine" in stages

    # 1. deterministic extraction (+ cross-sheet connector pairing for a multi-page set)
    t0 = time.perf_counter()
    doc = extract(pdf_path, cfg, pages)
    timings["extract"] = time.perf_counter() - t0
    extracted = doc
    if page_count(doc) > 1:
        # with refinement the links are computed again on the refined document (Gemma may
        # reclassify connectors); extracted.json gets the deterministic links of the vector result
        t0 = time.perf_counter()
        extracted = link_multisheet(doc.model_copy(deep=True) if refine else doc)
        timings["link_sheets"] = time.perf_counter() - t0
    paths["extracted"] = extracted.save(out / "extracted.json")

    # 2. Gemma refinement of symbols / drawing info / text ownership (/ ambiguous sheet links)
    if refine:
        t0 = time.perf_counter()
        doc = analyzer.refine_document(doc, pdf_path)
        link_multisheet(doc)  # the Gemma sheet-link step failed or did not run
        timings["refine"] = time.perf_counter() - t0
    paths["document"] = doc.save(out / "document.json")
    result = RunResult(document=doc, paths=paths, timings=timings, llm_backend=llm_label(client))

    # 3. analysis (deterministic rules + Gemma engineering review)
    if "analyze" in stages or "report" in stages:
        from .analysis.analyzer import analyze
        from .report.export import export_tables

        t0 = time.perf_counter()
        report = analyze(doc, reviewer=analyzer)
        timings["analyze"] = time.perf_counter() - t0
        paths["analysis"] = report.save(out / "analysis.json")
        for name, p in export_tables(report, out / "tables").items():
            paths[f"table:{name}"] = Path(p)
        result.report = report

    # 4. 3D arrangement + model + viewer
    if "model3d" in stages:
        from .model3d.builder import build_layout
        from .model3d.scene import export_glb, layout_to_scene
        from .model3d.viewer import write_viewer_html

        t0 = time.perf_counter()
        plan_client = None if planner == "heuristic" else client
        layout = build_layout(doc, client=plan_client)
        paths["layout"] = layout.save(out / "layout3d.json")
        scene = layout_to_scene(layout, doc)
        paths["glb"] = Path(export_glb(scene, out / "model.glb"))
        modes = ["threejs", "plotly"] if viewer == "both" else [viewer]
        for mode in modes:
            name = "viewer.html" if mode == modes[0] else f"viewer_{mode}.html"
            paths[f"viewer:{mode}"] = Path(write_viewer_html(paths["glb"], layout, doc, out / name, mode=mode))
        timings["model3d"] = time.perf_counter() - t0
        result.layout = layout

    # 5. HTML report
    if "report" in stages and result.report is not None:
        from .report.html import write_html_report

        t0 = time.perf_counter()
        viewer_href = "viewer.html" if "viewer:" + (viewer if viewer != "both" else "threejs") in paths else None
        paths["report"] = Path(write_html_report(result.report, out / "report.html", pdf_path=pdf_path,
                                                 viewer_href=viewer_href, layout=result.layout))
        timings["report"] = time.perf_counter() - t0

    log.info("pipeline finished in %.1fs: %s", sum(timings.values()),
             ", ".join(f"{k}={v:.1f}s" for k, v in timings.items()))
    return result


def summarize(result: RunResult) -> dict[str, Any]:
    """Short JSON-able summary of a run (printed by the CLI)."""
    doc = result.document
    out: dict[str, Any] = {
        "llm": result.llm_backend,
        "objects": len(doc.objects),
        "pipes": len(doc.pipes),
        "texts": len(doc.texts),
        "system": doc.info.system.value,
        "drawing_no": doc.info.drawing_no,
        "timings_s": {k: round(v, 2) for k, v in result.timings.items()},
        "outputs": {k: str(v) for k, v in result.paths.items() if not k.startswith("table:")},
    }
    if result.report is not None:
        verdicts: dict[str, int] = {}
        for f in result.report.findings:
            verdicts[f.verdict.value] = verdicts.get(f.verdict.value, 0) + 1
        out["findings"] = verdicts
        out["llm_usage"] = result.report.llm.model_dump()
    if result.layout is not None:
        out["layout"] = {"planner": result.layout.planner, "placements": len(result.layout.placements),
                         "routes": len(result.layout.routes)}
    meta = result.report.meta if result.report is not None else doc.meta
    skipped = {k: list(meta.get(k) or doc.meta.get(k) or []) for k in ("unreadable_pages", "empty_pages")}
    if any(skipped.values()):
        out["pages_not_analysed"] = skipped
    if doc.meta.get("damaged"):
        out["damaged_pdf"] = doc.meta["damaged"]
    return out


def nothing_extracted(result: RunResult) -> bool:
    """True when no page could be analysed (e.g. a scanned-only PDF): the CLI exits 1."""
    return not result.document.objects and bool(
        result.document.meta.get("unreadable_pages") or result.document.meta.get("empty_pages"))
