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
    ``client`` overrides the client built from ``cfg`` (tests, custom backends)."""
    cfg = cfg or load_config()
    stages = set(stages)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    paths: dict[str, Path] = {}

    client = client if client is not None else _client(cfg)
    analyzer = None
    if client is not None:
        from .analysis.gemma import GemmaAnalyzer

        analyzer = GemmaAnalyzer(client, crop_max_side=cfg.llm.max_image_side)

    # 1. deterministic extraction
    t0 = time.perf_counter()
    doc = extract(pdf_path, cfg, pages)
    timings["extract"] = time.perf_counter() - t0
    paths["extracted"] = doc.save(out / "extracted.json")

    # 2. Gemma refinement of symbols / drawing info / text ownership
    if analyzer is not None and "refine" in stages:
        t0 = time.perf_counter()
        doc = analyzer.refine_document(doc, pdf_path)
        timings["refine"] = time.perf_counter() - t0
    paths["document"] = doc.save(out / "document.json")
    result = RunResult(document=doc, paths=paths, timings=timings,
                       llm_backend=f"{client.backend.name}:{client.model}" if client else "none")

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
        if planner == "gemma" and plan_client is None:
            raise ValueError("planner='gemma' needs an LLM backend (see `shippid llm-check`)")
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
    return out
