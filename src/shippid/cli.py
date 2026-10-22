"""Command line interface: ``shippid <command> ...``.

Commands
    collect synthetic   generate a synthetic ship P&ID dataset (readable vector PDFs + ground truth)
    collect scan        index an archive of real PDF drawings (readability, dedupe, system guess)
    collect fetch       download PDFs listed in a sources YAML
    extract             deterministic vector extraction -> PIDDocument JSON
    analyze             extraction + Gemma refinement + analysis + HTML/CSV report
    model3d             3D arrangement (Gemma-planned or heuristic) -> GLB + viewer
    run                 all stages on one PDF
    evaluate            benchmark extraction against a generated dataset
    llm-check           test the connection to the Gemma 4 31B endpoint
    symbols             write the symbol library legend as a PDF
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from .config import Config, load_config


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def _add_llm_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Gemma 4 31B (LLM)")
    g.add_argument("--llm", dest="llm_backend", default=None,
                   help="backend: openai|vllm|sglang|ollama|llamacpp|lmstudio|gemini|none "
                        "(default: $SHIPPID_LLM_BACKEND or config; 'none' = deterministic only)")
    g.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. http://gpu-host:8000/v1")
    g.add_argument("--model", default=None, help="model id (default google/gemma-4-31B-it; Gemini API: gemma-4-31b-it)")
    g.add_argument("--api-key", default=None, help="API key (prefer $SHIPPID_LLM_API_KEY)")
    g.add_argument("--llm-cache", default=None, help="directory for the request/response replay cache")
    g.add_argument("--max-image-side", type=int, default=None, help="max pixel side of images sent to Gemma")
    g.add_argument("--config", default=None, help="shippid.yaml path")


def _config(args: argparse.Namespace) -> Config:
    overrides = {
        "backend": args.llm_backend, "base_url": args.base_url, "model": args.model,
        "api_key": args.api_key, "cache_dir": args.llm_cache, "max_image_side": args.max_image_side,
    }
    cfg = load_config(args.config, **{k: v for k, v in overrides.items() if v is not None})
    if args.base_url is not None:
        # an explicit --base-url wins over the per-backend default endpoint (llm.factory)
        cfg.llm.base_url_explicit = True  # type: ignore[attr-defined]
    return cfg


# --------------------------------------------------------------------------- commands


def cmd_collect_synthetic(args: argparse.Namespace) -> int:
    from .datasets.build import build_dataset

    systems = [s.strip() for s in args.systems.split(",")] if args.systems else None
    summary = build_dataset(args.out, args.count, seed=args.seed, workers=args.workers, systems=systems,
                            previews=args.previews)
    _print(summary)
    return 0 if summary.get("failed", 0) == 0 else 1


def cmd_collect_scan(args: argparse.Namespace) -> int:
    from .datasets.collect import scan_archive

    res = scan_archive(args.root, args.index)
    _print(res if not isinstance(res, list) else {"records": len(res), "index": args.index})
    return 0


def cmd_collect_fetch(args: argparse.Namespace) -> int:
    from .datasets.collect import fetch_sources

    _print(fetch_sources(args.sources, args.dest))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from .extract.pipeline import extract_pdf

    cfg = load_config(args.config)
    pages = [int(p) for p in args.pages.split(",")] if args.pages else None
    doc = extract_pdf(args.pdf, cfg, pages=pages)
    out = Path(args.output or Path(args.pdf).with_suffix(".extracted.json"))
    doc.save(out)
    _print({"output": str(out), "objects": len(doc.objects), "pipes": len(doc.pipes), "texts": len(doc.texts),
            "junctions": len(doc.junctions), "system": doc.info.system.value, "seconds": doc.meta.get("seconds")})
    return 0


def cmd_run(args: argparse.Namespace, stages: Sequence[str]) -> int:
    from .pipeline import run_pipeline, summarize

    cfg = _config(args)
    pages = [int(p) for p in args.pages.split(",")] if getattr(args, "pages", None) else None
    result = run_pipeline(args.pdf, args.out, cfg, pages=pages, planner=getattr(args, "planner", "auto"),
                          viewer=getattr(args, "viewer", "threejs"), stages=stages)
    _print(summarize(result))
    return 0


def cmd_model3d(args: argparse.Namespace) -> int:
    """3D from a PDF, a PIDDocument JSON or an AnalysisReport JSON."""
    src = Path(args.input)
    if src.suffix.lower() == ".pdf":
        args.pdf = str(src)
        return cmd_run(args, ("extract", "refine", "model3d"))
    from .model import AnalysisReport, PIDDocument
    from .model3d.builder import build_layout
    from .model3d.scene import export_glb, layout_to_scene
    from .model3d.viewer import write_viewer_html
    from .pipeline import _client

    data = json.loads(src.read_text(encoding="utf-8"))
    doc = AnalysisReport.model_validate(data).document if "document" in data else PIDDocument.model_validate(data)
    cfg = _config(args)
    client = None if args.planner == "heuristic" else _client(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layout = build_layout(doc, client=client)
    layout.save(out / "layout3d.json")
    glb = export_glb(layout_to_scene(layout, doc), out / "model.glb")
    modes = ["threejs", "plotly"] if args.viewer == "both" else [args.viewer]
    viewers = [str(write_viewer_html(glb, layout, doc, out / ("viewer.html" if i == 0 else f"viewer_{m}.html"), mode=m))
               for i, m in enumerate(modes)]
    _print({"planner": layout.planner, "placements": len(layout.placements), "routes": len(layout.routes),
            "glb": str(glb), "viewers": viewers, "notes": layout.notes[:10], "llm_usage": layout.llm.model_dump()})
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark

    cfg = _config(args)
    systems = [s.strip() for s in args.systems.split(",")] if args.systems else None
    res = run_benchmark(args.data, args.out, split=args.split, limit=args.limit, workers=args.workers, cfg=cfg,
                        use_llm=cfg.llm.enabled and args.with_llm, systems=systems,
                        save_predictions=args.save_predictions)
    bench = res.get("benchmark", {})
    _print({"micro": res.get("micro"), "macro": res.get("macro"), "documents": bench.get("documents"),
            "errors": len(bench.get("errors", [])), "seconds": bench.get("seconds"),
            "report_md": bench.get("report_md")})
    return 0


def cmd_llm_check(args: argparse.Namespace) -> int:
    from .llm.factory import check_connection, make_client

    cfg = _config(args)
    client = make_client(cfg.llm)
    if client is None:
        _print({"ok": False, "error": "LLM backend is 'none'; set --llm/--base-url or SHIPPID_LLM_*"})
        return 2
    res = check_connection(client)
    _print(res)
    return 0 if res.get("ok") else 1


def cmd_symbols(args: argparse.Namespace) -> int:
    import pymupdf

    from .model import ActuatorType, Category
    from .symbols import Placement, all_symbols, draw_symbol, placed_ports

    doc = pymupdf.open()
    page = doc.new_page(width=1190, height=842)
    page.insert_text((40, 40), "shippid symbol library (local frame: flow along +x; red dots = ports)", fontsize=12)
    shape = page.new_shape()
    specs = all_symbols()
    for i, spec in enumerate(specs):
        cx, cy = 90 + (i % 8) * 140, 110 + (i // 8) * 150
        draw_symbol(shape, spec, Placement(cx, cy, scale=28))
        for _, (x, y), _ in placed_ports(spec, Placement(cx, cy, scale=28)):
            shape.draw_circle(pymupdf.Point(x, y), 1.8)
            shape.finish(color=(0.85, 0, 0), fill=(0.85, 0, 0))
        page.insert_text((cx - 55, cy + 50), spec.cls.value, fontsize=7)
        page.insert_text((cx - 55, cy + 59), spec.label_en, fontsize=6, color=(0.3, 0.3, 0.3))
    y = 110 + ((len(specs) - 1) // 8 + 1) * 150
    for j, act in enumerate([a for a in ActuatorType if a != ActuatorType.NONE]):
        cx = 90 + j * 140
        from .model import SymbolClass
        from .symbols import get_symbol

        draw_symbol(shape, get_symbol(SymbolClass.GLOBE_VALVE), Placement(cx, y, scale=28), act)
        page.insert_text((cx - 55, y + 40), f"actuator: {act.value}", fontsize=7)
    shape.commit()
    doc.save(args.output)
    _print({"output": args.output, "symbols": len(specs)})
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shippid", description="Ship P&ID (readable PDF) extraction, "
                                "Gemma 4 31B analysis and 3D model generation")
    p.add_argument("-v", "--verbose", action="count", default=0)
    sub = p.add_subparsers(dest="command", required=True)

    col = sub.add_parser("collect", help="collect drawings").add_subparsers(dest="what", required=True)
    s = col.add_parser("synthetic", help="generate synthetic ship P&IDs (vector PDF + ground truth)")
    s.add_argument("--out", required=True)
    s.add_argument("--count", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--systems", default=None, help="comma separated SystemType values (default: all)")
    s.add_argument("--previews", action="store_true", help="also write PNG previews")
    s.set_defaults(func=cmd_collect_synthetic)
    s = col.add_parser("scan", help="index a directory tree of real PDF drawings")
    s.add_argument("root")
    s.add_argument("--index", default="index.jsonl")
    s.set_defaults(func=cmd_collect_scan)
    s = col.add_parser("fetch", help="download PDFs listed in a sources YAML")
    s.add_argument("--sources", required=True)
    s.add_argument("--dest", required=True)
    s.set_defaults(func=cmd_collect_fetch)

    s = sub.add_parser("extract", help="deterministic vector extraction to JSON")
    s.add_argument("pdf")
    s.add_argument("-o", "--output", default=None)
    s.add_argument("--pages", default=None, help="comma separated 0-based page indices")
    s.add_argument("--config", default=None)
    s.set_defaults(func=cmd_extract)

    for name, stages, help_ in (
        ("analyze", ("extract", "refine", "analyze", "report"), "extract + Gemma refinement + analysis + report"),
        ("run", ("extract", "refine", "analyze", "model3d", "report"), "all stages (analysis, 3D, reports)"),
    ):
        s = sub.add_parser(name, help=help_)
        s.add_argument("pdf")
        s.add_argument("-o", "--out", required=True, help="output directory")
        s.add_argument("--pages", default=None)
        s.add_argument("--planner", choices=["auto", "gemma", "heuristic"], default="auto")
        s.add_argument("--viewer", choices=["threejs", "plotly", "both"], default="threejs")
        _add_llm_args(s)
        s.set_defaults(func=lambda a, st=stages: cmd_run(a, st))

    s = sub.add_parser("model3d", help="3D model from a PDF, document JSON or analysis JSON")
    s.add_argument("input")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--pages", default=None)
    s.add_argument("--planner", choices=["auto", "gemma", "heuristic"], default="auto")
    s.add_argument("--viewer", choices=["threejs", "plotly", "both"], default="threejs")
    _add_llm_args(s)
    s.set_defaults(func=cmd_model3d)

    s = sub.add_parser("evaluate", help="benchmark extraction on a generated dataset")
    s.add_argument("--data", required=True, help="dataset directory with manifest.jsonl")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--split", default="test", help="train|val|test|all")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--systems", default=None)
    s.add_argument("--with-llm", action="store_true", help="also run Gemma refinement before scoring")
    s.add_argument("--save-predictions", action="store_true")
    _add_llm_args(s)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("llm-check", help="test the Gemma 4 31B endpoint (text + image + JSON)")
    _add_llm_args(s)
    s.set_defaults(func=cmd_llm_check)

    s = sub.add_parser("symbols", help="write the symbol library legend PDF")
    s.add_argument("-o", "--output", default="shippid_symbols.pdf")
    s.set_defaults(func=cmd_symbols)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * min(args.verbose, 2),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
