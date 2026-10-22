"""Benchmark the extraction (+ optional Gemma refinement) against a generated dataset.

Reads ``manifest.jsonl`` written by :func:`shippid.datasets.build.build_dataset`,
extracts every selected PDF, compares the result with its ground-truth JSON and
aggregates the metrics of :mod:`shippid.evaluate`.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Optional

from .config import Config, load_config
from .model import PIDDocument

log = logging.getLogger(__name__)


def load_manifest(data_dir: str | Path, split: Optional[str] = None, limit: Optional[int] = None,
                  systems: Optional[list[str]] = None) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    rows = []
    with (data_dir / "manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if split and split != "all" and r.get("split") != split:
                continue
            if systems and r.get("system") not in systems:
                continue
            rows.append(r)
    return rows[:limit] if limit else rows


def _paths(data_dir: Path, row: dict[str, Any]) -> tuple[Path, Path]:
    pdf = Path(row.get("pdf") or f"{row['id']}.pdf")
    gt = Path(row.get("gt") or f"{row['id']}.json")
    pdf = pdf if pdf.is_absolute() else data_dir / pdf.name
    gt = gt if gt.is_absolute() else data_dir / gt.name
    return pdf, gt


def _extract_one(pdf: str) -> dict[str, Any]:
    from .extract.pipeline import extract_pdf

    t0 = time.perf_counter()
    try:
        pred = extract_pdf(pdf)
        return {"pdf": pdf, "seconds": time.perf_counter() - t0, "pred": pred.model_dump_json()}
    except Exception as exc:  # keep the benchmark going; failures are reported
        log.exception("extraction failed for %s", pdf)
        return {"pdf": pdf, "error": f"{type(exc).__name__}: {exc}"}


def run_benchmark(data_dir: str | Path, out_dir: str | Path, *, split: str = "test", limit: Optional[int] = None,
                  workers: int = 1, cfg: Optional[Config] = None, use_llm: bool = False,
                  systems: Optional[list[str]] = None, save_predictions: bool = False) -> dict[str, Any]:
    """Extract + evaluate. With ``use_llm`` the Gemma refinement step runs too (sequentially)."""
    from .evaluate import evaluate_dataset, write_eval_report

    data_dir = Path(data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = out / "predictions" if save_predictions else None
    if pred_dir:
        pred_dir.mkdir(exist_ok=True)
    rows = load_manifest(data_dir, split, limit, systems)
    if not rows:
        raise ValueError(f"no manifest rows for split={split!r} in {data_dir}")
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    pairs: list[tuple[PIDDocument, PIDDocument]] = []
    per_doc: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    if use_llm:
        from .analysis.gemma import GemmaAnalyzer
        from .extract.pipeline import extract_pdf
        from .llm.factory import make_client

        client = make_client(cfg.llm)
        if client is None:
            raise ValueError("use_llm requires an LLM backend (SHIPPID_LLM_BACKEND / --llm)")
        analyzer = GemmaAnalyzer(client, crop_max_side=cfg.llm.max_image_side)
        for r in rows:
            pdf, gt = _paths(data_dir, r)
            try:
                pred = analyzer.refine_document(extract_pdf(pdf, cfg), pdf)
                if pred_dir:
                    pred.save(pred_dir / (pdf.stem + ".pred.json"))
                pairs.append((pred, PIDDocument.load(gt)))
                per_doc.append({"id": r["id"], "system": r.get("system")})
            except Exception as exc:
                errors.append({"id": r["id"], "error": f"{type(exc).__name__}: {exc}"})
        llm_usage = client.usage.model_dump()
    else:
        from .evaluate import evaluate_document

        jobs = [str(_paths(data_dir, r)[0]) for r in rows]
        if workers > 1:
            with mp.get_context().Pool(min(workers, len(jobs))) as pool:
                results = pool.map(_extract_one, jobs, chunksize=2)
        else:
            results = [_extract_one(j) for j in jobs]
        for r, res in zip(rows, results):
            if "error" in res:
                errors.append({"id": r["id"], "error": res["error"]})
                continue
            pred = PIDDocument.model_validate_json(res["pred"])
            gt = PIDDocument.load(_paths(data_dir, r)[1])
            if pred_dir:
                pred.save(pred_dir / f"{r['id']}.pred.json")
            m = evaluate_document(pred, gt)
            per_doc.append({"id": r["id"], "system": r.get("system"), "seconds": round(res["seconds"], 3),
                            "detection_f1": m.get("detection", {}).get("f1"),
                            "connection_f1": m.get("connections", {}).get("f1")})
            pairs.append((pred, gt))
        llm_usage = None

    result = evaluate_dataset(pairs)
    result["benchmark"] = {
        "data_dir": str(data_dir), "split": split, "documents": len(pairs), "errors": errors,
        "seconds": round(time.perf_counter() - t0, 2), "llm": "gemma" if use_llm else "none",
        "llm_usage": llm_usage, "per_document": per_doc,
    }
    json_path, md_path = write_eval_report(result, out)
    result["benchmark"]["report_json"] = str(json_path)
    result["benchmark"]["report_md"] = str(md_path)
    return result
