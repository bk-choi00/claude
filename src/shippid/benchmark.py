"""Benchmark the extraction (+ optional Gemma refinement) against a generated dataset.

Reads ``manifest.jsonl`` written by :func:`shippid.datasets.build.build_dataset` (or
:func:`shippid.datasets.variants.build_variant_suite`), extracts every selected PDF with the
given :class:`~shippid.config.Config`, pairs the off-page connectors of multi-page documents
across the sheets (:func:`shippid.pipeline.link_multisheet`, as ``shippid run`` / ``analyze()``
do, so cross-sheet connections are scored like single-sheet ones), compares the result with its
ground-truth JSON and aggregates the metrics of :mod:`shippid.evaluate` (every document is
evaluated once; predictions are parsed one at a time while they are scored).

``evaluation.json`` records under ``benchmark`` the data set, the documents, the errors, the
timings and ``llm`` -- ``"none"`` or ``"<backend>:<model>"`` of the client that refined the
predictions (a mock backend is recorded as ``mock:...``, never as Gemma). Saved predictions are
named after the sanitised manifest id (``predictions/<id>.pred.json``, never outside
``predictions/``).
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

from .config import Config, load_config
from .model import PIDDocument

if TYPE_CHECKING:  # pragma: no cover
    from .llm.base import LLMClient

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


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def prediction_name(row_id: Any, pdf: Path, used: Optional[set[str]] = None) -> str:
    """File name (``<name>.pred.json``) of a saved prediction: the manifest id reduced to
    ``[A-Za-z0-9._-]`` (no separators, no leading dots), else the PDF stem; made unique
    (case-insensitively) against ``used``, which is updated."""
    name = ""
    for raw in (row_id, pdf.stem, "document"):
        name = _UNSAFE.sub("_", str(raw or "")).strip("._")[:120]
        if name:
            break
    base, k = name, 1
    while used is not None and name.lower() in used:
        k += 1
        name = f"{base}_{k}"
    if used is not None:
        used.add(name.lower())
    return f"{name}.pred.json"


def _prediction_path(pred_dir: Path, name: str) -> Path:
    target = pred_dir / name
    if target.resolve().parent != pred_dir.resolve():  # defence in depth: prediction_name has no separators
        raise ValueError(f"prediction file {name!r} would be written outside {pred_dir}")
    return target


def _extract_one(job: tuple[str, Optional[Config]]) -> dict[str, Any]:
    from .extract.pipeline import extract_pdf
    from .pipeline import link_multisheet

    pdf, cfg = job
    t0 = time.perf_counter()
    try:
        pred = link_multisheet(extract_pdf(pdf, cfg))
        return {"pdf": pdf, "seconds": time.perf_counter() - t0, "pred": pred.model_dump_json()}
    except Exception as exc:  # keep the benchmark going; failures are reported
        log.exception("extraction failed for %s", pdf)
        return {"pdf": pdf, "error": f"{type(exc).__name__}: {exc}"}


def run_benchmark(data_dir: str | Path, out_dir: str | Path, *, split: str = "test", limit: Optional[int] = None,
                  workers: int = 1, cfg: Optional[Config] = None, use_llm: bool = False,
                  systems: Optional[list[str]] = None, save_predictions: bool = False,
                  client: Optional["LLMClient"] = None) -> dict[str, Any]:
    """Extract (with ``cfg``) + link sheets + evaluate. With ``use_llm`` the Gemma refinement runs
    too (sequentially, with the client built from ``cfg.llm`` or the given ``client``)."""
    from .evaluate import evaluate_dataset, write_eval_report
    from .pipeline import link_multisheet, llm_label, make_gemma_analyzer

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
    errors: list[dict[str, str]] = []
    done: list[tuple[dict[str, Any], dict[str, Any]]] = []  # (manifest row, result with the prediction JSON)

    if use_llm:
        from .extract.pipeline import extract_pdf
        from .llm.factory import make_client

        client = client if client is not None else make_client(cfg.llm)
        if client is None:
            raise ValueError("use_llm requires an LLM backend (SHIPPID_LLM_BACKEND / --llm)")
        analyzer = make_gemma_analyzer(client, cfg)
        for r in rows:
            pdf, _ = _paths(data_dir, r)
            t1 = time.perf_counter()
            try:
                pred = analyzer.refine_document(extract_pdf(pdf, cfg), pdf)
                link_multisheet(pred)  # the Gemma sheet-link step failed or did not run
                done.append((r, {"seconds": time.perf_counter() - t1, "pred": pred.model_dump_json()}))
            except Exception as exc:
                log.exception("extraction / refinement failed for %s", pdf)
                errors.append({"id": r["id"], "error": f"{type(exc).__name__}: {exc}"})
        llm_usage = client.usage.model_dump()
    else:
        jobs = [(str(_paths(data_dir, r)[0]), cfg) for r in rows]
        if workers > 1:
            with mp.get_context().Pool(min(workers, len(jobs))) as pool:
                results = pool.map(_extract_one, jobs, chunksize=2)
        else:
            results = [_extract_one(j) for j in jobs]
        for r, res in zip(rows, results):
            if "error" in res:
                errors.append({"id": r["id"], "error": res["error"]})
            else:
                done.append((r, res))
        llm_usage = None

    scored: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for r, res in done:
        gt_path = _paths(data_dir, r)[1]
        if not gt_path.is_file():
            errors.append({"id": r["id"], "error": f"ground truth not found: {gt_path}"})
        else:
            scored.append((r, res))
    used: set[str] = set()

    def pairs() -> Iterator[tuple[PIDDocument, PIDDocument]]:
        for r, res in scored:  # one prediction / ground truth pair in memory at a time
            pred = PIDDocument.model_validate_json(res.pop("pred"))
            if pred_dir:
                pred.save(_prediction_path(pred_dir, prediction_name(r.get("id"), _paths(data_dir, r)[0], used)))
            yield pred, PIDDocument.load(_paths(data_dir, r)[1])

    result = evaluate_dataset(pairs())
    per_doc = [{"id": r["id"], "system": r.get("system"), "seconds": round(res["seconds"], 3),
                "detection_f1": d.get("detection_f1"), "connection_f1": d.get("connection_f1")}
               for (r, res), d in zip(scored, result["documents"])]
    result["benchmark"] = {
        "data_dir": str(data_dir), "split": split, "documents": len(scored), "errors": errors,
        "seconds": round(time.perf_counter() - t0, 2), "llm": llm_label(client) if use_llm else "none",
        "llm_usage": llm_usage, "workers": 1 if use_llm else max(1, workers), "per_document": per_doc,
    }
    json_path, md_path = write_eval_report(result, out)
    _append_benchmark_md(md_path, result["benchmark"])
    result["benchmark"]["report_json"] = str(json_path)
    result["benchmark"]["report_md"] = str(md_path)
    return result


def _append_benchmark_md(md_path: Path, bench: dict[str, Any]) -> None:
    usage = bench.get("llm_usage") or {}
    llm = bench["llm"] if not usage else f"{bench['llm']} ({usage.get('calls', 0)} calls, backend {usage.get('backend')})"
    lines = ["", "## Benchmark", "", "| | |", "|---|---|", f"| data | `{bench['data_dir']}` (split {bench['split']}) |",
             f"| documents scored | {bench['documents']} |", f"| errors | {len(bench['errors'])} |",
             f"| LLM | {llm} |", f"| wall time | {bench['seconds']} s |"]
    if bench["errors"]:
        lines += ["", "| id | error |", "|---|---|"]
        lines += [f"| {e['id']} | {str(e['error'])[:200].replace('|', '/')} |" for e in bench["errors"][:50]]
    with md_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
