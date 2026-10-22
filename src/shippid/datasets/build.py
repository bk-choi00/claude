"""Mass generation of the synthetic ship P&ID dataset.

``build_dataset(out_dir, count, seed=0, workers=os.cpu_count(), systems=None, previews=False)``
writes for every item ``<id>.pdf`` (readable vector PDF), ``<id>.json`` (ground-truth
:class:`~shippid.model.PIDDocument`) and optionally ``previews/<id>.png``, plus
``manifest.jsonl`` (one record per item, in index order) and ``summary.json``.

Every item depends only on ``(seed, index)``: the output is identical for any number of
workers. Systems are assigned round-robin over a seed-shuffled system list (balanced
classes); splits are an exact 80/10/10 partition of a seed-shuffled index list.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from ..model import SystemType
from .pdf_writer import render_png, to_bytes
from .synthetic import ALL_SYSTEMS, generate

log = logging.getLogger(__name__)

SPLITS = (("train", 0.8), ("val", 0.1), ("test", 0.1))


def item_seed(seed: int, index: int) -> int:
    """Deterministic 63-bit seed of dataset item ``index``."""
    h = hashlib.sha256(f"shippid:{seed}:{index}".encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def item_id(seed: int, index: int) -> str:
    return f"syn{seed:04d}_{index:06d}"


def assign_systems(count: int, seed: int, systems: Optional[Sequence[SystemType]] = None) -> list[SystemType]:
    pool = list(systems) if systems else list(ALL_SYSTEMS)
    pool = [SystemType(s) for s in pool]
    if not pool or SystemType.UNKNOWN in pool:
        raise ValueError("systems must be a non-empty list of known SystemType values")
    random.Random(f"systems:{seed}").shuffle(pool)
    return [pool[i % len(pool)] for i in range(count)]


def assign_splits(count: int, seed: int) -> list[str]:
    order = list(range(count))
    random.Random(f"splits:{seed}").shuffle(order)
    n_train = int(round(count * SPLITS[0][1]))
    n_val = int(round(count * SPLITS[1][1]))
    out = [""] * count
    for rank, idx in enumerate(order):
        out[idx] = "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
    return out


def _build_one(job: tuple[str, int, int, str, str, bool, int]) -> dict[str, Any]:
    out_dir, index, seed, system, split, previews, dpi = job
    iid = item_id(seed, index)
    s = item_seed(seed, index)
    out = Path(out_dir)
    t0 = time.perf_counter()
    try:
        doc, spec = generate(s, SystemType(system))
        pdf_bytes = to_bytes(spec)
        (out / f"{iid}.pdf").write_bytes(pdf_bytes)
        doc.source = f"{iid}.pdf"
        (out / f"{iid}.json").write_text(doc.to_json(), encoding="utf-8")
        preview = None
        if previews:
            preview = f"previews/{iid}.png"
            render_png(pdf_bytes, out / preview, dpi=dpi)
        cats = Counter(o.category.value for o in doc.objects)
        classes = Counter(o.cls.value for o in doc.objects)
        page = doc.pages[0]
        return {
            "id": iid, "index": index, "seed": s, "split": split, "pdf": f"{iid}.pdf", "gt": f"{iid}.json",
            "preview": preview, "system": doc.info.system.value, "ship_type": doc.info.ship_type,
            "hull_no": doc.info.hull_no, "drawing_no": doc.info.drawing_no, "paper": spec.paper,
            "width": page.width, "height": page.height,
            "counts": {"objects": len(doc.objects), "pipes": len(doc.pipes), "junctions": len(doc.junctions),
                       "texts": len(doc.texts), "crossings": doc.meta.get("crossings", 0), **dict(sorted(cats.items()))},
            "classes": dict(sorted(classes.items())), "bytes": len(pdf_bytes),
            "seconds": round(time.perf_counter() - t0, 4),
        }
    except Exception as exc:  # recorded in the manifest summary, never silently dropped
        log.exception("item %s failed", iid)
        return {"id": iid, "index": index, "seed": s, "split": split, "system": system,
                "error": f"{type(exc).__name__}: {exc}"}


def build_dataset(out_dir: str | Path, count: int, seed: int = 0, workers: Optional[int] = None,
                  systems: Optional[Iterable[SystemType | str]] = None, previews: bool = False,
                  preview_dpi: int = 50, chunksize: int = 4) -> dict[str, Any]:
    """Generate ``count`` drawings into ``out_dir``; returns (and writes) a summary dict."""
    if count < 0:
        raise ValueError("count must be >= 0")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if previews:
        (out / "previews").mkdir(exist_ok=True)
    sys_list = assign_systems(count, seed, [SystemType(s) for s in systems] if systems else None)
    splits = assign_splits(count, seed)
    jobs = [(str(out), i, seed, sys_list[i].value, splits[i], previews, preview_dpi) for i in range(count)]
    workers = max(1, int(workers if workers is not None else (os.cpu_count() or 1)))
    t0 = time.perf_counter()
    if workers == 1 or count <= 1:
        results = [_build_one(j) for j in jobs]
    else:
        with mp.get_context().Pool(processes=min(workers, count)) as pool:
            results = list(pool.imap_unordered(_build_one, jobs, chunksize=max(1, chunksize)))
    seconds = time.perf_counter() - t0
    results.sort(key=lambda r: r["index"])
    manifest = out / "manifest.jsonl"
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    with manifest.open("w", encoding="utf-8") as fh:
        for r in ok:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "out_dir": str(out), "manifest": str(manifest), "count": count, "ok": len(ok), "failed": len(failed),
        "seed": seed, "workers": workers, "seconds": round(seconds, 3),
        "drawings_per_second": round(len(ok) / seconds, 2) if seconds > 0 else None,
        "by_system": dict(sorted(Counter(r["system"] for r in ok).items())),
        "by_split": dict(sorted(Counter(r["split"] for r in ok).items())),
        "by_paper": dict(sorted(Counter(r["paper"] for r in ok).items())),
        "objects": sum(r["counts"]["objects"] for r in ok),
        "failures": [{"id": r["id"], "system": r["system"], "error": r["error"]} for r in failed],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
