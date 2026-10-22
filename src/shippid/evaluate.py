"""Evaluation of an extracted :class:`~shippid.model.PIDDocument` against ground truth.

Metrics (all on the same page only):

* **objects** -- one-to-one Hungarian matching on bounding-box IoU >= ``iou``: class-aware
  (per-class / per-category TP, FP, FN, precision, recall, F1) and class-agnostic (detection);
  class accuracy and a confusion table on the class-agnostic matches;
* **tags** (normalised text equality) and **actuators** (valves) on class-agnostic matches;
* **connections** -- precision / recall / F1 over object pairs of
  :func:`shippid.analysis.graph.object_adjacency` (junction-collapsed, signal lines excluded),
  predicted pairs mapped into ground-truth ids through the class-agnostic matching (pairs with an
  unmatched object count as false positives);
* **pipes** -- length-based coverage of ground-truth lines by predicted lines (and vice versa)
  within ``pipe_tol`` points; line-number and size accuracy on ground-truth pipes, each compared
  with the predicted pipe that covers most of it;
* **texts** -- Hungarian matching on IoU >= ``text_iou``; role, content and owner accuracy.

Undefined ratios (no denominator) are ``None``. :func:`evaluate_dataset` aggregates documents
micro (summed counts) and macro (mean of per-document values) and :func:`write_eval_report`
writes JSON + Markdown.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .analysis.graph import object_adjacency
from .model import BBox, Category, PIDDocument, PipeKind, PipeRun

_DASHES = re.compile(r"[‐‑‒–—―−]")


# --------------------------------------------------------------------------- helpers


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Counts plus precision / recall / F1 (``None`` when undefined)."""
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def _acc(correct: int, total: int, **extra: Any) -> dict[str, Any]:
    return {"correct": correct, "total": total, "accuracy": correct / total if total else None, **extra}


def norm_text(s: Optional[str]) -> str:
    """Upper case, unified dashes, no whitespace."""
    return re.sub(r"\s+", "", _DASHES.sub("-", s or "")).upper()


def match_boxes(pred: Sequence[BBox], gt: Sequence[BBox], thr: float,
                compatible: Optional[Callable[[int, int], bool]] = None) -> list[tuple[int, int, float]]:
    """Maximum-IoU one-to-one matching (Hungarian); returns (pred index, gt index, IoU) with
    IoU >= ``thr`` and ``compatible(pi, gi)``."""
    if not pred or not gt:
        return []
    P = np.array([b.as_tuple() for b in pred], dtype=float)
    G = np.array([b.as_tuple() for b in gt], dtype=float)
    ix0 = np.maximum(P[:, None, 0], G[None, :, 0])
    iy0 = np.maximum(P[:, None, 1], G[None, :, 1])
    ix1 = np.minimum(P[:, None, 2], G[None, :, 2])
    iy1 = np.minimum(P[:, None, 3], G[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    pa = np.clip(P[:, 2] - P[:, 0], 0, None) * np.clip(P[:, 3] - P[:, 1], 0, None)
    ga = np.clip(G[:, 2] - G[:, 0], 0, None) * np.clip(G[:, 3] - G[:, 1], 0, None)
    union = pa[:, None] + ga[None, :] - inter
    iou = np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)
    valid = iou >= thr
    if compatible is not None:
        for i in range(len(pred)):
            for j in range(len(gt)):
                if valid[i, j] and not compatible(i, j):
                    valid[i, j] = False
    if not valid.any():
        return []
    cost = np.where(valid, 1.0 - iou, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(iou[i, j])) for i, j in zip(rows, cols) if valid[i, j]]


def _by_page(items: Sequence[Any]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for k, it in enumerate(items):
        out[it.page].append(k)
    return out


def _paged_match(pred: Sequence[Any], gt: Sequence[Any], thr: float,
                 same: Optional[Callable[[Any, Any], bool]] = None) -> list[tuple[int, int, float]]:
    pp, gp = _by_page(pred), _by_page(gt)
    out = []
    for page in sorted(set(pp) & set(gp)):
        pi, gi = pp[page], gp[page]
        comp = (lambda a, b: same(pred[pi[a]], gt[gi[b]])) if same else None
        for a, b, v in match_boxes([pred[k].bbox for k in pi], [gt[k].bbox for k in gi], thr, comp):
            out.append((pi[a], gi[b], v))
    return out


def _line(p: PipeRun) -> Optional[LineString]:
    pts = [(q.x, q.y) for q in p.points]
    pts = [pt for k, pt in enumerate(pts) if k == 0 or pt != pts[k - 1]]
    if len(pts) < 2:
        return None
    ls = LineString(pts)
    return ls if ls.length > 0 else None


def _coverage(a: list[LineString], b: list[LineString], tol: float) -> tuple[float, float]:
    """(length of ``a`` within ``tol`` of ``b``, total length of ``a``)."""
    total = float(sum(x.length for x in a))
    if not a or not b:
        return 0.0, total
    zone = unary_union([x.buffer(tol, cap_style=2) for x in b])
    return float(sum(x.intersection(zone).length for x in a)), total


def _pipe_map(pred: PIDDocument, gt: PIDDocument, tol: float,
              kinds: Iterable[PipeKind] = (PipeKind.PROCESS, PipeKind.INSTRUMENT_TAP)) -> dict[str, tuple[str, float]]:
    """gt pipe id -> (pred pipe id covering most of it, covered fraction of the gt pipe)."""
    kinds = set(kinds)
    out: dict[str, tuple[str, float]] = {}
    pred_lines = [(p, _line(p)) for p in pred.pipes if p.kind in kinds]
    pred_lines = [(p, ls) for p, ls in pred_lines if ls is not None]
    if not pred_lines:
        return out
    tree = STRtree([ls for _, ls in pred_lines])
    for g in gt.pipes:
        if g.kind not in kinds:
            continue
        gl = _line(g)
        if gl is None:
            continue
        zone = gl.buffer(tol, cap_style=2)
        best = None
        for k in tree.query(zone):
            p, pl = pred_lines[int(k)]
            if p.page != g.page:
                continue
            # length of the gt line explained by this pred pipe
            cov = gl.intersection(pl.buffer(tol, cap_style=2)).length
            if best is None or cov > best[1]:
                best = (p.id, cov)
        if best and best[1] >= 0.5 * gl.length:
            out[g.id] = (best[0], best[1] / gl.length)
    return out


def _pred_pipe_map(pred: PIDDocument, gt: PIDDocument, tol: float) -> dict[str, str]:
    """pred pipe id -> gt pipe id it overlaps most (for text owner mapping)."""
    inv = _pipe_map(gt, pred, tol, kinds=tuple(PipeKind))
    return {pid: gid for pid, (gid, _) in inv.items()}


# --------------------------------------------------------------------------- document


def evaluate_document(pred: PIDDocument, gt: PIDDocument, iou: float = 0.5, *, text_iou: float = 0.5,
                      pipe_tol: float = 3.0) -> dict[str, Any]:
    """Compare one predicted document with its ground truth (see module docstring)."""
    po, go = pred.objects, gt.objects
    agn = _paged_match(po, go, iou)
    cls_m = _paged_match(po, go, iou, lambda a, b: a.cls == b.cls)
    cat_m = _paged_match(po, go, iou, lambda a, b: a.category == b.category)

    per_class: dict[str, dict[str, Any]] = {}
    tp_c = Counter(go[g].cls.value for _, g, _ in cls_m)
    n_pred_c = Counter(o.cls.value for o in po)
    n_gt_c = Counter(o.cls.value for o in go)
    for c in sorted(set(n_pred_c) | set(n_gt_c)):
        row = prf(tp_c[c], n_pred_c[c] - tp_c[c], n_gt_c[c] - tp_c[c])
        row["support"] = n_gt_c[c]
        per_class[c] = row
    per_category: dict[str, dict[str, Any]] = {}
    tp_k = Counter(go[g].category.value for _, g, _ in cat_m)
    n_pred_k = Counter(o.category.value for o in po)
    n_gt_k = Counter(o.category.value for o in go)
    for c in sorted(set(n_pred_k) | set(n_gt_k)):
        row = prf(tp_k[c], n_pred_k[c] - tp_k[c], n_gt_k[c] - tp_k[c])
        row["support"] = n_gt_k[c]
        per_category[c] = row

    confusion: dict[str, Counter] = defaultdict(Counter)
    cls_ok = 0
    tag = Counter()
    act = Counter()
    for p, g, _ in agn:
        pg, gg = po[p], go[g]
        confusion[gg.cls.value][pg.cls.value] += 1
        cls_ok += pg.cls == gg.cls
        if gg.tag:
            tag["total"] += 1
            if norm_text(pg.tag) == norm_text(gg.tag):
                tag["correct"] += 1
            elif not pg.tag:
                tag["missing"] += 1
            else:
                tag["wrong"] += 1
        elif pg.tag:
            tag["spurious"] += 1
        if gg.category == Category.VALVE:
            act["total"] += 1
            act["correct"] += pg.actuator == gg.actuator

    # connections
    pmap = {po[p].id: go[g].id for p, g, _ in agn}
    gt_pairs = object_adjacency(gt)
    pred_pairs = object_adjacency(pred)
    mapped, unmapped = set(), 0
    for pair in pred_pairs:
        a, b = tuple(pair)
        if a in pmap and b in pmap:
            mapped.add(frozenset((pmap[a], pmap[b])))
        else:
            unmapped += 1
    ctp = len(mapped & gt_pairs)
    conn = prf(ctp, len(pred_pairs) - ctp, len(gt_pairs) - ctp)
    conn.update({"gt_pairs": len(gt_pairs), "pred_pairs": len(pred_pairs), "unmapped_pred_pairs": unmapped})

    # pipes
    pipes: dict[str, Any] = {}
    for name, kinds in (("process", (PipeKind.PROCESS, PipeKind.INSTRUMENT_TAP)), ("signal", (PipeKind.SIGNAL,))):
        cov_g = cov_p = tot_g = tot_p = 0.0
        for page in sorted({p.page for p in gt.pipes} | {p.page for p in pred.pipes}):
            gl = [ls for p in gt.pipes if p.page == page and p.kind in kinds for ls in [_line(p)] if ls]
            pl = [ls for p in pred.pipes if p.page == page and p.kind in kinds for ls in [_line(p)] if ls]
            c, t = _coverage(gl, pl, pipe_tol)
            cov_g, tot_g = cov_g + c, tot_g + t
            c, t = _coverage(pl, gl, pipe_tol)
            cov_p, tot_p = cov_p + c, tot_p + t
        rec = cov_g / tot_g if tot_g else None
        pre = cov_p / tot_p if tot_p else None
        f1 = 2 * pre * rec / (pre + rec) if pre is not None and rec is not None and pre + rec > 0 else (
            0.0 if pre is not None and rec is not None else None)
        pipes[name] = {"gt_length": round(tot_g, 1), "pred_length": round(tot_p, 1),
                       "gt_covered": round(cov_g, 1), "pred_covered": round(cov_p, 1),
                       "recall": rec, "precision": pre, "f1": f1}
    gmap = _pipe_map(pred, gt, pipe_tol)
    pred_by_id = {p.id: p for p in pred.pipes}
    ln = Counter()
    sz = Counter()
    for g in gt.pipes:
        if g.kind == PipeKind.SIGNAL:
            continue
        m = gmap.get(g.id)
        pp = pred_by_id[m[0]] if m else None
        if g.line_number:
            ln["total"] += 1
            ln["matched"] += pp is not None
            ln["correct"] += pp is not None and norm_text(pp.line_number) == norm_text(g.line_number)
        if g.size:
            sz["total"] += 1
            sz["correct"] += pp is not None and norm_text(pp.size) == norm_text(g.size)
    pipes["matched_gt_pipes"] = len(gmap)
    pipes["gt_pipes"] = sum(1 for p in gt.pipes if p.kind != PipeKind.SIGNAL)

    # texts
    tm = _paged_match(pred.texts, gt.texts, text_iou)
    pt_map = _pred_pipe_map(pred, gt, pipe_tol)
    roles = Counter()
    content = Counter()
    owner = Counter()
    role_conf: dict[str, Counter] = defaultdict(Counter)
    for p, g, _ in tm:
        pt, gtx = pred.texts[p], gt.texts[g]
        roles["correct"] += pt.role == gtx.role
        role_conf[gtx.role.value][pt.role.value] += 1
        content["correct"] += norm_text(pt.text) == norm_text(gtx.text)
        if gtx.owner_id:
            owner["total"] += 1
            mapped_owner = pmap.get(pt.owner_id or "") or pt_map.get(pt.owner_id or "")
            owner["correct"] += mapped_owner == gtx.owner_id
    texts = prf(len(tm), len(pred.texts) - len(tm), len(gt.texts) - len(tm))
    texts.update({
        "role": _acc(roles["correct"], len(tm)),
        "content": _acc(content["correct"], len(tm)),
        "owner": _acc(owner["correct"], owner["total"]),
        "role_confusion": {k: dict(v) for k, v in sorted(role_conf.items())},
    })

    det = prf(len(agn), len(po) - len(agn), len(go) - len(agn))
    det_cls = prf(len(cls_m), len(po) - len(cls_m), len(go) - len(cls_m))
    return {
        "level": "document",
        "source": gt.source,
        "pred_source": pred.source,
        "system": gt.info.system.value,
        "iou_threshold": iou,
        "counts": {"gt_objects": len(go), "pred_objects": len(po), "gt_pipes": len(gt.pipes),
                   "pred_pipes": len(pred.pipes), "gt_texts": len(gt.texts), "pred_texts": len(pred.texts),
                   "gt_junctions": len(gt.junctions), "pred_junctions": len(pred.junctions)},
        "detection": det,
        "detection_class_aware": det_cls,
        "classification": _acc(cls_ok, len(agn)),
        "per_class": per_class,
        "per_category": per_category,
        "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
        "tag": _acc(tag["correct"], tag["total"], missing=tag["missing"], wrong=tag["wrong"],
                    spurious=tag["spurious"]),
        "actuator": _acc(act["correct"], act["total"]),
        "connections": conn,
        "line_number": _acc(ln["correct"], ln["total"], matched=ln["matched"]),
        "pipe_size": _acc(sz["correct"], sz["total"]),
        "pipes": pipes,
        "texts": texts,
    }


# --------------------------------------------------------------------------- dataset

_PRF_KEYS = ("detection", "detection_class_aware", "connections", "texts")
_ACC_KEYS = (("classification", None), ("tag", None), ("actuator", None), ("line_number", None),
             ("pipe_size", None), ("text_role", ("texts", "role")), ("text_content", ("texts", "content")),
             ("text_owner", ("texts", "owner")))


def _get(d: dict[str, Any], path: Sequence[str]) -> Any:
    for k in path:
        d = d[k]
    return d


def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return float(sum(vals) / len(vals)) if vals else None


def evaluate_dataset(pairs: Iterable[tuple[PIDDocument, PIDDocument]], iou: float = 0.5,
                     **kwargs: Any) -> dict[str, Any]:
    """Evaluate (pred, gt) pairs; micro (summed counts) and macro (mean over documents)
    aggregates, a per-class table and one summary row per document."""
    docs = [evaluate_document(p, g, iou, **kwargs) for p, g in pairs]
    micro: dict[str, Any] = {}
    macro: dict[str, Any] = {}
    for key in _PRF_KEYS:
        tp = sum(d[key]["tp"] for d in docs)
        fp = sum(d[key]["fp"] for d in docs)
        fn = sum(d[key]["fn"] for d in docs)
        micro[key] = prf(tp, fp, fn)
        macro[key] = {m: _mean(d[key][m] for d in docs) for m in ("precision", "recall", "f1")}
    for name, path in _ACC_KEYS:
        path = path or (name,)
        c = sum(_get(d, path)["correct"] for d in docs)
        t = sum(_get(d, path)["total"] for d in docs)
        micro[name] = _acc(c, t)
        macro[name] = {"accuracy": _mean(_get(d, path)["accuracy"] for d in docs)}
    for kind in ("process", "signal"):
        cg = sum(d["pipes"][kind]["gt_covered"] for d in docs)
        tg = sum(d["pipes"][kind]["gt_length"] for d in docs)
        cp = sum(d["pipes"][kind]["pred_covered"] for d in docs)
        tpl = sum(d["pipes"][kind]["pred_length"] for d in docs)
        rec, pre = (cg / tg if tg else None), (cp / tpl if tpl else None)
        f1 = 2 * pre * rec / (pre + rec) if pre and rec else (0.0 if pre is not None and rec is not None else None)
        micro[f"pipes_{kind}"] = {"recall": rec, "precision": pre, "f1": f1}
        macro[f"pipes_{kind}"] = {m: _mean(d["pipes"][kind][m] for d in docs) for m in ("precision", "recall", "f1")}

    per_class: dict[str, dict[str, Any]] = {}
    for table in ("per_class", "per_category"):
        agg: dict[str, Counter] = defaultdict(Counter)
        for d in docs:
            for c, row in d[table].items():
                for k in ("tp", "fp", "fn", "support"):
                    agg[c][k] += row[k]
        rows = {}
        for c in sorted(agg):
            row = prf(agg[c]["tp"], agg[c]["fp"], agg[c]["fn"])
            row["support"] = agg[c]["support"]
            rows[c] = row
        if table == "per_class":
            per_class = rows
        else:
            per_category = rows
    confusion: dict[str, Counter] = defaultdict(Counter)
    for d in docs:
        for g, row in d["confusion"].items():
            confusion[g].update(row)
    macro["class_f1"] = _mean(r["f1"] for r in per_class.values() if r["support"] > 0)
    documents = [{
        "source": d["source"], "system": d["system"], "gt_objects": d["counts"]["gt_objects"],
        "pred_objects": d["counts"]["pred_objects"], "detection_f1": d["detection"]["f1"],
        "class_aware_f1": d["detection_class_aware"]["f1"], "connection_f1": d["connections"]["f1"],
        "tag_accuracy": d["tag"]["accuracy"], "line_number_accuracy": d["line_number"]["accuracy"],
        "text_role_accuracy": d["texts"]["role"]["accuracy"], "pipe_recall": d["pipes"]["process"]["recall"],
    } for d in docs]
    return {"level": "dataset", "iou_threshold": iou, "n_documents": len(docs), "micro": micro, "macro": macro,
            "per_class": per_class, "per_category": per_category,
            "confusion": {k: dict(v) for k, v in sorted(confusion.items())}, "documents": documents}


# --------------------------------------------------------------------------- report


def _f(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _prf_table(rows: dict[str, dict[str, Any]], first: str) -> list[str]:
    out = [f"| {first} | support | TP | FP | FN | precision | recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in rows.items():
        out.append(f"| {name} | {r.get('support', '-')} | {r['tp']} | {r['fp']} | {r['fn']} | {_f(r['precision'])} | "
                   f"{_f(r['recall'])} | {_f(r['f1'])} |")
    return out


def write_eval_report(result: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``evaluation.json`` and ``evaluation.md`` (Markdown tables) into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation.json"
    json_path.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    md = ["# shippid evaluation", ""]
    if result.get("level") == "dataset":
        md += [f"Documents: **{result['n_documents']}**, IoU threshold {result['iou_threshold']}.", "",
               "## Summary", "", "| metric | micro | macro |", "|---|---:|---:|"]
        for key in _PRF_KEYS:
            for m in ("precision", "recall", "f1"):
                md.append(f"| {key} {m} | {_f(result['micro'][key][m])} | {_f(result['macro'][key][m])} |")
        for name, _ in _ACC_KEYS:
            md.append(f"| {name} accuracy | {_f(result['micro'][name]['accuracy'])} | "
                      f"{_f(result['macro'][name]['accuracy'])} |")
        for kind in ("process", "signal"):
            for m in ("precision", "recall", "f1"):
                md.append(f"| pipe length ({kind}) {m} | {_f(result['micro'][f'pipes_{kind}'][m])} | "
                          f"{_f(result['macro'][f'pipes_{kind}'][m])} |")
        md.append(f"| class F1 (macro over classes) | - | {_f(result['macro']['class_f1'])} |")
    else:
        md += [f"Ground truth: `{result.get('source', '')}`, prediction: `{result.get('pred_source', '')}`, "
               f"IoU threshold {result['iou_threshold']}.", "", "## Summary", "",
               "| metric | value |", "|---|---:|"]
        for key in ("detection", "detection_class_aware", "connections", "texts"):
            r = result[key]
            md.append(f"| {key} P / R / F1 | {_f(r['precision'])} / {_f(r['recall'])} / {_f(r['f1'])} |")
        for key, path in _ACC_KEYS:
            r = _get(result, path or (key,))
            md.append(f"| {key} accuracy | {_f(r['accuracy'])} ({r['correct']}/{r['total']}) |")
        for kind in ("process", "signal"):
            r = result["pipes"][kind]
            md.append(f"| pipe length ({kind}) P / R / F1 | {_f(r['precision'])} / {_f(r['recall'])} / {_f(r['f1'])} |")
    md += ["", "## Per class", ""] + _prf_table(result["per_class"], "class")
    md += ["", "## Per category", ""] + _prf_table(result["per_category"], "category")
    if result.get("documents"):
        md += ["", "## Documents", "", "| source | system | GT objects | detection F1 | class-aware F1 | "
                                       "connection F1 | tag acc. | line no. acc. | text role acc. |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for d in result["documents"][:500]:
            md.append(f"| {d['source']} | {d['system']} | {d['gt_objects']} | {_f(d['detection_f1'])} | "
                      f"{_f(d['class_aware_f1'])} | {_f(d['connection_f1'])} | {_f(d['tag_accuracy'])} | "
                      f"{_f(d['line_number_accuracy'])} | {_f(d['text_role_accuracy'])} |")
        if len(result["documents"]) > 500:
            md.append(f"\n({len(result['documents']) - 500} more documents in evaluation.json)")
    md_path = out_dir / "evaluation.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return json_path, md_path
