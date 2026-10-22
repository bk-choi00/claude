"""Gemma 4 31B driven refinement and review of extracted P&IDs.

:class:`GemmaAnalyzer` adds the multimodal model on top of the deterministic
pipeline; every step keeps the deterministic result and records provenance:

* ``refine_symbols``  -- low-confidence / unknown / ambiguous symbols are shown to
  the model as a numbered contact sheet of crops + a JSON digest + the legend;
* ``read_drawing_info`` -- title-block texts (exact PDF text) + a title-block render;
* ``resolve_texts``   -- tags / line numbers / instrument texts with a missing or
  ambiguous owner, decided from candidate owners and distances;
* ``review``          -- engineering review of an :class:`AnalysisReport` (findings
  with rule references, English + Korean summaries); satisfies ``ReportReviewer``.

Each method returns a new object. Model output is validated against the ids
of the document (invalid output triggers a repair round inside
:meth:`LLMClient.generate_json`); an :class:`LLMError` leaves the deterministic
result in place and is logged in ``meta["llm_errors"]``. Results produced via
the test ``MockBackend`` are labelled ``mock``, never ``gemma``.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict

from ..catalog import SYSTEM_BY_CODE, SYSTEMS, detect_system, parse_line_number
from ..llm.base import ImagePart, LLMClient, LLMError, Message
from ..model import (
    ActuatorType,
    AnalysisReport,
    BBox,
    Category,
    DrawingInfo,
    Finding,
    LLMUsage,
    Medium,
    PIDDocument,
    PIDObject,
    PipeKind,
    PipeRun,
    Severity,
    SymbolClass,
    SystemType,
    TextItem,
    TextRole,
    Verdict,
)
from ..symbols import category_of
from . import prompts

if TYPE_CHECKING:
    from ..config import LLMConfig

log = logging.getLogger(__name__)

T = TypeVar("T")
RUNNER_UP_MARGIN = 0.08  # top-2 vector classes closer than this are re-checked
OWNER_TEXT_ROLES = (TextRole.TAG, TextRole.LINE_NUMBER, TextRole.INSTRUMENT)
_HANGUL = re.compile(r"[가-힣]")
_PRIM_KEYS = ("primitives", "primitive_summary", "prims", "signature", "strokes")


# --------------------------------------------------------------------------- output schemas


class _Out(BaseModel):
    # the schema forbids extra keys (guided decoding); validation tolerates them (prompt mode)
    model_config = ConfigDict(extra="ignore", json_schema_extra={"additionalProperties": False})


class SymbolVerdict(_Out):
    id: str
    reason: str = ""
    cls: SymbolClass
    actuator: ActuatorType = ActuatorType.NONE
    confidence: float


class SymbolBatch(_Out):
    items: list[SymbolVerdict]


class DrawingInfoOut(_Out):
    drawing_no: Optional[str] = None
    title: Optional[str] = None
    system: SystemType = SystemType.UNKNOWN
    systems: list[SystemType] = []
    hull_no: Optional[str] = None
    ship_type: Optional[str] = None
    revision: Optional[str] = None
    sheet: Optional[str] = None
    scale: Optional[str] = None
    evidence_text_ids: list[str] = []


class TextAssignment(_Out):
    text_id: str
    reason: str = ""
    owner_id: Optional[str] = None
    role: TextRole


class TextAssignments(_Out):
    assignments: list[TextAssignment]


class ReviewFinding(_Out):
    rule_id: str
    verdict: Verdict
    severity: Severity
    title: str
    description: str
    object_ids: list[str] = []
    reference: Optional[str] = None
    confidence: float = 0.7


class ReviewOut(_Out):
    findings: list[ReviewFinding]
    summary_en: str
    summary_ko: str


# --------------------------------------------------------------------------- helpers


def _clamp01(x: float) -> float:
    return 0.0 if not math.isfinite(x) else min(1.0, max(0.0, float(x)))


def _r(x: float, nd: int = 1) -> float:
    return round(float(x), nd)


def _rect_gap(a: BBox, b: BBox) -> float:
    dx = max(0.0, a.x0 - b.x1, b.x0 - a.x1)
    dy = max(0.0, a.y0 - b.y1, b.y0 - a.y1)
    return math.hypot(dx, dy)


def _seg_point(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
    """Closest point to (px, py) on segment a-b."""
    vx, vy = bx - ax, by - ay
    n2 = vx * vx + vy * vy
    t = 0.0 if n2 <= 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / n2))
    return ax + t * vx, ay + t * vy


def _pipe_gap(box: BBox, pipe: PipeRun) -> tuple[float, tuple[float, float]]:
    """Gap between a text box and a pipe polyline, and the closest pipe point."""
    from shapely.geometry import LineString, Point, box as sbox

    rect = sbox(box.x0, box.y0, box.x1, box.y1)
    pts = [(p.x, p.y) for p in pipe.points]
    if len(pts) == 1:
        geom = Point(pts[0])
    else:
        geom = LineString(pts)
    d = float(rect.distance(geom))
    c = box.center
    best, best_d = pts[0], math.inf
    for a, b in zip(pts, pts[1:] or pts):
        q = _seg_point(c.x, c.y, a[0], a[1], b[0], b[1])
        dd = math.hypot(q[0] - c.x, q[1] - c.y)
        if dd < best_d:
            best, best_d = q, dd
    return d, best


def _direction(dx: float, dy: float, gap: float) -> str:
    if gap <= 0.5:
        return "overlap"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "below" if dy > 0 else "above"


def _vector_candidates(o: PIDObject) -> list[tuple[str, float]]:
    """Distinct (class, score) pairs from ``attributes['candidates']``, best first.

    Accepts a list of dicts ({cls|class|name, score|confidence}), a list of
    (cls, score) pairs or a {cls: score} mapping.
    """
    raw = o.attributes.get("candidates")
    pairs: list[tuple[str, float]] = []
    if isinstance(raw, dict):
        items: Sequence[Any] = list(raw.items())
    elif isinstance(raw, (list, tuple)):
        items = raw
    else:
        return []
    for c in items:
        cls, score = None, None
        if isinstance(c, dict):
            cls = c.get("cls", c.get("class", c.get("name")))
            score = c.get("score", c.get("confidence"))
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            cls, score = c[0], c[1]
        if cls is None or score is None:
            continue
        try:
            pairs.append((SymbolClass(cls).value if not isinstance(cls, SymbolClass) else cls.value, float(score)))
        except (ValueError, TypeError):
            try:
                pairs.append((str(cls), float(score)))
            except (TypeError, ValueError):
                continue
    best: dict[str, float] = {}
    for cls, score in pairs:
        best[cls] = max(score, best.get(cls, -math.inf))
    return sorted(best.items(), key=lambda kv: -kv[1])


def _primitive_summary(o: PIDObject) -> Any:
    for key in _PRIM_KEYS:
        if key in o.attributes:
            value = o.attributes[key]
            text = prompts.to_json(value)
            return value if len(text) <= 300 else text[:300]
    return None


def _usage_delta(before: LLMUsage, after: LLMUsage) -> LLMUsage:
    return LLMUsage(
        backend=after.backend, model=after.model, calls=after.calls - before.calls,
        prompt_tokens=after.prompt_tokens - before.prompt_tokens,
        completion_tokens=after.completion_tokens - before.completion_tokens,
        seconds=round(after.seconds - before.seconds, 3), errors=list(after.errors[len(before.errors):]),
    )


def _usage_add(a: LLMUsage, b: LLMUsage) -> LLMUsage:
    return LLMUsage(
        backend=a.backend if a.backend not in ("", "none") else b.backend, model=a.model or b.model,
        calls=a.calls + b.calls, prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens, seconds=round(a.seconds + b.seconds, 3),
        errors=list(a.errors) + list(b.errors),
    )


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


# list-row keys that cost tokens without helping an engineering review
_DIGEST_DROP = frozenset({"type_ko", "actuator_ko", "type_en", "page", "x", "y", "source", "bbox"})


def _scalarize(row: dict[str, Any], max_len: int = 80, min_conf: float = 0.75) -> dict[str, Any]:
    """Keep the short scalar (and short list) fields of a list row for the review digest."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if v is None or v == "" or v == [] or k in _DIGEST_DROP:
            continue
        if k == "confidence" and isinstance(v, (int, float)) and v >= min_conf:
            continue
        if isinstance(v, (bool, int, float)):
            out[k] = round(v, 3) if isinstance(v, float) else v
        elif isinstance(v, str):
            out[k] = v[:max_len]
        elif hasattr(v, "value") and isinstance(getattr(v, "value"), str):
            out[k] = v.value
        elif isinstance(v, (list, tuple)) and len(v) <= 12 and all(isinstance(x, (str, int, float)) for x in v):
            out[k] = list(v)
    return out


# --------------------------------------------------------------------------- analyzer


class GemmaAnalyzer:
    """Gemma 4 31B analysis steps on top of the deterministic extraction / analysis."""

    def __init__(
        self,
        client: LLMClient,
        *,
        symbol_batch: int = 24,
        min_confidence: float = 0.75,
        review_thinking: bool = True,
        crop_max_side: int = 1536,
        text_batch: int = 40,
        concurrency: int = 1,
        runner_up_margin: float = RUNNER_UP_MARGIN,
        review_max_chars: int = 60000,
    ):
        self.client = client
        self.symbol_batch = max(1, int(symbol_batch))
        self.min_confidence = min_confidence
        self.review_thinking = review_thinking
        self.crop_max_side = crop_max_side
        self.text_batch = max(1, int(text_batch))
        self.concurrency = max(1, int(concurrency))
        self.runner_up_margin = runner_up_margin
        self.review_max_chars = review_max_chars  # digest budget (~17k tokens) for a 32k-context server
        # provenance label: output of the test MockBackend must never pass for Gemma output
        self.label = "mock" if client.backend.name == "mock" else "gemma"

    @property
    def usage(self) -> LLMUsage:
        return self.client.usage.model_copy(deep=True)

    # -- bookkeeping -------------------------------------------------------------------
    def _error(self, doc_meta: dict[str, Any], step: str, exc: BaseException) -> None:
        msg = f"{step}: {exc}"
        log.warning("Gemma %s failed, keeping the deterministic result: %s", step, str(exc)[:300])
        doc_meta.setdefault("llm_errors", []).append(msg[:1000])

    def _record_usage(self, doc: PIDDocument, before: LLMUsage) -> None:
        delta = _usage_delta(before, self.client.usage)
        if delta.calls == 0 and not delta.errors:
            return
        prev = doc.meta.get("llm_usage")
        total = _usage_add(LLMUsage.model_validate(prev), delta) if prev else delta
        doc.meta["llm_usage"] = total.model_dump(mode="json")

    def _gemma_meta(self, doc: PIDDocument) -> dict[str, Any]:
        return doc.meta.setdefault("gemma", {})

    def _run_parallel(self, jobs: Sequence[Callable[[], T]]) -> list[tuple[Optional[T], Optional[LLMError]]]:
        def run(job: Callable[[], T]) -> tuple[Optional[T], Optional[LLMError]]:
            try:
                return job(), None
            except LLMError as exc:
                return None, exc

        if self.concurrency <= 1 or len(jobs) <= 1:
            return [run(j) for j in jobs]
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(jobs))) as pool:
            return list(pool.map(run, jobs))

    @staticmethod
    def _open_pdf(pdf_path: Optional[str | Path]):
        if pdf_path is None:
            return None, "no PDF given"
        from ..llm.render import open_pdf

        try:
            return open_pdf(pdf_path), None
        except Exception as exc:  # pymupdf raises several types (FileNotFoundError, FileDataError, ...)
            return None, f"cannot open {pdf_path}: {exc}"

    # -- symbols -------------------------------------------------------------------------
    def needs_review(self, o: PIDObject) -> bool:
        """Unknown, low-confidence or ambiguous (close runner-up) objects not refined yet."""
        if "gemma_cls" in o.attributes:
            return False
        if o.cls == SymbolClass.UNKNOWN or o.confidence < self.min_confidence:
            return True
        cands = _vector_candidates(o)
        return len(cands) >= 2 and cands[0][1] - cands[1][1] < self.runner_up_margin

    @staticmethod
    def _nearby_texts(doc: PIDDocument, o: PIDObject, limit: int = 4) -> list[dict[str, Any]]:
        reach = max(30.0, 1.5 * max(o.bbox.width, o.bbox.height))
        own = set(o.text_ids)
        found = []
        for t in doc.texts:
            if t.page != o.page or t.role in (TextRole.TITLE_BLOCK, TextRole.LEGEND):
                continue
            gap = _rect_gap(t.bbox, o.bbox)
            if gap <= reach or t.id in own:
                found.append((0.0 if t.id in own else gap, t))
        found.sort(key=lambda x: x[0])
        return [{"text": t.text[:40], "role": t.role.value, "dist_pt": _r(d)} for d, t in found[:limit]]

    def _symbol_digest(self, doc: PIDDocument, o: PIDObject, n: int, pipe_ends: dict[str, int]) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "n": n, "id": o.id, "size_pt": [_r(o.bbox.width), _r(o.bbox.height)], "rotation": o.rotation,
            "vector_class": o.cls.value, "vector_confidence": round(o.confidence, 3),
            "vector_candidates": [{"cls": c, "score": round(s, 3)} for c, s in _vector_candidates(o)[:3]],
        }
        if o.actuator != ActuatorType.NONE:
            entry["vector_actuator"] = o.actuator.value
        prim = _primitive_summary(o)
        if prim is not None:
            entry["primitives"] = prim
        entry["nearby_text"] = self._nearby_texts(doc, o)
        entry["connected_pipes"] = pipe_ends.get(o.id, 0)
        if o.tag:
            entry["tag"] = o.tag
        if isinstance(o.attributes.get("name"), str):
            entry["name"] = o.attributes["name"][:60]
        return entry

    def _merge_symbol(self, o: PIDObject, v: SymbolVerdict) -> str:
        gconf = _clamp01(v.confidence)
        vcls, vconf = o.cls, o.confidence
        attrs = dict(o.attributes)
        attrs.setdefault("vector_cls", vcls.value)
        attrs.setdefault("vector_confidence", vconf)
        attrs.setdefault("vector_actuator", o.actuator.value)
        attrs.update(gemma_cls=v.cls.value, gemma_actuator=v.actuator.value, gemma_confidence=gconf,
                     gemma_reason=v.reason[:300], gemma_source=self.label)
        g_act = v.actuator if category_of(v.cls) == Category.VALVE else ActuatorType.NONE
        if v.cls == vcls:
            decision = "confirmed"
            if gconf > vconf and g_act != o.actuator:
                o.actuator = g_act
            o.confidence = max(vconf, gconf)
            o.source = "merged"
        elif vcls == SymbolClass.UNKNOWN or gconf > vconf:
            decision = "adopted"
            o.cls = v.cls
            o.category = category_of(v.cls)
            o.actuator = g_act
            o.confidence = gconf
            o.source = self.label
        else:
            decision = "kept_vector"
        attrs["refine_decision"] = decision
        o.attributes = attrs
        return decision

    def refine_symbols(self, doc: PIDDocument, pdf_path: Optional[str | Path] = None) -> PIDDocument:
        """Re-classify unknown / low-confidence / ambiguous symbols from crops + vector digest."""
        from ..llm.render import contact_sheet

        out = doc.model_copy(deep=True)
        before = self.client.usage.model_copy(deep=True)
        targets = [o for o in out.objects if self.needs_review(o)]
        stats: dict[str, Any] = {"source": self.label, "candidates": len(targets), "batches": 0,
                                 "failed_batches": 0, "adopted": 0, "confirmed": 0, "kept_vector": 0,
                                 "images": False}
        self._gemma_meta(out)["symbols"] = stats
        if not targets:
            return out
        pdf, pdf_err = self._open_pdf(pdf_path)
        if pdf_err:
            out.meta.setdefault("llm_notes", []).append(f"refine_symbols: {pdf_err}; digest-only prompts")
        stats["images"] = pdf is not None
        pipe_ends: dict[str, int] = defaultdict(int)
        for p in out.pipes:
            for e in (p.start, p.end):
                if e.object_id:
                    pipe_ends[e.object_id] += 1
        by_page: dict[int, list[PIDObject]] = defaultdict(list)
        for o in targets:
            by_page[o.page].append(o)
        system = out.info.system.value if out.info.system != SystemType.UNKNOWN else None
        sys_prompt = prompts.symbol_system_prompt()
        jobs: list[Callable[[], SymbolBatch]] = []
        batches: list[list[PIDObject]] = []
        try:
            for page_no in sorted(by_page):
                objs = by_page[page_no]
                page = pdf[page_no] if pdf is not None and 0 <= page_no < pdf.page_count else None
                for i in range(0, len(objs), self.symbol_batch):
                    batch = objs[i : i + self.symbol_batch]
                    digest = [self._symbol_digest(out, o, n, pipe_ends) for n, o in enumerate(batch, 1)]
                    parts: list[Any] = []
                    if page is not None:
                        crops = []
                        for n, o in enumerate(batch, 1):
                            m = max(6.0, 0.4 * max(o.bbox.width, o.bbox.height))
                            b = o.bbox.expand(m)
                            crops.append((str(n), _pymupdf_rect(b)))
                        sheets = contact_sheet(page, crops, max_side=self.crop_max_side)
                        for k, (png, labels) in enumerate(sheets, 1):
                            parts.append(ImagePart(png, label=f"contact sheet {k}/{len(sheets)}, candidates "
                                                               f"{labels[0]}-{labels[-1]}"))
                    head = {"system": system, "page": page_no, "count": len(batch),
                            "images": "contact sheet above" if parts else "not available - use the digest"}
                    parts.append("Drawing: " + prompts.to_json(head) + "\nCandidates:\n" + prompts.to_json(digest)
                                 + "\nReturn one item for each id: " + ", ".join(o.id for o in batch) + ".")
                    msgs = [Message("system", sys_prompt), Message("user", parts)]
                    ids = [o.id for o in batch]
                    jobs.append(self._symbol_job(msgs, ids))
                    batches.append(batch)
        finally:
            if pdf is not None:
                pdf.close()
        stats["batches"] = len(jobs)
        for batch, (result, err) in zip(batches, self._run_parallel(jobs)):
            if err is not None or result is None:
                stats["failed_batches"] += 1
                self._error(out.meta, f"refine_symbols[{batch[0].id}..{batch[-1].id}]", err or LLMError("no result"))
                continue
            verdicts = {v.id: v for v in result.items}
            for o in batch:
                stats[self._merge_symbol(o, verdicts[o.id])] += 1
        self._record_usage(out, before)
        return out

    def _symbol_job(self, msgs: list[Message], ids: list[str]) -> Callable[[], SymbolBatch]:
        expected = set(ids)

        def validate(res: SymbolBatch) -> Optional[str]:
            got = [it.id for it in res.items]
            problems = []
            unknown = [i for i in got if i not in expected]
            missing = [i for i in ids if i not in got]
            dups = sorted({i for i in got if got.count(i) > 1})
            if unknown:
                problems.append(f"unknown ids {unknown} (use only ids from the digest)")
            if missing:
                problems.append(f"missing ids {missing} (return exactly one item for each)")
            if dups:
                problems.append(f"duplicate ids {dups}")
            return "; ".join(problems) or None

        return lambda: self.client.generate_json(msgs, SymbolBatch, validator=validate)

    # -- drawing info ----------------------------------------------------------------------
    def read_drawing_info(self, doc: PIDDocument, pdf_path: Optional[str | Path] = None) -> PIDDocument:
        """Fill :class:`DrawingInfo` from the title block (texts + render); keyword hint as fallback."""
        from ..llm.render import render_clip

        out = doc.model_copy(deep=True)
        before = self.client.usage.model_copy(deep=True)
        det = out.info.model_copy(deep=True)
        stats: dict[str, Any] = {"source": "vector"}
        self._gemma_meta(out)["drawing_info"] = stats
        tb_all = [t for t in out.texts if t.role == TextRole.TITLE_BLOCK]
        page_no = tb_all[0].page if tb_all else 0
        page_w, page_h = _page_size(out, page_no)
        if not page_w and out.texts:
            page_w = max(t.bbox.x1 for t in out.texts)
            page_h = max(t.bbox.y1 for t in out.texts)
        if tb_all:
            tb = select_title_block([t for t in tb_all if t.page == page_no], page_w, page_h)
        else:  # no title block detected: ISO 7200 puts it bottom-right
            tb = [t for t in out.texts if t.page == page_no and t.bbox.x0 >= 0.55 * page_w and t.bbox.y0 >= 0.7 * page_h
                  and t.role not in OWNER_TEXT_ROLES]
        notes = [t for t in out.texts if t.role == TextRole.NOTE and t.page == page_no][:60]
        tb = tb[:200]
        stats["title_block_texts"] = len(tb)
        hint_text = " ".join([det.title or ""] + [t.text for t in tb] + [t.text for t in notes])
        keyword_system = detect_system(hint_text)
        stats["keyword_system"] = keyword_system.value if keyword_system else None
        region: Optional[BBox] = None
        if tb:
            region = tb[0].bbox
            for t in tb[1:]:
                region = region.union(t.bbox)
            region = region.expand(max(16.0, 0.04 * max(region.width, region.height)))
        elif page_w:
            region = BBox(x0=0.55 * page_w, y0=0.7 * page_h, x1=page_w, y1=page_h)
        image: Optional[bytes] = None
        pdf, pdf_err = self._open_pdf(pdf_path)
        if pdf is not None:
            try:
                if 0 <= page_no < pdf.page_count and region is not None:
                    image = render_clip(pdf[page_no], _pymupdf_rect(region), max_side=self.crop_max_side)
            finally:
                pdf.close()
        stats["image"] = image is not None
        if not tb and image is None:
            self._apply_keyword_system(out, keyword_system)
            out.meta.setdefault("llm_notes", []).append("read_drawing_info: no title-block text or image; skipped")
            return out
        items = [{"id": t.id, "text": t.text, "x": _r(t.bbox.x0), "y": _r(t.bbox.y0), "size": _r(t.size)} for t in tb]
        note_items = [{"id": t.id, "text": t.text[:200]} for t in notes]
        hints = {"keyword_system": stats["keyword_system"],
                 "parsed": det.model_dump(mode="json", exclude={"extra"}, exclude_defaults=True)}
        parts: list[Any] = [ImagePart(image, label="title-block region")] if image else []
        parts.append("Title-block texts:\n" + prompts.to_json(items) + "\nNotes:\n" + prompts.to_json(note_items)
                     + "\nHints:\n" + prompts.to_json(hints))
        msgs = [Message("system", prompts.drawing_info_system_prompt()), Message("user", parts)]
        known_ids = {t.id for t in tb} | {t.id for t in notes}
        corpus = "".join(_norm(t.text) for t in tb + notes)

        def validate(res: DrawingInfoOut) -> Optional[str]:
            problems = []
            bad = [i for i in res.evidence_text_ids if i not in known_ids]
            if bad:
                problems.append(f"evidence_text_ids {bad} are not among the given text ids")
            if corpus:
                for field in ("drawing_no", "hull_no"):
                    value = getattr(res, field)
                    if value and _norm(value) not in corpus:
                        problems.append(f"{field} {value!r} does not appear in the texts; copy it exactly or use null")
            return "; ".join(problems) or None

        try:
            res = self.client.generate_json(msgs, DrawingInfoOut, validator=validate)
        except LLMError as exc:
            self._error(out.meta, "read_drawing_info", exc)
            self._apply_keyword_system(out, keyword_system)
            self._record_usage(out, before)
            return out
        out.info = self._merge_info(det, res, keyword_system)
        stats["source"] = out.info.extra.get("source")
        self._record_usage(out, before)
        return out

    @staticmethod
    def _apply_keyword_system(doc: PIDDocument, keyword_system: Optional[SystemType]) -> None:
        if keyword_system and doc.info.system == SystemType.UNKNOWN:
            doc.info.system = keyword_system
            if keyword_system not in doc.info.systems:
                doc.info.systems = [keyword_system] + list(doc.info.systems)
            extra = dict(doc.info.extra)
            extra["field_sources"] = {**extra.get("field_sources", {}), "system": "keyword"}
            extra.setdefault("source", "vector")
            extra["keyword_system"] = keyword_system.value
            doc.info.extra = extra

    def _merge_info(self, det: DrawingInfo, res: DrawingInfoOut, keyword_system: Optional[SystemType]) -> DrawingInfo:
        fields = ("drawing_no", "title", "hull_no", "ship_type", "revision", "sheet", "scale")
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for f in fields:
            g = getattr(res, f)
            g = g.strip() if isinstance(g, str) else g
            if g:
                values[f], sources[f] = g, self.label
            elif getattr(det, f):
                values[f], sources[f] = getattr(det, f), "vector"
        if res.system != SystemType.UNKNOWN:
            values["system"], sources["system"] = res.system, self.label
        elif det.system != SystemType.UNKNOWN:
            values["system"], sources["system"] = det.system, "vector"
        elif keyword_system:
            values["system"], sources["system"] = keyword_system, "keyword"
        systems: list[SystemType] = []
        for s in [values.get("system", SystemType.UNKNOWN)] + list(res.systems) + list(det.systems):
            if s != SystemType.UNKNOWN and s not in systems:
                systems.append(s)
        used = set(sources.values())
        source = self.label if used and used <= {self.label} else ("merged" if self.label in used else "vector")
        extra = {**det.extra, "source": source, "field_sources": sources,
                 "vector_info": det.model_dump(mode="json", exclude={"extra"}, exclude_defaults=True),
                 "keyword_system": keyword_system.value if keyword_system else None,
                 "evidence_text_ids": list(res.evidence_text_ids)}
        return DrawingInfo(systems=systems, extra=extra, **values)

    # -- text ownership ----------------------------------------------------------------------
    def _owner_candidates(self, doc: PIDDocument, t: TextItem, limit: int = 5) -> list[dict[str, Any]]:
        height = t.size or min(t.bbox.width, t.bbox.height) or 3.0
        radius = max(40.0, 5.0 * height)
        c = t.bbox.center
        found: list[dict[str, Any]] = []
        if t.role == TextRole.LINE_NUMBER:
            for p in doc.pipes:
                if p.page != t.page or p.kind == PipeKind.SIGNAL or not p.points:
                    continue
                pb = BBox.from_points([(q.x, q.y) for q in p.points])
                if _rect_gap(pb, t.bbox) > radius:
                    continue
                gap, near = _pipe_gap(t.bbox, p)
                if gap <= radius:
                    entry = {"id": p.id, "kind": "pipe", "dist_pt": _r(gap),
                             "dir": _direction(near[0] - c.x, near[1] - c.y, gap)}
                    if p.line_number:
                        entry["line_number"] = p.line_number
                    found.append(entry)
        else:
            for o in doc.objects:
                if o.page != t.page:
                    continue
                gap = _rect_gap(t.bbox, o.bbox)
                if gap <= radius:
                    oc = o.bbox.center
                    entry = {"id": o.id, "kind": o.cls.value, "dist_pt": _r(gap),
                             "dir": _direction(oc.x - c.x, oc.y - c.y, gap)}
                    if o.tag:
                        entry["tag"] = o.tag
                    found.append(entry)
        found.sort(key=lambda e: (e["dist_pt"], e["id"]))
        return found[:limit]

    def ambiguous_texts(self, doc: PIDDocument) -> list[tuple[TextItem, list[dict[str, Any]]]]:
        """Owner-bearing texts with no valid owner, or several plausible owners."""
        valid = {o.id for o in doc.objects} | {p.id for p in doc.pipes}
        out = []
        for t in doc.texts:
            if t.role not in OWNER_TEXT_ROLES:
                continue
            cands = self._owner_candidates(doc, t)
            if not cands:
                continue
            if t.owner_id not in valid:
                out.append((t, cands))
                continue
            top_ids = [cnd["id"] for cnd in cands[:2]]
            if len(cands) >= 2:
                d1, d2 = cands[0]["dist_pt"], cands[1]["dist_pt"]
                if d2 <= max(1.5 * d1, d1 + 6.0) or t.owner_id not in top_ids:
                    out.append((t, cands))
            elif t.owner_id not in top_ids:
                out.append((t, cands))
        return out

    def resolve_texts(self, doc: PIDDocument) -> PIDDocument:
        """Let Gemma decide the owner of texts with a missing or ambiguous owner."""
        out = doc.model_copy(deep=True)
        before = self.client.usage.model_copy(deep=True)
        targets = self.ambiguous_texts(out)
        stats: dict[str, Any] = {"source": self.label, "texts": len(targets), "batches": 0, "failed_batches": 0,
                                 "changed": 0, "confirmed": 0}
        self._gemma_meta(out)["texts"] = stats
        if not targets:
            return out
        sys_prompt = prompts.text_owner_system_prompt()
        jobs, batches = [], []
        for i in range(0, len(targets), self.text_batch):
            batch = targets[i : i + self.text_batch]
            digest = [{"text_id": t.id, "text": t.text, "role": t.role.value,
                       "center": [_r(t.bbox.center.x), _r(t.bbox.center.y)], "rotation": t.rotation,
                       "current_owner": t.owner_id, "candidates": cands} for t, cands in batch]
            msgs = [Message("system", sys_prompt),
                    Message("user", "Texts:\n" + prompts.to_json(digest) + "\nReturn one assignment for each "
                            "text_id: " + ", ".join(t.id for t, _ in batch) + ".")]
            jobs.append(self._text_job(msgs, batch))
            batches.append(batch)
        stats["batches"] = len(jobs)
        records = out.meta.setdefault("text_resolution", [])
        for batch, (res, err) in zip(batches, self._run_parallel(jobs)):
            if err is not None or res is None:
                stats["failed_batches"] += 1
                self._error(out.meta, f"resolve_texts[{batch[0][0].id}..{batch[-1][0].id}]", err or LLMError("no result"))
                continue
            for a in res.assignments:
                changed = self._apply_assignment(out, a)
                stats["changed" if changed else "confirmed"] += 1
                records.append({"text_id": a.text_id, "owner_id": a.owner_id, "role": a.role.value,
                                "changed": changed, "reason": a.reason[:200], "source": self.label})
        self._record_usage(out, before)
        return out

    def _text_job(self, msgs: list[Message], batch: list[tuple[TextItem, list[dict[str, Any]]]]) -> Callable[[], TextAssignments]:
        allowed = {t.id: {c["id"] for c in cands} for t, cands in batch}
        order = [t.id for t, _ in batch]

        def validate(res: TextAssignments) -> Optional[str]:
            got = [a.text_id for a in res.assignments]
            problems = []
            unknown = [i for i in got if i not in allowed]
            missing = [i for i in order if i not in got]
            dups = sorted({i for i in got if got.count(i) > 1})
            bad_owner = [f"{a.text_id}->{a.owner_id}" for a in res.assignments
                         if a.text_id in allowed and a.owner_id is not None and a.owner_id not in allowed[a.text_id]]
            if unknown:
                problems.append(f"unknown text_ids {unknown}")
            if missing:
                problems.append(f"missing text_ids {missing} (one assignment each)")
            if dups:
                problems.append(f"duplicate text_ids {dups}")
            if bad_owner:
                problems.append(f"owner_id not among that text's candidates: {bad_owner} (pick a candidate id or null)")
            return "; ".join(problems) or None

        return lambda: self.client.generate_json(msgs, TextAssignments, validator=validate)

    @staticmethod
    def _apply_assignment(doc: PIDDocument, a: TextAssignment) -> bool:
        texts = doc.text_by_id()
        objs = doc.object_by_id()
        pipes = {p.id: p for p in doc.pipes}
        t = texts[a.text_id]
        old, new = t.owner_id, a.owner_id
        changed = old != new or t.role != a.role
        t.role = a.role
        if old == new:
            return changed
        if old in objs:
            o = objs[old]
            o.text_ids = [i for i in o.text_ids if i != t.id]
            if o.tag == t.text:
                o.tag = None
        elif old in pipes:
            p = pipes[old]
            p.text_ids = [i for i in p.text_ids if i != t.id]
            if p.line_number == t.text:
                parsed = parse_line_number(t.text)
                p.line_number = None
                if parsed and p.size == parsed["size"]:
                    p.size = None
        t.owner_id = new
        if new in objs:
            o = objs[new]
            if t.id not in o.text_ids:
                o.text_ids.append(t.id)
            if a.role in (TextRole.TAG, TextRole.INSTRUMENT) and not o.tag:
                o.tag = t.text
        elif new in pipes:
            p = pipes[new]
            if t.id not in p.text_ids:
                p.text_ids.append(t.id)
            if a.role == TextRole.LINE_NUMBER and not p.line_number:
                p.line_number = t.text
                parsed = parse_line_number(t.text)
                if parsed:
                    p.size = p.size or parsed["size"]
                    system = SYSTEM_BY_CODE.get(parsed["system"])
                    if system and p.medium == Medium.UNKNOWN:
                        p.medium = SYSTEMS[system].medium
        return True

    # -- whole document --------------------------------------------------------------------
    def refine_document(self, doc: PIDDocument, pdf_path: Optional[str | Path] = None) -> PIDDocument:
        """read_drawing_info + refine_symbols + resolve_texts; an LLMError in a step keeps its input."""
        out = doc
        steps: list[tuple[str, Callable[[PIDDocument], PIDDocument]]] = [
            ("read_drawing_info", lambda d: self.read_drawing_info(d, pdf_path)),
            ("refine_symbols", lambda d: self.refine_symbols(d, pdf_path)),
            ("resolve_texts", self.resolve_texts),
        ]
        for name, step in steps:
            try:
                out = step(out)
            except LLMError as exc:
                out = out.model_copy(deep=True) if out is doc else out
                self._error(out.meta, name, exc)
        if out is doc:
            out = doc.model_copy(deep=True)
        out.meta["refined_by"] = {"label": self.label, "backend": self.client.backend.name,
                                  "model": self.client.backend.model}
        return out

    # -- review ------------------------------------------------------------------------------
    def review_digest(self, report: AnalysisReport, max_items: int = 300) -> dict[str, Any]:
        """Compact JSON digest of a report for the review prompt."""
        doc = report.document
        truncated: dict[str, int] = {}

        def cap(name: str, rows: list[Any], limit: int) -> list[Any]:
            if len(rows) > limit:
                truncated[name] = len(rows)
            return rows[:limit]

        object_ids = {o.id for o in doc.objects}
        names: dict[str, list[str]] = defaultdict(list)
        for t in doc.texts:
            if t.owner_id and t.role in (TextRole.LABEL, TextRole.CONNECTOR):
                names[t.owner_id].append(t.text[:60])

        def obj_row(o: PIDObject) -> dict[str, Any]:
            row: dict[str, Any] = {"id": o.id, "cls": o.cls.value}
            if o.tag:
                row["tag"] = o.tag
            if o.category == Category.VALVE and o.actuator != ActuatorType.NONE:
                row["actuator"] = o.actuator.value
            name = o.attributes.get("name") if isinstance(o.attributes.get("name"), str) else None
            if name or names.get(o.id):
                row["name"] = (name or " / ".join(names[o.id][:2]))[:80]
            if o.confidence < self.min_confidence:
                row["conf"] = round(o.confidence, 2)
            return row

        def rows_or(name: str, given: list[dict[str, Any]], derived: Callable[[], list[dict[str, Any]]]) -> list[Any]:
            rows = [_scalarize(r, min_conf=self.min_confidence) for r in given] if given else derived()
            return cap(name, rows, max_items)

        cats = {c: [o for o in doc.objects if o.category == c] for c in Category}
        lines: dict[str, dict[str, Any]] = {}
        no_line = 0
        for p in doc.pipes:
            if p.kind == PipeKind.SIGNAL:
                continue
            if not p.line_number:
                no_line += 1
                continue
            row = lines.setdefault(p.line_number, {"line_number": p.line_number, "size": p.size,
                                                   "medium": p.medium.value, "pipes": []})
            row["pipes"].append(p.id)
        digest: dict[str, Any] = {
            "drawing": doc.info.model_dump(mode="json", exclude={"extra"}, exclude_defaults=True),
            "counts": report.counts or {c.value: len(v) for c, v in cats.items()},
            "equipment": rows_or("equipment", report.equipment_list,
                                 lambda: [obj_row(o) for o in cats[Category.EQUIPMENT]]),
            "valves": rows_or("valves", report.valve_list, lambda: [obj_row(o) for o in cats[Category.VALVE]]),
            "instruments": rows_or("instruments", report.instrument_list,
                                   lambda: [obj_row(o) for o in cats[Category.INSTRUMENT]]),
            "fittings_connectors": cap("fittings_connectors", [obj_row(o) for o in cats[Category.FITTING] +
                                                               cats[Category.CONNECTOR]], max_items),
            "lines": rows_or("lines", report.line_list, lambda: list(lines.values())),
            "process_pipes_without_line_number": no_line,
            "adjacency": cap("adjacency", [{"id": k, "to": v} for k, v in _adjacency(doc).items()], max_items),
            "flow_paths": cap("flow_paths", [
                {"from": fp.source_id, "to": fp.target_id,
                 "via": [n for n in fp.node_ids if n in object_ids][:25],
                 **({"desc": fp.description[:120]} if fp.description else {})} for fp in report.flow_paths],
                min(40, max_items)),
            "rule_checks": cap("rule_checks", [
                {"rule_id": f.rule_id, "verdict": f.verdict.value, "severity": f.severity.value, "title": f.title[:160],
                 "object_ids": f.object_ids[:12], "reference": f.reference}
                for f in report.findings if f.source not in ("gemma", "mock")], min(120, max_items)),
        }
        if truncated:
            digest["truncated_lists_total_lengths"] = truncated
        return digest

    def review(self, report: AnalysisReport) -> AnalysisReport:
        """Engineering review: append Gemma findings, set summaries, account LLM usage."""
        out = report.model_copy(deep=True)
        doc = out.document
        if doc.meta.get("llm_usage") and not out.meta.get("llm_refine_usage_included"):
            out.llm = _usage_add(out.llm, LLMUsage.model_validate(doc.meta["llm_usage"]))
            out.meta["llm_refine_usage_included"] = True
        before = self.client.usage.model_copy(deep=True)
        valid = {o.id for o in doc.objects} | {p.id for p in doc.pipes} | {j.id for j in doc.junctions}
        digest_text = prompts.to_json(self.review_digest(out))
        for max_items in (150, 80, 40, 20):  # shrink the lists until the digest fits the budget
            if len(digest_text) <= self.review_max_chars:
                break
            digest_text = prompts.to_json(self.review_digest(out, max_items=max_items))
        msgs = [Message("system", prompts.review_system_prompt()),
                Message("user", "P&ID digest:\n" + digest_text)]

        def validate(res: ReviewOut) -> Optional[str]:
            problems = []
            bad = sorted({i for f in res.findings for i in f.object_ids if i not in valid})
            if bad:
                problems.append(f"object_ids {bad[:20]} do not exist in the digest; use only listed ids")
            if not res.summary_en.strip():
                problems.append("summary_en is empty")
            if not _HANGUL.search(res.summary_ko):
                problems.append("summary_ko must be written in Korean (Hangul)")
            if any(not f.rule_id.strip() for f in res.findings):
                problems.append("every finding needs a rule_id")
            return "; ".join(problems) or None

        try:
            res = self.client.generate_json(msgs, ReviewOut, validator=validate,
                                            thinking=True if self.review_thinking else None)
        except LLMError as exc:
            self._error(out.meta, "review", exc)
            out.meta["review"] = {"source": self.label, "status": "error", "digest_chars": len(digest_text)}
            out.llm = _usage_add(out.llm, _usage_delta(before, self.client.usage))
            if f"review: {exc}"[:1000] not in out.llm.errors:
                out.llm.errors.append(f"review: {exc}"[:1000])
            return out
        prefix = f"{self.label.upper()}-"
        taken = {f.id for f in out.findings}
        n = 0
        for f in res.findings:
            n += 1
            fid = f"{self.label}-{n:03d}"
            while fid in taken:
                n += 1
                fid = f"{self.label}-{n:03d}"
            taken.add(fid)
            rule = re.sub(r"\s+", "-", f.rule_id.strip().upper())
            rule = re.sub(r"^(GEMMA|MOCK)-", "", rule)
            ids = list(dict.fromkeys(f.object_ids))
            out.findings.append(Finding(
                id=fid, rule_id=prefix + rule, verdict=f.verdict, severity=f.severity, title=f.title.strip(),
                description=f.description.strip(), object_ids=ids, reference=(f.reference or None),
                source=self.label, confidence=_clamp01(f.confidence)))
        if (out.summary_ko or out.summary_en) and out.meta.get("summary_source") not in ("gemma", "mock"):
            out.meta["summary_deterministic"] = {"ko": out.summary_ko, "en": out.summary_en}
        out.summary_ko = res.summary_ko.strip()
        out.summary_en = res.summary_en.strip()
        out.meta["summary_source"] = self.label
        out.meta["review"] = {"source": self.label, "status": "ok", "findings": len(res.findings),
                              "thinking": self.review_thinking, "digest_chars": len(digest_text)}
        out.llm = _usage_add(out.llm, _usage_delta(before, self.client.usage))
        return out


def make_analyzer(cfg: "LLMConfig", responder: Optional[Callable[..., Any]] = None, **kwargs: Any) -> Optional[GemmaAnalyzer]:
    """GemmaAnalyzer for an :class:`LLMConfig` (``None`` when the LLM is disabled).

    Wires ``max_image_side`` -> ``crop_max_side`` and ``concurrency``; extra keyword
    arguments go to :class:`GemmaAnalyzer`. ``responder`` is only for the ``mock`` backend.
    """
    from ..llm.factory import make_client

    client = make_client(cfg, responder=responder)
    if client is None:
        return None
    kwargs.setdefault("crop_max_side", cfg.max_image_side)
    kwargs.setdefault("concurrency", cfg.concurrency)
    return GemmaAnalyzer(client, **kwargs)


# --------------------------------------------------------------------------- module helpers

_TB_KEYWORDS = re.compile(r"\b(DWG|DRAWING|DRAWN|HULL|TITLE|SCALE|REV|REVISION|SHEET|DATE|APPROVED|CHECKED|"
                          r"DESIGNED|SHIP|PROJECT|OWNER|YARD|CLASS)\b|NO\.", re.I)


def select_title_block(texts: Sequence[TextItem], page_w: float, page_h: float) -> list[TextItem]:
    """The title-block texts among ``texts`` (all labelled title block by the extractor).

    Extractors often label the whole side panel (legend, notes, revision table)
    and the frame's zone markers as title block. Texts are clustered by gap
    (single link, 3x the median text height); the cluster with the most
    title-block keywords nearest the bottom-right corner (ISO 7200) wins,
    adjacent multi-text clusters (revision table, notes) are added, then every
    text lying inside the resulting region. Zone markers sit outside the frame
    and never join, so the region stays compact.
    """
    texts = list(texts)
    if len(texts) <= 2 or not page_w or not page_h:
        return texts
    sizes = sorted(t.size or t.bbox.height for t in texts)
    link = max(12.0, 3.0 * sizes[len(sizes) // 2])
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if _rect_gap(texts[i].bbox, texts[j].bbox) <= link:
                parent[find(i)] = find(j)
    groups: dict[int, list[TextItem]] = defaultdict(list)
    for i, t in enumerate(texts):
        groups[find(i)].append(t)
    clusters = list(groups.values())

    def bbox(ts: Sequence[TextItem]) -> BBox:
        b = ts[0].bbox
        for t in ts[1:]:
            b = b.union(t.bbox)
        return b

    def score(c: list[TextItem]) -> float:
        b = bbox(c)
        keywords = sum(1 for t in c if _TB_KEYWORDS.search(t.text))
        return 3.0 * keywords + 5.0 * (b.x1 / page_w + b.y1 / page_h) + 0.2 * min(len(c), 20)

    best = max(range(len(clusters)), key=lambda k: score(clusters[k]))
    used, region = {best}, bbox(clusters[best])
    grown = True
    while grown:
        grown = False
        for k, c in enumerate(clusters):
            if k in used or len(c) < 2:
                continue
            if _rect_gap(bbox(c), region) <= 2.5 * link:
                used.add(k)
                region = region.union(bbox(c))
                grown = True
    # isolated values inside the block (e.g. a hull number centred in a wide cell) belong to it too
    inner = region.expand(0.5 * link)
    chosen = [t for k, c in enumerate(clusters) for t in c
              if k in used or inner.contains_point(t.bbox.center.x, t.bbox.center.y)]
    return sorted(chosen, key=lambda t: (round(t.bbox.y0, 1), t.bbox.x0))



def _pymupdf_rect(b: BBox):
    import pymupdf

    return pymupdf.Rect(b.x0, b.y0, b.x1, b.y1)


def _page_size(doc: PIDDocument, page_no: int) -> tuple[float, float]:
    for p in doc.pages:
        if p.index == page_no:
            return p.width, p.height
    return 0.0, 0.0


def _adjacency(doc: PIDDocument) -> dict[str, list[str]]:
    """Object -> objects reachable through pipes, passing through junctions (signal lines excluded)."""
    objects = {o.id for o in doc.objects}
    graph: dict[str, set[str]] = defaultdict(set)
    for p in doc.pipes:
        if p.kind == PipeKind.SIGNAL:
            continue
        a, b = p.start.node_id, p.end.node_id
        if a and b and a != b:
            graph[a].add(b)
            graph[b].add(a)
    adj: dict[str, list[str]] = {}
    for oid in sorted(objects, key=_id_key):
        seen, stack, found = {oid}, list(graph.get(oid, ())), set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n in objects:
                found.add(n)
            else:
                stack.extend(graph.get(n, ()))
        if found:
            adj[oid] = sorted(found, key=_id_key)
    return adj


def _id_key(i: str) -> tuple[str, int]:
    m = re.match(r"([a-z]+)(\d+)$", i)
    return (m.group(1), int(m.group(2))) if m else (i, 0)
