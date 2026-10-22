"""Gemma 4 31B driven refinement and review of extracted P&IDs.

:class:`GemmaAnalyzer` adds the multimodal model on top of the deterministic
pipeline; every step keeps the deterministic result and records provenance:

* ``refine_symbols``  -- low-confidence / unknown / ambiguous symbols are shown to
  the model as a numbered contact sheet of crops + a JSON digest + the legend;
  a refined instrument / valve / equipment without a tag gets one from its bubble
  texts, from Gemma's tag (only if built from nearby texts) or a free tag text;
* ``read_drawing_info`` -- title-block texts (exact PDF text) + a title-block render;
* ``resolve_texts``   -- tags / line numbers / instrument texts with a missing or
  ambiguous owner, decided from candidate owners and distances;
* ``resolve_sheet_links`` -- multi-sheet sets: off-page connectors the deterministic
  linker (:mod:`shippid.analysis.sheets`) left ambiguous, decided among their candidates;
* ``review``          -- engineering review of an :class:`AnalysisReport` (findings
  with rule references, English + Korean summaries); satisfies ``ReportReviewer``.
  The request is sized in tokens to the server context (``context_tokens``) and the
  summaries may not claim compliance while deterministic rule checks fail.

Drawing text is untrusted data: every prompt says so, and model output that could
change deterministic verdicts is grounded first -- a drawing system is accepted only
with title-block evidence it cites (or line-number codes / the vector result), the
vector primary system is never replaced without title-block evidence (a disagreement
is recorded instead), and title-block fields must be printed in the title block.

Each method returns a new object. Model output is validated against the ids
of the document (invalid output triggers a repair round inside
:meth:`LLMClient.generate_json`); an :class:`LLMError` in a refinement step
leaves the deterministic result in place, is logged in ``meta["llm_errors"]``
and sets the step's status (``refined_by`` only names steps that got a valid
answer); a failed review sets ``meta["review"]["status"] = "error"``, which
``analyze()`` reports. Results produced via the test ``MockBackend`` are
labelled ``mock``, never ``gemma``.
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

from ..catalog import SYSTEM_BY_CODE, SYSTEMS, classify_tag, detect_system, parse_line_number
from ..llm.base import ImagePart, LLMClient, LLMError, Message, schema_for
from ..llm.budget import context_of, messages_tokens, repair_reserve
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
ADOPT_MIN_CONFIDENCE = 0.5  # Gemma classes below this are recorded, never adopted
OWNER_TEXT_ROLES = (TextRole.TAG, TextRole.LINE_NUMBER, TextRole.INSTRUMENT)
_FIXED_ROLES = (TextRole.TITLE_BLOCK, TextRole.NOTE, TextRole.LEGEND)  # texts that never get an owner
_TAGGABLE = (Category.INSTRUMENT, Category.VALVE, Category.EQUIPMENT)
_HANGUL = re.compile(r"[가-힣]")
_PRIM_KEYS = ("primitives", "primitive_summary", "prims", "signature", "strokes")
_TAG_SPLIT = re.compile(r"[-\s]+")
_INSTR_LETTERS = re.compile(r"^[A-Z]{1,5}$")
_INSTR_NUMBER = re.compile(r"^\d{1,5}[A-Z]?$")
_INSTR_JOINED = re.compile(r"^([A-Z]{1,5})(\d{1,5}[A-Z]?)$")
_TAG_KIND_FOR = {Category.VALVE: "valve", Category.EQUIPMENT: "equipment", Category.INSTRUMENT: "instrument"}
REVIEW_MIN_ANSWER_TOKENS = 2500  # findings + both summaries of a review answer
REVIEW_THINKING_TOKENS = 1500  # minimum room for thinking before the review JSON
_INFO_TEXT_FIELDS = ("title", "ship_type", "revision", "sheet", "scale")  # grounded in the title-block texts
_TOKEN_EDGE = ".,;:()[]{}\"'"


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
    tag: Optional[str] = None  # must be built from the candidate's nearby_text (checked on merge)


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


class SheetLinkDecision(_Out):
    connector: str
    reason: str = ""
    partner: Optional[str] = None
    confidence: float = 0.5


class SheetLinkDecisions(_Out):
    decisions: list[SheetLinkDecision]


# --------------------------------------------------------------------------- helpers


def _clamp01(x: float) -> float:
    return 0.0 if not math.isfinite(x) else min(1.0, max(0.0, float(x)))


def _r(x: float, nd: int = 1) -> float:
    return round(float(x), nd)


def _rect_gap(a: BBox, b: BBox) -> float:
    dx = max(0.0, a.x0 - b.x1, b.x0 - a.x1)
    dy = max(0.0, a.y0 - b.y1, b.y0 - a.y1)
    return math.hypot(dx, dy)


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


def normalize_text(s: str) -> str:
    """Upper case with single spaces (the extractor's tag / line-number normalisation)."""
    return re.sub(r"\s+", " ", s.strip().upper())


def tag_tokens(s: str) -> list[str]:
    """Tag tokens: upper case, split at hyphens and whitespace ('PI-101' -> ['PI', '101'])."""
    return [tok for tok in _TAG_SPLIT.split(s.strip().upper()) if tok]


def compose_instrument_tag(texts: Sequence[TextItem]) -> Optional[str]:
    """Letters + loop number written inside a bubble -> 'PI-101' (same rule as the extractor)."""
    tokens: list[str] = []
    for t in sorted(texts, key=lambda t: (t.bbox.y0, t.bbox.x0)):
        tokens.extend(tag_tokens(t.text))
    letters = next((tok for tok in tokens if _INSTR_LETTERS.match(tok)), None)
    number = next((tok for tok in tokens if _INSTR_NUMBER.match(tok)), None)
    if letters is None and len(tokens) == 1 and (m := _INSTR_JOINED.match(tokens[0])):
        letters, number = m.group(1), m.group(2)
    if letters and number:
        return f"{letters}-{number}"
    return " ".join(tokens) or None


def _is_instrument_tag(tag: Optional[str]) -> bool:
    return bool(tag) and re.fullmatch(r"[A-Z]{1,5}-\d{1,5}[A-Z]?", tag or "") is not None


def _covers(tag: str, text: str) -> bool:
    """Every token of ``text`` occurs in ``tag`` (the text is (part of) that tag)."""
    xs = tag_tokens(text)
    return bool(xs) and set(xs) <= set(tag_tokens(tag))


def _cover(want: set[str], texts: Sequence[TextItem]) -> list[TextItem]:
    """Texts (in the given nearest-first order) whose tokens make up exactly ``want``; [] if impossible.

    A single text carrying the whole tag wins; otherwise texts are added while
    they contribute new tokens (a tag split over bubble letters and loop number).
    """
    exact = [t for t in texts if set(tag_tokens(t.text)) == want]
    if exact:
        return exact[:1]
    picked: list[TextItem] = []
    covered: set[str] = set()
    for t in texts:
        toks = set(tag_tokens(t.text))
        if toks <= want and not toks <= covered:
            picked.append(t)
            covered |= toks
    return picked if covered == want else []


def _tag_like(text: str) -> bool:
    """A self-contained tag has letters and digits ('BA-V001', 'CF015V'); '601' or 'PI' alone are fragments."""
    t = normalize_text(text)
    return bool(re.search(r"[A-Z]", t)) and bool(re.search(r"\d", t))


def _tag_from_texts(o: PIDObject, texts: dict[str, TextItem]) -> Optional[str]:
    """An object's tag rebuilt from the tag texts it owns, or None when only fragments remain.

    Bubble texts are composed like the extractor ('TIC' + '601' -> 'TIC-601') and
    count only when letters and loop number are both there; otherwise the
    nearest self-contained TAG text is used.
    """
    owned = [texts[i] for i in o.text_ids if i in texts]
    bubble = [t for t in owned if t.role == TextRole.INSTRUMENT]
    composed = compose_instrument_tag(bubble) if bubble else None
    if _is_instrument_tag(composed):
        return composed
    tags = [t for t in owned if t.role == TextRole.TAG and _tag_like(t.text)]
    if tags:
        return normalize_text(min(tags, key=lambda t: _rect_gap(t.bbox, o.bbox)).text)
    return None


class _SpatialIndex:
    """STRtree over the objects and process pipes of each page (built once per document)."""

    def __init__(self, doc: PIDDocument):
        from shapely import STRtree
        from shapely.geometry import LineString, Point, box as sbox

        self._box = sbox
        self.objects: dict[int, list[PIDObject]] = defaultdict(list)
        self.pipes: dict[int, list[tuple[PipeRun, Any]]] = defaultdict(list)
        for o in doc.objects:
            self.objects[o.page].append(o)
        for p in doc.pipes:
            if p.kind == PipeKind.SIGNAL or not p.points:
                continue
            pts = [(q.x, q.y) for q in p.points]
            self.pipes[p.page].append((p, Point(pts[0]) if len(pts) == 1 else LineString(pts)))
        self._otree = {pg: STRtree([sbox(o.bbox.x0, o.bbox.y0, o.bbox.x1, o.bbox.y1) for o in objs])
                       for pg, objs in self.objects.items()}
        self._ptree = {pg: STRtree([g for _, g in items]) for pg, items in self.pipes.items()}

    def _window(self, b: BBox, radius: float) -> Any:
        return self._box(b.x0 - radius, b.y0 - radius, b.x1 + radius, b.y1 + radius)

    def objects_near(self, page: int, b: BBox, radius: float) -> list[tuple[PIDObject, float]]:
        tree = self._otree.get(page)
        if tree is None:
            return []
        objs = self.objects[page]
        found = []
        for k in tree.query(self._window(b, radius)).tolist():
            gap = _rect_gap(b, objs[k].bbox)
            if gap <= radius:
                found.append((objs[k], gap))
        return found

    def pipes_near(self, page: int, b: BBox, radius: float) -> list[tuple[PipeRun, float, tuple[float, float]]]:
        """(pipe, gap between box and polyline, polyline point closest to the box centre)."""
        from shapely.geometry import Point

        tree = self._ptree.get(page)
        if tree is None:
            return []
        items = self.pipes[page]
        rect = self._box(b.x0, b.y0, b.x1, b.y1)
        c = Point(b.center.x, b.center.y)
        found = []
        for k in tree.query(self._window(b, radius)).tolist():
            p, geom = items[k]
            gap = float(rect.distance(geom))
            if gap <= radius:
                q = geom if geom.geom_type == "Point" else geom.interpolate(geom.project(c))
                found.append((p, gap, (q.x, q.y)))
        return found


# list-row keys that cost tokens without helping an engineering review
_DIGEST_DROP = frozenset({"type_ko", "actuator_ko", "type_en", "page", "x", "y", "source", "bbox"})
_JOIN_SEP = "; "  # analysis.lists joins neighbour / valve / line-number lists with this separator


def _cap_list(items: Sequence[Any], max_items: int) -> list[Any]:
    items = list(items)
    if len(items) <= max_items:
        return items
    return items[:max_items] + [f"…(+{len(items) - max_items} more)"]


def _scalarize(row: dict[str, Any], max_len: int = 240, min_conf: float = 0.75, max_list: int = 40,
               keep: Sequence[str] = ()) -> dict[str, Any]:
    """The scalar and list fields of a list row for the review digest.

    Joined lists (``"a; b; c"`` and space-separated ``*_ids``) are split back into
    lists; long lists and long free text are cut with an explicit ``…(+n …)``
    marker, never silently (the digest-size guard limits the number of rows).
    ``keep`` names normally dropped keys to keep (``page`` for multi-sheet sets).
    """
    out: dict[str, Any] = {}
    for k, v in row.items():
        if v is None or v == "" or v == [] or (k in _DIGEST_DROP and k not in keep):
            continue
        if k == "confidence" and isinstance(v, (int, float)) and v >= min_conf:
            continue
        if isinstance(v, (bool, int, float)):
            out[k] = round(v, 3) if isinstance(v, float) else v
        elif isinstance(v, str):
            if k.endswith("_ids"):
                out[k] = _cap_list(v.split(), max_list)
            elif _JOIN_SEP in v:
                out[k] = _cap_list([x.strip() for x in v.split(_JOIN_SEP) if x.strip()], max_list)
            elif len(v) > max_len:
                out[k] = v[:max_len] + f"…(+{len(v) - max_len} chars)"
            else:
                out[k] = v
        elif hasattr(v, "value") and isinstance(getattr(v, "value"), str):
            out[k] = v.value
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float)) for x in v):
            out[k] = _cap_list(v, max_list)
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
        adopt_min_confidence: float = ADOPT_MIN_CONFIDENCE,
        context_tokens: Any = "auto",
        review_max_tokens: Optional[int] = None,
        sheet_link_batch: int = 30,
    ):
        """``context_tokens`` is the server context (prompt + max_tokens, vLLM ``--max-model-len``):
        ``"auto"`` = the client's (``LLMConfig.context_tokens``), else 32768; None = unlimited. The
        review digest is shrunk and its ``max_tokens`` (``review_max_tokens``, default the client's)
        capped so the request fits it; ``review_max_chars`` caps the digest size in addition."""
        self.client = client
        self.symbol_batch = max(1, int(symbol_batch))
        self.min_confidence = min_confidence
        self.adopt_min_confidence = adopt_min_confidence
        self.review_thinking = review_thinking
        self.crop_max_side = crop_max_side
        self.text_batch = max(1, int(text_batch))
        self.concurrency = max(1, int(concurrency))
        self.runner_up_margin = runner_up_margin
        self.review_max_chars = review_max_chars  # digest size cap (characters)
        self.context_tokens = context_of(client) if context_tokens == "auto" else context_tokens
        self.review_max_tokens = review_max_tokens
        self.sheet_link_batch = max(1, int(sheet_link_batch))
        # provenance label: output of the test MockBackend must never pass for Gemma output
        self.label = "mock" if client.backend.name == "mock" else "gemma"

    @property
    def usage(self) -> LLMUsage:
        return self.client.usage.model_copy(deep=True)

    # -- bookkeeping -------------------------------------------------------------------
    def _redact(self, text: Any) -> str:
        """Defence in depth: backends already keep credentials out of their error texts."""
        redact = getattr(self.client.backend, "redact", None)
        return redact(str(text)) if callable(redact) else str(text)

    def _error(self, doc_meta: dict[str, Any], step: str, exc: BaseException) -> None:
        msg = self._redact(f"{step}: {exc}")
        log.warning("Gemma %s failed, keeping the deterministic result: %s", step, self._redact(exc)[:300])
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
    def _nearby_texts(doc: PIDDocument, o: PIDObject, limit: int = 6) -> list[tuple[float, TextItem]]:
        """(gap, text) of the texts owned by or near ``o``, nearest first (what the digest shows)."""
        reach = max(30.0, 1.5 * max(o.bbox.width, o.bbox.height))
        own = set(o.text_ids)
        found = []
        for t in doc.texts:
            if t.page != o.page or t.role in (TextRole.TITLE_BLOCK, TextRole.LEGEND):
                continue
            gap = _rect_gap(t.bbox, o.bbox)
            if gap <= reach or t.id in own:
                found.append((0.0 if t.id in own else gap, t))
        found.sort(key=lambda x: (x[0], x[1].id))
        return found[:limit]

    def _symbol_digest(self, doc: PIDDocument, o: PIDObject, n: int, pipe_ends: dict[str, int],
                       nearby: list[tuple[float, TextItem]]) -> dict[str, Any]:
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
        entry["nearby_text"] = [{"text": t.text[:40], "role": t.role.value, "dist_pt": _r(d),
                                 **({"inside": True} if o.bbox.contains_point(t.bbox.center.x, t.bbox.center.y) else {})}
                                for d, t in nearby]
        entry["connected_pipes"] = pipe_ends.get(o.id, 0)
        if o.tag:
            entry["tag"] = o.tag
        if isinstance(o.attributes.get("name"), str):
            entry["name"] = o.attributes["name"][:60]
        entry.update(_extractor_hints(o))
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
        elif (vcls == SymbolClass.UNKNOWN or gconf > vconf) and gconf >= self.adopt_min_confidence:
            decision = "adopted"
            o.cls = v.cls
            o.category = category_of(v.cls)
            o.actuator = g_act
            o.confidence = gconf
            o.source = self.label
        else:
            decision = "kept_vector"
            if gconf < self.adopt_min_confidence:
                attrs["refine_note"] = f"{self.label} confidence {gconf:.2f} < {self.adopt_min_confidence:.2f}: not adopted"
        attrs["refine_decision"] = decision
        o.attributes = attrs
        return decision

    # -- tags of refined symbols -------------------------------------------------------------
    def _claim_tag(self, doc: PIDDocument, o: PIDObject, v: SymbolVerdict, nearby: list[tuple[float, TextItem]],
                   maps: tuple[dict[str, TextItem], dict[str, PIDObject], dict[str, PipeRun]]) -> Optional[dict[str, Any]]:
        """Give a refined, untagged instrument / valve / equipment its tag.

        In order: the texts inside an instrument bubble (composed like the
        extractor: 'PI' + '601' -> 'PI-601'); Gemma's ``tag`` when it is built
        from the candidate's nearby texts (otherwise recorded as rejected); the
        nearest unowned tag text of the matching kind for which ``o`` is the
        nearest object of its category. The texts used are moved to ``o``
        (previous owners' tags are rebuilt). Returns a change record or None.
        """
        texts, objs, _ = maps
        if v.tag:
            o.attributes["gemma_tag"] = v.tag[:60]
        if o.tag or o.category not in _TAGGABLE:
            return None
        pool = [t for _, t in nearby if t.role not in _FIXED_ROLES and t.role != TextRole.LINE_NUMBER]

        def free(t: TextItem) -> bool:  # not the tag source of another object
            other = objs.get(t.owner_id or "")
            if other is None or other.id == o.id:
                return True
            if other.category == Category.INSTRUMENT and t.role == TextRole.INSTRUMENT:
                return False
            return not (other.tag and _covers(other.tag, t.text))

        chosen: list[TextItem] = []
        tag, source, role = None, None, TextRole.TAG
        if o.category == Category.INSTRUMENT:
            inside = [t for t in pool if o.bbox.contains_point(t.bbox.center.x, t.bbox.center.y)
                      and t.role != TextRole.CONNECTOR and free(t)]
            composed = compose_instrument_tag(inside) if inside else None
            if _is_instrument_tag(composed):
                tag, source, role, chosen = composed, "bubble_text", TextRole.INSTRUMENT, inside
        want = set(tag_tokens(v.tag or ""))
        if tag is None and want:
            parts = [t for t in pool if tag_tokens(t.text) and set(tag_tokens(t.text)) <= want]
            picked = _cover(want, [t for t in parts if free(t)])
            if picked:
                tag, source, chosen = normalize_text(v.tag or ""), self.label, picked
                if o.category == Category.INSTRUMENT and all(
                        o.bbox.contains_point(t.bbox.center.x, t.bbox.center.y) for t in picked):
                    role = TextRole.INSTRUMENT
                    composed = compose_instrument_tag(picked)
                    tag = composed if _is_instrument_tag(composed) else tag
            else:
                o.attributes["gemma_tag_rejected"] = ("its texts already tag another object" if _cover(want, parts)
                                                      else "not built from the symbol's nearby texts")
        if tag is None:
            kind = _TAG_KIND_FOR[o.category]
            for t in pool:
                if t.role != TextRole.TAG or t.owner_id in objs or classify_tag(normalize_text(t.text)) != kind:
                    continue
                rivals = [p for p in objs.values() if p.id != o.id and p.page == o.page and p.category == o.category
                          and not p.tag and _rect_gap(t.bbox, p.bbox) < _rect_gap(t.bbox, o.bbox)]
                if not rivals:
                    tag, source, chosen = normalize_text(t.text), "nearby_text", [t]
                    break
        if tag is None:
            return None
        for t in chosen:
            self._apply_assignment(doc, TextAssignment(text_id=t.id, owner_id=o.id, role=role,
                                                       reason=f"tag of refined {o.cls.value}"), maps)
        o.tag = tag
        o.attributes["tag_source"] = source
        name = o.attributes.get("name")
        if isinstance(name, str) and role == TextRole.INSTRUMENT and set(tag_tokens(name)) <= set(tag_tokens(tag)):
            o.attributes["vector_name"] = o.attributes.pop("name")  # bubble letters read as an equipment name
        return {"object_id": o.id, "tag": tag, "source": source, "text_ids": [t.id for t in chosen]}

    def refine_symbols(self, doc: PIDDocument, pdf_path: Optional[str | Path] = None) -> PIDDocument:
        """Re-classify unknown / low-confidence / ambiguous symbols from crops + vector digest."""
        from ..llm.render import contact_sheet

        out = doc.model_copy(deep=True)
        before = self.client.usage.model_copy(deep=True)
        targets = [o for o in out.objects if self.needs_review(o)]
        stats: dict[str, Any] = {"source": self.label, "status": "skipped", "candidates": len(targets), "batches": 0,
                                 "failed_batches": 0, "adopted": 0, "confirmed": 0, "kept_vector": 0,
                                 "images": False, "tags": []}
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
        nearby = {o.id: self._nearby_texts(out, o) for o in targets}
        drawing_legend = drawing_legend_digest(out)
        if drawing_legend:
            stats["drawing_legend_rows"] = len(drawing_legend.get("rows", []))
        try:
            for page_no in sorted(by_page):
                objs = by_page[page_no]
                page = pdf[page_no] if pdf is not None and 0 <= page_no < pdf.page_count else None
                for i in range(0, len(objs), self.symbol_batch):
                    batch = objs[i : i + self.symbol_batch]
                    digest = [self._symbol_digest(out, o, n, pipe_ends, nearby[o.id]) for n, o in enumerate(batch, 1)]
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
                    legend_block = ("Drawing legend (this drawing's own legend; it overrides the built-in legend):\n"
                                    + prompts.to_json(drawing_legend) + "\n") if drawing_legend else ""
                    parts.append("Drawing: " + prompts.to_json(head) + "\n" + legend_block + "Candidates:\n"
                                 + prompts.to_json(digest)
                                 + "\nReturn one item for each id: " + ", ".join(o.id for o in batch) + ".")
                    msgs = [Message("system", sys_prompt), Message("user", parts)]
                    ids = [o.id for o in batch]
                    jobs.append(self._symbol_job(msgs, ids))
                    batches.append(batch)
        finally:
            if pdf is not None:
                pdf.close()
        stats["batches"] = len(jobs)
        maps = (out.text_by_id(), out.object_by_id(), {p.id: p for p in out.pipes})
        for batch, (result, err) in zip(batches, self._run_parallel(jobs)):
            if err is not None or result is None:
                stats["failed_batches"] += 1
                self._error(out.meta, f"refine_symbols[{batch[0].id}..{batch[-1].id}]", err or LLMError("no result"))
                continue
            verdicts = {v.id: v for v in result.items}
            for o in batch:
                decision = self._merge_symbol(o, verdicts[o.id])
                stats[decision] += 1
                if decision in ("adopted", "confirmed"):
                    change = self._claim_tag(out, o, verdicts[o.id], nearby[o.id], maps)
                    if change:
                        stats["tags"].append(change)
        stats["status"] = _step_status(stats["batches"], stats["failed_batches"])
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
        stats: dict[str, Any] = {"source": "vector", "status": "skipped"}
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
        line_systems = _line_number_systems(out)
        hints = {"keyword_system": stats["keyword_system"],
                 "parsed": det.model_dump(mode="json", exclude={"extra"}, exclude_defaults=True)}
        if line_systems:
            hints["line_number_systems"] = [s_.value for s_ in line_systems]
        parts: list[Any] = [ImagePart(image, label="title-block region")] if image else []
        parts.append("Title-block texts:\n" + prompts.to_json(items) + "\nNotes:\n" + prompts.to_json(note_items)
                     + "\nHints:\n" + prompts.to_json(hints))
        msgs = [Message("system", prompts.drawing_info_system_prompt()), Message("user", parts)]
        known_ids = {t.id for t in tb} | {t.id for t in notes}
        printed = printed_values(tb + notes)
        tb_printed = printed_values(tb)
        tb_tokens = {tok for t in tb for tok in value_tokens(t.text)}

        def validate(res: DrawingInfoOut) -> Optional[str]:
            problems = []
            bad = [i for i in res.evidence_text_ids if i not in known_ids]
            if bad:
                problems.append(f"evidence_text_ids {bad} are not among the given text ids")
            if printed:
                for field in ("drawing_no", "hull_no"):
                    value = getattr(res, field)
                    if value and _norm(value) not in printed:
                        problems.append(f"{field} {value!r} is not printed as a text item (or as the value after a "
                                        "field label); copy the complete value exactly or use null")
            if tb:
                for field in _INFO_TEXT_FIELDS:
                    value = getattr(res, field)
                    if isinstance(value, str) and value.strip() and not grounded_in(value, tb_printed, tb_tokens):
                        problems.append(f"{field} {value!r} is not printed in the title-block texts; copy it exactly "
                                        "from the title-block text items or use null")
            if res.drawing_no and res.hull_no and _norm(res.drawing_no) == _norm(res.hull_no):
                problems.append(f"drawing_no and hull_no are both {res.drawing_no!r}; they are different fields")
            return "; ".join(problems) or None

        try:
            res = self.client.generate_json(msgs, DrawingInfoOut, validator=validate)
        except LLMError as exc:
            self._error(out.meta, "read_drawing_info", exc)
            stats["status"] = "error"
            self._apply_keyword_system(out, keyword_system)
            self._record_usage(out, before)
            return out
        support = {"title_block": {t.id: t.text for t in tb}, "line_systems": line_systems,
                   "title_block_system": detect_system(" ".join(t.text for t in tb)) if tb else None}
        out.info = self._merge_info(det, res, keyword_system, support)
        stats["status"] = "ok"
        stats["source"] = out.info.extra.get("source")
        if out.info.extra.get("system_check"):
            stats["system_check"] = out.info.extra["system_check"]
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

    def _merge_info(self, det: DrawingInfo, res: DrawingInfoOut, keyword_system: Optional[SystemType],
                    support: Optional[dict[str, Any]] = None) -> DrawingInfo:
        """Gemma's title-block reading merged over the vector result.

        Text fields come from the (validated) answer. The drawing systems decide which rules apply,
        so they are grounded: Gemma's main system replaces the vector one only when a title-block
        text it cites as evidence names that system (``support["title_block"]``: id -> text); with
        no vector system it is accepted when cited title-block text, the line-number codes
        (``support["line_systems"]``), the title-block keywords or the keyword fallback support it.
        Secondary systems need the same support (or the vector result). Everything rejected is
        recorded in ``extra["system_check"]`` for the review instead of changing the verdicts.
        """
        support = support or {}
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
        unknown = SystemType.UNKNOWN
        tb_texts: dict[str, str] = support.get("title_block") or {}
        line_systems = {s_ for s_ in support.get("line_systems") or () if s_ != unknown}
        tb_system = support.get("title_block_system")
        cited = {s_ for s_ in (detect_system(tb_texts[i]) for i in res.evidence_text_ids if i in tb_texts) if s_}
        vector = {s_ for s_ in [det.system, *det.systems] if s_ != unknown}
        check: dict[str, Any] = {}
        g = res.system
        if g != unknown and det.system != unknown and g != det.system:
            if g in cited:
                values["system"], sources["system"] = g, self.label
                check = {"decision": "replaced", "gemma": g.value, "vector": det.system.value,
                         "basis": "cited title-block text"}
            else:
                values["system"], sources["system"] = det.system, "vector"
                check = {"decision": "kept_vector", "gemma": g.value, "vector": det.system.value,
                         "reason": "no cited title-block text names that system"}
        elif g != unknown and det.system == unknown:
            basis = ("cited title-block text" if g in cited else "line numbers" if g in line_systems else
                     "title-block keywords" if g == tb_system else "keyword fallback" if g == keyword_system else None)
            if basis:
                values["system"], sources["system"] = g, self.label
                check = {"decision": "accepted", "gemma": g.value, "basis": basis}
            else:
                check = {"decision": "rejected", "gemma": g.value,
                         "reason": "no title-block text, line number or keyword supports that system"}
                if keyword_system:
                    values["system"], sources["system"] = keyword_system, "keyword"
        elif g != unknown:  # confirms the vector system
            values["system"], sources["system"] = g, self.label
        elif det.system != unknown:
            values["system"], sources["system"] = det.system, "vector"
        elif keyword_system:
            values["system"], sources["system"] = keyword_system, "keyword"
        allowed = vector | line_systems | cited | ({tb_system} if tb_system else set())
        systems: list[SystemType] = []
        rejected: list[str] = []
        primary = values.get("system", unknown)
        for s_ in [primary] + list(res.systems) + list(det.systems):
            if s_ == unknown or s_ in systems:
                continue
            if s_ == primary or s_ in vector or s_ in allowed:
                systems.append(s_)
            elif s_.value not in rejected:
                rejected.append(s_.value)
        if rejected:
            check = {**check, "systems_rejected": rejected}
        if det.system != unknown and det.system not in systems:  # never drop the vector system silently
            systems.append(det.system)
        used = set(sources.values())
        source = self.label if used and used <= {self.label} else ("merged" if self.label in used else "vector")
        extra = {**det.extra, "source": source, "field_sources": sources,
                 "vector_info": det.model_dump(mode="json", exclude={"extra"}, exclude_defaults=True),
                 "keyword_system": keyword_system.value if keyword_system else None,
                 "evidence_text_ids": list(res.evidence_text_ids)}
        if check:
            extra["system_check"] = check
        return DrawingInfo(systems=systems, extra=extra, **values)

    # -- text ownership ----------------------------------------------------------------------
    def _owner_candidates(self, doc: PIDDocument, t: TextItem, limit: int = 5,
                          index: Optional[_SpatialIndex] = None) -> list[dict[str, Any]]:
        """Nearest pipes (line numbers) or objects (other texts) around ``t``, nearest first."""
        index = index or _SpatialIndex(doc)
        height = t.size or min(t.bbox.width, t.bbox.height) or 3.0
        radius = max(40.0, 5.0 * height)
        c = t.bbox.center
        found: list[dict[str, Any]] = []
        if t.role == TextRole.LINE_NUMBER:
            for p, gap, near in index.pipes_near(t.page, t.bbox, radius):
                entry = {"id": p.id, "kind": "pipe", "dist_pt": _r(gap),
                         "dir": _direction(near[0] - c.x, near[1] - c.y, gap)}
                if p.line_number:
                    entry["line_number"] = p.line_number
                found.append(entry)
        else:
            for o, gap in index.objects_near(t.page, t.bbox, radius):
                oc = o.bbox.center
                entry = {"id": o.id, "kind": o.cls.value, "dist_pt": _r(gap),
                         "dir": _direction(oc.x - c.x, oc.y - c.y, gap)}
                if o.tag:
                    entry["tag"] = o.tag
                found.append(entry)
        found.sort(key=lambda e: (e["dist_pt"], _id_key(e["id"])))
        return found[:limit]

    def ambiguous_texts(self, doc: PIDDocument) -> list[tuple[TextItem, list[dict[str, Any]]]]:
        """Owner-bearing texts with no valid owner, or several plausible owners."""
        valid = {o.id for o in doc.objects} | {p.id for p in doc.pipes}
        index = _SpatialIndex(doc)
        out = []
        for t in doc.texts:
            if t.role not in OWNER_TEXT_ROLES:
                continue
            cands = self._owner_candidates(doc, t, index=index)
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
        stats: dict[str, Any] = {"source": self.label, "status": "skipped", "texts": len(targets), "batches": 0,
                                 "failed_batches": 0, "changed": 0, "confirmed": 0}
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
        maps = (out.text_by_id(), out.object_by_id(), {p.id: p for p in out.pipes})
        for batch, (res, err) in zip(batches, self._run_parallel(jobs)):
            if err is not None or res is None:
                stats["failed_batches"] += 1
                self._error(out.meta, f"resolve_texts[{batch[0][0].id}..{batch[-1][0].id}]", err or LLMError("no result"))
                continue
            for a in res.assignments:
                changed = self._apply_assignment(out, a, maps)
                stats["changed" if changed else "confirmed"] += 1
                records.append({"text_id": a.text_id, "owner_id": a.owner_id, "role": a.role.value,
                                "changed": changed, "reason": a.reason[:200], "source": self.label})
        stats["status"] = _step_status(stats["batches"], stats["failed_batches"])
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
    def _apply_assignment(doc: PIDDocument, a: TextAssignment,
                          maps: Optional[tuple[dict[str, TextItem], dict[str, PIDObject], dict[str, PipeRun]]] = None,
                          ) -> bool:
        """Move a text to its new owner / role and keep tags and line numbers consistent.

        A text that leaves an object (or stops being a tag) takes its part of the
        object's tag with it: the tag is rebuilt from the tag texts the object
        still owns (bubble texts composed like the extractor, 'TIC' + '601' ->
        'TIC-601'). A text that joins an object sets or completes its tag the
        same way; tags and line numbers are normalised (upper case).
        """
        texts, objs, pipes = maps or (doc.text_by_id(), doc.object_by_id(), {p.id: p for p in doc.pipes})
        t = texts[a.text_id]
        old, new = t.owner_id, a.owner_id
        changed = old != new or t.role != a.role
        if not changed:
            return False
        t.role = a.role
        if old in objs:
            o = objs[old]
            if old != new:
                o.text_ids = [i for i in o.text_ids if i != t.id]
            if (old != new or a.role not in (TextRole.TAG, TextRole.INSTRUMENT)) and o.tag and _covers(o.tag, t.text):
                o.tag = _tag_from_texts(o, texts)
                o.attributes = {**o.attributes, "tag_source": "rebuilt"}
        elif old in pipes and old != new:
            p = pipes[old]
            p.text_ids = [i for i in p.text_ids if i != t.id]
            if p.line_number and _norm(p.line_number) == _norm(t.text):
                parsed = parse_line_number(normalize_text(t.text))
                rest = [texts[i] for i in p.text_ids if i in texts and texts[i].role == TextRole.LINE_NUMBER]
                p.line_number = normalize_text(rest[0].text) if rest else None
                if parsed and p.size == parsed["size"] and not rest:
                    p.size = None
        t.owner_id = new
        if new in objs:
            o = objs[new]
            if t.id not in o.text_ids:
                o.text_ids.append(t.id)
            if a.role == TextRole.INSTRUMENT:
                composed = _tag_from_texts(o, texts)
                if composed and (not o.tag or set(tag_tokens(o.tag)) <= set(tag_tokens(composed))):
                    o.tag = composed
            elif a.role == TextRole.TAG and not o.tag and _tag_like(t.text):
                o.tag = normalize_text(t.text)
        elif new in pipes:
            p = pipes[new]
            if t.id not in p.text_ids:
                p.text_ids.append(t.id)
            if a.role == TextRole.LINE_NUMBER and not p.line_number:
                p.line_number = normalize_text(t.text)
                parsed = parse_line_number(p.line_number)
                if parsed:
                    p.size = p.size or parsed["size"]
                    system = SYSTEM_BY_CODE.get(parsed["system"])
                    if system and p.medium == Medium.UNKNOWN:
                        p.medium = SYSTEMS[system].medium
        return changed

    # -- cross-sheet links ----------------------------------------------------------------
    def resolve_sheet_links(self, doc: PIDDocument, pdf_path: Optional[str | Path] = None) -> PIDDocument:
        """Let Gemma pair the off-page connectors the deterministic linker left ambiguous.

        A multi-page document without ``meta["sheet_links"]`` is linked first
        (:func:`shippid.analysis.sheets.link_sheets`, stored in the document so ``analyze()``
        reuses it). Each ambiguous connector (:func:`~shippid.analysis.sheets.ambiguous_links`:
        texts, references, names, equipment beside it, candidates with scores) is decided among
        its listed candidates only; one-to-one and mutually consistent answers are required, and
        decisions with confidence >= ``adopt_min_confidence`` are applied through
        :func:`~shippid.analysis.sheets.apply_link_decisions` with ``source`` = the label.
        """
        from ..llm.render import contact_sheet
        from .sheets import ambiguous_links, apply_link_decisions, link_sheets

        out = doc.model_copy(deep=True)
        before = self.client.usage.model_copy(deep=True)
        stats: dict[str, Any] = {"source": self.label, "status": "skipped", "connectors": 0, "batches": 0,
                                 "failed_batches": 0, "linked": 0, "unlinked": 0, "not_adopted": 0, "images": False}
        self._gemma_meta(out)["sheet_links"] = stats
        if _n_pages(out) <= 1:
            return out
        try:
            if "sheet_links" not in out.meta:
                link_sheets(out)
            amb = ambiguous_links(out, compute=False)
        except Exception as exc:  # the deterministic linker failed: analyze() links (and reports) again
            for key in ("sheet_links", "sheet_links_ambiguous", "sheet_link_stats"):
                out.meta.pop(key, None)
            stats["status"] = "error"
            self._error(out.meta, "resolve_sheet_links (deterministic linker)", exc)
            return out
        stats["rule_links"] = len(out.meta.get("sheet_links") or [])
        stats["connectors"] = len(amb)
        if not amb:
            return out
        pdf, pdf_err = self._open_pdf(pdf_path)
        if pdf_err and pdf_path is not None:
            out.meta.setdefault("llm_notes", []).append(f"resolve_sheet_links: {pdf_err}; digest-only prompts")
        objs = out.object_by_id()
        sys_prompt = prompts.sheet_link_system_prompt()
        jobs: list[Callable[[], SheetLinkDecisions]] = []
        batches: list[list[dict[str, Any]]] = []
        try:
            for i in range(0, len(amb), self.sheet_link_batch):
                batch = amb[i: i + self.sheet_link_batch]
                parts: list[Any] = []
                if pdf is not None:
                    ids = list(dict.fromkeys([e["connector"]["id"] for e in batch]
                                             + [c["id"] for e in batch for c in e["candidates"]]))
                    by_page: dict[int, list[tuple[str, Any]]] = defaultdict(list)
                    for oid in ids:
                        o = objs.get(oid)
                        if o is not None and 0 <= o.page < pdf.page_count:
                            m = max(30.0, 1.5 * max(o.bbox.width, o.bbox.height))
                            by_page[o.page].append((oid, _pymupdf_rect(o.bbox.expand(m))))
                    for page_no in sorted(by_page):
                        for png, labels in contact_sheet(pdf[page_no], by_page[page_no], max_side=self.crop_max_side):
                            parts.append(ImagePart(png, label=f"page {page_no} connectors {', '.join(labels)}"))
                    stats["images"] = stats["images"] or bool(parts)
                parts.append("Ambiguous connectors:\n" + prompts.to_json(batch) + "\nReturn one decision for each "
                             "connector: " + ", ".join(e["connector"]["id"] for e in batch) + ".")
                jobs.append(self._sheet_link_job([Message("system", sys_prompt), Message("user", parts)], batch))
                batches.append(batch)
        finally:
            if pdf is not None:
                pdf.close()
        stats["batches"] = len(jobs)
        records = out.meta.setdefault("sheet_link_resolution", [])
        linked: set[frozenset[str]] = set()
        for batch, (res, err) in zip(batches, self._run_parallel(jobs)):
            if err is not None or res is None:
                stats["failed_batches"] += 1
                self._error(out.meta, f"resolve_sheet_links[{batch[0]['connector']['id']}..{batch[-1]['connector']['id']}]",
                            err or LLMError("no result"))
                continue
            for d in res.decisions:
                conf = _clamp01(d.confidence)
                applied = False
                if d.partner is None:
                    stats["unlinked"] += 1
                elif conf < self.adopt_min_confidence:
                    stats["not_adopted"] += 1
                elif frozenset((d.connector, d.partner)) in linked:  # the partner's mirror decision
                    applied = True
                else:
                    added = apply_link_decisions(out, [{"a": d.connector, "b": d.partner, "confidence": conf,
                                                        "reason": d.reason}], source=self.label)
                    applied = bool(added)
                    stats["linked"] += len(added)
                    linked.update(frozenset((lk["a"], lk["b"])) for lk in added)
                records.append({"connector": d.connector, "partner": d.partner, "confidence": conf,
                                "applied": applied, "reason": d.reason[:200], "source": self.label})
        stats["status"] = _step_status(stats["batches"], stats["failed_batches"])
        self._record_usage(out, before)
        return out

    def _sheet_link_job(self, msgs: list[Message], batch: list[dict[str, Any]]) -> Callable[[], SheetLinkDecisions]:
        allowed = {e["connector"]["id"]: {c["id"] for c in e["candidates"]} for e in batch}
        order = [e["connector"]["id"] for e in batch]

        def validate(res: SheetLinkDecisions) -> Optional[str]:
            got = [d.connector for d in res.decisions]
            problems = []
            unknown = [i for i in got if i not in allowed]
            missing = [i for i in order if i not in got]
            dups = sorted({i for i in got if got.count(i) > 1})
            bad = [f"{d.connector}->{d.partner}" for d in res.decisions
                   if d.connector in allowed and d.partner is not None and d.partner not in allowed[d.connector]]
            chosen: dict[str, list[str]] = defaultdict(list)
            decided = {d.connector: d.partner for d in res.decisions}
            pairs = {a: b for a, b in decided.items() if b is not None}
            for a, b in pairs.items():
                chosen[b].append(a)
            shared = {b: sorted(a) for b, a in chosen.items() if len(a) > 1}
            inconsistent = sorted(f"{a}->{b} but {b}->{decided[b]}" for a, b in pairs.items()
                                  if b in decided and decided[b] != a)
            if unknown:
                problems.append(f"unknown connector ids {unknown} (decide only the listed connectors)")
            if missing:
                problems.append(f"missing connectors {missing} (one decision each)")
            if dups:
                problems.append(f"duplicate connectors {dups}")
            if bad:
                problems.append(f"partner not among that connector's candidates: {bad} (pick a candidate id or null)")
            if shared:
                problems.append(f"partners chosen for more than one connector: {shared} (each connector pairs with "
                                "one partner)")
            if inconsistent:
                problems.append(f"inconsistent pairs: {inconsistent}")
            return "; ".join(problems) or None

        return lambda: self.client.generate_json(msgs, SheetLinkDecisions, validator=validate)

    # -- whole document --------------------------------------------------------------------
    def refine_document(self, doc: PIDDocument, pdf_path: Optional[str | Path] = None) -> PIDDocument:
        """read_drawing_info + refine_symbols + resolve_texts (+ resolve_sheet_links for a multi-page
        set, which also stores the deterministic sheet links for ``analyze()``); an LLMError in a
        step keeps its input.

        ``meta["refine"]`` records every step's status (ok / partial / error /
        skipped); ``meta["refined_by"]`` is written only when at least one step got
        a valid model answer, and lists the steps that did and did not.
        """
        out = doc
        steps: list[tuple[str, str, Callable[[PIDDocument], PIDDocument]]] = [
            ("read_drawing_info", "drawing_info", lambda d: self.read_drawing_info(d, pdf_path)),
            ("refine_symbols", "symbols", lambda d: self.refine_symbols(d, pdf_path)),
            ("resolve_texts", "texts", self.resolve_texts),
        ]
        if _n_pages(doc) > 1:
            steps.append(("resolve_sheet_links", "sheet_links", lambda d: self.resolve_sheet_links(d, pdf_path)))
        status: dict[str, str] = {}
        for name, key, step in steps:
            try:
                out = step(out)
                status[name] = str(self._gemma_meta(out).get(key, {}).get("status", "ok"))
            except LLMError as exc:
                out = out.model_copy(deep=True) if out is doc else out
                self._error(out.meta, name, exc)
                status[name] = "error"
        if out is doc:
            out = doc.model_copy(deep=True)
        done = [n for n, st in status.items() if st in ("ok", "partial")]
        failed = [n for n, st in status.items() if st in ("error", "partial")]
        overall = ("skipped" if all(st == "skipped" for st in status.values()) else
                   "error" if not done else "partial" if failed else "ok")
        out.meta["refine"] = {"label": self.label, "backend": self.client.backend.name,
                              "model": self.client.backend.model, "status": overall, "steps": status}
        if done:
            out.meta["refined_by"] = {"label": self.label, "backend": self.client.backend.name,
                                      "model": self.client.backend.model, "steps_ok": done, "steps_failed": failed}
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
        multi = _n_pages(doc) > 1
        keep = ("page",) if multi else ()
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
            if multi:
                row["page"] = o.page
            return row

        def rows_or(name: str, given: list[dict[str, Any]], derived: Callable[[], list[dict[str, Any]]]) -> list[Any]:
            rows = [_scalarize(r, min_conf=self.min_confidence, keep=keep) for r in given] if given else derived()
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
            "verdict_counts": deterministic_verdicts(report),
        }
        if multi:
            digest["drawing"].pop("sheet", None)  # page 0's sheet number; sheets.sheet_of_page has every page
            digest["sheets"] = self._sheets_digest(doc, max_items)
        if truncated:
            digest["truncated_lists_total_lengths"] = truncated
        skipped = {k: list(report.meta.get(k) or doc.meta.get(k) or []) for k in ("unreadable_pages", "empty_pages")}
        if any(skipped.values()):
            # pages without readable vector content were not analysed: the review must not claim compliance
            digest["pages_not_analysed"] = skipped
        return digest

    @staticmethod
    def _sheets_digest(doc: PIDDocument, max_items: int) -> dict[str, Any]:
        """Sheet structure of a multi-page set: sheet number per page, linked connector pairs and
        the connectors left ambiguous (``doc.meta`` from :func:`~shippid.analysis.sheets.link_sheets`)."""
        from .graph import sheet_link_pairs
        from .sheets import page_sheet_numbers

        sheets = page_sheet_numbers(doc)
        objs = doc.object_by_id()
        sheet_of = {p: v["sheet"] for p, v in sorted(sheets.items())}
        links = []
        for a, b, lk in sheet_link_pairs(doc)[:max_items]:
            row: dict[str, Any] = {"a": a, "b": b, "sheets": [sheet_of.get(objs[a].page), sheet_of.get(objs[b].page)]}
            if lk.get("key"):
                row["key"] = str(lk["key"])[:60]
            if lk.get("source") and lk.get("source") != "rule":
                row["source"] = lk["source"]
            links.append(row)
        ambiguous = [row.get("id") for row in doc.meta.get("sheet_links_ambiguous") or []
                     if isinstance(row, dict) and row.get("id") in objs]
        out: dict[str, Any] = {"pages": _n_pages(doc), "sheet_of_page": {str(p): n for p, n in sheet_of.items()},
                               "links": links}
        if ambiguous:
            out["ambiguous_connectors"] = _cap_list(ambiguous, max_items)
        if "sheet_links" not in doc.meta:
            out["note"] = "connectors not linked yet"
        return out

    def _review_request(self, out: AnalysisReport) -> tuple[list[Message], str, dict[str, Any]]:
        """Review messages sized to the context: the digest shrinks (fewer rows per list) until system
        prompt + digest + schema note + a repair-round reserve + ``max_tokens`` fit ``context_tokens``
        (and the digest fits ``review_max_chars``); if even the smallest digest does not, ``max_tokens``
        is capped at what is left. Returns (messages, digest text, budget record)."""
        sys_prompt = prompts.review_system_prompt()
        describe = self.client.schema_in_prompt or self.client.json_mode != "json_schema"
        schema = schema_for(ReviewOut) if describe else None
        want = int(self.review_max_tokens or self.client.max_tokens)
        ctx = self.context_tokens
        reserve = repair_reserve(self.client.max_attempts)
        min_answer = REVIEW_MIN_ANSWER_TOKENS + (REVIEW_THINKING_TOKENS if self.review_thinking else 0)
        for max_items in (300, 150, 80, 40, 20, 10):
            digest_text = prompts.to_json(self.review_digest(out, max_items=max_items))
            msgs = [Message("system", sys_prompt), Message("user", "P&ID digest:\n" + digest_text)]
            prompt_tokens = messages_tokens(msgs, schema)
            if len(digest_text) <= self.review_max_chars and (not ctx or prompt_tokens + reserve + want <= ctx):
                break
        max_tokens = want
        if ctx:
            room = ctx - prompt_tokens - reserve
            if room < min_answer:  # no room for a repair round: give the first answer what is left
                room = ctx - prompt_tokens - 256
            max_tokens = max(256, min(want, room))
        budget = {"context_tokens": ctx, "prompt_tokens_est": prompt_tokens, "max_tokens": max_tokens,
                  "max_items": max_items}
        if ctx and prompt_tokens + max_tokens > ctx:
            budget["over_context"] = True
        return msgs, digest_text, budget

    def review(self, report: AnalysisReport) -> AnalysisReport:
        """Engineering review: append Gemma findings, set summaries, account LLM usage.

        ``meta["review"]["status"]`` is ``"ok"`` or ``"error"``; on error the report
        is returned unchanged apart from the error and the LLM usage of the failed
        attempts, and ``analyze()`` reports ``review_status="error"``.
        """
        out = report.model_copy(deep=True)
        doc = out.document
        if doc.meta.get("llm_usage") and not out.meta.get("llm_refine_usage_included"):
            out.llm = _usage_add(out.llm, LLMUsage.model_validate(doc.meta["llm_usage"]))
            out.meta["llm_refine_usage_included"] = True
        before = self.client.usage.model_copy(deep=True)
        valid = {o.id for o in doc.objects} | {p.id for p in doc.pipes} | {j.id for j in doc.junctions}
        msgs, digest_text, budget = self._review_request(out)
        verdicts = deterministic_verdicts(out)
        fails = verdicts["failed_rule_ids"]

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
            if verdicts["fail"]:
                for name, text in (("summary_en", res.summary_en), ("summary_ko", res.summary_ko)):
                    claim = compliance_claim(text)
                    if claim:
                        problems.append(f"{name} claims {claim!r}, but {verdicts['fail']} deterministic rule check(s) "
                                        f"failed ({', '.join(fails[:8])}); do not claim compliance, state the failures")
            return "; ".join(problems) or None

        try:
            res = self.client.generate_json(msgs, ReviewOut, validator=validate, max_tokens=budget["max_tokens"],
                                            thinking=True if self.review_thinking else None)
        except LLMError as exc:
            delta = _usage_delta(before, self.client.usage)
            self._error(out.meta, "review", exc)
            out.meta["review"] = {"source": self.label, "status": "error", "digest_chars": len(digest_text),
                                  "calls": delta.calls, **budget}
            out.llm = _usage_add(out.llm, delta)
            msg = self._redact(f"review: {exc}")[:1000]
            if msg not in out.llm.errors:
                out.llm.errors.append(msg)
            return out  # analyze() reports meta["review"]["status"] as the review_status
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
        skipped = sorted({*(out.meta.get("unreadable_pages") or []), *(out.meta.get("empty_pages") or [])})
        if skipped:  # keep the deterministic warning even when the model rewrites the summaries
            pages = ", ".join(str(int(p) + 1) for p in skipped)
            out.summary_en = f"WARNING: page(s) {pages} were not analysed (no readable vector content).\n" + out.summary_en
            out.summary_ko = f"경고: {pages} 페이지는 판독 가능한 벡터 데이터가 없어 분석하지 못했습니다.\n" + out.summary_ko
        out.meta["summary_source"] = self.label
        out.meta["review"] = {"source": self.label, "status": "ok", "findings": len(res.findings),
                              "thinking": self.review_thinking, "digest_chars": len(digest_text),
                              "deterministic_fail": verdicts["fail"], **budget}
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


def value_tokens(text: str) -> list[str]:
    """Upper-case words of a printed value with surrounding punctuation removed ('SYSTEM.' -> 'SYSTEM')."""
    return [w for w in (tok.strip(_TOKEN_EDGE) for tok in text.upper().split()) if w]


def grounded_in(value: str, printed: set[str], tokens: set[str]) -> bool:
    """``value`` is printed: a whole (labelled) value of a text item, or made only of printed words
    (a multi-line title joined with spaces, 'BALLAST WATER' + 'SYSTEM')."""
    if _norm(value) in printed:
        return True
    words = value_tokens(value)
    return bool(words) and all(w in tokens for w in words)


def _line_number_systems(doc: PIDDocument) -> list[SystemType]:
    """Systems named by the system codes of the drawing's line numbers (pipes and line-number texts),
    most frequent first."""
    counts: dict[SystemType, int] = defaultdict(int)
    numbers = [p.line_number for p in doc.pipes if p.line_number]
    numbers += [t.text for t in doc.texts if t.role == TextRole.LINE_NUMBER]
    for ln in numbers:
        parsed = parse_line_number(normalize_text(ln))
        system = SYSTEM_BY_CODE.get(parsed["system"]) if parsed else None
        if system:
            counts[system] += 1
    return [s_ for s_, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].value))]


_FIELD_LABEL = re.compile(r"(?:NO|NUMBER|DWG|DRAWING|HULL|SHIP|YARD|PROJECT|REV|REVISION|SHEET|SCALE)\.?:?$|.*:$", re.I)


def printed_values(texts: Sequence[TextItem]) -> set[str]:
    """Normalised values printed in title-block / note texts.

    A value is a whole text item, or what follows a field label inside one item
    ('DWG. NO. 1234-BA-001' -> '1234-BA-001', 'HULL NO: S1234' -> 'S1234').
    Fragments of a printed value (e.g. '7104' of '7104-PD-BA-09') are not values.
    """
    out: set[str] = set()
    for t in texts:
        text = t.text.strip()
        if not text:
            continue
        out.add(_norm(text))
        if ":" in text and text.rsplit(":", 1)[1].strip():
            out.add(_norm(text.rsplit(":", 1)[1]))
        tokens = text.split()
        for i in range(1, len(tokens)):
            if _FIELD_LABEL.match(tokens[i - 1]):
                out.add(_norm(" ".join(tokens[i:])))
    return out


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



def deterministic_verdicts(report: AnalysisReport) -> dict[str, Any]:
    """Counts of the deterministic (rule engine) findings per verdict and the failed rule ids."""
    det = [f for f in report.findings if f.source not in ("gemma", "mock")]
    out: dict[str, Any] = {v.value: sum(1 for f in det if f.verdict == v) for v in Verdict}
    out["failed_rule_ids"] = sorted({f.rule_id for f in det if f.verdict == Verdict.FAIL})
    return out


_NEGATION = re.compile(r"\b(?:not|never|nor|neither|cannot|fails?\s+to|failing\s+to)\b|n't\b", re.I)
_CLAIMS_EN = [re.compile(p, re.I) for p in (
    r"\ball\s+(?:of\s+)?(?:the\s+)?(?:\w+\s+){0,2}?(?:checks?|rules?|requirements?|regulations?|criteria)\s+"
    r"(?:have\s+|were\s+|are\s+|is\s+)?(?:been\s+)?(?:passed|pass|met|satisfied|fulfill?ed|complied\s+with)\b",
    r"\b(?:passes|passed|meets|satisfies|fulfill?s)\s+all\b",
    r"\b(?:fully|entirely|completely)\s+(?:compliant|complies|comply|conforms?|conformant)\b",
    r"\b(?:drawing|design|system|p&id|arrangement|installation)\s+(?:\w+\s+){0,2}?(?:complies|conforms|is\s+compliant)\b",
    r"\bno\s+(?:rule\s+|deterministic\s+)?(?:checks?\s+)?(?:failures?|violations?|non-?conformit(?:y|ies)|"
    r"non-?complian(?:ce|ces))\b(?!\s+(?:for|in|on|at|within|of|regarding|related)\b)",
    r"\bno\s+(?:rule\s+|deterministic\s+)?checks?\s+(?:have\s+|has\s+)?failed\b",
)]
_CLAIMS_KO = [re.compile(p) for p in (
    r"모두\s*(?:적합|통과|만족|충족|준수)",
    r"(?:부적합|위반|결함)\s*(?:사항|항목|건)?\s*(?:이|은|는|도)?\s*(?:전혀\s*)?없",
    r"모든\s*(?:규정|규칙|요건|요구\s*사항|기준|검토\s*항목|항목|검사)\S*\s*(?:을|를|이|가)?\s*(?:모두\s*)?"
    r"(?:만족|충족|준수|통과)(?:한다|합니다|함|하며|하고|했|하였|되었|됩니다)",
)]


def compliance_claim(text: str) -> Optional[str]:
    """The phrase of ``text`` that claims the drawing complies / has no failures, or None.

    Conservative on purpose (a scoped "no failures for the fire pumps" or a negated "does not
    comply with" is no claim); used only to reject summaries that contradict deterministic FAILs."""
    for pat in _CLAIMS_EN:
        for m in pat.finditer(text):
            if not _NEGATION.search(text[max(0, m.start() - 25): m.start()]):
                return m.group(0)
    for pat in _CLAIMS_KO:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


def _extractor_hints(o: PIDObject) -> dict[str, Any]:
    """What the extractor stored for the reviewer: the drawing-legend row an unknown symbol is drawn
    like (``attributes["legend"]``), why its confidence was lowered (``attributes["calibration"]``)
    and the legend entry a match came from (``attributes["template"]``)."""
    out: dict[str, Any] = {}
    leg = o.attributes.get("legend")
    if isinstance(leg, dict) and leg.get("description"):
        hint: dict[str, Any] = {"description": str(leg["description"])[:100]}
        if isinstance(leg.get("score"), (int, float)):
            hint["score"] = round(float(leg["score"]), 3)
        out["legend_hint"] = hint
    elif isinstance(leg, str) and leg.strip():
        out["legend_hint"] = {"description": leg.strip()[:100]}
    cal = o.attributes.get("calibration")
    if isinstance(cal, str):
        cal = [cal]
    if isinstance(cal, (list, tuple)) and cal:
        notes = [str(x)[:140] for x in cal if str(x).strip()]
        out["calibration"] = notes[:4] + ([f"…(+{len(notes) - 4} more)"] if len(notes) > 4 else [])
    tpl = o.attributes.get("template")
    if isinstance(tpl, str) and tpl.strip():
        out["template"] = tpl.strip()[:100]
    return out


def drawing_legend_digest(doc: PIDDocument, max_rows: int = 40) -> Optional[dict[str, Any]]:
    """The drawing's own legend (``doc.meta["legend"]`` from the extractor) for the symbol prompt:
    rows ``{description, class?, actuator?, status, page}`` and the unmapped descriptions; None if
    the drawing has no legend."""
    meta = doc.meta.get("legend") if isinstance(doc.meta, dict) else None
    if not isinstance(meta, dict):
        return None
    rows: list[dict[str, Any]] = []
    for lg in meta.get("legends") or []:
        if not isinstance(lg, dict):
            continue
        for r in lg.get("rows") or []:
            if not isinstance(r, dict) or not str(r.get("description") or "").strip():
                continue
            row: dict[str, Any] = {"description": str(r["description"])[:100]}
            if r.get("cls") and r.get("cls") != SymbolClass.UNKNOWN.value:
                row["class"] = r["cls"]
            if r.get("actuator") and r.get("actuator") != ActuatorType.NONE.value:
                row["actuator"] = r["actuator"]
            if r.get("status"):
                row["status"] = str(r["status"])[:60]
            if isinstance(r.get("page"), int):
                row["page"] = r["page"]
            rows.append(row)
    unmapped = [str(u.get("description"))[:100] for u in meta.get("unmapped") or []
                if isinstance(u, dict) and u.get("description")]
    if not rows and not unmapped:
        return None
    out: dict[str, Any] = {"rows": rows[:max_rows]}
    if len(rows) > max_rows:
        out["rows"].append(f"…(+{len(rows) - max_rows} more rows)")
    if unmapped:
        out["unmapped"] = list(dict.fromkeys(unmapped))[:max_rows]
    return out


def _step_status(batches: int, failed: int) -> str:
    """ok | partial | error | skipped for a step made of ``batches`` LLM requests."""
    if batches == 0:
        return "skipped"
    return "ok" if failed == 0 else ("error" if failed >= batches else "partial")


def _pymupdf_rect(b: BBox):
    import pymupdf

    return pymupdf.Rect(b.x0, b.y0, b.x1, b.y1)


def _n_pages(doc: PIDDocument) -> int:
    """Pages of a document (as ``analyze()`` counts them)."""
    return len({p.index for p in doc.pages} | {o.page for o in doc.objects})


def _page_size(doc: PIDDocument, page_no: int) -> tuple[float, float]:
    for p in doc.pages:
        if p.index == page_no:
            return p.width, p.height
    return 0.0, 0.0


def _adjacency(doc: PIDDocument) -> dict[str, list[str]]:
    """Object -> objects reachable through pipes, passing through junctions (signal lines excluded);
    paired off-page connectors (``doc.meta["sheet_links"]``) are adjacent to each other."""
    from .graph import sheet_link_pairs

    objects = {o.id for o in doc.objects}
    graph: dict[str, set[str]] = defaultdict(set)
    for p in doc.pipes:
        if p.kind == PipeKind.SIGNAL:
            continue
        a, b = p.start.node_id, p.end.node_id
        if a and b and a != b:
            graph[a].add(b)
            graph[b].add(a)
    for a, b, _ in sheet_link_pairs(doc):
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
