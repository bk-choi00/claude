"""Cross-sheet links: pairing off-page connectors of a multi-sheet P&ID set.

A ship system drawn on several sheets continues through off-page connectors: the line leaving
sheet 1 at ``TO NO.1 W.B.TK (P) / SH.2`` continues on sheet 2 at ``FROM BALLAST PUMP / SH.1``;
yards also write a shared continuation mark inside both arrows (``A1``) or a drawing / sheet
reference (``CONT. ON DWG 4446-PD-BA-12 SH.2``). :func:`link_sheets` pairs such connectors across
pages (and within one page, for shared marks) from normalised continuation keys:

* **sheet references** (``SH.2``, ``SHEET 2``, ``SHT 2``, a bare sheet number inside the arrow)
  against each page's own sheet number, read from its title block (``2 OF 3``, ``2/3``);
* **drawing numbers** (``DWG 4446-PD-BA-12``) against the drawing numbers in the title blocks
  (a reference to a drawing that is not in the document is an external connection: never paired);
* **continuation marks** (``A``, ``A1``, ``B-2``, ``11``) written inside both arrows;
* **destination / origin names**: ``TO <name>`` must name equipment that follows the partner on its
  sheet (the first equipment reached from it through valves / fittings / junctions), ``FROM <name>``
  equipment that precedes this connector on its own sheet (``BALLAST PUMPS`` matches ``NO.1 BALLAST
  PUMP``);
* **directions** (``TO`` / ``CONT. ON`` = outgoing, ``FROM`` / ``CONT. FROM`` = incoming): two
  outgoing or two incoming connectors are never paired;
* equal **line numbers** on the two connecting pipes (a continued line keeps its number).

A pair is a candidate only when *each* side's own text supports it (its mark equals the other's,
its sheet / drawing reference names the other's sheet, or its name matches the equipment beside
the other) and nothing contradicts it (other sheet named, different marks, same direction,
external drawing). Pairs are accepted one-to-one when they are each other's best candidate with a
score margin of at least ``margin`` over every competitor; accepting a pair can resolve others, so
this repeats until nothing changes. Connectors left with viable but unresolved candidates are
listed as ambiguous (:func:`ambiguous_links`) for a later Gemma step, which hands its choice back
through :func:`apply_link_decisions` (validated against the listed candidates).

Results are stored in ``doc.meta``:

* ``"sheet_links"``: ``[{"a", "b", "key", "confidence", "source", "directed", "evidence"}]`` -- ``a``
  is the outgoing (upstream) connector when ``directed`` (directions known), ``key`` the continuation keys that
  matched (``"mark:A1+sheet:1>2+name"``), ``source`` ``"rule"`` (``"gemma"`` / ``"manual"`` for
  decisions handed in later; ground truth of :mod:`shippid.datasets.multisheet` uses ``"gt"``);
* ``"sheet_links_ambiguous"``: ``[{"id", "page", "candidates": [{"id", "page", "score", "key"}]}]``;
* ``"sheet_link_stats"``: counts and the page -> sheet number map used.

:func:`shippid.analysis.graph.build_graph` turns every link into an edge ``kind="sheet_link"``
between the two connectors, so flow paths, adjacency, components, isolation and the rules see one
connected system across the sheets.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import networkx as nx

from ..model import Category, PIDDocument, PipeKind, SymbolClass, TextRole
from .graph import (
    INLINE_EQUIPMENT,
    PROCESS_KINDS,
    build_graph,
    effective_class,
    links,
    object_labels,
    sheet_link_pairs,
)

S = SymbolClass

SHEET_REF_RE = re.compile(r"\b(?:SHEET|SHT|SH)\s*\.?\s*(?:NO\.?\s*)?(\d{1,3})\b(?!\s*(?:OF|/)\s*\d)")
SHEET_OF_RE = re.compile(r"^\s*(?:(?:SHEET|SHT|SH)\s*\.?\s*(?:NO\.?\s*)?)?(\d{1,3})\s*(?:OF|/)\s*(\d{1,3})\s*$")
SHEET_OF_ANY_RE = re.compile(r"\b(?:SHEET|SHT|SH)\s*\.?\s*(?:NO\.?\s*)?(\d{1,3})\s*(?:OF|/)\s*(\d{1,3})\b")
SHEET_LABEL_RE = re.compile(r"^\s*(?:SHEET|SHT|SH)\s*\.?\s*(?:NO\.?)?\s*:?\s*$")
DWG_LABEL_RE = re.compile(r"^\s*(?:DWG|DRAWING)\s*\.?\s*(?:NO\s*\.?|NUMBER)\s*:?\s*(.*)$")
DWG_REF_RE = re.compile(r"\bDWG\s*\.?\s*(?:NO\s*\.?\s*)?[:#]?\s*([A-Z0-9][A-Z0-9./-]*\d[A-Z0-9./-]*)")
_CONT = r"CONT(?:'?D|INUED|\.)?\s*"
OUT_RE = re.compile(rf"(?:^|[\s(/])(?:TO|{_CONT}(?:ON|TO))\s")
IN_RE = re.compile(rf"(?:^|[\s(/])(?:FROM|FR\.|{_CONT}FROM)\s")
MARK_RE = re.compile(r"^[A-Z]{0,3}[-.]?\d{0,3}[A-Z]?$")
_NAME_RE = re.compile(r"^\s*(?:TO|FROM|FR\.)\s+(.+)$")
_REF_TAIL_RE = re.compile(r"\s*[/(]?\s*(?:\(?\s*(?:SHEET|SHT|SH)\s*\.?\s*(?:NO\.?\s*)?\d.*|\(?\s*DWG\b.*|CONT\b.*)$")
_NO_RE = re.compile(r"\bNO\.?\s*\d+[A-Z]?(?:\s*[~&,-]\s*\d+[A-Z]?)*")
_STOP = frozenset({"TO", "FROM", "FR", "CONT", "CONTD", "ON", "DWG", "NO", "SH", "SHEET", "SHT", "OF", "THE", "AND",
                   "SEE", "ETC", "P", "S", "C", "P&S", "PS", "PORT", "STBD", "&", "-", "/"})
_GENERIC = frozenset({"PUMP", "TK", "VALVE", "V/V", "UNIT", "LINE", "MAIN", "SYSTEM", "PIPE", "HEADER", "SUCTION",
                      "DISCHARGE", "INLET", "OUTLET"})
_SYNONYM = {"TANK": "TK", "TANKS": "TK", "TKS": "TK", "V/VS": "V/V"}

#: score weights of the continuation keys
W_MARK = 4.0
W_SHEET = 2.0
W_DWG = 1.0
W_NAME = 1.5
W_NAME_CONFLICT = 1.0  # penalty: the name written does not match the (fully named) equipment beside the partner
W_LINE = 1.0
W_DIRECTION = 0.5


# --------------------------------------------------------------------------- text keys


def norm_ref(s: str) -> str:
    """Reference comparison form: upper case letters and digits only (``A-1`` == ``A1``)."""
    return re.sub(r"[^0-9A-Z]", "", s.upper())


def name_tokens(text: str) -> set[str]:
    """Comparable words of an equipment / destination name (numbering, sides, plurals and
    connector words removed; ``TANKS`` -> ``TK``, ``PUMPS`` -> ``PUMP``)."""
    s = _NO_RE.sub(" ", text.upper())
    s = re.sub(r"\s/\s|[(),;:&]", " ", s)
    out = set()
    for tok in s.split():
        tok = tok.strip(".'\"")
        tok = _SYNONYM.get(tok, tok)
        if len(tok) > 3 and tok.endswith("S") and not tok.endswith("SS"):
            tok = tok[:-1].strip(".")
            tok = _SYNONYM.get(tok, tok)
        if not tok or tok in _STOP or tok.isdigit():
            continue
        out.add(tok)
    return out


def names_match(name: str, pool: set[str]) -> bool:
    """Most words of ``name`` (and at least one specific word) occur in the token ``pool``."""
    toks = name_tokens(name)
    if not toks:
        return False
    inter = toks & pool
    return len(inter) >= max(1, math.ceil(0.75 * len(toks))) and bool(inter - _GENERIC)


@dataclass
class ConnectorKeys:
    """Continuation keys read from one off-page connector."""

    id: str
    page: int
    sheet: Optional[int]  # sheet number of the connector's own page
    texts: list[str] = field(default_factory=list)
    inside: list[str] = field(default_factory=list)
    direction: Optional[str] = None  # "out" | "in"
    direction_hard: bool = False  # read from the text (or ground truth), not guessed from pipe flow
    sheet_refs: set[int] = field(default_factory=set)
    dwg_refs: set[str] = field(default_factory=set)
    mark: Optional[str] = None  # normalised continuation mark (not a sheet reference)
    names: list[str] = field(default_factory=list)  # destination / origin names
    near: set[str] = field(default_factory=set)  # name tokens of the equipment beside it on its sheet
    near_names: list[str] = field(default_factory=list)
    near_complete: bool = False  # every equipment item beside it is named (a mismatch is then evidence)
    line_numbers: set[str] = field(default_factory=set)

    @property
    def digit_mark(self) -> bool:
        return bool(self.mark) and self.mark.isdigit()

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "page": self.page, "sheet": self.sheet, "texts": self.texts, "direction": self.direction,
                "sheet_refs": sorted(self.sheet_refs), "dwg_refs": sorted(self.dwg_refs), "mark": self.mark,
                "names": self.names, "near": self.near_names[:6]}


def _drawing_no(title: Sequence) -> Optional[str]:
    """Normalised drawing number of one title block: the value after / below a ``DWG. NO.`` label."""
    for t in title:
        m = DWG_LABEL_RE.match(t.text.upper())
        if not m:
            continue
        rest = m.group(1).strip()
        if rest and re.search(r"\d", rest):
            return norm_ref(rest)
        h = max(t.bbox.height, 1.0)
        best = None
        for v in title:
            if v is t or not re.search(r"\d", v.text) or len(norm_ref(v.text)) < 4:
                continue
            if DWG_LABEL_RE.match(v.text.upper()):
                continue
            below = v.bbox.y0 >= t.bbox.y1 - 0.5 * h and v.bbox.y0 - t.bbox.y1 <= 4 * h
            right = abs(v.bbox.center.y - t.bbox.center.y) <= 0.7 * h and v.bbox.x0 >= t.bbox.x1 - h
            if not (below or right) or v.bbox.x0 < t.bbox.x0 - 2 * h or v.bbox.x0 > t.bbox.x0 + 60 * h:
                continue
            d = v.bbox.center.dist(t.bbox.center)
            if best is None or d < best[0]:
                best = (d, v)
        if best is not None:
            return norm_ref(best[1].text)
    return None


def page_sheet_numbers(doc: PIDDocument) -> dict[int, dict[str, Any]]:
    """Per page: ``{"sheet", "of", "inferred", "drawing_no", "titles"}`` -- the sheet number from the
    title block (``2 OF 3`` / ``2/3`` next to a SHEET label, or ``SHEET 2 OF 3`` anywhere), else the
    page index + 1 (``inferred``); the normalised drawing number (``None`` if not found) and the
    normalised title-block strings / words used to recognise drawing references."""
    pages = sorted({p.index for p in doc.pages} | {o.page for o in doc.objects} | {t.page for t in doc.texts})
    by_page: dict[int, list] = defaultdict(list)
    for t in doc.texts:
        by_page[t.page].append(t)
    out: dict[int, dict[str, Any]] = {}
    for p in pages:
        texts = by_page.get(p, [])
        title = [t for t in texts if t.role == TextRole.TITLE_BLOCK]
        found: Optional[tuple[int, int]] = None
        cands = [(t, SHEET_OF_RE.match(t.text.upper())) for t in title]
        # (a scale such as "1/100" also reads "k/n": prefer the value next to a SHEET label, then "k OF n")
        cands = [(t, m) for t, m in cands if m and 0 < int(m.group(1)) <= int(m.group(2)) <= 99]
        if cands:
            labels = [t for t in title if SHEET_LABEL_RE.match(t.text.upper())]
            lc = labels[0].bbox.center if labels else None
            cands.sort(key=lambda tm: (tm[0].bbox.center.dist(lc) if lc is not None else 0.0,
                                       "OF" not in tm[0].text.upper()))
            found = (int(cands[0][1].group(1)), int(cands[0][1].group(2)))
        else:
            for t in title or texts:
                m = SHEET_OF_ANY_RE.search(t.text.upper())
                if m and 0 < int(m.group(1)) <= int(m.group(2)):
                    found = (int(m.group(1)), int(m.group(2)))
                    break
        titles: set[str] = set()
        for t in title:
            titles.add(norm_ref(t.text))
            titles.update(norm_ref(w) for w in t.text.split())
        titles.discard("")
        out[p] = {"sheet": found[0] if found else p + 1, "of": found[1] if found else None,
                  "inferred": found is None, "drawing_no": _drawing_no(title), "titles": titles}
    return out


def _first_layer(G: nx.MultiGraph, start: str, limit: int = 400) -> tuple[list[str], set[str], bool]:
    """Names (labels / tags) of the equipment beside connector ``start`` on its own sheet: the
    first equipment reached through junctions, valves, fittings, instruments and in-line
    equipment (whose names are collected on the way); one more equipment layer when the first
    one is unnamed. Other off-page connectors end the search. The flag is True when every
    equipment item met has a name (label), so a destination name that matches none of them
    contradicts the pairing."""
    names: list[str] = []
    frontier = [start]
    seen = {start}
    complete = True
    for _layer in range(2):
        nxt: list[str] = []
        q = deque(frontier)
        while q and len(seen) < limit:
            n = q.popleft()
            for lk in links(G, n, PROCESS_KINDS):
                m = lk.other
                if m in seen:
                    continue
                seen.add(m)
                d = G.nodes[m]
                kind = d.get("kind")
                if kind == "junction":
                    q.append(m)
                    continue
                if kind != "object":
                    continue
                cls, cat = d.get("eff_cls"), d.get("eff_category")
                if cls == S.OFF_PAGE_CONNECTOR:
                    continue
                label = d.get("label") or ""
                tag = d.get("tag") or ""
                if cat in (Category.VALVE, Category.FITTING, Category.INSTRUMENT) or cls == S.UNKNOWN:
                    q.append(m)
                    continue
                if not label and cls not in INLINE_EQUIPMENT:
                    complete = False
                for v in (label, tag):
                    if v and v not in names:
                        names.append(v)
                if cls in INLINE_EQUIPMENT:
                    q.append(m)
                else:
                    nxt.append(m)
        if names or not nxt:
            break
        frontier = nxt
        complete = True
    tokens: set[str] = set()
    for v in names:
        tokens |= name_tokens(v)
    return names, tokens, complete and bool(names)


def _texts_of(doc: PIDDocument, obj) -> tuple[list[str], list[str]]:
    """(all texts, texts written inside the arrow) of a connector: owned texts in reading order,
    plus the ``name`` / ``rating`` / ``ref`` attributes."""
    owned = sorted((t for t in doc.texts if t.owner_id == obj.id and t.text.strip()),
                   key=lambda t: (round(t.bbox.y0, 1), t.bbox.x0))
    texts: list[str] = []
    inside: list[str] = []
    b = obj.bbox
    for t in owned:
        s = " ".join(t.text.split())
        texts.append(s)
        c = t.bbox.center
        if b.contains_point(c.x, c.y, tol=0.5):
            inside.append(s)
    for k in ("ref", "name", "rating", "label"):
        v = obj.attributes.get(k)
        if isinstance(v, str) and v.strip():
            v = " ".join(v.split())
            if k == "ref" and v not in inside:
                inside.append(v)
            if v not in texts and not any(v in x for x in texts):
                texts.append(v)
    return list(dict.fromkeys(texts)), list(dict.fromkeys(inside))


def connector_keys(doc: PIDDocument, G: Optional[nx.MultiGraph] = None,
                   sheets: Optional[dict[int, dict[str, Any]]] = None) -> dict[str, ConnectorKeys]:
    """Continuation keys of every off-page connector (``eff`` class, so connectors the extractor
    left unclassified but named ``TO ...`` / ``FROM ...`` count too)."""
    G = G if G is not None else build_graph(doc, with_sheet_links=False)
    sheets = sheets if sheets is not None else page_sheet_numbers(doc)
    labels = object_labels(doc)
    pipes_of: dict[str, list] = defaultdict(list)
    for p in doc.pipes:
        if p.kind != PipeKind.PROCESS:
            continue
        for ep, into in ((p.start, -1), (p.end, 1)):
            if ep.object_id:
                pipes_of[ep.object_id].append((p, into))
    out: dict[str, ConnectorKeys] = {}
    for o in doc.objects:
        if effective_class(o, labels.get(o.id)) != S.OFF_PAGE_CONNECTOR:
            continue
        texts, inside = _texts_of(doc, o)
        k = ConnectorKeys(id=o.id, page=o.page, sheet=sheets.get(o.page, {}).get("sheet"), texts=texts,
                          inside=inside)
        up = [t.upper() for t in texts]
        outs = sum(1 for t in up if OUT_RE.search(" " + t))
        ins = sum(1 for t in up if IN_RE.search(" " + t))
        if outs != ins:
            k.direction, k.direction_hard = ("out" if outs > ins else "in"), True
        elif o.attributes.get("direction") in ("in", "out"):
            k.direction, k.direction_hard = o.attributes["direction"], True
        else:
            flows = {(p.flow_direction or 0) * into for p, into in pipes_of.get(o.id, [])} - {0}
            if len(flows) == 1:
                k.direction = "out" if flows.pop() > 0 else "in"
        for t in up:
            k.sheet_refs.update(int(m.group(1)) for m in SHEET_REF_RE.finditer(t))
            k.dwg_refs.update(norm_ref(m.group(1)) for m in DWG_REF_RE.finditer(t))
            m = _NAME_RE.match(t)
            if m:
                name = _REF_TAIL_RE.sub("", m.group(1)).strip(" /(")
                if name and name_tokens(name):
                    k.names.append(name)
        for s in inside:
            su = s.upper().strip()
            if SHEET_REF_RE.search(su) or OUT_RE.search(" " + su) or IN_RE.search(" " + su):
                continue
            if MARK_RE.match(su.replace(" ", "")) and norm_ref(su):
                k.mark = norm_ref(su)
                break
        k.near_names, k.near, k.near_complete = _first_layer(G, o.id)
        k.line_numbers = {re.sub(r"\s+", "", p.line_number.upper()) for p, _ in pipes_of.get(o.id, [])
                          if p.line_number}
        out[o.id] = k
    return out


# --------------------------------------------------------------------------- candidates


@dataclass
class Candidate:
    a: str
    b: str
    score: float
    key: str
    evidence: list[str]
    directed: bool = False  # a is known to be the outgoing side
    conflict: bool = False  # both names contradict the equipment beside the partner: never auto-accepted


def _support(x: ConnectorKeys, y: ConnectorKeys, sheets: dict[int, dict[str, Any]],
             doc_dwgs: bool) -> tuple[Optional[list[str]], float, list[str]]:
    """What ``x``'s own text says about ``y`` being its partner: (None if contradicted, else the
    supporting keys), score, evidence."""
    keys: list[str] = []
    ev: list[str] = []
    score = 0.0
    ysheet = y.sheet
    refs = set(x.sheet_refs)
    if x.digit_mark and y.digit_mark and x.mark != y.mark:
        # two different bare numbers: the partner's sheet numbers, or a contradiction
        if not (int(x.mark) == ysheet and int(y.mark) == x.sheet):
            return None, 0.0, []
        refs.add(int(x.mark))
    elif x.digit_mark and not y.mark and int(x.mark) == ysheet and x.page != y.page:
        refs.add(int(x.mark))
    if refs:
        if ysheet in refs and (x.page != y.page or x.sheet in refs):
            keys.append("sheet")
            score += W_SHEET
            ev.append(f"{x.id} refers to sheet {ysheet}")
        elif x.page != y.page:
            return None, 0.0, []  # names another sheet
    if x.dwg_refs:
        here = sheets.get(y.page, {}).get("titles", set())
        known = any(d in sheets.get(p, {}).get("titles", set()) for p in sheets for d in x.dwg_refs)
        if any(d in here for d in x.dwg_refs):
            keys.append("dwg")
            score += W_DWG
            ev.append(f"{x.id} refers to the drawing of page {y.page}")
        elif x.page != y.page and (known or doc_dwgs):
            return None, 0.0, []  # another drawing of this set, or an external drawing
    if x.names and any(names_match(nm, y.near) for nm in x.names):
        keys.append("name")
        score += W_NAME
        ev.append(f"{x.id} names equipment beside {y.id}")
    elif x.names and y.near_complete:
        score -= W_NAME_CONFLICT
        ev.append(f"{x.id} names {x.names[0]!r}, not the equipment beside {y.id}")
        keys.append("!name")
    return keys, score, ev


def _pos(keys: list[str]) -> bool:
    return any(not k.startswith("!") for k in keys)


def _pair(x: ConnectorKeys, y: ConnectorKeys, sheets: dict[int, dict[str, Any]], doc_dwgs: bool) -> Optional[Candidate]:
    if x.id == y.id:
        return None
    if x.direction and y.direction and x.direction == y.direction and x.direction_hard and y.direction_hard:
        return None
    marks = x.mark is not None and y.mark is not None and not (x.digit_mark and y.digit_mark)
    if marks and x.mark != y.mark:
        return None  # different continuation marks
    mark_eq = x.mark is not None and x.mark == y.mark
    same_page = x.page == y.page
    if same_page and not mark_eq:
        return None  # within a sheet only a shared mark continues a line
    sx = _support(x, y, sheets, doc_dwgs)
    sy = _support(y, x, sheets, doc_dwgs)
    if sx[0] is None or sy[0] is None:
        return None
    dx, dy = sheets.get(x.page, {}).get("drawing_no"), sheets.get(y.page, {}).get("drawing_no")
    if not same_page and dx and dy and dx != dy and "dwg" not in sx[0] + sy[0]:
        return None  # marks and sheet numbers belong to one drawing: another drawing must be named
    if same_page:
        # a continuation within one sheet: the shared mark plus names that point at each other
        # (or bare marks); a reference shared by "TO X SYSTEM" / "FROM X SYSTEM" names a drawing
        if (x.names or y.names) and not ("name" in sx[0] and "name" in sy[0]):
            return None
    elif mark_eq:
        if not (_pos(sx[0]) or _pos(sy[0])) and (x.names or y.names):
            return None  # same reference, but the destinations named are not on the other sheet
    elif not _pos(sx[0]) or not _pos(sy[0]):
        return None  # each side's own text must point at the other
    score = sx[1] + sy[1]
    ev = sx[2] + sy[2]
    key_parts: list[str] = []
    if mark_eq:
        score += W_MARK
        key_parts.append(f"mark:{x.mark}")
        ev.insert(0, f"shared mark {x.mark}")
    if "sheet" in sx[0] or "sheet" in sy[0]:
        key_parts.append(f"sheet:{x.sheet}>{y.sheet}" if x.direction != "in" else f"sheet:{y.sheet}>{x.sheet}")
    if "dwg" in sx[0] or "dwg" in sy[0]:
        key_parts.append("dwg")
    if "name" in sx[0] or "name" in sy[0]:
        key_parts.append("name")
    conflict = "!name" in sx[0] and "!name" in sy[0]
    if conflict:
        ev.append("both names contradict: left for review")
    if x.direction and y.direction and x.direction != y.direction:
        score += W_DIRECTION
    if x.line_numbers & y.line_numbers:
        score += W_LINE
        key_parts.append("line")
        ev.append("same line number")
    a, b = (x, y)
    if x.direction == "in" or y.direction == "out":
        a, b = y, x
    directed = a.direction == "out" or b.direction == "in"
    return Candidate(a.id, b.id, round(score, 3), "+".join(key_parts), ev, directed, conflict)


def candidate_pairs(doc: PIDDocument, keys: Optional[dict[str, ConnectorKeys]] = None,
                    sheets: Optional[dict[int, dict[str, Any]]] = None) -> list[Candidate]:
    """Every viable connector pair (see module docstring), best first."""
    sheets = sheets if sheets is not None else page_sheet_numbers(doc)
    keys = keys if keys is not None else connector_keys(doc, sheets=sheets)
    doc_dwgs = any(sheets[p]["titles"] for p in sheets)
    ks = list(keys.values())
    out: list[Candidate] = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            c = _pair(ks[i], ks[j], sheets, doc_dwgs)
            if c is not None:
                out.append(c)
    out.sort(key=lambda c: (-c.score, c.a, c.b))
    return out


# --------------------------------------------------------------------------- linking


def _confidence(score: float, margin: float) -> float:
    return round(min(0.99, 0.4 + 0.08 * score + 0.05 * min(margin, 3.0)), 3)


def resolve(cands: Sequence[Candidate], margin: float = 1.0, min_score: float = 2.5,
            taken: Iterable[str] = ()) -> tuple[list[tuple[Candidate, float]], dict[str, list[Candidate]]]:
    """One-to-one acceptance: a pair whose two connectors are each other's best candidate with a
    margin of ``margin`` over every other candidate of either connector; repeated until stable.
    Returns the accepted (candidate, margin) pairs and the remaining candidates per unpaired
    connector."""
    used = set(taken)
    pool = [c for c in cands if c.score >= min_score and c.a not in used and c.b not in used]
    accepted: list[tuple[Candidate, float]] = []
    changed = True
    while changed:
        changed = False
        by: dict[str, list[Candidate]] = defaultdict(list)
        for c in pool:
            by[c.a].append(c)
            by[c.b].append(c)
        for c in sorted(pool, key=lambda c: (-c.score, c.a, c.b)):
            if c.a in used or c.b in used or c.conflict:
                continue
            others = [o for o in by[c.a] + by[c.b] if o is not c]
            best_other = max((o.score for o in others), default=None)
            m = c.score - best_other if best_other is not None else math.inf
            if m >= margin:
                accepted.append((c, m))
                used.update((c.a, c.b))
                changed = True
        pool = [c for c in pool if c.a not in used and c.b not in used]
    left: dict[str, list[Candidate]] = defaultdict(list)
    for c in pool:
        left[c.a].append(c)
        left[c.b].append(c)
    return accepted, dict(left)


def link_sheets(doc: PIDDocument, *, store: bool = True, margin: float = 1.0, min_score: float = 2.5,
                keep: Sequence[str] = ("gemma", "manual")) -> list[dict[str, Any]]:
    """Pair off-page connectors across (and within) pages; returns ``[{"a", "b", "key",
    "confidence", "source", "evidence"}]`` and, with ``store``, writes ``doc.meta["sheet_links"]``,
    ``["sheet_links_ambiguous"]`` and ``["sheet_link_stats"]``. Links already in the document whose
    ``source`` is in ``keep`` (Gemma / manual decisions) are kept and their connectors not re-paired."""
    sheets = page_sheet_numbers(doc)
    keys = connector_keys(doc, sheets=sheets)
    kept = [dict(lk) for lk in doc.meta.get("sheet_links", []) or []
            if isinstance(lk, dict) and lk.get("source") in keep and lk.get("a") in keys and lk.get("b") in keys]
    taken = {x for lk in kept for x in (lk["a"], lk["b"])}
    cands = candidate_pairs(doc, keys, sheets)
    accepted, left = resolve(cands, margin=margin, min_score=min_score, taken=taken)
    links_out = list(kept)
    for c, m in accepted:
        links_out.append({"a": c.a, "b": c.b, "key": c.key, "confidence": _confidence(c.score, m), "source": "rule",
                          "directed": c.directed, "score": c.score, "evidence": c.evidence})
    ambiguous = []
    order = {o.id: i for i, o in enumerate(doc.objects)}
    for cid in sorted(left, key=lambda i: order.get(i, 0)):
        cs = sorted(left[cid], key=lambda c: (-c.score, c.a, c.b))
        ambiguous.append({"id": cid, "page": keys[cid].page, "candidates": [
            {"id": c.b if c.a == cid else c.a, "page": keys[c.b if c.a == cid else c.a].page, "score": c.score,
             "key": c.key} for c in cs]})
    if store:
        doc.meta["sheet_links"] = links_out
        doc.meta["sheet_links_ambiguous"] = ambiguous
        doc.meta["sheet_link_stats"] = {
            "connectors": len(keys), "candidates": len(cands), "linked": len(links_out),
            "ambiguous": len(ambiguous), "pages": {str(p): v["sheet"] for p, v in sorted(sheets.items())},
            "drawing_nos": {str(p): v["drawing_no"] for p, v in sorted(sheets.items())},
            "sheet_numbers_inferred": sorted(p for p, v in sheets.items() if v["inferred"])}
    return links_out


def ambiguous_links(doc: PIDDocument, *, compute: bool = True) -> list[dict[str, Any]]:
    """Connectors left unpaired with more than one viable partner (or a partner below the margin),
    with everything a reviewer (Gemma) needs to decide: per connector its texts, page / sheet,
    direction, references, names and the equipment beside it, and per candidate the same plus the
    rule score and matched keys. Uses ``doc.meta["sheet_links_ambiguous"]`` (computed by
    :func:`link_sheets` first when missing and ``compute``)."""
    if "sheet_links_ambiguous" not in doc.meta:
        if not compute:
            return []
        link_sheets(doc)
    sheets = page_sheet_numbers(doc)
    keys = connector_keys(doc, sheets=sheets)
    out = []
    for row in doc.meta.get("sheet_links_ambiguous", []) or []:
        k = keys.get(row.get("id"))
        if k is None:
            continue
        cands = []
        for c in row.get("candidates", []):
            ck = keys.get(c.get("id"))
            if ck is not None:
                cands.append({**ck.summary(), "score": c.get("score"), "key": c.get("key")})
        if cands:
            out.append({"connector": k.summary(), "candidates": cands})
    return out


def apply_link_decisions(doc: PIDDocument, decisions: Iterable[dict[str, Any]],
                         source: str = "gemma") -> list[dict[str, Any]]:
    """Add reviewer decisions ``[{"a", "b", "confidence"?, "reason"?}]`` for ambiguous connectors.
    A decision is applied only when ``b`` is one of the listed candidates of ``a`` (or vice versa)
    and neither connector is linked already; returns the links added (``source`` recorded)."""
    amb = {row["id"]: {c["id"] for c in row.get("candidates", [])}
           for row in doc.meta.get("sheet_links_ambiguous", []) or [] if isinstance(row, dict) and "id" in row}
    links_now = list(doc.meta.get("sheet_links", []) or [])
    taken = {x for lk in links_now if isinstance(lk, dict) for x in (lk.get("a"), lk.get("b"))}
    added = []
    for d in decisions:
        a, b = d.get("a"), d.get("b")
        if not a or not b or a == b or a in taken or b in taken:
            continue
        if b not in amb.get(a, set()) and a not in amb.get(b, set()):
            continue
        conf = d.get("confidence")
        lk = {"a": a, "b": b, "key": "review", "confidence": float(conf) if isinstance(conf, (int, float)) else 0.5,
              "source": source}
        if d.get("reason"):
            lk["evidence"] = [str(d["reason"])[:300]]
        links_now.append(lk)
        added.append(lk)
        taken.update((a, b))
    if added:
        doc.meta["sheet_links"] = links_now
        doc.meta["sheet_links_ambiguous"] = [row for row in doc.meta.get("sheet_links_ambiguous", [])
                                             if row.get("id") not in taken]
    return added


def linked_pairs(doc: PIDDocument) -> set[frozenset[str]]:
    """The connector pairs of ``doc.meta["sheet_links"]`` that exist in the document."""
    return {frozenset((a, b)) for a, b, _ in sheet_link_pairs(doc)}
