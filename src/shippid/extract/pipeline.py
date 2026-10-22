"""Deterministic vector extraction: PDF page -> :class:`~shippid.model.PIDDocument`.

Stages (see ``docs/ARCHITECTURE.md`` section 2): read the page, find sheet
furniture (frame, title block, notes, legend), separate pipe lines from symbol
strokes, group and classify symbol candidates by template matching, build the
pipe network and finally assign texts. Every object and pipe carries
``source="vector"``; per-candidate class alternatives are kept for Gemma.

Text whose visible glyphs are vector strokes (AutoCAD SHX text exported with PDFSHX=1 / 2,
see :mod:`shippid.extract.textlines`) takes its strokes out of symbol and pipe search.
Confidence is calibrated so that drawings the library does not explain go to review
(``confidence`` below 0.75, reasons in ``attributes["calibration"]``): unexplained or missing
ink (:func:`shippid.extract.symbols_match.ink_penalty`), pipes meeting an instrument circle
off its radius, a second circle drawn inside a bubble.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ..catalog import MEDIUM_COLORS, medium_rgb
from ..config import Config
from ..model import (
    ActuatorType,
    BBox,
    Category,
    DrawingInfo,
    Endpoint,
    Junction,
    Medium,
    PageInfo,
    PIDDocument,
    PIDObject,
    PipeKind,
    PipeRun,
    Point,
    Port,
    SymbolClass,
    TextItem,
    TextRole,
)
from ..pdf.reader import PageContent, open_pdf, read_page
from ..symbols import category_of, get_symbol
from .candidates import Candidate, group_candidates
from .legend import (
    LEGEND_TIE,
    LegendBook,
    PageGeometry,
    find_legends,
    heading_lines,
    is_learned_variant,
    name_stub_ports,
    page_has_heading,
    page_lines,
    prepare_page,
)
from .legend import rerank as legend_rerank
from .pipes import MLine, Separation, separate
from .primitives import PrimSet, bbox_dist, bbox_inside, bbox_union, estimate_unit, point_prim_dist, point_seg_dist
from .regions import Regions, detect_regions
from .symbols_match import (
    CandidateShape,
    MatchCache,
    MatchResult,
    TemplateLibrary,
    candidate_shape,
    coarse_scores,
    get_library,
    match_candidate,
    review_band,
)
from .text import ObjRef, PipeRef, assign_texts, drawing_info, medium_of_line, line_number_info
from .textlines import TextLine, build_lines
from .topology import Body, Network, Node, build_network

log = logging.getLogger(__name__)

BBox4 = tuple[float, float, float, float]


@dataclass
class ExtractOptions:
    accept_score: float = 0.62  # below: SymbolClass.UNKNOWN (alternatives kept)
    seed_units: float = 1.2  # straight open strokes longer than this (symbol units) are pipes
    group_units: float = 0.12  # proximity (symbol units) that still joins strokes into one candidate
    tap_max_units: float = 6.0  # solid runs to an instrument up to this length are instrument taps
    min_unknown_units: float = 0.3  # smaller unmatched stroke groups are dropped as noise
    contain_units: float = 4.5  # closed outlines up to this size (symbol units) hold their inner strokes
    boundary_symbols: int = 2  # an outline enclosing this many complete symbols is a boundary, not equipment
    decompose_budget: float = 1.0  # per candidate on the page (at least 100), for splitting candidates into
    # symbols bolted together (fine match = 1, coarse scoring = 0.2); measured use: <= 0.4 per candidate
    # on dense sheets, random clutter would use ~10x more. The search stops when the budget is spent.
    propagate_line_numbers: bool = True  # unlabelled runs take the nearest line number along the line
    keep_legend_symbols: bool = False
    learn_legends: bool = True  # extract_pdf: symbol templates learned from the document's legends


class IdAllocator:
    """Sequential ``o<n>``/``j<n>``/``p<n>``/``t<n>`` ids shared across pages."""

    def __init__(self) -> None:
        self.n = {"o": 0, "j": 0, "p": 0, "t": 0}

    def next(self, prefix: str) -> str:
        self.n[prefix] += 1
        return f"{prefix}{self.n[prefix]}"


@dataclass
class Detection:
    prim_ids: list[int]
    shape: CandidateShape
    results: list[MatchResult]
    cls: SymbolClass = SymbolClass.UNKNOWN
    score: float = 0.0
    bbox: BBox4 = (0.0, 0.0, 0.0, 0.0)
    body: BBox4 = (0.0, 0.0, 0.0, 0.0)
    internals: list[int] = field(default_factory=list)  # unmatched strokes drawn inside (tank heating coil ...)
    n_internal: int = 0  # number of absorbed inner stroke groups
    notes: list[str] = field(default_factory=list)  # why the confidence was lowered (calibration)

    @property
    def best(self) -> Optional[MatchResult]:
        return self.results[0] if self.results else None

    @property
    def size(self) -> float:
        return max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


# --------------------------------------------------------------------------- classification


def _rings_of(ps: PrimSet, prim_ids: Sequence[int]) -> list:
    """Rings all of whose primitives are among ``prim_ids``."""
    ids = set(prim_ids)
    rids = {ps.prims[i].ring for i in prim_ids if ps.prims[i].ring >= 0}
    out = []
    for rid in sorted(rids):
        r = ps.rings.get(rid)
        if r is not None and r.prim_ids and all(i in ids for i in r.prim_ids):
            out.append(r)
    return out


def _shape_for(ps: PrimSet, prim_ids: Sequence[int]) -> CandidateShape:
    return candidate_shape([ps.prims[i] for i in prim_ids], _rings_of(ps, prim_ids))


def _score(d: Detection) -> float:
    return d.best.score if d.best else 0.0


@dataclass
class _Matcher:
    """Matching context of one classification pass over a page."""

    ps: PrimSet
    lib: TemplateLibrary
    unit: float
    cache: MatchCache
    accept: float
    prior: bool = False  # size prior of stretchable equipment (the unit is anchored by matched valves)
    budget: float = 0.0  # remaining decomposition budget of the page (see ExtractOptions.decompose_budget)
    no_split: set = field(default_factory=set)  # shape keys whose decomposition found nothing
    no_peel: set = field(default_factory=set)  # shape keys of containment candidates that stay whole
    pipe_ends: Optional[np.ndarray] = None  # (n, 2) end points of pipe lines
    tol: float = 2.5  # snap tolerance (points)
    group_units: float = 0.12
    pipe_segs: Optional[np.ndarray] = None  # (n, 4) pipe lines (legend templates with stripped stubs check them)

    def detect(self, prim_ids: Sequence[int]) -> Detection:
        cs = _shape_for(self.ps, prim_ids)
        res = match_candidate(cs, self.lib, self.unit, cache=self.cache, size_prior=self.prior)
        if self.lib.n_extra:  # templates learned from the legend: prefer them, check stripped stubs
            res = legend_rerank(res, self.lib, self.pipe_segs, self.tol)
        return Detection(sorted(prim_ids), cs, res)

    def piped(self, prim_ids: Sequence[int]) -> bool:
        """A pipe line ends on these strokes (library symbols take pipes only at their ports
        on the outline, never on inner decoration: such strokes are piping or equipment)."""
        if self.pipe_ends is None or not len(self.pipe_ends):
            return False
        touch = max(0.6, 0.06 * self.unit)
        prims = [self.ps.prims[i] for i in prim_ids]
        bb = bbox_union(p.bbox for p in prims)
        e = self.pipe_ends
        sel = e[(e[:, 0] >= bb[0] - touch) & (e[:, 0] <= bb[2] + touch) & (e[:, 1] >= bb[1] - touch) & (e[:, 1] <= bb[3] + touch)]
        return any(point_prim_dist(float(x), float(y), p) <= touch for x, y in sel for p in prims)


def _finalise(d: Detection, accept: float) -> None:
    best = d.best
    if best is not None and best.score >= accept:
        d.cls = best.variant.cls
        d.score = best.score
        tb = [float(v) for v in best.bbox()]
        d.bbox = (min(tb[0], tb[2]), min(tb[1], tb[3]), max(tb[0], tb[2]), max(tb[1], tb[3]))
        bb = [float(v) for v in best.body_bbox()]
        d.body = (min(bb[0], bb[2]), min(bb[1], bb[3]), max(bb[0], bb[2]), max(bb[1], bb[3]))
    else:
        d.cls = SymbolClass.UNKNOWN
        d.score = best.score if best else 0.0
        d.bbox = d.shape.bbox
        d.body = d.shape.bbox


def classify_candidates(ps: PrimSet, cands: Sequence[Candidate], unit: float, opts: ExtractOptions,
                        lib: Optional[TemplateLibrary] = None, cache: Optional[MatchCache] = None,
                        size_prior: bool = False, pipes: Sequence[MLine] = (), tol: float = 2.5,
                        stats: Optional[dict] = None) -> list[Detection]:
    """Classify symbol candidates. ``size_prior``: the unit is anchored by matched valves,
    so stretchable equipment drawn implausibly small is penalised; ``pipes``: the pipe
    lines found by stroke separation; ``stats`` receives bookkeeping (budget use)."""
    ends = np.array([pt for ml in pipes for pt in ml.endpoints()], dtype=float).reshape(-1, 2)
    segs = np.array([(ml.x0, ml.y0, ml.x1, ml.y1) for ml in pipes], dtype=float).reshape(-1, 4)
    budget = max(100.0, opts.decompose_budget * len(cands))
    m = _Matcher(ps, lib or get_library(), unit, cache if cache is not None else MatchCache(), opts.accept_score,
                 size_prior, budget, pipe_ends=ends, tol=tol, group_units=opts.group_units, pipe_segs=segs)
    accept = m.accept
    dets: list[Detection] = []
    for c in cands:
        prims = [ps.prims[i] for i in c.prim_ids]
        if c.size < opts.min_unknown_units * unit and not any(p.is_core for p in prims):
            for p in prims:
                p.role = "noise"
            continue
        if _is_hatching(prims, unit):
            for p in prims:
                p.role = "hatch"
            continue
        d = m.detect(c.prim_ids)
        if len(c.parts) >= 2:
            peeled = _peel(m, c, d)
            if peeled is not None:
                dets.extend(peeled)
                continue
        if _score(d) < accept and len(c.clusters) >= 2:
            split = _try_split(m, c, d)
            if split is not None:
                dets.extend(split)
                continue
        if _worth_decomposing(m, d):
            parts = _decompose(m, d)
            if parts is not None:
                dets.extend(parts)
                continue
            m.no_split.add(d.shape.signature)
        dets.append(d)
    for d in dets:
        _finalise(d, accept)
    dets = _absorb_fragments(m, dets)
    _size_prior(dets, accept)
    dets = _concentric_bubbles(m.ps, dets)
    if stats is not None:
        stats["decompose_budget_exhausted"] = m.budget <= 0
        stats["decompose_budget_used"] = round(budget - m.budget, 1)
    return _resolve_containers(m, dets, opts)


def _concentric_bubbles(ps: PrimSet, dets: list[Detection]) -> list[Detection]:
    """A circle drawn concentrically inside an instrument bubble (a double-circle bubble of a
    yard convention) is part of that bubble, not a second instrument: it becomes the bubble's
    internals, and the bubble -- ink the template does not explain -- goes to review."""
    def circle_of(d: Detection):
        cs = [ps.prims[i] for i in d.prim_ids if ps.prims[i].kind == "circle" and not ps.prims[i].filled]
        return max(cs, key=lambda p: p.r) if cs else None

    hosts = [(d, circle_of(d)) for d in dets if d.cls != SymbolClass.UNKNOWN and category_of(d.cls) == Category.INSTRUMENT]
    hosts = [(d, c) for d, c in hosts if c is not None]
    if not hosts:
        return dets
    drop: set[int] = set()
    for d, c in sorted(hosts, key=lambda dc: -dc[1].r):
        if id(d) in drop:
            continue
        for o in dets:
            if o is d or id(o) in drop:
                continue
            prims = [ps.prims[i] for i in o.prim_ids]
            if not prims or any(p.kind != "circle" or p.filled for p in prims):
                continue
            if all(math.hypot(p.cx - c.cx, p.cy - c.cy) <= 0.08 * c.r and p.r < 0.97 * c.r for p in prims):
                drop.add(id(o))
                d.internals.extend(o.prim_ids)
                d.n_internal += 1
                d.score = min(d.score, review_band(d.score))
                d.notes.append("concentric inner circle (double-circle bubble) not in the template")
    return [d for d in dets if id(d) not in drop]


def _is_hatching(prims: Sequence, unit: float) -> bool:
    """A group dominated by many parallel strokes at different offsets (section /
    insulation hatching, shaded areas): not a symbol."""
    segs = [p for p in prims if p.kind == "seg" and not p.curve]
    if len(segs) < 8:
        return False
    total = sum(p.length for p in prims)
    groups: dict[int, list] = defaultdict(list)
    for p in segs:
        groups[int(round(math.degrees(math.atan2(p.y1 - p.y0, p.x1 - p.x0)) % 180.0 / 3.0)) % 60].append(p)
    for ang, grp in groups.items():
        if len(grp) < 8:
            continue
        a = math.radians(ang * 3.0)
        nx, ny = -math.sin(a), math.cos(a)  # normal of the stroke direction
        offsets = {int(round((p.x0 * nx + p.y0 * ny) / max(0.04 * unit, 0.3))) for p in grp}
        if len(offsets) >= 8 and sum(p.length for p in grp) >= 0.5 * total:
            return True
    return False


def _peel(m: _Matcher, c: Candidate, whole: Detection) -> Optional[list[Detection]]:
    """A closed outline that took in the strokes inside it (``c.parts[0]`` the outline's
    group, the rest the inner groups): keep the union when it is the better symbol, else
    match the outline and its contents separately (a tank with a level switch or a
    heating coil inside; an instrument drawn inside a package outline)."""
    sw = _score(whole)
    piped = any(m.piped(g) for g in c.parts[1:])
    key = whole.shape.signature
    if not piped and (sw >= 0.97 or key in m.no_peel):
        return None  # a confident union: no split can beat it by the required margin
    outer = m.detect(c.parts[0])
    so = _score(outer)
    if so >= m.accept:
        inners = [m.detect(g) for g in c.parts[1:]]
        if outer.best.variant.stretchable and (sw < m.accept or piped):
            # no symbol as a whole, or piping inside the outline (a heating coil fed through the tank top):
            # unmatched contents become the outline's internals later
            return [outer] + inners
        if all(_score(d) >= m.accept for d in inners) and min([so] + [_score(d) for d in inners]) >= sw + 0.03:
            return [outer] + inners
    if not piped:
        m.no_peel.add(key)
    return None


def _worth_decomposing(m: _Matcher, d: Detection) -> bool:
    """Try to split a candidate into symbols bolted flange-to-flange? Not for a confident
    match, a shape already known not to split, or an accepted rigid symbol at the drawing's
    symbol scale (the expensive search is reserved for poor matches and oversized ones)."""
    s = _score(d)
    if s >= 0.93 or m.budget <= 0 or d.shape.signature in m.no_split:
        return False
    if s >= m.accept:
        b = d.best
        if not b.variant.stretchable and 0.7 * m.unit <= b.scale <= 1.45 * m.unit:
            return False
    return True


def _try_split(m: _Matcher, c: Candidate, whole: Detection) -> Optional[list[Detection]]:
    """Split a poorly matching group into its touch clusters (two symbols drawn
    flange-to-flange); small clusters go with their nearest big neighbour."""
    prims, unit = m.ps.prims, m.unit
    infos = []
    for cl in c.clusters:
        bb = bbox_union(prims[i].bbox for i in cl)
        infos.append((cl, bb, max(bb[2] - bb[0], bb[3] - bb[1])))
    big = [x for x in infos if x[2] >= 0.45 * unit]
    small = [x for x in infos if x[2] < 0.45 * unit]
    if len(big) < 2:
        return None
    groups = [list(b[0]) for b in big]
    for cl, bb, _ in small:
        k = min(range(len(big)), key=lambda j: bbox_dist(bb, big[j][1]))
        groups[k].extend(cl)
    parts = [m.detect(g) for g in groups]
    scores = [_score(p) for p in parts]
    if min(scores) >= 0.9 * m.accept and sum(scores) / len(scores) > _score(whole) + 0.05:
        return parts
    return None


def _decompose(m: _Matcher, whole: Detection) -> Optional[list[Detection]]:
    """Symbols bolted flange-to-flange form one touching candidate. The candidate is
    cut into 'atoms' (closed outlines with their contents, other strokes); cuts
    across either axis that do not pass through a closed outline are ranked by
    coarse score, the best are fine-matched, and a split is kept when the
    point-weighted score improves by a margin. Parts are split recursively
    (chains of three or more symbols) within a small match budget."""
    ps, unit, accept = m.ps, m.unit, m.accept
    prims = ps.prims
    atoms: list[tuple[list[int], tuple[float, float, float, float], bool]] = []  # (prim ids, bbox, solid)
    in_ring: set[int] = set()
    for ring in _rings_of(ps, whole.prim_ids):
        if ring.prim_ids[0] in in_ring:
            continue
        atoms.append((list(ring.prim_ids), ring.bbox, True))
        in_ring.update(ring.prim_ids)
    for i in whole.prim_ids:
        if i not in in_ring:
            p = prims[i]
            atoms.append(([i], p.bbox, p.kind == "circle" and not p.filled and p.r >= 0.12 * unit))
    # strokes well inside a closed outline can never be cut away from it: merge them in
    mg = 0.06 * unit
    solid_idx = [k for k, a in enumerate(atoms) if a[2]]
    absorbed: set[int] = set()
    for k, a in enumerate(atoms):
        if a[2]:
            continue
        cx, cy = (a[1][0] + a[1][2]) / 2, (a[1][1] + a[1][3]) / 2
        for s_ in solid_idx:
            b = atoms[s_][1]
            if b[0] + mg < cx < b[2] - mg and b[1] + mg < cy < b[3] - mg:
                atoms[s_][0].extend(a[0])
                absorbed.add(k)
                break
    atoms = [a for k, a in enumerate(atoms) if k not in absorbed]
    if len(atoms) < 2 or len(atoms) > 80:
        return None

    coarse_memo: dict[tuple[int, ...], tuple[float, float]] = {}

    def coarse(group: list[int]) -> tuple[float, float]:
        key = tuple(sorted(group))
        if key not in coarse_memo:
            m.budget -= 0.2
            cs = _shape_for(ps, [i for a in group for i in atoms[a][0]])
            sc, _, _ = coarse_scores(m.lib, cs, unit, size_prior=m.prior)
            coarse_memo[key] = (float(sc.max()) if len(sc) else 0.0, float(len(cs.pts)))
        return coarse_memo[key]

    def partitions(group: list[int], axis: int) -> list[tuple[list[int], list[int]]]:
        centre = {a: (atoms[a][1][axis] + atoms[a][1][axis + 2]) / 2 for a in group}
        solid = [atoms[a][1] for a in group if atoms[a][2]]
        cuts = sorted({v for a in group for v in (atoms[a][1][axis], atoms[a][1][axis + 2], centre[a])})
        eps = 0.02 * unit
        out, seen = [], set()
        for c in cuts:
            if any(b[axis] + 0.05 * unit < c < b[axis + 2] - 0.05 * unit for b in solid):
                continue  # never cut through a closed outline
            on_cut = [a for a in group if abs(centre[a] - c) <= eps]
            options = [set()]
            if len(on_cut) >= 2:  # a shared flange face drawn twice: one copy for each symbol
                options.append(set(on_cut[1::2]))
            for to_right in options:
                left = tuple(sorted(a for a in group if centre[a] <= c + eps and a not in to_right))
                right = tuple(sorted(a for a in group if centre[a] > c + eps or a in to_right))
                if left and right and left not in seen:
                    seen.add(left)
                    out.append((list(left), list(right)))
        return out

    def weighted(parts: list[Detection]) -> float:
        w = [float(len(p.shape.pts)) for p in parts]
        return sum(_score(p) * wi for p, wi in zip(parts, w)) / max(sum(w), 1e-9)

    budget = [24]  # match calls allowed for this candidate

    def detect(side: list[int]) -> Detection:
        budget[0] -= 1
        m.budget -= 1
        return m.detect([i for a in side for i in atoms[a][0]])

    def splittable(side: list[int], d: Detection) -> bool:
        """Worth splitting further: poorly matched and large enough for two symbols."""
        if _score(d) >= 0.85 or budget[0] <= 0 or sum(1 for a in side if atoms[a][2]) < 1:
            return False
        b = bbox_union(atoms[a][1] for a in side)
        return max(b[2] - b[0], b[3] - b[1]) >= 1.1 * unit

    def split_group(group: list[int], current: Detection, depth: int) -> list[Detection]:
        """Best decomposition of ``group`` (``[current]`` when no split helps)."""
        if len(group) < 2 or depth > 3 or budget[0] <= 0 or m.budget <= 0:
            return [current]
        gb = bbox_union(atoms[a][1] for a in group)
        main = 0 if (gb[2] - gb[0]) >= (gb[3] - gb[1]) else 1
        best_parts, best_val = [current], _score(current) + 0.06
        for axis in (main, 1 - main):
            ranked = []
            for left, right in partitions(group, axis):
                (sa, wa), (sb, wb) = coarse(left), coarse(right)
                ranked.append(((sa * wa + sb * wb) / max(wa + wb, 1e-9), left, right))
            ranked.sort(key=lambda r: -r[0])
            for _, left, right in ranked[: (3 if depth == 0 else 2)]:
                if budget[0] <= 1 or m.budget <= 0:
                    break
                leaves = []
                for side in (left, right):
                    d = detect(side)
                    # a part that is itself several symbols is split further (chains of three or more)
                    leaves.extend(split_group(side, d, depth + 1) if splittable(side, d) else [d])
                if min(_score(p) for p in leaves) < accept:
                    continue
                val = weighted(leaves)
                if val > best_val:
                    best_parts, best_val = leaves, val
            if len(best_parts) > 1:
                break
        return best_parts

    parts = split_group(list(range(len(atoms))), whole, 0)
    return parts if len(parts) > 1 else None


def _absorb_fragments(m: _Matcher, dets: list[Detection]) -> list[Detection]:
    """Merge an unmatched small fragment into a neighbour when the union matches
    better (e.g. an actuator drawn with a gap to its valve body)."""
    unit, accept = m.unit, m.accept
    out = list(dets)
    changed = True
    rounds = 0
    while changed and rounds < 3:
        changed = False
        rounds += 1
        unknown = [d for d in out if d.cls == SymbolClass.UNKNOWN and d.size <= 1.5 * unit]
        for d in unknown:
            if d not in out:
                continue
            best = None
            for o in out:
                if o is d or bbox_dist(o.shape.bbox, d.shape.bbox) > 0.4 * unit:
                    continue
                mm = m.detect(o.prim_ids + d.prim_ids)
                _finalise(mm, accept)
                if mm.cls == SymbolClass.UNKNOWN or mm.score < max(o.score - 0.03, accept):
                    continue
                if best is None or mm.score > best[0].score:
                    best = (mm, o)
            if best is not None:
                mm, o = best
                out.remove(o)
                out.remove(d)
                out.append(mm)
                changed = True
    return out


def _size_prior(dets: list[Detection], accept: float) -> None:
    """Prefer alternatives whose scale agrees with the drawing's symbol scale when
    two classes score almost equally."""
    scales = [d.best.scale for d in dets if d.cls != SymbolClass.UNKNOWN and d.best and not d.best.variant.stretchable]
    if len(scales) < 5:
        return
    ref = float(np.median(scales))
    for d in dets:
        if len(d.results) < 2:
            continue
        def adj(r: MatchResult) -> float:
            if r.variant.stretchable:
                return r.score
            z = math.log(max(r.scale, 1e-6) / ref)
            return r.score * (0.9 + 0.1 * math.exp(-z * z / (2 * 0.6 ** 2)))
        ranked = sorted(d.results, key=lambda r: -adj(r))
        top = d.results[0]
        if is_learned_variant(top.variant) and top.score >= ranked[0].score - LEGEND_TIE:
            continue  # the drawing's legend decided between look-alikes (its templates have their own size)
        if ranked[0].variant.cls != top.variant.cls and ranked[0].score >= top.score - 0.03:
            # switch class only; keep the canonical orientation chosen by the matcher
            winner = ranked[0].variant.cls
            d.results = [r for r in d.results if r.variant.cls == winner] + [r for r in d.results if r.variant.cls != winner]
            _finalise(d, accept)


def _inside(inner: BBox4, outer: BBox4, margin: float) -> bool:
    return (inner[0] >= outer[0] + margin and inner[1] >= outer[1] + margin and inner[2] <= outer[2] - margin
            and inner[3] <= outer[3] - margin)


def _free_boundary(m: _Matcher, d: Detection) -> list[Detection]:
    """Drop the outline of a boundary detection (its largest ring) and classify whatever
    else was grouped with it (symbols touching the outline) on its own."""
    rings = _rings_of(m.ps, d.prim_ids)
    outline = max(rings, key=lambda r: r.polygon.area) if rings else None
    ring_ids = set(outline.prim_ids) if outline is not None else set(d.prim_ids)
    for i in ring_ids:
        m.ps.prims[i].role = "boundary"
    rest = [i for i in d.prim_ids if i not in ring_ids]
    out: list[Detection] = []
    for c in group_candidates(m.ps, rest, m.unit, m.tol, m.group_units):
        prims = [m.ps.prims[i] for i in c.prim_ids]
        if c.size < 0.3 * m.unit and not any(p.is_core for p in prims):
            continue
        nd = m.detect(c.prim_ids)
        _finalise(nd, m.accept)
        out.append(nd)
    return out


def _resolve_containers(m: _Matcher, dets: list[Detection], opts: ExtractOptions) -> list[Detection]:
    """Large closed outlines and what is drawn inside them.

    * An outline (unmatched, or matched as box-like stretchable equipment) enclosing
      ``opts.boundary_symbols`` or more complete symbols (valves, fittings, equipment) is
      a package / skid boundary or a hull or plan outline drawn behind the diagram: it is
      dropped (its strokes become ``boundary``) and the symbols inside stand on their own.
    * Unmatched strokes inside accepted box-like equipment are its internals (heating
      coil, baffles): when outline + contents match a library symbol (a large sea chest
      or BWTS unit drawn as a box with its insert), that symbol is taken, otherwise the
      contents are kept as the equipment's internals.
    """
    unit, accept = m.unit, m.accept
    margin = 0.02 * unit

    def boxy(d: Detection) -> bool:
        if d.cls == SymbolClass.UNKNOWN:
            return bool(_rings_of(m.ps, d.prim_ids)) and d.size >= 1.5 * unit
        return get_symbol(d.cls).stretchable

    solid = [d for d in dets if d.cls != SymbolClass.UNKNOWN and category_of(d.cls) != Category.INSTRUMENT]
    drop: set[int] = set()
    freed: list[Detection] = []
    for d in sorted((d for d in dets if boxy(d)), key=lambda d: -d.size):
        n_inside = sum(1 for o in solid if o is not d and id(o) not in drop and _inside(o.bbox, d.body, margin))
        if n_inside >= opts.boundary_symbols:
            drop.add(id(d))
            freed.extend(_free_boundary(m, d))
    out = [d for d in dets if id(d) not in drop] + freed
    taken: set[int] = set()
    result: list[Detection] = []
    for d in sorted(out, key=lambda d: -d.size):
        if id(d) in taken:
            continue
        if d.cls == SymbolClass.UNKNOWN or not get_symbol(d.cls).stretchable:
            continue
        inner = [o for o in out if o is not d and id(o) not in taken and o.cls == SymbolClass.UNKNOWN
                 and _inside(o.shape.bbox, d.body, margin)]
        if not inner:
            continue
        for o in inner:
            taken.add(id(o))
        union = None
        if not any(m.piped(o.prim_ids) for o in inner):  # piping inside is never part of a symbol
            union = m.detect(d.prim_ids + [i for o in inner for i in o.prim_ids])
            _finalise(union, accept)
        # only box-like equipment is re-joined here (symbol-size outlines were merged during grouping)
        if union is not None and union.cls not in (SymbolClass.UNKNOWN, SymbolClass.TANK) and union.score >= accept \
                and union.best.variant.stretchable:
            taken.add(id(d))
            result.append(union)
        else:
            d.internals.extend(i for o in inner for i in o.prim_ids)
            d.n_internal += len(inner)
    result.extend(d for d in out if id(d) not in taken)
    return result


def _straight_unknowns_to_pipes(ps: PrimSet, dets: list[Detection], unit: float, tol: float, start_id: int) -> tuple[list[Detection], list[MLine]]:
    """Unmatched candidates that are just one long straight (diagonal) stroke are pipes."""
    keep: list[Detection] = []
    lines: list[MLine] = []
    for d in dets:
        prims = [ps.prims[i] for i in d.prim_ids]
        if d.cls != SymbolClass.UNKNOWN or any(p.kind != "seg" or p.closed or p.curve for p in prims):
            keep.append(d)
            continue
        pts = [pt for p in prims for pt in p.endpoints()]
        a, b = max(((p, q) for p in pts for q in pts), key=lambda pq: math.dist(*pq))
        L = math.dist(a, b)
        if L < 2.0 * unit or any(point_seg_dist(x, y, a[0], a[1], b[0], b[1]) > tol for x, y in pts):
            keep.append(d)
            continue
        m = MLine(start_id + len(lines), "d", a[0], a[1], b[0], b[1], list(d.prim_ids), all(p.dashed for p in prims),
                  max(p.width for p in prims), prims[0].color, [], True)
        lines.append(m)
        for p in prims:
            p.role = "pipe"
    return keep, lines


# --------------------------------------------------------------------------- page


def _medium_from_color(color) -> Medium:
    if color is None:
        return Medium.UNKNOWN
    r, g, b = color
    if max(r, g, b) - min(r, g, b) < 0.15:  # black / grey: the default line colour, no information
        return Medium.UNKNOWN
    best, bd = Medium.UNKNOWN, 0.2
    for med in MEDIUM_COLORS:
        if med == Medium.UNKNOWN:
            continue
        mr = medium_rgb(med)
        if max(mr) - min(mr) < 0.15:
            continue
        d = math.dist(color, mr)
        if d < bd:
            best, bd = med, d
    return best


def _ports_dir(d: tuple[float, float]) -> Point:
    return Point(x=round(d[0], 4), y=round(d[1], 4))


@dataclass
class _Stage:
    ps: PrimSet
    regions: Regions
    sep: Separation
    cands: list[Candidate]
    dets: list[Detection]
    diag_lines: list[MLine]
    unit: float
    anchored: bool  # the unit comes from matched valve-like symbols (not from the stroke-cluster median)
    stats: dict = field(default_factory=dict)


def _analyse(pc: PageContent, lines: list[TextLine], cfg: Config, opts: ExtractOptions, lib: TemplateLibrary,
             cache: MatchCache, unit: Optional[float], timing: dict[str, float], anchored: bool = False,
             legend_boxes: Sequence[BBox4] = (), geometry: Optional[PageGeometry] = None) -> _Stage:
    """Regions, stroke separation and symbol classification for one symbol unit. ``legend_boxes``:
    legend regions found by :func:`shippid.extract.legend.find_legends` (also headings such as
    '범례' the region finder does not know) -- excluded like the region finder's legends;
    ``geometry``: the page as prepared for legend finding (the same first steps, reused once)."""
    tol = cfg.snap_tol
    t = time.perf_counter()
    if geometry is not None and unit is None and geometry.lines is lines and geometry.tol == tol:
        ps, unit0, regions = geometry.ps, geometry.unit, geometry.regions
    else:
        ps = PrimSet.from_page(pc)
        _mark_text_strokes(ps, lines)
        unit0 = unit or estimate_unit(ps.prims, [p.id for p in ps.prims if p.role != "text"], max(0.6, 0.5 * tol))
        regions = detect_regions(ps, lines, pc.width, pc.height, tol, unit0)
    for box in legend_boxes:
        if not any(bbox_inside(box, b, tol) for b in regions.legends):
            regions.legends.append(tuple(box))
            for p in ps.prims:
                if p.role == "" and bbox_inside(p.bbox, box, tol):
                    p.role = "excluded"
    if unit is None:
        free_ids = [p.id for p in ps.prims if p.role == ""]
        unit = estimate_unit(ps.prims, free_ids, max(0.6, 0.5 * tol), default=unit0)
        probed = _probe_unit(ps, free_ids, unit, tol, lib, cache)
        if probed is not None:
            unit, anchored = probed, True
    timing["regions"] = timing.get("regions", 0.0) + time.perf_counter() - t

    t = time.perf_counter()
    sep = separate(ps, unit, tol, opts.seed_units, opts.contain_units)
    timing["separate"] = timing.get("separate", 0.0) + time.perf_counter() - t

    t = time.perf_counter()
    cands = group_candidates(ps, sep.symbol_ids, unit, tol, opts.group_units, opts.contain_units)
    timing["symbols"] = timing.get("symbols", 0.0) + time.perf_counter() - t
    return _classify_stage(_Stage(ps, regions, sep, cands, [], [], unit, anchored), cfg, opts, lib, cache, timing)


def _mark_text_strokes(ps: PrimSet, lines: Sequence[TextLine]) -> int:
    """Stroke glyphs of SHX text: primitives lying inside a text whose string comes from
    invisible text or an "AutoCAD SHX Text" annotation (:attr:`TextLine.geometric`) are
    lettering, never symbols or pipes (an 'O' would pass for an instrument bubble, an 'I'
    for a pipe stub). Returns the number of primitives marked ``text``."""
    geo = [t for t in lines if t.geometric]
    if not geo or not ps.prims:
        return 0
    bb = ps.bbox_array(range(len(ps.prims)))
    width = np.array([p.width for p in ps.prims])
    free = np.array([p.role == "" for p in ps.prims])
    n = 0
    for t in geo:
        pad = 0.15 * max(t.size, 1.0) + 0.5 * width
        x0, y0, x1, y1 = t.bbox
        sel = free & (bb[:, 0] >= x0 - pad) & (bb[:, 1] >= y0 - pad) & (bb[:, 2] <= x1 + pad) & (bb[:, 3] <= y1 + pad)
        for i in np.nonzero(sel)[0].tolist():
            ps.prims[i].role = "text"
            n += 1
        free &= ~sel
    return n


OFF_RADIUS_DEG = 35.0  # a pipe arriving at a bubble more than this off the radial direction is not a tap


def _off_radius_attachments(ps: PrimSet, d: Detection, ports) -> int:
    """Pipes that meet an instrument circle off its radius. Instrument connections and signal
    lines run to a bubble's centre; a line touching the circle tangentially or at a slant is
    drawing structure the bubble template does not explain (a centrifugal pump drawn as a
    circle with a tangential discharge stub, which merges into the discharge pipe)."""
    circ = [ps.prims[i] for i in d.prim_ids if ps.prims[i].kind == "circle"]
    if not circ:
        return 0
    c = max(circ, key=lambda p: p.r)
    lim = math.cos(math.radians(OFF_RADIUS_DEG))
    n = 0
    for _, (x, y), (dx, dy) in ports:
        rx, ry = x - c.cx, y - c.cy
        dist = math.hypot(rx, ry)
        dn = math.hypot(dx, dy)
        if dist < 1e-6 or dn < 1e-9 or abs(dist - c.r) > max(0.15 * c.r, 1.0):
            continue  # not on the circle (e.g. on the square of a shared-display symbol)
        if abs(rx * dx + ry * dy) / (dist * dn) < lim:
            n += 1
    return n


def _classify_stage(st: _Stage, cfg: Config, opts: ExtractOptions, lib: TemplateLibrary, cache: MatchCache,
                    timing: dict[str, float]) -> _Stage:
    """Classify the stage's candidates (again, e.g. once the unit is anchored)."""
    t = time.perf_counter()
    tol = cfg.snap_tol
    stats: dict = {}
    dets = classify_candidates(st.ps, st.cands, st.unit, opts, lib, cache, size_prior=st.anchored,
                               pipes=st.sep.pipe_lines, tol=tol, stats=stats)
    dets, diag_lines = _straight_unknowns_to_pipes(st.ps, dets, st.unit, tol, len(st.sep.lines))
    if not opts.keep_legend_symbols:
        dets = [d for d in dets if st.regions.kind_at(d.bbox, tol) is None]
    dets.sort(key=lambda d: (round(d.bbox[1], 1), round(d.bbox[0], 1)))
    timing["symbols"] = timing.get("symbols", 0.0) + time.perf_counter() - t
    return _Stage(st.ps, st.regions, st.sep, st.cands, dets, diag_lines, st.unit, st.anchored, stats)


def _probe_unit(ps: PrimSet, ids: Sequence[int], unit: float, tol: float, lib: TemplateLibrary,
                cache: MatchCache, top: int = 12) -> Optional[float]:
    """Symbol unit from the most frequent symbol-only stroke clusters (valve bodies:
    bow-ties, discs) matched against the library; None when nothing valve-like repeats."""
    from collections import Counter

    from .primitives import DSU, prim_distance
    from .symbols_match import _prim_signature

    prims = ps.prims
    touch = max(0.6, 0.5 * tol)
    core = [i for i in ids if prims[i].is_core and not prims[i].dashed]
    if len(core) < 6:
        return None
    dsu = DSU(len(prims))
    for i, j in ps.pairs_within(core, touch):
        if prim_distance(prims[i], prims[j]) <= touch:
            dsu.union(i, j)
    groups: dict[bytes, list[list[int]]] = {}
    for members in dsu.groups(core).values():
        bb = bbox_union(prims[m].bbox for m in members)
        D = max(bb[2] - bb[0], bb[3] - bb[1])
        if not (0.4 * unit <= D <= 3.0 * unit):
            continue
        mset = set(members)
        rings = [r for r in ps.rings.values() if r.prim_ids and all(i in mset for i in r.prim_ids)]
        groups.setdefault(_prim_signature([prims[m] for m in members], rings, bb, D), []).append(members)
    counts = Counter({k: len(v) for k, v in groups.items() if len(v) >= 2})
    scales: list[float] = []
    n_sig = 0
    for key, n in counts.most_common(top):
        res = match_candidate(_shape_for(ps, groups[key][0]), lib, None, cache=cache)
        if res and res[0].shape_score >= 0.7 and res[0].variant.spec.category == Category.VALVE:
            scales.extend([res[0].scale] * n)
            n_sig += 1
            if n_sig >= 3 and len(scales) >= 8:
                break  # enough evidence: the most frequent valve drawings agree on the unit
    if len(scales) < 3:
        return None
    return float(np.median(scales))


def _valve_unit(dets: Sequence[Detection], min_n: int = 3) -> Optional[float]:
    """Median matched scale of confidently classified valves (else other rigid symbols)."""
    # the shape fit (before the unexplained-ink penalty): a yard-style valve still gives the drawing's scale
    valves = [d.best.scale for d in dets if d.best is not None and d.cls != SymbolClass.UNKNOWN
              and d.best.shape_score >= 0.85 and d.best.variant.spec.category == Category.VALVE]
    if len(valves) >= min_n:
        return float(np.median(valves))
    rigid = [d.best.scale for d in dets if d.best is not None and d.cls != SymbolClass.UNKNOWN
             and d.best.shape_score >= 0.85 and not d.best.variant.stretchable
             and d.best.variant.spec.category != Category.EQUIPMENT]
    if len(rigid) >= min_n:
        return float(np.median(rigid))
    return None


def extract_page(page, page_index: int = 0, config: Optional[Config] = None, ids: Optional[IdAllocator] = None,
                 options: Optional[ExtractOptions] = None, cache: Optional[MatchCache] = None,
                 content: Optional[PageContent] = None, legend: Optional[LegendBook] = None,
                 geometry: Optional[PageGeometry] = None) -> PIDDocument:
    """Extract one PyMuPDF page into a single-page :class:`PIDDocument`. ``legend``: what was learned
    from the document's legends (:class:`shippid.extract.legend.LegendBook`); its templates are
    matched alongside the built-in library (a ``cache`` must belong to that combined library).
    ``geometry``: this page as prepared by :func:`shippid.extract.legend.prepare_page` (reused,
    not recomputed; it must not be used for anything else afterwards)."""
    cfg = config or Config()
    opts = options or ExtractOptions()
    ids = ids or IdAllocator()
    cache = cache if cache is not None else MatchCache(cfg.symbol_max_size)
    timing: dict[str, float] = {}
    t0 = time.perf_counter()

    if geometry is not None and content is not None and geometry.pc is not content:
        geometry = None  # prepared from another reading of the page
    pc = content or (geometry.pc if geometry is not None else None) or read_page(page)
    timing["read"] = time.perf_counter() - t0
    doc = PIDDocument(source=str(getattr(getattr(page, "parent", None), "name", "") or ""))
    doc.pages.append(PageInfo(index=page_index, width=pc.width, height=pc.height, rotation=pc.rotation))
    page_meta: dict = {"image_coverage": round(pc.image_coverage, 4), "layers": pc.layers, "readable": pc.readable,
                       "empty": pc.empty, "vector_paths": pc.n_paths, "chars": pc.n_chars,
                       "hidden_chars": pc.n_hidden_chars, "clipped_paths": pc.n_clipped_paths}
    doc.meta["pages"] = {str(page_index): page_meta}
    doc.meta["extractor"] = "vector"
    if pc.empty:
        log.info("page %d is empty; skipped", page_index)
        doc.meta["empty_pages"] = [page_index]
        return doc
    if not pc.readable:
        log.warning("page %d looks scanned/raster-only (image coverage %.2f, %d vector paths, %d visible / %d hidden "
                    "chars); skipped", page_index, pc.image_coverage, pc.n_paths, pc.n_chars, pc.n_hidden_chars)
        doc.meta["unreadable_pages"] = [page_index]
        return doc

    tol = cfg.snap_tol
    lib = (legend.library if legend is not None else None) or get_library()
    leg_boxes = legend.regions(page_index) if legend is not None else []
    if geometry is not None:
        lines = geometry.lines
    else:
        rules = [(s.x0, s.y0, s.x1, s.y1) for s in pc.segments if s.curve < 0 and s.length >= 2.0]
        lines = build_lines(pc.words, separators=rules)
    page_meta["annotation_chars"] = pc.n_annot_chars
    page_meta["tessellated_arcs"] = pc.n_arcs
    page_meta["text_sources"] = {s: sum(1 for t in lines if t.source == s) for s in ("text", "hidden", "annotation")}
    st = _analyse(pc, lines, cfg, opts, lib, cache, None, timing, legend_boxes=leg_boxes, geometry=geometry)
    ref = _valve_unit(st.dets)
    if ref is not None and abs(ref / st.unit - 1.0) > 0.15:
        # the first estimate (median size of symbol-only stroke clusters) is biased by the mix of
        # symbols on the sheet; matched valve scales are the reliable symbol unit -> redo with it
        log.debug("page %d: symbol unit %.2f -> %.2f (from matched valves)", page_index, st.unit, ref)
        st = _analyse(pc, lines, cfg, opts, lib, cache, ref, timing, anchored=True, legend_boxes=leg_boxes)
    elif not st.anchored:
        # too few repeated valve drawings to probe the unit: one or two confidently matched
        # valves still anchor it, and the size prior of stretchable equipment then applies
        few = _valve_unit(st.dets, min_n=1)
        if few is not None and abs(few / st.unit - 1.0) <= 0.15:  # same unit: only the classification changes
            st = _classify_stage(dataclasses.replace(st, anchored=True), cfg, opts, lib, cache, timing)
        elif few is not None:
            st = _analyse(pc, lines, cfg, opts, lib, cache, few, timing, anchored=True, legend_boxes=leg_boxes)
    ps, regions, sep, cands, dets, diag_lines, unit = st.ps, st.regions, st.sep, st.cands, st.dets, st.diag_lines, st.unit

    # ---- network
    t = time.perf_counter()
    bodies = []
    for k, d in enumerate(dets):
        spec = get_symbol(d.cls) if d.cls != SymbolClass.UNKNOWN else None
        ports = d.best.ports() if (d.best is not None and spec is not None) else []
        bodies.append(Body(k, d.bbox, d.body, [ps.prims[i] for i in d.prim_ids + d.internals], ports,
                           boundary_ports=(spec is None or spec.boundary_ports),
                           is_valve=spec is not None and spec.category == Category.VALVE))
    pipe_lines = sep.pipe_lines + diag_lines
    net = build_network(pipe_lines, bodies, sep.dots, unit, tol)
    if legend is not None:  # a learned stub merged into its pipe: the attachment takes the stub's port name
        name_stub_ports(net, [(k, d.best) for k, d in enumerate(dets) if d.cls != SymbolClass.UNKNOWN and d.best], tol)
    review: dict[int, list[str]] = defaultdict(list)  # calibration notes per detection (confidence lowered)
    for k, d in enumerate(dets):
        b = d.best
        if b is not None and d.cls != SymbolClass.UNKNOWN and b.raw is not None:
            review[k].append(f"unexplained ink {b.unexplained:.2f}, missing template ink {b.missing:.2f}")
        review[k].extend(d.notes)
        if d.cls != SymbolClass.UNKNOWN and category_of(d.cls) == Category.INSTRUMENT:
            n_off = _off_radius_attachments(ps, d, net.ports.get(k, []))
            if n_off:
                d.score = min(d.score, review_band(d.score))
                review[k].append(f"{n_off} pipe(s) meet the circle off its radius (not an instrument connection)")
    timing["network"] = time.perf_counter() - t

    # ---- objects
    t = time.perf_counter()
    obj_ids = [ids.next("o") for _ in dets]
    objects: list[PIDObject] = []
    for k, d in enumerate(dets):
        best = d.best
        alt: list[dict] = []
        seen: set = set()
        for r in d.results:
            if r.variant.cls in seen:
                continue
            seen.add(r.variant.cls)
            alt.append({"cls": r.variant.cls.value, "score": round(r.score, 3), "precision": round(r.precision, 3),
                        "recall": round(r.recall, 3), "unexplained": round(r.unexplained, 3),
                        "missing": round(r.missing, 3)})
            if len(alt) == 3:
                break
        accepted = d.cls != SymbolClass.UNKNOWN
        guess_cls = best.variant.cls if best is not None else SymbolClass.UNKNOWN
        category = category_of(d.cls if accepted else guess_cls)
        prims = [ps.prims[i] for i in d.prim_ids]
        attrs: dict = {
            "candidates": alt,
            "signature": {
                "segments": sum(1 for p in prims if p.kind == "seg" and not p.curve),
                "curves": len({(p.path, p.sub) for p in prims if p.curve}),
                "circles": sum(1 for p in prims if p.kind == "circle"),
                "filled": len({(p.path, p.sub) for p in prims if p.filled}),
            },
        }
        if best is not None:
            attrs["scale"] = round(best.scale, 3)
        if d.n_internal:
            attrs["internals"] = d.n_internal  # unmatched stroke groups drawn inside (heating coil, baffles ...)
        if accepted and best is not None and best.variant.stretchable:
            attrs["stretch"] = [round(best.sx / best.scale, 3), round(best.sy / best.scale, 3)]
        if review.get(k):
            attrs["calibration"] = review[k]  # why the confidence is below the review threshold
        actuator = best.variant.actuator if (accepted and best and category == Category.VALVE) else ActuatorType.NONE
        if legend is not None and best is not None:
            if accepted:
                tpl = legend.template_for(best.variant)
                if tpl:
                    attrs["template"] = tpl  # learned from / confirmed by the drawing's legend
                if category == Category.VALVE:
                    actuator = legend.actuator_for(best.variant, actuator)
            else:
                hit = legend.describe_unknown(d.shape)
                if hit:
                    attrs["legend"] = hit  # drawn like a legend row whose description names no known class
        obj = PIDObject(
            id=obj_ids[k],
            cls=d.cls,
            category=category,
            bbox=BBox(x0=d.bbox[0], y0=d.bbox[1], x1=d.bbox[2], y1=d.bbox[3]),
            page=page_index,
            rotation=best.variant.rotation if (accepted and best) else 0,
            mirrored=bool(best.variant.mirrored) if (accepted and best) else False,
            actuator=actuator,
            ports=[Port(name=n, position=Point(x=x, y=y), direction=_ports_dir(dv)) for n, (x, y), dv in net.ports.get(k, [])],
            attributes=attrs,
            confidence=round(float(min(1.0, max(0.0, d.score))), 4),
            source="vector",
        )
        objects.append(obj)

    # ---- junctions and pipe runs
    junc_ids: dict[int, str] = {}
    order = sorted(net.junctions, key=lambda j: (round(j.y, 1), round(j.x, 1)))
    junctions = []
    for j in order:
        jid = ids.next("j")
        junc_ids[j.id] = jid
        junctions.append(Junction(id=jid, position=Point(x=j.x, y=j.y), page=page_index, kind=j.kind, dotted=j.dotted))

    def endpoint(n: Node) -> Endpoint:
        if n.body is not None:
            return Endpoint(object_id=obj_ids[n.body], port=n.port)
        if n.junction is not None:
            return Endpoint(junction_id=junc_ids[n.junction])
        return Endpoint()

    runs = sorted(net.runs, key=lambda r: (round(min(p[1] for p in r.points), 1), round(min(p[0] for p in r.points), 1)))
    pipes: list[PipeRun] = []
    for r in runs:
        start, end = endpoint(r.start), endpoint(r.end)
        length = sum(math.dist(a, b) for a, b in zip(r.points, r.points[1:]))
        kind = PipeKind.SIGNAL if r.dashed else PipeKind.PROCESS
        if kind == PipeKind.PROCESS and length <= opts.tap_max_units * unit:
            for n in (r.start, r.end):
                if n.body is not None and objects[n.body].category == Category.INSTRUMENT:
                    kind = PipeKind.INSTRUMENT_TAP
        pipes.append(PipeRun(
            id=ids.next("p"), page=page_index, points=[Point(x=x, y=y) for x, y in r.points], start=start, end=end,
            kind=kind, medium=_medium_from_color(r.color) if kind != PipeKind.SIGNAL else Medium.UNKNOWN,
            confidence=1.0, source="vector",
        ))
    timing["assemble"] = time.perf_counter() - t

    # ---- texts
    t = time.perf_counter()
    objrefs = []
    for k, d in enumerate(dets):
        o = objects[k]
        circle = None
        if o.category == Category.INSTRUMENT:
            circ = [ps.prims[i] for i in d.prim_ids if ps.prims[i].kind == "circle"]
            if circ:
                c = max(circ, key=lambda p: p.r)
                circle = (c.cx, c.cy, c.r)
        holds = get_symbol(d.cls).holds_text if d.cls != SymbolClass.UNKNOWN else False
        objrefs.append(ObjRef(k, d.cls, o.category, d.bbox, d.body, o.rotation, holds, circle))
    piperefs = [PipeRef(k, [(a.x, a.y, b.x, b.y) for a, b in zip(p.points, p.points[1:])], p.kind == PipeKind.SIGNAL)
                for k, p in enumerate(pipes)]
    ta = assign_texts(lines, regions, objrefs, piperefs, unit)
    texts: list[TextItem] = []
    for k, tl in enumerate(lines):
        res = ta.results[k]
        owner = None
        if res.owner is not None:
            owner = obj_ids[res.owner[1]] if res.owner[0] == "obj" else pipes[res.owner[1]].id
        tid = ids.next("t")
        texts.append(TextItem(id=tid, text=tl.text, bbox=BBox(x0=tl.bbox[0], y0=tl.bbox[1], x1=tl.bbox[2], y1=tl.bbox[3]),
                              page=page_index, rotation=tl.rotation, size=round(tl.size, 2), role=res.role, owner_id=owner))
        if res.owner is not None:
            if res.owner[0] == "obj":
                objects[res.owner[1]].text_ids.append(tid)
            else:
                pipes[res.owner[1]].text_ids.append(tid)
    for k, tag in ta.object_tags.items():
        objects[k].tag = tag
    for k, attrs in ta.object_attrs.items():
        objects[k].attributes.update(attrs)
    label_pos: dict[int, tuple[float, float]] = {}
    for k, ln in ta.pipe_line_numbers.items():
        _set_line_number(pipes[k], ln)
    for kk, res in enumerate(ta.results):
        if res.role == TextRole.LINE_NUMBER and res.owner is not None and res.owner[0] == "pipe":
            label_pos.setdefault(res.owner[1], (lines[kk].cx, lines[kk].cy))
    if opts.propagate_line_numbers:
        _propagate_line_numbers(pipes, objects, label_pos)
    _flow_directions(pipes, runs, net, sep, objects)
    timing["text"] = time.perf_counter() - t

    title_lines = [tl for k, tl in enumerate(lines) if ta.results[k].role == TextRole.TITLE_BLOCK
                   and regions.kind_at(tl.bbox, 0.5) == "title_block"]
    doc.info = drawing_info(title_lines, lines, [p.line_number for p in pipes if p.line_number], regions.cell_at)
    da = regions.drawing_area
    for tb in regions.title_blocks:  # drop a full-width title strip from the drawing area
        if tb[2] - tb[0] >= 0.9 * (da[2] - da[0]) and tb[1] > (da[1] + da[3]) / 2:
            da = (da[0], da[1], da[2], min(da[3], tb[1]))
        elif tb[3] - tb[1] >= 0.9 * (da[3] - da[1]) and tb[0] > (da[0] + da[2]) / 2:
            da = (da[0], da[1], min(da[2], tb[0]), da[3])
    doc.pages[0].drawing_area = BBox(x0=da[0], y0=da[1], x1=da[2], y1=da[3])
    doc.objects = sorted(objects, key=lambda o: int(o.id[1:]))
    doc.junctions = junctions
    doc.pipes = pipes
    doc.texts = texts
    timing["total"] = time.perf_counter() - t0
    page_meta.update({
        "unit": round(unit, 3),
        "unit_anchored": st.anchored,
        "decompose_budget_exhausted": bool(st.stats.get("decompose_budget_exhausted")),
        "decompose_budget_used": st.stats.get("decompose_budget_used", 0.0),
        "regions": {
            "drawing_area": [round(v, 2) for v in regions.drawing_area],
            "title_blocks": [[round(v, 2) for v in b] for b in regions.title_blocks],
            "notes": [[round(v, 2) for v in b] for b in regions.notes],
            "legends": [[round(v, 2) for v in b] for b in regions.legends],
        },
        "counts": {"primitives": len(ps.prims), "text_strokes": sum(1 for p in ps.prims if p.role == "text"),
                   "candidates": len(cands), "objects": len(objects),
                   "unknown": sum(1 for o in objects if o.cls == SymbolClass.UNKNOWN), "pipes": len(pipes),
                   "junctions": len(junctions), "texts": len(texts), "dots": len(sep.dots), "arrows": len(sep.arrows)},
        "timing": {k: round(v, 4) for k, v in timing.items()},
    })
    return doc


def _set_line_number(p: PipeRun, ln: str) -> None:
    p.line_number = ln
    info = line_number_info(ln)
    if info:
        p.size = info["size"]
    med = medium_of_line(ln)
    if med != Medium.UNKNOWN:
        p.medium = med


def _incidence(pipes: list[PipeRun]) -> dict[str, list[tuple[int, str, tuple[float, float]]]]:
    """node id -> [(pipe index, 'start' | 'end', unit direction leaving the node along the pipe)]."""
    out: dict[str, list[tuple[int, str, tuple[float, float]]]] = defaultdict(list)
    for k, p in enumerate(pipes):
        pts = p.points
        for which, e, a, b in (("start", p.start, pts[0], pts[1] if len(pts) > 1 else pts[0]),
                               ("end", p.end, pts[-1], pts[-2] if len(pts) > 1 else pts[-1])):
            node = e.object_id or e.junction_id
            if not node:
                continue
            dx, dy = b.x - a.x, b.y - a.y
            n = math.hypot(dx, dy) or 1.0
            out[node].append((k, which, (dx / n, dy / n)))
    return out


def _straight_pairs_at(inc: list[tuple[int, str, tuple[float, float]]],
                       pipes: list[PipeRun]) -> list[tuple[tuple[int, str], tuple[int, str]]]:
    """Pairs of (run, end) of two different process runs leaving a junction in opposite
    directions (the line continues straight through)."""
    proc = [(k, w, d) for k, w, d in inc if pipes[k].kind == PipeKind.PROCESS]
    pairs = []
    for i in range(len(proc)):
        for j in range(i + 1, len(proc)):
            (a, wa, da), (b, wb, db) = proc[i], proc[j]
            if a != b and da[0] * db[0] + da[1] * db[1] < -0.97:
                pairs.append(((a, wa), (b, wb)))
    return pairs


def _two_runs(proc: list) -> bool:
    """Exactly two incident entries that belong to two different runs (a run with both
    ends on the same node -- a loop -- does not pass anything through that node)."""
    return len(proc) == 2 and proc[0][0] != proc[1][0]


def _propagate_line_numbers(pipes: list[PipeRun], objects: list[PIDObject], labels: dict[int, tuple[float, float]]) -> None:
    """A line number labels the whole line between equipment. Unlabelled runs take the
    number of the nearest label measured along the pipe network, crossing only inline
    valves / fittings and continuing straight through tees (``labels``: run index ->
    label position)."""
    import heapq

    inc = _incidence(pipes)
    by_id = {o.id: o for o in objects}
    lengths = [max(p.length, 1e-6) for p in pipes]
    links: dict[tuple[int, str], list[tuple[int, str]]] = defaultdict(list)  # (run, end) -> [(run, end)]

    def link(a: int, wa: str, b: int, wb: str) -> None:
        links[(a, wa)].append((b, wb))
        links[(b, wb)].append((a, wa))

    for node, lst in inc.items():
        o = by_id.get(node)
        proc = [(k, w) for k, w, _ in lst if pipes[k].kind == PipeKind.PROCESS]
        if o is not None:
            if o.category in (Category.VALVE, Category.FITTING) and _two_runs(proc):
                (a, wa), (b, wb) = proc
                link(a, wa, b, wb)
            continue
        for (a, wa), (b, wb) in _straight_pairs_at(lst, pipes):
            link(a, wa, b, wb)

    dist: dict[tuple[int, str], float] = {}
    best_mid: dict[int, tuple[float, str]] = {}
    heap: list[tuple[float, int, str, str]] = []
    for k, p in enumerate(pipes):
        if not p.line_number or k not in labels:
            continue
        pos = _arc_position(p, labels[k])
        for w, d in (("start", pos), ("end", lengths[k] - pos)):
            heapq.heappush(heap, (d, k, w, p.line_number))
    while heap:
        d, k, w, ln = heapq.heappop(heap)
        if (k, w) in dist:
            continue
        dist[(k, w)] = d
        for b, wb in links.get((k, w), ()):
            if pipes[b].line_number and b in labels:
                continue
            mid = d + lengths[b] / 2
            if b not in best_mid or mid < best_mid[b][0]:
                best_mid[b] = (mid, ln)
            far = "end" if wb == "start" else "start"
            if (b, far) not in dist:
                heapq.heappush(heap, (d + lengths[b], b, far, ln))
    for b, (_, ln) in best_mid.items():
        if not pipes[b].line_number:
            _set_line_number(pipes[b], ln)


def _arc_position(p: PipeRun, xy: tuple[float, float]) -> float:
    """Arc length from the run's start to the point of the run closest to ``xy``."""
    best, acc, pos = None, 0.0, 0.0
    for a, b in zip(p.points, p.points[1:]):
        L = a.dist(b)
        if L > 0:
            t = max(0.0, min(1.0, ((xy[0] - a.x) * (b.x - a.x) + (xy[1] - a.y) * (b.y - a.y)) / (L * L)))
            d = math.hypot(a.x + t * (b.x - a.x) - xy[0], a.y + t * (b.y - a.y) - xy[1])
            if best is None or d < best:
                best, pos = d, acc + t * L
        acc += L
    return pos


_FLOW_IN = {"in", "suction", "tube_in", "shell_in", "a_in", "b_in"}
_FLOW_OUT = {"out", "discharge", "tube_out", "shell_out", "a_out", "b_out"}
_DIRECTIONAL = {SymbolClass.CHECK_VALVE, SymbolClass.SDNR_VALVE, SymbolClass.PUMP_CENTRIFUGAL, SymbolClass.PUMP_SCREW,
                SymbolClass.SAFETY_VALVE, SymbolClass.COMPRESSOR}


def _flow_directions(pipes: list[PipeRun], runs, net: Network, sep: Separation, objects: list[PIDObject]) -> None:
    """Flow direction of process runs: seeded by flow arrows and directional symbols
    (pumps, check valves, relief valves), then propagated through two-port symbols,
    straight through instrument-tap tees and by flow conservation at tees."""
    by_id = {o.id: o for o in objects}
    piece_run: dict[int, int] = {}
    for k, r in enumerate(runs):
        for i in r.pieces:
            piece_run[i] = k
    votes: dict[int, int] = defaultdict(int)
    for a in sep.arrows:
        for i, pc in enumerate(net.pieces):
            if not pc.alive or i not in piece_run or pc.line_id != a.line_id:
                continue
            if point_seg_dist(a.x, a.y, pc.x0, pc.y0, pc.x1, pc.y1) > 1.0:
                continue
            k = piece_run[i]
            pts = runs[k].points
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                if point_seg_dist(a.x, a.y, x0, y0, x1, y1) <= 1.0:
                    votes[k] += 2 if (x1 - x0) * a.dx + (y1 - y0) * a.dy > 0 else -2
                    break
            break
    for k, p in enumerate(pipes):
        for e, sign in ((p.end, 1), (p.start, -1)):
            if not e.object_id or by_id[e.object_id].cls not in _DIRECTIONAL:
                continue
            if e.port in _FLOW_IN:
                votes[k] += sign
            elif e.port in _FLOW_OUT:
                votes[k] -= sign
    # sources and sinks: 'FROM ...' / 'TO ...' connectors, sea chests, overboard discharges
    for k, p in enumerate(pipes):
        if p.kind != PipeKind.PROCESS:
            continue
        for e, sign in ((p.end, 1), (p.start, -1)):
            o = by_id.get(e.object_id) if e.object_id else None
            if o is None:
                continue
            name = str(o.attributes.get("name", "")).upper()
            if o.cls == SymbolClass.SEA_CHEST or (o.cls == SymbolClass.OFF_PAGE_CONNECTOR and name.startswith("FROM")):
                votes[k] -= sign  # flow leaves the source
            elif o.cls == SymbolClass.OVERBOARD or (o.cls == SymbolClass.OFF_PAGE_CONNECTOR and name.startswith("TO")):
                votes[k] += sign  # flow enters the sink
    flow: dict[int, int] = {k: (1 if v > 0 else -1) for k, v in votes.items() if v}

    def into(k: int, which: str) -> Optional[bool]:
        f = flow.get(k)
        if f is None:
            return None
        return (f > 0) == (which == "end")

    def set_into(k: int, which: str, value: bool) -> bool:
        if k in flow:
            return False
        flow[k] = (1 if value else -1) if which == "end" else (-1 if value else 1)
        return True

    inc = _incidence(pipes)
    changed = True
    while changed:
        changed = False
        for node, lst in inc.items():
            proc = [(k, w, d) for k, w, d in lst if pipes[k].kind == PipeKind.PROCESS]
            o = by_id.get(node)
            if o is not None:
                if not _two_runs(proc) or o.category == Category.INSTRUMENT or o.cls == SymbolClass.TANK:
                    continue
                (a, wa, _), (b, wb, _) = proc
                ia, ib = into(a, wa), into(b, wb)
                if ia is not None and ib is None:
                    changed |= set_into(b, wb, not ia)
                elif ib is not None and ia is None:
                    changed |= set_into(a, wa, not ib)
                continue
            if _two_runs(proc) and len(lst) > 2:  # tee whose branch is not a process line (instrument tap)
                (a, wa, _), (b, wb, _) = proc
                ia, ib = into(a, wa), into(b, wb)
                if ia is not None and ib is None:
                    changed |= set_into(b, wb, not ia)
                elif ib is not None and ia is None:
                    changed |= set_into(a, wa, not ib)
                continue
            known = [into(k, w) for k, w, _ in proc]
            unknown = [(k, w) for (k, w, _), v in zip(proc, known) if v is None]
            if len(proc) >= 3 and len(unknown) == 1 and len({k for k, _, _ in proc}) == len(proc):
                vals = [v for v in known if v is not None]
                if all(vals):
                    changed |= set_into(*unknown[0], False)
                elif not any(vals):
                    changed |= set_into(*unknown[0], True)
    _flow_from_pumps(pipes, objects, flow)
    for k, f in flow.items():
        pipes[k].flow_direction = f


_PASSABLE_EQUIPMENT = {SymbolClass.STRAINER, SymbolClass.FILTER, SymbolClass.DUPLEX_FILTER, SymbolClass.FLOW_METER,
                       SymbolClass.HEATER, SymbolClass.BWTS_UNIT}


def _flow_from_pumps(pipes: list[PipeRun], objects: list[PIDObject], flow: dict[int, int]) -> None:
    """Remaining process runs: flow towards the nearest pump suction / away from the nearest
    pump discharge, measured along the network through valves, fittings, inline equipment
    and junctions (tanks, connectors and other equipment end the search)."""
    import heapq

    by_id = {o.id: o for o in objects}
    adj: dict[str, list[tuple[int, str]]] = defaultdict(list)  # node -> [(run, other node)]
    for k, p in enumerate(pipes):
        a, b = p.start.object_id or p.start.junction_id, p.end.object_id or p.end.junction_id
        if p.kind == PipeKind.PROCESS and a and b:
            adj[a].append((k, b))
            adj[b].append((k, a))

    def passable(node: str) -> bool:
        o = by_id.get(node)
        if o is None:
            return True  # junction
        return o.category in (Category.VALVE, Category.FITTING) or o.cls in _PASSABLE_EQUIPMENT

    heap: list[tuple[float, str, int, str]] = []  # (distance, node, run, 'suction' | 'discharge')
    for k, p in enumerate(pipes):
        if p.kind != PipeKind.PROCESS:
            continue
        for e in (p.start, p.end):
            o = by_id.get(e.object_id) if e.object_id else None
            if o is not None and o.cls in (SymbolClass.PUMP_CENTRIFUGAL, SymbolClass.PUMP_SCREW) and e.port in ("suction", "discharge", "in", "out"):
                side = "suction" if e.port in ("suction", "in") else "discharge"
                heapq.heappush(heap, (0.0, o.id, k, side))
    seen: set[int] = set()
    while heap:
        d, node, k, side = heapq.heappop(heap)
        if k in seen:
            continue
        seen.add(k)
        p = pipes[k]
        start_node = p.start.object_id or p.start.junction_id
        toward = start_node == node  # the run is entered at its start
        if k not in flow:
            # suction side: flow runs towards ``node``; discharge side: away from it
            flow[k] = (-1 if toward else 1) if side == "suction" else (1 if toward else -1)
        far = (p.end.object_id or p.end.junction_id) if toward else start_node
        if not far or not passable(far):
            continue
        for k2, _ in adj.get(far, ()):
            if k2 not in seen:
                heapq.heappush(heap, (d + p.length, far, k2, side))


# --------------------------------------------------------------------------- document


def _merge_into(doc: PIDDocument, part: PIDDocument) -> None:
    doc.pages.extend(part.pages)
    doc.objects.extend(part.objects)
    doc.junctions.extend(part.junctions)
    doc.pipes.extend(part.pipes)
    doc.texts.extend(part.texts)
    doc.meta.setdefault("pages", {}).update(part.meta.get("pages", {}))
    for key in ("unreadable_pages", "empty_pages"):
        if part.meta.get(key):
            doc.meta.setdefault(key, []).extend(part.meta[key])


def _merge_info(infos: list[DrawingInfo]) -> DrawingInfo:
    out = DrawingInfo()
    for inf in infos:
        for name in ("drawing_no", "title", "hull_no", "ship_type", "revision", "sheet", "scale"):
            if getattr(out, name) is None and getattr(inf, name) is not None:
                setattr(out, name, getattr(inf, name))
        if out.system.value == "unknown" and inf.system.value != "unknown":
            out.system = inf.system
        for s in inf.systems:
            if s not in out.systems:
                out.systems.append(s)
        for k, v in inf.extra.items():
            out.extra.setdefault(k, v)
    if out.system.value != "unknown" and out.system in out.systems:
        out.systems.remove(out.system)
        out.systems.insert(0, out.system)
    return out


def _learn_legends(pdf, page_list: Sequence[int], cfg: Config, contents: dict[int, PageContent],
                   geoms: dict[int, PageGeometry]) -> Optional[LegendBook]:
    """Legends of every page of the document. Selected pages are read once (their content is kept
    in ``contents``, and the prepared geometry of pages with a legend heading in ``geoms``, for the
    extraction); other pages are read only when their text has a legend heading word. None when the
    document has no legend rows."""
    legends = []
    selected = set(page_list)
    for idx in range(pdf.page_count):
        page = pdf[idx]
        if idx in selected:
            pc = read_page(page)
            contents[idx] = pc
        else:
            try:
                if not page_has_heading(page.get_text("text")):
                    continue
            except Exception:  # pragma: no cover - a page whose text cannot be read is not a legend sheet
                continue
            pc = read_page(page)
        if pc.empty or not pc.readable or not page_has_heading(pc):
            continue
        try:
            lines = page_lines(pc)
            if not heading_lines(lines):
                continue
            g = prepare_page(pc, cfg, lines)
            if idx in selected:
                geoms[idx] = g
            legends.extend(find_legends(pc, idx, cfg, geometry=g))
        except Exception:  # pragma: no cover - legend learning must never stop the extraction
            log.exception("legend reading failed on page %d; built-in symbols only", idx)
            geoms.pop(idx, None)
    if not any(lg.rows for lg in legends):
        return None
    return LegendBook.learn(legends)


def extract_pdf(path: str | Path, config: Optional[Config] = None, pages: Optional[Iterable[int]] = None,
                options: Optional[ExtractOptions] = None) -> PIDDocument:
    """Extract every (or the selected) page of a readable P&ID PDF.

    With ``options.learn_legends`` (default) the legends of the document -- on the selected pages
    or on any other page, e.g. a separate legend sheet -- are read first and their symbols are
    matched alongside the built-in library for this document (:mod:`shippid.extract.legend`);
    ``doc.meta["legend"]`` reports the rows, the learned templates and the unmapped rows."""
    cfg = config or Config()
    opts = options or ExtractOptions()
    t0 = time.perf_counter()
    pdf = open_pdf(path)
    try:
        doc = PIDDocument(source=str(path))
        doc.meta["extractor"] = "vector"
        ids = IdAllocator()
        infos = []
        page_list = list(pages) if pages is not None else list(range(pdf.page_count))
        for idx in page_list:
            if idx < 0 or idx >= pdf.page_count:
                raise IndexError(f"page {idx} out of range (document has {pdf.page_count} pages)")
        contents: dict[int, PageContent] = {}
        geoms: dict[int, PageGeometry] = {}
        book: Optional[LegendBook] = None
        if opts.learn_legends:
            t1 = time.perf_counter()
            book = _learn_legends(pdf, page_list, cfg, contents, geoms)
            if book is not None:
                doc.meta["legend"] = book.to_meta()
                doc.meta["legend"]["seconds"] = round(time.perf_counter() - t1, 3)
        cache = MatchCache(cfg.symbol_max_size)
        for idx in page_list:
            part = extract_page(pdf[idx], idx, cfg, ids, opts, cache, content=contents.pop(idx, None), legend=book,
                                geometry=geoms.pop(idx, None))
            _merge_into(doc, part)
            infos.append(part.info)
        doc.info = _merge_info(infos)
        doc.meta["seconds"] = round(time.perf_counter() - t0, 3)
        doc.meta["match_cache"] = {"hits": cache.hits, "misses": cache.misses}
        return doc
    finally:
        pdf.close()
