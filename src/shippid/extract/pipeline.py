"""Deterministic vector extraction: PDF page -> :class:`~shippid.model.PIDDocument`.

Stages (see ``docs/ARCHITECTURE.md`` section 2): read the page, find sheet
furniture (frame, title block, notes, legend), separate pipe lines from symbol
strokes, group and classify symbol candidates by template matching, build the
pipe network and finally assign texts. Every object and pipe carries
``source="vector"``; per-candidate class alternatives are kept for Gemma.
"""

from __future__ import annotations

import logging
import math
import re
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
from .pipes import MLine, Separation, separate
from .primitives import PrimSet, bbox_dist, bbox_union, estimate_unit, point_seg_dist
from .regions import Regions, detect_regions
from .symbols_match import CandidateShape, MatchCache, MatchResult, TemplateLibrary, candidate_shape, get_library, match_candidate
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
    propagate_line_numbers: bool = True  # copy a line number across inline 2-port symbols
    keep_legend_symbols: bool = False


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


def _detect(ps: PrimSet, prim_ids: list[int], lib: TemplateLibrary, unit: float, cache: MatchCache) -> Detection:
    cs = _shape_for(ps, prim_ids)
    res = match_candidate(cs, lib, unit, cache=cache)
    return Detection(sorted(prim_ids), cs, res)


def _finalise(d: Detection, accept: float) -> None:
    best = d.best
    if best is not None and best.score >= accept:
        d.cls = best.variant.cls
        d.score = best.score
        tb = best.bbox()
        d.bbox = (min(tb[0], tb[2]), min(tb[1], tb[3]), max(tb[0], tb[2]), max(tb[1], tb[3]))
        bb = best.body_bbox()
        d.body = (min(bb[0], bb[2]), min(bb[1], bb[3]), max(bb[0], bb[2]), max(bb[1], bb[3]))
    else:
        d.cls = SymbolClass.UNKNOWN
        d.score = best.score if best else 0.0
        d.bbox = d.shape.bbox
        d.body = d.shape.bbox


def classify_candidates(ps: PrimSet, cands: Sequence[Candidate], unit: float, opts: ExtractOptions,
                        lib: Optional[TemplateLibrary] = None, cache: Optional[MatchCache] = None) -> list[Detection]:
    lib = lib or get_library()
    cache = cache if cache is not None else MatchCache()
    accept = opts.accept_score
    dets: list[Detection] = []
    for c in cands:
        prims = [ps.prims[i] for i in c.prim_ids]
        if c.size < opts.min_unknown_units * unit and not any(p.is_core for p in prims):
            for p in prims:
                p.role = "noise"
            continue
        d = _detect(ps, list(c.prim_ids), lib, unit, cache)
        if (d.best is None or d.best.score < accept) and len(c.clusters) >= 2:
            split = _try_split(ps, c, d, lib, unit, cache, accept)
            if split is not None:
                dets.extend(split)
                continue
        if d.best is None or d.best.score < 0.93:
            parts = _decompose(ps, d, lib, unit, cache, accept)
            if parts is not None:
                dets.extend(parts)
                continue
        dets.append(d)
    for d in dets:
        _finalise(d, accept)
    dets = _absorb_fragments(ps, dets, lib, unit, cache, accept)
    _size_prior(dets, accept)
    return dets


def _try_split(ps: PrimSet, c: Candidate, whole: Detection, lib, unit, cache, accept) -> Optional[list[Detection]]:
    """Split a poorly matching group into its touch clusters (two symbols drawn
    flange-to-flange); small clusters go with their nearest big neighbour."""
    prims = ps.prims
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
    parts = [_detect(ps, g, lib, unit, cache) for g in groups]
    scores = [p.best.score if p.best else 0.0 for p in parts]
    whole_score = whole.best.score if whole.best else 0.0
    if min(scores) >= 0.9 * accept and sum(scores) / len(scores) > whole_score + 0.05:
        return parts
    return None


def _decompose(ps: PrimSet, whole: Detection, lib, unit, cache, accept) -> Optional[list[Detection]]:
    """Symbols bolted flange-to-flange form one touching candidate. Try cuts across
    the candidate at the boundaries of its closed outlines (never through one),
    rank the partitions by coarse score and keep the best split when the
    point-weighted fine score improves."""
    from .symbols_match import coarse_scores

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
    m = 0.06 * unit
    solid_idx = [k for k, a in enumerate(atoms) if a[2]]
    merged: list[tuple[list[int], tuple[float, float, float, float], bool]] = []
    absorbed: set[int] = set()
    for k, a in enumerate(atoms):
        if a[2]:
            continue
        cx, cy = (a[1][0] + a[1][2]) / 2, (a[1][1] + a[1][3]) / 2
        for s_ in solid_idx:
            b = atoms[s_][1]
            if b[0] + m < cx < b[2] - m and b[1] + m < cy < b[3] - m:
                atoms[s_][0].extend(a[0])
                absorbed.add(k)
                break
    merged = [a for k, a in enumerate(atoms) if k not in absorbed]
    atoms = merged
    if len(atoms) < 2 or len(atoms) > 80:
        return None

    def score(d: Detection) -> float:
        return d.best.score if d.best else 0.0

    coarse_memo: dict[tuple[int, ...], tuple[float, float]] = {}

    def coarse(group: list[int]) -> tuple[float, float]:
        key = tuple(sorted(group))
        if key not in coarse_memo:
            cs = _shape_for(ps, [i for a in group for i in atoms[a][0]])
            sc, _, _ = coarse_scores(lib, cs, unit)
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
        return sum(score(p) * wi for p, wi in zip(parts, w)) / max(sum(w), 1e-9)

    budget = [24]  # match calls allowed for this candidate

    def detect(side: list[int]) -> Detection:
        budget[0] -= 1
        return _detect(ps, [i for a in side for i in atoms[a][0]], lib, unit, cache)

    def splittable(side: list[int], d: Detection) -> bool:
        """Worth splitting further: poorly matched and large enough for two symbols."""
        if score(d) >= 0.85 or budget[0] <= 0 or sum(1 for a in side if atoms[a][2]) < 1:
            return False
        b = bbox_union(atoms[a][1] for a in side)
        return max(b[2] - b[0], b[3] - b[1]) >= 1.1 * unit

    def split_group(group: list[int], current: Detection, depth: int) -> list[Detection]:
        """Best decomposition of ``group`` (``[current]`` when no split helps)."""
        if len(group) < 2 or depth > 3 or budget[0] <= 0:
            return [current]
        gb = bbox_union(atoms[a][1] for a in group)
        main = 0 if (gb[2] - gb[0]) >= (gb[3] - gb[1]) else 1
        best_parts, best_val = [current], score(current) + 0.06
        for axis in (main, 1 - main):
            ranked = []
            for left, right in partitions(group, axis):
                (sa, wa), (sb, wb) = coarse(left), coarse(right)
                ranked.append(((sa * wa + sb * wb) / max(wa + wb, 1e-9), left, right))
            ranked.sort(key=lambda r: -r[0])
            for _, left, right in ranked[: (3 if depth == 0 else 2)]:
                if budget[0] <= 1:
                    break
                leaves = []
                for side in (left, right):
                    d = detect(side)
                    # a part that is itself several symbols is split further (chains of three or more)
                    leaves.extend(split_group(side, d, depth + 1) if splittable(side, d) else [d])
                if min(score(p) for p in leaves) < accept:
                    continue
                val = weighted(leaves)
                if val > best_val:
                    best_parts, best_val = leaves, val
            if len(best_parts) > 1:
                break
        return best_parts

    parts = split_group(list(range(len(atoms))), whole, 0)
    return parts if len(parts) > 1 else None


def _absorb_fragments(ps: PrimSet, dets: list[Detection], lib, unit, cache, accept) -> list[Detection]:
    """Merge an unmatched small fragment into a neighbour when the union matches
    better (e.g. an actuator drawn with a gap to its valve body)."""
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
                m = _detect(ps, o.prim_ids + d.prim_ids, lib, unit, cache)
                _finalise(m, accept)
                if m.cls == SymbolClass.UNKNOWN or m.score < max(o.score - 0.03, accept):
                    continue
                if best is None or m.score > best[0].score:
                    best = (m, o)
            if best is not None:
                m, o = best
                out.remove(o)
                out.remove(d)
                out.append(m)
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
        if ranked[0].variant.cls != top.variant.cls and ranked[0].score >= top.score - 0.03:
            # switch class only; keep the canonical orientation chosen by the matcher
            winner = ranked[0].variant.cls
            d.results = [r for r in d.results if r.variant.cls == winner] + [r for r in d.results if r.variant.cls != winner]
            _finalise(d, accept)


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


def _analyse(pc: PageContent, lines: list[TextLine], cfg: Config, opts: ExtractOptions, lib: TemplateLibrary,
             cache: MatchCache, unit: Optional[float], timing: dict[str, float]) -> _Stage:
    """Regions, stroke separation and symbol classification for one symbol unit."""
    tol = cfg.snap_tol
    t = time.perf_counter()
    ps = PrimSet.from_page(pc)
    unit0 = unit or estimate_unit(ps.prims, range(len(ps.prims)), max(0.6, 0.5 * tol))
    regions = detect_regions(ps, lines, pc.width, pc.height, tol, unit0)
    if unit is None:
        free_ids = [p.id for p in ps.prims if p.role == ""]
        unit = estimate_unit(ps.prims, free_ids, max(0.6, 0.5 * tol), default=unit0)
        probed = _probe_unit(ps, free_ids, unit, tol, lib, cache)
        if probed is not None:
            unit = probed
    timing["regions"] = timing.get("regions", 0.0) + time.perf_counter() - t

    t = time.perf_counter()
    sep = separate(ps, unit, tol, opts.seed_units)
    timing["separate"] = timing.get("separate", 0.0) + time.perf_counter() - t

    t = time.perf_counter()
    cands = group_candidates(ps, sep.symbol_ids, unit, tol, opts.group_units)
    dets = classify_candidates(ps, cands, unit, opts, lib, cache)
    dets, diag_lines = _straight_unknowns_to_pipes(ps, dets, unit, tol, len(sep.lines))
    if not opts.keep_legend_symbols:
        dets = [d for d in dets if regions.kind_at(d.bbox, tol) is None]
    dets.sort(key=lambda d: (round(d.bbox[1], 1), round(d.bbox[0], 1)))
    timing["symbols"] = timing.get("symbols", 0.0) + time.perf_counter() - t
    return _Stage(ps, regions, sep, cands, dets, diag_lines, unit)


def _probe_unit(ps: PrimSet, ids: Sequence[int], unit: float, tol: float, lib: TemplateLibrary,
                cache: MatchCache, top: int = 6) -> Optional[float]:
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
    for key, n in counts.most_common(top):
        res = match_candidate(_shape_for(ps, groups[key][0]), lib, None, cache=cache)
        if res and res[0].score >= 0.7 and res[0].variant.spec.category == Category.VALVE:
            scales.extend([res[0].scale] * n)
    if len(scales) < 3:
        return None
    return float(np.median(scales))


def _valve_unit(dets: Sequence[Detection]) -> Optional[float]:
    """Median matched scale of confidently classified valves (else other rigid symbols)."""
    valves = [d.best.scale for d in dets if d.best is not None and d.cls != SymbolClass.UNKNOWN and d.score >= 0.85
              and d.best.variant.spec.category == Category.VALVE]
    if len(valves) >= 3:
        return float(np.median(valves))
    rigid = [d.best.scale for d in dets if d.best is not None and d.cls != SymbolClass.UNKNOWN and d.score >= 0.85
             and not d.best.variant.stretchable and d.best.variant.spec.category != Category.EQUIPMENT]
    if len(rigid) >= 3:
        return float(np.median(rigid))
    return None


def extract_page(page, page_index: int = 0, config: Optional[Config] = None, ids: Optional[IdAllocator] = None,
                 options: Optional[ExtractOptions] = None, cache: Optional[MatchCache] = None,
                 content: Optional[PageContent] = None) -> PIDDocument:
    """Extract one PyMuPDF page into a single-page :class:`PIDDocument`."""
    cfg = config or Config()
    opts = options or ExtractOptions()
    ids = ids or IdAllocator()
    cache = cache if cache is not None else MatchCache()
    timing: dict[str, float] = {}
    t0 = time.perf_counter()

    pc = content or read_page(page)
    timing["read"] = time.perf_counter() - t0
    doc = PIDDocument(source=str(getattr(getattr(page, "parent", None), "name", "") or ""))
    doc.pages.append(PageInfo(index=page_index, width=pc.width, height=pc.height))
    page_meta: dict = {"image_coverage": round(pc.image_coverage, 4), "layers": pc.layers, "readable": pc.readable}
    doc.meta["pages"] = {str(page_index): page_meta}
    doc.meta["extractor"] = "vector"
    if not pc.readable:
        log.warning("page %d looks scanned/raster-only (image coverage %.2f, %d chars); skipped",
                    page_index, pc.image_coverage, pc.n_chars)
        doc.meta["unreadable_pages"] = [page_index]
        return doc

    tol = cfg.snap_tol
    lib = get_library()
    lines = build_lines(pc.words)
    st = _analyse(pc, lines, cfg, opts, lib, cache, None, timing)
    ref = _valve_unit(st.dets)
    if ref is not None and abs(ref / st.unit - 1.0) > 0.15:
        # the first estimate (median size of symbol-only stroke clusters) is biased by the mix of
        # symbols on the sheet; matched valve scales are the reliable symbol unit -> redo with it
        log.debug("page %d: symbol unit %.2f -> %.2f (from matched valves)", page_index, st.unit, ref)
        st = _analyse(pc, lines, cfg, opts, lib, cache, ref, timing)
    ps, regions, sep, cands, dets, diag_lines, unit = st.ps, st.regions, st.sep, st.cands, st.dets, st.diag_lines, st.unit

    # ---- network
    t = time.perf_counter()
    bodies = []
    for k, d in enumerate(dets):
        spec = get_symbol(d.cls) if d.cls != SymbolClass.UNKNOWN else None
        ports = d.best.ports() if (d.best is not None and spec is not None) else []
        bodies.append(Body(k, d.bbox, d.body, [ps.prims[i] for i in d.prim_ids], ports,
                           boundary_ports=(spec is None or spec.boundary_ports),
                           is_valve=spec is not None and spec.category == Category.VALVE))
    pipe_lines = sep.pipe_lines + diag_lines
    net = build_network(pipe_lines, bodies, sep.dots, unit, tol)
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
            alt.append({"cls": r.variant.cls.value, "score": round(r.score, 3)})
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
        if accepted and best is not None and best.variant.stretchable:
            attrs["stretch"] = [round(best.sx / best.scale, 3), round(best.sy / best.scale, 3)]
        obj = PIDObject(
            id=obj_ids[k],
            cls=d.cls,
            category=category,
            bbox=BBox(x0=d.bbox[0], y0=d.bbox[1], x1=d.bbox[2], y1=d.bbox[3]),
            page=page_index,
            rotation=best.variant.rotation if (accepted and best) else 0,
            mirrored=bool(best.variant.mirrored) if (accepted and best) else False,
            actuator=best.variant.actuator if (accepted and best and category == Category.VALVE) else ActuatorType.NONE,
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
        "regions": {
            "drawing_area": [round(v, 2) for v in regions.drawing_area],
            "title_blocks": [[round(v, 2) for v in b] for b in regions.title_blocks],
            "notes": [[round(v, 2) for v in b] for b in regions.notes],
            "legends": [[round(v, 2) for v in b] for b in regions.legends],
        },
        "counts": {"primitives": len(ps.prims), "candidates": len(cands), "objects": len(objects),
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


def _straight_pairs_at(inc: list[tuple[int, str, tuple[float, float]]], pipes: list[PipeRun]) -> list[tuple[int, int]]:
    """Pairs of process runs leaving a junction in opposite directions (the run continues through)."""
    proc = [(k, d) for k, _, d in inc if pipes[k].kind == PipeKind.PROCESS]
    pairs = []
    for i in range(len(proc)):
        for j in range(i + 1, len(proc)):
            (a, da), (b, db) = proc[i], proc[j]
            if da[0] * db[0] + da[1] * db[1] < -0.97:
                pairs.append((a, b))
    return pairs


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
            if o.category in (Category.VALVE, Category.FITTING) and len({k for k, _ in proc}) == 2:
                (a, wa), (b, wb) = proc
                link(a, wa, b, wb)
            continue
        ends = {k: w for k, w, _ in lst}
        for a, b in _straight_pairs_at(lst, pipes):
            link(a, ends[a], b, ends[b])

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
                if len(proc) != 2 or o.category == Category.INSTRUMENT or o.cls == SymbolClass.TANK:
                    continue
                (a, wa, _), (b, wb, _) = proc
                ia, ib = into(a, wa), into(b, wb)
                if ia is not None and ib is None:
                    changed |= set_into(b, wb, not ia)
                elif ib is not None and ia is None:
                    changed |= set_into(a, wa, not ib)
                continue
            if len(proc) == 2 and len(lst) > 2:  # tee whose branch is not a process line (instrument tap)
                (a, wa, _), (b, wb, _) = proc
                ia, ib = into(a, wa), into(b, wb)
                if ia is not None and ib is None:
                    changed |= set_into(b, wb, not ia)
                elif ib is not None and ia is None:
                    changed |= set_into(a, wa, not ib)
                continue
            known = [into(k, w) for k, w, _ in proc]
            unknown = [(k, w) for (k, w, _), v in zip(proc, known) if v is None]
            if len(proc) >= 3 and len(unknown) == 1:
                vals = [v for v in known if v is not None]
                if all(vals):
                    changed |= set_into(*unknown[0], False)
                elif not any(vals):
                    changed |= set_into(*unknown[0], True)
    for k, f in flow.items():
        pipes[k].flow_direction = f


# --------------------------------------------------------------------------- document


def _merge_into(doc: PIDDocument, part: PIDDocument) -> None:
    doc.pages.extend(part.pages)
    doc.objects.extend(part.objects)
    doc.junctions.extend(part.junctions)
    doc.pipes.extend(part.pipes)
    doc.texts.extend(part.texts)
    doc.meta.setdefault("pages", {}).update(part.meta.get("pages", {}))
    if part.meta.get("unreadable_pages"):
        doc.meta.setdefault("unreadable_pages", []).extend(part.meta["unreadable_pages"])


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


def extract_pdf(path: str | Path, config: Optional[Config] = None, pages: Optional[Iterable[int]] = None,
                options: Optional[ExtractOptions] = None) -> PIDDocument:
    """Extract every (or the selected) page of a readable P&ID PDF."""
    cfg = config or Config()
    t0 = time.perf_counter()
    pdf = open_pdf(path)
    try:
        doc = PIDDocument(source=str(path))
        doc.meta["extractor"] = "vector"
        ids = IdAllocator()
        cache = MatchCache()
        infos = []
        page_list = list(pages) if pages is not None else list(range(pdf.page_count))
        for idx in page_list:
            if idx < 0 or idx >= pdf.page_count:
                raise IndexError(f"page {idx} out of range (document has {pdf.page_count} pages)")
            part = extract_page(pdf[idx], idx, cfg, ids, options, cache)
            _merge_into(doc, part)
            infos.append(part.info)
        doc.info = _merge_info(infos)
        doc.meta["seconds"] = round(time.perf_counter() - t0, 3)
        doc.meta["match_cache"] = {"hits": cache.hits, "misses": cache.misses}
        return doc
    finally:
        pdf.close()
