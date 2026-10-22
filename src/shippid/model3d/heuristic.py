"""Deterministic arrangement planner (marine rules of thumb) and shared P&ID graph helpers.

The heuristic plan is the fallback for every Gemma-planned item and the baseline when no
LLM is configured. Rules (engine-room practice, see docs/RESEARCH.md):

* tanks: double-bottom tanks (bilge well, sludge, drain, sump, overflow, water ballast and
  other remote tanks) in the double bottom; settling/service tanks elevated at the ship side;
  expansion/header tanks on the upper platform; other tanks on the tank top at the side;
* sea chests in the hull bottom at the side; overboard discharges on the shell;
* pumps on the tank top near their suction source (short suction, positive head); parallel
  pumps side by side;
* coolers, filters, BWTS, compressors on the floor; purifiers and heaters on the 3rd-deck
  platform; air receivers vertical next to the compressors; OWS on the tank top;
* main engine on the centre line on the tank top (aft), generator engines on the floor;
* order: engines first, then the flow order (topological from sources); transverse /
  longitudinal bands from the drawing position when no rule applies (bow to the right, port
  at the top of the sheet, the usual plan-view convention).

Multi-sheet sets and noisy extractions (:func:`prepare_document`, applied by every entry point of
the 3D stage): each pair of off-page connectors in ``doc.meta["sheet_links"]`` becomes one
zero-length process pipe ``sheet_link:<a>:<b>`` between the partners, and the two connectors are
pass-through items (not planned: the system continues across the sheet boundary, so adjacency,
the Gemma digest and the router see one network and the 3D pipe is continuous, with both markers
on it). Drawing positions are normalised per page with that page's own drawing area and the
sheets are laid side by side in sheet order (:class:`DrawingFrame`). Unrecognised symbols
(class ``unknown``) without a pipe connection are not planned as equipment (the router shows them
as small markers at their drawing position), and pipe-connected ones only up to a cap.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from ..catalog import EQUIPMENT_TAG_RE, SYSTEM_BY_CODE, SYSTEMS, VALVE_TAG_RE, parse_line_number
from ..model import (Category, Endpoint, Medium, PIDDocument, PIDObject, PipeKind, PipeRun, Point, SymbolClass,
                     SystemType, TextRole)
from .plan import MODEL3D_ROLE, ElevationBand, LayoutPlan, PlanItem, RoutingPlan, is_planned
from .zones import EngineRoom

S = SymbolClass

# ---------------------------------------------------------------------------- labels / systems

_NAME_KEYS = ("name", "label", "title", "ref", "service")


def object_labels(doc: PIDDocument) -> dict[str, str]:
    """Readable name per object: attributes (name/label/title/ref/service), else owned label /
    connector texts, else the tag."""
    owned: dict[str, list] = defaultdict(list)
    for t in doc.texts:
        if t.owner_id and t.role in (TextRole.LABEL, TextRole.CONNECTOR) and t.text.strip():
            owned[t.owner_id].append(t)
    out: dict[str, str] = {}
    for o in doc.objects:
        parts: list[str] = []
        for k in _NAME_KEYS:
            v = o.attributes.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(" ".join(v.split()))
            elif isinstance(v, (list, tuple)):
                parts.extend(" ".join(str(x).split()) for x in v if str(x).strip())
        if not parts:
            ts = sorted(owned.get(o.id, []), key=lambda t: (round(t.bbox.y0, 1), t.bbox.x0))
            parts = [" ".join(t.text.split()) for t in ts]
        if not parts and o.tag:
            parts = [o.tag]
        if parts:
            out[o.id] = " ".join(dict.fromkeys(parts))[:120]
    return out


_MEDIUM_SYSTEM = {info.medium: s for s, info in SYSTEMS.items()}


def pipe_system(pipe: PipeRun, default: SystemType = SystemType.UNKNOWN) -> SystemType:
    """System of a pipe from its line number code, else its medium, else ``default``."""
    if pipe.line_number:
        parsed = parse_line_number(pipe.line_number)
        if parsed and parsed["system"] in SYSTEM_BY_CODE:
            return SYSTEM_BY_CODE[parsed["system"]]
    if pipe.medium in _MEDIUM_SYSTEM and pipe.medium != Medium.FUEL_OIL:
        return _MEDIUM_SYSTEM[pipe.medium]
    if pipe.medium == Medium.FUEL_OIL and default in (SystemType.FUEL_OIL_SUPPLY, SystemType.FUEL_OIL_TRANSFER):
        return default
    if pipe.medium == Medium.FUEL_OIL:
        return SystemType.FUEL_OIL_TRANSFER
    return default


def pipe_medium(pipe: PipeRun, default: Medium = Medium.UNKNOWN) -> Medium:
    """Medium of a pipe: its own, else from the line-number system code, else ``default``."""
    if pipe.medium != Medium.UNKNOWN:
        return pipe.medium
    if pipe.line_number:
        parsed = parse_line_number(pipe.line_number)
        if parsed and parsed["system"] in SYSTEM_BY_CODE:
            return SYSTEMS[SYSTEM_BY_CODE[parsed["system"]]].medium
    return default


def doc_system(doc: PIDDocument) -> SystemType:
    return doc.info.system if doc.info.system != SystemType.UNKNOWN else (
        doc.info.systems[0] if doc.info.systems else SystemType.UNKNOWN)


def object_systems(doc: PIDDocument) -> dict[str, SystemType]:
    """System per object: tag system code, else majority of attached pipes, else the drawing system."""
    default = doc_system(doc)
    votes: dict[str, dict[SystemType, int]] = defaultdict(lambda: defaultdict(int))
    for p in doc.pipes:
        if p.kind == PipeKind.SIGNAL:
            continue
        s = pipe_system(p, default)
        if s == SystemType.UNKNOWN:
            continue
        for e in (p.start, p.end):
            if e.object_id:
                votes[e.object_id][s] += 1
    out: dict[str, SystemType] = {}
    for o in doc.objects:
        sys_: Optional[SystemType] = None
        if o.tag:
            t = o.tag.strip().upper()
            m = EQUIPMENT_TAG_RE.match(t) or VALVE_TAG_RE.match(t)
            if m:
                code = next((v for k, v in m.groupdict().items() if k.startswith("sys") and v), None)
                sys_ = SYSTEM_BY_CODE.get(code or "")
        if sys_ is None and votes.get(o.id):
            sys_ = max(votes[o.id].items(), key=lambda kv: (kv[1], kv[0].value))[0]
        out[o.id] = sys_ or default
    return out


# ---------------------------------------------------------------------------- drawing frame


def _extent(pts: list[tuple[float, float]]) -> Optional[tuple[float, float, float, float]]:
    if not pts:
        return None
    xs, ys = zip(*pts)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return (x0, y0, max(x1, x0 + 1.0), max(y1, y0 + 1.0))


def _sheet_numbers(doc: PIDDocument) -> dict[int, int]:
    """Page index -> sheet number from ``doc.meta["sheets"]`` (``[{"page", "sheet", ...}]``), if given."""
    out: dict[int, int] = {}
    raw = doc.meta.get("sheets") if isinstance(doc.meta, dict) else None
    if isinstance(raw, list):
        for s in raw:
            if isinstance(s, dict) and isinstance(s.get("page"), int) and isinstance(s.get("sheet"), int):
                out.setdefault(s["page"], s["sheet"])
    return out


class DrawingFrame:
    """Page point -> normalised drawing position (u right, v down, both in [0, 1]).

    Every page is normalised over its own drawing area (frame interior, else the extent of that
    page's objects and pipes, else the page size). A multi-page document has one slot per page that
    carries objects, pipes or junctions, side by side along u in sheet order (``doc.meta["sheets"]``
    sheet numbers when every used page has one, else page order), each as wide as its share of the
    objects (a sheet with more equipment gets more of the room length): sheets never overlay each
    other and a larger later sheet is not clamped into the frame of the first one. A single-page
    document maps exactly as before (u over the frame width)."""

    def __init__(self, doc: PIDDocument):
        pages_info = {p.index: p for p in doc.pages}
        used: set[int] = {o.page for o in doc.objects} | {j.page for j in doc.junctions}
        used |= {p.page for p in doc.pipes if p.points}
        if not used:
            used = {doc.pages[0].index} if doc.pages else {0}
        sheets = _sheet_numbers(doc)
        if all(pg in sheets for pg in used):
            order = sorted(used, key=lambda pg: (sheets[pg], pg))
        else:
            order = sorted(used)
        self.slots: dict[int, int] = {pg: i for i, pg in enumerate(order)}
        self.n = len(order)
        count: dict[int, int] = defaultdict(int)
        for o in doc.objects:
            count[o.page] += 1
        weights = [max(1, count.get(pg, 0)) for pg in order]
        total = float(sum(weights))
        self.spans: dict[int, tuple[float, float]] = {}  # page -> (u offset, u width)
        acc = 0.0
        for pg, w in zip(order, weights):
            self.spans[pg] = (acc / total, w / total)
            acc += w
        pts: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for o in doc.objects:
            pts[o.page] += [(o.bbox.x0, o.bbox.y0), (o.bbox.x1, o.bbox.y1)]
        for pipe in doc.pipes:
            pts[pipe.page] += [(p.x, p.y) for p in pipe.points]
        self._pts = pts
        self._info = pages_info
        self.frames: dict[int, tuple[float, float, float, float]] = {}
        for pg in order:
            self.frames[pg] = self._frame(pg)

    def _frame(self, pg: int) -> tuple[float, float, float, float]:
        info = self._info.get(pg)
        if info is not None and info.drawing_area is not None:
            b = info.drawing_area
            return (b.x0, b.y0, b.x1, b.y1)
        ext = _extent(self._pts.get(pg, []))
        if ext is not None:
            return ext
        if info is not None:
            return (0.0, 0.0, info.width, info.height)
        return (0.0, 0.0, 1.0, 1.0)

    def frame(self, page: int = 0) -> tuple[float, float, float, float]:
        if page not in self.frames:
            self.frames[page] = self._frame(page)
        return self.frames[page]

    def uv(self, x: float, y: float, page: int = 0) -> tuple[float, float]:
        x0, y0, x1, y1 = self.frame(page)
        u = min(1.0, max(0.0, (x - x0) / max(1e-9, x1 - x0)))
        v = min(1.0, max(0.0, (y - y0) / max(1e-9, y1 - y0)))
        if self.n > 1:
            off, width = self.spans.get(page, (0.0, 1.0 / self.n))
            u = off + width * u
        return (u, v)


def drawing_area(doc: PIDDocument, page: Optional[int] = None) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of the drawing area of ``page`` (default: the first used sheet): frame
    interior, else the extent of that page's objects and pipes, else the page size."""
    fr = DrawingFrame(doc)
    if page is None:
        page = min(fr.slots, key=lambda pg: fr.slots[pg])
    return fr.frame(page)


def normalized_position(doc: PIDDocument, x: float, y: float, page: int = 0) -> tuple[float, float]:
    """Page point -> (u, v) in [0, 1]^2 (see :class:`DrawingFrame`; build one frame when mapping
    many points)."""
    return DrawingFrame(doc).uv(x, y, page)


def drawing_positions(doc: PIDDocument) -> dict[str, tuple[float, float]]:
    """Normalised bbox centre of every object (bow right / port top convention; sheets side by side)."""
    fr = DrawingFrame(doc)
    return {o.id: fr.uv(o.center.x, o.center.y, o.page) for o in doc.objects}


# ---------------------------------------------------------------------------- 3D document

SHEET_LINK_PREFIX = "sheet_link:"
SHEET_LINK_PORT = "sheet_link"
PREPARED_KEY = "model3d_prepared"
# pipe-connected unknown symbols planned as equipment: at least this many, at most as many as the
# recognised planned objects (larger symbols first); the rest is placed in the pipe run
UNKNOWN_PLAN_MIN = 12


def is_sheet_link(pipe: PipeRun) -> bool:
    return pipe.id.startswith(SHEET_LINK_PREFIX)


def sheet_link_runs(doc: PIDDocument) -> list[PipeRun]:
    """One zero-length process :class:`PipeRun` ``sheet_link:<a>:<b>`` per valid pair of
    ``doc.meta["sheet_links"]`` (:func:`shippid.analysis.graph.sheet_link_pairs`), from ``a`` to
    ``b`` (port ``sheet_link`` on both), with the size / medium of the largest pipe attached to
    either connector, the line number when both sides carry the same one, and flow ``a -> b`` when
    the link is directed. Pairs already present as pipes are skipped."""
    from ..analysis.graph import sheet_link_pairs

    pairs = sheet_link_pairs(doc)
    if not pairs:
        return []
    have = {p.id for p in doc.pipes}
    objs = doc.object_by_id()
    attached: dict[str, list[PipeRun]] = defaultdict(list)
    for p in doc.pipes:
        if p.kind == PipeKind.SIGNAL or is_sheet_link(p):
            continue
        for e in (p.start, p.end):
            if e.object_id:
                attached[e.object_id].append(p)
    out: list[PipeRun] = []
    for a, b, lk in pairs:
        pid = f"{SHEET_LINK_PREFIX}{a}:{b}"
        if pid in have:
            continue
        side = attached.get(a, []) + attached.get(b, [])
        ref = max(side, key=lambda p: (pipe_dn(p) or 0, p.medium != Medium.UNKNOWN), default=None)
        size = None
        if ref is not None:
            size = ref.size or ((parse_line_number(ref.line_number) or {}).get("size") if ref.line_number else None)
        lines_a = {p.line_number for p in attached.get(a, []) if p.line_number}
        lines_b = {p.line_number for p in attached.get(b, []) if p.line_number}
        common = sorted(lines_a & lines_b)
        medium = next((pipe_medium(p) for p in side if pipe_medium(p) != Medium.UNKNOWN), Medium.UNKNOWN)
        c = objs[a].center
        conf = lk.get("confidence")
        out.append(PipeRun(id=pid, page=objs[a].page, points=[Point(x=c.x, y=c.y)],
                           start=Endpoint(object_id=a, port=SHEET_LINK_PORT),
                           end=Endpoint(object_id=b, port=SHEET_LINK_PORT), kind=PipeKind.PROCESS,
                           line_number=common[0] if common else None, size=size, medium=medium,
                           flow_direction=1 if lk.get("directed") else None,
                           confidence=float(conf) if isinstance(conf, (int, float)) else 1.0, source="sheet_link"))
    return out


def _bbox_area(o: PIDObject) -> float:
    return max(0.0, o.bbox.x1 - o.bbox.x0) * max(0.0, o.bbox.y1 - o.bbox.y0)


def model3d_roles(doc: PIDDocument) -> dict[str, str]:
    """Objects the class rules would plan as equipment that the 3D stage does not plan:

    * ``"sheet_link"``: off-page connector paired with another one (a pass-through continuation);
    * ``"unconnected_unknown"``: unrecognised symbol (class ``unknown``) without any process pipe:
      a small marker at its drawing position (placed by the router);
    * ``"unknown_not_modelled"``: such symbols beyond the cap (mostly extractor noise such as
      hatching or SHX text strokes): left out of the 3D model (one aggregated note);
    * ``"unknown_over_cap"``: pipe-connected unrecognised symbol beyond the cap: placed in its pipe
      run by the router instead of as an equipment box.

    The cap is the larger of :data:`UNKNOWN_PLAN_MIN` and the number of recognised planned objects,
    separately for connected and unconnected symbols; larger symbols are kept first."""
    if doc.meta.get(PREPARED_KEY):
        return {o.id: o.attributes[MODEL3D_ROLE] for o in doc.objects if o.attributes.get(MODEL3D_ROLE)}
    from ..analysis.graph import sheet_link_pairs

    roles: dict[str, str] = {}
    for a, b, _ in sheet_link_pairs(doc):
        roles[a] = roles[b] = "sheet_link"
    degree: dict[str, int] = defaultdict(int)
    for p in doc.pipes:
        if p.kind == PipeKind.SIGNAL or not p.points:
            continue
        for e in (p.start, p.end):
            if e.object_id:
                degree[e.object_id] += 1
    unknown = [o for o in doc.objects if o.id not in roles and o.cls == SymbolClass.UNKNOWN
               and o.category == Category.EQUIPMENT and is_planned(o)]
    connected = [o for o in unknown if degree.get(o.id, 0) > 0]
    loose = [o for o in unknown if degree.get(o.id, 0) == 0]
    recognised = sum(1 for o in doc.objects if o.id not in roles and o.cls != SymbolClass.UNKNOWN and is_planned(o))
    cap = max(UNKNOWN_PLAN_MIN, recognised)
    for group, kept_role, over_role in ((connected, None, "unknown_over_cap"),
                                        (loose, "unconnected_unknown", "unknown_not_modelled")):
        keep = {o.id for o in sorted(group, key=lambda o: -_bbox_area(o))[:cap]}  # stable: document order on ties
        for o in group:
            role = kept_role if o.id in keep else over_role
            if role:
                roles[o.id] = role
    return roles


def _id_list(ids: list[str], k: int = 12) -> str:
    return ", ".join(ids[:k]) + (f" (+{len(ids) - k} more)" if len(ids) > k else "")


def prepare_document(doc: PIDDocument, pair_sheets: bool = False) -> tuple[PIDDocument, list[str]]:
    """The document as the 3D stage sees it, and notes on what changed (idempotent; the input is
    not modified: lists, meta and the flagged objects are copied).

    * ``pair_sheets``: a multi-page document without ``meta["sheet_links"]`` gets its off-page
      connectors paired first (:func:`shippid.analysis.sheets.link_sheets`, as the analysis does);
    * every sheet link becomes a pipe (:func:`sheet_link_runs`) appended to ``pipes``;
    * the objects of :func:`model3d_roles` carry ``attributes["model3d_role"]`` (so
      :func:`shippid.model3d.plan.is_planned` leaves them out) and linked connectors also
      ``attributes["model3d_partner"]``;
    * ``meta["model3d_prepared"]`` = True."""
    if doc.meta.get(PREPARED_KEY):
        return doc, []
    notes: list[str] = []
    meta = dict(doc.meta)
    d = doc.model_copy(update={"meta": meta, "pipes": list(doc.pipes), "objects": list(doc.objects)})
    pages = {p.index for p in doc.pages} | {o.page for o in doc.objects}
    if pair_sheets and len(pages) > 1 and "sheet_links" not in meta:
        try:
            from ..analysis.sheets import link_sheets

            links = link_sheets(d)
            notes.append(f"sheets: {len(pages)} pages, {len(links)} off-page connector pair(s) linked across the "
                         f"sheets by the deterministic connector matcher for the 3D model")
        except Exception as exc:  # pairing is an improvement, never a reason to fail the 3D stage
            meta.pop("sheet_links", None)
            notes.append(f"sheets: off-page connector pairing failed ({type(exc).__name__}); sheets kept separate")
    runs = sheet_link_runs(d)
    d.pipes.extend(runs)
    roles = model3d_roles(d)
    partner = {}
    for p in runs:
        partner[p.start.object_id] = p.end.object_id
        partner[p.end.object_id] = p.start.object_id
    if roles:
        new = []
        for o in d.objects:
            if o.id in roles:
                attrs = dict(o.attributes)
                attrs[MODEL3D_ROLE] = roles[o.id]
                if o.id in partner:
                    attrs["model3d_partner"] = partner[o.id]
                o = o.model_copy(update={"attributes": attrs})
            new.append(o)
        d.objects = new
    meta[PREPARED_KEY] = True
    if runs:
        notes.append(f"sheets: {len(runs)} linked off-page connector pair(s) routed as one continuous pipe "
                     f"(route ids sheet_link:<a>:<b>), both connectors shown as markers on it: "
                     + _id_list([f"{p.start.object_id}-{p.end.object_id}" for p in runs], 8))
    by_role: dict[str, list[str]] = defaultdict(list)
    for oid, r in roles.items():
        by_role[r].append(oid)
    shown, dropped = by_role.get("unconnected_unknown", []), by_role.get("unknown_not_modelled", [])
    if shown or dropped:
        notes.append(f"{len(shown) + len(dropped)} unrecognised symbol(s) (class unknown) without a pipe connection "
                     f"not planned as equipment: {len(shown)} largest shown as small markers at their drawing "
                     f"position" + (f": {_id_list(shown)}" if shown else "")
                     + (f"; {len(dropped)} left out of the 3D model (likely extraction noise such as hatching or "
                        f"SHX text strokes): {_id_list(dropped)}" if dropped else ""))
    if by_role.get("unknown_over_cap"):
        ids = by_role["unknown_over_cap"]
        n_unknown = len(ids) + sum(1 for o in d.objects if o.cls == SymbolClass.UNKNOWN and is_planned(o))
        notes.append(f"{n_unknown} pipe-connected unrecognised symbol(s): the {n_unknown - len(ids)} largest planned "
                     f"as equipment, {len(ids)} placed in their pipe run by the router: " + _id_list(ids))
    return d, notes


def planned_ids(doc: PIDDocument) -> list[str]:
    """Ids the planner must place, in document order: :func:`shippid.model3d.plan.is_planned`
    minus the objects of :func:`model3d_roles`."""
    roles = model3d_roles(doc)
    return [o.id for o in doc.objects if is_planned(o) and o.id not in roles]


_planned_ids = planned_ids  # equipment_adjacency's parameter shadows the name


# ---------------------------------------------------------------------------- connectivity

_OUT_PORTS = {"out", "discharge", "tube_out", "shell_out", "a_out", "b_out", "outlet"}
_IN_PORTS = {"in", "suction", "tube_in", "shell_in", "a_in", "b_in", "inlet"}


def pipe_direction(pipe: PipeRun, neutral: frozenset[str] = frozenset()) -> Optional[int]:
    """+1 if flow goes start->end, -1 if end->start, None if unknown (from flow_direction or port
    names). Port names of the objects in ``neutral`` say nothing about the flow (an off-page
    connector's only port is called "in" whether the line leaves or enters the sheet)."""
    if pipe.flow_direction in (1, -1):
        return pipe.flow_direction
    sp, ep = (pipe.start.port or "").lower(), (pipe.end.port or "").lower()
    s_obj = pipe.start.object_id is not None and pipe.start.object_id not in neutral
    e_obj = pipe.end.object_id is not None and pipe.end.object_id not in neutral
    fwd = (s_obj and sp in _OUT_PORTS) or (e_obj and ep in _IN_PORTS)
    back = (s_obj and sp in _IN_PORTS) or (e_obj and ep in _OUT_PORTS)
    if fwd and not back:
        return 1
    if back and not fwd:
        return -1
    return None


def pipe_dn(pipe: PipeRun) -> Optional[int]:
    """Nominal size (DN / JIS A) of a pipe from its size or line number."""
    for s in (pipe.size, (parse_line_number(pipe.line_number) or {}).get("size") if pipe.line_number else None):
        if s:
            m = re.search(r"(\d{2,3})", s)
            if m:
                return int(m.group(1))
    return None


@dataclass
class Neighbor:
    """A planned object reachable from another one without passing through planned objects."""

    id: str
    pipes: list[str] = field(default_factory=list)
    via: list[str] = field(default_factory=list)  # inline object ids on the way
    lines: list[str] = field(default_factory=list)
    dn: Optional[int] = None
    medium: Medium = Medium.UNKNOWN
    direction: Optional[str] = None  # "out" (flow from the object to the neighbour), "in", None
    port: Optional[str] = None  # port of the object the path leaves from


def process_pipes(doc: PIDDocument) -> list[PipeRun]:
    return [p for p in doc.pipes if p.kind != PipeKind.SIGNAL and len(p.points) >= 1]


def equipment_adjacency(doc: PIDDocument, planned_ids: Optional[set[str]] = None) -> dict[str, list[Neighbor]]:
    """Object-adjacency summary between planned objects through inline objects and junctions.

    For each planned object, a BFS over pipes (signal lines excluded) that stops at other
    planned objects. Flow direction is aggregated from the pipes on the path. Paired off-page
    connectors are passed through like inline items (their sheet link is a zero-length pipe, see
    :func:`prepare_document`), so neighbours on both sheets connect (``via`` lists the connectors).
    """
    doc, _ = prepare_document(doc)
    objs = doc.object_by_id()
    if planned_ids is None:
        planned_ids = set(_planned_ids(doc))
    adj: dict[str, list[tuple[str, PipeRun, int, Optional[str]]]] = defaultdict(list)
    connectors = frozenset(o.id for o in doc.objects
                           if o.cls == SymbolClass.OFF_PAGE_CONNECTOR or o.category == Category.CONNECTOR)
    for p in process_pipes(doc):
        a, b = p.start.node_id, p.end.node_id
        if a is None or b is None or a == b:
            continue
        d = pipe_direction(p, connectors)
        adj[a].append((b, p, d or 0, p.start.port if p.start.object_id else None))
        adj[b].append((a, p, -(d or 0), p.end.port if p.end.object_id else None))
    out: dict[str, list[Neighbor]] = {}
    for src in planned_ids:
        found: dict[str, Neighbor] = {}
        fwd_votes: dict[str, list[int]] = defaultdict(list)
        # state: node, pipes on path, via objects, first port
        queue = deque([(src, [], [], [], None)])
        visited = {src}
        while queue:
            node, pipes, via, dirs, port = queue.popleft()
            for nb, pipe, d, from_port in adj.get(node, []):
                if nb in visited and nb not in planned_ids:
                    continue
                npipes = pipes + [pipe]
                ndirs = dirs + [d]
                nport = port if node != src else from_port
                if nb in planned_ids:
                    if nb == src:
                        continue
                    n = found.get(nb)
                    if n is None:
                        n = found[nb] = Neighbor(id=nb, port=nport)
                    for pp in npipes:
                        if pp.id not in n.pipes:
                            n.pipes.append(pp.id)
                        if pp.line_number and pp.line_number not in n.lines:
                            n.lines.append(pp.line_number)
                        dn = pipe_dn(pp)
                        if dn and (n.dn is None or dn > n.dn):
                            n.dn = dn
                        if n.medium == Medium.UNKNOWN:
                            n.medium = pipe_medium(pp)
                    for v in via:
                        if v not in n.via:
                            n.via.append(v)
                    fwd_votes[nb].extend(ndirs)
                    continue
                visited.add(nb)
                nvia = via + ([nb] if nb in objs else [])
                if len(npipes) < 40:
                    queue.append((nb, npipes, nvia, ndirs, nport))
        for nb, n in found.items():
            votes = fwd_votes[nb]
            pos, neg = sum(1 for v in votes if v > 0), sum(1 for v in votes if v < 0)
            n.direction = "out" if pos and not neg else ("in" if neg and not pos else None)
        out[src] = sorted(found.values(), key=lambda n: n.id)
    return out


def flow_order(doc: PIDDocument, planned_ids: list[str], adjacency: dict[str, list[Neighbor]]) -> list[str]:
    """Planned ids in flow order: topological over the condensation of the directed adjacency,
    ties broken by drawing position (left to right, top to bottom)."""
    pos = drawing_positions(doc)
    g = nx.DiGraph()
    g.add_nodes_from(planned_ids)
    for a in planned_ids:
        for n in adjacency.get(a, []):
            if n.id in g and n.direction == "out":
                g.add_edge(a, n.id)
    cond = nx.condensation(g)
    members = cond.graph["mapping"]
    comp_nodes: dict[int, list[str]] = defaultdict(list)
    for node, c in members.items():
        comp_nodes[c].append(node)

    def key(c: int) -> tuple:
        return min((pos.get(n, (0.5, 0.5))[0], pos.get(n, (0.5, 0.5))[1], n) for n in comp_nodes[c])

    order: list[str] = []
    for c in nx.lexicographical_topological_sort(cond, key=key):
        order.extend(sorted(comp_nodes[c], key=lambda n: (pos.get(n, (0.5, 0.5)), n)))
    return order


# ---------------------------------------------------------------------------- rules

_DB_TANK = re.compile(r"D\.?\s?B\b|DOUBLE\s*BOTTOM|BILGE|SLUDGE|DRAIN|SUMP|\bWELL\b|OVERFLOW|LEAK|COFFERDAM|"
                      r"W\.?\s?B\.?\s?T|WATER\s*BALLAST|BALLAST|PEAK|DEEP\s*TK|CARGO|SEPARATED|DIRTY|WASTE|SEWAGE")
_REMOTE_TANK = re.compile(r"W\.?\s?B\.?\s?T|WATER\s*BALLAST|BALLAST\s*T|PEAK|CARGO|DEEP\s*TK|\bHOLD\b|WING\s*T")
_ELEVATED_TANK = re.compile(r"SETT|SERV|\bDAY\b|SETTLING|SERVICE|GRAVITY")
_HIGH_TANK = re.compile(r"EXPANSION|HEADER|\bEXP\b|DEAERAT|CASCADE|OBSERVATION|HOT\s*WELL")
_PORT = re.compile(r"\(P\)|\bP\)|\bPORT\b|\(PS\)|\bP/S\b")
_STBD = re.compile(r"\(S\)|\bS\)|\bSTBD\b|\(SB\)|STARBOARD|\bS/S\b")
_CENTER = re.compile(r"\(C\)|\bCENTER\b|\bCENTRE\b|\(C/L\)")
_MAIN_ENGINE = re.compile(r"\bM/E\b|MAIN\s*ENGINE|\bME\b|PROPULSION")
_GEN_ENGINE = re.compile(r"\bG/E\b|\bD/G\b|GENERATOR|\bGE\b|AUX\.?\s*ENGINE|\bA/E\b|DIESEL\s*GEN")
_BOILER = re.compile(r"BOILER|ECONOMI[SZ]ER|\bEGE\b|COMPOSITE")
_LOW = re.compile(r"\bLOW\b")
_HIGH = re.compile(r"\bHIGH\b")


def name_side(label: str) -> Optional[str]:
    up = label.upper()
    if _PORT.search(up):
        return "port"
    if _STBD.search(up):
        return "starboard"
    if _CENTER.search(up):
        return "center"
    return None


def drawing_side(v: float) -> str:
    return "port" if v < 1 / 3 else ("starboard" if v > 2 / 3 else "center")


def drawing_long(u: float) -> str:
    return "aft" if u < 1 / 3 else ("fore" if u > 2 / 3 else "mid")


def _outer_side(side: str, v: float) -> str:
    """Side for items that belong at the ship side: never the centre band."""
    if side != "center":
        return side
    return "port" if v <= 0.5 else "starboard"


DEFAULT_ELEVATIONS: dict[Medium, tuple[str, float]] = {
    # (deck, height above deck in m) of the preferred pipe centre line -- HEURISTIC: mains of
    # sea water / ballast / bilge run under the floor plates on the tank top, oil and water
    # services under the platforms, air and steam high.
    Medium.BALLAST_WATER: ("tank_top", 1.2),
    Medium.BILGE: ("tank_top", 0.8),
    Medium.SEA_WATER: ("tank_top", 1.5),
    Medium.FIRE_WATER: ("floor", 2.6),
    Medium.FRESH_WATER: ("floor", 2.8),
    Medium.FUEL_OIL: ("floor", 2.4),
    Medium.LUBE_OIL: ("floor", 2.2),
    Medium.COMPRESSED_AIR: ("floor", 3.0),
    Medium.STEAM: ("3rd_deck", 2.6),
    Medium.UNKNOWN: ("floor", 2.5),
}


def default_routing(doc: PIDDocument) -> RoutingPlan:
    """Elevation band per medium present in the drawing and priority = line numbers by size."""
    media = sorted({pipe_medium(p) for p in process_pipes(doc)}, key=lambda m: m.value)
    bands = [ElevationBand(target=m.value, deck=DEFAULT_ELEVATIONS[m][0], height_m=DEFAULT_ELEVATIONS[m][1])
             for m in media]
    sizes: dict[str, int] = {}
    for p in process_pipes(doc):
        if p.line_number:
            sizes[p.line_number] = max(sizes.get(p.line_number, 0), pipe_dn(p) or 0)
    priority = [ln for ln, _ in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))]
    return RoutingPlan(elevation_bands=bands, priority=priority)


def _rule(obj: PIDObject, label: str, uv: tuple[float, float]) -> dict:
    """Zone / band / relation-free placement rule for one planned object."""
    u, v = uv
    up = label.upper()
    nside = name_side(label)
    side = nside or drawing_side(v)
    long_ = drawing_long(u)
    cls = obj.cls
    r: dict = {"zone": "floor", "side": side, "longitudinal": long_, "orientation_deg": 0, "rationale": ""}
    if cls == S.TANK:
        if _HIGH_TANK.search(up):
            r.update(zone="upper_platform", rationale="expansion/header tank kept high for static head")
        elif _ELEVATED_TANK.search(up):
            r.update(zone="floor", side=_outer_side(side, v),
                     rationale="settling/service tank elevated at the ship side (gravity head to the pumps)")
        elif _REMOTE_TANK.search(up):
            r.update(zone="double_bottom", longitudinal="fore", side=nside or side,
                     rationale="tank outside the engine room (cargo area / peak); represented in the forward "
                               "double bottom")
        elif _DB_TANK.search(up):
            r.update(zone="double_bottom", rationale="double-bottom tank (drains by gravity, lowest point)")
        else:
            r.update(zone="tank_top", side=_outer_side(side, v),
                     rationale="storage tank at the ship side on the tank top")
    elif cls == S.SEA_CHEST:
        r.update(zone="bottom", side=nside or _outer_side(side, v), longitudinal="aft" if u < 0.5 else "mid",
                 rationale="sea chest in the hull bottom (low) / side (high) below the waterline")
        if _LOW.search(up) and not nside:
            r["side"] = "center"
    elif cls == S.OVERBOARD:
        sd = nside or _outer_side(side, v)
        r.update(zone="ship_side_port" if sd == "port" else "ship_side_starboard", side=sd, deck="floor",
                 rationale="overboard discharge through the shell with shell valve")
    elif cls in (S.PUMP_CENTRIFUGAL, S.PUMP_SCREW):
        r.update(zone="tank_top", rationale="pump on the tank top near its suction source (positive suction head)")
    elif cls == S.ENGINE:
        if _BOILER.search(up):
            r.update(zone="upper_platform", side="center" if not nside else nside, longitudinal="aft",
                     rationale="boiler / exhaust-gas economiser high in the casing near the funnel")
        elif _GEN_ENGINE.search(up):
            r.update(zone="floor", side=_outer_side(side, v), longitudinal="fore",
                     rationale="generator engine on the floor deck")
        elif _MAIN_ENGINE.search(up) or "ENGINE" in up or not up.strip():
            r.update(zone="tank_top", side="center", longitudinal="aft",
                     rationale="main engine on the centre line on the tank top, in line with the shaft")
        else:
            r.update(zone="floor", rationale="consumer on the floor deck")
    elif cls in (S.HEAT_EXCHANGER, S.PLATE_HEAT_EXCHANGER):
        r.update(zone="floor", rationale="cooler / heat exchanger on the floor deck near its pumps")
    elif cls == S.HEATER:
        r.update(zone="middle_platform", rationale="heater next to the purifiers / service pumps")
    elif cls == S.PURIFIER:
        r.update(zone="middle_platform", rationale="purifier room on the 3rd-deck platform")
    elif cls == S.COMPRESSOR:
        r.update(zone="floor", rationale="air compressor on the floor deck")
    elif cls == S.AIR_RECEIVER:
        r.update(zone="floor", rationale="vertical air receiver next to the compressors")
    elif cls == S.BWTS_UNIT:
        r.update(zone="floor", rationale="BWTS unit on the floor deck downstream of the ballast pumps")
    elif cls == S.OILY_WATER_SEPARATOR:
        r.update(zone="tank_top", rationale="oily water separator on the tank top near the bilge tanks")
    elif cls == S.DUPLEX_FILTER:
        r.update(zone="floor", rationale="duplex / auto-backflush filter on the floor deck in the pump line")
    elif cls == S.OFF_PAGE_CONNECTOR:
        r.update(zone="floor", rationale="off-page connector: pipe continues on another drawing")
    else:
        r.update(zone="floor", rationale="equipment on the floor deck (no specific rule)")
    return r


def heuristic_plan(doc: PIDDocument, room: EngineRoom, ids: Optional[list[str]] = None,
                   adjacency: Optional[dict[str, list[Neighbor]]] = None) -> LayoutPlan:
    """Deterministic layout plan for all planned objects (:func:`planned_ids`; or only ``ids``, in
    plan order)."""
    doc, _ = prepare_document(doc)
    objs = doc.object_by_id()
    planned = _planned_ids(doc)
    if ids is not None:
        extra = [i for i in ids if i in objs and i not in planned]
        planned = planned + extra
    adjacency = adjacency if adjacency is not None else equipment_adjacency(doc, set(planned))
    labels = object_labels(doc)
    pos = drawing_positions(doc)
    order = flow_order(doc, planned, adjacency)
    engines = [i for i in order if objs[i].cls == S.ENGINE]
    order = engines + [i for i in order if i not in engines]
    connectors = [i for i in order if objs[i].cls == S.OFF_PAGE_CONNECTOR]
    order = [i for i in order if i not in connectors] + connectors
    if ids is not None:
        wanted = set(ids)
        order = [i for i in order if i in wanted]

    items: list[PlanItem] = []
    placed: dict[str, PlanItem] = {}
    group_last: dict[str, str] = {}
    for oid in order:
        obj = objs[oid]
        label = labels.get(oid, "")
        r = _rule(obj, label, pos[oid])
        anchor, relation, max_m, group = None, "none", None, None
        nbs = [n for n in adjacency.get(oid, []) if n.id in placed]
        ins = [n for n in nbs if n.direction == "in"]
        nearest = sorted(nbs, key=lambda n: _ddist(pos, oid, n.id))
        if obj.cls in (S.PUMP_CENTRIFUGAL, S.PUMP_SCREW, S.PURIFIER, S.COMPRESSOR):
            sources = sorted({n.id for n in adjacency.get(oid, []) if n.direction == "in"})
            group = f"{obj.cls.value}:{'+'.join(sources[:3]) or label[:20] or oid}"
            if group in group_last:
                anchor, relation, max_m = group_last[group], "adjacent_to", 1.5
                prev = placed[anchor]
                r.update(zone=prev.zone, side=prev.side, longitudinal=prev.longitudinal)
            else:
                src = sorted(ins, key=lambda n: _ddist(pos, oid, n.id)) or nearest
                if src:
                    anchor, relation, max_m = src[0].id, "near", 6.0
                    a = placed[src[0].id]
                    if obj.cls in (S.PUMP_CENTRIFUGAL, S.PUMP_SCREW) and a.zone in ("double_bottom", "bottom",
                                                                                     "tank_top", "floor"):
                        r.update(side=a.side if a.side != "center" or obj.cls != S.PUMP_CENTRIFUGAL else r["side"],
                                 longitudinal=a.longitudinal)
            group_last[group] = oid
        elif obj.cls == S.AIR_RECEIVER:
            comp = [n for n in nbs if objs[n.id].cls == S.COMPRESSOR] or nearest
            if comp:
                anchor, relation, max_m = comp[0].id, "adjacent_to", 2.0
                a = placed[comp[0].id]
                r.update(zone=a.zone if a.zone in ("floor", "tank_top") else r["zone"], side=a.side,
                         longitudinal=a.longitudinal)
        elif obj.cls == S.HEATER:
            pur = [n for n in nbs if objs[n.id].cls == S.PURIFIER]
            if pur:
                anchor, relation, max_m = pur[0].id, "adjacent_to", 2.0
            elif nearest:
                anchor, relation, max_m = nearest[0].id, "near", 6.0
        elif obj.cls in (S.HEAT_EXCHANGER, S.PLATE_HEAT_EXCHANGER, S.BWTS_UNIT, S.DUPLEX_FILTER,
                         S.OILY_WATER_SEPARATOR) and (ins or nearest):
            src = sorted(ins, key=lambda n: _ddist(pos, oid, n.id)) or nearest
            anchor, relation, max_m = src[0].id, "near", 6.0
            a = placed[src[0].id]
            if a.zone not in ("ship_side_port", "ship_side_starboard", "upper_platform"):
                r.update(side=a.side, longitudinal=a.longitudinal)
        elif obj.cls == S.OFF_PAGE_CONNECTOR and nearest:
            a = placed[nearest[0].id]
            zone = a.zone if a.zone in ("tank_top", "floor", "middle_platform", "upper_platform") else "tank_top"
            r.update(zone=zone, side=a.side, longitudinal=a.longitudinal)
            anchor, relation, max_m = nearest[0].id, "near", 4.0
        item = PlanItem(object_id=oid, zone=r["zone"], deck=r.get("deck"), side=r["side"],
                        longitudinal=r["longitudinal"], anchor_id=anchor, relation=relation, max_m=max_m,
                        orientation_deg=r["orientation_deg"], group=group, rationale=r["rationale"],
                        source="heuristic")
        items.append(item)
        placed[oid] = item
    notes = ["heuristic arrangement from marine rules of thumb and the drawing position (bow right, port top)"]
    return LayoutPlan(items=items, routing=default_routing(doc), notes=notes, planner="heuristic")


def _ddist(pos: dict[str, tuple[float, float]], a: str, b: str) -> float:
    pa, pb = pos.get(a, (0.5, 0.5)), pos.get(b, (0.5, 0.5))
    return abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
