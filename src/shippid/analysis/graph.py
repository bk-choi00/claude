"""Pipe-network graph of a P&ID and deterministic flow analysis.

:func:`build_graph` turns a :class:`~shippid.model.PIDDocument` into an undirected
:class:`networkx.MultiGraph`:

* object nodes (``kind="object"``) carry ``cls``, ``category``, ``tag``, ``label`` (equipment
  name / connector text) and ``eff_cls`` (the class used by the analysis: the extracted class,
  or a class inferred from the label when the extractor left the object ``UNKNOWN``);
* junction nodes (``kind="junction"``);
* ``free:<pipe_id>:start|end`` nodes (``kind="free"``) for pipe ends attached to nothing (or to
  an id that does not exist in the document);
* one edge per :class:`~shippid.model.PipeRun` (key = pipe id) with ``pipe_id``, ``kind``,
  ``medium``, ``line_number``, ``size``, ``length``, ``flow_direction`` and the port name at each
  end (``start_port`` / ``end_port``). Ports the extractor did not name are inferred from the
  symbol geometry (rotation, mirroring, library port positions).

Flow semantics (used by :func:`flow_paths` and the rules):

* only ``PROCESS`` pipes carry flow; pipe ``flow_direction`` is respected when set;
* tanks, sea chests, overboards, engines and off-page connectors end a path (never passed
  through); instruments are never passed through;
* pumps and compressors pass flow only from suction/inlet to discharge/outlet, check / SDNR /
  safety valves only from ``in`` to ``out``; heat exchangers only within one circuit;
* sources: tanks, sea chests, compressors, ``FROM ...`` connectors, boilers; sinks: tanks,
  overboards, engines, ``TO ...`` connectors; single-connection equipment is both;
* a flow path goes source -> at least one driver (pump / compressor) -> sink. Drawings without
  any driver fall back to undriven source -> sink paths.

Multi-sheet sets: every pair of off-page connectors in ``doc.meta["sheet_links"]`` (see
:mod:`shippid.analysis.sheets`) becomes one more edge, ``kind="sheet_link"`` (key / ``pipe_id``
``sheet_link:<a>:<b>``, zero length, ``flow_direction`` +1 from ``a`` to ``b`` when the link is
``directed``). The edge kind is part of ``PROCESS_KINDS`` / ``CONNECT_KINDS`` / ``ALL_KINDS``, and
paired connectors (node attribute ``sheet_link=True``, ``linked_to``) are passed through like
junctions: they are neither sources nor sinks, flow paths, isolation searches, adjacency and
components continue on the partner's sheet. ``build_graph(doc, with_sheet_links=False)`` gives
the per-sheet graph.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import networkx as nx

from ..model import (
    Category,
    FlowPath,
    Medium,
    PIDDocument,
    PIDObject,
    PipeKind,
    Point,
    SymbolClass,
    TextRole,
)
from ..symbols import ACTUATOR_PRIMS, Circle, Line, Poly, Rect, SymbolSpec, category_of, get_symbol

S = SymbolClass

#: edge kind of a cross-sheet continuation between two paired off-page connectors (not a PipeKind:
#: no pipe of the document carries it)
SHEET_LINK = "sheet_link"
PROCESS_KINDS = frozenset({PipeKind.PROCESS, SHEET_LINK})
CONNECT_KINDS = frozenset({PipeKind.PROCESS, PipeKind.INSTRUMENT_TAP, SHEET_LINK})
ALL_KINDS = frozenset(PipeKind) | {SHEET_LINK}

PUMP_CLASSES = frozenset({S.PUMP_CENTRIFUGAL, S.PUMP_SCREW})
DRIVER_CLASSES = PUMP_CLASSES | {S.COMPRESSOR}
NRV_CLASSES = frozenset({S.CHECK_VALVE, S.SDNR_VALVE})
TERMINAL_CLASSES = frozenset({S.TANK, S.SEA_CHEST, S.OVERBOARD, S.ENGINE, S.OFF_PAGE_CONNECTOR})
STRAINER_CLASSES = frozenset({S.STRAINER, S.FILTER, S.DUPLEX_FILTER})
INLINE_EQUIPMENT = STRAINER_CLASSES | {S.FLOW_METER}
INSTRUMENT_CLASSES = frozenset({S.INSTRUMENT_LOCAL, S.INSTRUMENT_PANEL, S.INSTRUMENT_SHARED})
#: valves that can be closed to isolate equipment (check and relief valves cannot)
ISOLATING_VALVES = frozenset({S.GATE_VALVE, S.GLOBE_VALVE, S.BALL_VALVE, S.BUTTERFLY_VALVE, S.SDNR_VALVE,
                              S.THREE_WAY_VALVE})
DIRECTIONAL_CLASSES = DRIVER_CLASSES | NRV_CLASSES | {S.SAFETY_VALVE}
INLET_PORTS = frozenset({"in", "suction"})
OUTLET_PORTS = frozenset({"out", "discharge"})
CIRCUITS: dict[SymbolClass, tuple[frozenset[str], ...]] = {
    S.HEAT_EXCHANGER: (frozenset({"tube_in", "tube_out"}), frozenset({"shell_in", "shell_out"})),
    S.PLATE_HEAT_EXCHANGER: (frozenset({"a_in", "a_out"}), frozenset({"b_in", "b_out"})),
}

# Label keywords used only for objects the extractor left UNKNOWN (order matters:
# "OWS FEED PUMP" is a pump, "L.O. PURIFIER HEATER" a heater, "M/E L.O. SUMP TK" a tank).
_LABEL_CLASSES: tuple[tuple[re.Pattern[str], SymbolClass], ...] = tuple(
    (re.compile(p), c)
    for p, c in [
        (r"\bPUMPS?\b|\bP/P\b|펌프", S.PUMP_CENTRIFUGAL),
        (r"COMPRESSOR|\bAIR\s*COMP\b|압축기", S.COMPRESSOR),
        (r"SEA\s*CHEST|SEA\s*INLET|씨체스트", S.SEA_CHEST),
        (r"OVERBOARD|\bO\.?B\.?D\b|선외", S.OVERBOARD),
        (r"STRAINER|MUD\s*BOX", S.STRAINER),
        (r"FILTER", S.FILTER),
        (r"HEATER", S.HEATER),
        (r"COOLER|HEAT\s*EXCH", S.HEAT_EXCHANGER),
        (r"PURIFIER", S.PURIFIER),
        (r"SEPARATOR|\bO\.?W\.?S\b", S.OILY_WATER_SEPARATOR),
        (r"RECEIVER|AIR\s*BOTTLE|AIR\s*RESERVOIR", S.AIR_RECEIVER),
        (r"\bTK\b|TANK|\bWELL\b|COFFERDAM|탱크", S.TANK),
        (r"\bM/E\b|\bG/E\b|ENGINE|BOILER|ECONOMI[SZ]ER", S.ENGINE),
    ]
)
_SOURCE_ENGINE_RE = re.compile(r"BOILER|ECONOMI[SZ]ER|EVAPORATOR|F\.?\s?W\.?\s+GENERATOR")
_FROM_RE = re.compile(r"(?:^|[\s(])(FROM|FR\.)\s")
_TO_RE = re.compile(r"(?:^|[\s(/])TO\s")
_NAME_KEYS = ("name", "label", "title")


# --------------------------------------------------------------------------- labels / classes


def object_labels(doc: PIDDocument) -> dict[str, str]:
    """Human-readable name of each object: ``attributes['name'|'label'|'title']``, else the
    LABEL / CONNECTOR texts owned by the object (reading order)."""
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
                parts.append(v.strip())
            elif isinstance(v, (list, tuple)):
                parts.extend(str(x).strip() for x in v if str(x).strip())
        if not parts:
            ts = sorted(owned.get(o.id, []), key=lambda t: (round(t.bbox.y0, 1), t.bbox.x0))
            parts = [" ".join(t.text.split()) for t in ts]
        if parts:
            out[o.id] = " ".join(dict.fromkeys(parts))
    return out


def effective_class(obj: PIDObject, label: Optional[str] = None) -> SymbolClass:
    """Class used by the analysis: the object's class, or for ``UNKNOWN`` objects a class
    inferred from the label keywords (e.g. ``"NO.2 BILGE PUMP"`` -> pump)."""
    if obj.cls != S.UNKNOWN or not label:
        return obj.cls
    up = label.upper()
    for pat, cls in _LABEL_CLASSES:
        if pat.search(up):
            return cls
    return obj.cls


def effective_category(obj: PIDObject, eff: SymbolClass) -> Category:
    return obj.category if eff == obj.cls else category_of(eff)


def node_name(G: nx.MultiGraph, n: str, with_tag: bool = True) -> str:
    """Display name of a graph node: ``label [tag]`` / tag / class for objects."""
    d = G.nodes[n]
    kind = d.get("kind")
    if kind == "object":
        tag, label = d.get("tag"), d.get("label")
        if label and tag and with_tag and tag not in label:
            return f"{label} [{tag}]"
        if label or tag:
            return label or tag
        return f"{_value(d.get('eff_cls') or d.get('cls'))}:{n}"
    if kind == "free":
        return f"open end of {d.get('pipe_id')}"
    return n


def _value(x: object) -> str:
    return getattr(x, "value", str(x))


# --------------------------------------------------------------------------- port inference


def _spec(cls: SymbolClass) -> Optional[SymbolSpec]:
    try:
        return get_symbol(cls)
    except KeyError:
        return None


def _prims_extent(prims: Iterable) -> Optional[tuple[float, float, float, float]]:
    us: list[float] = []
    vs: list[float] = []
    for p in prims:
        if isinstance(p, Line):
            us += [p.x0, p.x1]
            vs += [p.y0, p.y1]
        elif isinstance(p, Poly):
            us += [q[0] for q in p.pts]
            vs += [q[1] for q in p.pts]
        elif isinstance(p, Circle):
            us += [p.cx - p.r, p.cx + p.r]
            vs += [p.cy - p.r, p.cy + p.r]
        elif isinstance(p, Rect):
            us += [p.x0, p.x1]
            vs += [p.y0, p.y1]
    if not us:
        return None
    return (min(us), min(vs), max(us), max(vs))


def _to_local(obj: PIDObject, x: float, y: float) -> tuple[float, float]:
    """Page offset from the bbox centre -> symbol-local axes (u along flow, v across), page units."""
    c = obj.bbox.center
    dx, dy = x - c.x, y - c.y
    r = obj.rotation % 360
    if r == 90:
        u, v = dy, -dx
    elif r == 180:
        u, v = -dx, -dy
    elif r == 270:
        u, v = -dy, dx
    elif r == 0:
        u, v = dx, dy
    else:
        a = math.radians(r)
        u, v = dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)
    return (u, -v if obj.mirrored else v)


def _port_from_geometry(obj: PIDObject, spec: SymbolSpec, x: float, y: float) -> Optional[str]:
    """Nearest library port of a symbol to a page point, using the object's bbox, rotation and
    mirroring (the bbox is mapped onto the symbol's local extent, actuator included)."""
    prims = list(spec.prims) + (list(ACTUATOR_PRIMS.get(obj.actuator, ())) if spec.category == Category.VALVE else [])
    ext = _prims_extent(prims)
    if ext is None:
        return None
    u0, v0, u1, v1 = ext
    du, dv = _to_local(obj, x, y)
    swap = obj.rotation % 180 == 90
    span_u = obj.bbox.height if swap else obj.bbox.width
    span_v = obj.bbox.width if swap else obj.bbox.height
    lu = (u0 + u1) / 2 + (du / span_u * (u1 - u0) if span_u > 1e-9 else 0.0)
    lv = (v0 + v1) / 2 + (dv / span_v * (v1 - v0) if span_v > 1e-9 else 0.0)
    best = min(spec.ports, key=lambda p: math.hypot(p.x - lu, p.y - lv), default=None)
    return best.name if best else None


def resolve_port(obj: PIDObject, port: Optional[str], point: Optional[Point]) -> Optional[str]:
    """Port name of a pipe end on ``obj``: the given name when it is a library port of the
    class, else the nearest declared/library port to the pipe's end point."""
    spec = _spec(obj.cls)
    if spec is None or spec.boundary_ports or not spec.ports or port == "actuator":
        return port
    names = {p.name for p in spec.ports}
    if port in names or point is None:
        return port
    declared = [p for p in obj.ports if p.name in names]
    if declared:
        best = min(declared, key=lambda p: p.position.dist(point))
        if best.position.dist(point) <= 0.75 * max(obj.bbox.width, obj.bbox.height, 1.0):
            return best.name
    return _port_from_geometry(obj, spec, point.x, point.y) or port


# --------------------------------------------------------------------------- graph


def sheet_link_pairs(doc: PIDDocument) -> list[tuple[str, str, dict]]:
    """Valid entries of ``doc.meta["sheet_links"]``: ``(a, b, link)`` for two distinct objects of the
    document, each connector in at most one link (first entry wins), in stored order."""
    raw = doc.meta.get("sheet_links") if isinstance(doc.meta, dict) else None
    if not raw:
        return []
    ids = {o.id for o in doc.objects}
    used: set[str] = set()
    out = []
    for lk in raw:
        if not isinstance(lk, dict):
            continue
        a, b = lk.get("a"), lk.get("b")
        if not isinstance(a, str) or not isinstance(b, str) or a == b or a not in ids or b not in ids:
            continue
        if a in used or b in used:
            continue
        used.update((a, b))
        out.append((a, b, lk))
    return out


def build_graph(doc: PIDDocument, *, with_sheet_links: bool = True) -> nx.MultiGraph:
    """Objects + junctions as nodes, one edge per pipe run, plus one ``sheet_link`` edge per paired
    off-page connectors unless ``with_sheet_links`` is False (see module docstring)."""
    G = nx.MultiGraph(source=doc.source)
    labels = object_labels(doc)
    objs = doc.object_by_id()
    for i, o in enumerate(doc.objects):
        eff = effective_class(o, labels.get(o.id))
        c = o.center
        G.add_node(o.id, kind="object", cls=o.cls, category=o.category, tag=o.tag, label=labels.get(o.id),
                   eff_cls=eff, eff_category=effective_category(o, eff), actuator=o.actuator, page=o.page,
                   pos=(c.x, c.y), confidence=o.confidence, order=i)
    for i, j in enumerate(doc.junctions):
        if j.id in G:
            continue
        G.add_node(j.id, kind="junction", junction_kind=j.kind, dotted=j.dotted, page=j.page,
                   pos=(j.position.x, j.position.y), order=i)
    junction_ids = {j.id for j in doc.junctions}

    for i, p in enumerate(doc.pipes):
        ends = []
        for which, ep, pt in (("start", p.start, p.points[0] if p.points else None),
                              ("end", p.end, p.points[-1] if p.points else None)):
            ref = ep.object_id or ep.junction_id
            if ep.object_id and ep.object_id in objs:
                node, port = ep.object_id, resolve_port(objs[ep.object_id], ep.port, pt)
            elif not ep.object_id and ep.junction_id and ep.junction_id in junction_ids:
                node, port = ep.junction_id, None
            else:
                node, port = f"free:{p.id}:{which}", None
                G.add_node(node, kind="free", pipe_id=p.id, end=which, ref=ref, page=p.page,
                           pos=(pt.x, pt.y) if pt else None)
            ends.append((node, port, ep.port))
        (u, up, uraw), (v, vp, vraw) = ends
        fd = p.flow_direction if p.flow_direction in (1, -1) else None
        G.add_edge(u, v, key=p.id, pipe_id=p.id, kind=p.kind, medium=p.medium, line_number=p.line_number,
                   size=p.size, length=p.length, flow_direction=fd, start_node=u, end_node=v,
                   start_port=up, end_port=vp, raw_start_port=uraw, raw_end_port=vraw, order=i)
    if with_sheet_links:
        _add_sheet_links(G, doc)
    return G


def _add_sheet_links(G: nx.MultiGraph, doc: PIDDocument) -> None:
    pairs = sheet_link_pairs(doc)
    if not pairs:
        return
    media: dict[str, Medium] = {}
    for p in doc.pipes:
        if p.kind == PipeKind.PROCESS and p.medium != Medium.UNKNOWN:
            for ep in (p.start, p.end):
                if ep.object_id:
                    media.setdefault(ep.object_id, p.medium)
    base = len(doc.pipes)
    for k, (a, b, lk) in enumerate(pairs):
        key = f"sheet_link:{a}:{b}"
        G.add_edge(a, b, key=key, pipe_id=key, kind=SHEET_LINK, medium=media.get(a) or media.get(b) or Medium.UNKNOWN,
                   line_number=None, size=None, length=0.0, flow_direction=1 if lk.get("directed") else None,
                   start_node=a, end_node=b, start_port=None, end_port=None, raw_start_port=None, raw_end_port=None,
                   order=base + k, sheet_link=True, link_key=lk.get("key"), confidence=lk.get("confidence"),
                   link_source=lk.get("source"))
        for n, m in ((a, b), (b, a)):
            G.nodes[n]["sheet_link"] = True
            G.nodes[n].setdefault("linked_to", []).append(m)


def is_linked(G: nx.MultiGraph, n: str) -> bool:
    """True for an off-page connector paired with another one (a continuation, not a terminal)."""
    return bool(G.nodes[n].get("sheet_link")) if n in G else False


@dataclass(frozen=True)
class Link:
    """One pipe as seen from one of its end nodes."""

    pipe_id: str
    node: str
    port: Optional[str]  # port at ``node``
    other: str
    other_port: Optional[str]
    kind: PipeKind
    forward: bool  # flow may go node -> other
    backward: bool  # flow may go other -> node
    order: int = 0


def links(G: nx.MultiGraph, node: str, kinds: frozenset[PipeKind] = PROCESS_KINDS) -> list[Link]:
    """Pipes attached to ``node`` (self loops skipped), in document order."""
    out: list[Link] = []
    if node not in G:
        return out
    for _, other, key, d in G.edges(node, keys=True, data=True):
        if d["kind"] not in kinds or d["start_node"] == d["end_node"]:
            continue
        fd = d.get("flow_direction")
        if d["start_node"] == node:
            out.append(Link(key, node, d["start_port"], other, d["end_port"], d["kind"], fd != -1, fd != 1,
                            d.get("order", 0)))
        else:
            out.append(Link(key, node, d["end_port"], other, d["start_port"], d["kind"], fd != 1, fd != -1,
                            d.get("order", 0)))
    out.sort(key=lambda lk: lk.order)
    return out


def object_ids(G: nx.MultiGraph) -> list[str]:
    return sorted((n for n, d in G.nodes(data=True) if d.get("kind") == "object"),
                  key=lambda n: G.nodes[n].get("order", 0))


def object_adjacency(doc: PIDDocument, G: Optional[nx.MultiGraph] = None) -> set[frozenset[str]]:
    """Pairs of objects connected directly or through junctions by process / instrument-tap
    pipes (junction-collapsed; signal lines and free ends excluded)."""
    G = G if G is not None else build_graph(doc)
    pairs: set[frozenset[str]] = set()
    for o in object_ids(G):
        seen = {o}
        q = deque([o])
        while q:
            n = q.popleft()
            for lk in links(G, n, CONNECT_KINDS):
                m = lk.other
                if m in seen:
                    continue
                seen.add(m)
                kind = G.nodes[m].get("kind")
                if kind == "object":
                    pairs.add(frozenset((o, m)))
                elif kind == "junction":
                    q.append(m)
    return pairs


def components(doc: PIDDocument, G: Optional[nx.MultiGraph] = None, include_signal: bool = False) -> list[list[str]]:
    """Connected groups of object ids (junctions collapsed), largest first; isolated objects
    form single-element groups."""
    G = G if G is not None else build_graph(doc)
    kinds = ALL_KINDS if include_signal else CONNECT_KINDS
    H = nx.Graph()
    H.add_nodes_from(n for n in G.nodes if G.nodes[n].get("kind") != "free")
    for u, v, d in G.edges(data=True):
        if d["kind"] in kinds and G.nodes[u].get("kind") != "free" and G.nodes[v].get("kind") != "free":
            H.add_edge(u, v)
    order = {n: G.nodes[n].get("order", 0) for n in G.nodes}
    groups = []
    for comp in nx.connected_components(H):
        objs = sorted((n for n in comp if G.nodes[n].get("kind") == "object"), key=order.get)
        if objs:
            groups.append(objs)
    groups.sort(key=lambda g: (-len(g), order[g[0]]))
    return groups


# --------------------------------------------------------------------------- flow network

State = tuple[str, Optional[str]]  # (node, pipe the flow arrived by)


@dataclass
class Search:
    """Result of a :meth:`FlowNet.search`: BFS tree over (node, arrival pipe) states."""

    start: State
    parent: dict[State, Optional[tuple[State, str]]] = field(default_factory=dict)
    dist: dict[State, int] = field(default_factory=dict)
    by_node: dict[str, list[State]] = field(default_factory=lambda: defaultdict(list))

    def reached(self, node: str) -> bool:
        return node in self.by_node and node != self.start[0]

    def states_at(self, node: str) -> list[State]:
        return list(self.by_node.get(node, []))

    def best(self, node: str) -> Optional[State]:
        st = self.by_node.get(node)
        return st[0] if st else None

    def path(self, state: State) -> tuple[list[str], list[str]]:
        nodes, pipes = [state[0]], []
        cur = self.parent.get(state)
        while cur is not None:
            prev, pipe = cur
            nodes.append(prev[0])
            pipes.append(pipe)
            cur = self.parent.get(prev)
        return nodes[::-1], pipes[::-1]


class FlowNet:
    """Directional view of the process pipe network (see module docstring for the rules)."""

    def __init__(self, doc: PIDDocument, G: Optional[nx.MultiGraph] = None):
        self.doc = doc
        self.G = G if G is not None else build_graph(doc)
        self.adj: dict[str, list[Link]] = {n: links(self.G, n, PROCESS_KINDS) for n in self.G.nodes}
        self.port_of: dict[tuple[str, str], Optional[str]] = {
            (n, lk.pipe_id): lk.port for n, lks in self.adj.items() for lk in lks}

    def kind(self, n: str) -> str:
        return self.G.nodes[n].get("kind", "")

    def eff(self, n: str) -> Optional[SymbolClass]:
        return self.G.nodes[n].get("eff_cls")

    def passable(self, n: str) -> bool:
        d = self.G.nodes[n]
        if d.get("kind") == "junction" or d.get("sheet_link"):
            return True
        if d.get("kind") != "object":
            return False
        return d["eff_cls"] not in TERMINAL_CLASSES and d.get("eff_category") != Category.INSTRUMENT

    def allowed(self, n: str, pin: Optional[str], pout: Optional[str]) -> bool:
        """May flow entering ``n`` at port ``pin`` leave at port ``pout``?"""
        if self.kind(n) != "object":
            return True
        if pin is not None and pin == pout:
            return True  # tee at a port: the flow does not cross the body
        cls = self.eff(n)
        if cls in DIRECTIONAL_CLASSES:
            return not (pin in OUTLET_PORTS or pout in INLET_PORTS)
        for circuit in CIRCUITS.get(cls, ()):
            if pin in circuit or pout in circuit:
                return pin is None or pout is None or (pin in circuit and pout in circuit)
        return True

    def crosses(self, n: str, pin: Optional[str], pout: Optional[str]) -> bool:
        """True when a path through driver ``n`` really crosses it from inlet to outlet."""
        if pin is not None and pin == pout:
            return False
        return pin not in OUTLET_PORTS and pout not in INLET_PORTS

    def search(self, start: str, in_pipe: Optional[str] = None, *, blocked: Iterable[str] = (),
               stop: Iterable[str] = (), depart: Optional[Callable[[Link], bool]] = None) -> Search:
        """BFS over (node, arrival pipe) states from ``start`` (entered by ``in_pipe``),
        honouring pipe flow direction and directional objects; ``blocked`` nodes are never
        entered, ``stop`` nodes are reached but not passed through, ``depart`` filters the
        first pipe."""
        blocked = set(blocked)
        stop = set(stop)
        s0: State = (start, in_pipe)
        res = Search(start=s0)
        res.parent[s0] = None
        res.dist[s0] = 0
        res.by_node[start].append(s0)
        q = deque([s0])
        while q:
            st = q.popleft()
            n, ip = st
            if st != s0 and (n == start or n in stop or not self.passable(n)):
                continue
            pin = self.port_of.get((n, ip)) if ip else None
            for lk in self.adj.get(n, ()):
                if not lk.forward or lk.pipe_id == ip or lk.other in blocked or self.kind(lk.other) == "free":
                    continue
                if st == s0 and depart is not None and not depart(lk):
                    continue
                if not self.allowed(n, pin, lk.port):
                    continue
                ns: State = (lk.other, lk.pipe_id)
                if ns in res.parent:
                    continue
                res.parent[ns] = (st, lk.pipe_id)
                res.dist[ns] = res.dist[st] + 1
                res.by_node[lk.other].append(ns)
                q.append(ns)
        return res


# --------------------------------------------------------------------------- sources / sinks / paths


def terminal_role(G: nx.MultiGraph, n: str) -> tuple[bool, bool]:
    """(is_source, is_sink) of an object node."""
    d = G.nodes[n]
    if d.get("kind") != "object" or d.get("sheet_link"):
        return (False, False)
    cls = d["eff_cls"]
    label = (d.get("label") or "").upper()
    deg = len(links(G, n, PROCESS_KINDS))
    if deg == 0:
        return (False, False)
    if cls == S.SEA_CHEST:
        return (True, False)
    if cls == S.OVERBOARD:
        return (False, True)
    if cls == S.TANK:
        return (True, True)
    if cls == S.ENGINE:
        return (bool(_SOURCE_ENGINE_RE.search(label)), True)
    if cls == S.COMPRESSOR:
        return (True, False)
    if cls == S.OFF_PAGE_CONNECTOR:
        src, snk = bool(_FROM_RE.search(" " + label)), bool(_TO_RE.search(" " + label))
        return (True, True) if src == snk else (src, snk)
    if (d.get("eff_category") == Category.EQUIPMENT and cls not in DRIVER_CLASSES | INLINE_EQUIPMENT
            and deg == 1):
        return (True, True)
    return (False, False)


def sources_and_sinks(doc: PIDDocument, G: Optional[nx.MultiGraph] = None) -> tuple[list[str], list[str]]:
    """Object ids where flow starts / ends (see module docstring), in document order."""
    G = G if G is not None else build_graph(doc)
    sources, sinks = [], []
    for n in object_ids(G):
        src, snk = terminal_role(G, n)
        if src:
            sources.append(n)
        if snk:
            sinks.append(n)
    return sources, sinks


def _short_name(G: nx.MultiGraph, n: str) -> str:
    return node_name(G, n, with_tag=False)


def flow_paths(doc: PIDDocument, G: Optional[nx.MultiGraph] = None, max_paths: int = 200, *,
               net: Optional[FlowNet] = None) -> list[FlowPath]:
    """Driven flow paths source -> pump/compressor -> sink (shortest per source, driver and
    sink), node-simple, sorted by source, sink and length, at most ``max_paths``."""
    G = G if G is not None else (net.G if net is not None else build_graph(doc))
    net = net if net is not None else FlowNet(doc, G)
    sources, sinks = sources_and_sinks(doc, G)
    order = {n: G.nodes[n].get("order", 0) for n in G.nodes}
    drivers = [n for n in object_ids(G) if G.nodes[n]["eff_cls"] in DRIVER_CLASSES and net.adj.get(n)]
    found: dict[tuple[str, ...], tuple[list[str], list[str]]] = {}

    def add(nodes: list[str], pipes: list[str]) -> None:
        if len(nodes) >= 2 and len(set(nodes)) == len(nodes):
            found.setdefault(tuple(nodes), (nodes, pipes))

    cache: dict[State, Search] = {}
    for s in sources:
        if not drivers or s in drivers:
            res = net.search(s)
            for t in sinks:
                st = res.best(t) if t != s else None
                if st is not None:
                    add(*res.path(st))
            continue
        res = net.search(s)
        for d in drivers:
            for st in res.states_at(d):
                pin = net.port_of.get(st)
                if pin in OUTLET_PORTS:
                    continue
                res2 = cache.get(st)
                if res2 is None:
                    res2 = cache[st] = net.search(d, st[1])
                n1, p1 = res.path(st)
                for t in sinks:
                    if t in (s, d):
                        continue
                    st2 = res2.best(t)
                    if st2 is None:
                        continue
                    n2, p2 = res2.path(st2)
                    if not p2 or not net.crosses(d, pin, net.port_of.get((d, p2[0]))):
                        continue
                    add(n1 + n2[1:], p1 + p2)
    items = sorted(found.values(), key=lambda np_: (order[np_[0][0]], order[np_[0][-1]], len(np_[1]), np_[0]))
    counts: dict[str, int] = defaultdict(int)
    for n in object_ids(G):
        counts[_short_name(G, n)] += 1

    def label(n: str) -> str:
        nm = _short_name(G, n)
        return f"{nm} ({n})" if counts[nm] > 1 and not nm.endswith(f":{n}") else nm

    out: list[FlowPath] = []
    for nodes, pipes in items[: max(0, max_paths)]:
        via = [n for n in nodes[1:-1] if G.nodes[n].get("kind") == "object" and G.nodes[n]["eff_cls"] in DRIVER_CLASSES]
        names = [label(nodes[0])] + [label(n) for n in via] + [label(nodes[-1])]
        out.append(FlowPath(source_id=nodes[0], target_id=nodes[-1], node_ids=list(nodes), pipe_ids=list(pipes),
                            description=" -> ".join(names)))
    return out


# --------------------------------------------------------------------------- isolation


@dataclass
class IsolationBranch:
    pipe_id: str
    port: Optional[str]
    valves: list[str] = field(default_factory=list)
    open_to: list[str] = field(default_factory=list)  # nodes reached without passing an isolating valve

    @property
    def isolated(self) -> bool:
        return bool(self.valves) and not self.open_to


def _isolating(G: nx.MultiGraph, n: str) -> bool:
    d = G.nodes[n]
    if d.get("eff_category") != Category.VALVE:
        return False
    return d["eff_cls"] in ISOLATING_VALVES or d["eff_cls"] == S.UNKNOWN


def _isolation_passable(G: nx.MultiGraph, n: str) -> bool:
    d = G.nodes[n]
    if d.get("kind") == "junction" or d.get("sheet_link"):
        return True
    if d.get("kind") != "object":
        return False
    cls, cat = d["eff_cls"], d.get("eff_category")
    return cat == Category.FITTING or cls in INLINE_EQUIPMENT or cls == S.CHECK_VALVE or cls == S.UNKNOWN


def isolation_analysis(doc: PIDDocument, G: Optional[nx.MultiGraph], object_id: str) -> list[IsolationBranch]:
    """Per process connection of ``object_id``: the nearest isolating valves and the nodes that
    can be reached without passing one (open branches). Check valves, fittings, strainers,
    filters and flow meters are passed through; relief valves are treated as closed."""
    G = G if G is not None else build_graph(doc)
    out: list[IsolationBranch] = []
    for lk in links(G, object_id, PROCESS_KINDS):
        br = IsolationBranch(lk.pipe_id, lk.port)
        seen = {object_id}
        q = deque([lk.other])
        seen.add(lk.other)
        while q:
            n = q.popleft()
            d = G.nodes[n]
            if d.get("kind") == "free":
                br.open_to.append(n)
                continue
            if d.get("kind") == "object":
                if _isolating(G, n):
                    br.valves.append(n)
                    continue
                if d["eff_cls"] == S.SAFETY_VALVE or d.get("eff_category") == Category.INSTRUMENT:
                    continue
                if not _isolation_passable(G, n):
                    br.open_to.append(n)
                    continue
            for nl in links(G, n, PROCESS_KINDS):
                if nl.other not in seen:
                    seen.add(nl.other)
                    q.append(nl.other)
        out.append(br)
    return out


def isolation_valves(doc: PIDDocument, G: Optional[nx.MultiGraph], object_id: str) -> list[str]:
    """Nearest isolating valves on every process branch of ``object_id`` (document order)."""
    G = G if G is not None else build_graph(doc)
    vs = {v for br in isolation_analysis(doc, G, object_id) for v in br.valves}
    return sorted(vs, key=lambda n: G.nodes[n].get("order", 0))
