"""Deterministic marine engineering checks and drawing-quality checks.

Every rule has a stable ``rule_id``, a severity (used for FAIL verdicts), a reference and
returns :class:`~shippid.model.Finding` objects with a verdict:

* ``PASS`` / ``FAIL`` -- the requirement could be evaluated on this drawing;
* ``NOT_APPLICABLE`` -- the drawing's system (``doc.info.system(s)``, else inferred from the
  title, line numbers or title block) or its content makes the rule irrelevant;
* ``INSUFFICIENT_DATA`` -- the rule applies but the drawing (or the extraction) does not show
  enough to decide (e.g. an unconnected pump, an unnamed tank).

Rules look at the *effective* class of objects (the extracted class, or a class inferred from
the equipment name for ``UNKNOWN`` objects), at port names (inferred from the symbol geometry
when missing) and at equipment names, so they degrade gracefully on imperfect extraction.
Requirements whose fulfilment may be shown on another drawing (second bilge / fire pump,
emergency fire pump, built-in relief valves) say so in the finding description.

References are given at regulation level (SOLAS / MARPOL / Load Lines / IACS); class-rule
references are generic because the exact paragraph depends on the classification society.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import networkx as nx

from ..catalog import SYSTEM_BY_CODE, SYSTEMS, detect_system, parse_line_number
from ..model import (
    ActuatorType,
    Category,
    Finding,
    Medium,
    PIDDocument,
    PipeKind,
    Severity,
    SymbolClass,
    SystemType,
    TextRole,
    Verdict,
)
from .graph import (
    ALL_KINDS,
    DRIVER_CLASSES,
    INLET_PORTS,
    INLINE_EQUIPMENT,
    NRV_CLASSES,
    OUTLET_PORTS,
    PROCESS_KINDS,
    PUMP_CLASSES,
    STRAINER_CLASSES,
    FlowNet,
    Link,
    Search,
    build_graph,
    isolation_analysis,
    links,
    node_name,
    object_ids,
)

log = logging.getLogger(__name__)

S = SymbolClass
V = Verdict
ST = SystemType

# --------------------------------------------------------------------------- drawing system


def drawing_systems(doc: PIDDocument) -> tuple[list[SystemType], str]:
    """Systems shown on the drawing and the basis of that answer:
    ``drawing_info`` > ``title`` > ``line_numbers`` (majority system code) > ``title_block``."""
    systems = [s for s in [doc.info.system, *doc.info.systems] if s != ST.UNKNOWN]
    if systems:
        return list(dict.fromkeys(systems)), "drawing_info"
    if doc.info.title:
        s = detect_system(doc.info.title)
        if s:
            return [s], "title"
    codes = Counter()
    for p in doc.pipes:
        parsed = parse_line_number(p.line_number) if p.line_number else None
        if parsed and parsed["system"] in SYSTEM_BY_CODE:
            codes[parsed["system"]] += 1
    if codes:
        (code, n), total = codes.most_common(1)[0], sum(codes.values())
        if n >= 0.4 * total:
            return [SYSTEM_BY_CODE[code]], "line_numbers"
    tb = " ".join(t.text for t in doc.texts if t.role == TextRole.TITLE_BLOCK)
    s = detect_system(tb) if tb else None
    if s:
        return [s], "title_block"
    return [], "unknown"


# --------------------------------------------------------------------------- context


class RuleContext:
    """Graph, flow network and lookups shared by all rules (computed once per document)."""

    def __init__(self, doc: PIDDocument, G: Optional[nx.MultiGraph] = None, net: Optional[FlowNet] = None):
        self.doc = doc
        self.G = G if G is not None else (net.G if net is not None else build_graph(doc))
        self.net = net if net is not None else FlowNet(doc, self.G)
        self.systems, self.system_basis = drawing_systems(doc)
        self.objs = doc.object_by_id()
        self._links: dict[tuple[str, frozenset], list[Link]] = {}
        self._pump_fwd: dict[str, Search] = {}
        self._pump_flow: dict[str, Search] = {}
        self._passed: dict[str, set[str]] = {}

    # -- lookups -----------------------------------------------------------------
    def node(self, n: str) -> dict:
        return self.G.nodes[n]

    def is_obj(self, n: str) -> bool:
        return self.G.nodes[n].get("kind") == "object"

    def is_junction(self, n: str) -> bool:
        return self.G.nodes[n].get("kind") == "junction"

    def eff(self, n: str) -> Optional[SymbolClass]:
        return self.G.nodes[n].get("eff_cls")

    def cat(self, n: str) -> Optional[Category]:
        return self.G.nodes[n].get("eff_category")

    def label(self, n: str) -> str:
        return (self.G.nodes[n].get("label") or "").upper()

    def tag(self, n: str) -> Optional[str]:
        return self.G.nodes[n].get("tag")

    def name(self, n: str) -> str:
        return node_name(self.G, n)

    def names(self, ns: Iterable[str], limit: int = 8) -> str:
        ns = list(ns)
        out = ", ".join(self.name(n) for n in ns[:limit])
        return out + (f" (+{len(ns) - limit} more)" if len(ns) > limit else "")

    def ids(self, *classes: SymbolClass) -> list[str]:
        return [n for n in object_ids(self.G) if self.eff(n) in classes]

    def links(self, n: str, kinds: frozenset = PROCESS_KINDS) -> list[Link]:
        key = (n, kinds)
        if key not in self._links:
            self._links[key] = links(self.G, n, kinds)
        return self._links[key]

    def conf(self, ns: Iterable[str]) -> float:
        vals = [self.G.nodes[n].get("confidence", 1.0) for n in ns if n in self.G and self.is_obj(n)]
        return round(min(vals), 3) if vals else 1.0

    def has_system(self, *systems: SystemType) -> bool:
        return any(s in self.systems for s in systems)

    def is_primary(self, *systems: SystemType) -> bool:
        """The drawing's main system (``info.system``, else the first listed / inferred one)."""
        return bool(self.systems) and self.systems[0] in systems

    @property
    def systems_known(self) -> bool:
        return bool(self.systems)

    # -- searches ----------------------------------------------------------------
    def reach(self, start: str, *, passable: Callable[[str], bool], via: Optional[Callable[[Link], bool]] = None,
              blocked: Iterable[str] = (), kinds: frozenset = PROCESS_KINDS,
              limit: int = 20000) -> tuple[list[str], dict[str, Optional[str]]]:
        """Undirected BFS from ``start``: visited nodes (BFS order, start excluded) and the port by
        which each was first reached. Non-passable nodes are visited but not expanded."""
        blocked = set(blocked)
        seen = {start}
        order: list[str] = []
        arrival: dict[str, Optional[str]] = {}
        q: deque[str] = deque()

        def visit(lk: Link) -> None:
            m = lk.other
            if m in seen or m in blocked:
                return
            seen.add(m)
            arrival[m] = lk.other_port
            order.append(m)
            q.append(m)

        for lk in self.links(start, kinds):
            if via is None or via(lk):
                visit(lk)
        while q and len(order) < limit:
            n = q.popleft()
            if self.G.nodes[n].get("kind") == "free" or not passable(n):
                continue
            for lk in self.links(n, kinds):
                visit(lk)
        return order, arrival

    def pump_forward(self, p: str) -> Search:
        """Forward search along a driver's discharge branches: other drivers are never entered
        and relief valves are reached but not passed through (a relief valve back to the suction
        must not make the suction header look like part of the discharge)."""
        if p not in self._pump_fwd:
            drivers = set(self.ids(*DRIVER_CLASSES)) - {p}
            self._pump_fwd[p] = self.net.search(p, blocked=drivers, stop=self.ids(S.SAFETY_VALVE),
                                                depart=lambda lk: lk.port not in INLET_PORTS)
        return self._pump_fwd[p]

    def pump_flow(self, p: str) -> Search:
        """Everywhere a driver's discharge can flow (relief valves passed when lifting), not
        entering other drivers."""
        if p not in self._pump_flow:
            drivers = set(self.ids(*DRIVER_CLASSES)) - {p}
            self._pump_flow[p] = self.net.search(p, blocked=drivers, depart=lambda lk: lk.port not in INLET_PORTS)
        return self._pump_flow[p]

    def passed(self, p: str) -> set[str]:
        """Nodes the discharge-branch search from driver ``p`` passes through (not merely reaches)."""
        if p not in self._passed:
            res = self.pump_forward(p)
            self._passed[p] = {parent[0][0] for parent in res.parent.values() if parent} - {p}
        return self._passed[p]


# --------------------------------------------------------------------------- registry


_DOUBLE_STOP_RE = re.compile(r"(?<!\.)\.\.(?!\.)(?=\s|$)")


@dataclass(frozen=True)
class Rule:
    rule_id: str
    title: str
    title_ko: str
    severity: Severity
    reference: str
    func: Callable[["RuleContext", "Rule"], list[Finding]]
    group: str = "marine"  # "marine" | "quality"

    def make(self, verdict: Verdict, description: str, object_ids: Iterable[str] = (), confidence: float = 1.0,
             title: Optional[str] = None, severity: Optional[Severity] = None) -> Finding:
        if severity is None:
            if verdict == V.FAIL:
                severity = self.severity
            elif verdict == V.INSUFFICIENT_DATA:
                severity = Severity.WARNING if self.severity == Severity.ERROR else Severity.INFO
            else:
                severity = Severity.INFO
        description = _DOUBLE_STOP_RE.sub(".", description)  # names ending in "." ("NO.2 AIR COMP.") + full stop
        return Finding(id="", rule_id=self.rule_id, verdict=verdict, severity=severity, title=title or self.title,
                       description=description, object_ids=list(dict.fromkeys(object_ids)),
                       reference=self.reference, source="rule", confidence=max(0.0, min(1.0, confidence)))

    def na(self, description: str) -> Finding:
        return self.make(V.NOT_APPLICABLE, description)


RULES: dict[str, Rule] = {}


def rule(rule_id: str, title: str, title_ko: str, severity: Severity, reference: str, group: str = "marine"):
    """Register a rule function ``f(ctx, rule) -> list[Finding]``."""

    def deco(fn: Callable[[RuleContext, Rule], list[Finding]]):
        RULES[rule_id] = Rule(rule_id, title, title_ko, severity, reference, fn, group)
        return fn

    return deco


def run_rules(doc: PIDDocument, G: Optional[nx.MultiGraph] = None, *, context: Optional[RuleContext] = None,
              rule_ids: Optional[Iterable[str]] = None) -> list[Finding]:
    """Run the registered rules (all, or ``rule_ids``) and return their findings with stable
    ids ``<rule_id>:<n>``. A rule that raises yields one INSUFFICIENT_DATA finding."""
    ctx = context or RuleContext(doc, G)
    selected = list(rule_ids) if rule_ids is not None else list(RULES)
    out: list[Finding] = []
    for rid in selected:
        r = RULES[rid]
        try:
            found = r.func(ctx, r)
        except Exception as exc:  # a rule bug must not hide the other findings
            log.exception("rule %s failed", rid)
            found = [r.make(V.INSUFFICIENT_DATA, f"Rule evaluation failed: {type(exc).__name__}: {exc}",
                            confidence=0.0)]
        for k, f in enumerate(found, 1):
            out.append(f.model_copy(update={"id": f"{rid}:{k}"}))
    return out


# --------------------------------------------------------------------------- helpers

_FIRE_RE = re.compile(r"FIRE|소화")
_NOT_FIRE_PUMP_RE = re.compile(r"JOCKEY|PRIMING|STRIPPING|TRANSFER|FEED|CIRC|BOOSTER|SUPPLY")
_BILGE_WORD_RE = re.compile(r"BILGE|빌지")
# positive evidence of a bilge suction point: wells, cofferdams, compartments / spaces
_WELL_RE = re.compile(
    r"WELL\b|BILGE\s*SUCTION|COFFERDAM|DUCT\s*KEEL|PIPE\s*DUCT|CHAIN\s*LOCKER|THRUSTER\s*R(?:OO)?M|STEERING\s*GEAR|"
    r"PUMP\s*R(?:OO)?M|\bE/R\b|ENGINE\s*R(?:OO)?M|\bHOLD\b|TUNNEL|\bVOID\b|\bSPACE\b|\bROOM\b|COMP'?T\b|COMPARTMENT|"
    r"웰|코퍼댐|구역")
# oily-water process / storage tanks that merely carry "BILGE" in their name
_NOT_WELL_RE = re.compile(
    r"HOLDING|SLUDGE|SEPARATED|WASTE|\bOIL|OILY|SEWAGE|OVERFLOW|DRAIN\s*T|HOT\s*WELL|CASCADE|PRIMARY|SETTLING|SETT\b|"
    r"WATER\s*(?:TK|TANK)\b|SEPARATOR|STORAGE|\bSTOR\b|핫\s*웰|침전|저장")
_TANK_WORD_RE = re.compile(r"\bTK\b|TANK|탱크")
_BILGE_PUMP_EXCL_RE = re.compile(r"SEPARATOR|\bO\.?W\.?S\b|FEED|SLUDGE")
_DOUBLE_BOTTOM_RE = re.compile(r"\(D\.?\s?B\.?\)|\bD\.\s?B\b\.?|\bDB\b|DOUBLE\s*BOTTOM|이중저")
_REMOTE_ACTUATORS = frozenset({ActuatorType.HYDRAULIC, ActuatorType.ELECTRIC, ActuatorType.DIAPHRAGM})
_DISTILLATE_RE = re.compile(r"\bM\.?\s?D\.?\s?O\b|\bM\.?\s?G\.?\s?O\b|\bD\.?\s?O\b|DIESEL|GAS\s*OIL|\bDM[ABZ]\b|경유")
# auxiliary air plant that does not count towards the main-engine starting-air requirement
_NOT_START_AIR_RE = re.compile(
    r"EMERG|EM'CY|EMCY|SERVICE|\bSERV\b|WORKING|GENERAL|\bG/S\b|CONTROL|\bCONT\b|INSTRUMENT|\bINSTR?\b|TOPPING|"
    r"\bG/E\b|\bD/G\b|GENERATOR|AUX|DECK|비상|잡용|제어|계장|발전기")
_FUEL_LUBE_RE = re.compile(
    r"\bF\.?\s?O\b|FUEL|\bH\.?\s?F\.?\s?O\b|\bM\.?\s?D\.?\s?O\b|\bM\.?\s?G\.?\s?O\b|\bD\.?\s?O\b|\bL\.?\s?O\b|LUB|"
    r"DIESEL|GAS\s*OIL|VLSFO|ULSFO|LSHFO|L\.S\.H\.F\.O|연료|윤활")
_QC_FUNC_RE = re.compile(r"STOR|SETT|SERV|DAILY|\bDAY\b|저장|침전|서비스")
_QC_EXCL_RE = re.compile(r"SUMP|DRAIN|SLUDGE|OVERFLOW|BILGE|LEAK|WASTE|DIRTY|EXPANSION|CYL|SAMPL|SPILL|RETURN")
_SERVICE_RE = re.compile(r"SERV|DAILY|\bDAY\b|서비스")
_CAPACITY_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(M3|M³|㎥|CUM|LTR|LITRES?|LITERS?|L)(?![\w.])")
_SEA_INLET_RE = re.compile(r"SEA\s*CHEST|SEA\s*INLET|씨체스트")
_SW_PUMP_RE = re.compile(r"S\.?\s?W\.?|SEA\s*W|COOLING")
_BILGE_DEST_RE = re.compile(r"BILGE|HOLDING|\bWELL\b|SLUDGE|OWS|SEPARATOR")


def _value(x: object) -> str:
    return getattr(x, "value", str(x))


def _p_inline(ctx: RuleContext, n: str) -> bool:
    """Junctions, fittings, strainers / filters / flow meters and unknown symbols."""
    if ctx.is_junction(n):
        return True
    if not ctx.is_obj(n):
        return False
    return ctx.cat(n) == Category.FITTING or ctx.eff(n) in INLINE_EQUIPMENT or ctx.eff(n) == S.UNKNOWN


def _is_valve(ctx: RuleContext, n: str) -> bool:
    return ctx.is_obj(n) and ctx.cat(n) == Category.VALVE


def _p_through_valves(ctx: RuleContext, n: str) -> bool:
    """Inline items plus every valve except relief valves."""
    return _p_inline(ctx, n) or (_is_valve(ctx, n) and ctx.eff(n) != S.SAFETY_VALVE)


def _is_closing_valve(ctx: RuleContext, n: str) -> bool:
    return _is_valve(ctx, n) and ctx.eff(n) not in (S.CHECK_VALVE, S.SAFETY_VALVE)


def _is_nrv_like(ctx: RuleContext, n: str) -> bool:
    return ctx.is_obj(n) and ctx.eff(n) in (S.CHECK_VALVE, S.SDNR_VALVE, S.SAFETY_VALVE)


def _p_any_valve(ctx: RuleContext, n: str) -> bool:
    """Inline items plus every valve (relief valves included: they pass flow when lifting)."""
    return _p_inline(ctx, n) or _is_valve(ctx, n)


def _leaves(ctx: RuleContext, start: str, blocked: Iterable[str] = (),
            passable: Optional[Callable[[RuleContext, str], bool]] = None) -> list[str]:
    """Equipment / terminals / pumps reachable from ``start`` through valves and inline items."""
    passable = passable or _p_through_valves
    order, _ = ctx.reach(start, passable=lambda n: passable(ctx, n), blocked=blocked)
    return [n for n in order if ctx.is_obj(n) and not passable(ctx, n) and not _is_valve(ctx, n)
            and ctx.cat(n) != Category.INSTRUMENT]


def _fire_pumps(ctx: RuleContext) -> list[str]:
    fire_sys = ctx.is_primary(ST.FIRE_MAIN)  # other pumps count only on a drawing *of* the fire main
    out = []
    for p in ctx.ids(*PUMP_CLASSES):
        lab = ctx.label(p)
        if _FIRE_RE.search(lab):
            out.append(p)
        elif fire_sys and not _NOT_FIRE_PUMP_RE.search(lab):
            out.append(p)
    return out


def _well_kind(label: str, bilge_dwg: bool) -> Optional[str]:
    """``"well"`` (a bilge suction point), ``"uncertain"`` or ``None`` (not a bilge suction) for a
    tank label. Oily-water process tanks (holding / primary / settling ...) are never wells; a
    name with BILGE but no tank word ("E/R BILGE (P)") is a compartment bilge; WELL / COFFERDAM /
    compartment names count on bilge drawings; other BILGE ... TK names are uncertain."""
    if _NOT_WELL_RE.search(label):
        return None
    bilge = bool(_BILGE_WORD_RE.search(label))
    if _WELL_RE.search(label):
        return "well" if (bilge or bilge_dwg) else None
    if bilge:
        return "uncertain" if _TANK_WORD_RE.search(label) else "well"
    return None


def _bilge_wells(ctx: RuleContext) -> tuple[list[str], set[str]]:
    """Bilge suction points (tanks named BILGE WELL / COFFERDAM / compartment bilges); unnamed
    tanks on a bilge drawing and BILGE-named tanks of unclear purpose are included as uncertain."""
    wells, uncertain = [], set()
    bilge_dwg = ctx.has_system(ST.BILGE) or not ctx.systems_known
    for t in ctx.ids(S.TANK):
        lab = ctx.label(t)
        if lab:
            kind = _well_kind(lab, bilge_dwg)
            if kind is None:
                continue
            wells.append(t)
            if kind == "uncertain":
                uncertain.add(t)
        elif ctx.has_system(ST.BILGE):
            wells.append(t)
            uncertain.add(t)
    return wells, uncertain


def _none_shown(ctx: RuleContext, r: Rule, system: SystemType, what: str) -> Finding:
    """The rule's system applies but none of the equipment it counts is shown: insufficient data
    on a drawing *of* that system, not applicable when the system is only listed as secondary."""
    if ctx.is_primary(system):
        return r.make(V.INSUFFICIENT_DATA, f"No {what} recognised on this {SYSTEMS[system].name_en.lower()} drawing.")
    return r.na(f"{SYSTEMS[system].name_en} is only a secondary system of this drawing and no {what} is shown.")


def _driver_crossable_states(ctx: RuleContext, res: Search, d: str) -> list:
    return [st for st in res.states_at(d) if st[1] is not None and ctx.net.port_of.get(st) not in OUTLET_PORTS]


@dataclass
class _NrvCheck:
    ok: list[str] = field(default_factory=list)  # NRVs on the pump's own discharge branch
    shared: list[str] = field(default_factory=list)  # NRVs reached only through a common header
    reversed: list[str] = field(default_factory=list)
    connected: bool = True
    side_known: bool = True


def _discharge_nrv(ctx: RuleContext, p: str) -> _NrvCheck:
    res = _NrvCheck()
    lks = ctx.links(p)
    if not lks:
        res.connected = False
        return res
    res.side_known = any(lk.port in INLET_PORTS | OUTLET_PORTS for lk in lks)
    fwd = ctx.pump_forward(p)
    common: set[str] = set()
    for q in ctx.ids(*PUMP_CLASSES):
        if q != p:
            common |= ctx.passed(q)
    for n in fwd.by_node:
        if n == p or not ctx.is_obj(n) or ctx.eff(n) not in NRV_CLASSES:
            continue
        good = [st for st in fwd.states_at(n) if ctx.net.port_of.get(st) not in OUTLET_PORTS]
        if not good:
            continue
        nodes, _ = fwd.path(good[0])
        if any(ctx.is_obj(m) and ctx.eff(m) in NRV_CLASSES for m in nodes[1:-1]):
            continue  # report the nearest non-return valve(s) only
        (res.shared if any(m in common for m in nodes[1:-1]) else res.ok).append(n)
    if not res.ok:
        # an NRV drawn backwards right on the pump's own line
        for lk in lks:
            if lk.port in INLET_PORTS:
                continue
            prev, cur, port = p, lk.other, lk.other_port
            for _ in range(12):
                if not ctx.is_obj(cur) and not ctx.is_junction(cur):
                    break
                if ctx.is_obj(cur) and ctx.eff(cur) in NRV_CLASSES:
                    if port in OUTLET_PORTS:
                        res.reversed.append(cur)
                    break
                nxt = [x for x in ctx.links(cur) if x.other != prev]
                if len(nxt) != 1 or (ctx.is_obj(cur) and not _p_through_valves(ctx, cur)):
                    break
                prev, cur, port = cur, nxt[0].other, nxt[0].other_port
    return res


def _parallel_with(ctx: RuleContext, p: str) -> list[str]:
    mine = ctx.passed(p)
    return [q for q in ctx.ids(*PUMP_CLASSES) if q != p and mine & ctx.passed(q)]


def _size_key(size: Optional[str]) -> Optional[tuple[int, str]]:
    if not size:
        return None
    s = size.strip().upper()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    unit = "in" if (s.endswith("B") or '"' in s or "INCH" in s) else "mm"
    return (int(round(float(m.group(1)))), unit)


def _pipe_size(ctx: RuleContext, d: dict) -> Optional[tuple[int, str]]:
    key = _size_key(d.get("size"))
    if key is None and d.get("line_number"):
        parsed = parse_line_number(d["line_number"])
        key = _size_key(parsed["size"]) if parsed else None
    return key


def _capacity_litres(label: str) -> Optional[float]:
    m = _CAPACITY_RE.search(label)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    return val * 1000 if m.group(2) in ("M3", "M³", "㎥", "CUM") else val


# --------------------------------------------------------------------------- marine rules


@rule("R_PUMP_DISCHARGE_NRV", "Non-return valve on pump discharge", "펌프 토출측 체크(역지) 밸브", Severity.ERROR,
      "Class rules (pumps operating in parallel); good marine engineering practice")
def _r_pump_nrv(ctx: RuleContext, r: Rule) -> list[Finding]:
    fire = set(_fire_pumps(ctx))
    pumps = [p for p in ctx.ids(*PUMP_CLASSES) if p not in fire]
    if not pumps:
        return [r.na("No pumps other than fire pumps on this drawing (fire pumps: R_FIRE_PUMP_DISCHARGE_NRV).")]
    out = []
    for p in pumps:
        chk = _discharge_nrv(ctx, p)
        conf = ctx.conf([p, *chk.ok, *chk.reversed]) * (1.0 if chk.side_known else 0.6)
        name = ctx.name(p)
        if not chk.connected:
            out.append(r.make(V.INSUFFICIENT_DATA, f"{name} has no process connections; discharge cannot be traced.",
                              [p], conf))
        elif chk.ok:
            out.append(r.make(V.PASS, f"{name}: non-return valve {ctx.names(chk.ok)} on the discharge.",
                              [p, *chk.ok], conf))
        elif chk.reversed:
            out.append(r.make(V.FAIL, f"{name}: non-return valve {ctx.names(chk.reversed)} on the discharge line is "
                                      f"drawn reversed (closing towards the header).", [p, *chk.reversed], conf))
        else:
            par = _parallel_with(ctx, p)
            if par:
                shared = (f" (only the common non-return valve {ctx.names(chk.shared)} downstream of the header)"
                          if chk.shared else "")
                out.append(r.make(V.FAIL, f"{name} discharges into a common line with {ctx.names(par)} but has no "
                                          f"non-return valve on its own discharge branch{shared}; an idle pump would "
                                          f"be back-fed.", [p, *par, *chk.shared], conf))
            else:
                out.append(r.make(V.NOT_APPLICABLE, f"{name} does not run in parallel with another pump on this "
                                                    f"drawing; a discharge non-return valve is not required by this "
                                                    f"rule.", [p], conf))
    return out


@rule("R_PD_PUMP_RELIEF", "Relief valve for positive-displacement (screw) pump", "용적형(스크류) 펌프 안전밸브",
      Severity.WARNING, "Class rules: positive-displacement pumps to be protected by a relief valve")
def _r_pd_relief(ctx: RuleContext, r: Rule) -> list[Finding]:
    pumps = ctx.ids(S.PUMP_SCREW)
    if not pumps:
        return [r.na("No positive-displacement (screw / gear) pumps on this drawing.")]
    out = []
    for p in pumps:
        name = ctx.name(p)
        obj_attrs = ctx.objs[p].attributes
        if obj_attrs.get("integral_relief") or obj_attrs.get("relief_valve"):
            out.append(r.make(V.PASS, f"{name}: integral relief valve recorded in the object attributes.", [p]))
            continue
        if not ctx.links(p):
            out.append(r.make(V.INSUFFICIENT_DATA, f"{name} has no process connections.", [p], ctx.conf([p])))
            continue
        fwd = ctx.pump_forward(p)
        common: set[str] = set()
        for q in ctx.ids(*PUMP_CLASSES):
            if q != p:
                common |= ctx.passed(q)
        own, shared, reversed_ = [], [], []
        for n in fwd.by_node:
            if n == p or not ctx.is_obj(n) or ctx.eff(n) != S.SAFETY_VALVE:
                continue
            good = [st for st in fwd.states_at(n) if ctx.net.port_of.get(st) not in OUTLET_PORTS]
            if not good:  # reached only at its outlet: drawn reversed, it cannot relieve the discharge
                reversed_.append(n)
                continue
            nodes, _ = fwd.path(good[0])
            (shared if any(m in common for m in nodes[1:-1]) else own).append(n)
        if own:
            out.append(r.make(V.PASS, f"{name}: relief valve {ctx.names(own)} on the discharge side.", [p, *own],
                              ctx.conf([p, *own])))
        elif shared:
            out.append(r.make(V.PASS, f"{name}: only a shared relief valve {ctx.names(shared)} downstream of the "
                                      f"common header protects this pump; verify it cannot be isolated from the pump.",
                              [p, *shared], 0.7 * ctx.conf([p, *shared])))
        elif reversed_:
            out.append(r.make(V.FAIL, f"{name}: relief valve {ctx.names(reversed_)} is drawn reversed (its outlet faces "
                                      f"the pump discharge), so it cannot relieve the discharge side.",
                              [p, *reversed_], ctx.conf([p, *reversed_])))
        else:
            out.append(r.make(V.FAIL, f"{name}: no relief valve found on the discharge side. Required unless the pump "
                                      f"has a built-in relief valve (state it on the drawing).", [p], ctx.conf([p])))
    return out


@rule("R_QC_VALVE_FO_LO_TANK", "Quick-closing valve on FO/LO tank outlets", "연료유/윤활유 탱크 출구 급속차단밸브",
      Severity.ERROR, "SOLAS II-2/4.2.2.3.4 (fuel oil), II-2/4.2.3 (lubricating oil)")
def _r_qc_valve(ctx: RuleContext, r: Rule) -> list[Finding]:
    fo_lo_sys = ctx.has_system(ST.FUEL_OIL_TRANSFER, ST.FUEL_OIL_SUPPLY, ST.LUBE_OIL)
    tanks: list[tuple[str, float, str]] = []  # (id, confidence factor, kind)
    for t in ctx.ids(S.TANK):
        lab = ctx.label(t)
        media = {ctx.G.edges[t, lk.other, lk.pipe_id]["medium"] for lk in ctx.links(t)}
        oily = bool(media & {Medium.FUEL_OIL, Medium.LUBE_OIL})
        if lab:
            if _QC_EXCL_RE.search(lab):
                continue
            cap = _capacity_litres(lab)
            if cap is not None and cap < 500:
                continue
            if _FUEL_LUBE_RE.search(lab) or ((fo_lo_sys or oily) and _QC_FUNC_RE.search(lab)):
                kind = "double_bottom" if _DOUBLE_BOTTOM_RE.search(lab) else "named"
                tanks.append((t, 1.0 if _QC_FUNC_RE.search(lab) else 0.7, kind))
        elif fo_lo_sys or oily:
            tanks.append((t, 0.5, "unnamed"))
    if not tanks:
        why = "this is not a fuel / lubricating oil drawing" if not fo_lo_sys else "no storage / settling / service tank"
        return [r.na(f"No FO/LO storage, settling or daily service tank on this drawing ({why}).")]
    drivers = ctx.ids(*DRIVER_CLASSES)
    out = []
    for t, cf, kind in tanks:
        name = ctx.name(t)
        if kind == "double_bottom":
            out.append(r.make(V.NOT_APPLICABLE, f"{name} is a double-bottom tank; the quick-closing requirement "
                                                f"applies to FO/LO tanks situated above the double bottom.", [t],
                              cf * ctx.conf([t])))
            continue
        lks = ctx.links(t)
        if not lks:
            out.append(r.make(V.INSUFFICIENT_DATA, f"{name} has no pipe connections.", [t], cf * ctx.conf([t])))
            continue
        bad, unknown, good, remote, inlet_only = [], [], [], [], []
        for lk in lks:
            res = ctx.net.search(t, depart=lambda x, pid=lk.pipe_id: x.pipe_id == pid)
            outlet = any(_driver_crossable_states(ctx, res, d) for d in drivers)
            inlet = any((t, lk.pipe_id) in ctx.pump_flow(d).parent for d in drivers)
            order, _ = ctx.reach(t, via=lambda x, pid=lk.pipe_id: x.pipe_id == pid,
                                 passable=lambda n: ctx.is_junction(n) or (ctx.is_obj(n) and ctx.cat(n) == Category.FITTING))
            first_valves = [n for n in order if _is_valve(ctx, n)]
            open_to = [n for n in order if ctx.is_obj(n) and not _is_valve(ctx, n) and ctx.cat(n) != Category.FITTING
                       and ctx.cat(n) != Category.INSTRUMENT]
            qc = [v for v in first_valves if ctx.node(v).get("actuator") == ActuatorType.QUICK_CLOSING]
            rem = [v for v in first_valves if ctx.node(v).get("actuator") in _REMOTE_ACTUATORS]
            manual = [v for v in first_valves if v not in qc and v not in rem]
            if outlet:
                if manual or open_to or not first_valves:
                    bad.append((lk.pipe_id, manual, open_to))
                else:
                    good.extend(qc)
                    remote.extend(rem)
            elif inlet:
                inlet_only.append(lk.pipe_id)
            elif first_valves and not manual and not open_to:
                good.extend(qc)
                remote.extend(rem)
            else:
                unknown.append((lk.pipe_id, manual, open_to))
        involved = [t] + good + remote + [v for _, vs, _ in bad + unknown for v in vs]
        conf = cf * ctx.conf(involved)
        if bad:
            parts = []
            for pid, vs, open_to in bad:
                if vs:
                    acts = ", ".join(sorted({_value(ctx.node(v).get("actuator") or ActuatorType.NONE) for v in vs}))
                    parts.append(f"outlet {pid}: first valve {ctx.names(vs)} is not quick-closing or remotely "
                                 f"operated (actuator: {acts})")
                elif open_to:
                    parts.append(f"outlet {pid}: no valve before {ctx.names(open_to)}")
                else:
                    parts.append(f"outlet {pid}: no valve directly on the tank")
            verdict = V.INSUFFICIENT_DATA if kind == "unnamed" else V.FAIL
            note = " (tank is unnamed; confirm it is a storage/settling/service tank)" if kind == "unnamed" else ""
            out.append(r.make(verdict, f"{name}: " + "; ".join(parts) + note +
                              ". Outlet valves of FO/LO tanks above the double bottom must be closable from outside "
                              "the space (quick-closing or remotely operated); double-bottom tanks are exempt.",
                              involved, conf))
        elif unknown:
            out.append(r.make(V.INSUFFICIENT_DATA, f"{name}: flow direction of {', '.join(p for p, _, _ in unknown)} "
                                                   f"cannot be determined and its first valve is not quick-closing or "
                                                   f"remotely operated.", involved, conf))
        elif remote:
            what = f"remotely operated valve {ctx.names(dict.fromkeys(remote))}"
            if good:
                what = f"quick-closing valve {ctx.names(dict.fromkeys(good))} and " + what
            out.append(r.make(V.PASS, f"{name}: {what} directly on the tank outlet; verify that the remote closing "
                                      f"position is outside the space and that the valve can be closed from there in "
                                      f"the event of a fire.", involved, 0.8 * conf))
        elif good:
            out.append(r.make(V.PASS, f"{name}: quick-closing valve {ctx.names(dict.fromkeys(good))} directly on the "
                                      f"tank outlet.", involved, conf))
        else:
            out.append(r.make(V.NOT_APPLICABLE, f"{name}: only filling connections ({', '.join(inlet_only)}) on this "
                                                f"drawing; the tank outlet is shown elsewhere.", [t], conf))
    return out


@rule("R_FO_SERVICE_TANKS_MIN2", "At least two fuel oil service tanks", "연료유 서비스 탱크 2기 이상", Severity.WARNING,
      "SOLAS II-1/26.11; IACS UI SC123")
def _r_fo_service(ctx: RuleContext, r: Rule) -> list[Finding]:
    tanks = [t for t in ctx.ids(S.TANK) if _SERVICE_RE.search(ctx.label(t)) and _FUEL_LUBE_RE.search(ctx.label(t))
             and not re.search(r"\bL\.?\s?O\b|LUB", ctx.label(t))]
    applicable = ctx.has_system(ST.FUEL_OIL_SUPPLY) or (not ctx.systems_known and tanks)
    if not applicable:
        return [r.na("Not a fuel oil supply / service drawing.")]
    if not tanks:
        return [_none_shown(ctx, r, ST.FUEL_OIL_SUPPLY, "named fuel oil service tank")]
    dist = [t for t in tanks if _DISTILLATE_RE.search(ctx.label(t))]
    resid = [t for t in tanks if t not in dist]  # HFO / LSHFO / VLSFO and plain "F.O."
    conf = ctx.conf(tanks)
    groups = "; ".join(f"{label}: {ctx.names(ts)}" for label, ts in (("residual", resid), ("distillate", dist)) if ts)
    if len(tanks) == 1:
        return [r.make(V.FAIL, f"Only one fuel oil service tank ({ctx.names(tanks)}); two service tanks for each type "
                               f"of fuel used for propulsion and vital systems (or an equivalent arrangement) are "
                               f"required. Verify other drawings.", tanks, conf)]
    if all(len(g) >= 2 for g in (resid, dist) if g):
        return [r.make(V.PASS, f"At least two service tanks for each fuel type ({groups}).", tanks, conf)]
    if len(resid) >= 2:  # one distillate tank: the UI SC123 standard arrangement (cold start / flushing)
        return [r.make(V.PASS, f"Two or more residual fuel service tanks and one distillate service tank ({groups}). "
                               f"Sufficient when the distillate is only used for cold starting / repair work; if it "
                               f"also serves propulsion or generators (e.g. ECA operation), two distillate service "
                               f"tanks are required.", tanks, 0.7 * conf)]
    if len(dist) >= 2:  # one residual tank backed by two distillate tanks: UI SC123 equivalent arrangement
        return [r.make(V.PASS, f"One residual fuel service tank backed by {len(dist)} distillate service tanks "
                               f"({groups}): equivalent arrangement under IACS UI SC123 provided the distillate tanks "
                               f"can supply propulsion, generators and boilers at full load for 8 h - verify the "
                               f"capacities.", tanks, 0.6 * conf)]
    return [r.make(V.INSUFFICIENT_DATA, f"Only one service tank per fuel type ({groups}). SOLAS II-1/26.11 requires "
                                        f"two per fuel type; one residual plus one distillate tank is acceptable only "
                                        f"as an equivalent arrangement (IACS UI SC123) where each tank can supply "
                                        f"propulsion, generators and boilers at full load for 8 h, which cannot be "
                                        f"verified from this drawing.", tanks, conf, severity=Severity.WARNING)]


@rule("R_BILGE_SUCTION_NRV", "Non-return valve on every bilge suction branch", "빌지 흡입 분기관 역지(SDNR) 밸브",
      Severity.ERROR, "SOLAS II-1/35-1 (bilge suction branches: non-return valves)")
def _r_bilge_nrv(ctx: RuleContext, r: Rule) -> list[Finding]:
    wells, uncertain = _bilge_wells(ctx)
    if not wells:
        if ctx.has_system(ST.BILGE):
            return [_none_shown(ctx, r, ST.BILGE, "bilge well / suction")]
        return [r.na("No bilge suctions on this drawing.")]
    nrvs = [n for n in object_ids(ctx.G) if ctx.eff(n) in NRV_CLASSES]
    out = []
    for w in wells:
        name = ctx.name(w)
        cf = 0.5 if w in uncertain else 1.0
        if not ctx.links(w):
            out.append(r.make(V.INSUFFICIENT_DATA, f"{name} is not connected.", [w], cf * ctx.conf([w])))
            continue
        if not _leaves(ctx, w):
            out.append(r.make(V.INSUFFICIENT_DATA, f"{name}: suction line leads to no pump or main on this drawing.",
                              [w], cf * ctx.conf([w])))
            continue
        leaks = _leaves(ctx, w, blocked=nrvs)
        order, _ = ctx.reach(w, passable=lambda n: _p_through_valves(ctx, n) and ctx.eff(n) not in NRV_CLASSES)
        found = [n for n in order if ctx.is_obj(n) and ctx.eff(n) in NRV_CLASSES]
        conf = cf * ctx.conf([w, *found])
        if not leaks:
            out.append(r.make(V.PASS, f"{name}: suction branch protected by non-return valve {ctx.names(found)}.",
                              [w, *found], conf))
        elif w in uncertain:
            what = f"Tank {name} (not named as a bilge well)" if ctx.label(w) else f"Unnamed tank {name}"
            out.append(r.make(V.INSUFFICIENT_DATA, f"{what} reaches {ctx.names(leaks)} without a non-return valve; "
                                                   f"confirm whether it is a bilge suction.", [w, *leaks], conf))
        else:
            out.append(r.make(V.FAIL, f"{name}: suction branch reaches {ctx.names(leaks)} without a screw-down "
                                      f"non-return / check valve; back-flooding of the compartment is possible.",
                              [w, *leaks], conf))
    return out


@rule("R_BILGE_PUMPS_MIN2", "At least two power bilge pumps", "동력 빌지 펌프 2대 이상", Severity.WARNING,
      "SOLAS II-1/35-1 (number of power bilge pumps)")
def _r_bilge_pumps(ctx: RuleContext, r: Rule) -> list[Finding]:
    wells, uncertain = _bilge_wells(ctx)
    if not (ctx.has_system(ST.BILGE) or (not ctx.systems_known and wells)):
        return [r.na("Not a bilge drawing.")]
    pumps: list[str] = []
    certain = [w for w in wells if w not in uncertain]
    for group in (certain, [w for w in wells if w in uncertain]):  # uncertain suctions only as a fallback
        for w in group:
            res = ctx.net.search(w)
            pumps += [p for p in ctx.ids(*PUMP_CLASSES) if _driver_crossable_states(ctx, res, p)]
        pumps = [p for p in dict.fromkeys(pumps) if not _BILGE_PUMP_EXCL_RE.search(ctx.label(p))]
        if pumps:
            break
    basis = "connected to the bilge suctions"
    if not pumps:
        pumps = [p for p in ctx.ids(*PUMP_CLASSES)
                 if "BILGE" in ctx.label(p) and not _BILGE_PUMP_EXCL_RE.search(ctx.label(p))]
        basis = "named as bilge pumps"
    if not pumps:
        return [_none_shown(ctx, r, ST.BILGE, "bilge pump")]
    if len(pumps) >= 2:
        return [r.make(V.PASS, f"{len(pumps)} power pumps {basis}: {ctx.names(pumps)}.", pumps, ctx.conf(pumps))]
    return [r.make(V.FAIL, f"Only one power pump {basis} ({ctx.names(pumps)}). At least two power bilge pumps "
                           f"connected to the bilge main are required; another pump (e.g. ballast or general service "
                           f"pump) may be connected on a different drawing - verify.", pumps, ctx.conf(pumps))]


@rule("R_FIRE_PUMPS_MIN2", "At least two independently driven fire pumps", "독립 구동 소화 펌프 2대 이상",
      Severity.WARNING, "SOLAS II-2/10.2.2")
def _r_fire_pumps(ctx: RuleContext, r: Rule) -> list[Finding]:
    pumps = _fire_pumps(ctx)
    if not (ctx.has_system(ST.FIRE_MAIN) or (not ctx.systems_known and pumps)):
        return [r.na("Not a fire main drawing.")]
    if not pumps:
        return [_none_shown(ctx, r, ST.FIRE_MAIN, "fire pump")]
    if len(pumps) >= 2:
        return [r.make(V.PASS, f"{len(pumps)} fire pumps: {ctx.names(pumps)}.", pumps, ctx.conf(pumps))]
    return [r.make(V.FAIL, f"Only one fire pump on this drawing ({ctx.names(pumps)}); at least two independently "
                           f"driven fire pumps are required (the emergency fire pump may be on another drawing - "
                           f"verify).", pumps, ctx.conf(pumps))]


@rule("R_FIRE_PUMP_DISCHARGE_NRV", "Non-return valve on each fire pump discharge", "소화 펌프 토출측 역지 밸브",
      Severity.ERROR, "SOLAS II-2/10.2 (fire pumps and fire main)")
def _r_fire_nrv(ctx: RuleContext, r: Rule) -> list[Finding]:
    pumps = _fire_pumps(ctx)
    if not pumps:
        return [r.na("No fire pumps on this drawing.")]
    out = []
    for p in pumps:
        chk = _discharge_nrv(ctx, p)
        name = ctx.name(p)
        conf = ctx.conf([p, *chk.ok, *chk.reversed]) * (1.0 if chk.side_known else 0.6)
        if not chk.connected:
            out.append(r.make(V.INSUFFICIENT_DATA, f"{name} has no process connections.", [p], conf))
        elif chk.ok:
            out.append(r.make(V.PASS, f"{name}: non-return valve {ctx.names(chk.ok)} on the discharge.",
                              [p, *chk.ok], conf))
        elif chk.reversed:
            out.append(r.make(V.FAIL, f"{name}: discharge non-return valve {ctx.names(chk.reversed)} drawn reversed.",
                              [p, *chk.reversed], conf))
        else:
            out.append(r.make(V.FAIL, f"{name}: no non-return valve on its own discharge branch.", [p], conf))
    return out


@rule("R_OVERBOARD_VALVES", "Overboard discharge: positive-closing valve and non-return valve",
      "선외 배출: 차단 밸브 및 역지 밸브", Severity.WARNING, "ICLL Reg. 22; IACS UI LL11")
def _r_overboard(ctx: RuleContext, r: Rule) -> list[Finding]:
    obs = ctx.ids(S.OVERBOARD)
    if not obs:
        return [r.na("No overboard discharges on this drawing.")]
    nrv = [n for n in object_ids(ctx.G) if _is_nrv_like(ctx, n)]
    closing = [n for n in object_ids(ctx.G) if _is_closing_valve(ctx, n)]
    out = []
    for ob in obs:
        name = ctx.name(ob)
        leaves = _leaves(ctx, ob, passable=_p_any_valve)
        if not leaves:
            out.append(r.make(V.INSUFFICIENT_DATA, f"{name} is not connected to any equipment on this drawing.", [ob],
                              ctx.conf([ob])))
            continue
        order, _ = ctx.reach(ob, passable=lambda n: _p_any_valve(ctx, n))
        valves = [n for n in order if _is_valve(ctx, n)]
        no_nrv = _leaves(ctx, ob, blocked=nrv, passable=_p_any_valve)
        no_close = _leaves(ctx, ob, blocked=closing, passable=_p_any_valve)
        conf = ctx.conf([ob, *valves])
        if not no_nrv and not no_close:
            out.append(r.make(V.PASS, f"{name}: every route to the shell passes a positive-closing valve and a "
                                      f"non-return valve ({ctx.names(valves)}).", [ob, *valves], conf))
            continue
        missing = []
        if no_nrv:
            missing.append(f"no non-return valve on the route from {ctx.names(no_nrv, 4)}")
        if no_close:
            missing.append(f"no positive-closing valve on the route from {ctx.names(no_close, 4)}")
        out.append(r.make(V.FAIL, f"{name}: " + "; ".join(missing) + ". Shell discharges need an automatic non-return "
                                                                     "valve with positive means of closing (an SDNR "
                                                                     "valve satisfies both); locally operated valves "
                                                                     "may be accepted in manned machinery spaces.",
                          [ob, *valves, *no_nrv, *no_close], conf))
    return out


@rule("R_SW_SEA_CHESTS_MIN2", "At least two sea inlets for sea water cooling", "냉각 해수 씨체스트 2개 이상",
      Severity.WARNING, "Class rules: at least two sea chests (high / low) for cooling water")
def _r_sea_chests(ctx: RuleContext, r: Rule) -> list[Finding]:
    chests = ctx.ids(S.SEA_CHEST)
    sw_pumps = [p for p in ctx.ids(*PUMP_CLASSES) if _SW_PUMP_RE.search(ctx.label(p))]
    applicable = ctx.has_system(ST.SEA_WATER_COOLING) or (not ctx.systems_known and chests and sw_pumps)
    if not applicable:
        return [r.na("Not a sea water cooling drawing.")]
    inlets = chests + [c for c in ctx.ids(S.OFF_PAGE_CONNECTOR) if _SEA_INLET_RE.search(ctx.label(c))
                       and "FROM" in ctx.label(c)]
    if not inlets:
        return [_none_shown(ctx, r, ST.SEA_WATER_COOLING, "sea chest / sea inlet")]
    if len(inlets) >= 2:
        return [r.make(V.PASS, f"{len(inlets)} sea inlets: {ctx.names(inlets)}.", inlets, ctx.conf(inlets))]
    return [r.make(V.FAIL, f"Only one sea inlet ({ctx.names(inlets)}) feeds the sea water cooling system; at least two "
                           f"sea chests (e.g. high and low) are required.", inlets, ctx.conf(inlets))]


@rule("R_STRAINER_BEFORE_PUMP", "Strainer upstream of pumps drawing from sea chests / bilges",
      "씨체스트·빌지 흡입 펌프 전단 스트레이너", Severity.WARNING,
      "Class rules / good practice: sea suctions and bilge suctions fitted with strainers or mud boxes")
def _r_strainer(ctx: RuleContext, r: Rule) -> list[Finding]:
    wells, uncertain = _bilge_wells(ctx)
    sources = ctx.ids(S.SEA_CHEST) + wells
    if not sources:
        return [r.na("No sea chests or bilge suctions on this drawing.")]
    pumps = ctx.ids(*PUMP_CLASSES)
    strainers = [n for n in object_ids(ctx.G) if ctx.eff(n) in STRAINER_CLASSES]
    fed: dict[str, list[str]] = defaultdict(list)
    unstrained: dict[str, list[str]] = defaultdict(list)
    for s in sources:
        res = ctx.net.search(s)
        res_b = ctx.net.search(s, blocked=strainers)
        for p in pumps:
            if _driver_crossable_states(ctx, res, p):
                fed[p].append(s)
                if _driver_crossable_states(ctx, res_b, p):
                    unstrained[p].append(s)
    if not fed:
        return [r.make(V.INSUFFICIENT_DATA, f"No pump suction could be traced from {ctx.names(sources)}.",
                       sources, ctx.conf(sources))]
    out = []
    for p in pumps:
        if p not in fed:
            continue
        name = ctx.name(p)
        sure = [s for s in unstrained[p] if s not in uncertain]
        cf = 0.5 if all(s in uncertain for s in fed[p]) else 1.0
        if sure:
            unsure = [s for s in unstrained[p] if s in uncertain]
            extra = f" (also from {ctx.names(unsure)}, not confirmed as a bilge suction)" if unsure else ""
            out.append(r.make(V.FAIL, f"{name} draws from {ctx.names(sure)} without a strainer / mud box on the "
                                      f"suction{extra}.", [p, *sure, *unsure], ctx.conf([p, *sure])))
        elif unstrained[p]:
            out.append(r.make(V.INSUFFICIENT_DATA, f"{name} draws from {ctx.names(unstrained[p])} without a strainer / "
                                                   f"mud box, but the tank is not confirmed as a bilge suction (unnamed "
                                                   f"or not named as a bilge well).", [p, *unstrained[p]],
                              0.5 * ctx.conf([p, *unstrained[p]])))
        else:
            out.append(r.make(V.PASS, f"{name}: every suction route from {ctx.names(fed[p])} passes a strainer.",
                              [p, *fed[p]], cf * ctx.conf([p])))
    return out


def _ows_region(ctx: RuleContext, ows: str) -> list[str]:
    order, _ = ctx.reach(ows, passable=lambda n: _p_inline(ctx, n) or (_is_valve(ctx, n)
                                                                        and ctx.eff(n) != S.THREE_WAY_VALVE))
    return order


@rule("R_OWS_3WAY_RECIRC", "OWS outlet: three-way valve recirculating to bilge", "유수분리기 출구 3방향 밸브(재순환)",
      Severity.ERROR, "MARPOL Annex I Reg. 14.7 (automatic stopping / recirculation at 15 ppm)")
def _r_ows_3way(ctx: RuleContext, r: Rule) -> list[Finding]:
    owss = ctx.ids(S.OILY_WATER_SEPARATOR)
    if not owss:
        if ctx.has_system(ST.BILGE):
            return [r.na("No oily water separator on this bilge drawing.")]
        return [r.na("No oily water separator on this drawing.")]
    out = []
    for ows in owss:
        name = ctx.name(ows)
        if not ctx.links(ows):
            out.append(r.make(V.INSUFFICIENT_DATA, f"{name} is not connected.", [ows], ctx.conf([ows])))
            continue
        region = _ows_region(ctx, ows)
        threes = [n for n in region if ctx.is_obj(n) and ctx.eff(n) == S.THREE_WAY_VALVE]
        if not threes:
            out.append(r.make(V.FAIL, f"{name}: no three-way valve on the separator outlet to stop overboard "
                                      f"discharge and recirculate above 15 ppm.", [ows], ctx.conf([ows])))
            continue
        overboards = ctx.ids(S.OVERBOARD)
        recirc: list[str] = []
        for tv in threes:
            branch = [lk for lk in ctx.links(tv) if lk.port == "branch"]
            via = (lambda lk: lk.port == "branch") if branch else None
            order, _ = ctx.reach(tv, via=via, passable=lambda n: _p_through_valves(ctx, n),
                                 blocked=[ows, *overboards])
            for n in order:
                if not ctx.is_obj(n) or _p_through_valves(ctx, n):
                    continue
                if ctx.eff(n) == S.TANK or (ctx.eff(n) == S.OFF_PAGE_CONNECTOR
                                            and _BILGE_DEST_RE.search(ctx.label(n))):
                    recirc.append(n)
        conf = ctx.conf([ows, *threes])
        if recirc:
            out.append(r.make(V.PASS, f"{name}: three-way valve {ctx.names(threes)} recirculates to "
                                      f"{ctx.names(dict.fromkeys(recirc))}.", [ows, *threes, *recirc], conf))
        else:
            out.append(r.make(V.FAIL, f"{name}: three-way valve {ctx.names(threes)} found, but its branch does not "
                                      f"lead back to a bilge / holding tank on this drawing.", [ows, *threes], conf))
    return out


@rule("R_OWS_15PPM_ALARM", "OWS outlet: 15 ppm oil content meter / alarm", "유수분리기 출구 15ppm 경보장치",
      Severity.ERROR, "MARPOL Annex I Reg. 14.7")
def _r_ows_alarm(ctx: RuleContext, r: Rule) -> list[Finding]:
    owss = ctx.ids(S.OILY_WATER_SEPARATOR)
    if not owss:
        return [r.na("No oily water separator on this drawing.")]
    out = []
    for ows in owss:
        name = ctx.name(ows)
        region = [ows] + _ows_region(ctx, ows)
        instruments: list[str] = []
        for n in dict.fromkeys(region):
            for lk in ctx.links(n, ALL_KINDS):
                m = lk.other
                if ctx.is_obj(m) and ctx.cat(m) == Category.INSTRUMENT:
                    instruments.append(m)
                elif ctx.is_junction(m):
                    instruments += [x.other for x in ctx.links(m, frozenset({PipeKind.INSTRUMENT_TAP}))
                                    if ctx.is_obj(x.other) and ctx.cat(x.other) == Category.INSTRUMENT]
        meters = []
        for i in dict.fromkeys(instruments):
            letters = (ctx.objs[i].attributes.get("letters") or ctx.tag(i) or "").upper()
            if re.match(r"^[QA]", letters) or re.search(r"PPM|OIL\s*CONTENT|OCM|BILGE\s*ALARM", ctx.label(i)):
                meters.append(i)
        if meters:
            out.append(r.make(V.PASS, f"{name}: oil content meter / alarm {ctx.names(meters)} on the outlet.",
                              [ows, *meters], ctx.conf([ows, *meters])))
        else:
            out.append(r.make(V.FAIL, f"{name}: no oil content meter / 15 ppm bilge alarm (Q*/A* instrument) found on "
                                      f"the separator outlet.", [ows], ctx.conf([ows])))
    return out


def _air_applicable(ctx: RuleContext) -> bool:
    return ctx.has_system(ST.COMPRESSED_AIR) or (
        not ctx.systems_known and bool(ctx.ids(S.COMPRESSOR) or ctx.ids(S.AIR_RECEIVER)))


def _start_air_count(ctx: RuleContext, r: Rule, cls: SymbolClass, what: str) -> list[Finding]:
    """Count main starting-air units of ``cls``; emergency / service / control / topping-up /
    generator units are listed but not counted."""
    if not _air_applicable(ctx):
        return [r.na("Not a compressed / starting air drawing.")]
    units = ctx.ids(cls)
    if not units:
        return [_none_shown(ctx, r, ST.COMPRESSED_AIR, f"air {what}")]
    main = [u for u in units if not _NOT_START_AIR_RE.search(ctx.label(u))]
    other = [u for u in units if u not in main]
    excl = f" Not counted (emergency / service / control / auxiliary units): {ctx.names(other)}." if other else ""
    unnamed = [u for u in main if not ctx.label(u)]
    conf = ctx.conf(main or units) * (0.8 if unnamed else 1.0)
    if len(main) >= 2:
        return [r.make(V.PASS, f"{len(main)} main starting air {what}s: {ctx.names(main)}.{excl}", main, conf)]
    if not main:
        return [r.make(V.INSUFFICIENT_DATA, f"No main starting air {what} on this drawing; only {ctx.names(other)}, "
                                            f"which do not count towards the starting-air requirement. The main "
                                            f"starting air system may be on another drawing.", other, conf)]
    return [r.make(V.FAIL, f"Only one main starting air {what} ({ctx.names(main)}); at least two are required.{excl}",
                   main + other, conf)]


@rule("R_START_AIR_COMPRESSORS_MIN2", "At least two starting air compressors", "시동 공기 압축기 2대 이상",
      Severity.WARNING, "Class rules (main engine starting air: two or more compressors)")
def _r_air_comp(ctx: RuleContext, r: Rule) -> list[Finding]:
    return _start_air_count(ctx, r, S.COMPRESSOR, "compressor")


@rule("R_START_AIR_RECEIVERS_MIN2", "At least two starting air receivers", "시동 공기 저장조 2기 이상",
      Severity.WARNING, "Class rules (main engine starting air: two or more receivers)")
def _r_air_recv(ctx: RuleContext, r: Rule) -> list[Finding]:
    return _start_air_count(ctx, r, S.AIR_RECEIVER, "receiver")


@rule("R_AIR_RECEIVER_RELIEF", "Safety valve on each air receiver", "공기 저장조 안전밸브", Severity.ERROR,
      "Class rules (pressure vessels: safety valve on each air receiver)")
def _r_air_relief(ctx: RuleContext, r: Rule) -> list[Finding]:
    recs = ctx.ids(S.AIR_RECEIVER)
    if not recs:
        return [r.na("No air receivers on this drawing.")]
    out = []
    for rv in recs:
        name = ctx.name(rv)
        if not ctx.links(rv):
            out.append(r.make(V.INSUFFICIENT_DATA, f"{name} is not connected.", [rv], ctx.conf([rv])))
            continue
        order, _ = ctx.reach(rv, passable=lambda n: ctx.is_junction(n) or (ctx.is_obj(n)
                                                                            and ctx.cat(n) == Category.FITTING))
        svs = [n for n in order if ctx.is_obj(n) and ctx.eff(n) == S.SAFETY_VALVE]
        if svs:
            out.append(r.make(V.PASS, f"{name}: safety valve {ctx.names(svs)} fitted.", [rv, *svs],
                              ctx.conf([rv, *svs])))
        else:
            out.append(r.make(V.FAIL, f"{name}: no safety valve connected directly to the receiver.", [rv],
                              ctx.conf([rv])))
    return out


@rule("R_PUMP_ISOLATION", "Pump suction and discharge isolation valves", "펌프 흡입·토출 차단 밸브", Severity.INFO,
      "Good practice (maintenance isolation of pumps)")
def _r_pump_isolation(ctx: RuleContext, r: Rule) -> list[Finding]:
    pumps = ctx.ids(*PUMP_CLASSES)
    if not pumps:
        return [r.na("No pumps on this drawing.")]
    out = []
    for p in pumps:
        name = ctx.name(p)
        branches = isolation_analysis(ctx.doc, ctx.G, p)
        if not branches:
            out.append(r.make(V.INSUFFICIENT_DATA, f"{name} has no process connections.", [p], ctx.conf([p])))
            continue
        open_br = [b for b in branches if not b.isolated]
        valves = list(dict.fromkeys(v for b in branches for v in b.valves))
        if not open_br:
            out.append(r.make(V.PASS, f"{name} can be isolated by {ctx.names(valves)}.", [p, *valves],
                              ctx.conf([p])))
        else:
            what = "; ".join(f"{b.port or b.pipe_id} side open to {ctx.names(b.open_to, 3) or 'nothing (no valve)'}"
                             for b in open_br)
            out.append(r.make(V.FAIL, f"{name} cannot be fully isolated: {what}.", [p, *valves], ctx.conf([p])))
    return out


# --------------------------------------------------------------------------- drawing quality


@rule("DQ_UNCONNECTED_OBJECTS", "Objects without any pipe connection", "배관 연결이 없는 객체", Severity.WARNING,
      "Drawing quality", group="quality")
def _q_unconnected(ctx: RuleContext, r: Rule) -> list[Finding]:
    lonely = [n for n in object_ids(ctx.G) if not ctx.links(n, ALL_KINDS)]
    if not lonely:
        return [r.make(V.PASS, f"All {len(object_ids(ctx.G))} objects are connected.")]
    by_cat = Counter(_value(ctx.cat(n)) for n in lonely)
    detail = ", ".join(f"{k}: {v}" for k, v in sorted(by_cat.items()))
    return [r.make(V.FAIL, f"{len(lonely)} objects have no pipe or signal connection ({detail}): {ctx.names(lonely, 12)}.",
                   lonely, 1.0)]


@rule("DQ_OPEN_INLINE_ITEMS", "Valves / fittings connected on one side only", "한쪽만 연결된 밸브·부속",
      Severity.INFO, "Drawing quality", group="quality")
def _q_open_inline(ctx: RuleContext, r: Rule) -> list[Finding]:
    items = [n for n in object_ids(ctx.G)
             if ctx.cat(n) in (Category.VALVE, Category.FITTING) and ctx.eff(n) != S.SAFETY_VALVE
             and len(ctx.links(n)) == 1]
    if not items:
        return [r.make(V.PASS, "Every in-line valve and fitting is connected on both sides.")]
    return [r.make(V.FAIL, f"{len(items)} valves/fittings have a single process connection (open end or missed "
                           f"connection): {ctx.names(items, 12)}.", items)]


@rule("DQ_DANGLING_PIPE_ENDS", "Pipe ends not connected to anything", "연결되지 않은 배관 끝단", Severity.WARNING,
      "Drawing quality", group="quality")
def _q_dangling(ctx: RuleContext, r: Rule) -> list[Finding]:
    free = [n for n, d in ctx.G.nodes(data=True) if d.get("kind") == "free"]
    if not free:
        return [r.make(V.PASS, f"All {ctx.G.number_of_edges()} pipe runs end at an object or a junction.")]
    pipes = list(dict.fromkeys(ctx.G.nodes[n]["pipe_id"] for n in free))
    refs = sorted({ctx.G.nodes[n]["ref"] for n in free if ctx.G.nodes[n].get("ref")})
    extra = f" ({len(refs)} reference ids that do not exist: {', '.join(refs[:8])})" if refs else ""
    kinds = Counter(_value(ctx.G.edges[e]["kind"]) for e in ctx.G.edges(keys=True)
                    if any(ctx.G.nodes[x].get("kind") == "free" for x in e[:2]))
    return [r.make(V.FAIL, f"{len(free)} free pipe ends on {len(pipes)} runs ({', '.join(f'{k}: {v}' for k, v in sorted(kinds.items()))})"
                           f"{extra}: {', '.join(pipes[:15])}.", pipes)]


def _norm_tag(t: str) -> str:
    return re.sub(r"\s+", "", t.upper())


@rule("DQ_DUPLICATE_TAGS", "Duplicate tags", "중복 태그", Severity.WARNING, "Drawing quality", group="quality")
def _q_duplicates(ctx: RuleContext, r: Rule) -> list[Finding]:
    by_tag: dict[str, list[str]] = defaultdict(list)
    for n in object_ids(ctx.G):
        if ctx.tag(n):
            by_tag[_norm_tag(ctx.tag(n))].append(n)
    dups = {t: ns for t, ns in by_tag.items() if len(ns) > 1}
    if not dups:
        return [r.make(V.PASS, f"All {len(by_tag)} tags are unique.")]
    return [r.make(V.FAIL, f"Tag {t} is used by {len(ns)} objects: {ctx.names(ns)}.", ns) for t, ns in sorted(dups.items())]


@rule("DQ_MISSING_TAGS", "Equipment / valves / instruments without identification", "식별 태그 누락",
      Severity.INFO, "Drawing quality (ISA 5.1 / yard tagging practice)", group="quality")
def _q_missing_tags(ctx: RuleContext, r: Rule) -> list[Finding]:
    missing: dict[str, list[str]] = defaultdict(list)
    for n in object_ids(ctx.G):
        c = ctx.cat(n)
        if c == Category.EQUIPMENT and not ctx.tag(n) and not ctx.label(n):
            missing["equipment"].append(n)
        elif c in (Category.VALVE, Category.INSTRUMENT) and not ctx.tag(n):
            missing[_value(c)].append(n)
    ids = [n for v in missing.values() for n in v]
    if not ids:
        return [r.make(V.PASS, "Every equipment item has a tag or name; every valve and instrument has a tag.")]
    detail = ", ".join(f"{k}: {len(v)}" for k, v in sorted(missing.items()))
    return [r.make(V.FAIL, f"{len(ids)} objects without identification ({detail}): {ctx.names(ids, 12)}.", ids)]


@rule("DQ_SIZE_CHANGE_NO_REDUCER", "Pipe size change without reducer", "레듀서 없는 관경 변화", Severity.WARNING,
      "Drawing quality", group="quality")
def _q_size_change(ctx: RuleContext, r: Rule) -> list[Finding]:
    issues: list[tuple[str, list[str], str]] = []
    checked = 0
    for n in ctx.G.nodes:
        d = ctx.node(n)
        if d.get("kind") == "object":
            if ctx.cat(n) not in (Category.VALVE, Category.FITTING) or ctx.eff(n) in (S.REDUCER, S.THREE_WAY_VALVE,
                                                                                      S.SAFETY_VALVE):
                continue
        elif d.get("kind") != "junction":
            continue
        lks = ctx.links(n)
        pairs = []
        if len(lks) == 2:
            pairs.append((lks[0], lks[1]))
        elif d.get("kind") == "junction" and len(lks) > 2:
            # the run of a tee may legitimately reduce (reducing tee) unless it is the same line
            pairs += [(a, b) for a, b in _collinear_pairs(ctx, n, lks) if _same_line(ctx, n, a, b)]
        for a, b in pairs:
            sa = _pipe_size(ctx, ctx.G.edges[n, a.other, a.pipe_id])
            sb = _pipe_size(ctx, ctx.G.edges[n, b.other, b.pipe_id])
            if sa is None or sb is None:
                continue
            checked += 1
            if sa != sb:
                issues.append((n, [a.pipe_id, b.pipe_id], f"{sa[0]}{'A' if sa[1] == 'mm' else 'B'} / "
                                                          f"{sb[0]}{'A' if sb[1] == 'mm' else 'B'}"))
    if not any(_pipe_size(ctx, d) for _, _, d in ctx.G.edges(data=True) if d["kind"] == PipeKind.PROCESS):
        return [r.make(V.INSUFFICIENT_DATA, "No pipe sizes known on this drawing.")]
    if not issues:
        return [r.make(V.PASS, f"No size change without a reducer ({checked} run-to-run transitions checked).")]
    return [r.make(V.FAIL, f"Size changes at {node_name(ctx.G, n)} ({sizes}) between {a} and {b} without a reducer.",
                   ([n] if ctx.is_obj(n) else []) + [a, b]) for n, (a, b), sizes in issues]


def _line_serial(line_number: Optional[str]) -> Optional[tuple[str, str]]:
    parsed = parse_line_number(line_number) if line_number else None
    return (parsed["system"], parsed["number"]) if parsed else None


def _same_line(ctx: RuleContext, n: str, a: Link, b: Link) -> bool:
    sa = _line_serial(ctx.G.edges[n, a.other, a.pipe_id].get("line_number"))
    sb = _line_serial(ctx.G.edges[n, b.other, b.pipe_id].get("line_number"))
    return sa is not None and sa == sb


def _collinear_pairs(ctx: RuleContext, n: str, lks: list[Link]) -> list[tuple[Link, Link]]:
    """Pairs of pipes continuing straight through a junction (the run of a tee)."""
    pos = ctx.node(n).get("pos")
    if not pos:
        return []
    dirs = []
    for lk in lks:
        e = ctx.G.edges[n, lk.other, lk.pipe_id]
        pts = ctx.doc.pipes[e["order"]].points
        if len(pts) < 2:
            dirs.append(None)
            continue
        near, nxt = (pts[0], pts[1]) if e["start_node"] == n else (pts[-1], pts[-2])
        dx, dy = nxt.x - near.x, nxt.y - near.y
        norm = (dx * dx + dy * dy) ** 0.5
        dirs.append((dx / norm, dy / norm) if norm > 1e-9 else None)
    out = []
    for i in range(len(lks)):
        for j in range(i + 1, len(lks)):
            if dirs[i] and dirs[j] and dirs[i][0] * dirs[j][0] + dirs[i][1] * dirs[j][1] < -0.95:
                out.append((lks[i], lks[j]))
    return out


@rule("DQ_LINE_SYSTEM_CODE", "Line number system code matches the drawing system", "라인 번호 계통 코드 일치",
      Severity.INFO, "Drawing quality (line numbering convention)", group="quality")
def _q_line_codes(ctx: RuleContext, r: Rule) -> list[Finding]:
    if not ctx.systems_known:
        return [r.na("Drawing system unknown.")]
    codes = {SYSTEMS[s].code for s in ctx.systems if s in SYSTEMS}
    bad: dict[str, list[str]] = defaultdict(list)
    interface: set[str] = set()
    total = 0
    for u, v, k, d in ctx.G.edges(keys=True, data=True):
        if d["kind"] != PipeKind.PROCESS or not d.get("line_number"):
            continue
        parsed = parse_line_number(d["line_number"])
        if not parsed:
            continue
        total += 1
        if parsed["system"] in codes:
            continue
        ends = [x for x in (u, v) if ctx.is_obj(x)]
        if any(ctx.eff(x) in (S.OFF_PAGE_CONNECTOR, S.HEAT_EXCHANGER, S.PLATE_HEAT_EXCHANGER, S.HEATER) for x in ends):
            interface.add(d["line_number"])
            continue
        bad[d["line_number"]].append(k)
    if not total:
        return [r.make(V.INSUFFICIENT_DATA, "No parsable line numbers on this drawing.")]
    note = f" ({len(interface)} interface lines to other systems accepted)" if interface else ""
    if not bad:
        return [r.make(V.PASS, f"{total} numbered pipe runs use the drawing system code "
                               f"{'/'.join(sorted(codes))}{note}.")]
    return [r.make(V.FAIL, f"Line {ln}: system code {parse_line_number(ln)['system']} differs from the drawing system "
                           f"code {'/'.join(sorted(codes))} (verify whether it is an interface line){note}.", pids)
            for ln, pids in sorted(bad.items())]
