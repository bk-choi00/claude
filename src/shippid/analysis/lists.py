"""Equipment, valve, instrument and line lists (one ``dict`` per row, JSON/CSV friendly).

Rows only contain plain values (str / int / float / bool / None); multi-valued cells are
joined with ``"; "``. Names follow the analysis graph: ``label [tag]`` where both exist.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Optional, Sequence

import networkx as nx

from ..catalog import (
    INSTRUMENT_TAG_RE,
    ISA_FIRST_LETTER,
    ISA_FUNCTIONS,
    SYSTEM_BY_CODE,
    SYSTEMS,
    parse_line_number,
)
from ..model import ActuatorType, Category, FlowPath, Medium, PIDDocument, PipeKind, SymbolClass, TextRole
from ..symbols import get_symbol
from .graph import (
    ALL_KINDS,
    CONNECT_KINDS,
    PROCESS_KINDS,
    build_graph,
    isolation_valves,
    links,
    node_name,
)

S = SymbolClass

# ISA 5.1 letters not in the catalogue: first letters (measured / initiating variable),
# modifiers that may follow the first letter, readout / function letters, and the qualifiers
# that follow an alarm or switch letter (H / L / M = high / low / intermediate; O / C = open /
# closed, e.g. ZSO, HSC).
_ISA_FIRST_EXTRA = {"B": "burner/flame", "E": "voltage", "H": "hand", "I": "current", "J": "power", "K": "time",
                    "M": "moisture", "R": "radiation", "U": "multivariable", "W": "weight/force", "X": "unclassified",
                    "Y": "event/state"}
_ISA_MODIFIERS = {"D": "differential", "F": "ratio", "Q": "integrate/totalize", "J": "scan", "K": "time rate"}
_ISA_EXTRA_FUNCTIONS = {"Y": "relay/compute", "Z": "actuator", "K": "control station", "L": "light",
                        "O": "orifice", "W": "well", "B": "burner", "U": "multifunction", "X": "unclassified",
                        "P": "test point"}
_LEVEL_QUALIFIERS = {"H": "high", "L": "low", "M": "intermediate"}
_STATE_QUALIFIERS = {"O": "open", "C": "closed"}

_ACTUATOR_KO = {
    ActuatorType.NONE: "-", ActuatorType.HANDWHEEL: "수동(핸드휠)", ActuatorType.DIAPHRAGM: "공압(다이어프램)",
    ActuatorType.HYDRAULIC: "유압 원격", ActuatorType.ELECTRIC: "전동", ActuatorType.QUICK_CLOSING: "급속차단",
}
_REMOTE = {ActuatorType.DIAPHRAGM, ActuatorType.HYDRAULIC, ActuatorType.ELECTRIC, ActuatorType.QUICK_CLOSING}
_LOCATION = {S.INSTRUMENT_LOCAL: "local (field)", S.INSTRUMENT_PANEL: "panel / remote",
             S.INSTRUMENT_SHARED: "shared display (AMS/DCS)"}


def _join(xs: Sequence[Any]) -> str:
    return "; ".join(str(x) for x in dict.fromkeys(x for x in xs if x not in (None, "")))


def _type_labels(cls: SymbolClass) -> tuple[str, str]:
    try:
        spec = get_symbol(cls)
        return spec.label_en, spec.label_ko
    except KeyError:
        return ("Unknown symbol", "미분류 기호")


def _pos(G: nx.MultiGraph, n: str) -> tuple[Optional[float], Optional[float]]:
    p = G.nodes[n].get("pos")
    return (round(p[0], 1), round(p[1], 1)) if p else (None, None)


def _neighbours(G: nx.MultiGraph, n: str, kinds=CONNECT_KINDS) -> list[str]:
    """Objects connected to ``n`` directly or through junctions (junction-collapsed)."""
    seen, out, stack = {n}, [], [n]
    while stack:
        cur = stack.pop()
        for lk in links(G, cur, kinds):
            m = lk.other
            if m in seen:
                continue
            seen.add(m)
            kind = G.nodes[m].get("kind")
            if kind == "object":
                out.append(m)
            elif kind == "junction":
                stack.append(m)
    return sorted(out, key=lambda m: G.nodes[m].get("order", 0))


def _pipe_attrs(G: nx.MultiGraph, n: str, kinds=PROCESS_KINDS) -> list[dict]:
    return [G.edges[n, lk.other, lk.pipe_id] for lk in links(G, n, kinds)]


def _medium_of(edges: list[dict]) -> str:
    media = [e["medium"] for e in edges if e["medium"] != Medium.UNKNOWN]
    if media:
        return Counter(media).most_common(1)[0][0].value
    for e in edges:
        parsed = parse_line_number(e["line_number"]) if e.get("line_number") else None
        if parsed and parsed["system"] in SYSTEM_BY_CODE:
            return SYSTEMS[SYSTEM_BY_CODE[parsed["system"]]].medium.value
    return Medium.UNKNOWN.value


def _size_of(e: dict) -> Optional[str]:
    if e.get("size"):
        return e["size"]
    parsed = parse_line_number(e["line_number"]) if e.get("line_number") else None
    return parsed["size"] if parsed else None


def _size_num(size: Optional[str]) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", size or "")
    return float(m.group(1)) if m else -1.0


def _base(G: nx.MultiGraph, doc: PIDDocument, n: str) -> dict[str, Any]:
    o = doc.object_by_id()[n]
    eff = G.nodes[n]["eff_cls"]
    en, ko = _type_labels(eff)
    x, y = _pos(G, n)
    return {"id": n, "tag": o.tag or "", "name": G.nodes[n].get("label") or "", "class": eff.value,
            "type_en": en, "type_ko": ko, "page": o.page, "x": x, "y": y,
            "confidence": round(o.confidence, 3), "source": o.source}


# --------------------------------------------------------------------------- lists


def equipment_list(doc: PIDDocument, G: Optional[nx.MultiGraph] = None) -> list[dict[str, Any]]:
    """Equipment rows: tag, name, type, connections, connected objects, isolation valves."""
    G = G if G is not None else build_graph(doc)
    rows = []
    for o in doc.objects:
        if G.nodes[o.id].get("eff_category") != Category.EQUIPMENT:
            continue
        row = _base(G, doc, o.id)
        rating = o.attributes.get("rating") or o.attributes.get("capacity")
        row["rating"] = str(rating) if rating else ""
        row["connections"] = len(links(G, o.id, PROCESS_KINDS))
        row["connected_to"] = _join(node_name(G, m) for m in _neighbours(G, o.id, PROCESS_KINDS))
        row["isolation_valves"] = _join(node_name(G, v) for v in isolation_valves(doc, G, o.id))
        rows.append(row)
    return rows


def valve_list(doc: PIDDocument, G: Optional[nx.MultiGraph] = None) -> list[dict[str, Any]]:
    """Valve rows: type, actuator, remote operation, size, line numbers, medium, neighbours."""
    G = G if G is not None else build_graph(doc)
    rows = []
    for o in doc.objects:
        if G.nodes[o.id].get("eff_category") != Category.VALVE:
            continue
        row = _base(G, doc, o.id)
        edges = _pipe_attrs(G, o.id)
        sizes = [s for s in (_size_of(e) for e in edges) if s]
        row.update({
            "actuator": o.actuator.value,
            "actuator_ko": _ACTUATOR_KO.get(o.actuator, o.actuator.value),
            "remote_operated": o.actuator in _REMOTE,
            "size": max(sizes, key=_size_num) if sizes else "",
            "line_numbers": _join(e.get("line_number") for e in edges),
            "medium": _medium_of(edges),
            "connections": len(edges),
            "connected_to": _join(node_name(G, m) for m in _neighbours(G, o.id, PROCESS_KINDS)),
            "service": str(o.attributes.get("service") or ""),
        })
        rows.append(row)
    return rows


def decode_isa(letters: str) -> tuple[str, list[str]]:
    """ISA 5.1 letters -> (measured variable, functions), e.g. ``"PIC"`` ->
    ``("pressure", ["indicator", "controller"])``, ``"LAL"`` -> ``("level", ["alarm low"])``,
    ``"PSLL"`` -> ``("pressure", ["switch low low"])``, ``"ZSO"`` -> ``("position", ["switch open"])``,
    ``"PSV"`` -> ``("pressure safety", ["valve"])``."""
    letters = letters.upper().strip()
    if not letters:
        return ("", [])
    first = letters[0]
    variable = ISA_FIRST_LETTER.get(first) or _ISA_FIRST_EXTRA.get(first) or f"'{first}'"
    funcs: list[str] = []
    rest = letters[1:]
    if rest in ("SV", "SE"):  # safety valve / safety element (rupture disc): S is the "safety" modifier
        return (f"{variable} safety", [ISA_FUNCTIONS[rest[1]]])
    if rest and rest[0] in _ISA_MODIFIERS and rest[0] not in ISA_FUNCTIONS:
        variable = f"{variable} {_ISA_MODIFIERS[rest[0]]}"
        rest = rest[1:]
    i = 0
    while i < len(rest):
        c = rest[i]
        name = ISA_FUNCTIONS.get(c) or _ISA_EXTRA_FUNCTIONS.get(c) or f"'{c}'"
        # qualifiers: high / low after an alarm or switch, open / closed after a switch or light
        allowed = {**(_LEVEL_QUALIFIERS if c in ("A", "S") else {}), **(_STATE_QUALIFIERS if c in ("S", "L") else {})}
        while i + 1 < len(rest) and rest[i + 1] in allowed:
            name += " " + allowed[rest[i + 1]]
            i += 1
        funcs.append(name)
        i += 1
    return (variable, funcs)


def instrument_letters(doc: PIDDocument, obj_id: str) -> tuple[str, str]:
    """(ISA letters, loop number) from attributes, the tag or the text inside the bubble."""
    o = doc.object_by_id()[obj_id]
    letters = str(o.attributes.get("letters") or "")
    number = str(o.attributes.get("loop") or "")
    if not letters and o.tag:
        m = INSTRUMENT_TAG_RE.match(o.tag.strip().upper().replace(" ", "-"))
        if m:
            letters, number = m.group("letters"), number or m.group("number")
        else:
            m = re.match(r"^([A-Z]{1,5})[-\s]?(\d{1,5}[A-Z]?)$", o.tag.strip().upper())
            if m:
                letters, number = m.group(1), number or m.group(2)
    if not letters:
        inside = sorted((t for t in doc.texts if t.owner_id == obj_id and t.role == TextRole.INSTRUMENT),
                        key=lambda t: (t.bbox.y0, t.bbox.x0))
        for t in inside:
            txt = t.text.strip().upper()
            if not letters and re.fullmatch(r"[A-Z]{1,5}", txt):
                letters = txt
            elif not number and re.fullmatch(r"\d{1,5}[A-Z]?", txt):
                number = txt
    return letters, number


def instrument_list(doc: PIDDocument, G: Optional[nx.MultiGraph] = None) -> list[dict[str, Any]]:
    """Instrument rows with ISA letters decoded via the catalogue."""
    G = G if G is not None else build_graph(doc)
    rows = []
    for o in doc.objects:
        if G.nodes[o.id].get("eff_category") != Category.INSTRUMENT:
            continue
        row = _base(G, doc, o.id)
        letters, number = instrument_letters(doc, o.id)
        variable, funcs = decode_isa(letters) if letters else ("", [])
        measured_on: list[str] = []
        for lk in links(G, o.id, CONNECT_KINDS):
            m = lk.other
            kind = G.nodes[m].get("kind")
            if kind == "object":
                measured_on.append(node_name(G, m))
            elif kind == "junction":
                lines = [G.edges[m, x.other, x.pipe_id].get("line_number") for x in links(G, m, PROCESS_KINDS)]
                lines = [ln for ln in lines if ln]
                near = [node_name(G, q) for q in _neighbours(G, m, PROCESS_KINDS)]
                measured_on.append(lines[0] if lines else (" / ".join(near[:2]) or m))
        signal_to = [node_name(G, m) for m in _neighbours(G, o.id, frozenset({PipeKind.SIGNAL}))]
        row.update({
            "letters": letters,
            "loop": number,
            "variable": variable,
            "function": ", ".join(funcs),
            "description": " ".join([variable] + funcs).strip(),
            "location": _LOCATION.get(G.nodes[o.id]["eff_cls"], ""),
            "measured_on": _join(measured_on),
            "signal_to": _join(signal_to),
            "service": str(o.attributes.get("service") or ""),
        })
        rows.append(row)
    return rows


class _DSU:
    def __init__(self, items: Sequence[str]):
        self.parent = {i: i for i in items}
        self.labels: dict[str, set[str]] = {i: set() for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if len(self.labels[ra] | self.labels[rb]) > 1:
            return False
        self.parent[rb] = ra
        self.labels[ra] |= self.labels[rb]
        return True


def line_groups(doc: PIDDocument, G: nx.MultiGraph) -> list[list[str]]:
    """Process pipe runs grouped into lines: runs sharing a line number, plus unnumbered runs
    continuing them through in-line valves / fittings / corners (never merging two numbers)."""
    pipes = [p for p in doc.pipes if p.kind == PipeKind.PROCESS]
    dsu = _DSU([p.id for p in pipes])
    by_number: dict[str, list[str]] = defaultdict(list)
    for p in pipes:
        if p.line_number:
            num = p.line_number.strip().upper()
            dsu.labels[p.id].add(num)
            by_number[num].append(p.id)
    for ids in by_number.values():
        for x in ids[1:]:
            dsu.union(ids[0], x)
    changed = True
    while changed:  # propagate through 2-way in-line nodes until stable
        changed = False
        for n, d in G.nodes(data=True):
            kind = d.get("kind")
            inline = kind == "junction" or (kind == "object" and d.get("eff_category") in (Category.VALVE,
                                                                                            Category.FITTING))
            if not inline:
                continue
            lks = links(G, n, PROCESS_KINDS)
            if len(lks) != 2:
                continue
            a, b = lks[0].pipe_id, lks[1].pipe_id
            if dsu.find(a) != dsu.find(b) and dsu.union(a, b):
                changed = True
    groups: dict[str, list[str]] = defaultdict(list)
    for p in pipes:
        groups[dsu.find(p.id)].append(p.id)
    return list(groups.values())


def line_list(doc: PIDDocument, G: Optional[nx.MultiGraph] = None,
              flow_paths: Optional[Sequence[FlowPath]] = None) -> list[dict[str, Any]]:
    """One row per line (see :func:`line_groups`): number, size, medium, system, spec, from / to,
    in-line valves, length in points. Direction from the flow paths when they cover the line,
    else left-to-right."""
    G = G if G is not None else build_graph(doc)
    pipe_by_id = {p.id: p for p in doc.pipes}
    edge_by_id = {k: d for _, _, k, d in G.edges(keys=True, data=True)}
    rows = []
    for ids in line_groups(doc, G):
        members = set(ids)
        deg: Counter[str] = Counter()
        for pid in ids:
            e = edge_by_id[pid]
            deg[e["start_node"]] += 1
            deg[e["end_node"]] += 1
        terminals = [n for n, c in deg.items() if c == 1 or G.nodes[n].get("kind") == "free"
                     or (G.nodes[n].get("kind") == "object" and G.nodes[n].get("eff_category")
                         not in (Category.VALVE, Category.FITTING))]
        terminals = _order_terminals(G, terminals, flow_paths or [])
        numbers = [pipe_by_id[p].line_number for p in ids if pipe_by_id[p].line_number]
        number = Counter(n.strip() for n in numbers).most_common(1)[0][0] if numbers else ""
        parsed = parse_line_number(number) if number else None
        edges = [edge_by_id[p] for p in ids]
        sizes = [s for s in (_size_of(e) for e in edges) if s]
        size = Counter(sizes).most_common(1)[0][0] if sizes else (parsed["size"] if parsed else "")
        inline = [n for n, c in deg.items() if c >= 2 and G.nodes[n].get("kind") == "object"
                  and G.nodes[n].get("eff_category") in (Category.VALVE, Category.FITTING)]
        inline.sort(key=lambda n: G.nodes[n].get("order", 0))
        names = [_terminal_name(G, n, members) for n in terminals]
        rows.append({
            "line_number": number,
            "size": size or "",
            "medium": _medium_of(edges),
            "system_code": parsed["system"] if parsed else "",
            "spec": parsed["spec"] if parsed else "",
            "from": names[0] if names else "",
            "to": _join(names[1:]),
            "valves_fittings": _join(node_name(G, n) for n in inline),
            "runs": len(ids),
            "length_pt": round(sum(pipe_by_id[p].length for p in ids), 1),
            "pipe_ids": " ".join(ids),
            "page": pipe_by_id[ids[0]].page,
        })
    rows.sort(key=lambda r: (r["line_number"] == "", r["line_number"], r["pipe_ids"]))
    return rows


def _order_terminals(G: nx.MultiGraph, terminals: list[str], paths: Sequence[FlowPath]) -> list[str]:
    tset = set(terminals)
    for fp in paths:
        hits = [n for n in fp.node_ids if n in tset]
        if len(hits) >= 2:
            first = hits[0]
            rest = sorted((t for t in terminals if t != first), key=lambda n: _xkey(G, n))
            return [first] + rest
    return sorted(terminals, key=lambda n: _xkey(G, n))


def _xkey(G: nx.MultiGraph, n: str) -> tuple[float, float]:
    p = G.nodes[n].get("pos")
    return (p[0], p[1]) if p else (float("inf"), float("inf"))


def _terminal_name(G: nx.MultiGraph, n: str, members: set[str]) -> str:
    kind = G.nodes[n].get("kind")
    if kind == "object":
        return node_name(G, n)
    if kind == "free":
        return "open end"
    others = [G.edges[n, lk.other, lk.pipe_id].get("line_number") for lk in links(G, n, ALL_KINDS)
              if lk.pipe_id not in members and lk.kind == PipeKind.PROCESS]
    others = [o for o in others if o]
    return f"tee {n} on {others[0]}" if others else f"tee {n}"
