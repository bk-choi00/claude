"""Randomised blueprints of ship piping systems for the synthetic P&ID generator.

A blueprint turns ``(system, rng)`` into a :class:`SystemTopology`: an abstract,
layout-free description of one ship system drawn left to right as *stages*:

* stage 0 -- sources (tanks, sea chests, bilge wells, compressors, off-page connectors),
* middle stages -- parallel *trains* (rows) such as pump trains (suction valve,
  strainer, gauges, pump, check valve, discharge valve) or process units
  (filters, heaters, coolers, purifiers, BWTS, OWS, flow meters, three-way TCVs),
* last stage -- destinations (tanks, overboard, engines, off-page connectors).

Neighbouring stages are joined by vertical headers (one per ``Train.group``).
Decorations that are not plain inline items are listed separately: bypasses,
relief valves across screw pumps, heat-exchanger shell / plate-cooler secondary
connections, instrument attachments on equipment and return lines to tanks.

Naming follows marine practice (``NO.1 W.B.TK (P)``, ``NO.2 BALLAST PUMP``,
``H.F.O. SETT. TK``, ``M/E``), valve actuators follow the usual conventions
(hydraulic butterfly valves on ballast, quick-closing valves on FO/LO tank
outlets, SDNR valves on bilge suctions, relief valves across screw pumps, check
valves on pump discharges, three-way TCVs on cooling circuits), instruments use
ISA 5.1 letters and line numbers use one of several yard styles.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from ..catalog import SYSTEMS
from ..model import ActuatorType, Category, Medium, SymbolClass, SystemType
from ..symbols import category_of

S = SymbolClass
A = ActuatorType

# --------------------------------------------------------------------------- topology types


@dataclass
class Item:
    """One drawn object (equipment, valve, instrument, fitting or connector)."""

    key: str
    cls: SymbolClass
    name: tuple[str, ...] = ()  # label lines (equipment name, capacity) -- TextRole.LABEL
    tag: Optional[str] = None  # tag text -- TextRole.TAG (instruments: "PI-101")
    actuator: ActuatorType = ActuatorType.NONE
    in_port: Optional[str] = None  # main-flow inlet (None: "in" / a boundary port)
    out_port: Optional[str] = None  # main-flow outlet (None: "out" / a boundary port)
    size: Optional[tuple[float, float]] = None  # (width, height) in symbol units for stretchable equipment
    inside: tuple[str, ...] = ()  # text drawn inside (instrument letters + number, connector ref)
    name_inside: bool = False  # draw ``name`` inside the outline (tanks, engines)
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> Category:
        return category_of(self.cls)


@dataclass
class Tap:
    """Instrument connected to the pipe at this point by a short tap line."""

    instr: Item
    side: int = -1  # -1: above the pipe (local -v), +1: below
    signal_to: Optional[str] = None  # key of a control valve receiving a dashed signal line


@dataclass
class Branch:
    """A tee position on the train referenced by a bypass / relief line."""

    key: str


Element = Union[Item, Tap, Branch]


@dataclass
class Train:
    key: str
    elements: list[Element]
    source: Optional[Item] = None  # None: the train starts at the header on its left
    sink: Optional[Item] = None  # None: the train ends at the header on its right (or a return line)
    group: int = 0  # header group linking it to the neighbouring stages
    template: str = ""  # trains sharing a template get identical spacing (aligned columns)
    size_class: str = "main"
    end_return: Optional[str] = None  # open end routed back to this tank (see ReturnLine)


@dataclass
class Stage:
    trains: list[Train]
    fanout: bool = False  # destinations may be drawn as vertical branches off a horizontal main


@dataclass
class Bypass:
    start: str  # Branch key
    end: str  # Branch key, or item key when ``end_port`` is set
    valve: Item
    end_port: Optional[str] = None  # e.g. the "branch" port of a three-way TCV
    side: int = 1  # +1 below the train, -1 above


@dataclass
class Relief:
    inlet: str  # Branch key on the pump discharge side
    outlet: str  # Branch key on the pump suction side
    valve: Item  # SAFETY_VALVE


@dataclass
class SideLink:
    """Secondary connection of a multi-port unit (cooler shell side) to an off-page connector."""

    item: str
    port: str
    connector: Item
    outgoing: bool  # flow from the unit to the connector
    medium: Medium
    code: str  # system code used in the line number
    size_class: str = "branch"


@dataclass
class Attachment:
    """Object hung off boundary-port equipment (level transmitter on a tank, safety valve on a receiver)."""

    item: str
    obj: Item
    side: str = "outer"  # "outer" (away from the pipe) | "top"
    kind: str = "tap"  # "tap" (instrument tap line) | "process"


@dataclass
class ReturnLine:
    """Line back to a source tank (usually crossing other lines on its way).

    It starts either at the open end of ``train`` or at ``port`` of ``item``
    (e.g. the branch port of a three-way valve).
    """

    target: str  # tank item key (stage 0 source)
    train: Optional[str] = None
    item: Optional[str] = None
    port: Optional[str] = None
    valve: Optional[Item] = None


@dataclass
class Drain:
    """Drain / overflow line dropped from the top train of ``stage`` to an off-page connector below
    the stage (it crosses the trains underneath). Optional: skipped if no free route exists."""

    stage: "Stage"
    connector: Item
    valve: Optional[Item] = None


@dataclass
class ShipIdentity:
    hull_no: str
    ship_type: str
    yard: str
    owner: str
    project: str


@dataclass
class SystemTopology:
    system: SystemType
    code: str
    medium: Medium
    title: tuple[str, ...]  # drawing title lines (contain catalogue keywords of the system)
    stages: list[Stage] = field(default_factory=list)
    bypasses: list[Bypass] = field(default_factory=list)
    reliefs: list[Relief] = field(default_factory=list)
    side_links: list[SideLink] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)
    returns: list[ReturnLine] = field(default_factory=list)
    drains: list[Drain] = field(default_factory=list)
    sizes: dict[str, int] = field(default_factory=dict)  # size class -> nominal size (mm, 'A')
    notes: list[str] = field(default_factory=list)
    namer: Optional["Namer"] = None
    ship: Optional[ShipIdentity] = None

    def items(self) -> list[Item]:
        """Every item of the topology (trains, decorations), in drawing order."""
        out: list[Item] = []
        for st in self.stages:
            for tr in st.trains:
                if tr.source:
                    out.append(tr.source)
                for el in tr.elements:
                    if isinstance(el, Item):
                        out.append(el)
                    elif isinstance(el, Tap):
                        out.append(el.instr)
                if tr.sink:
                    out.append(tr.sink)
        out += [b.valve for b in self.bypasses] + [r.valve for r in self.reliefs]
        out += [s.connector for s in self.side_links] + [a.obj for a in self.attachments]
        out += [r.valve for r in self.returns if r.valve]
        for d in self.drains:
            out += [d.connector] + ([d.valve] if d.valve else [])
        return out

    def train_count(self) -> int:
        return sum(len(s.trains) for s in self.stages)


# --------------------------------------------------------------------------- naming

VALVE_TAG_STYLES = ("{c}-V{n:03d}", "V-{c}-{n:03d}", "{c}{n:03d}V")
LINE_NO_STYLES = ("{size}A-{c}-{n:03d}-{spec}", "{c}-{n:03d}-{size}A", "{size}-{c}-{n:03d}")
PIPE_SPECS = ("1S", "2S", "3S", "1A", "2B", "B1", "C2", "S1", "10", "11")
EQUIP_KIND = {
    S.PUMP_CENTRIFUGAL: "P", S.PUMP_SCREW: "P", S.TANK: "T", S.FILTER: "F", S.DUPLEX_FILTER: "F",
    S.STRAINER: "F", S.HEATER: "H", S.HEAT_EXCHANGER: "E", S.PLATE_HEAT_EXCHANGER: "E", S.PURIFIER: "S",
    S.COMPRESSOR: "K", S.AIR_RECEIVER: "R", S.OILY_WATER_SEPARATOR: "S", S.BWTS_UNIT: "U", S.FLOW_METER: "M",
}


class Namer:
    """Per-drawing allocation of valve / instrument / equipment tags and line numbers."""

    def __init__(self, code: str, rng: random.Random):
        self.code = code
        self.valve_style = rng.choice(VALVE_TAG_STYLES)
        self.line_style = rng.choice(LINE_NO_STYLES)
        self.spec = rng.choice(PIPE_SPECS)
        self.equipment_tags = rng.random() < 0.35
        self.inst_sep = rng.choice(["-", "-", " ", ""])
        self._valve = rng.choice([1, 1, 11, 21, 101])
        self._line = rng.choice([1, 1, 11, 101])
        base = rng.choice([1, 2, 3, 5, 6, 7, 8, 9]) * 100
        self._inst: dict[str, int] = {}
        self._inst_base = base + rng.choice([0, 0, 10, 20])
        self._equip: dict[str, int] = {}

    def valve(self) -> str:
        n, self._valve = self._valve, self._valve + 1
        return self.valve_style.format(c=self.code, n=n)

    def instrument(self, letters: str) -> tuple[str, str]:
        """(letters, loop number); loops are numbered per first letter (P, T, L, F, ...)."""
        first = letters[0]
        n = self._inst.get(first, self._inst_base)
        self._inst[first] = n + 1
        return letters, f"{n:03d}"

    def instrument_tag(self, letters: str, number: str) -> str:
        return f"{letters}-{number}"

    def equipment(self, cls: SymbolClass) -> Optional[str]:
        kind = EQUIP_KIND.get(cls)
        if not kind or not self.equipment_tags:
            return None
        n = self._equip.get(kind, 1)
        self._equip[kind] = n + 1
        return f"{self.code}-{kind}{n:02d}"

    def line_number(self, size_mm: int, code: Optional[str] = None) -> str:
        n, self._line = self._line, self._line + 1
        return self.line_style.format(size=size_mm, c=code or self.code, n=n, spec=self.spec)


# --------------------------------------------------------------------------- ship identities

SHIP_TYPES = (
    "300K VLCC", "319K VLCC", "158K SUEZMAX TANKER", "115K AFRAMAX TANKER", "50K PRODUCT/CHEMICAL TANKER",
    "24,000 TEU CONTAINER CARRIER", "15,000 TEU CONTAINER CARRIER", "8,200 TEU CONTAINER CARRIER",
    "174K LNG CARRIER", "180K LNG CARRIER", "91K VLGC", "180K BULK CARRIER", "82K KAMSARMAX BULK CARRIER",
    "7,000 CEU PCTC",
)
YARDS = (
    "AURORA BAY SHIPBUILDING CO., LTD.", "SEAFORGE HEAVY INDUSTRIES CO., LTD.", "MERIDIAN MARINE ENGINEERING",
    "CORALINE SHIPYARD CO., LTD.", "HALCYON OFFSHORE & SHIPBUILDING", "TIDEMARK MARINE WORKS",
)
OWNERS = (
    "OCEANIC CARRIERS S.A.", "BLUE HORIZON SHIPPING LTD.", "ATLAS TANKERS INC.", "POLARIS MARITIME CORP.",
    "NORDWIND REEDEREI GMBH", "CRESCENT LINES PTE. LTD.",
)


def ship_identity(rng: random.Random) -> ShipIdentity:
    hull = rng.choice(["H.{n}", "H.{n}", "S{n}", "HN {n}"]).format(n=rng.randint(1100, 9899))
    st = rng.choice(SHIP_TYPES)
    return ShipIdentity(hull_no=hull, ship_type=st, yard=rng.choice(YARDS), owner=rng.choice(OWNERS),
                        project=f"{st} ({hull})")


# --------------------------------------------------------------------------- builder


class _Builder:
    """Helpers shared by the system blueprints."""

    def __init__(self, system: SystemType, rng: random.Random, complexity: float):
        self.rng = rng
        self.c = complexity
        info = SYSTEMS[system]
        self.code = info.code
        self.namer = Namer(info.code, rng)
        self.topo = SystemTopology(system=system, code=info.code, medium=info.medium, title=(), namer=self.namer)
        self._n = 0
        self.tap_cls = rng.choice([S.INSTRUMENT_LOCAL, S.INSTRUMENT_LOCAL, S.INSTRUMENT_PANEL])
        self.remote_cls = rng.choice([S.INSTRUMENT_SHARED, S.INSTRUMENT_SHARED, S.INSTRUMENT_PANEL])

    # -- counts ----------------------------------------------------------------
    def count(self, lo: int, hi: int) -> int:
        """Random count in [lo, hi] biased towards the upper end (real installations are mostly
        duplicated / triplicated); the upper bound shrinks with the complexity factor."""
        hi = max(lo, int(round(lo + (hi - lo) * self.c)))
        values = list(range(lo, hi + 1))
        return self.rng.choices(values, weights=[1 + 2 * i for i in range(len(values))])[0]

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def key(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    # -- items -----------------------------------------------------------------
    def valve(self, cls: SymbolClass = S.GATE_VALVE, actuator: ActuatorType = A.HANDWHEEL, tagged: bool = True,
              **attrs: Any) -> Item:
        return Item(self.key("v"), cls, tag=self.namer.valve() if tagged else None, actuator=actuator,
                    attrs=dict(attrs))

    def check(self, cls: SymbolClass = S.CHECK_VALVE, tagged: Optional[bool] = None) -> Item:
        tagged = self.chance(0.6) if tagged is None else tagged
        act = A.HANDWHEEL if cls == S.SDNR_VALVE else A.NONE
        return self.valve(cls, act, tagged)

    def equip(self, cls: SymbolClass, *name: str, size: Optional[tuple[float, float]] = None,
              inside: bool = False, **attrs: Any) -> Item:
        return Item(self.key("e"), cls, name=tuple(n for n in name if n), tag=self.namer.equipment(cls),
                    size=size, name_inside=inside, attrs=dict(attrs))

    def tank(self, name: str, capacity: Optional[str] = None, size: Optional[tuple[float, float]] = None) -> Item:
        if size is None:
            size = (self.rng.uniform(2.4, 3.6), self.rng.uniform(1.4, 2.0))
        lines = (name,) + ((capacity,) if capacity and self.chance(0.5) else ())
        return self.equip(S.TANK, *lines, size=size, inside=True)

    def connector(self, label: str, ref: Optional[str] = None) -> Item:
        ref = ref or self.rng.choice(["{a}-{n}", "{a}{n}", "S.{n}"]).format(
            a=self.rng.choice("ABCDEFGHJK"), n=self.rng.randint(1, 12))
        return Item(self.key("c"), S.OFF_PAGE_CONNECTOR, name=(label,), inside=(ref,))

    def instr(self, letters: str, cls: Optional[SymbolClass] = None, service: str = "") -> Item:
        cls = cls or (self.remote_cls if letters[-1] in "TCA" and len(letters) > 1 and letters[1] != "I"
                      else self.tap_cls)
        letters, number = self.namer.instrument(letters)
        return Item(self.key("i"), cls, tag=self.namer.instrument_tag(letters, number), inside=(letters, number),
                    attrs={"letters": letters, "loop": number, "service": service})

    def tap(self, letters: str, side: int = -1, cls: Optional[SymbolClass] = None, signal_to: Optional[str] = None,
            service: str = "") -> Tap:
        return Tap(self.instr(letters, cls, service), side=side, signal_to=signal_to)

    def branch(self) -> Branch:
        return Branch(self.key("b"))

    def safety(self) -> Item:
        return self.valve(S.SAFETY_VALVE, A.NONE, tagged=self.chance(0.7))

    # -- structure -------------------------------------------------------------
    def stage(self, fanout: bool = False) -> Stage:
        st = Stage([], fanout=fanout)
        self.topo.stages.append(st)
        return st

    def train(self, stage: Stage, elements: list[Element], source: Optional[Item] = None, sink: Optional[Item] = None,
              group: int = 0, template: str = "", size_class: str = "main") -> Train:
        tr = Train(self.key("t"), list(elements), source, sink, group, template or self.key("tpl"), size_class)
        stage.trains.append(tr)
        return tr

    def pump_elements(self, pump: Item, suction: tuple[SymbolClass, ActuatorType],
                      discharge: tuple[SymbolClass, ActuatorType], *, strainer: bool, check: bool,
                      gauges: bool, transmitter: bool, relief: bool, gauge_side: int,
                      check_cls: SymbolClass = S.CHECK_VALVE, flex: bool = False,
                      reducer: bool = False) -> list[Element]:
        """Suction valve, [strainer], [PI], [flexible joint], [relief tee], pump, [relief tee], [PI], [PT],
        [reducer], [check], discharge valve."""
        el: list[Element] = [self.valve(*suction)]
        if strainer:
            el.append(self.equip(S.STRAINER))
        if gauges:
            el.append(self.tap("PI", side=gauge_side, service="pump suction"))
        if flex:
            el.append(Item(self.key("f"), S.FLEXIBLE_HOSE))
        b_in = b_out = None
        if relief:
            b_in = self.branch()
            el.append(b_in)
        el.append(pump)
        if relief:
            b_out = self.branch()
            el.append(b_out)
        if gauges:
            el.append(self.tap("PI", side=gauge_side, service="pump discharge"))
        if transmitter:
            el.append(self.tap("PT", side=gauge_side, cls=self.remote_cls, service="pump discharge"))
        if reducer:
            el.append(Item(self.key("f"), S.REDUCER))
        if check:
            el.append(self.check(check_cls))
        el.append(self.valve(*discharge))
        if relief and b_in and b_out:
            self.topo.reliefs.append(Relief(inlet=b_out.key, outlet=b_in.key, valve=self.safety()))
        return el

    def pump_stage(self, n: int, name_fmt: str, cls: SymbolClass, capacity: str,
                   suction: tuple[SymbolClass, ActuatorType], discharge: tuple[SymbolClass, ActuatorType],
                   *, strainer: Optional[bool] = None, check: Optional[bool] = True, relief: Optional[bool] = None,
                   group: int = 0, stage: Optional[Stage] = None, check_cls: SymbolClass = S.CHECK_VALVE,
                   size_class: str = "main") -> Stage:
        """``n`` identical parallel pump trains (aligned)."""
        st = stage or self.stage()
        strainer = self.chance(0.4) if strainer is None else strainer
        check = self.chance(0.75) if check is None else check
        relief = (cls == S.PUMP_SCREW and self.chance(0.85)) if relief is None else relief
        gauges = self.chance(0.9)
        transmitter = self.chance(0.4)
        flex = self.chance(0.3)
        reducer = self.chance(0.3)
        gauge_side = 1 if relief else self.rng.choice([-1, -1, 1])
        tpl = self.key("pumps")
        for i in range(n):
            name = name_fmt.format(n=i + 1, s=("(P)", "(S)", "(C)")[i % 3])
            pump = self.equip(cls, name, capacity if self.chance(0.8) else "")
            self.train(st, self.pump_elements(pump, suction, discharge, strainer=strainer, check=check,
                                              gauges=gauges, transmitter=transmitter, relief=relief,
                                              gauge_side=gauge_side, check_cls=check_cls, flex=flex,
                                              reducer=reducer),
                       group=group, template=tpl, size_class=size_class)
        return st

    def sinks(self, stage: Stage, items: list[tuple[Item, list[Element]]], template: str = "",
              size_class: str = "branch", group: int = 0) -> None:
        tpl = template or self.key("snk")
        for sink, el in items:
            self.train(stage, el, sink=sink, template=tpl, size_class=size_class, group=group)

    def cooler_with_tcv(self, cooler: Item, *, cooling_code: str, cooling_medium: Medium, side_ports: tuple[str, str],
                        tcv_service: str, tcv_actuator: ActuatorType = A.DIAPHRAGM,
                        bypass_side: int = 1) -> list[Element]:
        """[bypass tee], cooler, 3-way TCV (bypass into its branch), TIC tap with signal to the TCV."""
        b0 = self.branch()
        tcv = self.valve(S.THREE_WAY_VALVE, tcv_actuator, tagged=False)
        tcv.tag = None
        letters, number = self.namer.instrument("TCV")
        tcv.tag = self.namer.instrument_tag(letters, number)
        tcv.attrs["service"] = tcv_service
        tcv.attrs["branch_side"] = bypass_side
        self.topo.bypasses.append(Bypass(start=b0.key, end=tcv.key, end_port="branch", side=bypass_side,
                                         valve=self.valve(S.GLOBE_VALVE, A.HANDWHEEL)))
        self.side_pair(cooler, side_ports, cooling_code, cooling_medium)
        tic = self.tap("TIC", side=-1, cls=self.remote_cls, signal_to=tcv.key, service=tcv_service)
        return [b0, cooler, tcv, tic]

    def side_pair(self, unit: Item, ports: tuple[str, str], code: str, medium: Medium) -> None:
        name = {"CF": "C.F.W.", "SW": "C.S.W.", "ST": "STEAM", "FW": "F.W."}.get(code, code)
        self.topo.side_links.append(SideLink(unit.key, ports[0], self.connector(f"FROM {name} SYSTEM"), False,
                                             medium, code))
        self.topo.side_links.append(SideLink(unit.key, ports[1], self.connector(f"TO {name} SYSTEM"), True,
                                             medium, code))

    def return_from(self, train: Train, tank: Item, valve: bool = False) -> None:
        train.end_return = tank.key
        self.topo.returns.append(ReturnLine(target=tank.key, train=train.key,
                                            valve=self.valve(S.GLOBE_VALVE, A.HANDWHEEL) if valve else None))

    def spectacle(self) -> Item:
        return Item(self.key("f"), S.SPECTACLE_FLANGE)

    def drain(self, stage: Stage, labels: tuple[str, ...], p: float = 0.65) -> None:
        """Maybe add a drain line from the top train of ``stage`` (needs >= 2 rows to cross)."""
        if len(stage.trains) >= 2 and self.chance(p):
            valve = self.valve(self.rng.choice([S.GLOBE_VALVE, S.GATE_VALVE, S.BALL_VALVE]), A.HANDWHEEL,
                               tagged=self.chance(0.7))
            self.topo.drains.append(Drain(stage, self.connector(self.rng.choice(labels)), valve))

    def level_gauge(self, tank: Item, letters: str = "LT") -> None:
        if self.chance(0.6):
            self.topo.attachments.append(Attachment(tank.key, self.instr(letters, self.remote_cls, "tank level")))


def _side(i: int) -> str:
    return ("(P)", "(S)")[i % 2]


def _cap(rng: random.Random, values: tuple, unit: str) -> str:
    q = rng.choice(values)
    return f"{q:,} {unit}" if isinstance(q, int) else f"{q} {unit}"


# --------------------------------------------------------------------------- system blueprints


def _ballast(b: _Builder) -> None:
    r = b.rng
    b.topo.title = r.choice([("BALLAST WATER SYSTEM",), ("BALLAST SYSTEM", "IN ENGINE ROOM"),
                             ("BALLAST WATER TREATMENT", "& BALLAST SYSTEM")])
    b.topo.sizes = {"main": r.choice([300, 350, 400, 450, 500]), "branch": r.choice([250, 300, 350]), "small": 50}
    hyd = (S.BUTTERFLY_VALVE, A.HYDRAULIC)
    s0 = b.stage()
    chests = ["HIGH SEA CHEST", "LOW SEA CHEST"][: b.count(1, 2)]
    for name in chests:
        el: list[Element] = [b.valve(*hyd)]
        if b.chance(0.4):
            el.append(b.equip(S.STRAINER, "SEA W. STRAINER" if b.chance(0.4) else ""))
        b.train(s0, el, source=b.equip(S.SEA_CHEST, name, size=(r.uniform(1.6, 2.4), r.uniform(1.0, 1.4))),
                template="chest")
    if b.chance(0.35):
        b.train(s0, [b.valve(*hyd)], source=b.connector("FROM DEBALLAST MAIN"))
    n = b.count(1, 3)
    st = b.pump_stage(n, "NO.{n} BALLAST PUMP", S.PUMP_CENTRIFUGAL,
                      _cap(r, (1500, 2000, 2500, 3000, 3500), "M3/H x 30 MTH"), hyd, hyd, strainer=False, check=None)
    b.drain(st, ("TO BILGE WELL", "TO EDUCTOR SUCTION", "DRAIN TO BILGE"))
    s2 = b.stage()
    for k in range(2 if n >= 2 and b.chance(0.3) else 1):
        b0, b1 = b.branch(), b.branch()
        el = [b0, b.equip(S.DUPLEX_FILTER, f"NO.{k + 1} BWTS FILTER" if k or b.chance(0.5) else "BWTS FILTER"),
              b.equip(S.BWTS_UNIT, "BWTS UV REACTOR" if b.chance(0.6) else "B.W.T. UNIT",
                      _cap(r, (500, 750, 1000, 1500), "M3/H"), size=(r.uniform(1.8, 2.6), r.uniform(0.8, 1.1))),
              b1]
        if b.chance(0.6):
            el.append(b.tap("QT", side=-1, cls=b.remote_cls, service="UV intensity"))
        if b.chance(0.7):
            el.append(b.equip(S.FLOW_METER))
            el.append(b.tap("FT", side=-1, cls=b.remote_cls, service="ballast flow"))
        el.append(b.valve(*hyd))
        b.topo.bypasses.append(Bypass(b0.key, b1.key, b.valve(*hyd), side=1))
        b.train(s2, el, template="bwts")
    last = b.stage(fanout=True)
    tanks = []
    names = [f"NO.{i // 2 + 1} W.B.TK {_side(i)}" for i in range(8)]
    if b.chance(0.4):
        names = ["F.P.TK"] + names
    for i in range(b.count(3, 6)):
        t = b.tank(names[i], _cap(r, (2400, 3100, 4500, 6200), "M3"))
        b.level_gauge(t)
        tanks.append((t, [b.valve(*hyd)]))
    b.sinks(last, tanks, template="wbt")
    ob = [b.valve(*hyd)] + ([b.check(S.SDNR_VALVE)] if b.chance(0.5) else [])
    b.train(last, ob, sink=b.equip(S.OVERBOARD, "OVERBOARD"), size_class="branch")
    if b.chance(0.4):
        b.train(last, [b.valve(*hyd)], sink=b.connector(r.choice(["TO A.P.TK", "TO NO.5 W.B.TK (P&S)", "TO TOP SIDE TK"])),
                size_class="branch")
    b.topo.notes += [
        "BALLAST WATER TREATMENT SYSTEM TO COMPLY WITH IMO BWM CONVENTION REG. D-2.",
        "ALL BALLAST VALVES ARE HYDRAULICALLY REMOTE OPERATED FROM BALLAST CONTROL CONSOLE.",
    ]


def _bilge(b: _Builder) -> None:
    r = b.rng
    b.topo.title = r.choice([("BILGE SYSTEM",), ("BILGE & OILY WATER", "SEPARATOR SYSTEM"), ("E/R BILGE SYSTEM",)])
    b.topo.sizes = {"main": r.choice([65, 80, 100]), "branch": r.choice([50, 65]), "small": 25}
    s0 = b.stage()
    wells = ["BILGE WELL (P)", "BILGE WELL (S)", "AFT BILGE WELL", "E/R FWD BILGE WELL", "COFFERDAM"]
    nw = b.count(2, 4)
    holding_pos = r.randint(0, nw)
    holding = None
    for i in range(nw + 1):
        if i == holding_pos:
            holding = b.tank("BILGE HOLDING TK", _cap(r, (20, 30, 45), "M3"))
            b.level_gauge(holding)
            b.train(s0, [b.valve(S.GATE_VALVE, A.HANDWHEEL)], source=holding, group=1)
            continue
        well = b.tank(wells[i - (i > holding_pos)], size=(r.uniform(1.4, 2.0), r.uniform(1.0, 1.4)))
        b.train(s0, [b.equip(S.STRAINER, "MUD BOX" if b.chance(0.3) else ""), b.check(S.SDNR_VALVE, tagged=True)],
                source=well, template="well", group=0)
    s0.trains.sort(key=lambda t: t.group)  # header groups must be vertically contiguous
    s1 = b.stage()
    if b.chance(0.6):
        name, cls, cap = "NO.{n} BILGE PUMP", r.choice([S.PUMP_CENTRIFUGAL, S.PUMP_SCREW]), \
            _cap(r, (10, 15, 20, 25), "M3/H x 30 MTH")
    else:  # the large general service pumps double as bilge pumps
        name, cls, cap = "NO.{n} BILGE, FIRE & G/S PUMP", S.PUMP_CENTRIFUGAL, _cap(r, (150, 200, 250), "M3/H x 35 MTH")
    b.pump_stage(b.count(1, 2), name, cls, cap, (S.GLOBE_VALVE, A.HANDWHEEL), (S.GLOBE_VALVE, A.HANDWHEEL),
                 strainer=False, stage=s1, group=0)
    feed = b.equip(S.PUMP_SCREW, "OWS FEED PUMP" if b.chance(0.5) else "BILGE SEPARATOR PUMP",
                   _cap(r, (2.5, 5.0), "M3/H x 3.5 BAR"))
    els = b.pump_elements(feed, (S.GATE_VALVE, A.HANDWHEEL), (S.GATE_VALVE, A.HANDWHEEL), strainer=True,
                          check=True, gauges=True, transmitter=False, relief=b.chance(0.7), gauge_side=1)
    b.train(s1, els, group=1, size_class="branch")
    b.drain(s1, ("TO SLUDGE TK", "TO OILY BILGE TK", "TO SLUDGE PUMP"))
    s2 = b.stage()
    b.train(s2, [b.check(S.SDNR_VALVE, tagged=True)], sink=b.equip(S.OVERBOARD, "OVERBOARD"), group=0)
    if b.chance(0.6):
        b.train(s2, [b.valve(S.GLOBE_VALVE, A.HANDWHEEL)] + ([b.spectacle()] if b.chance(0.5) else []),
                sink=b.connector("TO SHORE CONN. (P&S)"), group=0, size_class="branch")
    ows = b.equip(S.OILY_WATER_SEPARATOR, "OILY WATER SEPARATOR", _cap(r, (2.5, 5.0), "M3/H"),
                  size=(r.uniform(1.6, 2.2), r.uniform(1.2, 1.6)), inside=b.chance(0.4))
    three = b.valve(S.THREE_WAY_VALVE, A.ELECTRIC, tagged=True)
    three.attrs["branch_side"] = 1
    els = [ows, b.tap("QIA", side=-1, cls=b.remote_cls, service="15 ppm bilge alarm"), three]
    if b.chance(0.5):
        els.append(b.equip(S.FLOW_METER))
    els.append(b.check(S.SDNR_VALVE, tagged=True))
    b.train(s2, els, sink=b.equip(S.OVERBOARD, "OVERBOARD (OWS)" if b.chance(0.5) else "OVERBOARD"), group=1,
            size_class="small")
    if holding is not None:  # OCM alarm: the three-way valve recirculates to the holding tank
        b.topo.returns.append(ReturnLine(target=holding.key, item=three.key, port="branch",
                                         valve=b.valve(S.GLOBE_VALVE, A.HANDWHEEL)))
    b.topo.notes += [
        "OILY WATER SEPARATOR DISCHARGE TO BE MONITORED BY 15 PPM BILGE ALARM (MARPOL ANNEX I).",
        "BILGE SUCTIONS TO BE FITTED WITH SCREW-DOWN NON-RETURN VALVES AND MUD BOXES.",
    ]


def _fo_transfer(b: _Builder) -> None:
    r = b.rng
    b.topo.title = r.choice([("FUEL OIL TRANSFER SYSTEM",), ("F.O. TRANSFER SYSTEM",),
                             ("FUEL OIL TRANSFER", "& PURIFYING SYSTEM")])
    b.topo.sizes = {"main": r.choice([80, 100, 125, 150]), "branch": r.choice([65, 80, 100]), "small": 25}
    qc = (r.choice([S.GLOBE_VALVE, S.GATE_VALVE]), A.QUICK_CLOSING)
    s0 = b.stage()
    names = ["NO.1 H.F.O. STOR. TK (P)", "NO.1 H.F.O. STOR. TK (S)", "NO.2 H.F.O. STOR. TK (P)",
             "NO.2 H.F.O. STOR. TK (S)", "M.D.O. STOR. TK", "L.S.H.F.O. STOR. TK"]
    tanks = []
    for i in range(b.count(2, 5)):
        t = b.tank(names[i], _cap(r, (1200, 1850, 2600, 3400), "M3"))
        b.level_gauge(t, "LT")
        tanks.append(t)
        b.train(s0, [b.valve(*qc)], source=t, template="stor")
    if b.chance(0.4):
        b.train(s0, [b.valve(S.GATE_VALVE, A.HANDWHEEL)] + ([b.spectacle()] if b.chance(0.6) else []),
                source=b.connector("FROM BUNKER STATION (P&S)"))
    st = b.pump_stage(b.count(1, 2), "NO.{n} H.F.O. TRANSFER PUMP", S.PUMP_SCREW,
                      _cap(r, (50, 80, 100, 150), "M3/H x 4.0 BAR"), (S.GATE_VALVE, A.HANDWHEEL),
                      (S.GLOBE_VALVE, A.HANDWHEEL), strainer=True)
    b.drain(st, ("TO F.O. OVERFLOW TK", "TO SLUDGE TK", "TO F.O. DRAIN TK"))
    if b.chance(0.5):
        mid = b.stage()
        el: list[Element] = [b.equip(S.FLOW_METER, "F.O. FLOW METER" if b.chance(0.4) else ""),
                             b.tap("FQI", side=-1, service="transfer flow")]
        b.train(mid, el)
    last = b.stage(fanout=True)
    snames = ["NO.1 H.F.O. SETT. TK", "NO.2 H.F.O. SETT. TK", "M.D.O. SERV. TK", "L.S.H.F.O. SETT. TK"]
    items = []
    for i in range(b.count(1, 3)):
        t = b.tank(snames[i], _cap(r, (80, 120, 150), "M3"))
        b.level_gauge(t, r.choice(["LT", "LIA"]))
        items.append((t, [b.valve(S.GLOBE_VALVE, A.HANDWHEEL)]))
    b.sinks(last, items, template="sett")
    if b.chance(0.6):
        b.train(last, [b.valve(S.GLOBE_VALVE, A.HANDWHEEL)], sink=b.connector("TO F.O. PURIFIERS"),
                size_class="branch")
    if b.chance(0.4):
        b.train(last, [b.valve(S.GATE_VALVE, A.HANDWHEEL), b.spectacle()], sink=b.connector("TO BUNKER STATION"),
                size_class="branch")
    if len(tanks) >= 2 and b.chance(0.6):  # recirculation back to a (lower) storage tank
        tr = b.train(last, [b.valve(S.GLOBE_VALVE, A.HANDWHEEL)], size_class="branch")
        last.trains.insert(0, last.trains.pop())
        b.return_from(tr, tanks[r.randint(1, len(tanks) - 1)])
    b.topo.notes += [
        "QUICK-CLOSING VALVES TO BE REMOTELY OPERATED FROM FIRE CONTROL STATION (SOLAS II-2/4.2.2.3.4).",
        "ALL F.O. TANKS TO BE FITTED WITH HIGH LEVEL ALARMS.",
    ]


def _fo_supply(b: _Builder) -> None:
    r = b.rng
    b.topo.title = r.choice([("FUEL OIL SUPPLY SYSTEM",), ("M/E FUEL OIL SUPPLY", "& BOOSTER SYSTEM"),
                             ("F.O. SUPPLY SYSTEM",)])
    b.topo.sizes = {"main": r.choice([40, 50, 65]), "branch": r.choice([32, 40, 50]), "small": 15}
    s0 = b.stage()
    for name in ["H.F.O. SERV. TK", "M.D.O. SERV. TK", "L.S.H.F.O. SERV. TK"][: b.count(1, 3)]:
        t = b.tank(name, _cap(r, (60, 80, 120), "M3"))
        b.train(s0, [b.valve(r.choice([S.GLOBE_VALVE, S.GATE_VALVE]), A.QUICK_CLOSING)], source=t, template="serv")
    b.pump_stage(b.count(1, 2), "NO.{n} F.O. SUPPLY PUMP", S.PUMP_SCREW, _cap(r, (5, 8, 10, 14), "M3/H x 5.0 BAR"),
                 (S.GATE_VALVE, A.HANDWHEEL), (S.GATE_VALVE, A.HANDWHEEL), strainer=False)
    if b.chance(0.7):
        st = b.stage()
        b0, b1 = b.branch(), b.branch()
        b.topo.bypasses.append(Bypass(b0.key, b1.key, b.valve(S.GATE_VALVE, A.HANDWHEEL), side=1))
        b.train(st, [b0, b.equip(S.FLOW_METER, "F.O. FLOW METER" if b.chance(0.5) else ""), b1,
                     b.tap("FQI", side=-1, service="F.O. consumption")])
    st = b.pump_stage(b.count(1, 2), "NO.{n} F.O. CIRC. PUMP", S.PUMP_SCREW, _cap(r, (10, 14, 18), "M3/H x 10 BAR"),
                      (S.GATE_VALVE, A.HANDWHEEL), (S.GATE_VALVE, A.HANDWHEEL), strainer=False)
    b.drain(st, ("TO F.O. DRAIN TK", "TO SLUDGE TK", "TO F.O. OVERFLOW TK"))
    st = b.stage()
    nh = b.count(1, 2)
    for k in range(nh):
        b.train(st, [b.valve(S.GLOBE_VALVE, A.HANDWHEEL),
                     b.equip(S.HEATER, f"NO.{k + 1} F.O. HEATER" if nh > 1 else "F.O. HEATER"),
                     b.tap("TI", side=-1, service="heater outlet"),
                     b.valve(S.GLOBE_VALVE, A.HANDWHEEL)], template="heater")
    st = b.stage()
    b0, b1 = b.branch(), b.branch()
    el: list[Element] = [b.tap("VIC", side=-1, cls=b.remote_cls, service="viscosity control"), b0,
                         b.equip(S.DUPLEX_FILTER, "F.O. AUTO. FILTER" if b.chance(0.5) else "F.O. FILTER"), b1,
                         b.tap("PI", side=-1, service="engine inlet")]
    if b.chance(0.5):
        el.append(b.tap("TI", side=-1, service="engine inlet"))
    b.topo.bypasses.append(Bypass(b0.key, b1.key, b.valve(S.GATE_VALVE, A.HANDWHEEL), side=1))
    b.train(st, el)
    last = b.stage()
    b.train(last, [b.valve(S.GATE_VALVE, A.HANDWHEEL)],
            sink=b.equip(S.ENGINE, "M/E", r.choice(["", "MAIN ENGINE", "6G70ME-C10.5"]),
                         size=(r.uniform(3.0, 4.2), r.uniform(2.2, 3.0)), inside=True))
    if b.chance(0.5):
        b.train(last, [b.valve(S.GATE_VALVE, A.HANDWHEEL)], sink=b.connector("TO G/E F.O. SUPPLY UNIT"),
                size_class="branch")
    b.topo.notes += [
        "F.O. SUPPLY PIPES BETWEEN HIGH PRESSURE PUMPS AND INJECTORS TO BE JACKETED (SOLAS II-2/4.2.2.5.2).",
        "TRACE HEATING AND INSULATION FOR ALL H.F.O. LINES EXCEPT M.D.O. LINES.",
    ]


def _lube_oil(b: _Builder) -> None:
    r = b.rng
    b.topo.title = r.choice([("MAIN LUBRICATING OIL SYSTEM",), ("M/E L.O. SYSTEM",),
                             ("LUB. OIL SERVICE SYSTEM",)])
    b.topo.sizes = {"main": r.choice([200, 250, 300, 350]), "branch": r.choice([100, 150]), "small": 40}
    s0 = b.stage()
    sump = b.tank("M/E L.O. SUMP TK", _cap(r, (40, 55, 70), "M3"))
    b.level_gauge(sump, "LIA")
    b.train(s0, [b.valve(S.BUTTERFLY_VALVE, A.HANDWHEEL)], source=sump)
    if b.chance(0.4):
        b.train(s0, [b.valve(S.GLOBE_VALVE, A.QUICK_CLOSING)] + ([b.spectacle()] if b.chance(0.5) else []),
                source=b.connector("FROM L.O. STOR. TK"), size_class="branch")
    s1 = b.pump_stage(b.count(1, 2), "NO.{n} M/E L.O. PUMP", r.choice([S.PUMP_SCREW, S.PUMP_CENTRIFUGAL]),
                      _cap(r, (400, 600, 800, 1000), "M3/H x 4.5 BAR"), (S.BUTTERFLY_VALVE, A.HANDWHEEL),
                      (S.BUTTERFLY_VALVE, A.HANDWHEEL), strainer=False)
    b.drain(s1, ("TO L.O. DRAIN TK", "TO SLUDGE TK", "TO L.O. SETT. TK"))
    purifier = b.chance(0.6)
    if purifier:
        pump = b.equip(S.PUMP_SCREW, "L.O. PURIFIER FEED PUMP" if b.chance(0.5) else "")
        els = [b.valve(S.GATE_VALVE, A.HANDWHEEL), pump, b.equip(S.HEATER, "L.O. PURIFIER HEATER"),
               b.equip(S.PURIFIER, "M/E L.O. PURIFIER", _cap(r, (3000, 4500, 6000), "L/H")),
               b.valve(S.GATE_VALVE, A.HANDWHEEL)]
        tr = b.train(s1, els, size_class="small", template="purifier")
        b.return_from(tr, sump)
    st = b.stage()
    if b.chance(0.6):  # plate cooler with a three-way TCV mixing in the cooler bypass
        cooler = b.equip(S.PLATE_HEAT_EXCHANGER, "M/E L.O. COOLER")
        cooler.in_port, cooler.out_port = "b_in", "b_out"
        b.train(st, b.cooler_with_tcv(cooler, cooling_code="CF", cooling_medium=Medium.FRESH_WATER,
                                      side_ports=("a_in", "a_out"), tcv_service="L.O. temperature control"))
    else:  # shell & tube cooler, temperature controlled on the cooling water side
        cooler = b.equip(S.HEAT_EXCHANGER, "M/E L.O. COOLER")
        cooler.in_port, cooler.out_port = "tube_in", "tube_out"
        b.side_pair(cooler, ("shell_in", "shell_out"), "CF", Medium.FRESH_WATER)
        b.train(st, [b.valve(S.BUTTERFLY_VALVE, A.HANDWHEEL), cooler, b.tap("TI", side=-1, service="L.O. outlet"),
                     b.valve(S.BUTTERFLY_VALVE, A.HANDWHEEL)])
    st = b.stage()
    b0, b1 = b.branch(), b.branch()
    b.topo.bypasses.append(Bypass(b0.key, b1.key, b.valve(S.BUTTERFLY_VALVE, A.HANDWHEEL), side=1))
    el: list[Element] = [b0, b.equip(S.DUPLEX_FILTER, "L.O. AUTO. BACK-FLUSHING FILTER" if b.chance(0.5)
                                     else "L.O. AUTO FILTER"), b1, b.tap("PI", side=-1, service="engine inlet")]
    if b.chance(0.6):
        el.append(b.tap("PT", side=-1, cls=b.remote_cls, service="L.O. inlet pressure"))
    if b.chance(0.5):
        el.append(b.tap("TI", side=-1, service="engine inlet"))
    b.train(st, el)
    last = b.stage()
    b.train(last, [b.valve(S.BUTTERFLY_VALVE, A.HANDWHEEL)],
            sink=b.equip(S.ENGINE, "M/E", r.choice(["", "MAIN ENGINE", "L.O. INLET"]),
                         size=(r.uniform(3.0, 4.2), r.uniform(2.2, 3.0)), inside=True))
    if b.chance(0.4):
        b.train(last, [b.valve(S.GLOBE_VALVE, A.HANDWHEEL)], sink=b.connector("TO STERN TUBE L.O. SYSTEM"),
                size_class="small")
    b.topo.notes += ["L.O. COOLER BY-PASS CONTROLLED BY TEMPERATURE CONTROL VALVE (TCV) VIA AMS.",
                     "L.O. PIPES TO BE PICKLED AND FLUSHED BEFORE CONNECTION TO MAIN ENGINE."]


def _sw_cooling(b: _Builder) -> None:
    r = b.rng
    b.topo.title = r.choice([("SEA WATER COOLING SYSTEM",), ("CENTRAL COOLING S.W. SYSTEM",),
                             ("MAIN SEA WATER COOLING SYSTEM",)])
    b.topo.sizes = {"main": r.choice([350, 400, 450, 500]), "branch": r.choice([250, 300]), "small": 50}
    s0 = b.stage()
    act = r.choice([A.HYDRAULIC, A.HANDWHEEL])
    for name in ["HIGH SEA CHEST", "LOW SEA CHEST"][: b.count(1, 2)]:
        b.train(s0, [b.valve(S.BUTTERFLY_VALVE, act), b.equip(S.STRAINER, "S.W. STRAINER" if b.chance(0.5) else "")],
                source=b.equip(S.SEA_CHEST, name, size=(r.uniform(1.6, 2.4), r.uniform(1.0, 1.4))), template="chest")
    st = b.pump_stage(b.count(1, 3), "NO.{n} MAIN C.S.W. PUMP", S.PUMP_CENTRIFUGAL,
                      _cap(r, (600, 900, 1200, 1500), "M3/H x 25 MTH"), (S.BUTTERFLY_VALVE, A.HANDWHEEL),
                      (S.BUTTERFLY_VALVE, A.HANDWHEEL), strainer=False)
    b.drain(st, ("TO BILGE WELL", "DRAIN TO BILGE", "TO OVERBOARD"))
    st = b.stage()
    nc = b.count(1, 2)
    for k in range(nc):
        cooler = b.equip(S.PLATE_HEAT_EXCHANGER, f"NO.{k + 1} CENTRAL COOLER" if nc > 1 else "CENTRAL F.W. COOLER")
        cooler.in_port, cooler.out_port = "a_in", "a_out"
        b.side_pair(cooler, ("b_in", "b_out"), "CF", Medium.FRESH_WATER)
        el: list[Element] = [b.valve(S.BUTTERFLY_VALVE, A.HANDWHEEL), cooler]
        if b.chance(0.6):
            el.append(b.tap("TI", side=-1, service="cooler S.W. outlet"))
        el.append(b.valve(S.BUTTERFLY_VALVE, A.HANDWHEEL))
        b.train(st, el, template="cooler")
    last = b.stage(fanout=False)
    b.train(last, [b.valve(S.BUTTERFLY_VALVE, act)] + ([b.check()] if b.chance(0.4) else []),
            sink=b.equip(S.OVERBOARD, "OVERBOARD"))
    if b.chance(0.5):
        b.train(last, [b.valve(S.GLOBE_VALVE, A.HANDWHEEL)], sink=b.connector("TO FIRE & G/S PUMP SUCTION"),
                size_class="branch")
    if b.chance(0.4):
        b.train(last, [b.valve(S.BUTTERFLY_VALVE, A.HANDWHEEL)], sink=b.connector("TO SEA CHEST (ICE CLEARING)"),
                size_class="branch")
    b.topo.notes += ["SEA WATER PIPES TO BE GALVANIZED STEEL OR CU-NI 90/10.",
                     "SEA CHESTS TO BE PROVIDED WITH VENT AND STEAM / AIR BLOW-OUT CONNECTIONS."]


def _fw_cooling(b: _Builder) -> None:
    r = b.rng
    b.topo.title = r.choice([("CENTRAL FRESH WATER COOLING SYSTEM",), ("CENTRAL COOLING F.W. SYSTEM",),
                             ("LOW TEMP. CENTRAL COOLING", "FRESH WATER SYSTEM")])
    b.topo.sizes = {"main": r.choice([250, 300, 350, 400]), "branch": r.choice([100, 150, 200]), "small": 40}
    s0 = b.stage()
    b.train(s0, [b.valve(S.BUTTERFLY_VALVE, A.HANDWHEEL)],
            source=b.connector("FROM M/E & AUX. COOLERS (RETURN)"))
    exp = b.tank("C.F.W. EXPANSION TK", _cap(r, (3, 5, 8), "M3"), size=(r.uniform(2.0, 2.8), r.uniform(1.2, 1.6)))
    b.level_gauge(exp, "LAL")
    b.train(s0, [b.valve(S.GATE_VALVE, A.HANDWHEEL)], source=exp, size_class="small")
    st = b.pump_stage(b.count(1, 3), "NO.{n} CENTRAL C.F.W. PUMP", S.PUMP_CENTRIFUGAL,
                      _cap(r, (500, 700, 900, 1100), "M3/H x 30 MTH"), (S.BUTTERFLY_VALVE, A.HANDWHEEL),
                      (S.BUTTERFLY_VALVE, A.HANDWHEEL), strainer=False)
    b.drain(st, ("TO C.F.W. DRAIN TK", "TO DRAIN TK", "DRAIN TO BILGE"))
    st = b.stage()
    cooler = b.equip(S.PLATE_HEAT_EXCHANGER, "CENTRAL F.W. COOLER" if b.chance(0.6) else "CENTRAL COOLER")
    cooler.in_port, cooler.out_port = "b_in", "b_out"
    b.train(st, b.cooler_with_tcv(cooler, cooling_code="SW", cooling_medium=Medium.SEA_WATER,
                                  side_ports=("a_in", "a_out"), tcv_service="C.F.W. temperature control",
                                  bypass_side=1))
    last = b.stage()
    consumers = ["TO M/E AIR COOLER", "TO M/E JACKET F.W. COOLER", "TO M/E L.O. COOLER", "TO G/E COOLERS",
                 "TO AIR COMPRESSORS", "TO A/C & REF. PLANT"]
    r.shuffle(consumers)
    items = [(b.connector(c), [b.valve(r.choice([S.BUTTERFLY_VALVE, S.GLOBE_VALVE]), A.HANDWHEEL)])
             for c in consumers[: b.count(2, 5)]]
    if b.chance(0.5):
        ge = b.equip(S.ENGINE, "NO.1 G/E", size=(r.uniform(2.4, 3.2), r.uniform(1.6, 2.2)), inside=True)
        items.insert(0, (ge, [b.valve(S.GLOBE_VALVE, A.HANDWHEEL)]))
    b.sinks(last, items, size_class="branch")
    b.topo.notes += ["C.F.W. TO BE TREATED WITH CORROSION INHIBITOR (NITRITE-BORATE).",
                     "EXPANSION TANK TO BE LOCATED ABOVE THE HIGHEST POINT OF THE SYSTEM."]


def _fire_main(b: _Builder) -> None:
    r = b.rng
    b.topo.title = r.choice([("FIRE MAIN SYSTEM",), ("FIRE & WASH DECK SYSTEM",), ("FIRE MAIN & HYDRANT SYSTEM",)])
    b.topo.sizes = {"main": r.choice([125, 150, 200]), "branch": r.choice([65, 80, 100]), "small": 25}
    s0 = b.stage()
    for name in ["SEA CHEST (P)", "SEA CHEST (S)"][: b.count(1, 2)]:
        b.train(s0, [b.valve(S.GATE_VALVE, A.HANDWHEEL), b.equip(S.STRAINER)],
                source=b.equip(S.SEA_CHEST, name, size=(r.uniform(1.4, 2.0), r.uniform(0.9, 1.2))), template="chest")
    if b.chance(0.4):
        b.train(s0, [b.valve(S.GATE_VALVE, A.HANDWHEEL)], source=b.connector("FROM S.W. CROSS-OVER MAIN"))
    n = b.count(1, 2)
    st = b.pump_stage(n, "NO.{n} FIRE & G/S PUMP" if b.chance(0.6) else "NO.{n} FIRE PUMP", S.PUMP_CENTRIFUGAL,
                      _cap(r, (72, 100, 150, 200), "M3/H x 90 MTH"), (S.GATE_VALVE, A.HANDWHEEL),
                      (S.GLOBE_VALVE, A.HANDWHEEL), strainer=False,
                      check_cls=r.choice([S.CHECK_VALVE, S.SDNR_VALVE]))
    b.drain(st, ("TO BILGE WELL", "DRAIN TO BILGE", "TO DECK SCUPPER"))
    if b.chance(0.5):
        st = b.stage()
        b.train(st, [b.valve(S.GATE_VALVE, A.HANDWHEEL), b.tap("PI", side=-1, service="fire main")])
    last = b.stage(fanout=True)
    outlets = ["TO FIRE HYDRANT (E/R FLOOR)", "TO FIRE HYDRANT (E/R 3RD DK)", "TO FIRE HYDRANT (UPPER DK)",
               "TO ACCOM. HYDRANTS", "TO FOAM SYSTEM", "TO WASH DECK", "TO ANCHOR WASHING"]
    r.shuffle(outlets)
    items = [(b.connector(o), [b.valve(S.GLOBE_VALVE, A.HANDWHEEL)]) for o in outlets[: b.count(2, 5)]]
    b.sinks(last, items, size_class="branch")
    if b.chance(0.6):
        b.train(last, [b.valve(S.GATE_VALVE, A.HANDWHEEL)] + ([b.spectacle()] if b.chance(0.5) else []),
                sink=b.connector("TO INT'L SHORE CONNECTION"), size_class="branch")
    if b.chance(0.5):
        b.train(last, [b.valve(S.SAFETY_VALVE, A.NONE, tagged=b.chance(0.6))], sink=b.equip(S.OVERBOARD, "OVERBOARD"),
                size_class="small")
    b.topo.notes += ["FIRE MAIN TO BE ARRANGED IN ACCORDANCE WITH SOLAS II-2/10.2.",
                     "ISOLATING VALVE TO BE LOCATED OUTSIDE MACHINERY SPACE."]


def _compressed_air(b: _Builder) -> None:
    r = b.rng
    b.topo.title = r.choice([("COMPRESSED AIR SYSTEM",), ("STARTING AIR SYSTEM", "(COMPRESSED AIR)"),
                             ("STARTING & COMPRESSED AIR", "SYSTEM")])
    b.topo.sizes = {"main": r.choice([50, 65, 80]), "branch": r.choice([25, 32, 40]), "small": 15}
    s0 = b.stage()
    for i in range(b.count(1, 3)):
        comp = b.equip(S.COMPRESSOR, f"NO.{i + 1} MAIN AIR COMPRESSOR" if b.chance(0.7) else f"NO.{i + 1} AIR COMP.",
                       _cap(r, (150, 250, 360, 500), "M3/H x 30 BAR"))
        el: list[Element] = [b.tap("TI", side=-1, service="compressor discharge")] if b.chance(0.5) else []
        el += [b.check(), b.valve(S.GLOBE_VALVE, A.HANDWHEEL)]
        b.train(s0, el, source=comp, template="comp")
    b.drain(s0, ("DRAIN TO BILGE", "TO OILY BILGE TK", "TO BILGE WELL"))
    st = b.stage()
    nr = b.count(1, 2)
    for i in range(nr):
        rec = b.equip(S.AIR_RECEIVER, f"NO.{i + 1} MAIN AIR RECEIVER" if nr > 1 else "MAIN AIR RECEIVER",
                      _cap(r, (5, 8, 10, 12), "M3 x 30 BAR"), size=(r.uniform(0.9, 1.2), r.uniform(1.9, 2.5)))
        b.train(st, [b.valve(S.GLOBE_VALVE, A.HANDWHEEL), rec, b.valve(S.GLOBE_VALVE, A.HANDWHEEL)],
                template="recv")
        b.topo.attachments.append(Attachment(rec.key, b.safety(), side="top", kind="process"))
        if b.chance(0.6):
            b.topo.attachments.append(Attachment(rec.key, b.instr("PI", service="receiver pressure"), side="outer"))
    last = b.stage()
    b.train(last, [b.valve(S.GLOBE_VALVE, A.HANDWHEEL)],
            sink=b.equip(S.ENGINE, "M/E", "STARTING AIR INLET" if b.chance(0.5) else "",
                         size=(r.uniform(3.0, 4.0), r.uniform(2.0, 2.8)), inside=True))
    if b.chance(0.7):
        pcv = b.valve(S.GLOBE_VALVE, A.DIAPHRAGM, tagged=False)
        letters, number = b.namer.instrument("PCV")
        pcv.tag = b.namer.instrument_tag(letters, number)
        el = [b.valve(S.GLOBE_VALVE, A.HANDWHEEL), b.equip(S.FILTER), pcv,
              b.tap("PIC", side=-1, cls=b.remote_cls, signal_to=pcv.key, service="reduced air pressure")]
        b.train(last, el, sink=b.connector("TO CONTROL AIR SYSTEM (7 BAR)"), size_class="small")
    out_cls = r.choice([S.GLOBE_VALVE, S.BALL_VALVE])
    for c in ["TO G/E STARTING AIR", "TO GENERAL SERVICE AIR", "TO WORKSHOP"][: b.count(1, 3)]:
        b.train(last, [b.valve(out_cls, A.HANDWHEEL)], sink=b.connector(c), size_class="branch",
                template="ca_out")
    b.topo.notes += ["AIR RECEIVERS TO BE FITTED WITH SAFETY VALVES AND DRAIN VALVES.",
                     "STARTING AIR CAPACITY: 12 CONSECUTIVE STARTS WITHOUT REFILLING."]


def _steam(b: _Builder) -> None:
    r = b.rng
    b.topo.title = r.choice([("STEAM & CONDENSATE SYSTEM",), ("AUX. BOILER STEAM", "& CONDENSATE SYSTEM"),
                             ("STEAM SUPPLY & CONDENSATE", "RETURN SYSTEM")])
    b.topo.sizes = {"main": r.choice([65, 80, 100]), "branch": r.choice([25, 32, 40]), "small": 15}
    s0 = b.stage()
    boiler = b.equip(S.ENGINE, "AUX. BOILER", _cap(r, (2000, 2500, 3500), "KG/H x 7 BAR"),
                     size=(r.uniform(2.6, 3.4), r.uniform(1.8, 2.4)), inside=True)
    b.train(s0, [b.valve(S.GLOBE_VALVE, A.HANDWHEEL, service="main steam stop")], source=boiler)
    if b.chance(0.5):
        b.train(s0, [b.valve(S.GLOBE_VALVE, A.HANDWHEEL)], source=b.connector("FROM EXH. GAS ECONOMIZER"))
    st = b.stage()
    consumers = ["H.F.O. SETT. TK HEATING", "H.F.O. SERV. TK HEATING", "F.O. PURIFIER HEATER",
                 "L.O. PURIFIER HEATER", "CALORIFIER", "F.O. HEATER"]
    r.shuffle(consumers)
    for name in consumers[: b.count(1, 3)]:
        el: list[Element] = [b.valve(S.GLOBE_VALVE, A.HANDWHEEL), b.equip(S.HEATER, name)]
        if b.chance(0.4):
            el.append(b.tap("TI", side=-1, service="condensate"))
        el += [b.equip(S.STRAINER), b.check(), b.valve(S.GATE_VALVE, A.HANDWHEEL)]
        b.train(st, el, template="heat", size_class="branch")
    b.drain(st, ("TO DRAIN COOLER", "TO OBSERVATION TK", "TO HOT WELL"))
    st = b.stage()
    cascade = b.equip(S.TANK, "CASCADE TK" if b.chance(0.5) else "HOT WELL", _cap(r, (3, 4, 6), "M3"),
                      size=(r.uniform(2.2, 3.0), r.uniform(1.4, 1.8)), inside=True)
    b.train(st, [b.tap("TI", side=-1, service="condensate return"), cascade, b.valve(S.GATE_VALVE, A.HANDWHEEL)])
    b.pump_stage(b.count(1, 2), "NO.{n} BOILER FEED W. PUMP", S.PUMP_CENTRIFUGAL,
                 _cap(r, (3, 5, 8), "M3/H x 12 BAR"), (S.GATE_VALVE, A.HANDWHEEL), (S.GLOBE_VALVE, A.HANDWHEEL),
                 strainer=True, check=True, size_class="branch")
    last = b.stage()
    tr = b.train(last, [b.valve(S.GLOBE_VALVE, A.HANDWHEEL)], size_class="branch")
    if b.chance(0.6):
        b.return_from(tr, boiler)
    else:
        tr.sink = b.connector("TO AUX. BOILER FEED")
    b.topo.notes += ["STEAM TRAPS WITH STRAINER AND BY-PASS TO BE FITTED AT EACH HEATING COIL OUTLET.",
                     "CONDENSATE FROM F.O. HEATING TO BE LED VIA OBSERVATION TANK."]


BLUEPRINTS: dict[SystemType, Callable[[_Builder], None]] = {
    SystemType.BALLAST: _ballast,
    SystemType.BILGE: _bilge,
    SystemType.FUEL_OIL_TRANSFER: _fo_transfer,
    SystemType.FUEL_OIL_SUPPLY: _fo_supply,
    SystemType.LUBE_OIL: _lube_oil,
    SystemType.SEA_WATER_COOLING: _sw_cooling,
    SystemType.FRESH_WATER_COOLING: _fw_cooling,
    SystemType.FIRE_MAIN: _fire_main,
    SystemType.COMPRESSED_AIR: _compressed_air,
    SystemType.STEAM: _steam,
}

GENERIC_NOTES = (
    "ALL VALVES ARE SHOWN IN NORMAL OPERATING POSITION.",
    "LINE NUMBER DESIGNATION: SIZE - SYSTEM CODE - SERIAL NO. - PIPE CLASS.",
    "FOR SYMBOLS AND ABBREVIATIONS SEE PIPING SYMBOL LIST.",
    "PIPE SIZES ARE NOMINAL BORE (JIS 'A').",
    "INSTRUMENT IDENTIFICATION IN ACCORDANCE WITH ISA 5.1.",
    "ALL PIPING TO BE IN ACCORDANCE WITH CLASS RULES AND SOLAS REQUIREMENTS.",
    "DIMENSIONS AND ELEVATIONS SEE PIPING ARRANGEMENT DRAWINGS.",
)


def build_topology(system: SystemType, rng: random.Random, complexity: float = 1.0) -> SystemTopology:
    """Random topology of ``system``; ``complexity`` in (0, 1] scales the optional counts."""
    if system not in BLUEPRINTS:
        raise ValueError(f"no blueprint for system {system!r}")
    b = _Builder(system, rng, max(0.0, min(1.0, complexity)))
    BLUEPRINTS[system](b)
    topo = b.topo
    topo.stages = [s for s in topo.stages if s.trains]
    notes = list(GENERIC_NOTES)
    rng.shuffle(notes)
    topo.notes = notes[: rng.randint(2, 4)] + topo.notes[: rng.randint(1, len(topo.notes))]
    topo.ship = ship_identity(rng)
    return topo
