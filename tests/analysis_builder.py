"""Hand-built P&ID documents for the analysis / report / evaluation tests.

Objects are placed with the symbol library (exact bboxes and port positions), pipes start and
end on the named ports (or on the bbox edge facing the other end for boundary-port symbols)
and are routed orthogonally. ``send_port=False`` drops the port name from the endpoint so the
analysis has to infer it from the geometry.
"""

from __future__ import annotations

from typing import Optional, Union

from shippid.model import (
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
    SystemType,
    TextItem,
    TextRole,
)
from shippid.symbols import Placement, get_symbol, placed_bbox, placed_ports

S = SymbolClass
A = ActuatorType

Node = Union[str, tuple[str, str], None]  # id, (id, port) or None (free end)

_OUT = {S.PUMP_CENTRIFUGAL: "discharge"}
_IN = {S.PUMP_CENTRIFUGAL: "suction"}


class DocBuilder:
    def __init__(self, system: SystemType = SystemType.UNKNOWN, title: Optional[str] = None,
                 systems: Optional[list[SystemType]] = None, width: float = 1200, height: float = 800):
        self.info = DrawingInfo(system=system, title=title, systems=systems or [])
        self.objects: list[PIDObject] = []
        self.junctions: list[Junction] = []
        self.pipes: list[PipeRun] = []
        self.texts: list[TextItem] = []
        self.pages = [PageInfo(index=0, width=width, height=height)]

    # -- nodes ---------------------------------------------------------------------
    def obj(self, cls: SymbolClass, x: float, y: float, *, tag: Optional[str] = None, name: Optional[str] = None,
            rotation: int = 0, mirrored: bool = False, actuator: ActuatorType = A.NONE, scale: float = 20,
            stretch: tuple[float, float] = (1.0, 1.0), confidence: float = 1.0, name_as_text: bool = False,
            attrs: Optional[dict] = None, page: int = 0) -> str:
        spec = get_symbol(cls)
        pl = Placement(x, y, scale, rotation, mirrored, stretch)
        bb = placed_bbox(spec, pl, actuator)
        ports = [Port(name=n, position=Point(x=px, y=py), direction=Point(x=dx, y=dy))
                 for n, (px, py), (dx, dy) in placed_ports(spec, pl)]
        oid = f"o{len(self.objects) + 1}"
        attributes = dict(attrs or {})
        if name and not name_as_text:
            attributes["name"] = name
        self.objects.append(PIDObject(id=oid, cls=cls, category=spec.category,
                                      bbox=BBox(x0=bb[0], y0=bb[1], x1=bb[2], y1=bb[3]), rotation=rotation,
                                      mirrored=mirrored, tag=tag, actuator=actuator, ports=ports,
                                      attributes=attributes, confidence=confidence, page=page))
        if name and name_as_text:
            role = TextRole.CONNECTOR if cls == S.OFF_PAGE_CONNECTOR else TextRole.LABEL
            self.text(name, bb[0], bb[3] + 2, role=role, owner=oid)
        return oid

    def unknown(self, x: float, y: float, *, name: Optional[str] = None, w: float = 20, h: float = 20,
                category: Category = Category.EQUIPMENT) -> str:
        oid = f"o{len(self.objects) + 1}"
        self.objects.append(PIDObject(id=oid, cls=S.UNKNOWN, category=category,
                                      bbox=BBox(x0=x - w / 2, y0=y - h / 2, x1=x + w / 2, y1=y + h / 2),
                                      attributes={"name": name} if name else {}, confidence=0.4))
        return oid

    def junction(self, x: float, y: float, dotted: bool = True) -> str:
        jid = f"j{len(self.junctions) + 1}"
        self.junctions.append(Junction(id=jid, position=Point(x=x, y=y), dotted=dotted))
        return jid

    def text(self, text: str, x: float, y: float, *, role: TextRole = TextRole.UNKNOWN, owner: Optional[str] = None,
             w: Optional[float] = None, h: float = 6) -> str:
        tid = f"t{len(self.texts) + 1}"
        self.texts.append(TextItem(id=tid, text=text, bbox=BBox(x0=x, y0=y, x1=x + (w or 4 * len(text)), y1=y + h),
                                   size=h, role=role, owner_id=owner))
        return tid

    # -- pipes ---------------------------------------------------------------------
    def _o(self, oid: str) -> PIDObject:
        return next(o for o in self.objects if o.id == oid)

    def _point(self, node: Node, other_xy: tuple[float, float], default_port: Optional[str],
               free_xy: Optional[tuple[float, float]]) -> tuple[tuple[float, float], Endpoint]:
        if node is None:
            assert free_xy is not None
            return free_xy, Endpoint()
        nid, port = (node if isinstance(node, tuple) else (node, None))
        if nid.startswith("j"):
            j = next(j for j in self.junctions if j.id == nid)
            return (j.position.x, j.position.y), Endpoint(junction_id=nid)
        o = self._o(nid)
        names = {p.name: p for p in o.ports}
        if port is None and default_port == "@nearest" and names:
            port = min(names, key=lambda n: (names[n].position.x - other_xy[0]) ** 2 +
                       (names[n].position.y - other_xy[1]) ** 2)
        port = port or (default_port if default_port in names else None)
        if port in names:
            p = names[port].position
            return (p.x, p.y), Endpoint(object_id=nid, port=port)
        c = o.bbox.center
        dx, dy = other_xy[0] - c.x, other_xy[1] - c.y
        if abs(dx) >= abs(dy):
            xy = (o.bbox.x1 if dx > 0 else o.bbox.x0, c.y)
        else:
            xy = (c.x, o.bbox.y1 if dy > 0 else o.bbox.y0)
        return xy, Endpoint(object_id=nid, port=port)

    def _center(self, node: Node, free_xy: Optional[tuple[float, float]]) -> tuple[float, float]:
        if node is None:
            return free_xy  # type: ignore[return-value]
        nid = node[0] if isinstance(node, tuple) else node
        if nid.startswith("j"):
            j = next(j for j in self.junctions if j.id == nid)
            return (j.position.x, j.position.y)
        c = self._o(nid).bbox.center
        return (c.x, c.y)

    def _default(self, node: Node, outgoing: bool) -> Optional[str]:
        if node is None or isinstance(node, tuple) or node.startswith("j"):
            return None
        cls = self._o(node).cls
        return (_OUT.get(cls, "out") if outgoing else _IN.get(cls, "in"))

    def pipe(self, a: Node, b: Node, *, kind: PipeKind = PipeKind.PROCESS, line: Optional[str] = None,
             size: Optional[str] = None, medium: Medium = Medium.UNKNOWN, flow: Optional[int] = None,
             send_port: bool = True, a_xy: Optional[tuple[float, float]] = None,
             b_xy: Optional[tuple[float, float]] = None, page: int = 0) -> str:
        ca, cb = self._center(a, a_xy), self._center(b, b_xy)
        da = self._default(a, True) if send_port else "@nearest"
        db = self._default(b, False) if send_port else "@nearest"
        pa, ea = self._point(a, cb, da, a_xy)
        pb, eb = self._point(b, ca, db, b_xy)
        if not send_port:
            ea = Endpoint(object_id=ea.object_id, junction_id=ea.junction_id)
            eb = Endpoint(object_id=eb.object_id, junction_id=eb.junction_id)
        pts = [pa]
        if abs(pa[0] - pb[0]) > 1e-6 and abs(pa[1] - pb[1]) > 1e-6:
            pts.append((pb[0], pa[1]))
        pts.append(pb)
        pid = f"p{len(self.pipes) + 1}"
        self.pipes.append(PipeRun(id=pid, points=[Point(x=x, y=y) for x, y in pts], start=ea, end=eb, kind=kind,
                                  line_number=line, size=size, medium=medium, flow_direction=flow, page=page))
        return pid

    def chain(self, *nodes: Node, **kw) -> list[str]:
        return [self.pipe(a, b, **kw) for a, b in zip(nodes, nodes[1:])]

    def build(self, source: str = "test") -> PIDDocument:
        return PIDDocument(source=source, pages=self.pages, info=self.info, objects=self.objects,
                           junctions=self.junctions, pipes=self.pipes, texts=self.texts)


def pump_station(system: SystemType = SystemType.UNKNOWN, *, n_pumps: int = 2, nrv: tuple[bool, ...] = (True, True),
                 source: SymbolClass = S.TANK, source_name: str = "NO.1 TK", pump_cls: SymbolClass = S.PUMP_CENTRIFUGAL,
                 pump_name: str = "NO.{n} PUMP", strainer: bool = True, relief: tuple[bool, ...] = (False, False),
                 sink: SymbolClass = S.TANK, sink_name: str = "NO.2 TK",
                 title: Optional[str] = None) -> tuple[DocBuilder, dict[str, list[str] | str]]:
    """source -> valve -> [strainer] -> suction header -> n x (valve, pump, [NRV], valve) -> discharge
    header -> valve -> sink. ``relief[i]`` adds a relief valve from pump i's discharge to its suction."""
    b = DocBuilder(system, title=title)
    ids: dict[str, list[str] | str] = {"pumps": [], "nrvs": [], "reliefs": [], "suction_valves": [],
                                       "discharge_valves": []}
    src = b.obj(source, 60, 300, name=source_name, stretch=(1.0, 1.0))
    v0 = b.obj(S.GATE_VALVE, 160, 300, tag="V-001", actuator=A.HANDWHEEL)
    nodes: list[Node] = [src, v0]
    if strainer:
        nodes.append(b.obj(S.STRAINER, 230, 300, tag="F-001"))
    hs = b.junction(300, 300)
    b.chain(*nodes, hs)
    hd = b.junction(900, 300)
    for i in range(n_pumps):
        y = 300 + i * 120
        js = b.junction(300, y) if i else hs
        if i:
            b.pipe(prev_s, js)  # noqa: F821 - defined in the previous iteration
        sv = b.obj(S.GATE_VALVE, 380, y, tag=f"V-1{i}1", actuator=A.HANDWHEEL)
        p = b.obj(pump_cls, 520, y, name=pump_name.format(n=i + 1), tag=f"P-{i + 1:02d}")
        chain: list[Node] = [js, sv]
        if relief[i] if i < len(relief) else False:
            ja = b.junction(460, y)
            jb = b.junction(600, y)
            chain += [ja, p, jb]
        else:
            chain += [p]
        if nrv[i] if i < len(nrv) else False:
            cv = b.obj(S.CHECK_VALVE, 680, y, tag=f"V-1{i}2")
            chain.append(cv)
            ids["nrvs"].append(cv)  # type: ignore[union-attr]
        dv = b.obj(S.GLOBE_VALVE, 780, y, tag=f"V-1{i}3", actuator=A.HANDWHEEL)
        jd = b.junction(900, y) if i else hd
        if i:
            b.pipe(prev_d, jd)  # noqa: F821
        chain += [dv, jd]
        b.chain(*chain)
        if relief[i] if i < len(relief) else False:
            rv = b.obj(S.SAFETY_VALVE, 530, y - 50, tag=f"RV-{i + 1}")
            b.pipe(jb, (rv, "in"))
            b.pipe((rv, "out"), ja)
            ids["reliefs"].append(rv)  # type: ignore[union-attr]
        ids["pumps"].append(p)  # type: ignore[union-attr]
        ids["suction_valves"].append(sv)  # type: ignore[union-attr]
        ids["discharge_valves"].append(dv)  # type: ignore[union-attr]
        prev_s, prev_d = js, jd  # noqa: F841
    v9 = b.obj(S.GATE_VALVE, 980, 300, tag="V-009", actuator=A.HANDWHEEL)
    snk = b.obj(sink, 1100, 300, name=sink_name)
    b.chain(hd, v9, snk)
    ids.update(source=src, sink=snk, header_s=hs, header_d=hd)
    return b, ids
