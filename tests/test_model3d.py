"""Tests for shippid.model3d: zones, heuristic plan, solver, router, Gemma planner, GLB + viewers.

The P&IDs are hand built (no dependency on shippid.datasets) with the symbol library, so
object bboxes and ports are exactly where the extractor would report them.
"""

from __future__ import annotations

import base64
import json
import math
import re
import struct
import time
from typing import Optional, Union

import pytest

from shippid.model import (ActuatorType, BBox, Category, DrawingInfo, Endpoint, Junction, Medium, PageInfo,
                           PIDDocument, PIDObject, PipeKind, PipeRun, Point, Port, SymbolClass, SystemType)
from shippid.symbols import Placement, get_symbol, placed_bbox, placed_ports

S = SymbolClass
Ref = Union[str, tuple[str, str]]  # junction id or (object id, port name)


class DocBuilder:
    """Minimal P&ID document builder (page points, y down)."""

    def __init__(self, title: str, system: SystemType, ship_type: str = "VLCC"):
        self.doc = PIDDocument(source=f"test:{title}",
                               pages=[PageInfo(index=0, width=1684, height=1190,
                                               drawing_area=BBox(x0=20, y0=20, x1=1664, y1=1000))],
                               info=DrawingInfo(title=title, system=system, systems=[system], ship_type=ship_type,
                                                drawing_no="TEST-" + system.value.upper()))
        self._pos: dict[tuple[str, str], tuple[float, float]] = {}
        self._nb: dict[str, int] = {}

    def obj(self, cls: SymbolClass, x: float, y: float, tag: Optional[str] = None, name: Optional[str] = None,
            rotation: int = 0, scale: float = 18.0, stretch: tuple[float, float] = (1.0, 1.0),
            actuator: ActuatorType = ActuatorType.NONE) -> str:
        spec = get_symbol(cls)
        pl = Placement(x, y, scale=scale, rotation=rotation if spec.rotatable else 0, stretch=stretch)
        act = actuator if spec.category == Category.VALVE else ActuatorType.NONE
        x0, y0, x1, y1 = placed_bbox(spec, pl, act)
        oid = f"o{len(self.doc.objects) + 1}"
        ports = []
        for name_, (px, py), (dx, dy) in placed_ports(spec, pl):
            ports.append(Port(name=name_, position=Point(x=px, y=py), direction=Point(x=dx, y=dy)))
            self._pos[(oid, name_)] = (px, py)
        attrs = {"name": name} if name else {}
        self.doc.objects.append(PIDObject(id=oid, cls=cls, category=spec.category,
                                          bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1), rotation=pl.rotation, tag=tag,
                                          actuator=act, ports=ports, attributes=attrs))
        return oid

    def bport(self, oid: str, x: float, y: float) -> tuple[str, str]:
        """New boundary port n<k> on a boundary object at a page point on its outline."""
        o = self.doc.object_by_id()[oid]
        k = self._nb.get(oid, 0)
        self._nb[oid] = k + 1
        name = f"n{k}"
        c = o.center
        dx, dy = x - c.x, y - c.y
        n = math.hypot(dx, dy) or 1.0
        o.ports.append(Port(name=name, position=Point(x=x, y=y), direction=Point(x=dx / n, y=dy / n)))
        self._pos[(oid, name)] = (x, y)
        return (oid, name)

    def junction(self, x: float, y: float) -> str:
        jid = f"j{len(self.doc.junctions) + 1}"
        self.doc.junctions.append(Junction(id=jid, position=Point(x=x, y=y), dotted=True))
        self._pos[(jid, "")] = (x, y)
        return jid

    def _where(self, r: Ref) -> tuple[tuple[float, float], Endpoint]:
        if isinstance(r, str):
            return self._pos[(r, "")], Endpoint(junction_id=r)
        return self._pos[r], Endpoint(object_id=r[0], port=r[1])

    def pipe(self, a: Ref, b: Ref, line: Optional[str] = None, size: Optional[str] = None,
             medium: Medium = Medium.UNKNOWN, kind: PipeKind = PipeKind.PROCESS, vfirst: bool = False) -> str:
        (ax, ay), ea = self._where(a)
        (bx, by), eb = self._where(b)
        pts = [(ax, ay)]
        if abs(ax - bx) > 1e-6 and abs(ay - by) > 1e-6:
            pts.append((ax, by) if vfirst else (bx, ay))
        pts.append((bx, by))
        pid = f"p{len(self.doc.pipes) + 1}"
        self.doc.pipes.append(PipeRun(id=pid, points=[Point(x=x, y=y) for x, y in pts], start=ea, end=eb,
                                      kind=kind, line_number=line, size=size, medium=medium))
        return pid


def ballast_doc() -> PIDDocument:
    """Sea chest -> strainer -> 2 ballast pumps (valves, check valves, gauge) -> BWTS -> 2 WB tanks / overboard."""
    b = DocBuilder("BALLAST WATER SYSTEM", SystemType.BALLAST, "300K VLCC")
    BW = Medium.BALLAST_WATER
    sc = b.obj(S.SEA_CHEST, 120, 760, name="HIGH SEA CHEST (S)", stretch=(1.4, 1.2))
    st = b.obj(S.STRAINER, 120, 640, tag="BA-F001", name="MUD BOX", rotation=270)
    j1 = b.junction(120, 520)
    b.pipe((sc, "out"), (st, "in"), "350A-BA-001", "350A", BW)
    b.pipe((st, "out"), j1, "350A-BA-001", "350A", BW)
    j_dis = None
    pumps, chk = [], []
    for i, y in enumerate((520, 380)):
        v_in = b.obj(S.BUTTERFLY_VALVE, 260, y, tag=f"BA-V{11 + i:03d}", actuator=ActuatorType.HYDRAULIC)
        p = b.obj(S.PUMP_CENTRIFUGAL, 400, y, tag=f"BA-P{i + 1:03d}", name=f"NO.{i + 1} BALLAST PUMP")
        cv = b.obj(S.CHECK_VALVE, 520, y - 60, tag=f"BA-V{21 + i:03d}")
        v_out = b.obj(S.BUTTERFLY_VALVE, 640, y - 60, tag=f"BA-V{31 + i:03d}", actuator=ActuatorType.HYDRAULIC)
        if i == 0:
            b.pipe(j1, (v_in, "in"), "300A-BA-002", "300A", BW)
        else:  # second branch tees off the suction main between j1 and the first valve
            jb = b.junction(180, 520)
            b.pipe(j1, jb, "300A-BA-002", "300A", BW)
            b.pipe(jb, (v_in, "in"), "300A-BA-003", "300A", BW, vfirst=True)
        b.pipe((v_in, "out"), (p, "suction"), f"300A-BA-{2 + i:03d}", "300A", BW)
        jg = b.junction(400, y - 60)
        b.pipe((p, "discharge"), jg, f"250A-BA-01{i}", "250A", BW, vfirst=True)
        pi = b.obj(S.INSTRUMENT_LOCAL, 400, y - 110, tag=f"PI-10{i + 1}")
        b.pipe(jg, b.bport(pi, 400, y - 110 + 7.2), None, "15A", BW, kind=PipeKind.INSTRUMENT_TAP)
        b.pipe(jg, (cv, "in"), f"250A-BA-01{i}", "250A", BW)
        b.pipe((cv, "out"), (v_out, "in"), f"250A-BA-01{i}", "250A", BW)
        if j_dis is None:
            j_dis = b.junction(760, 320)
        b.pipe((v_out, "out"), j_dis, f"250A-BA-01{i}", "250A", BW)
        pumps.append(p)
        chk.append(cv)
    bw = b.obj(S.BWTS_UNIT, 900, 320, tag="BA-U001", name="B.W.T.S. UV UNIT")
    b.pipe(j_dis, (bw, "in"), "350A-BA-020", "350A", BW)
    j4 = b.junction(1100, 320)
    b.pipe((bw, "out"), j4, "350A-BA-021", "350A", BW)
    tanks = []
    for k, (name, y) in enumerate((("NO.1 W.B.TK (P)", 160), ("NO.1 W.B.TK (S)", 480))):
        v = b.obj(S.BUTTERFLY_VALVE, 1250, y, tag=f"BA-V{41 + k:03d}", actuator=ActuatorType.HYDRAULIC)
        t = b.obj(S.TANK, 1450, y, name=name, stretch=(1.2, 1.0), scale=30)
        b.pipe(j4, (v, "in"), f"300A-BA-03{k}", "300A", BW, vfirst=True)
        b.pipe((v, "out"), b.bport(t, 1450 - 45 * 1.2, y), f"300A-BA-03{k}", "300A", BW)
        tanks.append(t)
    vo = b.obj(S.GATE_VALVE, 1250, 700, tag="BA-V051", actuator=ActuatorType.HANDWHEEL)
    ob = b.obj(S.OVERBOARD, 1450, 700, name="OVERBOARD (S)")
    b.pipe(j4, (vo, "in"), "300A-BA-040", "300A", BW, vfirst=True)
    b.pipe((vo, "out"), (ob, "in"), "300A-BA-040", "300A", BW)
    return b.doc


def fo_transfer_doc() -> PIDDocument:
    """Two HFO storage tanks (quick-closing valves) -> strainer -> 2 screw pumps with relief valves ->
    2 settling tanks + off-page connector to the purifiers."""
    b = DocBuilder("FUEL OIL TRANSFER SYSTEM", SystemType.FUEL_OIL_TRANSFER, "14000 TEU CONTAINER CARRIER")
    FO = Medium.FUEL_OIL
    j_suc = b.junction(420, 500)
    for k, (name, y) in enumerate((("NO.1 H.F.O. STORAGE TK (P)", 300), ("NO.1 H.F.O. STORAGE TK (S)", 700))):
        t = b.obj(S.TANK, 120, y, name=name, scale=30)
        qc = b.obj(S.GATE_VALVE, 260, y, tag=f"FO-V{1 + k:03d}", actuator=ActuatorType.QUICK_CLOSING)
        b.pipe(b.bport(t, 120 + 45, y), (qc, "in"), "150A-FO-001", "150A", FO)
        b.pipe((qc, "out"), j_suc, "150A-FO-001", "150A", FO)
    st = b.obj(S.DUPLEX_FILTER, 520, 500, tag="FO-F001", name="F.O. TRANSFER PUMP SUCTION FILTER")
    b.pipe(j_suc, (st, "in"), "150A-FO-002", "150A", FO)
    j_a = b.junction(620, 500)
    b.pipe((st, "out"), j_a, "150A-FO-002", "150A", FO)
    j_out = b.junction(1100, 500)
    for i, y in enumerate((420, 580)):
        jy = b.junction(620, y)
        b.pipe(j_a, jy, "125A-FO-003", "125A", FO)
        p = b.obj(S.PUMP_SCREW, 800, y, tag=f"FO-P{i + 1:03d}", name=f"NO.{i + 1} F.O. TRANSFER PUMP")
        b.pipe(jy, (p, "in"), "125A-FO-003", "125A", FO)
        jd = b.junction(900, y)
        b.pipe((p, "out"), jd, "100A-FO-004", "100A", FO)
        sv = b.obj(S.SAFETY_VALVE, 760, y - 50 if i == 0 else y + 50, tag=f"FO-V{11 + i:03d}")
        b.pipe(jd, (sv, "in"), "65A-FO-005", "65A", FO, vfirst=True)
        b.pipe((sv, "out"), jy, "65A-FO-005", "65A", FO)
        cv = b.obj(S.CHECK_VALVE, 980, y, tag=f"FO-V{21 + i:03d}")
        b.pipe(jd, (cv, "in"), "100A-FO-004", "100A", FO)
        b.pipe((cv, "out"), j_out, "100A-FO-004", "100A", FO)
    for k, (name, y) in enumerate((("NO.1 H.F.O. SETT. TK", 260), ("NO.2 H.F.O. SETT. TK", 740))):
        v = b.obj(S.GLOBE_VALVE, 1250, y, tag=f"FO-V{31 + k:03d}", actuator=ActuatorType.HANDWHEEL)
        t = b.obj(S.TANK, 1450, y, name=name, scale=30)
        b.pipe(j_out, (v, "in"), "100A-FO-006", "100A", FO, vfirst=True)
        b.pipe((v, "out"), b.bport(t, 1450 - 45, y), "100A-FO-006", "100A", FO)
    opc = b.obj(S.OFF_PAGE_CONNECTOR, 1450, 500, name="TO F.O. PURIFIER")
    b.pipe(j_out, (opc, "in"), "80A-FO-007", "80A", FO)
    return b.doc


_TRAIN_SPECS = [
    # (system, medium, code, source class, source name, unit classes, destination class, destination name)
    (SystemType.SEA_WATER_COOLING, Medium.SEA_WATER, "SW", S.SEA_CHEST, "LOW SEA CHEST",
     [S.PLATE_HEAT_EXCHANGER], S.OVERBOARD, "OVERBOARD (P)"),
    (SystemType.FRESH_WATER_COOLING, Medium.FRESH_WATER, "CF", S.TANK, "F.W. EXPANSION TK",
     [S.PLATE_HEAT_EXCHANGER], S.ENGINE, "M/E"),
    (SystemType.LUBE_OIL, Medium.LUBE_OIL, "LO", S.TANK, "M/E L.O. SUMP TK",
     [S.HEAT_EXCHANGER, S.DUPLEX_FILTER], S.ENGINE, "M/E"),
    (SystemType.FUEL_OIL_SUPPLY, Medium.FUEL_OIL, "FS", S.TANK, "NO.1 H.F.O. SERV. TK (P)",
     [S.HEATER, S.DUPLEX_FILTER, S.FLOW_METER], S.ENGINE, "M/E"),
    (SystemType.FUEL_OIL_TRANSFER, Medium.FUEL_OIL, "FO", S.TANK, "NO.2 H.F.O. STORAGE TK (S)",
     [S.PURIFIER, S.HEATER], S.TANK, "NO.2 H.F.O. SETT. TK"),
    (SystemType.BILGE, Medium.BILGE, "BG", S.TANK, "BILGE HOLDING TK",
     [S.OILY_WATER_SEPARATOR], S.OVERBOARD, "OVERBOARD (S)"),
    (SystemType.COMPRESSED_AIR, Medium.COMPRESSED_AIR, "CA", S.COMPRESSOR, "NO.1 MAIN AIR COMPRESSOR",
     [S.AIR_RECEIVER], S.OFF_PAGE_CONNECTOR, "TO M/E STARTING AIR"),
    (SystemType.FIRE_MAIN, Medium.FIRE_WATER, "FM", S.SEA_CHEST, "HIGH SEA CHEST (P)",
     [], S.OFF_PAGE_CONNECTOR, "TO FIRE MAIN DECK"),
]


def large_doc(target: int = 150) -> PIDDocument:
    """Multi-system engine-room P&ID with >= ``target`` objects: per system a source, suction valve and
    strainer, two pump trains (valves, check valve, gauge), process units and a destination; the
    cooling / oil trains share one main engine."""
    b = DocBuilder("ENGINE ROOM MACHINERY SYSTEMS", SystemType.SEA_WATER_COOLING, "174K LNG CARRIER")
    b.doc.pages[0] = PageInfo(index=0, width=3370, height=2384, drawing_area=BBox(x0=20, y0=20, x1=3350, y1=2200))
    me = b.obj(S.ENGINE, 3100, 1100, name="M/E", scale=60, stretch=(1.5, 3.0))
    k = 0
    while len(b.doc.objects) < target:
        sys_, med, code, scls, sname, units, dcls, dname = _TRAIN_SPECS[k % len(_TRAIN_SPECS)]
        rep = k // len(_TRAIN_SPECS)
        y = 120 + (k % len(_TRAIN_SPECS)) * 260 + rep * 60
        x0 = 80 + rep * 40
        n = f"{k:02d}"
        src = b.obj(scls, x0, y, name=sname if rep == 0 else f"{sname} {rep + 1}", scale=30 if scls == S.TANK else 18)
        src_ref = b.bport(src, x0 + 45, y) if get_symbol(scls).boundary_ports else (src, get_symbol(scls).ports[-1].name)
        sv = b.obj(S.GATE_VALVE, x0 + 160, y, tag=f"{code}-V{k}01", actuator=ActuatorType.HANDWHEEL)
        st = b.obj(S.STRAINER, x0 + 260, y, tag=f"{code}-F{k}01")
        size = "200A" if med in (Medium.SEA_WATER, Medium.FIRE_WATER) else "100A"
        line = f"{size}-{code}-{k}01"
        b.pipe(src_ref, (sv, "in"), line, size, med)
        b.pipe((sv, "out"), (st, "in"), line, size, med)
        js = b.junction(x0 + 360, y)
        b.pipe((st, "out"), js, line, size, med)
        jd = b.junction(x0 + 1100, y)
        for i, dy in enumerate((-60, 60)):
            jy = b.junction(x0 + 360, y + dy)
            b.pipe(js, jy, line, size, med)
            vi = b.obj(S.BUTTERFLY_VALVE, x0 + 460, y + dy, tag=f"{code}-V{k}1{i}")
            pcls = S.PUMP_SCREW if med in (Medium.FUEL_OIL, Medium.LUBE_OIL) else S.PUMP_CENTRIFUGAL
            if med == Medium.COMPRESSED_AIR:
                pcls = S.COMPRESSOR
            p = b.obj(pcls, x0 + 600, y + dy, tag=f"{code}-P{k}{i}", name=f"NO.{i + 1} {code} PUMP {n}")
            pin = "suction" if pcls == S.PUMP_CENTRIFUGAL else "in"
            pout = "discharge" if pcls == S.PUMP_CENTRIFUGAL else "out"
            b.pipe(jy, (vi, "in"), line, size, med)
            b.pipe((vi, "out"), (p, pin), line, size, med)
            jg = b.junction(x0 + 720, y + dy)
            b.pipe((p, pout), jg, f"{size}-{code}-{k}02", size, med, vfirst=True)
            gi = b.obj(S.INSTRUMENT_LOCAL, x0 + 720, y + dy - 40, tag=f"PI-{k}{i}1")
            b.pipe(jg, b.bport(gi, x0 + 720, y + dy - 40 + 7.2), None, "15A", med, kind=PipeKind.INSTRUMENT_TAP)
            cv = b.obj(S.CHECK_VALVE, x0 + 820, y + dy, tag=f"{code}-V{k}2{i}")
            vo = b.obj(S.GLOBE_VALVE, x0 + 940, y + dy, tag=f"{code}-V{k}3{i}", actuator=ActuatorType.HANDWHEEL)
            b.pipe(jg, (cv, "in"), f"{size}-{code}-{k}02", size, med)
            b.pipe((cv, "out"), (vo, "in"), f"{size}-{code}-{k}02", size, med)
            b.pipe((vo, "out"), jd, f"{size}-{code}-{k}02", size, med, vfirst=True)
        prev: Ref = jd
        x = x0 + 1250
        for ucls in units:
            u = b.obj(ucls, x, y, tag=f"{code}-E{k}{x}", name=f"{code} {ucls.value.upper()} {n}",
                      scale=24 if ucls != S.AIR_RECEIVER else 30)
            spec = get_symbol(ucls)
            if spec.boundary_ports:
                uin, uout = b.bport(u, x - 12, y), b.bport(u, x + 12, y)
            else:
                names = [p.name for p in spec.ports]
                uin, uout = (u, names[0]), (u, names[1])
            b.pipe(prev, uin, f"{size}-{code}-{k}03", size, med)
            prev = uout
            x += 250
        vd = b.obj(S.BUTTERFLY_VALVE, x, y, tag=f"{code}-V{k}41", actuator=ActuatorType.HYDRAULIC)
        b.pipe(prev, (vd, "in"), f"{size}-{code}-{k}04", size, med)
        if dname == "M/E":
            mo = b.doc.object_by_id()[me]
            b.pipe((vd, "out"), b.bport(me, mo.bbox.x0, y), f"{size}-{code}-{k}04", size, med)
        else:
            d = b.obj(dcls, x + 200, y, name=dname if rep == 0 else f"{dname} {rep + 1}",
                      scale=30 if dcls == S.TANK else 18)
            dref = b.bport(d, x + 200 - 45, y) if get_symbol(dcls).boundary_ports else (d, get_symbol(dcls).ports[0].name)
            b.pipe((vd, "out"), dref, f"{size}-{code}-{k}04", size, med)
        k += 1
    return b.doc
