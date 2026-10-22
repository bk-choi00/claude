"""Tests for shippid.model3d: zones, heuristic plan, solver, router, Gemma planner, GLB + viewers.

The P&IDs are hand built (no dependency on shippid.datasets) with the symbol library, so
object bboxes and ports are exactly where the extractor would report them.
"""

from __future__ import annotations

import base64
import json
import math
import re
import time
from typing import Optional, Union

import pytest
from pydantic import ValidationError

from shippid.model import (ActuatorType, BBox, Category, DrawingInfo, Endpoint, Junction, Layout3D, Medium,  # noqa: E501
                           PageInfo, PIDDocument, PIDObject, PipeKind, PipeRun, Point, Port, SymbolClass, SystemType)
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
        sspec = get_symbol(scls)
        src_ref = b.bport(src, x0 + 45, y) if sspec.boundary_ports else (src, sspec.ports[-1].name)
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
            dspec = get_symbol(dcls)
            dref = b.bport(d, x + 155, y) if dspec.boundary_ports else (d, dspec.ports[0].name)
            b.pipe((vd, "out"), dref, f"{size}-{code}-{k}04", size, med)
        k += 1
    return b.doc


# =========================================================================== helpers for checks

from shippid.llm.base import LLMClient, MockBackend  # noqa: E402
from shippid.model3d.builder import build_layout  # noqa: E402
from shippid.model3d.gemma_planner import GemmaLayoutPlanner  # noqa: E402
from shippid.model3d.heuristic import heuristic_plan  # noqa: E402
from shippid.model3d.plan import LayoutPlan, PlanItem, PlanItemDraft, check_plan, is_planned  # noqa: E402
from shippid.model3d.router import route, segment_hits_box  # noqa: E402
from shippid.model3d.scene import export_glb, layout_to_scene, read_glb_json  # noqa: E402
from shippid.model3d.solver import (MARKERS, MARKER_CLEARANCE, STRUCTURAL, nozzle_normal, placement_box,  # noqa: E402
                                    solve, solve_detailed)
from shippid.model3d.viewer import write_viewer_html  # noqa: E402
from shippid.model3d.zones import engine_room_for, load_engine_room  # noqa: E402

DOCS = {"ballast": ballast_doc, "fo_transfer": fo_transfer_doc}


def _check_no_overlap(placements, room, clearance=0.6):
    """Hard constraints of the solver: inside the room, clearance between equipment boxes."""
    bounds = room.bounds
    boxes = [(p, placement_box(p)) for p in placements]
    for p, b in boxes:
        assert bounds.x0 - 1e-6 <= b.x0 and b.x1 <= bounds.x1 + 1e-6, p.object_id
        assert bounds.y0 - 1e-6 <= b.y0 and b.y1 <= bounds.y1 + 1e-6, p.object_id
        assert bounds.z0 - 1e-6 <= b.z0 and b.z1 <= bounds.z1 + 1e-6, p.object_id
    for i, (p, a) in enumerate(boxes):
        for q, b in boxes[i + 1:]:
            if not (a.z0 < b.z1 - 1e-6 and b.z0 < a.z1 - 1e-6):
                continue
            c = clearance
            if p.cls in STRUCTURAL and q.cls in STRUCTURAL:
                c = 0.0
            elif p.cls in MARKERS or q.cls in MARKERS:
                c = MARKER_CLEARANCE
            gx = max(a.x0 - b.x1, b.x0 - a.x1)
            gy = max(a.y0 - b.y1, b.y0 - a.y1)
            assert gx >= c - 1e-6 or gy >= c - 1e-6, (p.object_id, q.object_id, gx, gy)


def _on_surface(p, pt, tol=1e-3):
    b = placement_box(p)
    inside = b.x0 - tol <= pt.x <= b.x1 + tol and b.y0 - tol <= pt.y <= b.y1 + tol and b.z0 - tol <= pt.z <= b.z1 + tol
    on_face = min(abs(pt.x - b.x0), abs(pt.x - b.x1), abs(pt.y - b.y0), abs(pt.y - b.y1), abs(pt.z - b.z0),
                  abs(pt.z - b.z1)) < tol
    return inside and on_face


def _orthogonal(points):
    return all(sum(1 for c in "xyz" if abs(getattr(a, c) - getattr(b, c)) > 1e-6) <= 1
               for a, b in zip(points, points[1:]))


def _seg_dist(p1, q1, p2, q2) -> float:
    """Exact closest distance between two 3D segments (independent of the router's code)."""
    import numpy as np

    p1, q1, p2, q2 = (np.asarray(x, dtype=float) for x in (p1, q1, p2, q2))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = d1 @ r
        if e < 1e-12:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def _rendered_bounds(layout, doc):
    """World AABB of every rendered object, from the scene meshes (stems, actuators, vertical-run
    orientation included) -- not from the router's own bookkeeping."""
    import numpy as np
    from shippid.model3d.scene import scene_items

    out = {}
    for it in scene_items(layout, doc, outline=False):
        if it.kind != "object":
            continue
        pts = np.vstack([(np.c_[np.asarray(p.mesh.vertices), np.ones(len(p.mesh.vertices))] @ it.transform.T)[:, :3]
                         for p in it.parts])
        out[it.extras["id"]] = (pts.min(axis=0), pts.max(axis=0))
    return out


def _layout_clashes(layout, doc, room, tol=0.02):
    """Clashes in the rendered model: pipes through inline bodies (bounds of the mesh vs the pipe
    cylinder's box; a pipe's own end objects excepted), pipes closer than the sum of their radii
    (exact segment distance; pairs sharing a node excepted), overlapping bodies of objects not
    joined by a pipe (solver equipment among themselves excepted) and bodies outside the room."""
    import numpy as np

    pipes = {p.id: p for p in doc.pipes}
    bounds = _rendered_bounds(layout, doc)
    router_ids = sorted(p.object_id for p in layout.placements if (p.rationale or "").startswith("[router]"))
    linked = {frozenset((p.start.object_id, p.end.object_id)) for p in doc.pipes
              if p.start.object_id and p.end.object_id}
    out = {"pipe_item": set(), "pipe_pipe": set(), "body_overlap": set(), "outside": set()}
    segs = [(r.pipe_id, a, b, r.diameter / 2) for r in layout.routes
            for a, b in zip([v.as_tuple() for v in r.points], [v.as_tuple() for v in r.points][1:])]
    S_lo = np.array([np.minimum(a, b) for _, a, b, _ in segs]).reshape(-1, 3)
    S_hi = np.array([np.maximum(a, b) for _, a, b, _ in segs]).reshape(-1, 3)
    S_r = np.array([r for *_, r in segs])
    I_lo = np.array([bounds[o][0] for o in router_ids]).reshape(-1, 3)
    I_hi = np.array([bounds[o][1] for o in router_ids]).reshape(-1, 3)
    for k, (pid, a, b, rad) in enumerate(segs):  # pipe cylinder box vs body bounds
        ov = np.all(np.minimum(S_hi[k] + rad, I_hi) - np.maximum(S_lo[k] - rad, I_lo) > tol, axis=1)
        ends = {pipes[pid].start.object_id, pipes[pid].end.object_id}
        out["pipe_item"] |= {(pid, router_ids[i]) for i in np.nonzero(ov)[0] if router_ids[i] not in ends}
    nodes = {pid: {p.start.node_id, p.end.node_id} - {None} for pid, p in pipes.items()}
    for k, (pa, a0, a1, ra) in enumerate(segs):  # box distance prefilter, exact distance to confirm
        gap = np.maximum(0.0, np.maximum(S_lo[k + 1:] - S_hi[k], S_lo[k] - S_hi[k + 1:]))
        for j in np.nonzero(np.sqrt((gap ** 2).sum(axis=1)) < ra + S_r[k + 1:])[0]:
            pb, b0, b1, rb = segs[k + 1 + j]
            if pa != pb and not nodes[pa] & nodes[pb] and _seg_dist(a0, a1, b0, b1) < ra + rb - 1e-3:
                out["pipe_pipe"].add(tuple(sorted((pa, pb))))
    ids = sorted(bounds)
    A_lo = np.array([bounds[o][0] for o in ids])
    A_hi = np.array([bounds[o][1] for o in ids])
    rset = set(router_ids)
    for i, a in enumerate(ids):
        ov = np.all(np.minimum(A_hi[i], A_hi[i + 1:]) - np.maximum(A_lo[i], A_lo[i + 1:]) > tol, axis=1)
        for j in np.nonzero(ov)[0]:
            b = ids[i + 1 + j]
            if (a in rset or b in rset) and frozenset((a, b)) not in linked:
                out["body_overlap"].add((a, b))
    rb_ = room.bounds
    for oid in router_ids:
        lo, hi = bounds[oid]
        if (lo < np.array([rb_.x0, rb_.y0, rb_.z0]) - 1e-3).any() or (hi > np.array([rb_.x1, rb_.y1, rb_.z1]) + 1e-3).any():
            out["outside"].add(oid)
    return {k: sorted(v) for k, v in out.items()}


def _router_counts(notes) -> dict:
    note = next(n for n in notes if n.startswith("router:"))
    keys = ("segments through equipment", "pipe/inline-item clashes", "pipe-pipe clashes", "inline item overlaps",
            "items outside the room", "soft obstacles", "Manhattan fallback")
    return {k: int(re.search(re.escape(k) + r" (\d+)", note).group(1)) for k in keys}


# =========================================================================== zones / plan schema


def test_engine_room_per_ship_type_and_overrides(tmp_path):
    vlcc = engine_room_for(DrawingInfo(ship_type="300K VLCC"))
    box = engine_room_for(DrawingInfo(ship_type="14,000 TEU CONTAINER CARRIER"))
    generic = engine_room_for(DrawingInfo())
    assert vlcc.ship_type == "vlcc" and box.ship_type == "container" and generic.ship_type == "generic"
    assert vlcc.length > box.length > generic.length - 5
    assert vlcc.deck_z("tank_top") > generic.deck_z("tank_top")
    assert [d.name for d in vlcc.to_decks()] == ["tank_top", "floor", "3rd_deck", "2nd_deck", "upper_deck"]
    # bands partition the zone
    b = vlcc.band("tank_top", "port", "aft")
    assert b.y1 == pytest.approx(vlcc.breadth / 2) and b.x0 == 0 and b.x1 == pytest.approx(vlcc.length / 3)
    assert vlcc.zone("double_bottom").box.z1 == vlcc.deck_z("tank_top")
    # overrides: mapping, DrawingInfo.extra and YAML
    o = engine_room_for(DrawingInfo(ship_type="VLCC"), {"length": 30, "decks": {"floor": 5.0}})
    assert o.length == 30 and o.deck_z("floor") == 5.0 and o.source == "override"
    e = engine_room_for(DrawingInfo(extra={"engine_room": {"breadth": 22.0}}))
    assert e.breadth == 22.0
    y = tmp_path / "er.yaml"
    y.write_text("engine_room:\n  length: 28\n  clearance: 0.8\n", encoding="utf-8")
    assert load_engine_room(y).length == 28 and load_engine_room(y).clearance == 0.8


def test_plan_item_normalisation_and_checks():
    it = PlanItemDraft.model_validate({"object_id": "o1", "zone": "Tank Top", "side": "STBD", "longitudinal": "fwd",
                                       "relation": "next to", "orientation_deg": 85, "size_m": [1, 2],
                                       "deck": None, "anchor_id": "null", "max_m": -3})
    assert (it.zone, it.side, it.longitudinal, it.relation, it.orientation_deg) == \
        ("tank_top", "starboard", "fore", "adjacent_to", 90)
    assert it.size_m is None and it.anchor_id is None and it.max_m is None
    with pytest.raises(ValidationError):
        PlanItemDraft.model_validate({"object_id": "o1", "zone": "galley"})
    items = [PlanItemDraft(object_id="o1", zone="floor"),
             PlanItemDraft(object_id="o2", zone="floor", anchor_id="o3", relation="near"),
             PlanItemDraft(object_id="o3", zone="tank_top", deck="2nd_deck"),
             PlanItemDraft(object_id="o1", zone="floor"),
             PlanItemDraft(object_id="o9", zone="floor")]
    problems = "\n".join(check_plan(items, ["o1", "o2", "o3", "o4"], ["o1", "o2", "o3", "o4"]))
    assert "unknown object_id 'o9'" in problems and "duplicate object_id 'o1'" in problems
    assert "anchor 'o3' is later in the list" in problems and "not valid for zone 'tank_top'" in problems
    assert "missing items" in problems and "o4" in problems


# =========================================================================== heuristic + solver


@pytest.mark.parametrize("name", list(DOCS))
def test_heuristic_plan_and_solver(name):
    doc = DOCS[name]()
    room = engine_room_for(doc.info)
    plan = heuristic_plan(doc, room)
    planned = {o.id for o in doc.objects if is_planned(o)}
    assert {it.object_id for it in plan.items} == planned
    assert plan.planner == "heuristic" and all(it.source == "heuristic" for it in plan.items)
    seen = set()
    for it in plan.items:  # anchors always refer to earlier items
        assert it.anchor_id is None or it.anchor_id in seen
        seen.add(it.object_id)
    res = solve_detailed(plan, doc, room)
    assert {p.object_id for p in res.placements} == planned
    assert all(d.status == "ok" for d in res.diagnostics.values()), res.issues()
    _check_no_overlap(res.placements, room)
    by_id = {p.object_id: p for p in res.placements}
    objs = doc.object_by_id()
    # every pipe end on a planned object has a nozzle on the box surface, reachable from outside
    for pipe in doc.pipes:
        for e in (pipe.start, pipe.end):
            if e.object_id in by_id:
                p = by_id[e.object_id]
                nz = p.nozzles[e.port]
                assert _on_surface(p, nz), (p.object_id, e.port)
                n = nozzle_normal(p, nz)
                probe = (nz.x + 0.3 * n[0], nz.y + 0.3 * n[1], nz.z + 0.3 * n[2])
                assert room.bounds.contains(*probe), (p.object_id, e.port)
                for q in res.placements:
                    if q.object_id != p.object_id:
                        assert not placement_box(q).expand(-0.01).contains(*probe), (p.object_id, q.object_id)
    # domain rules
    for p in res.placements:
        cls = objs[p.object_id].cls
        if cls in (S.PUMP_CENTRIFUGAL, S.PUMP_SCREW):
            assert p.deck == "tank_top"
        if cls == S.SEA_CHEST:
            assert p.position.z == 0.0 and abs(abs(p.position.y) + p.size.y / 2 - room.breadth / 2) < 1e-6
        if cls == S.OVERBOARD:
            assert abs(abs(p.position.y) + p.size.y / 2 - room.breadth / 2) < 1e-6
    if name == "ballast":  # parallel pumps side by side, W.B. tanks remote -> forward double bottom
        pumps = [by_id[o.id] for o in doc.objects if o.cls == S.PUMP_CENTRIFUGAL]
        a, b = placement_box(pumps[0]), placement_box(pumps[1])
        assert max(a.x0 - b.x1, b.x0 - a.x1, a.y0 - b.y1, b.y0 - a.y1) <= 1.6 + 1e-6
        for o in doc.objects:
            if o.cls == S.TANK:
                assert by_id[o.id].deck == "base_line" and by_id[o.id].position.x > 2 * room.length / 3
                assert (by_id[o.id].position.y > 0) == ("(P)" in o.attributes["name"])


def test_solver_crowded_band_relocates_and_reports():
    doc = ballast_doc()
    room = engine_room_for(doc.info)
    plan = heuristic_plan(doc, room)
    # cram everything into one small band with oversized boxes
    crowded = LayoutPlan(items=[it.model_copy(update={"zone": "floor", "side": "port", "longitudinal": "aft",
                                                      "size_m": [6.0, 6.0, 2.0], "anchor_id": None,
                                                      "relation": "none"}) for it in plan.items])
    res = solve_detailed(crowded, doc, room)
    statuses = {d.status for d in res.diagnostics.values()}
    assert "relocated" in statuses
    assert any("no free space" in m for m in res.issues())
    _check_no_overlap(res.placements, room)
    placements, notes = solve(crowded, doc, room)
    assert len(placements) == len(plan.items) and notes


def _probe_blockers(placements, room, steps=(0.1, 0.25, 0.4)):
    """(object, nozzle, blocker) for every nozzle whose outside is not free (room / other equipment)."""
    out = []
    for p in placements:
        for k, nz in p.nozzles.items():
            n = nozzle_normal(p, nz)
            for d in steps:
                q = (nz.x + d * n[0], nz.y + d * n[1], nz.z + d * n[2])
                hit = [o.object_id for o in placements if o.object_id != p.object_id
                       and placement_box(o).expand(-0.01).contains(*q)]
                if hit or not room.bounds.contains(*q):
                    out.append((p.object_id, k, (hit or ["room"])[0]))
                    break
    return out


def test_solver_nozzles_reachable_on_large_drawing():
    """Regression: on the 300-object drawing the LO sump tank's top nozzle sat under a screw pump and
    the pipe climbed through the pump; every nozzle's outside must be free (fixed ports included),
    covering items are moved and the move is reported."""
    doc = large_doc(300)
    room = engine_room_for(doc.info)
    res = solve_detailed(heuristic_plan(doc, room), doc, room)
    assert _probe_blockers(res.placements, room) == []
    assert not any("is covered" in i for i in res.issues())
    assert any("not covered by room" in i or "accessible" in i for i in res.issues())  # relocations, reported


def _lo_doc() -> PIDDocument:
    b = DocBuilder("LUB. OIL SYSTEM", SystemType.LUBE_OIL)
    t = b.obj(S.TANK, 200, 500, name="M/E L.O. SUMP TK", scale=30)
    pump = b.obj(S.PUMP_CENTRIFUGAL, 600, 500, tag="LO-P001", name="NO.1 L.O. PUMP")
    b.pipe(b.bport(t, 245, 500), (pump, "suction"), "150A-LO-001", "150A", Medium.LUBE_OIL)
    return b.doc


def test_solver_keeps_a_covered_tank_nozzle_reachable():
    """A pump planned right on top of a double-bottom tank covers its whole top: the nozzle goes to
    the upper part of a side face (not under the pump), and the pipe does not climb through the pump."""
    doc = _lo_doc()
    room = engine_room_for(doc.info)
    plan = LayoutPlan(items=[
        PlanItem(object_id="o1", zone="double_bottom", side="center", longitudinal="mid"),
        PlanItem(object_id="o2", zone="tank_top", side="center", longitudinal="mid", anchor_id="o1",
                 relation="above", size_m=[5.0, 4.0, 1.0]),
    ])
    res = solve_detailed(plan, doc, room)
    by_id = {p.object_id: p for p in res.placements}
    tank, pump = placement_box(by_id["o1"]), placement_box(by_id["o2"])
    assert pump.x0 <= tank.x0 and pump.x1 >= tank.x1 and pump.y0 <= tank.y0 and pump.y1 >= tank.y1  # covered
    nz = by_id["o1"].nozzles["n0"]
    assert nz.z < tank.z1 - 1e-6 and _probe_blockers(res.placements, room) == []
    routes, inline, notes = route(doc, res.placements, room, plan.routing)
    pts = [v.as_tuple() for v in routes[0].points]
    assert not any(segment_hits_box(a, b_, pump) for a, b_ in zip(pts, pts[1:]))
    assert _router_counts(notes)["segments through equipment"] == 0


def test_solver_reports_an_unreachable_fixed_nozzle_for_repair():
    """A pump against the ship side turned so that its suction flange faces the shell cannot be fixed
    by moving it along the shell: the covered nozzle is reported in the issues fed back to Gemma."""
    doc = _lo_doc()
    room = engine_room_for(doc.info)
    plan = LayoutPlan(items=[
        PlanItem(object_id="o1", zone="double_bottom", side="center", longitudinal="mid"),
        PlanItem(object_id="o2", zone="ship_side_port", deck="tank_top", orientation_deg=270),
    ])
    res = solve_detailed(plan, doc, room)
    pump = next(p for p in res.placements if p.object_id == "o2")
    assert abs(placement_box(pump).y1 - room.breadth / 2) < 1e-6  # against the port shell
    issues = res.issues()
    assert any(i.startswith("o2: nozzle suction is covered by the room boundary") for i in issues), issues
    assert any("nozzle(s) without free access" in n for n in res.notes)


def test_compute_nozzles_reports_a_fixed_port_facing_the_wall():
    doc = ballast_doc()
    room = engine_room_for(doc.info)
    pump = next(o for o in doc.objects if o.cls == S.PUMP_CENTRIFUGAL)
    from shippid.model import Placement3D, Vec3
    from shippid.model3d.solver import compute_nozzles_detailed

    pl = Placement3D(object_id=pump.id, cls=pump.cls, position=Vec3(x=0.4, y=0.0, z=room.deck_z("tank_top")),
                     size=Vec3(x=0.8, y=0.8, z=1.6), deck="tank_top")  # suction flange on the -x face, at x = 0
    noz, blocked = compute_nozzles_detailed(doc, pump, pl, {pump.id: pl}, room)
    assert "suction" in noz and any(b.key == "suction" and b.blocker == "room" for b in blocked)


# =========================================================================== router


@pytest.mark.parametrize("name", list(DOCS))
def test_router_orthogonal_routes_per_pipe_and_inline_items(name):
    doc = DOCS[name]()
    room = engine_room_for(doc.info)
    plan = heuristic_plan(doc, room)
    placements, _ = solve(plan, doc, room)
    routes, inline, notes = route(doc, placements, room, plan.routing)
    process = [p for p in doc.pipes if p.kind != PipeKind.SIGNAL]
    assert [r.pipe_id for r in routes] == [p.id for p in process]
    assert not any("Manhattan fallback route" in n for n in notes)
    all_pl = {p.object_id: p for p in placements + inline}
    assert set(all_pl) == {o.id for o in doc.objects}
    boxes = [placement_box(p) for p in placements]
    pipes = {p.id: p for p in doc.pipes}
    for r in routes:
        assert len(r.points) >= 2 and _orthogonal(r.points), r.pipe_id
        pipe = pipes[r.pipe_id]
        assert r.medium == pipe.medium and r.line_number == pipe.line_number
        if pipe.size:
            from shippid.catalog import pipe_od_m
            assert r.diameter == pytest.approx(pipe_od_m(pipe.size), abs=1e-4)
        # sample points every 0.1 m: never inside an equipment box
        for a, b in zip(r.points, r.points[1:]):
            pa, pb = a.as_tuple(), b.as_tuple()
            n = max(1, int(math.dist(pa, pb) / 0.1))
            for t in range(n + 1):
                q = tuple(pa[i] + (pb[i] - pa[i]) * t / n for i in range(3))
                for bx in boxes:
                    assert not bx.expand(-0.02).contains(*q), (r.pipe_id, q)
            assert not any(segment_hits_box(pa, pb, bx) for bx in boxes)
        # route ends at the nozzles of the objects it connects
        for e, end in ((pipe.start, r.points[0]), (pipe.end, r.points[-1])):
            if e.object_id:
                nz = all_pl[e.object_id].nozzles[e.port]
                assert math.dist(nz.as_tuple(), end.as_tuple()) < 2e-3, (r.pipe_id, e.object_id, e.port)
    # inline valves sit on their route: in/out nozzles on the pipe ends, body along the segment
    objs = doc.object_by_id()
    for o in doc.objects:
        if o.category == Category.VALVE:
            p = all_pl[o.id]
            assert (p.rationale or "").startswith("[router]")
            assert {"in", "out"} <= set(p.nozzles) or o.cls == S.SAFETY_VALVE or len(p.nozzles) >= 2
            a, b = p.nozzles["in"].as_tuple(), p.nozzles["out"].as_tuple()
            assert math.dist(a, b) > 0.05 and sum(1 for i in range(3) if abs(a[i] - b[i]) > 1e-6) == 1
    assert objs  # documents are non-trivial
    # the rendered model is clash-free: inline bodies (stems / actuators included) are obstacles for
    # the pipes routed after them, pipes keep clear of each other, bodies stay inside the room
    # (regression: ballast p3 300A-BA-002 ran through the hydraulic actuator of BA-V012)
    lo, hi = room.bounds_vec()
    layout = Layout3D(planner="heuristic", decks=room.to_decks(), bounds_min=lo, bounds_max=hi,
                      placements=placements + inline, routes=routes)
    assert _layout_clashes(layout, doc, room) == {"pipe_item": [], "pipe_pipe": [], "body_overlap": [],
                                                  "outside": []}
    counts = _router_counts(notes)  # the router's own audit agrees
    assert all(v == 0 for v in counts.values()), counts



def test_inline_body_clamped_to_room():
    """A 300A valve on a pipe running 0.25 m from the ship side would poke 0.06 m through the shell
    (reviewer: o144 y1 18.06 > 18.0); its body is reduced until it fits."""
    from types import SimpleNamespace
    from shippid.model3d.router import _clamp_to_room, _make_inline, item_box

    doc = ballast_doc()
    room = engine_room_for(doc.info)
    valve = next(o for o in doc.objects if o.cls == S.BUTTERFLY_VALVE)
    y = room.breadth / 2 - 0.25
    ctx = SimpleNamespace(room=room)
    size = (0.31, 0.82, 1.03)
    make = lambda sz: _make_inline(ctx, valve, sz, (5.0, y, 3.0), (5.31, y, 3.0), "in", "out", False, "test")
    assert item_box(make(size), valve).y1 > room.bounds.y1 + 0.05  # the nominal body pokes out
    pl, box, clamped = _clamp_to_room(ctx, valve, size, make)
    assert clamped and box.y1 <= room.bounds.y1 + 1e-3 and pl.size.y < size[1]
    assert box == item_box(pl, valve)


# =========================================================================== Gemma planner (mock / HTTP)

_ZONE = {"tank": "double_bottom", "pump_centrifugal": "tank_top", "pump_screw": "tank_top", "sea_chest": "bottom",
         "overboard": "ship_side_starboard"}


def _digest(messages) -> dict:
    user = next(m for m in messages if m.role == "user")
    text = user.parts()[0] if hasattr(user, "parts") else user
    return json.loads(text.split("\n\n", 1)[1].split("\n", 1)[0])


def _rule_plan(d: dict) -> dict:
    """What a sensible model would answer for the digest (deterministic stand-in for tests)."""
    items, prev_pump = [], None
    for k, o in enumerate(o for o in d["objects"] if o["required"]):
        it = {"object_id": o["id"], "zone": _ZONE.get(o["class"], "floor"), "deck": None,
              "side": ("port", "starboard")[k % 2], "longitudinal": ("aft", "mid", "fore")[k % 3],
              "anchor_id": None, "relation": "none", "orientation_deg": 0, "size_m": None, "group": None,
              "rationale": f"{o['class']} by rule"}
        if o["class"].startswith("pump") and prev_pump:
            it.update(anchor_id=prev_pump, relation="adjacent_to", group="pumps", side=items[-1]["side"],
                      longitudinal=items[-1]["longitudinal"])
        if o["class"].startswith("pump"):
            prev_pump = o["id"]
        items.append(it)
    lines = d["line_numbers"][:2]
    return {"items": items, "notes": ["rule-based stand-in"],
            "routing": {"elevation_bands": [{"target": m, "deck": "tank_top", "height_m": 1.1} for m in d["media"]],
                        "priority": lines}}


def _bad_plan(d: dict) -> dict:
    good = _rule_plan(d)["items"]
    items = [dict(it, zone="tank_top", side="port", longitudinal="aft", size_m=[9.0, 9.0, 1.8], anchor_id=None,
                  relation="none") for it in good]
    items.insert(0, dict(items[0], object_id="o999"))
    items.append(dict(items[1]))  # duplicate
    items[2] = dict(items[2], anchor_id=items[-2]["object_id"], relation="near")  # anchor to a later item
    return {"items": items, "routing": {"elevation_bands": [], "priority": []}}


def test_gemma_planner_valid_plan_mock_backend():
    doc = ballast_doc()
    room = engine_room_for(doc.info)
    backend = MockBackend(lambda m, s, n: _rule_plan(_digest(m)))
    client = LLMClient(backend)
    plan, usage = GemmaLayoutPlanner(client).plan(doc, room)
    required = {o.id for o in doc.objects if is_planned(o)}
    assert {it.object_id for it in plan.items} == required
    # provenance: the test backend is labelled "mock", never "gemma"
    assert plan.planner == "mock" and all(it.source == "mock" for it in plan.items)
    assert usage.backend == "mock" and usage.calls == 1 and not usage.errors
    msgs, schema, name = backend.calls[0]
    assert name == "LayoutPlanDraft" and schema is not None and "items" in schema["properties"]
    system = msgs[0].text()
    assert msgs[0].role == "system" and "Never output coordinates" in system and "sea chest" in system.lower()
    digest = _digest(msgs)
    assert all("size_prior_m" in o and "connections" in o for o in digest["objects"])
    assert {o["id"] for o in digest["objects"] if not o["required"]} == {"o2"}  # strainer is optional
    pumps = [o for o in digest["objects"] if o["class"] == "pump_centrifugal"]
    assert any(c["to"] == "o2" and c["flow"] == "in" for c in pumps[0]["connections"])  # via the strainer
    strainer = next(o for o in digest["objects"] if o["id"] == "o2")
    assert any(c["to"] == "o1" and c["flow"] == "in" for c in strainer["connections"])  # ... from the sea chest
    assert plan.routing.priority[:2] == digest["line_numbers"][:2]
    layout = build_layout(doc, client=client)
    assert layout.planner == "mock" and layout.llm.backend == "mock"
    assert all(p.rationale.startswith(("[mock]", "[router]")) for p in layout.placements)


def test_gemma_planner_repair_round_on_unknown_ids_and_overlaps():
    doc = ballast_doc()
    room = engine_room_for(doc.info)
    answers = []

    def responder(m, s, n):
        d = _digest(m)
        answers.append(len(m))
        return _bad_plan(d) if len(answers) == 1 else _rule_plan(d)

    backend = MockBackend(responder)
    planner = GemmaLayoutPlanner(LLMClient(backend), max_rounds=3)
    plan, usage = planner.plan(doc, room)
    assert usage.calls == 2 and len(planner.log) == 2
    first = "\n".join(planner.log[0]["problems"] + planner.log[0]["issues"])
    assert "unknown object_id 'o999'" in first and "duplicate object_id" in first and "later in the list" in first
    assert "no free space" in first or "solver moved it" in first
    assert not planner.log[1]["problems"] and not planner.log[1]["issues"]
    feedback = backend.calls[1][0]
    assert feedback[-2].role == "assistant" and "The solver checked your plan" in feedback[-1].text()
    assert plan.planner == "mock" and {it.object_id for it in plan.items} == {o.id for o in doc.objects
                                                                               if is_planned(o)}


def test_gemma_planner_missing_items_filled_by_heuristic():
    doc = fo_transfer_doc()
    room = engine_room_for(doc.info)

    def responder(m, s, n):
        out = _rule_plan(_digest(m))
        out["items"] = [it for it in out["items"] if it["object_id"] not in ("o13", "o15")]
        return out

    planner = GemmaLayoutPlanner(LLMClient(MockBackend(responder)), max_rounds=1)
    plan, usage = planner.plan(doc, room)
    src = {it.object_id: it.source for it in plan.items}
    assert src["o13"] == src["o15"] == "heuristic" and src["o6"] == "mock"
    assert plan.planner == "mock" and usage.calls == 2  # one repair round asked for the missing items
    assert any("heuristic placement for 2 item(s)" in n for n in plan.notes)


def test_gemma_planner_garbage_falls_back_to_heuristic():
    doc = ballast_doc()
    room = engine_room_for(doc.info)
    client = LLMClient(MockBackend(lambda m, s, n: "Sorry, I can not help with that."), max_attempts=2)
    plan, usage = GemmaLayoutPlanner(client).plan(doc, room)
    assert plan.planner == "heuristic" and all(it.source == "heuristic" for it in plan.items)
    assert usage.calls == 2 and usage.errors and any("no valid plan" in e for e in usage.errors)
    layout = build_layout(doc, client=LLMClient(MockBackend(lambda m, s, n: "{not json"), max_attempts=1))
    assert layout.planner == "heuristic" and layout.llm.errors
    assert len(layout.placements) == len(doc.objects)


def test_gemma_planner_chunks_large_drawings():
    doc = large_doc(60)
    room = engine_room_for(doc.info)
    backend = MockBackend(lambda m, s, n: _rule_plan(_digest(m)))
    planner = GemmaLayoutPlanner(LLMClient(backend), chunk_size=8, max_rounds=0)
    plan, usage = planner.plan(doc, room)
    required = [o.id for o in doc.objects if is_planned(o)]
    assert sorted(it.object_id for it in plan.items) == sorted(required)
    assert usage.calls >= 3 and len(backend.calls) == usage.calls
    later = _digest(backend.calls[-1][0])
    assert later["already_placed"] and len(later["objects"]) <= 8


def test_gemma_planner_failing_repair_round_keeps_the_best_plan():
    """Regression: an LLMError in a repair round (here: unparseable output after LLMClient's own
    attempts) used to discard the valid round-1 plan and fall back to the heuristic."""
    doc = ballast_doc()
    room = engine_room_for(doc.info)
    calls = []

    def responder(m, s, n):
        calls.append(1)
        if len(calls) == 1:
            plan = _rule_plan(_digest(m))
            plan["items"].append(dict(plan["items"][0], object_id="o999"))  # one fixable problem
            return plan
        return "sorry, I cannot help with that"

    planner = GemmaLayoutPlanner(LLMClient(MockBackend(responder), max_attempts=1), max_rounds=3)
    plan, usage = planner.plan(doc, room)
    required = {o.id for o in doc.objects if is_planned(o)}
    assert plan.planner == "mock" and {it.object_id for it in plan.items} == required
    assert all(it.source == "mock" for it in plan.items)
    assert usage.calls == 2 and any("repair round 2 failed" in e for e in usage.errors)
    assert any("kept the plan of round 1" in n for n in plan.notes)
    assert not any("no valid plan" in n for n in plan.notes)
    assert planner.log[-1]["error"] and planner.log[0]["problems"]


def test_gemma_planner_http_400_in_repair_round_keeps_the_best_plan():
    """vLLM answers a prompt that overflows --max-model-len with HTTP 400 (non-retryable): the
    planner keeps the round-1 plan instead of throwing it away."""
    httpx = pytest.importorskip("httpx")
    oc = pytest.importorskip("shippid.llm.openai_compat")
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) > 1:
            return httpx.Response(400, json={
                "object": "error", "type": "BadRequestError", "param": None, "code": 400,
                "message": "This model's maximum context length is 32768 tokens. However, you requested 33012 "
                           "tokens (21012 in the messages, 12000 in the completion). Please reduce the length of "
                           "the messages or completion."})
        user = next(m for m in body["messages"] if m["role"] == "user")["content"]
        plan = _rule_plan(json.loads(user.split("\n\n", 1)[1].split("\n", 1)[0]))
        plan["items"].append(dict(plan["items"][0], object_id="o999"))
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant",
                                                                          "content": json.dumps(plan)}}],
            "usage": {"prompt_tokens": 2100, "completion_tokens": 640, "total_tokens": 2740}})

    backend = oc.OpenAICompatBackend("google/gemma-4-31B-it", "http://gpu:8000/v1",
                                     transport=httpx.MockTransport(handler), backoff=0)
    doc = ballast_doc()
    plan, usage = GemmaLayoutPlanner(LLMClient(backend)).plan(doc, engine_room_for(doc.info))
    assert len(seen) == 2 and plan.planner == "gemma" and all(it.source == "gemma" for it in plan.items)
    assert any("HTTP 400" in e and "repair round 2 failed" in e for e in usage.errors)


def test_gemma_planner_partial_repair_answer_is_merged():
    """Regression: a repair answer with only the corrected item scored better than a complete plan
    with soft issues, so 6 of 7 items fell back to the heuristic. It is now a patch of the plan."""
    doc = ballast_doc()
    room = engine_room_for(doc.info)
    calls = []

    def responder(m, s, n):
        calls.append(1)
        plan = _rule_plan(_digest(m))
        if len(calls) == 1:  # complete and valid, but three relations the solver cannot meet
            first = plan["items"][0]["object_id"]
            for it in plan["items"][1:4]:
                it.update(anchor_id=first, relation="above")
            return plan
        plan["items"] = plan["items"][1:2]  # the repair answer lists only the item it changed
        return plan

    planner = GemmaLayoutPlanner(LLMClient(MockBackend(responder)), max_rounds=2)
    plan, usage = planner.plan(doc, room)
    required = {o.id for o in doc.objects if is_planned(o)}
    assert {it.object_id for it in plan.items} == required and all(it.source == "mock" for it in plan.items)
    assert planner.log[1]["merged"] and planner.log[1]["score"][0] == 0  # nothing missing after the merge
    assert usage.calls == 3


class _RecordingMock(MockBackend):
    """MockBackend that also records the max_tokens of each call."""

    def __init__(self, responder):
        super().__init__(responder)
        self.max_tokens: list[int] = []

    def complete(self, messages, **kw):
        self.max_tokens.append(kw["max_tokens"])
        return super().complete(messages, **kw)


def test_gemma_planner_repair_context_is_bounded_and_fits_the_context():
    """Regression: every repair round resent the whole conversation (41k -> 80k characters for a
    49-item chunk) with max_tokens 12000, past vLLM's 32k context. Repair prompts are now system +
    request + latest plan + latest feedback, and max_tokens is sized to what the context leaves."""
    from shippid.model3d.gemma_planner import estimate_tokens

    doc = large_doc(150)
    room = engine_room_for(doc.info)

    def responder(m, s, n):
        plan = _rule_plan(_digest(m))
        first = plan["items"][0]["object_id"]
        for it in plan["items"][1:]:
            it.update(anchor_id=first, relation="above")  # every round has solver issues
        return plan

    backend = _RecordingMock(responder)
    planner = GemmaLayoutPlanner(LLMClient(backend), max_rounds=3)
    plan, usage = planner.plan(doc, room)
    assert usage.calls == 4 and plan.planner == "mock"
    sizes = []
    for (msgs, _, _), mt in zip(backend.calls, backend.max_tokens):
        assert len(msgs) <= 4 and msgs[0].role == "system"
        assert msgs[1].parts()[0] == backend.calls[0][0][1].parts()[0]  # the original request, unchanged
        prompt = sum(estimate_tokens(m.text()) + 8 for m in msgs)
        assert prompt + mt <= 32768, (prompt, mt)
        sizes.append(sum(len(m.text()) for m in msgs))
    assert sizes[3] <= sizes[1] * 1.05  # no growth over the repair rounds
    assert all(mt <= 12000 for mt in backend.max_tokens)


def test_gemma_planner_splits_a_chunk_whose_prompt_does_not_fit():
    from shippid.model3d.gemma_planner import estimate_tokens

    doc = large_doc(60)
    room = engine_room_for(doc.info)
    backend = _RecordingMock(lambda m, s, n: _rule_plan(_digest(m)))
    planner = GemmaLayoutPlanner(LLMClient(backend), max_rounds=0, context_tokens=14000)
    plan, usage = planner.plan(doc, room)
    assert any("split in two" in n for n in plan.notes)
    assert sorted(it.object_id for it in plan.items) == sorted(o.id for o in doc.objects if is_planned(o))
    for (msgs, _, _), mt in zip(backend.calls, backend.max_tokens):
        assert sum(estimate_tokens(m.text()) + 8 for m in msgs) + mt <= 14000


def test_gemma_planner_over_openai_compatible_http():
    """vLLM-style /chat/completions emulated with httpx.MockTransport -> planner label 'gemma'."""
    httpx = pytest.importorskip("httpx")
    oc = pytest.importorskip("shippid.llm.openai_compat")
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.url.path == "/v1/chat/completions"
        assert body["model"] == "google/gemma-4-31B-it"
        assert body["response_format"]["type"] == "json_schema"
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        assert body["messages"][0]["role"] == "system"
        user = next(m for m in body["messages"] if m["role"] == "user")["content"]
        plan = _rule_plan(json.loads(user.split("\n\n", 1)[1].split("\n", 1)[0]))
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(plan),
                                     "reasoning": "Pumps go on the tank top near the sea chest."}}],
            "usage": {"prompt_tokens": 2100, "completion_tokens": 640, "total_tokens": 2740}})

    backend = oc.OpenAICompatBackend("google/gemma-4-31B-it", "http://gpu:8000/v1",
                                     transport=httpx.MockTransport(handler), backoff=0)
    doc = ballast_doc()
    plan, usage = GemmaLayoutPlanner(LLMClient(backend), thinking=True).plan(doc, engine_room_for(doc.info))
    assert plan.planner == "gemma" and all(it.source == "gemma" for it in plan.items)
    assert usage.backend == "openai" and usage.prompt_tokens == 2100 and usage.completion_tokens == 640
    assert len(seen) == 1


# =========================================================================== GLB / viewers


def test_glb_export_names_extras_materials(tmp_path):
    doc = ballast_doc()
    layout = build_layout(doc)
    assert layout.planner == "heuristic" and len(layout.placements) == len(doc.objects)
    assert len(layout.routes) == len([p for p in doc.pipes if p.kind != PipeKind.SIGNAL])
    path = export_glb(layout_to_scene(layout, doc), tmp_path / "ballast.glb")
    js = read_glb_json(path)
    names = [n["name"] for n in js["nodes"]]
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names)
    by_id = {n["extras"]["id"]: n for n in js["nodes"] if n.get("extras", {}).get("kind") in ("object", "pipe")}
    assert {o.id for o in doc.objects} <= set(by_id) and {r.pipe_id for r in layout.routes} <= set(by_id)
    pump = by_id["o4"]["extras"]
    assert pump["tag"] == "BA-P001" and pump["class"] == "pump_centrifugal" and pump["system"] == "ballast"
    # human label under its own key: GLTFLoader sets userData.name to the node name before the extras,
    # so a "name" extra would be shadowed by the slug "BA-P001_o4" (reviewer: 명칭 BA-V012_o8)
    assert pump["label"] == "NO.1 BALLAST PUMP" and "name" not in pump
    assert all("name" not in n.get("extras", {}) for n in js["nodes"])
    assert pump["deck"] == "tank_top" and pump["rationale"] and pump["medium"] == "ballast_water"
    assert by_id["p1"]["extras"]["line_number"] == "350A-BA-001" and by_id["p1"]["extras"]["medium"] == "ballast_water"
    groups = [n["extras"] for n in js["nodes"] if n.get("extras", {}).get("kind") == "group"]
    assert {"system", "category", "structure"} <= {g["group_type"] for g in groups}
    # one PBR material per mesh, linear colours, translucent decks
    assert len(js["materials"]) == len(js["meshes"])
    assert all("baseColorFactor" in m["pbrMetallicRoughness"] for m in js["materials"])
    assert any(m.get("alphaMode") == "BLEND" for m in js["materials"])
    assert all("NORMAL" in p["attributes"] for m in js["meshes"] for p in m["primitives"])
    root = js["nodes"][js["scenes"][0]["nodes"][0]]
    assert root["name"] == "engine_room" and root["matrix"][:12] == [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0]
    assert js["scenes"][0]["extras"]["drawing"]["drawing_no"] == "TEST-BALLAST"


def test_viewers_threejs_and_plotly(tmp_path):
    doc = fo_transfer_doc()
    layout = build_layout(doc)
    glb = export_glb(layout_to_scene(layout, doc), tmp_path / "fo.glb")
    page = write_viewer_html(glb, layout, doc, tmp_path / "fo.html").read_text(encoding="utf-8")
    imap = json.loads(re.search(r'<script type="importmap">(.*?)</script>', page, re.S).group(1))
    assert imap["imports"]["three"] == "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js"
    assert imap["imports"]["three/addons/"] == "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"
    b64 = re.search(r'<script id="glb-data" type="application/octet-stream">(.*?)</script>', page, re.S).group(1)
    assert base64.b64decode(b64) == glb.read_bytes()
    data = re.search(r'<script id="viewer-data" type="application/json">(.*?)</script>', page, re.S).group(1)
    data = json.loads(data)
    assert data["planner"] == "heuristic" and data["legend"][0]["medium"] == "fuel_oil"
    assert data["fields"]["tag"] == "태그"
    for snippet in ("GLTFLoader().parseAsync", "OrbitControls", "visibleChain", "setFromObject", "runSearch"):
        assert snippet in page
    out = write_viewer_html(None, layout, doc, tmp_path / "fo_plotly.html", mode="plotly")
    html_text = out.read_text(encoding="utf-8")
    assert not re.search(r"<script[^>]+src=", html_text)  # fully offline: plotly.js embedded
    assert '"type":"mesh3d"' in html_text.replace(" ", "") and '"aspectmode":"data"' in html_text.replace(" ", "")
    with pytest.raises(ValueError):
        write_viewer_html(glb, layout, doc, tmp_path / "x.html", mode="vr")
    assert data["fields"]["label"] == "명칭" and "name" not in data["fields"]
    assert "'label', 'line_number'" in page and "u.tag || u.label || u.id" in page
    n_planned = sum(1 for o in doc.objects if is_planned(o))  # badge counts: plan items vs router-placed
    assert data["sources"] == {"heuristic": n_planned, "router": len(layout.placements) - n_planned}


def test_viewer_survives_script_breaking_strings(tmp_path):
    """Regression: a title with '<!--<script>' switched the HTML parser into the double-escaped
    script state, the viewer-data element swallowed the GLB and the module, nothing loaded."""
    doc = ballast_doc()
    nasty = "BALLAST <!--<script> SYSTEM </script> & \u2028 <b>x</b>"
    doc.info.title = nasty
    layout = build_layout(doc)
    layout.notes.append("model note: </script><script>alert(1)</script>")
    glb = export_glb(layout_to_scene(layout, doc), tmp_path / "b.glb")
    page = write_viewer_html(glb, layout, doc, tmp_path / "b.html").read_text(encoding="utf-8")
    assert len(re.findall(r"<script\b", page)) == 4  # importmap, viewer-data, glb-data, module
    raw = re.search(r'<script id="viewer-data" type="application/json">(.*?)</script>', page, re.S).group(1)
    assert not set(raw) & {"<", ">", "&", "\u2028"}
    data = json.loads(raw)
    assert data["title"] == nasty and data["notes"][-1] == "model note: </script><script>alert(1)</script>"
    b64 = re.search(r'<script id="glb-data" type="application/octet-stream">(.*?)</script>', page, re.S).group(1)
    assert base64.b64decode(b64) == glb.read_bytes()
    assert html_title(page) == f"{nasty} – 3D"


def html_title(page: str) -> str:
    import html as html_mod

    return html_mod.unescape(re.search(r"<title>(.*?)</title>", page, re.S).group(1))


# =========================================================================== scale


def test_large_drawing_performance():
    doc = large_doc(150)
    assert len(doc.objects) >= 150
    t0 = time.time()
    layout = build_layout(doc)
    t_layout = time.time() - t0
    scene = layout_to_scene(layout, doc)
    t_total = time.time() - t0
    assert len(layout.placements) == len(doc.objects)
    assert len(layout.routes) == len([p for p in doc.pipes if p.kind != PipeKind.SIGNAL])
    assert not any("Manhattan fallback route" in n for n in layout.notes)
    counts = _router_counts(layout.notes)
    assert all(v == 0 for v in counts.values()), counts
    t_check = time.time()
    clashes = _layout_clashes(layout, doc, engine_room_for(doc.info))  # independent of the router's audit
    assert clashes == {"pipe_item": [], "pipe_pipe": [], "body_overlap": [], "outside": []}, clashes
    t_check = time.time() - t_check
    assert len(scene.geometry) > 300
    print(f"\n150-object drawing: {len(doc.objects)} objects, {len(doc.pipes)} pipes: layout {t_layout:.2f} s, "
          f"layout + scene {t_total:.2f} s, clash check {t_check:.2f} s")
    assert t_layout < 60


def test_degraded_extraction_output_still_builds(tmp_path):
    """Extractor-like defects: unknown classes, unnamed ports, free pipe ends, signal-only and
    unconnected objects -> every object placed, every process pipe routed, nothing crashes."""
    doc = ballast_doc()
    objs = doc.object_by_id()
    objs["o3"].cls = S.UNKNOWN  # unclassified valve-like symbol (category stays VALVE)
    for p in doc.pipes[:6]:  # ports not located
        p.start.port = None if p.start.object_id else p.start.port
    doc.pipes.append(PipeRun(id="p900", points=[Point(x=1300, y=900), Point(x=1250, y=900), Point(x=1250, y=700)],
                             start=Endpoint(), end=Endpoint(object_id="o18", port=None), line_number="50A-BA-099",
                             size="50A", medium=Medium.BALLAST_WATER))
    doc.objects.append(PIDObject(id="o90", cls=S.INSTRUMENT_PANEL, category=Category.INSTRUMENT, tag="LT-301",
                                 bbox=BBox(x0=990, y0=880, x1=1004, y1=894)))
    doc.pipes.append(PipeRun(id="p901", points=[Point(x=997, y=887), Point(x=900, y=887), Point(x=900, y=330)],
                             start=Endpoint(object_id="o90"), end=Endpoint(object_id="o13"), kind=PipeKind.SIGNAL))
    doc.objects.append(PIDObject(id="o91", cls=S.UNKNOWN, category=Category.EQUIPMENT,
                                 bbox=BBox(x0=600, y0=900, x1=660, y1=950)))
    layout = build_layout(doc)
    assert {p.object_id for p in layout.placements} == {o.id for o in doc.objects}
    process = [p.id for p in doc.pipes if p.kind != PipeKind.SIGNAL]
    assert [r.pipe_id for r in layout.routes] == process
    assert all(_orthogonal(r.points) for r in layout.routes)
    assert any("signal line" in n for n in layout.notes)
    lt = next(p for p in layout.placements if p.object_id == "o90")
    assert "signal-linked" in lt.rationale
    assert _layout_clashes(layout, doc, engine_room_for(doc.info)) == {"pipe_item": [], "pipe_pipe": [],
                                                                       "body_overlap": [], "outside": []}
    js = read_glb_json(export_glb(layout_to_scene(layout, doc), tmp_path / "degraded.glb"))
    ids = {n["extras"]["id"] for n in js["nodes"] if n.get("extras", {}).get("kind") == "object"}
    assert ids == {o.id for o in doc.objects}
