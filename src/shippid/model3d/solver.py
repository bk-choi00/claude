"""Deterministic solver: :class:`LayoutPlan` -> :class:`Placement3D` list.

Items are placed in plan order. For each item the solver enumerates centre positions
on a grid (plus "snug" positions touching already placed boxes at exactly the clearance)
inside the requested zone / band, rejects positions that leave the room or collide with
placed boxes (hard constraints: room bounds and the >= 0.6 m maintenance clearance; tanks
and sea chests may share bulkheads), and scores the rest (soft): relation to the anchor,
distance to the drawing-position hint, group cohesion and short connections. If the band
is full it widens the search (whole zone, then neighbouring zones, then a relaxed
clearance) and records what it did, which the Gemma planner feeds back for repair.

Nozzle positions: fixed symbol ports are mapped from the symbol's local frame onto the
box faces (local +u = +x face, drawing "up" = top face); boundary-port equipment (tanks,
engines, receivers, overboards, connectors) gets each nozzle on the face nearest to the
equipment it connects to (top face for double-bottom tanks and sea chests, searched over the
whole face, then the upper part of the side faces). Every nozzle, fixed ports included, is
probe-checked: the space just outside it must be inside the room and free of other equipment.
A nozzle that stays covered makes the solver move the covering item once (keeping the access
column clear); if that fails it is reported in :meth:`SolveResult.issues` (and so fed back to
the Gemma planner).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..catalog import TYPICAL_DIMENSIONS
from ..model import Category, Endpoint, PIDDocument, PIDObject, Placement3D, SymbolClass, Vec3
from ..symbols import get_symbol
from .heuristic import (_BOILER, _GEN_ENGINE, _HIGH_TANK, _MAIN_ENGINE, Neighbor, drawing_positions,
                        equipment_adjacency, object_labels, process_pipes)
from .plan import LayoutPlan, PlanItem, effective_deck
from .zones import Box3, EngineRoom

S = SymbolClass

STRUCTURAL: frozenset[SymbolClass] = frozenset({S.TANK, S.SEA_CHEST})  # may share bulkheads
MARKERS: frozenset[SymbolClass] = frozenset({S.OFF_PAGE_CONNECTOR, S.OVERBOARD})
MARKER_CLEARANCE = 0.2
BOUNDARY_FACE_CLASSES: frozenset[SymbolClass] = frozenset({S.OVERBOARD, S.OFF_PAGE_CONNECTOR})
_SCALED_BY_PIPE = frozenset({S.PUMP_CENTRIFUGAL, S.PUMP_SCREW, S.HEAT_EXCHANGER, S.PLATE_HEAT_EXCHANGER, S.HEATER,
                             S.DUPLEX_FILTER, S.FILTER, S.STRAINER, S.FLOW_METER, S.BWTS_UNIT, S.PURIFIER,
                             S.COMPRESSOR})
ZONE_FALLBACK: dict[str, tuple[str, ...]] = {
    "double_bottom": ("tank_top", "floor"),
    "bottom": ("double_bottom", "tank_top"),
    "tank_top": ("floor", "middle_platform"),
    "floor": ("tank_top", "middle_platform", "upper_platform"),
    "middle_platform": ("floor", "upper_platform", "tank_top"),
    "upper_platform": ("middle_platform", "floor"),
    "ship_side_port": ("ship_side_starboard", "floor"),
    "ship_side_starboard": ("ship_side_port", "floor"),
}
# (face, tangential fraction, height fraction) overrides of the generic port -> face mapping
_FACE_OVERRIDES: dict[SymbolClass, dict[str, tuple[str, float, float]]] = {
    # vertical in-line centrifugal pump: suction / discharge flanges on the casing, opposite sides
    S.PUMP_CENTRIFUGAL: {"suction": ("-x", 0.0, 0.25), "discharge": ("+x", 0.0, 0.25)},
    # horizontal screw pump: end suction, discharge on top of the pump body (motor on +x)
    S.PUMP_SCREW: {"in": ("-x", 0.0, 0.5), "out": ("top", -0.3, 1.0)},
}


# ---------------------------------------------------------------------------- geometry helpers


def yaw_footprint(size: tuple[float, float, float], yaw_deg: float) -> tuple[float, float]:
    """Axis-aligned footprint (dx, dy) of a box of ``size`` rotated by ``yaw_deg`` about z."""
    a = math.radians(yaw_deg)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return (size[0] * c + size[1] * s, size[0] * s + size[1] * c)


def placement_box(p: Placement3D) -> Box3:
    """World AABB of a placement (position = footprint centre at the base)."""
    fx, fy = yaw_footprint(p.size.as_tuple(), p.yaw_deg)
    return Box3(p.position.x - fx / 2, p.position.y - fy / 2, p.position.z,
                p.position.x + fx / 2, p.position.y + fy / 2, p.position.z + p.size.z)


def local_to_world(p: Placement3D, lx: float, ly: float, lz: float) -> tuple[float, float, float]:
    a = math.radians(p.yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return (p.position.x + c * lx - s * ly, p.position.y + s * lx + c * ly, p.position.z + lz)


def world_to_local(p: Placement3D, x: float, y: float, z: float) -> tuple[float, float, float]:
    a = math.radians(p.yaw_deg)
    c, s = math.cos(a), math.sin(a)
    dx, dy = x - p.position.x, y - p.position.y
    return (c * dx + s * dy, -s * dx + c * dy, z - p.position.z)


def nozzle_normal(p: Placement3D, point: Vec3) -> tuple[float, float, float]:
    """Outward unit normal (world) of the box face a nozzle point lies on."""
    l, w, h = p.size.as_tuple()
    lx, ly, lz = world_to_local(p, point.x, point.y, point.z)
    cands = [(abs(lx - l / 2), (1, 0, 0)), (abs(lx + l / 2), (-1, 0, 0)), (abs(ly - w / 2), (0, 1, 0)),
             (abs(ly + w / 2), (0, -1, 0)), (abs(lz - h), (0, 0, 1)), (abs(lz), (0, 0, -1))]
    _, (nx, ny, nz) = min(cands, key=lambda c: c[0])
    a = math.radians(p.yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return (round(c * nx - s * ny, 9), round(s * nx + c * ny, 9), float(nz))


def port_key(endpoint: Endpoint, pipe_id: str) -> str:
    """Nozzle key for a pipe end on an object: the port name, or ``auto_<pipe>`` if unnamed."""
    return endpoint.port or f"auto_{pipe_id}"


def axis_height(cls: SymbolClass, size: tuple[float, float, float]) -> float:
    """Height of the pipe axis above the base of an inline item."""
    l, w, h = size
    cat = get_symbol(cls).category if cls != S.UNKNOWN else Category.EQUIPMENT
    if cat == Category.INSTRUMENT:
        return 0.0
    if cat == Category.VALVE:
        return min(w, h) / 2
    if cls == S.FILTER:
        return 0.7 * h
    if cls == S.SPECTACLE_FLANGE:
        return 0.35 * h
    return h / 2


def scale_for_od(size: tuple[float, float, float], od: float, exponent: float = 0.7) -> tuple[float, float, float]:
    """Scale an inline item's catalogue size with the pipe OD (reference DN100)."""
    f = max(1.0, od / 0.1143) ** exponent
    return (size[0] * f, size[1] * f, size[2] * f)


# ---------------------------------------------------------------------------- sizing


def object_size(obj: PIDObject, room: EngineRoom, zone: str, label: str = "", dn: Optional[int] = None,
                requested: Optional[Sequence[float]] = None) -> tuple[tuple[float, float, float], str]:
    """Overall (length, width, height) of a planned object and a short note on its origin."""
    cls = obj.cls
    l, w, h = TYPICAL_DIMENSIONS.get(cls, TYPICAL_DIMENSIONS[S.UNKNOWN])
    origin = "catalog"
    up = label.upper()
    if requested:
        l, w, h = (float(requested[0]), float(requested[1]), float(requested[2]))
        origin = "plan"
    else:
        if cls in _SCALED_BY_PIPE and dn:
            f = min(1.5, max(0.75, math.sqrt(dn / 150.0)))
            l, w, h = l * f, w * f, h * f
            origin = f"catalog x{f:.2f} (DN{dn})"
        if cls == S.ENGINE:
            if _BOILER.search(up):
                l, w, h = 4.5, 3.5, 6.0
            elif _GEN_ENGINE.search(up):
                l, w, h = 7.0, 2.2, 3.6
            elif not (_MAIN_ENGINE.search(up) or not up.strip() or "ENGINE" in up):
                l, w, h = 3.0, 2.0, 2.5
        if cls == S.TANK:
            if zone in ("double_bottom", "bottom"):
                l, w, h = (5.0, 4.0, room.deck_z("tank_top")) if "BALLAST" in up or "W.B" in up or "WB" in up \
                    else (4.0, 3.0, room.deck_z("tank_top"))
                origin = "double-bottom height"
            elif _HIGH_TANK.search(up) or zone == "upper_platform":
                l, w, h = 1.6, 1.2, 1.6
            elif zone in ("floor", "middle_platform"):
                above = room.deck_above("floor" if zone == "floor" else "3rd_deck")
                gap = (above[1] if above else room.height) - room.deck_z("floor" if zone == "floor" else "3rd_deck")
                l, w, h = 3.5, 3.0, max(2.0, min(5.0, 0.8 * gap))
                origin = "elevated tank, 0.8 x deck height"
            else:
                gap = room.deck_z("floor") - room.deck_z("tank_top")
                l, w, h = 4.0, 3.0, max(1.5, gap - 0.1)
                origin = "tank-top tank below the floor"
        if cls == S.SEA_CHEST:
            h = min(h, max(0.6, room.deck_z("tank_top") - 0.2))
    l = min(max(l, 0.05), room.length - 0.2)
    w = min(max(w, 0.05), room.breadth - 0.2)
    h = min(max(h, 0.05), room.height - 0.2)
    return (round(l, 3), round(w, 3), round(h, 3)), origin


# ---------------------------------------------------------------------------- diagnostics


@dataclass
class ItemDiag:
    object_id: str
    status: str  # ok | relocated | relaxed | forced | skipped
    requested: str
    actual: str = ""
    relation: str = "none"
    relation_ok: Optional[bool] = None
    relation_detail: str = ""
    message: str = ""
    nozzle_issues: list[str] = field(default_factory=list)


@dataclass
class SolveResult:
    placements: list[Placement3D]
    notes: list[str]
    diagnostics: dict[str, ItemDiag]
    boxes: dict[str, Box3]

    def issues(self, ids: Optional[Sequence[str]] = None) -> list[str]:
        """Feedback lines for the planner (placement problems and unsatisfied relations)."""
        wanted = set(ids) if ids is not None else None
        out: list[str] = []
        for oid, d in self.diagnostics.items():
            if wanted is not None and oid not in wanted:
                continue
            if d.status != "ok":
                out.append(f"{oid}: {d.message}")
            if d.relation_ok is False:
                out.append(f"{oid}: relation {d.relation} not satisfied ({d.relation_detail})")
            out += [f"{oid}: {m}" for m in d.nozzle_issues]
        return out


# ---------------------------------------------------------------------------- solver core


@dataclass
class _Placed:
    oid: str
    box: Box3
    clearance: float
    structural: bool
    marker: bool
    group: Optional[str]


@dataclass
class _Ctx:
    room: EngineRoom
    step: float
    placed: list[_Placed] = field(default_factory=list)
    arr: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    clear: np.ndarray = field(default_factory=lambda: np.zeros(0))
    struct: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    marker: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def add(self, p: _Placed) -> None:
        self.placed.append(p)
        self.arr = np.vstack([self.arr, np.array(p.box.as_tuple())[None, :]])
        self.clear = np.append(self.clear, p.clearance)
        self.struct = np.append(self.struct, p.structural)
        self.marker = np.append(self.marker, p.marker)


def _region(room: EngineRoom, zone: str, side: Optional[str], long_: Optional[str], fx: float, fy: float
            ) -> Optional[tuple[float, float, float, float]]:
    """Feasible range (cx0, cx1, cy0, cy1) of the footprint centre: centre inside the band,
    box inside the zone's horizontal extent (ship-side zones and side sea chests: against the shell)."""
    zb = room.zone(zone).box
    bb = room.band(zone, side, long_)
    cx0, cx1 = max(bb.x0, zb.x0 + fx / 2), min(bb.x1, zb.x1 - fx / 2)
    if zone == "ship_side_port" or (zone == "bottom" and side == "port"):
        cy0 = cy1 = room.breadth / 2 - fy / 2
    elif zone == "ship_side_starboard" or (zone == "bottom" and side == "starboard"):
        cy0 = cy1 = -room.breadth / 2 + fy / 2
    else:
        cy0, cy1 = max(bb.y0, zb.y0 + fy / 2), min(bb.y1, zb.y1 - fy / 2)
    if cx0 > cx1 + 1e-9:
        if fx <= zb.x1 - zb.x0:  # item longer than the band: centre as close to the band as possible
            mid = min(max((bb.x0 + bb.x1) / 2, zb.x0 + fx / 2), zb.x1 - fx / 2)
            cx0 = cx1 = mid
        else:
            return None
    if cy0 > cy1 + 1e-9:
        if fy <= room.breadth:
            mid = min(max((bb.y0 + bb.y1) / 2, -room.breadth / 2 + fy / 2), room.breadth / 2 - fy / 2)
            cy0 = cy1 = mid
        else:
            return None
    return (cx0, cx1, cy0, cy1)


def _grid(a: float, b: float, step: float) -> np.ndarray:
    if b - a < 1e-9:
        return np.array([a])
    n = max(1, int(math.floor((b - a) / step)))
    vals = np.linspace(a, b, n + 1)
    return np.unique(np.concatenate([vals, [(a + b) / 2]]))


def _candidates(ctx: _Ctx, region: tuple[float, float, float, float], fx: float, fy: float, z0: float, z1: float,
                c_self: float, structural: bool) -> np.ndarray:
    cx0, cx1, cy0, cy1 = region
    xs, ys = _grid(cx0, cx1, ctx.step), _grid(cy0, cy1, ctx.step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = [np.stack([gx.ravel(), gy.ravel()], axis=1)]
    if len(ctx.placed):
        a = ctx.arr
        zov = (a[:, 2] < z1 - 1e-6) & (z0 < a[:, 5] - 1e-6)
        for i in np.nonzero(zov)[0]:
            b = a[i]
            if structural and ctx.struct[i]:
                c = 0.0
            else:
                c = MARKER_CLEARANCE if ctx.marker[i] else max(c_self, ctx.clear[i])
            c += 1e-3
            xs_s = [b[3] + c + fx / 2, b[0] - c - fx / 2]
            ys_s = [b[4] + c + fy / 2, b[1] - c - fy / 2]
            ys_al = [b[1] + fy / 2, (b[1] + b[4]) / 2, b[4] - fy / 2]
            xs_al = [b[0] + fx / 2, (b[0] + b[3]) / 2, b[3] - fx / 2]
            pts.append(np.array([(x, y) for x in xs_s for y in ys_al] + [(x, y) for y in ys_s for x in xs_al]))
    p = np.concatenate(pts, axis=0)
    ok = (p[:, 0] >= cx0 - 1e-9) & (p[:, 0] <= cx1 + 1e-9) & (p[:, 1] >= cy0 - 1e-9) & (p[:, 1] <= cy1 + 1e-9)
    return p[ok]


def _feasible(ctx: _Ctx, pts: np.ndarray, fx: float, fy: float, z0: float, z1: float, c_self: float,
              structural: bool, marker: bool) -> np.ndarray:
    if len(pts) == 0 or len(ctx.placed) == 0:
        return np.ones(len(pts), dtype=bool)
    a = ctx.arr
    zov = (a[:, 2] < z1 - 1e-6) & (z0 < a[:, 5] - 1e-6)
    if not zov.any():
        return np.ones(len(pts), dtype=bool)
    a = a[zov]
    c = np.maximum(ctx.clear[zov], c_self)
    if marker:
        c = np.full_like(c, MARKER_CLEARANCE)
    c = np.where(ctx.marker[zov], MARKER_CLEARANCE, c)
    if structural:
        c = np.where(ctx.struct[zov], 0.0, c)
    x0 = pts[:, 0:1] - fx / 2
    x1 = pts[:, 0:1] + fx / 2
    y0 = pts[:, 1:2] - fy / 2
    y1 = pts[:, 1:2] + fy / 2
    gx = np.maximum(a[None, :, 0] - x1, x0 - a[None, :, 3])
    gy = np.maximum(a[None, :, 1] - y1, y0 - a[None, :, 4])
    conflict = (gx < c[None, :] - 1e-6) & (gy < c[None, :] - 1e-6)
    return ~conflict.any(axis=1)


def _gap(pts: np.ndarray, fx: float, fy: float, b: Box3) -> np.ndarray:
    gx = np.maximum(0.0, np.maximum(b.x0 - (pts[:, 0] + fx / 2), (pts[:, 0] - fx / 2) - b.x1))
    gy = np.maximum(0.0, np.maximum(b.y0 - (pts[:, 1] + fy / 2), (pts[:, 1] - fy / 2) - b.y1))
    return np.hypot(gx, gy)


def _relation_penalty(item: PlanItem, pts: np.ndarray, fx: float, fy: float, z0: float, z1: float,
                      anchor: Optional[Box3], clearance: float) -> np.ndarray:
    if anchor is None or (item.relation == "none" and item.min_m is None and item.max_m is None):
        return np.zeros(len(pts))
    gap = _gap(pts, fx, fy, anchor)
    acx, acy, _ = anchor.center
    rel = item.relation
    same_level = abs(z0 - anchor.z0) < 0.05
    if rel == "adjacent_to":
        lo = item.min_m if item.min_m is not None else clearance
        hi = item.max_m if item.max_m is not None else clearance + 1.0
        pen = np.maximum(0, lo - gap) + np.maximum(0, gap - hi) + (0.0 if same_level else 3.0)
    elif rel in ("above", "below"):
        horiz = np.hypot(pts[:, 0] - acx, pts[:, 1] - acy)
        wrong = (z0 < anchor.z1 - 0.05) if rel == "above" else (z1 > anchor.z0 + 0.05)
        pen = horiz + (10.0 if wrong else 0.0)
    elif rel in ("left_of", "right_of"):
        sgn = 1.0 if rel == "left_of" else -1.0  # left = port (+y) when looking forward
        need = (fy + (anchor.y1 - anchor.y0)) / 2 + (item.min_m if item.min_m is not None else clearance)
        off = sgn * (pts[:, 1] - acy)
        hi = item.max_m if item.max_m is not None else 4.0
        pen = np.maximum(0, need - off) + 0.5 * np.abs(pts[:, 0] - acx) + np.maximum(0, gap - hi)
    elif rel == "in_line_with":
        pen = np.minimum(np.abs(pts[:, 0] - acx), np.abs(pts[:, 1] - acy))
        hi = item.max_m if item.max_m is not None else 6.0
        pen = pen + np.maximum(0, gap - hi)
    else:  # near (or bare distance bounds)
        lo = item.min_m if item.min_m is not None else 0.0
        hi = item.max_m if item.max_m is not None else 4.0
        pen = np.maximum(0, lo - gap) + np.maximum(0, gap - hi)
    return pen


def _relation_detail(item: PlanItem, pen: float, gap: float) -> str:
    if item.relation == "adjacent_to":
        lo = item.min_m if item.min_m is not None else 0.6
        hi = item.max_m if item.max_m is not None else lo + 1.0
        return f"edge gap {gap:.1f} m to {item.anchor_id}, expected {lo:.1f}-{hi:.1f} m on the same deck"
    if item.relation in ("above", "below"):
        return f"must be {item.relation} {item.anchor_id} on a different deck (penalty {pen:.1f})"
    return f"edge gap {gap:.1f} m to {item.anchor_id} (penalty {pen:.1f})"


@dataclass
class _ItemCtx:
    item: PlanItem
    obj: PIDObject
    size: tuple[float, float, float]
    fx: float
    fy: float
    structural: bool
    marker: bool
    c_self: float
    hint: tuple[float, float]
    anchor_box: Optional[Box3]
    group_centres: list
    nb_centres: list


def _attempt(ctx: _Ctx, it: _ItemCtx, zone: str, side: Optional[str], long_: Optional[str], relax: bool = False
             ) -> Optional[tuple[float, float, float, str, float]]:
    """Best feasible (cx, cy, z0, deck, height) for one item in a zone / band, or None."""
    room, item, fx, fy = ctx.room, it.item, it.fx, it.fy
    deck = effective_deck(item) if zone == item.zone else room.zone(zone).deck
    z0 = room.deck_z(deck)
    if zone.startswith("ship_side") and it.obj.cls == S.OVERBOARD:
        z0 += 0.5
    h = min(it.size[2], room.height - z0)
    if h <= 0.05:
        return None
    z1 = z0 + h
    reg = _region(room, zone, side, long_, fx, fy)
    if reg is None:
        return None
    cs = MARKER_CLEARANCE if relax else it.c_self
    pts = _candidates(ctx, reg, fx, fy, z0, z1, cs, it.structural)
    if len(pts) == 0:
        return None
    pts = pts[_feasible(ctx, pts, fx, fy, z0, z1, cs, it.structural, it.marker or relax)]
    if len(pts) == 0:
        return None
    tx = min(max(it.hint[0], reg[0]), reg[1])
    ty = min(max(it.hint[1], reg[2]), reg[3])
    score = 0.25 * np.hypot(pts[:, 0] - tx, pts[:, 1] - ty)
    score += _relation_penalty(item, pts, fx, fy, z0, z1, it.anchor_box, room.clearance)
    if it.group_centres:
        gc = np.mean(np.array(it.group_centres), axis=0)
        score += 0.15 * np.hypot(pts[:, 0] - gc[0], pts[:, 1] - gc[1])
    if it.nb_centres:
        nc = np.array(it.nb_centres)
        d = np.hypot(pts[:, None, 0] - nc[None, :, 0], pts[:, None, 1] - nc[None, :, 1]).mean(axis=1)
        score += 0.08 * d
    order = np.lexsort((pts[:, 1], pts[:, 0], np.round(score, 6)))
    best = pts[order[0]]
    return (float(best[0]), float(best[1]), z0, deck, h)


def solve(plan: LayoutPlan, doc: PIDDocument, room: EngineRoom) -> tuple[list[Placement3D], list[str]]:
    """Place every plan item; returns placements and notes (see :func:`solve_detailed`)."""
    res = solve_detailed(plan, doc, room)
    return res.placements, res.notes


def _place(ctx: _Ctx, it: _ItemCtx) -> tuple[tuple[float, float, float, str, float], str, str]:
    """Position of one item: requested band, whole zone, fallback zones, relaxed clearance, forced.
    Returns ((cx, cy, z0, deck, height), status, zone used)."""
    room, item, fx, fy = ctx.room, it.item, it.fx, it.fy
    status = "ok"
    res = _attempt(ctx, it, item.zone, item.side, item.longitudinal)
    zone_used = item.zone
    if res is None:
        res = _attempt(ctx, it, item.zone, None, None)
        if res is not None:
            status = "relocated"
    if res is None:
        for alt in ZONE_FALLBACK.get(item.zone, ()):
            res = _attempt(ctx, it, alt, item.side, item.longitudinal) or _attempt(ctx, it, alt, None, None)
            if res is not None:
                status, zone_used = "relocated", alt
                break
    if res is None:
        res = _attempt(ctx, it, item.zone, None, None, relax=True)
        if res is not None:
            status = "relaxed"
    if res is None:
        deck = effective_deck(item)
        z0 = room.deck_z(deck)
        bb = room.band(item.zone, item.side, item.longitudinal)
        cx = min(max((bb.x0 + bb.x1) / 2, fx / 2), room.length - fx / 2)
        cy = min(max((bb.y0 + bb.y1) / 2, -room.breadth / 2 + fy / 2), room.breadth / 2 - fy / 2)
        res = (cx, cy, z0, deck, min(it.size[2], room.height - z0))
        status = "forced"
    return res, status, zone_used


def _diagnose(it: _ItemCtx, room: EngineRoom, res: tuple[float, float, float, str, float], status: str,
              zone_used: str) -> ItemDiag:
    item, size, fx, fy = it.item, it.size, it.fx, it.fy
    cx, cy, z0, _, h = res
    requested = f"{item.zone}/{item.side}/{item.longitudinal}"
    side_a, long_a = room.classify(cx, cy)
    actual = f"{zone_used}/{side_a}/{long_a}"
    msg = ""
    if status == "relocated":
        msg = f"no free space for {fx:.1f} x {fy:.1f} m in {requested}; solver moved it to {actual}"
    elif status == "relaxed":
        msg = f"placed in {actual} with reduced clearance ({MARKER_CLEARANCE} m): zone is crowded"
    elif status == "forced":
        msg = f"no feasible position anywhere; forced into {actual} and overlaps other equipment"
    if h < size[2]:
        msg = (msg + "; " if msg else "") + f"height clipped to {h:.1f} m by the room"
        if status == "ok":
            status = "relocated"
    if item.size_m and (abs(size[0] - item.size_m[0]) > 1e-6 or abs(size[1] - item.size_m[1]) > 1e-6):
        msg = (msg + "; " if msg else "") + f"size_m {item.size_m} exceeds the room; clipped to {list(size)}"
        if status == "ok":
            status = "relocated"
    diag = ItemDiag(object_id=item.object_id, status=status, requested=requested, actual=actual,
                    relation=item.relation, message=msg)
    anchor_box = it.anchor_box
    if anchor_box is not None and (item.relation != "none" or item.min_m is not None or item.max_m is not None):
        pt = np.array([[cx, cy]])
        pen = float(_relation_penalty(item, pt, fx, fy, z0, z0 + h, anchor_box, room.clearance)[0])
        gap = float(_gap(pt, fx, fy, anchor_box)[0])
        diag.relation_ok = pen <= 0.3
        diag.relation_detail = _relation_detail(item, pen, gap)
    return diag


def solve_detailed(plan: LayoutPlan, doc: PIDDocument, room: EngineRoom, step: float = 0.5,
                   adjacency: Optional[dict[str, list[Neighbor]]] = None) -> SolveResult:
    objs = doc.object_by_id()
    labels = object_labels(doc)
    dpos = drawing_positions(doc)
    planned_ids = [it.object_id for it in plan.items if it.object_id in objs]
    if adjacency is None:
        adjacency = equipment_adjacency(doc, set(planned_ids))
    ctx = _Ctx(room=room, step=step)
    placements: dict[str, Placement3D] = {}
    boxes: dict[str, Box3] = {}
    diags: dict[str, ItemDiag] = {}
    item_ctx: dict[str, _ItemCtx] = {}
    notes: list[str] = []
    groups: dict[str, list[str]] = defaultdict(list)

    for item in plan.items:
        oid = item.object_id
        obj = objs.get(oid)
        if obj is None:
            notes.append(f"plan item {oid}: no such object, skipped")
            continue
        if oid in placements:
            notes.append(f"plan item {oid}: duplicate, skipped")
            continue
        dn = max((n.dn or 0 for n in adjacency.get(oid, [])), default=0) or None
        size, size_origin = object_size(obj, room, item.zone, labels.get(oid, ""), dn, item.size_m)
        yaw = float(item.orientation_deg % 360)
        fx, fy = yaw_footprint(size, yaw)
        structural = obj.cls in STRUCTURAL
        marker = obj.cls in MARKERS
        c_self = MARKER_CLEARANCE if marker else room.clearance
        anchor_box = boxes.get(item.anchor_id) if item.anchor_id else None
        u, v = dpos.get(oid, (0.5, 0.5))
        hint = (u * room.length, (0.5 - v) * room.breadth)
        nb_centres = [boxes[n.id].center[:2] for n in adjacency.get(oid, []) if n.id in boxes]
        group_centres = [boxes[g].center[:2] for g in groups.get(item.group or "", []) if g in boxes]
        it = _ItemCtx(item=item, obj=obj, size=size, fx=fx, fy=fy, structural=structural, marker=marker,
                      c_self=c_self, hint=hint, anchor_box=anchor_box, group_centres=group_centres,
                      nb_centres=nb_centres)
        item_ctx[oid] = it
        res, status, zone_used = _place(ctx, it)
        cx, cy, z0, deck, h = res
        if status == "forced":
            notes.append(f"{oid}: forced placement overlaps other equipment")
        diags[oid] = _diagnose(it, room, res, status, zone_used)
        tag = f"[{item.source}] " if item.source else ""
        placements[oid] = Placement3D(
            object_id=oid, cls=obj.cls, position=Vec3(x=round(cx, 4), y=round(cy, 4), z=round(z0, 4)),
            yaw_deg=yaw, size=Vec3(x=size[0], y=size[1], z=round(h, 4)), deck=deck,
            rationale=(tag + (item.rationale or "") + (f" (size: {size_origin})" if size_origin else "")).strip(),
        )
        boxes[oid] = Box3(cx - fx / 2, cy - fy / 2, z0, cx + fx / 2, cy + fy / 2, z0 + h)
        ctx.add(_Placed(oid, boxes[oid], c_self, structural, marker, item.group))
        if item.group:
            groups[item.group].append(oid)

    _nozzles_with_access(doc, plan, room, placements, boxes, diags, item_ctx, notes, step)
    counts = defaultdict(int)
    for d in diags.values():
        counts[d.status] += 1
    unsat = sum(1 for d in diags.values() if d.relation_ok is False)
    covered = sum(len(d.nozzle_issues) for d in diags.values())
    notes.append("solver: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) +
                 f"; unsatisfied relations {unsat}; covered nozzles {covered}")
    return SolveResult(list(placements.values()), notes, diags, boxes)


def _nozzles_with_access(doc: PIDDocument, plan: LayoutPlan, room: EngineRoom, placements: dict[str, Placement3D],
                         boxes: dict[str, Box3], diags: dict[str, ItemDiag], item_ctx: dict[str, _ItemCtx],
                         notes: list[str], step: float) -> None:
    """Compute every nozzle; when one is covered by other equipment, move the covering item (or,
    if that is structural, the item itself) once with the nozzle's access column kept clear, then
    recompute. Nozzles still covered are recorded as diagnostics issues."""
    objs = doc.object_by_id()
    zone_of = plan.item_by_id()
    moved: set[str] = set()
    for attempt in range(2):
        blocked: list[NozzleBlock] = []
        for pl in placements.values():
            pl.nozzles = {}
        for oid, pl in placements.items():
            noz, blk = compute_nozzles_detailed(doc, objs[oid], pl, placements, room, zone_of=zone_of)
            pl.nozzles = noz
            blocked += blk
        if not blocked:
            return
        if attempt == 1:
            break
        any_moved = False
        def movable(oid: str) -> bool:  # double-bottom tanks / sea chests are packed in the hull: never moved
            it = zone_of.get(oid)
            return oid in placements and oid not in moved and not (it is not None and it.zone in
                                                                    ("double_bottom", "bottom"))

        for b in blocked:
            # prefer moving the covering item (a non-structural one first), else the nozzle's owner
            cands = [o for o in (b.blocker, b.object_id) if movable(o)]
            cands.sort(key=lambda o: (objs[o].cls in STRUCTURAL, o != b.blocker))
            if not cands:
                continue
            mover = cands[0]
            keep_out = b.access_box(0.2)
            if _relocate(room, step, placements, boxes, diags, item_ctx, mover, keep_out, b):
                moved.add(mover)
                any_moved = True
        if not any_moved:
            break
    for b in blocked:
        what = f"by {b.blocker}" if b.blocker != "room" else "by the room boundary"
        diags[b.object_id].nozzle_issues.append(
            f"nozzle {b.key} is covered {what}: no free face point for the pipe connection; "
            f"move {b.blocker if b.blocker in placements else b.object_id} or give it another zone / side")
    if blocked:
        notes.append(f"solver: {len(blocked)} nozzle(s) without free access: "
                     + ", ".join(f"{b.object_id}.{b.key} ({b.blocker})" for b in blocked[:10]))


def _relocate(room: EngineRoom, step: float, placements: dict[str, Placement3D], boxes: dict[str, Box3],
              diags: dict[str, ItemDiag], item_ctx: dict[str, _ItemCtx], mover: str, keep_out: Box3,
              why: "NozzleBlock") -> bool:
    """Re-place ``mover`` with all other boxes plus ``keep_out`` as obstacles."""
    it = item_ctx.get(mover)
    if it is None:
        return False
    ctx = _Ctx(room=room, step=step)
    for oid, bx in boxes.items():
        if oid == mover:
            continue
        o = item_ctx.get(oid)
        if o is None:
            continue
        ctx.add(_Placed(oid, bx, o.c_self, o.structural, o.marker, o.item.group))
    ctx.add(_Placed("__nozzle_access__", keep_out, 0.0, False, True, None))
    res, status, zone_used = _place(ctx, it)
    if status == "forced":
        return False
    cx, cy, z0, deck, h = res
    pl = placements[mover]
    diag = _diagnose(it, room, res, status, zone_used)
    if diag.status == "ok":
        diag.status = "relocated"
    what = (f"moved to keep nozzle {why.key} of {why.object_id} accessible (it was covering it)"
            if mover != why.object_id else f"moved so that its nozzle {why.key} is not covered by {why.blocker}")
    diag.message = ((diag.message + "; ") if diag.message else "") + what
    diags[mover] = diag
    pl.position = Vec3(x=round(cx, 4), y=round(cy, 4), z=round(z0, 4))
    pl.size = Vec3(x=pl.size.x, y=pl.size.y, z=round(h, 4))
    pl.deck = deck
    boxes[mover] = Box3(cx - it.fx / 2, cy - it.fy / 2, z0, cx + it.fx / 2, cy + it.fy / 2, z0 + h)
    return True


# ---------------------------------------------------------------------------- nozzles


def _face_point(pl: Placement3D, face: str, t: float, zf: float) -> tuple[float, float, float]:
    l, w, h = pl.size.as_tuple()
    m = 0.08
    if face in ("+x", "-x"):
        sgn = 1 if face == "+x" else -1
        ly = max(-w / 2 + m, min(w / 2 - m, t * w)) if w > 2 * m else 0.0
        return local_to_world(pl, sgn * l / 2, ly, max(0.05 * h, min(0.95 * h, zf * h)))
    if face in ("+y", "-y"):
        sgn = 1 if face == "+y" else -1
        lx = max(-l / 2 + m, min(l / 2 - m, t * l)) if l > 2 * m else 0.0
        return local_to_world(pl, lx, sgn * w / 2, max(0.05 * h, min(0.95 * h, zf * h)))
    lx = max(-l / 2 + m, min(l / 2 - m, t * l)) if l > 2 * m else 0.0
    return local_to_world(pl, lx, 0.0, h)


def _spec_face(cls: SymbolClass, port: str) -> Optional[tuple[str, float, float]]:
    over = _FACE_OVERRIDES.get(cls, {}).get(port)
    if over:
        return over
    if cls == S.UNKNOWN or cls in BOUNDARY_FACE_CLASSES:
        return None
    spec = get_symbol(cls)
    if spec.boundary_ports:
        return None
    ps = next((p for p in spec.ports if p.name == port), None)
    if ps is None:
        return None
    u0, v0, u1, v1 = spec.extent
    a = (ps.x - (u0 + u1) / 2) / max(1e-9, u1 - u0)
    b = (ps.y - (v0 + v1) / 2) / max(1e-9, v1 - v0)
    if abs(ps.dx) >= abs(ps.dy):
        return ("+x" if ps.dx > 0 else "-x", 0.0, min(0.8, max(0.2, 0.5 - b)))
    if ps.dy < 0:
        return ("top", a, 1.0)
    return ("-y", a, 0.3)


def _neighbor_targets(doc: PIDDocument, obj: PIDObject, placements: dict[str, Placement3D]
                      ) -> dict[str, tuple[float, float, float]]:
    """Port key -> world point of the nearest placed object along that pipe (or a direction hint)."""
    objs = doc.object_by_id()
    adj: dict[str, list[tuple[str, object]]] = defaultdict(list)
    pipes = process_pipes(doc)
    for p in pipes:
        a, b = p.start.node_id, p.end.node_id
        if a and b and a != b:
            adj[a].append((b, p))
            adj[b].append((a, p))
    me = placements[obj.id]
    out: dict[str, tuple[float, float, float]] = {}
    for p in pipes:
        for end, other in ((p.start, p.end), (p.end, p.start)):
            if end.object_id != obj.id:
                continue
            key = port_key(end, p.id)
            if key in out:
                continue
            target = None
            start = other.node_id
            if start and start != obj.id:
                seen, q = {obj.id, start}, deque([start])
                while q and target is None:
                    n = q.popleft()
                    if n in placements and n != obj.id:
                        b = placement_box(placements[n])
                        target = b.center
                        break
                    if n in objs and n not in placements and len(seen) > 200:
                        break
                    for nb, _ in adj.get(n, []):
                        if nb not in seen:
                            seen.add(nb)
                            q.append(nb)
            if target is None:
                pts = p.points if end is p.start else list(reversed(p.points))
                c = obj.center
                far = pts[min(len(pts) - 1, 1)] if len(pts) > 1 else pts[0]
                dx, dy = far.x - c.x, far.y - c.y
                n = math.hypot(dx, dy) or 1.0
                cx, cy, cz = placement_box(me).center
                target = (cx + 5 * dx / n, cy - 5 * dy / n, cz)
            out[key] = target
    return out


NOZZLE_PROBE = 0.4  # m: the space just outside a nozzle must be free for the pipe stub
_PROBE_STEPS = (0.1, 0.25, NOZZLE_PROBE)
NOZZLE_STUB = 0.5  # m of pipe stub assumed straight out of a nozzle
NOZZLE_SPACING = 0.55  # m between the stubs of nozzles of different objects where possible (> one routing cell)


def _seg_dist(p1: Sequence[float], q1: Sequence[float], p2: Sequence[float], q2: Sequence[float]) -> float:
    """Closest distance between two 3D segments."""
    a0, a1, b0, b1 = (np.asarray(x, dtype=float) for x in (p1, q1, p2, q2))
    d1, d2, r = a1 - a0, b1 - b0, a0 - b0
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s_, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e < 1e-12:
            t, s_ = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s_ = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-12 else 0.0
            t = (b * s_ + f) / e
            if t < 0.0:
                t, s_ = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t, s_ = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return float(np.linalg.norm((a0 + d1 * s_) - (b0 + d2 * t)))
_LOCAL_NORMALS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "top": (0, 0, 1)}


@dataclass
class NozzleBlock:
    """A nozzle whose outside is covered by other equipment (or lies against the room boundary)."""

    object_id: str
    key: str
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    blocker: str  # object id, or "room"

    def access_box(self, half: float = 0.2) -> Box3:
        """Column in front of the nozzle that must stay free (pipe stub + elbow)."""
        a = self.point
        b = tuple(a[i] + 0.8 * self.normal[i] for i in range(3))
        return Box3(min(a[0], b[0]) - half, min(a[1], b[1]) - half, min(a[2], b[2]) - half,
                    max(a[0], b[0]) + half, max(a[1], b[1]) + half, max(a[2], b[2]) + half)


def _world_normal(pl: Placement3D, face: str) -> tuple[float, float, float]:
    nx, ny, nz = _LOCAL_NORMALS[face]
    a = math.radians(pl.yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return (c * nx - s * ny, s * nx + c * ny, float(nz))


_SHIFTS = (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.8, -1.8, 2.6, -2.6)


def _face_candidates(pl: Placement3D, face: str, lx: float, ly: float, lz: float, high: bool = False
                     ) -> list[tuple[float, float, float]]:
    """Local points on ``face`` closest to the target, then shifted along the face (0.4 m steps);
    the top face is searched in both directions, ordered by distance from the closest point.
    ``high``: side-face points at the top of the face (low tanks: just under the tank top)."""
    l, w, h = pl.size.as_tuple()
    m = 0.15

    def clamp(v: float, half: float) -> float:
        return max(-half + m, min(half - m, v)) if half > m else 0.0

    pz = (max(0.5 * h, h - 0.2) if high else max(0.25 * h, min(0.75 * h, lz))) if h > 0.3 else h / 2
    out: list[tuple[float, float, float]] = []
    if face == "top":
        cand = []
        for dx in _SHIFTS:
            for dy in _SHIFTS:
                q = (clamp(lx + dx, l / 2), clamp(ly + dy, w / 2), h)
                cand.append((math.hypot(q[0] - clamp(lx, l / 2), q[1] - clamp(ly, w / 2)), q))
        for _, q in sorted(cand, key=lambda c: c[0]):
            if q not in out:
                out.append(q)
        return out
    for d in _SHIFTS:
        if face in ("+x", "-x"):
            q = ((l / 2) * (1 if face == "+x" else -1), clamp(ly + d, w / 2), pz)
        else:
            q = (clamp(lx + d, l / 2), (w / 2) * (1 if face == "+y" else -1), pz)
        if q not in out:
            out.append(q)
    return out


def compute_nozzles(doc: PIDDocument, obj: PIDObject, pl: Placement3D, placements: dict[str, Placement3D],
                    room: EngineRoom, zone_of: Optional[dict[str, PlanItem]] = None) -> dict[str, Vec3]:
    """World positions of every nozzle of a planned object (see :func:`compute_nozzles_detailed`)."""
    return compute_nozzles_detailed(doc, obj, pl, placements, room, zone_of)[0]


def compute_nozzles_detailed(doc: PIDDocument, obj: PIDObject, pl: Placement3D, placements: dict[str, Placement3D],
                             room: EngineRoom, zone_of: Optional[dict[str, PlanItem]] = None
                             ) -> tuple[dict[str, Vec3], list[NozzleBlock]]:
    """Nozzle positions (ports + pipe ends without port names) and the nozzles left covered.

    Fixed symbol ports keep their mapped face. Boundary nozzles go on the face pointing to the
    connected equipment (top face for double-bottom tanks / sea chests, then the upper part of
    their side faces), at the point closest to it, moved along the face (or to the next best
    face) until the space just outside the nozzle is free of other equipment, at least 0.3 m
    from the object's other nozzles and, where possible, its stub (NOZZLE_STUB out along the
    normal) NOZZLE_SPACING from the stubs of nozzles already placed on other objects. Every
    nozzle is probe-checked (0.1 / 0.25 / 0.4 m out along its normal); a covered one is returned
    as a :class:`NozzleBlock` naming the covering item.
    """
    keys: list[str] = [p.name for p in obj.ports]
    for p in process_pipes(doc):
        for end in (p.start, p.end):
            if end.object_id == obj.id:
                k = port_key(end, p.id)
                if k not in keys:
                    keys.append(k)
    if not keys:
        return {}, []
    targets = _neighbor_targets(doc, obj, placements)
    item = (zone_of or {}).get(obj.id)
    low = item is not None and item.zone in ("double_bottom", "bottom")
    box = placement_box(pl)
    cx, cy, cz = box.center
    others = [(oid, placement_box(q).expand(0.05)) for oid, q in placements.items() if oid != obj.id]
    near = box.expand(NOZZLE_STUB + NOZZLE_SPACING + 0.5)
    stubs = []  # stubs of nozzles of nearby objects
    for oid, q in placements.items():
        if oid == obj.id:
            continue
        for nz in q.nozzles.values():
            if near.contains(nz.x, nz.y, nz.z):
                nn = nozzle_normal(q, nz)
                a = nz.as_tuple()
                stubs.append((a, (a[0] + NOZZLE_STUB * nn[0], a[1] + NOZZLE_STUB * nn[1], a[2] + NOZZLE_STUB * nn[2])))
    bounds = room.bounds
    l, w, h = pl.size.as_tuple()
    out: dict[str, tuple[float, float, float]] = {}
    used: list[tuple[float, float, float]] = []
    blocked: list[NozzleBlock] = []

    def blocker(pt: tuple[float, float, float], n: tuple[float, float, float]) -> Optional[str]:
        """Id of the item covering the nozzle's outside ("room" if it leaves the room), else None."""
        for d in _PROBE_STEPS:
            probe = (pt[0] + d * n[0], pt[1] + d * n[1], pt[2] + d * n[2])
            if not bounds.contains(*probe):
                return "room"
            for oid, b in others:
                if b.contains(*probe):
                    return oid
        return None

    def crowded(pt: tuple[float, float, float], n: tuple[float, float, float]) -> bool:
        tip = (pt[0] + NOZZLE_STUB * n[0], pt[1] + NOZZLE_STUB * n[1], pt[2] + NOZZLE_STUB * n[2])
        return any(_seg_dist(pt, tip, a, b) < NOZZLE_SPACING for a, b in stubs)

    for k in keys:
        spec_face = None if low else _spec_face(obj.cls, k)
        if spec_face is not None:  # fixed port of the symbol: face given by the symbol, only probe-checked
            pt = _face_point(pl, *spec_face)
            n = _world_normal(pl, spec_face[0])
            out[k] = pt
            used.append(pt)
            who = blocker(pt, n)
            if who is not None:
                blocked.append(NozzleBlock(obj.id, k, pt, n, who))
            continue
        tx, ty, tz = targets.get(k, (cx, cy, box.z1 + 1.0))
        lx, ly, lz = world_to_local(pl, tx, ty, tz)
        rel = {"+x": lx / (l / 2), "-x": -lx / (l / 2), "+y": ly / (w / 2), "-y": -ly / (w / 2),
               "top": (lz - h / 2) / (h / 2)}
        sides = [f for f, _ in sorted(rel.items(), key=lambda kv: -kv[1]) if f != "top"]
        if low:  # through the tank top; the side faces (just under the top) only if the top is covered
            order = [("top", False)] + [(f, True) for f in sides]
        else:
            faces = [f for f, _ in sorted(rel.items(), key=lambda kv: -kv[1])]
            if obj.cls == S.OVERBOARD and item is not None and item.zone.startswith("ship_side"):
                # the discharge pipe comes from inboard, perpendicular to the shell
                inboard = _local_face_for_world(pl, "-y" if item.zone.endswith("port") else "+y")
                faces = [inboard] + [f for f in faces if f != inboard]
            order = [(f, False) for f in faces]
        chosen = None
        spare = None  # free but close to another object's nozzle stub
        first_block: Optional[NozzleBlock] = None
        for face, high in order:
            n = _world_normal(pl, face)
            for q in _face_candidates(pl, face, lx, ly, lz, high):
                wpt = local_to_world(pl, *q)
                if any(math.dist(wpt, u) < 0.3 for u in used):
                    continue
                who = blocker(wpt, n)
                if who is None:
                    if not stubs or not crowded(wpt, n):
                        chosen = wpt
                        break
                    spare = spare or wpt
                elif first_block is None:
                    first_block = NozzleBlock(obj.id, k, wpt, n, who)
            if chosen is not None:
                break
        if chosen is None and spare is not None:
            chosen = spare
        if chosen is None:  # every face is covered: closest point on the preferred face, reported
            face = order[0][0]
            chosen = local_to_world(pl, *_face_candidates(pl, face, lx, ly, lz, order[0][1])[0])
            n = _world_normal(pl, face)
            who = blocker(chosen, n) or (first_block.blocker if first_block else "room")
            blocked.append(NozzleBlock(obj.id, k, chosen, n, who))
        out[k] = chosen
        used.append(chosen)
    return {k: Vec3(x=round(x, 4), y=round(y, 4), z=round(z, 4)) for k, (x, y, z) in out.items()}, blocked


def _local_face_for_world(pl: Placement3D, world_face: str) -> str:
    """Local face name whose outward normal points along a world horizontal axis (+y / -y)."""
    target = (0.0, 1.0) if world_face == "+y" else (0.0, -1.0)
    a = math.radians(pl.yaw_deg)
    c, s = math.cos(a), math.sin(a)
    best, best_dot = "+y", -2.0
    for name, (nx, ny) in (("+x", (1, 0)), ("-x", (-1, 0)), ("+y", (0, 1)), ("-y", (0, -1))):
        wx, wy = c * nx - s * ny, s * nx + c * ny
        d = wx * target[0] + wy * target[1]
        if d > best_dot:
            best, best_dot = name, d
    return best
