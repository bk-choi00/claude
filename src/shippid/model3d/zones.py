"""Engine-room envelope, decks and named zones used by the 3D arrangement planner.

A P&ID carries no spatial information, so the 3D model is a *plausible* engine-room
arrangement, not a survey. The room is a box in a ship-fixed frame (metres):

* x -- forward, from the aft engine-room bulkhead (x = 0) to the forward bulkhead (x = L)
* y -- to port, centre line at y = 0 (port side y > 0, starboard y < 0)
* z -- up, from the base line (z = 0, inner bottom of the hull)

Decks follow :data:`shippid.catalog.DEFAULT_DECKS` scaled per ship type. Named zones are
vertical slabs between decks (double bottom, tank top, floor, platforms) plus hull zones
for sea chests (``bottom``) and shell penetrations (``ship_side_port`` / ``ship_side_starboard``).
Each zone is divided into port / center / starboard bands and aft / mid / fore thirds.

Room dimensions per ship type are HEURISTIC order-of-magnitude values for large ships
(usable engine-room envelope at tank-top level, not moulded hull dimensions); override
them per project with a mapping or YAML file (:func:`load_engine_room`,
``DrawingInfo.extra["engine_room"]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ..catalog import DEFAULT_DECKS
from ..model import Deck, DrawingInfo, Vec3

ZONE_NAMES: tuple[str, ...] = (
    "double_bottom",
    "tank_top",
    "floor",
    "middle_platform",
    "upper_platform",
    "bottom",
    "ship_side_port",
    "ship_side_starboard",
)
SIDES: tuple[str, ...] = ("port", "center", "starboard")
LONGITUDINALS: tuple[str, ...] = ("aft", "mid", "fore")
# "base_line" is the inner bottom of the hull (floor of the double bottom); the others
# are the deck names of catalog.DEFAULT_DECKS.
DECK_NAMES: tuple[str, ...] = ("base_line",) + tuple(name for name, _ in DEFAULT_DECKS)

# default standing deck of each zone, and the decks an item in the zone may stand on
ZONE_DECK: dict[str, str] = {
    "double_bottom": "base_line",
    "bottom": "base_line",
    "tank_top": "tank_top",
    "floor": "floor",
    "middle_platform": "3rd_deck",
    "upper_platform": "2nd_deck",
    "ship_side_port": "floor",
    "ship_side_starboard": "floor",
}
_SHELL_DECKS = ("tank_top", "floor", "3rd_deck", "2nd_deck")
ZONE_ALLOWED_DECKS: dict[str, tuple[str, ...]] = {
    "double_bottom": ("base_line",),
    "bottom": ("base_line",),
    "tank_top": ("tank_top",),
    "floor": ("floor",),
    "middle_platform": ("3rd_deck",),
    "upper_platform": ("2nd_deck",),
    "ship_side_port": _SHELL_DECKS,
    "ship_side_starboard": _SHELL_DECKS,
}
ZONE_DESCRIPTIONS: dict[str, str] = {
    "double_bottom": "inside the double bottom between base line and tank top: double-bottom tanks "
                     "(bilge well, sludge, drain, LO sump, overflow, storage, water ballast)",
    "tank_top": "standing on the tank top, the lowest engine-room level: pumps, main engine, OWS, "
                "large coolers, sea-water and ballast machinery",
    "floor": "standing on the floor (lowest platform) deck: coolers, filters, BWTS, compressors, "
             "air receivers, generator engines, settling/service tanks at the side",
    "middle_platform": "standing on the 3rd deck platform: purifier room, heaters, FO/LO service equipment",
    "upper_platform": "standing on the 2nd deck platform: boilers, expansion/header tanks, workshop level",
    "bottom": "in the hull bottom / bilge inside the double-bottom height: sea chests (sea inlets)",
    "ship_side_port": "against the port shell plating: overboard discharges and shell valves",
    "ship_side_starboard": "against the starboard shell plating: overboard discharges and shell valves",
}


@dataclass(frozen=True)
class Box3:
    """Axis-aligned box in room coordinates."""

    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float

    @property
    def size(self) -> tuple[float, float, float]:
        return (self.x1 - self.x0, self.y1 - self.y0, self.z1 - self.z0)

    @property
    def center(self) -> tuple[float, float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2, (self.z0 + self.z1) / 2)

    def contains(self, x: float, y: float, z: float, tol: float = 0.0) -> bool:
        return (self.x0 - tol <= x <= self.x1 + tol and self.y0 - tol <= y <= self.y1 + tol
                and self.z0 - tol <= z <= self.z1 + tol)

    def expand(self, d: float) -> "Box3":
        return Box3(self.x0 - d, self.y0 - d, self.z0 - d, self.x1 + d, self.y1 + d, self.z1 + d)

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.x0, self.y0, self.z0, self.x1, self.y1, self.z1)


@dataclass(frozen=True)
class Zone:
    name: str
    box: Box3
    deck: str  # default standing deck
    allowed_decks: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ShipProfile:
    key: str
    keywords: tuple[str, ...]
    length: float  # engine-room length (m)
    breadth: float  # usable engine-room breadth at tank-top level (m)
    deck_scale: float  # factor applied to catalog.DEFAULT_DECKS elevations
    note: str


# HEURISTIC engine-room envelopes (order of magnitude for large ships; no single primary
# source -- tune per hull). deck_scale follows the moulded depth relative to the ~24 m
# upper deck of DEFAULT_DECKS (VLCC D ~ 29 m, large container ship / LNG carrier D ~ 26 m).
SHIP_PROFILES: tuple[ShipProfile, ...] = (
    ShipProfile("vlcc", ("VLCC", "ULCC", "CRUDE", "TANKER", "SUEZMAX", "AFRAMAX", "PRODUCT"), 42.0, 40.0, 1.2,
                "VLCC/tanker: ~42 m ER length, ~40 m usable breadth, D ~ 29 m"),
    ShipProfile("lng", ("LNG", "LPG", "GAS CARRIER", "FSRU"), 40.0, 36.0, 1.1,
                "LNG/LPG carrier: ~40 m ER length, ~36 m usable breadth, D ~ 26 m"),
    ShipProfile("container", ("CONTAINER", "TEU", "BOX"), 38.0, 34.0, 1.1,
                "large container ship: ~38 m ER length, ~34 m usable breadth, D ~ 26 m"),
    ShipProfile("bulk", ("BULK", "CAPESIZE", "PANAMAX", "KAMSARMAX", "ORE", "NEWCASTLEMAX"), 32.0, 30.0, 1.0,
                "bulk carrier: ~32 m ER length, ~30 m usable breadth, D ~ 24 m"),
)
GENERIC_PROFILE = ShipProfile("generic", (), 34.0, 30.0, 1.0,
                              "generic large merchant ship: ~34 m ER length, ~30 m usable breadth, D ~ 24 m")


@dataclass
class EngineRoom:
    """Engine-room box, decks and zones (see module docstring for the frame)."""

    length: float
    breadth: float
    height: float
    decks: list[tuple[str, float]]  # (name, z) incl. ("base_line", 0.0), ascending z
    ship_type: str = "generic"
    clearance: float = 0.6  # maintenance clearance between equipment (m)
    side_band: float = 1.5  # thickness of the ship-side shell zones (m)
    source: str = "heuristic"
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [n for n, _ in self.decks]
        if "base_line" not in names:
            self.decks = [("base_line", 0.0)] + list(self.decks)
        self.decks = sorted(((str(n), float(z)) for n, z in self.decks), key=lambda d: d[1])
        missing = [n for n in DECK_NAMES if n not in dict(self.decks)]
        if missing:
            raise ValueError(f"engine room is missing decks {missing}")
        if self.length <= 0 or self.breadth <= 0 or self.height <= 0:
            raise ValueError("engine room dimensions must be positive")
        self.height = max(self.height, self.deck_z("2nd_deck") + 2.5)

    # -- decks -------------------------------------------------------------------
    def deck_z(self, name: str) -> float:
        for n, z in self.decks:
            if n == name:
                return z
        raise KeyError(name)

    def deck_above(self, name: str) -> Optional[tuple[str, float]]:
        """Next deck above ``name`` (None for the top deck)."""
        z0 = self.deck_z(name)
        above = [(n, z) for n, z in self.decks if z > z0 + 1e-6]
        return above[0] if above else None

    def deck_below_z(self, z: float) -> str:
        """Name of the highest deck at or below elevation ``z``."""
        best = self.decks[0][0]
        for n, zz in self.decks:
            if zz <= z + 1e-6:
                best = n
        return best

    def to_decks(self) -> list[Deck]:
        """Real decks for :class:`shippid.model.Layout3D` (base line excluded)."""
        return [Deck(name=n, z=z) for n, z in self.decks if n != "base_line"]

    @property
    def bounds(self) -> Box3:
        return Box3(0.0, -self.breadth / 2, 0.0, self.length, self.breadth / 2, self.height)

    def bounds_vec(self) -> tuple[Vec3, Vec3]:
        b = self.bounds
        return Vec3(x=b.x0, y=b.y0, z=b.z0), Vec3(x=b.x1, y=b.y1, z=b.z1)

    # -- zones -------------------------------------------------------------------
    @property
    def zones(self) -> dict[str, Zone]:
        return {name: self.zone(name) for name in ZONE_NAMES}

    def zone(self, name: str) -> Zone:
        L, B, H = self.length, self.breadth, self.height
        tt = self.deck_z("tank_top")
        if name in ("double_bottom", "bottom"):
            box = Box3(0.0, -B / 2, 0.0, L, B / 2, tt)
        elif name == "ship_side_port":
            box = Box3(0.0, B / 2 - self.side_band, tt, L, B / 2, H)
        elif name == "ship_side_starboard":
            box = Box3(0.0, -B / 2, tt, L, -B / 2 + self.side_band, H)
        elif name in ZONE_DECK:
            deck = ZONE_DECK[name]
            z0 = self.deck_z(deck)
            above = self.deck_above(deck)
            box = Box3(0.0, -B / 2, z0, L, B / 2, above[1] if above else H)
        else:
            raise KeyError(f"unknown zone {name!r}; expected one of {ZONE_NAMES}")
        return Zone(name, box, ZONE_DECK[name], ZONE_ALLOWED_DECKS[name], ZONE_DESCRIPTIONS[name])

    def band(self, zone: str, side: Optional[str] = None, longitudinal: Optional[str] = None) -> Box3:
        """Sub-box of a zone: transverse band (port/center/starboard) and longitudinal third."""
        b = self.zone(zone).box
        x0, x1, y0, y1 = b.x0, b.x1, b.y0, b.y1
        if longitudinal in LONGITUDINALS:
            i = LONGITUDINALS.index(longitudinal)
            x0, x1 = b.x0 + (b.x1 - b.x0) * i / 3, b.x0 + (b.x1 - b.x0) * (i + 1) / 3
        if side in SIDES and not zone.startswith("ship_side"):
            third = self.breadth / 3
            if side == "port":
                y0, y1 = max(y0, self.breadth / 2 - third), y1
            elif side == "starboard":
                y0, y1 = y0, min(y1, -self.breadth / 2 + third)
            else:
                y0, y1 = max(y0, -third / 2), min(y1, third / 2)
        return Box3(x0, y0, b.z0, x1, y1, b.z1)

    def classify(self, x: float, y: float) -> tuple[str, str]:
        """(side, longitudinal) band names of a point."""
        third = self.breadth / 3
        side = "port" if y > third / 2 else ("starboard" if y < -third / 2 else "center")
        i = min(2, max(0, int(x / (self.length / 3))))
        return side, LONGITUDINALS[i]

    # -- io ----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "length": self.length, "breadth": self.breadth, "height": self.height,
            "decks": {n: z for n, z in self.decks}, "ship_type": self.ship_type,
            "clearance": self.clearance, "side_band": self.side_band, "source": self.source,
        }

    def describe(self) -> dict[str, Any]:
        """Compact description for prompts / reports."""
        return {
            "frame": "x forward from the aft engine-room bulkhead, y to port (centre line y=0), z up from base line",
            "length_m": round(self.length, 2), "breadth_m": round(self.breadth, 2), "height_m": round(self.height, 2),
            "decks": {n: round(z, 2) for n, z in self.decks},
            "ship_type": self.ship_type,
            "maintenance_clearance_m": self.clearance,
        }


def _profile_for(ship_type: Optional[str]) -> ShipProfile:
    up = (ship_type or "").upper()
    for prof in SHIP_PROFILES:
        if any(k in up for k in prof.keywords):
            return prof
    return GENERIC_PROFILE


def engine_room_from_mapping(data: Mapping[str, Any], base: Optional[EngineRoom] = None) -> EngineRoom:
    """Engine room from a mapping (keys: length, breadth, height, decks {name: z} or
    [[name, z], ...], ship_type, clearance, side_band), on top of ``base`` (or generic)."""
    room = base or engine_room_for(DrawingInfo())
    decks = dict(room.decks)
    raw = data.get("decks")
    if isinstance(raw, Mapping):
        decks.update({str(k): float(v) for k, v in raw.items()})
    elif isinstance(raw, (list, tuple)):
        decks.update({str(k): float(v) for k, v in raw})
    height = float(data.get("height", decks.get("upper_deck", room.height)))
    return EngineRoom(
        length=float(data.get("length", room.length)),
        breadth=float(data.get("breadth", room.breadth)),
        height=height,
        decks=list(decks.items()),
        ship_type=str(data.get("ship_type", room.ship_type)),
        clearance=float(data.get("clearance", room.clearance)),
        side_band=float(data.get("side_band", room.side_band)),
        source="override",
        notes=list(room.notes) + ["engine room dimensions overridden by configuration"],
    )


def load_engine_room(path: str | Path, base: Optional[EngineRoom] = None) -> EngineRoom:
    """Engine room from a YAML file (top-level keys as :func:`engine_room_from_mapping`,
    optionally nested under ``engine_room:``)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if isinstance(data, Mapping) and isinstance(data.get("engine_room"), Mapping):
        data = data["engine_room"]
    return engine_room_from_mapping(data, base)


def engine_room_for(info: DrawingInfo, overrides: Optional[Mapping[str, Any] | str | Path] = None) -> EngineRoom:
    """Heuristic engine room for the drawing's ship type.

    Ship type keywords (VLCC/tanker, LNG/LPG, container, bulk) pick a :data:`SHIP_PROFILES`
    entry; decks are :data:`DEFAULT_DECKS` scaled by the profile. ``overrides`` (mapping or
    YAML path) and ``info.extra["engine_room"]`` replace any value.
    """
    prof = _profile_for(info.ship_type)
    decks = [(n, round(z * prof.deck_scale, 2)) for n, z in DEFAULT_DECKS]
    height = dict(decks)["upper_deck"]
    room = EngineRoom(
        length=prof.length, breadth=prof.breadth, height=height, decks=decks, ship_type=prof.key,
        source=f"heuristic:{prof.key}",
        notes=[f"engine room envelope is a heuristic for a {prof.note} "
               f"(ship type {info.ship_type or 'unknown'!s}); not a survey"],
    )
    extra = info.extra.get("engine_room") if isinstance(info.extra, Mapping) else None
    if isinstance(extra, Mapping):
        room = engine_room_from_mapping(extra, room)
    if isinstance(overrides, (str, Path)):
        room = load_engine_room(overrides, room)
    elif isinstance(overrides, Mapping):
        room = engine_room_from_mapping(overrides, room)
    return room
