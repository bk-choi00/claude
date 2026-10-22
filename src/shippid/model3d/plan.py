"""Layout plan: the fixed-vocabulary arrangement that Gemma (or the heuristic) produces.

The plan never contains coordinates. Each item says *where in words* an object goes
(zone, deck, transverse band, longitudinal third), optionally relative to an earlier item
(anchor + relation + distance bounds), with an orientation and an optional size. Items are
ordered: the solver places them one by one in this order, so an anchor must appear
before the items that reference it. Pipe geometry is never planned; the routing section
only sets preferred elevations and the routing priority.

``PlanItemDraft`` / ``LayoutPlanDraft`` are the model-facing schemas (what Gemma returns);
``PlanItem`` / ``LayoutPlan`` add provenance (``source`` / ``planner``).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Literal, Optional, Sequence

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from ..model import Category, PIDObject, SymbolClass
from .zones import DECK_NAMES, LONGITUDINALS, SIDES, ZONE_ALLOWED_DECKS, ZONE_DECK, ZONE_NAMES

ZoneName = Literal["double_bottom", "tank_top", "floor", "middle_platform", "upper_platform", "bottom",
                   "ship_side_port", "ship_side_starboard"]
DeckName = Literal["base_line", "tank_top", "floor", "3rd_deck", "2nd_deck", "upper_deck"]
Side = Literal["port", "center", "starboard"]
Longitudinal = Literal["aft", "mid", "fore"]
Relation = Literal["adjacent_to", "above", "below", "left_of", "right_of", "in_line_with", "near", "none"]
RELATIONS: tuple[str, ...] = ("adjacent_to", "above", "below", "left_of", "right_of", "in_line_with", "near", "none")

assert ZoneName.__args__ == ZONE_NAMES and DeckName.__args__ == DECK_NAMES  # type: ignore[attr-defined]
assert Side.__args__ == SIDES and Longitudinal.__args__ == LONGITUDINALS  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------- object roles

# Small equipment that normally sits in the pipe run; placed inline by the router unless
# the plan places it explicitly.
INLINE_EQUIPMENT: frozenset[SymbolClass] = frozenset({SymbolClass.STRAINER, SymbolClass.FILTER,
                                                      SymbolClass.FLOW_METER})
INLINE_CATEGORIES: frozenset[Category] = frozenset({Category.VALVE, Category.INSTRUMENT, Category.FITTING})


def is_planned(obj: PIDObject) -> bool:
    """Equipment-level object that the planner must place (equipment and off-page connectors)."""
    if obj.cls == SymbolClass.OFF_PAGE_CONNECTOR or obj.category == Category.CONNECTOR:
        return True
    return obj.category == Category.EQUIPMENT and obj.cls not in INLINE_EQUIPMENT


def is_optional_planned(obj: PIDObject) -> bool:
    """Inline-capable equipment the planner may place explicitly (strainer, filter, flow meter)."""
    return obj.category == Category.EQUIPMENT and obj.cls in INLINE_EQUIPMENT


# ---------------------------------------------------------------------------- normalisers

_SYNONYMS: dict[str, str] = {
    # sides
    "ps": "port", "p": "port", "portside": "port", "port_side": "port",
    "s": "starboard", "sb": "starboard", "stbd": "starboard", "stb": "starboard", "starboard_side": "starboard",
    "c": "center", "centre": "center", "centerline": "center", "centreline": "center", "center_line": "center",
    "centre_line": "center", "middle": "center", "cl": "center",
    # longitudinal
    "a": "aft", "after": "aft", "aftmost": "aft", "m": "mid", "midships": "mid", "midship": "mid",
    "amidships": "mid", "f": "fore", "forward": "fore", "fwd": "fore", "front": "fore",
    # zones
    "db": "double_bottom", "double_bottom_tank": "double_bottom", "inner_bottom": "double_bottom",
    "tanktop": "tank_top", "engine_room_floor": "floor", "floor_deck": "floor", "lower_platform": "floor",
    "3rd_deck_platform": "middle_platform", "middle_deck": "middle_platform", "third_deck": "3rd_deck",
    "2nd_deck_platform": "upper_platform", "second_deck": "2nd_deck", "upper_deck_platform": "upper_platform",
    "hull_bottom": "bottom", "sea_chest": "bottom", "shell_port": "ship_side_port",
    "shell_starboard": "ship_side_starboard", "side_port": "ship_side_port", "side_starboard": "ship_side_starboard",
    "baseline": "base_line", "base": "base_line", "bottom_shell": "base_line",
    # relations
    "adjacent": "adjacent_to", "next_to": "adjacent_to", "beside": "adjacent_to", "side_by_side": "adjacent_to",
    "on_top_of": "above", "over": "above", "under": "below", "beneath": "below", "inline_with": "in_line_with",
    "in_line": "in_line_with", "inline": "in_line_with", "aligned_with": "in_line_with", "close_to": "near",
    "near_to": "near", "port_of": "left_of", "starboard_of": "right_of", "": "none", "null": "none",
}


def _norm_word(value: Any) -> Any:
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_").replace(" ", "_").replace(".", "")
        return _SYNONYMS.get(key, key)
    return value


class _PlanBase(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class PlanItemDraft(_PlanBase):
    object_id: str = Field(description="id of the P&ID object, copied exactly (e.g. 'o12')")
    zone: ZoneName = Field(description="named engine-room zone the object goes into")
    deck: Optional[DeckName] = Field(default=None, description="deck the object stands on; null = zone default")
    side: Side = Field(default="center", description="transverse band: port, center or starboard")
    longitudinal: Longitudinal = Field(default="mid", description="longitudinal third: aft, mid or fore")
    anchor_id: Optional[str] = Field(default=None, description="id of an EARLIER item this one is placed relative to")
    relation: Relation = Field(default="none", description="spatial relation to the anchor")
    min_m: Optional[float] = Field(default=None, description="minimum edge-to-edge distance to the anchor (m)")
    max_m: Optional[float] = Field(default=None, description="maximum edge-to-edge distance to the anchor (m)")
    orientation_deg: Literal[0, 90, 180, 270] = Field(default=0, description="yaw: 0 = length along the ship")
    size_m: Optional[list[float]] = Field(default=None, description="[length, width, height] in metres, or null")
    group: Optional[str] = Field(default=None, description="group name for items kept together (e.g. pump train)")
    rationale: str = Field(default="", description="one short sentence: why here")

    @field_validator("zone", "deck", "side", "longitudinal", "relation", mode="before")
    @classmethod
    def _words(cls, v: Any) -> Any:
        return _norm_word(v)

    @field_validator("relation", mode="before")
    @classmethod
    def _relation_none(cls, v: Any) -> Any:
        return "none" if v is None else v

    @field_validator("side", "longitudinal", mode="before")
    @classmethod
    def _band_default(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None:
            return "center" if info.field_name == "side" else "mid"
        return v

    @field_validator("anchor_id", "group", mode="before")
    @classmethod
    def _empty_none(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().lower() in ("", "none", "null", "-"):
            return None
        return v

    @field_validator("orientation_deg", mode="before")
    @classmethod
    def _snap_orientation(cls, v: Any) -> Any:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0 if v is None else v
        if not math.isfinite(f):
            return 0
        return int(round(f / 90.0)) * 90 % 360

    @field_validator("min_m", "max_m", mode="before")
    @classmethod
    def _distance(cls, v: Any) -> Any:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) and 0.0 <= f <= 100.0 else None

    @field_validator("size_m", mode="before")
    @classmethod
    def _size(cls, v: Any) -> Any:
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            return None
        try:
            vals = [float(x) for x in v]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(x) and 0.02 <= x <= 60.0 for x in vals):
            return None
        return vals

    @field_validator("rationale", mode="before")
    @classmethod
    def _text(cls, v: Any) -> Any:
        return "" if v is None else str(v)[:300]


class PlanItem(PlanItemDraft):
    source: str = "heuristic"  # "gemma" | "mock" | "heuristic"


class ElevationBand(_PlanBase):
    target: str = Field(description="a medium value (e.g. 'ballast_water') or a line number")
    deck: DeckName = Field(default="tank_top", description="deck the elevation is measured from")
    height_m: float = Field(default=1.2, description="pipe centre-line height above that deck (m)")

    @field_validator("deck", mode="before")
    @classmethod
    def _words(cls, v: Any) -> Any:
        return _norm_word(v)

    @field_validator("height_m", mode="before")
    @classmethod
    def _height(cls, v: Any) -> Any:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 1.2
        return min(8.0, max(0.2, f)) if math.isfinite(f) else 1.2


class RoutingPlan(_PlanBase):
    elevation_bands: list[ElevationBand] = Field(default_factory=list,
                                                 description="preferred pipe elevations per medium or line")
    priority: list[str] = Field(default_factory=list, description="line numbers to route first")

    @field_validator("priority", mode="before")
    @classmethod
    def _priority(cls, v: Any) -> Any:
        if v is None:
            return []
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]


class LayoutPlanDraft(_PlanBase):
    items: list[PlanItemDraft] = Field(description="ordered placement list (anchors before dependants)")
    routing: RoutingPlan = Field(default_factory=RoutingPlan)
    notes: list[str] = Field(default_factory=list)

    @field_validator("notes", mode="before")
    @classmethod
    def _notes(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]

    @field_validator("routing", mode="before")
    @classmethod
    def _routing(cls, v: Any) -> Any:
        return {} if v is None else v


class LayoutPlan(_PlanBase):
    items: list[PlanItem] = Field(default_factory=list)
    routing: RoutingPlan = Field(default_factory=RoutingPlan)
    notes: list[str] = Field(default_factory=list)
    planner: str = "heuristic"  # "gemma" | "mock" | "heuristic"

    def item_by_id(self) -> dict[str, PlanItem]:
        return {it.object_id: it for it in self.items}


# ---------------------------------------------------------------------------- validation


def effective_deck(item: PlanItemDraft) -> str:
    """Deck the item stands on: the requested deck if the zone allows it, else the zone default."""
    if item.deck and item.deck in ZONE_ALLOWED_DECKS[item.zone]:
        return item.deck
    return ZONE_DECK[item.zone]


def check_plan(items: Sequence[PlanItemDraft], known_ids: Iterable[str], required_ids: Iterable[str] = (),
               preplaced_ids: Iterable[str] = ()) -> list[str]:
    """Structural problems of an ordered item list (empty list = valid).

    ``known_ids`` may be placed, ``required_ids`` must be placed, ``preplaced_ids`` were placed
    before this list (valid anchors).
    """
    known = set(known_ids)
    placed = set(preplaced_ids)
    problems: list[str] = []
    seen: set[str] = set()
    for i, it in enumerate(items):
        oid = it.object_id
        if oid not in known:
            problems.append(f"items[{i}]: unknown object_id {oid!r} (not in the object list)")
            continue
        if oid in seen:
            problems.append(f"items[{i}]: duplicate object_id {oid!r} (each object exactly once)")
            continue
        if it.anchor_id is not None:
            if it.anchor_id == oid:
                problems.append(f"{oid}: anchor_id refers to itself")
            elif it.anchor_id not in seen and it.anchor_id not in placed:
                where = "later in the list" if any(x.object_id == it.anchor_id for x in items[i + 1:]) else \
                    "not in the plan"
                problems.append(f"{oid}: anchor {it.anchor_id!r} is {where}; anchors must be placed earlier")
            if it.relation == "none" and it.max_m is None and it.min_m is None:
                problems.append(f"{oid}: anchor {it.anchor_id!r} given but relation is 'none'")
        elif it.relation != "none":
            problems.append(f"{oid}: relation {it.relation!r} needs an anchor_id")
        if it.deck and it.deck not in ZONE_ALLOWED_DECKS[it.zone]:
            problems.append(f"{oid}: deck {it.deck!r} is not valid for zone {it.zone!r} "
                            f"(allowed: {', '.join(ZONE_ALLOWED_DECKS[it.zone])})")
        if it.min_m is not None and it.max_m is not None and it.min_m > it.max_m:
            problems.append(f"{oid}: min_m {it.min_m} > max_m {it.max_m}")
        seen.add(oid)
    # anchor cycles (ignoring order), e.g. A anchored to B and B anchored to A
    g = nx.DiGraph()
    g.add_edges_from((it.object_id, it.anchor_id) for it in items if it.anchor_id and it.anchor_id != it.object_id)
    for cyc in nx.simple_cycles(g):
        problems.append("anchor cycle: " + " -> ".join(cyc + [cyc[0]]))
    missing = [r for r in required_ids if r not in seen and r not in placed]
    if missing:
        problems.append(f"missing items (every listed object needs exactly one item): {', '.join(missing)}")
    return problems


def sanitize_items(items: Sequence[PlanItemDraft], known_ids: Iterable[str],
                   preplaced_ids: Iterable[str] = ()) -> list[PlanItemDraft]:
    """Drop unknown/duplicate items and anchors that are not placed earlier; fix decks."""
    known = set(known_ids)
    placed = set(preplaced_ids)
    out: list[PlanItemDraft] = []
    for it in items:
        if it.object_id not in known or it.object_id in placed:
            continue
        upd: dict[str, Any] = {}
        if it.anchor_id is not None and (it.anchor_id == it.object_id or it.anchor_id not in placed):
            upd.update(anchor_id=None, relation="none", min_m=None, max_m=None)
        elif it.anchor_id is None and it.relation != "none":
            upd.update(relation="none")
        if it.deck and it.deck not in ZONE_ALLOWED_DECKS[it.zone]:
            upd["deck"] = None
        if it.min_m is not None and it.max_m is not None and it.min_m > it.max_m:
            upd.update(min_m=it.max_m, max_m=it.min_m)
        out.append(it.model_copy(update=upd) if upd else it)
        placed.add(it.object_id)
    return out
