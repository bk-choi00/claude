"""Gemma 4 31B plans the engine-room arrangement in a fixed vocabulary; the solver places it.

Following LayoutGPT / Holodeck / LayoutVLM experience (docs/RESEARCH.md, "LLM-planned 3D"),
Gemma never outputs coordinates or pipe geometry. It receives the equipment-level objects
(valves, instruments and fittings are placed inline by the router), their connections (an
object-adjacency summary computed from pipes and junctions), size priors, the zone / deck
vocabulary and the room size, and returns an ordered :class:`LayoutPlanDraft`.

Each answer is validated (unknown / duplicate ids, anchors not placed earlier, anchor cycles,
missing items) and solved; structural problems plus the solver's diagnostics (relocations
because a band is full, size clipping, overlaps, unsatisfied relations) are fed back for up
to ``max_rounds`` repair rounds; the best round is kept. Items still missing come from
:func:`heuristic_plan`. Large drawings are planned in chunks of whole graph components
grouped by system; later chunks see (and may anchor to) the items already planned.

Provenance: items produced by the model carry ``source="gemma"`` (``"mock"`` for the test
backend, whose output must never pass for Gemma's); ``LayoutPlan.planner`` is that label only
if the model produced at least one item, else ``"heuristic"``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Optional

import networkx as nx

from ..llm.base import LLMClient, LLMError, Message
from ..model import LLMUsage, PIDDocument, SymbolClass
from .heuristic import (Neighbor, default_routing, doc_system, drawing_positions, equipment_adjacency,
                        flow_order, heuristic_plan, object_labels, object_systems, pipe_medium, process_pipes)
from .plan import (LayoutPlan, LayoutPlanDraft, PlanItem, PlanItemDraft, RoutingPlan, check_plan, is_optional_planned,
                   is_planned, sanitize_items)
from .solver import object_size, solve_detailed
from .zones import DECK_NAMES, LONGITUDINALS, SIDES, ZONE_ALLOWED_DECKS, ZONE_DECK, ZONE_NAMES, EngineRoom

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior marine engineer who plans engine-room machinery arrangements of large commercial ships (VLCC and other tankers, container ships, LNG carriers, bulk carriers).

TASK: you receive the equipment of one P&ID (piping and instrumentation diagram) and plan a plausible 3D arrangement of it inside the engine room. A P&ID has no spatial information, so you decide WHERE each item goes in words. A deterministic solver converts your plan into coordinates (respecting room bounds and a 0.6 m maintenance clearance) and pipes are routed automatically. Never output coordinates or pipe geometry.

FRAME: x forward from the aft engine-room bulkhead, y to port (centre line y = 0), z up from the base line; metres. "left_of" means on the port side of the anchor and "right_of" on the starboard side, looking forward.

ZONES (zone: what goes there; deck the item stands on)
{zones}
Every zone is split into transverse bands side = port | center | starboard (thirds of the breadth) and longitudinal thirds longitudinal = aft | mid | fore. In ship_side_port / ship_side_starboard the item is against the shell (side is implied) and deck selects the height (tank_top, floor, 3rd_deck, 2nd_deck).

MARINE ARRANGEMENT PRACTICE
- Main engine: tank_top, center, aft (in line with the shaft). Generator engines: floor at a side, fore. Boilers / exhaust-gas economisers: upper_platform.
- Sea chests (sea inlets): zone bottom, at a side (high sea chest) or center (low sea chest near the keel); sea-water, fire and ballast pumps stand on the tank_top close to their sea chest (short suction, positive suction head).
- Pumps generally stand on the tank_top near their suction source (tank or sea chest); parallel duty / standby pumps side by side (relation adjacent_to the first pump, same group, same orientation).
- Double-bottom tanks (bilge well, bilge holding, sludge, drain, LO sump, FO overflow): zone double_bottom. Water-ballast, peak and cargo-area tanks lie outside the engine room: put them in double_bottom, longitudinal fore, at the side given by their name ((P) = port, (S) = starboard).
- FO settling / service tanks: elevated structural tanks at the ship side (floor or middle_platform, port or starboard). Storage / bunker tanks: tank_top or double_bottom at the side. Expansion / header tanks: upper_platform.
- Coolers (shell-and-tube, central plate coolers), duplex filters, BWTS units and oily water separators: floor or tank_top near their pumps; the BWTS just downstream of the ballast pumps.
- Purifiers and their heaters together in the purifier room (middle_platform), purifiers side by side.
- Air compressors on the floor; air receivers stand vertical adjacent_to the compressors.
- Overboard discharges: ship_side_port / ship_side_starboard (the side in their name, else the side nearest their pumps), deck floor.
- Off-page connectors are small markers where a pipe continues on another drawing: put them near the item they connect to.
- Strainers, filters and flow meters marked optional normally sit in the pipe run; include them only if you want them placed as stand-alone equipment.
- Respect size: a band is about L/3 x B/3; do not crowd one band with more equipment than fits (sum of footprints plus clearances); spread groups over neighbouring bands instead.

OUTPUT: a JSON object {{"items": [...], "routing": {{...}}, "notes": [...]}}.
items: ORDERED placement list; the solver places them in this order. Exactly one item per required object, using its id. Fields:
- object_id: the object id copied exactly (e.g. "o12").
- zone: one of {zone_names}.
- deck: one of {deck_names}, or null for the zone's default deck (it must be one of the zone's allowed decks).
- side: port | center | starboard.
- longitudinal: aft | mid | fore.
- anchor_id: id of an item that appears EARLIER in your list (or in already_placed) that this item is positioned relative to; null if none. Never anchor to a later item; no cycles.
- relation: one of adjacent_to (same deck, edge gap between min_m and max_m, default 0.6-1.6 m), near (edge gap <= max_m, default 4 m, any deck), above / below (on a higher / lower deck over / under the anchor), left_of / right_of (port / starboard side of the anchor), in_line_with (aligned with the anchor along x or y), none. Use none when anchor_id is null.
- min_m, max_m: optional edge-to-edge distance bounds to the anchor in metres (null if not needed).
- orientation_deg: 0 (length along the ship), 90, 180 or 270.
- size_m: [length, width, height] in metres if the prior is unsuitable (e.g. a larger pump for a bigger line), else null.
- group: short name shared by items that belong together (e.g. "ballast_pumps"), or null.
- rationale: one short sentence (why this place).
routing: {{"elevation_bands": [{{"target": medium value or line number, "deck": deck name, "height_m": pipe centre-line height above that deck}}], "priority": [line numbers to route first, largest and most important first]}}. Typical: sea water / ballast / bilge mains under the floor plates on the tank_top (0.8-1.5 m above it), oil and fresh-water services under the platforms (2-3 m above the floor), compressed air and steam higher.
notes: a few short remarks about the arrangement (optional)."""

USER_INTRO = ("Plan the arrangement of the objects below. Required objects must each get exactly one item; "
              "optional objects may be included or left out. Items in already_placed are fixed and may be used "
              "as anchors.")


def _zone_text(room: EngineRoom) -> str:
    lines = []
    for name in ZONE_NAMES:
        z = room.zone(name)
        decks = ", ".join(ZONE_ALLOWED_DECKS[name])
        lines.append(f"- {name}: {z.description}; stands on {ZONE_DECK[name]} "
                     f"(allowed decks: {decks}); z {z.box.z0:.1f}-{z.box.z1:.1f} m")
    return "\n".join(lines)


def system_prompt(room: EngineRoom) -> str:
    return SYSTEM_PROMPT.format(zones=_zone_text(room), zone_names=", ".join(ZONE_NAMES),
                                deck_names=", ".join(DECK_NAMES))


def _to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


class GemmaLayoutPlanner:
    """Gemma-driven arrangement planner with solver feedback and heuristic fallback."""

    def __init__(self, client: LLMClient, max_rounds: int = 3, chunk_size: int = 60, thinking: bool = True,
                 max_tokens: Optional[int] = None, max_feedback_lines: int = 40):
        self.client = client
        self.max_rounds = max(0, int(max_rounds))
        self.chunk_size = max(4, int(chunk_size))
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.max_feedback_lines = max_feedback_lines
        # provenance label: output of the test MockBackend must never pass for Gemma output
        self.label = "mock" if client.backend.name == "mock" else "gemma"
        self.log: list[dict[str, Any]] = []  # per chunk / round record (for reports and tests)

    # -- public ------------------------------------------------------------------
    def plan(self, doc: PIDDocument, room: EngineRoom) -> tuple[LayoutPlan, LLMUsage]:
        before = self.client.usage.model_copy(deep=True)
        errors: list[str] = []
        objs = doc.object_by_id()
        required = [o.id for o in doc.objects if is_planned(o)]
        optional = [o.id for o in doc.objects if is_optional_planned(o)]
        candidates = required + optional
        adjacency = equipment_adjacency(doc, set(candidates))
        req_adjacency = equipment_adjacency(doc, set(required))
        reference = heuristic_plan(doc, room, adjacency=req_adjacency)
        ref_items = {it.object_id: it for it in reference.items}
        chunks = self._chunks(doc, candidates, adjacency)
        final: list[PlanItem] = []
        routing: Optional[RoutingPlan] = None
        notes: list[str] = []
        for ci, chunk in enumerate(chunks):
            req_chunk = [i for i in chunk if i in set(required)]
            try:
                items, draft_routing, draft_notes = self._plan_chunk(doc, room, chunk, req_chunk, final, adjacency,
                                                                     ref_items, ci, len(chunks))
            except LLMError as exc:
                msg = f"chunk {ci + 1}/{len(chunks)}: no valid plan from the model ({str(exc)[:200]})"
                errors.append(msg)
                notes.append(msg + "; heuristic used for this chunk")
                items, draft_routing, draft_notes = [], None, []
            final.extend(items)
            if draft_routing is not None and routing is None:
                routing = draft_routing
            notes.extend(draft_notes)
            fill = self._fill(req_chunk, final, ref_items, doc)
            if fill:
                notes.append(f"chunk {ci + 1}: heuristic placement for {len(fill)} item(s) the model left out: "
                             + ", ".join(it.object_id for it in fill))
            final.extend(fill)
        n_model = sum(1 for it in final if it.source == self.label)
        planner = self.label if n_model else "heuristic"
        plan = LayoutPlan(items=final, routing=self._merge_routing(doc, routing), planner=planner,
                          notes=[f"{self.label} planned {n_model}/{len(final)} items in {len(chunks)} chunk(s)"] + notes)
        usage = self._usage_delta(before, errors)
        return plan, usage

    # -- chunking ------------------------------------------------------------------
    def _chunks(self, doc: PIDDocument, candidates: list[str], adjacency: dict[str, list[Neighbor]]) -> list[list[str]]:
        """Whole graph components grouped by system, packed into chunks of <= chunk_size items."""
        if len(candidates) <= self.chunk_size:
            return [flow_order(doc, candidates, adjacency)] if candidates else []
        g = nx.Graph()
        g.add_nodes_from(candidates)
        for a in candidates:
            for n in adjacency.get(a, []):
                if n.id in g:
                    g.add_edge(a, n.id)
        systems = object_systems(doc)
        comps = []
        for comp in nx.connected_components(g):
            order = flow_order(doc, sorted(comp), adjacency)
            votes: dict[str, int] = defaultdict(int)
            for i in order:
                votes[systems[i].value] += 1
            sys_ = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
            comps.append((sys_, -len(order), order[0], order))
        comps.sort()
        chunks: list[list[str]] = []
        cur: list[str] = []
        for _, _, _, order in comps:
            pieces = [order[i:i + self.chunk_size] for i in range(0, len(order), self.chunk_size)]
            for piece in pieces:
                if cur and len(cur) + len(piece) > self.chunk_size:
                    chunks.append(cur)
                    cur = []
                cur = cur + piece
        if cur:
            chunks.append(cur)
        return chunks

    # -- prompting -----------------------------------------------------------------
    def _digest(self, doc: PIDDocument, room: EngineRoom, chunk: list[str], required: list[str],
                placed: list[PlanItem], adjacency: dict[str, list[Neighbor]], ref_items: dict[str, PlanItem]
                ) -> dict[str, Any]:
        objs = doc.object_by_id()
        labels = object_labels(doc)
        systems = object_systems(doc)
        pos = drawing_positions(doc)
        req = set(required)
        objects = []
        for oid in chunk:
            o = objs[oid]
            nbs = adjacency.get(oid, [])
            dn = max((n.dn or 0 for n in nbs), default=0) or None
            ref = ref_items.get(oid)
            size, _ = object_size(o, room, ref.zone if ref else "floor", labels.get(oid, ""), dn)
            conns = []
            for n in sorted(nbs, key=lambda n: (-(n.dn or 0), n.id))[:10]:
                c: dict[str, Any] = {"to": n.id, "flow": n.direction or "unknown"}
                if n.lines:
                    c["lines"] = n.lines[:3]
                if n.dn:
                    c["dn"] = n.dn
                if n.medium.value != "unknown":
                    c["medium"] = n.medium.value
                if n.via:
                    c["via"] = [objs[v].cls.value for v in n.via[:6] if v in objs]
                conns.append(c)
            entry: dict[str, Any] = {"id": oid, "class": o.cls.value}
            if o.tag:
                entry["tag"] = o.tag
            if labels.get(oid) and labels.get(oid) != o.tag:
                entry["name"] = labels[oid]
            entry.update({"system": systems[oid].value, "required": oid in req,
                          "size_prior_m": [round(x, 2) for x in size],
                          "drawing_uv": [round(pos[oid][0], 3), round(pos[oid][1], 3)],
                          "connections": conns})
            objects.append(entry)
        already = [{"id": it.object_id, "class": objs[it.object_id].cls.value, "tag": objs[it.object_id].tag,
                    "zone": it.zone, "side": it.side, "longitudinal": it.longitudinal} for it in placed]
        lines = sorted({p.line_number for p in process_pipes(doc) if p.line_number})
        media = sorted({pipe_medium(p).value for p in process_pipes(doc)})
        return {
            "drawing": {"title": doc.info.title, "drawing_no": doc.info.drawing_no, "system": doc_system(doc).value,
                        "ship_type": doc.info.ship_type},
            "engine_room": room.describe(),
            "zone_vocabulary": list(ZONE_NAMES), "sides": list(SIDES), "longitudinal": list(LONGITUDINALS),
            "drawing_uv_note": "normalised drawing position (u right, v down; bow is usually drawn to the right "
                               "and port at the top) - a weak hint only",
            "already_placed": already,
            "objects": objects,
            "media": media,
            "line_numbers": lines[:200],
        }

    def _messages(self, doc: PIDDocument, room: EngineRoom, digest: dict[str, Any], ci: int, n_chunks: int
                  ) -> list[Message]:
        part = f" (part {ci + 1} of {n_chunks}; other parts are planned separately)" if n_chunks > 1 else ""
        return [Message("system", system_prompt(room)),
                Message("user", USER_INTRO + part + "\n\n" + _to_json(digest))]

    # -- rounds --------------------------------------------------------------------
    def _plan_chunk(self, doc: PIDDocument, room: EngineRoom, chunk: list[str], required: list[str],
                    placed: list[PlanItem], adjacency: dict[str, list[Neighbor]], ref_items: dict[str, PlanItem],
                    ci: int, n_chunks: int) -> tuple[list[PlanItem], Optional[RoutingPlan], list[str]]:
        digest = self._digest(doc, room, chunk, required, placed, adjacency, ref_items)
        msgs = self._messages(doc, room, digest, ci, n_chunks)
        placed_ids = [it.object_id for it in placed]
        best: Optional[tuple[int, int, list[PlanItem], LayoutPlanDraft]] = None
        for rnd in range(self.max_rounds + 1):
            draft = self.client.generate_json(msgs, LayoutPlanDraft, max_tokens=self.max_tokens,
                                              thinking=True if self.thinking else None)
            problems = check_plan(draft.items, chunk, required, placed_ids)
            items = [PlanItem(**it.model_dump(), source=self.label)
                     for it in sanitize_items(draft.items, chunk, placed_ids)]
            fill = self._fill(required, placed + items, ref_items, doc)
            trial = LayoutPlan(items=placed + items + fill, planner=self.label)
            res = solve_detailed(trial, doc, room)
            issues = res.issues([it.object_id for it in items])
            feedback = problems + issues
            score = len(problems) * 2 + len(issues)
            self.log.append({"chunk": ci, "round": rnd, "items": len(draft.items), "problems": problems,
                             "issues": issues})
            if best is None or score < best[0]:
                best = (score, rnd, items, draft)
            if not feedback:
                break
            if rnd < self.max_rounds:
                shown = feedback[: self.max_feedback_lines]
                more = len(feedback) - len(shown)
                msgs = msgs + [
                    Message("assistant", _to_json(draft.model_dump(mode="json"))[-12000:]),
                    Message("user", "The solver checked your plan and reported these problems:\n- " +
                            "\n- ".join(shown) + (f"\n(+{more} more)" if more > 0 else "") +
                            "\nReturn the COMPLETE corrected plan (all required items, same JSON format). Fix "
                            "unknown/duplicate/missing ids and anchors first; move items out of crowded bands, "
                            "choose other zones, sides or longitudinal thirds, or adjust relations and sizes."),
                ]
        assert best is not None
        _, rnd, items, draft = best
        notes = [f"chunk {ci + 1}: best plan from round {rnd + 1} ({len(items)} items)"]
        notes += [f"model note: {n}"[:300] for n in draft.notes[:5]]
        return items, self._valid_routing(doc, draft.routing), notes

    def _fill(self, required: list[str], placed: list[PlanItem], ref_items: dict[str, PlanItem],
              doc: PIDDocument) -> list[PlanItem]:
        """Heuristic items for required ids not yet placed (anchors kept only if already placed)."""
        have = {it.object_id for it in placed}
        out: list[PlanItem] = []
        for oid in [i for i in ref_items if i in set(required)]:
            if oid in have:
                continue
            it = ref_items[oid]
            if it.anchor_id and it.anchor_id not in have:
                it = it.model_copy(update={"anchor_id": None, "relation": "none", "max_m": None, "min_m": None})
            out.append(it.model_copy(update={"source": "heuristic"}))
            have.add(oid)
        return out

    def _valid_routing(self, doc: PIDDocument, routing: RoutingPlan) -> Optional[RoutingPlan]:
        lines = {p.line_number.strip().upper() for p in process_pipes(doc) if p.line_number}
        media = {pipe_medium(p).value.upper() for p in process_pipes(doc)}
        bands = [b for b in routing.elevation_bands if b.target.strip().upper() in lines | media]
        prio = [ln for ln in routing.priority if ln.strip().upper() in lines]
        if not bands and not prio:
            return None
        return RoutingPlan(elevation_bands=bands, priority=prio)

    def _merge_routing(self, doc: PIDDocument, routing: Optional[RoutingPlan]) -> RoutingPlan:
        base = default_routing(doc)
        if routing is None:
            return base
        targets = {b.target.strip().upper() for b in routing.elevation_bands}
        bands = list(routing.elevation_bands) + [b for b in base.elevation_bands
                                                 if b.target.strip().upper() not in targets]
        prio = list(dict.fromkeys(list(routing.priority) + base.priority))
        return RoutingPlan(elevation_bands=bands, priority=prio)

    def _usage_delta(self, before: LLMUsage, errors: list[str]) -> LLMUsage:
        u = self.client.usage
        return LLMUsage(backend=u.backend, model=u.model, calls=u.calls - before.calls,
                        prompt_tokens=u.prompt_tokens - before.prompt_tokens,
                        completion_tokens=u.completion_tokens - before.completion_tokens,
                        seconds=round(u.seconds - before.seconds, 3),
                        errors=list(u.errors[len(before.errors):]) + errors)
