"""Top-level 3D layout builder: plan (Gemma or heuristic) -> solve -> route -> :class:`Layout3D`."""

from __future__ import annotations

import logging
import time
from typing import Optional

from ..llm.base import LLMClient
from ..model import Layout3D, LLMUsage, PIDDocument
from .heuristic import heuristic_plan
from .plan import LayoutPlan
from .router import route
from .solver import solve_detailed
from .zones import EngineRoom, engine_room_for

log = logging.getLogger(__name__)

DISCLAIMER = ("3D arrangement inferred from a P&ID (which has no spatial data): a plausible engine-room "
              "layout for visualisation, not a survey or a design")


def build_layout(doc: PIDDocument, client: Optional[LLMClient] = None, room: Optional[EngineRoom] = None,
                 *, grid_res: float = 0.5, route_time_budget: float = 120.0, planner_kwargs: Optional[dict] = None,
                 plan: Optional[LayoutPlan] = None) -> Layout3D:
    """Build a :class:`Layout3D` with a placement for every object and a route for every process pipe.

    ``client`` set: Gemma plans the arrangement (``planner`` = "gemma", or "mock" for the test
    backend) with the heuristic filling anything it leaves out; on failure the heuristic plan is
    used and the error is recorded in ``llm.errors`` and ``notes``. ``plan`` bypasses planning.
    """
    t0 = time.time()
    room = room or engine_room_for(doc.info)
    notes: list[str] = [DISCLAIMER] + list(room.notes)
    usage = LLMUsage()
    if plan is None and client is not None:
        from .gemma_planner import GemmaLayoutPlanner

        try:
            plan, usage = GemmaLayoutPlanner(client, **(planner_kwargs or {})).plan(doc, room)
        except Exception as exc:  # deterministic fallback, recorded honestly
            log.warning("Gemma layout planning failed: %s", exc)
            usage = LLMUsage(backend=client.backend.name, model=client.model,
                             errors=[f"layout planner failed: {str(exc)[:300]}"])
            notes.append(f"Gemma layout planning failed ({type(exc).__name__}); heuristic plan used")
            plan = None
    if plan is None:
        plan = heuristic_plan(doc, room)
    notes += [f"plan: {n}" for n in plan.notes]
    by_source: dict[str, int] = {}
    for it in plan.items:
        by_source[it.source] = by_source.get(it.source, 0) + 1
    notes.append(f"planner: {plan.planner}; items by source: " +
                 ", ".join(f"{k} {v}" for k, v in sorted(by_source.items())))
    res = solve_detailed(plan, doc, room)
    notes += res.notes
    notes += [f"placement: {msg}" for msg in res.issues()]
    routes, inline, rnotes = route(doc, res.placements, room, plan.routing, res=grid_res,
                                   time_budget=route_time_budget)
    notes += rnotes
    placements = res.placements + inline
    missing = [o.id for o in doc.objects if o.id not in {p.object_id for p in placements}]
    if missing:
        notes.append(f"objects without placement: {', '.join(missing)}")
    lo, hi = room.bounds_vec()
    notes.append(f"layout built in {time.time() - t0:.2f} s")
    return Layout3D(planner=plan.planner, decks=room.to_decks(), bounds_min=lo, bounds_max=hi,
                    placements=placements, routes=routes, notes=notes, llm=usage)
