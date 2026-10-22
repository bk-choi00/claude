"""Grouping of symbol strokes into symbol candidates.

Strokes are first joined into *touch clusters* (strokes that touch or overlap),
then clusters are merged when one lies inside a closed outline of another (the
two small circles of a screw pump, the plates of a cooler) or when they are
closer than a fraction of the symbol unit (the bars of a butterfly valve are not
connected to its disc).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import shapely

from .primitives import DSU, Prim, PrimSet, bbox_union, prim_distance

BBox4 = tuple[float, float, float, float]


@dataclass
class Candidate:
    id: int
    prim_ids: list[int]
    clusters: list[list[int]] = field(default_factory=list)  # touch clusters
    bbox: BBox4 = (0.0, 0.0, 0.0, 0.0)

    def refresh(self, prims: Sequence[Prim]) -> None:
        self.bbox = bbox_union(prims[i].bbox for i in self.prim_ids)

    @property
    def size(self) -> float:
        return max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


def touch_tol(unit: float, tol: float, width: float = 0.0) -> float:
    return max(0.6, min(tol, 0.06 * unit), 1.2 * width)


def group_candidates(ps: PrimSet, ids: Sequence[int], unit: float, tol: float, group_units: float = 0.12) -> list[Candidate]:
    prims = ps.prims
    ids = list(ids)
    if not ids:
        return []
    base_touch = touch_tol(unit, tol)
    gtol = max(base_touch, group_units * unit)
    pairs = ps.pairs_within(ids, gtol)
    dist = {}
    touch = DSU(len(prims))
    for i, j in pairs:
        d = prim_distance(prims[i], prims[j])
        dist[(i, j)] = d
        if d <= max(base_touch, 1.2 * max(prims[i].width, prims[j].width)):
            touch.union(i, j)

    group = DSU(len(prims))
    for i, j in pairs:
        if touch.find(i) == touch.find(j):
            group.union(i, j)
    # containment: strokes inside a closed outline (ring or circle) of another cluster
    containers = []
    owners = []
    idset = set(ids)
    for ring in ps.rings.values():
        if ring.dashed or not ring.prim_ids or ring.prim_ids[0] not in idset:
            continue
        containers.append(ring.polygon.buffer(base_touch))
        owners.append(ring.prim_ids[0])
    for i in ids:
        p = prims[i]
        if p.kind == "circle" and not p.filled and p.r > 0.12 * unit:
            containers.append(shapely.Point(p.cx, p.cy).buffer(p.r + base_touch, quad_segs=8))
            owners.append(i)
    if containers:
        tree = shapely.STRtree(containers)
        geoms = []
        for i in ids:
            p = prims[i]
            if p.kind == "circle":
                geoms.append(shapely.Point(p.cx, p.cy).buffer(max(p.r, 1e-3), quad_segs=4))
            else:
                geoms.append(shapely.LineString([(p.x0, p.y0), (p.x1, p.y1)]))
        gi, ci = tree.query(geoms, predicate="within")
        for g, c in zip(gi.tolist(), ci.tolist()):
            group.union(ids[g], owners[c])
    # proximity between clusters
    for i, j in pairs:
        if group.find(i) != group.find(j) and dist[(i, j)] <= gtol:
            group.union(i, j)

    out: list[Candidate] = []
    for members in group.groups(ids).values():
        members.sort()
        cl: dict[int, list[int]] = {}
        for m in members:
            cl.setdefault(touch.find(m), []).append(m)
        cand = Candidate(len(out), members, list(cl.values()))
        cand.refresh(prims)
        out.append(cand)
    out.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
    for k, c in enumerate(out):
        c.id = k
    return out
