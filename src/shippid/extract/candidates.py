"""Grouping of symbol strokes into symbol candidates.

Strokes are first joined into *touch clusters* (strokes that touch or overlap),
then clusters are merged when they are closer than a fraction of the symbol unit
(the bars of a butterfly valve are not connected to its disc) or when one lies
inside a closed outline of another (the two small circles of a screw pump, the
plates of a cooler). Containment only applies to outlines of symbol size (up to
``contain_units`` symbol units): a package boundary, a hull outline or a large tank
does not swallow what is drawn inside it. A candidate merged by containment keeps
its parts (outline group first) so the classifier can match them separately.
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
    parts: list[list[int]] = field(default_factory=list)  # containment merge: [outline group, inner groups...]

    def refresh(self, prims: Sequence[Prim]) -> None:
        self.bbox = bbox_union(prims[i].bbox for i in self.prim_ids)

    @property
    def size(self) -> float:
        return max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


def touch_tol(unit: float, tol: float, width: float = 0.0) -> float:
    return max(0.6, min(tol, 0.06 * unit), 1.2 * width)


def group_candidates(ps: PrimSet, ids: Sequence[int], unit: float, tol: float, group_units: float = 0.12,
                     contain_units: float = 4.5) -> list[Candidate]:
    prims = ps.prims
    ids = list(ids)
    if not ids:
        return []
    base_touch = touch_tol(unit, tol)
    gtol = max(base_touch, group_units * unit)
    pairs = ps.pairs_within(ids, gtol)
    touch = DSU(len(prims))
    base = DSU(len(prims))  # touch + proximity
    for i, j in pairs:
        d = prim_distance(prims[i], prims[j])
        if d <= max(base_touch, 1.2 * max(prims[i].width, prims[j].width)):
            touch.union(i, j)
        if d <= max(gtol, 1.2 * max(prims[i].width, prims[j].width)):
            base.union(i, j)

    group = DSU(len(prims))
    for i, j in pairs:
        if base.find(i) == base.find(j):
            group.union(i, j)
    # containment: strokes inside a closed outline (ring or circle) of symbol size
    max_size = contain_units * unit
    containers = []
    owners = []
    areas = []
    idset = set(ids)
    for ring in ps.rings.values():
        if ring.dashed or not ring.prim_ids or ring.prim_ids[0] not in idset:
            continue
        bb = ring.bbox
        if max(bb[2] - bb[0], bb[3] - bb[1]) > max_size:
            continue
        containers.append(ring.polygon.buffer(base_touch))
        owners.append(ring.prim_ids[0])
        areas.append((bb[2] - bb[0]) * (bb[3] - bb[1]))
    for i in ids:
        p = prims[i]
        if p.kind == "circle" and not p.filled and 0.12 * unit < p.r <= 0.5 * max_size:
            containers.append(shapely.Point(p.cx, p.cy).buffer(p.r + base_touch, quad_segs=8))
            owners.append(i)
            areas.append(4 * p.r * p.r)
    holder: dict[int, tuple[float, int]] = {}  # base group -> (area, owner) of the largest outline it sits in
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
            i, o = ids[g], owners[c]
            if base.find(i) == base.find(o):
                continue
            group.union(i, o)
            bi = base.find(i)
            if bi not in holder or areas[c] > holder[bi][0]:
                holder[bi] = (areas[c], o)

    out: list[Candidate] = []
    for members in group.groups(ids).values():
        members.sort()
        cl: dict[int, list[int]] = {}
        bg: dict[int, list[int]] = {}
        for m in members:
            cl.setdefault(touch.find(m), []).append(m)
            bg.setdefault(base.find(m), []).append(m)
        parts: list[list[int]] = []
        if len(bg) >= 2:
            # the outline group: holds the largest container and is itself inside none
            outer_b = None
            if any(b in holder for b in bg):
                best_owner = max((holder[b] for b in bg if b in holder), key=lambda t: t[0])[1]
                outer_b = base.find(best_owner)
                if outer_b in holder:  # nested outlines: walk up to the outermost
                    seen = {outer_b}
                    while outer_b in holder and base.find(holder[outer_b][1]) not in seen:
                        outer_b = base.find(holder[outer_b][1])
                        seen.add(outer_b)
            if outer_b is not None and outer_b in bg:
                parts = [bg[outer_b]] + [v for k, v in bg.items() if k != outer_b]
        cand = Candidate(len(out), members, list(cl.values()), parts=parts)
        cand.refresh(prims)
        out.append(cand)
    out.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
    for k, c in enumerate(out):
        c.id = k
    return out
