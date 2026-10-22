"""Layout3D -> trimesh scene (GLB) with metadata, shared by the three.js and plotly viewers.

Scene graph (glTF node names are unique and three.js-safe: only ``[A-Za-z0-9_-]``)::

    engine_room                     root; matrix turns the z-up room frame into glTF +Y-up
      structure                     decks (translucent plates + grid) and the room outline
      sys_<system>                  one group per piping system      (extras.kind = "group")
        sys_<system>__<category>    equipment / valve / instrument / fitting / connector / pipe
          <TAG>_<id>                object or pipe node with metadata (extras.kind = "object"/"pipe")
            <TAG>_<id>__<part>      one mesh per part, one PBR material per mesh

Node extras (-> three.js ``userData``): id, tag, name, class, category, system, line_number, size,
medium, deck, rationale, dimensions_m, source. ``scene.metadata`` (-> glTF scene extras) holds the
drawing information, planner and counts.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from ..catalog import MEDIUM_COLORS, SYSTEMS
from ..model import Category, Layout3D, Medium, PIDDocument, PipeKind, Placement3D, SymbolClass, SystemType
from .heuristic import object_labels, object_systems, pipe_dn, pipe_medium, pipe_system, doc_system
from .meshes import Part, deck_parts, make_material, object_parts, outline_part, pipe_color, pipe_mesh
from .solver import axis_height

S = SymbolClass

# z-up room frame -> glTF y-up: (x, y, z) -> (x, z, -y)
Z_UP_TO_Y_UP = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1.0]])
_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def slugify(text: str, default: str = "node") -> str:
    """three.js-safe node name fragment (no whitespace or ``[ ] . : /``)."""
    s = _SAFE.sub("_", text or "").strip("_")
    return s or default


@dataclass
class SceneItem:
    node: str
    kind: str  # object | pipe | deck | outline
    system: str
    category: str
    transform: np.ndarray  # local -> room (z-up)
    parts: list[Part]
    extras: dict[str, Any] = field(default_factory=dict)


class _Names:
    def __init__(self) -> None:
        self.used: set[str] = set()

    def get(self, base: str) -> str:
        name = slugify(base)
        k, out = 1, name
        while out in self.used:
            k += 1
            out = f"{name}_{k}"
        self.used.add(out)
        return out


def _rz(deg: float) -> np.ndarray:
    return trimesh.transformations.rotation_matrix(math.radians(deg), (0, 0, 1))


def _tr(p) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = p
    return m


def placement_transform(p: Placement3D, category: Category) -> np.ndarray:
    """Local -> room transform; inline items on vertical runs are turned so local x follows the nozzles."""
    if category in (Category.VALVE, Category.FITTING) or p.cls in (S.STRAINER, S.FILTER, S.FLOW_METER):
        noz = p.nozzles
        a = noz.get("in") or (list(noz.values())[0] if len(noz) >= 2 else None)
        b = noz.get("out") or ([v for k, v in noz.items() if v is not a][0] if len(noz) >= 2 else None)
        if a is not None and b is not None:
            d = np.array(b.as_tuple()) - np.array(a.as_tuple())
            if abs(d[2]) > max(abs(d[0]), abs(d[1])) and abs(d[2]) > 1e-6:
                mid = (np.array(a.as_tuple()) + np.array(b.as_tuple())) / 2
                ang = -math.pi / 2 if d[2] > 0 else math.pi / 2  # Ry(-90) maps +x -> +z
                rot = trimesh.transformations.rotation_matrix(ang, (0, 1, 0))
                za = axis_height(p.cls, p.size.as_tuple())
                return _tr(mid) @ rot @ _tr((0, 0, -za))
    return _tr(p.position.as_tuple()) @ _rz(p.yaw_deg)


def _object_pipe_info(doc: PIDDocument) -> dict[str, dict[str, Any]]:
    """Main line / size / medium of the pipes attached to each object (largest size wins)."""
    best: dict[str, tuple[int, Any]] = {}
    lines: dict[str, list[str]] = {}
    for p in doc.pipes:
        if p.kind == PipeKind.SIGNAL:
            continue
        dn = pipe_dn(p) or 0
        for e in (p.start, p.end):
            if not e.object_id:
                continue
            if p.line_number:
                lines.setdefault(e.object_id, [])
                if p.line_number not in lines[e.object_id]:
                    lines[e.object_id].append(p.line_number)
            if e.object_id not in best or dn > best[e.object_id][0]:
                best[e.object_id] = (dn, p)
    out = {}
    for oid, (dn, p) in best.items():
        out[oid] = {"line_number": p.line_number, "size": p.size or (f"{dn}A" if dn else None),
                    "medium": pipe_medium(p).value, "lines": lines.get(oid, [])}
    return out


def _system_label(s: SystemType) -> str:
    info = SYSTEMS.get(s)
    return f"{info.name_en} / {info.name_ko}" if info else s.value


def scene_items(layout: Layout3D, doc: PIDDocument, outline: bool = True) -> list[SceneItem]:
    """Meshes (local frame + transform) and metadata of everything in the layout."""
    objs = doc.object_by_id()
    labels = object_labels(doc)
    systems = object_systems(doc)
    pinfo = _object_pipe_info(doc)
    default_sys = doc_system(doc)
    names = _Names()
    for reserved in ("engine_room", "structure"):
        names.used.add(reserved)
    items: list[SceneItem] = []
    for d in layout.decks:
        if d.z >= layout.bounds_max.z - 1e-6:  # the roof of the envelope is drawn by the outline only
            continue
        L = layout.bounds_max.x - layout.bounds_min.x
        B = layout.bounds_max.y - layout.bounds_min.y
        parts = deck_parts(L, B, d.z)
        tr = _tr((layout.bounds_min.x, (layout.bounds_min.y + layout.bounds_max.y) / 2, 0.0))
        items.append(SceneItem(names.get(f"deck_{d.name}"), "deck", "structure", "deck", tr, parts,
                               {"kind": "structure", "category": "deck", "deck": d.name, "z_m": d.z,
                                "name": f"deck {d.name} (z = {d.z:.2f} m)"}))
    if outline:
        bmin, bmax = layout.bounds_min, layout.bounds_max
        items.append(SceneItem(names.get("engine_room_outline"), "outline", "structure", "outline", np.eye(4),
                               [outline_part(bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z)],
                               {"kind": "structure", "category": "outline", "name": "engine room envelope"}))
    for p in layout.placements:
        o = objs.get(p.object_id)
        cls = o.cls if o else p.cls
        cat = o.category if o else Category.EQUIPMENT
        act = o.actuator if o else None
        parts = object_parts(cls, p.size.as_tuple(), act) if act is not None else object_parts(cls, p.size.as_tuple())
        info = pinfo.get(p.object_id, {})
        sys_ = systems.get(p.object_id, default_sys)
        tag = o.tag if o else None
        rationale = p.rationale or ""
        m = re.match(r"^\[(\w+)\]", rationale)
        extras = {
            "kind": "object", "id": p.object_id, "tag": tag,
            "name": labels.get(p.object_id) if labels.get(p.object_id) != tag else None, "class": cls.value,
            "category": cat.value, "system": sys_.value, "line_number": info.get("line_number"),
            "lines": info.get("lines", []), "size": info.get("size"), "medium": info.get("medium"),
            "deck": p.deck, "rationale": rationale,
            "dimensions_m": [round(p.size.x, 3), round(p.size.y, 3), round(p.size.z, 3)],
            "position_m": [round(p.position.x, 3), round(p.position.y, 3), round(p.position.z, 3)],
            "yaw_deg": p.yaw_deg, "source": m.group(1) if m else layout.planner,
            "actuator": o.actuator.value if o and o.actuator.value != "none" else None,
        }
        base = f"{tag}_{p.object_id}" if tag else f"{cls.value}_{p.object_id}"
        items.append(SceneItem(names.get(base), "object", sys_.value, cat.value, placement_transform(p, cat), parts,
                               extras))
    pipes = {pp.id: pp for pp in doc.pipes}
    for r in layout.routes:
        pp = pipes.get(r.pipe_id)
        pts = [v.as_tuple() for v in r.points]
        mesh = pipe_mesh(pts, r.diameter)
        if mesh is None:
            continue
        sys_ = pipe_system(pp, default_sys) if pp else default_sys
        length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        mc = MEDIUM_COLORS.get(r.medium, MEDIUM_COLORS[Medium.UNKNOWN])
        extras = {
            "kind": "pipe", "id": r.pipe_id, "tag": r.line_number, "class": "pipe", "category": "pipe",
            "system": sys_.value, "line_number": r.line_number, "size": pp.size if pp else None,
            "medium": r.medium.value, "iso14726": f"{mc[0]} ({mc[1]})", "diameter_m": round(r.diameter, 4),
            "length_m": round(length, 2),
            "from": (pp.start.object_id or pp.start.junction_id) if pp else None,
            "to": (pp.end.object_id or pp.end.junction_id) if pp else None,
            "pipe_kind": pp.kind.value if pp else None, "deck": None,
            "rationale": "orthogonal route (A* on a voxel grid) between the placed items",
        }
        base = f"{r.line_number}_{r.pipe_id}" if r.line_number else f"pipe_{r.pipe_id}"
        items.append(SceneItem(names.get(base), "pipe", sys_.value, "pipe", np.eye(4),
                               [Part("pipe", mesh, pipe_color(r.medium), 0.35, 0.45)], extras))
    return items


def layout_to_scene(layout: Layout3D, doc: PIDDocument, outline: bool = True) -> trimesh.Scene:
    """Build the trimesh scene (see module docstring)."""
    scene = trimesh.Scene()
    g = scene.graph
    info = doc.info
    g.update(frame_to="engine_room", frame_from=g.base_frame, matrix=Z_UP_TO_Y_UP,
             metadata={"kind": "root", "units": "m", "up_axis": "room frame z-up (x forward, y port); "
                                                             "this node rotates it to glTF +Y up"})
    g.update(frame_to="structure", frame_from="engine_room", matrix=np.eye(4),
             metadata={"kind": "group", "group_type": "structure", "label": "decks / outline"})
    items = scene_items(layout, doc, outline)
    names = {it.node for it in items} | {"engine_room", "structure"}
    groups: dict[tuple[str, str], str] = {}
    sys_nodes: dict[str, str] = {}

    def unique(base: str) -> str:
        k, out = 1, slugify(base)
        while out in names:
            k += 1
            out = f"{slugify(base)}_{k}"
        names.add(out)
        return out

    for it in items:
        if it.kind in ("deck", "outline"):
            parent = "structure"
        else:
            if it.system not in sys_nodes:
                sys_nodes[it.system] = unique(f"sys_{it.system}")
                st = SystemType(it.system) if it.system in SystemType._value2member_map_ else SystemType.UNKNOWN
                g.update(frame_to=sys_nodes[it.system], frame_from="engine_room", matrix=np.eye(4),
                         metadata={"kind": "group", "group_type": "system", "system": it.system,
                                   "label": _system_label(st)})
            key = (it.system, it.category)
            if key not in groups:
                groups[key] = unique(f"sys_{it.system}__{it.category}")
                g.update(frame_to=groups[key], frame_from=sys_nodes[it.system], matrix=np.eye(4),
                         metadata={"kind": "group", "group_type": "category", "system": it.system,
                                   "category": it.category})
            parent = groups[key]
        clean = {k: v for k, v in it.extras.items() if v is not None and v != []}
        g.update(frame_to=it.node, frame_from=parent, matrix=it.transform, metadata=clean)
        for part in it.parts:
            mesh = part.mesh.copy()
            mesh.metadata = {}
            pname = unique(f"{it.node}__{part.name}")
            mesh.visual = trimesh.visual.TextureVisuals(material=make_material(
                f"mat_{pname}", part.color, part.metallic, part.roughness, part.alpha))
            scene.add_geometry(mesh, node_name=pname, geom_name=pname, parent_node_name=it.node,
                               transform=np.eye(4), metadata={"part": part.name, "owner": it.node})
    counts: dict[str, int] = {}
    for it in items:
        counts[it.kind] = counts.get(it.kind, 0) + 1
    scene.metadata.update({
        "generator": "shippid.model3d",
        "units": "m",
        "planner": layout.planner,
        "drawing": {"drawing_no": info.drawing_no, "title": info.title, "system": info.system.value,
                    "ship_type": info.ship_type, "hull_no": info.hull_no, "revision": info.revision,
                    "source": doc.source},
        "counts": counts,
        "bounds_m": [layout.bounds_min.as_tuple(), layout.bounds_max.as_tuple()],
        "llm": {"backend": layout.llm.backend, "model": layout.llm.model, "calls": layout.llm.calls},
    })
    return scene


def export_glb(scene: trimesh.Scene, path: str | Path) -> Path:
    """Write the scene as binary glTF (with vertex normals)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = scene.export(file_type="glb", include_normals=True)
    path.write_bytes(data)
    return path


def read_glb_json(path: str | Path) -> dict[str, Any]:
    """The glTF JSON chunk of a GLB file (for checks and tests)."""
    import json
    import struct

    raw = Path(path).read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("not a glTF 2.0 binary")
    length, ctype = struct.unpack_from("<I4s", raw, 12)
    if ctype != b"JSON":
        raise ValueError("first GLB chunk is not JSON")
    return json.loads(raw[20:20 + length].decode("utf-8"))
