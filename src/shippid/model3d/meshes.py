"""Parametric trimesh meshes per :class:`SymbolClass`, pipes, decks and materials.

Every object is built in its *local frame*: origin at the centre of the footprint at the base,
x along the object's length (flow axis for inline items), y across, z up, overall size
``(l, w, h)`` = :attr:`Placement3D.size`. Inline items have their pipe axis at
:func:`shippid.model3d.solver.axis_height`. Each part carries its own sRGB colour; the scene
turns it into one :class:`trimesh.visual.material.PBRMaterial` per mesh with linear colours.

Equipment uses neutral greys / blues, valves are coloured by function (isolation, non-return,
control, safety) with the actuator in its own colour, pipes use ISO 14726 colours
(:func:`shippid.catalog.medium_rgb`).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
import trimesh
from trimesh.visual.color import srgb_to_linear
from trimesh.visual.material import PBRMaterial

from ..catalog import medium_rgb
from ..model import ActuatorType, Category, Medium, SymbolClass
from ..symbols import get_symbol
from .solver import axis_height

S = SymbolClass
RGB = tuple[float, float, float]

# --------------------------------------------------------------------------- palette (sRGB 0-1)
COLORS: dict[str, RGB] = {
    "casing": (0.30, 0.44, 0.62),  # pump / compressor casings (steel blue)
    "motor": (0.22, 0.27, 0.34),
    "base": (0.36, 0.37, 0.39),
    "tank": (0.80, 0.81, 0.82),
    "vessel": (0.70, 0.74, 0.78),
    "hx": (0.46, 0.58, 0.70),
    "plates": (0.78, 0.80, 0.83),
    "engine": (0.42, 0.47, 0.50),
    "engine_dark": (0.28, 0.31, 0.34),
    "unit": (0.62, 0.67, 0.72),
    "cabinet": (0.86, 0.87, 0.84),
    "sea_chest": (0.28, 0.46, 0.42),
    "grating": (0.20, 0.22, 0.22),
    "connector": (0.95, 0.72, 0.18),
    "fitting": (0.50, 0.50, 0.53),
    "dial": (0.96, 0.96, 0.96),
    "rim": (0.12, 0.12, 0.12),
    "unknown": (0.65, 0.40, 0.65),
    "deck": (0.62, 0.66, 0.70),
    "grid": (0.40, 0.44, 0.48),
    "outline": (0.35, 0.38, 0.42),
}
VALVE_FAMILY: dict[SymbolClass, str] = {
    S.GATE_VALVE: "isolation", S.GLOBE_VALVE: "isolation", S.BALL_VALVE: "isolation", S.BUTTERFLY_VALVE: "isolation",
    S.CHECK_VALVE: "non_return", S.SDNR_VALVE: "non_return", S.THREE_WAY_VALVE: "control",
    S.SAFETY_VALVE: "safety",
}
VALVE_COLORS: dict[str, RGB] = {
    "isolation": (0.20, 0.36, 0.58),
    "non_return": (0.12, 0.55, 0.55),
    "control": (0.55, 0.30, 0.66),
    "safety": (0.88, 0.46, 0.10),
}
ACTUATOR_COLORS: dict[ActuatorType, RGB] = {
    ActuatorType.NONE: (0.30, 0.30, 0.32),
    ActuatorType.HANDWHEEL: (0.78, 0.12, 0.12),
    ActuatorType.HYDRAULIC: (0.95, 0.78, 0.15),
    ActuatorType.ELECTRIC: (0.15, 0.36, 0.80),
    ActuatorType.DIAPHRAGM: (0.86, 0.86, 0.90),
    ActuatorType.QUICK_CLOSING: (0.85, 0.12, 0.12),
}


@dataclass
class Part:
    name: str
    mesh: trimesh.Trimesh
    color: RGB  # sRGB 0-1
    metallic: float = 0.25
    roughness: float = 0.6
    alpha: float = 1.0


def make_material(name: str, color: RGB, metallic: float = 0.25, roughness: float = 0.6,
                  alpha: float = 1.0, emissive: Optional[RGB] = None) -> PBRMaterial:
    """PBR material with the sRGB colour converted to linear (glTF baseColorFactor is linear)."""
    lin = srgb_to_linear(np.array(color, dtype=np.float64))
    rgba = [float(lin[0]), float(lin[1]), float(lin[2]), float(alpha)]
    return PBRMaterial(name=name, baseColorFactor=rgba, metallicFactor=float(metallic),
                       roughnessFactor=float(roughness), alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
                       doubleSided=alpha < 1.0,
                       emissiveFactor=None if emissive is None else list(srgb_to_linear(np.array(emissive))))


# --------------------------------------------------------------------------- primitive helpers


def _tr(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


_ROT = {
    "x": trimesh.transformations.rotation_matrix(math.pi / 2, (0, 1, 0)),
    "y": trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0)),
    "z": np.eye(4),
}


def box(l: float, w: float, h: float, cx: float = 0.0, cy: float = 0.0, z0: float = 0.0) -> trimesh.Trimesh:
    """Box of size (l, w, h) centred at (cx, cy) with its bottom at z0."""
    return trimesh.creation.box(extents=(max(l, 1e-3), max(w, 1e-3), max(h, 1e-3)), transform=_tr(cx, cy, z0 + h / 2))


def cyl(r: float, length: float, axis: str = "z", c: Sequence[float] = (0, 0, 0), sections: int = 20
        ) -> trimesh.Trimesh:
    """Cylinder of radius ``r`` and ``length`` along ``axis`` centred at ``c``."""
    m = trimesh.creation.cylinder(radius=max(r, 1e-3), height=max(length, 1e-3), sections=sections)
    m.apply_transform(_ROT[axis])
    m.apply_translation(c)
    return m


def sphere(r: float, c: Sequence[float], subdivisions: int = 2) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=max(r, 1e-3))
    m.apply_translation(c)
    return m


def frustum(r0: float, r1: float, length: float, axis: str = "x", c: Sequence[float] = (0, 0, 0),
            sections: int = 20) -> trimesh.Trimesh:
    """Cone frustum (reducer / dome) from radius r0 to r1 along ``axis``, centred at ``c``."""
    m = trimesh.creation.revolve([[0, 0], [max(r0, 1e-3), 0], [max(r1, 1e-3), length], [0, length]],
                                 sections=sections)
    m.apply_translation((0, 0, -length / 2))
    m.apply_transform(_ROT[axis])
    m.apply_translation(c)
    return m


def capsule(r: float, total: float, c: Sequence[float]) -> trimesh.Trimesh:
    """Vertical capsule (vessel with dished ends) of overall height ``total``."""
    m = trimesh.creation.capsule(height=max(total - 2 * r, 1e-3), radius=max(r, 1e-3), count=[16, 12])
    m.apply_translation(c)
    return m


def torus(R: float, r: float, c: Sequence[float]) -> trimesh.Trimesh:
    m = trimesh.creation.torus(major_radius=max(R, 2e-3), minor_radius=max(r, 1e-3), major_sections=20,
                               minor_sections=6)
    m.apply_translation(c)
    return m


def chamfered_box(l: float, w: float, h: float, ch: float) -> trimesh.Trimesh:
    """Tank body: box with chamfered top edges (bottom at z = 0)."""
    ch = max(0.0, min(ch, l / 3, w / 3, h / 3))
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            pts += [(sx * l / 2, sy * w / 2, 0), (sx * l / 2, sy * w / 2, h - ch),
                    (sx * (l / 2 - ch), sy * (w / 2 - ch), h)]
    return trimesh.convex.convex_hull(np.array(pts))


def _merge(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    return trimesh.util.concatenate([m for m in meshes if m is not None and len(m.faces)])


# --------------------------------------------------------------------------- equipment


def _pump_centrifugal(l, w, h):
    r = 0.30 * min(l, w)
    zc = 0.25 * h
    return [
        Part("base", box(0.9 * l, w, 0.08 * h), COLORS["base"]),
        Part("casing", _merge([cyl(r, 0.30 * h, "z", (0, 0, zc)),
                               cyl(0.12 * w + 0.02, l, "x", (0, 0, zc)),
                               cyl(0.7 * r, 0.12 * h, "z", (0, 0, 0.46 * h))]), COLORS["casing"], 0.4, 0.5),
        Part("motor", _merge([cyl(0.38 * min(l, w), 0.38 * h, "z", (0, 0, 0.72 * h)),
                              cyl(0.22 * min(l, w), 0.06 * h, "z", (0, 0, 0.94 * h))]), COLORS["motor"], 0.3, 0.5),
    ]


def _pump_screw(l, w, h):
    r = 0.36 * min(w, h)
    return [
        Part("base", box(l, 0.9 * w, 0.12 * h), COLORS["base"]),
        Part("casing", _merge([cyl(r, 0.5 * l, "x", (-0.22 * l, 0, 0.5 * h)),
                               cyl(0.45 * r, 0.25 * h, "z", (-0.3 * l, 0, 0.85 * h))]), COLORS["casing"], 0.4, 0.5),
        Part("motor", _merge([cyl(1.1 * r, 0.42 * l, "x", (0.28 * l, 0, 0.55 * h)),
                              cyl(0.3 * r, 0.1 * l, "x", (0.03 * l, 0, 0.5 * h))]), COLORS["motor"]),
    ]


def _shell_hx(l, w, h, color="hx"):
    r = 0.45 * min(w, h)
    zc = 0.55 * h if h > 2 * r else r
    sl = 0.78 * l
    return [
        Part("shell", _merge([cyl(r, sl, "x", (0, 0, zc)),
                              cyl(1.12 * r, 0.04 * l, "x", (-sl / 2, 0, zc)),
                              cyl(1.12 * r, 0.04 * l, "x", (sl / 2, 0, zc)),
                              cyl(0.18 * r, h - zc, "z", (-0.3 * l, 0, zc + (h - zc) / 2))]), COLORS[color], 0.4, 0.45),
        Part("heads", _merge([frustum(r, 0.55 * r, (l - sl) / 2, "x", (-(sl + (l - sl) / 2) / 2, 0, zc)),
                              frustum(0.55 * r, r, (l - sl) / 2, "x", ((sl + (l - sl) / 2) / 2, 0, zc))]),
             COLORS["casing"], 0.4, 0.45),
        Part("saddles", _merge([box(0.08 * l, 1.4 * r, zc, -0.3 * l, 0), box(0.08 * l, 1.4 * r, zc, 0.3 * l, 0)]),
             COLORS["base"]),
    ]


def _plate_hx(l, w, h):
    return [
        Part("frame", _merge([box(0.1 * l, w, h, -0.45 * l, 0), box(0.1 * l, 0.9 * w, 0.95 * h, 0.45 * l, 0),
                              box(l, 0.12 * w, 0.06 * h, 0, 0, 0.94 * h), box(l, 0.12 * w, 0.06 * h, 0, 0, 0.0)]),
             COLORS["hx"], 0.4, 0.45),
        Part("plates", box(0.68 * l, 0.82 * w, 0.82 * h, -0.02 * l, 0, 0.08 * h), COLORS["plates"], 0.6, 0.35),
    ]


def _filter(l, w, h, duplex=False):
    r = 0.38 * min(l, w if not duplex else w * 0.9)
    pots = [(0.0, 0.0)] if not duplex else [(0.0, -0.25 * w), (0.0, 0.25 * w)]
    if duplex:
        r = 0.22 * min(l, w) + 0.05
    body = [cyl(r, 0.65 * h, "z", (px, py, 0.1 * h + 0.325 * h)) for px, py in pots]
    covers = [cyl(1.1 * r, 0.06 * h, "z", (px, py, 0.78 * h)) for px, py in pots]
    parts = [Part("body", _merge(body + [cyl(0.25 * min(w, h), l, "x", (0, 0, 0.7 * h))]), COLORS["vessel"], 0.4, 0.5),
             Part("cover", _merge(covers), COLORS["casing"], 0.4, 0.5)]
    if duplex:
        parts.append(Part("changeover", box(0.3 * l, 0.3 * w, 0.18 * h, 0, 0, 0.82 * h), COLORS["motor"]))
    return parts


def _strainer(l, w, h):
    return [Part("body", _merge([box(0.7 * l, 0.8 * w, 0.75 * h, 0, 0, 0.0),
                                 cyl(0.3 * min(w, h), l, "x", (0, 0, 0.5 * h))]),
                 COLORS["vessel"], 0.4, 0.5),
            Part("cover", cyl(0.35 * min(l, w), 0.12 * h, "z", (0, 0, 0.81 * h)), COLORS["casing"])]


def _sea_chest(l, w, h):
    return [Part("box", box(l, w, 0.9 * h, 0, 0, 0.1 * h), COLORS["sea_chest"], 0.3, 0.7),
            Part("grating", box(l, w, 0.1 * h), COLORS["grating"], 0.5, 0.6)]


def _overboard(l, w, h):
    r = 0.4 * min(w, h)
    return [Part("shell_valve", _merge([cyl(r, 0.6 * l, "x", (0, 0, h / 2)),
                                        cyl(1.2 * r, 0.1 * l, "x", (0.35 * l, 0, h / 2))]),
                 COLORS["base"], 0.5, 0.5)]


def _purifier(l, w, h):
    r = 0.3 * min(l, w)
    return [Part("frame", box(l, w, 0.4 * h), COLORS["unit"], 0.3, 0.5),
            Part("bowl", _merge([cyl(r, 0.35 * h, "z", (-0.1 * l, 0, 0.575 * h)),
                                 frustum(r, 0.4 * r, 0.25 * h, "z", (-0.1 * l, 0, 0.875 * h))]),
                 COLORS["vessel"], 0.5, 0.4),
            Part("motor", cyl(0.18 * min(l, w), 0.35 * h, "z", (0.3 * l, 0, 0.575 * h)), COLORS["motor"])]


def _compressor(l, w, h):
    return [Part("frame", box(l, w, 0.12 * h), COLORS["base"]),
            Part("block", _merge([box(0.5 * l, 0.7 * w, 0.45 * h, -0.2 * l, 0, 0.12 * h),
                                  cyl(0.15 * min(l, w), 0.35 * h, "z", (-0.3 * l, 0, 0.75 * h)),
                                  cyl(0.15 * min(l, w), 0.35 * h, "z", (-0.1 * l, 0, 0.75 * h))]),
                 COLORS["casing"], 0.4, 0.5),
            Part("motor", cyl(0.3 * min(w, h), 0.4 * l, "x", (0.28 * l, 0, 0.42 * h)), COLORS["motor"])]


def _air_receiver(l, w, h):
    r = 0.45 * min(l, w)
    legs = [cyl(0.03 + 0.03 * r, 0.18 * h, "z", (0.7 * r * math.cos(a), 0.7 * r * math.sin(a), 0.09 * h))
            for a in (0.0, 2.094, 4.189)]
    return [Part("vessel", capsule(r, 0.82 * h, (0, 0, 0.18 * h + 0.41 * h)), COLORS["vessel"], 0.5, 0.4),
            Part("legs", _merge(legs), COLORS["base"])]


def _ows(l, w, h):
    r = 0.3 * min(l, w)
    return [Part("skid", box(l, w, 0.1 * h), COLORS["base"]),
            Part("separator", capsule(r, 0.85 * h, (-0.2 * l, 0, 0.1 * h + 0.425 * h)), COLORS["vessel"], 0.4, 0.5),
            Part("cabinet", box(0.3 * l, 0.6 * w, 0.6 * h, 0.3 * l, 0, 0.1 * h), COLORS["cabinet"])]


def _bwts(l, w, h):
    r = 0.32 * min(w, h)
    return [Part("skid", box(l, w, 0.1 * h), COLORS["base"]),
            Part("reactor", _merge([cyl(r, 0.65 * l, "x", (-0.12 * l, 0, 0.55 * h)),
                                    cyl(1.15 * r, 0.04 * l, "x", (-0.44 * l, 0, 0.55 * h)),
                                    cyl(1.15 * r, 0.04 * l, "x", (0.2 * l, 0, 0.55 * h))]), COLORS["vessel"], 0.5, 0.4),
            Part("cabinet", box(0.2 * l, 0.8 * w, 0.85 * h, 0.38 * l, 0, 0.1 * h), COLORS["cabinet"])]


def _engine(l, w, h):
    return [Part("bedplate", box(l, w, 0.22 * h), COLORS["engine_dark"], 0.3, 0.6),
            Part("frame", _merge([box(0.92 * l, 0.8 * w, 0.4 * h, 0, 0, 0.22 * h),
                                  box(0.86 * l, 0.6 * w, 0.28 * h, -0.02 * l, 0, 0.62 * h)]),
                 COLORS["engine"], 0.3, 0.6),
            Part("top", _merge([box(0.8 * l, 0.45 * w, 0.1 * h, -0.03 * l, 0, 0.9 * h),
                                cyl(0.18 * w, 0.2 * w, "y", (0.42 * l, 0, 0.82 * h))]),
                 COLORS["engine_dark"], 0.4, 0.5)]


def _flow_meter(l, w, h):
    return [Part("body", cyl(0.3 * min(w, h), l, "x", (0, 0, 0.5 * h)), COLORS["vessel"], 0.5, 0.4),
            Part("head", box(0.4 * l, 0.6 * w, 0.25 * h, 0, 0, 0.72 * h), COLORS["cabinet"])]


def _tank(l, w, h):
    return [Part("tank", chamfered_box(l, w, h, 0.08 * min(l, w, h)), COLORS["tank"], 0.2, 0.7)]


def _connector(l, w, h):
    return [Part("marker", _merge([box(0.6 * l, 0.6 * w, 0.6 * h, -0.15 * l, 0, 0.2 * h),
                                   frustum(0.35 * min(w, h), 0.02, 0.4 * l, "x", (0.3 * l, 0, 0.5 * h))]),
                 COLORS["connector"], 0.1, 0.5)]


def _unknown(l, w, h):
    return [Part("body", box(l, w, h), COLORS["unknown"], 0.1, 0.8)]


# --------------------------------------------------------------------------- valves / instruments / fittings


def _actuator(act: ActuatorType, w: float, z0: float, z1: float) -> Optional[Part]:
    span = max(0.02, z1 - z0)
    col = ACTUATOR_COLORS.get(act, ACTUATOR_COLORS[ActuatorType.NONE])
    if act == ActuatorType.HANDWHEEL:
        return Part("actuator", torus(0.35 * w, 0.025 * w + 0.004, (0, 0, z1 - 0.03 * w)), col, 0.3, 0.5)
    if act == ActuatorType.HYDRAULIC:
        return Part("actuator", box(0.55 * w, 0.55 * w, span, 0, 0, z0), col, 0.3, 0.5)
    if act == ActuatorType.ELECTRIC:
        return Part("actuator", _merge([box(0.5 * w, 0.4 * w, 0.4 * span, 0, 0, z0),
                                        cyl(0.18 * w, 0.6 * span, "z", (0, 0, z0 + 0.7 * span))]), col, 0.3, 0.5)
    if act == ActuatorType.DIAPHRAGM:
        return Part("actuator", _merge([frustum(0.5 * w, 0.15 * w, 0.6 * span, "z", (0, 0, z0 + 0.3 * span)),
                                        cyl(0.5 * w, 0.1 * span, "z", (0, 0, z0 + 0.05 * span))]), col, 0.6, 0.3)
    if act == ActuatorType.QUICK_CLOSING:
        return Part("actuator", _merge([box(0.35 * w, 0.35 * w, 0.4 * span, 0, 0, z0),
                                        frustum(0.02, 0.4 * w, 0.5 * span, "z", (0, 0, z0 + 0.7 * span))]),
                    col, 0.3, 0.5)
    return None


def _valve(cls: SymbolClass, act: ActuatorType, l: float, w: float, h: float) -> list[Part]:
    za = axis_height(cls, (l, w, h))
    col = VALVE_COLORS[VALVE_FAMILY.get(cls, "isolation")]
    rb = 0.5 * min(w, 2 * za)
    parts: list[trimesh.Trimesh] = []
    fl = 0.1 * l
    flanges = [cyl(rb, fl, "x", (-l / 2 + fl / 2, 0, za)), cyl(rb, fl, "x", (l / 2 - fl / 2, 0, za))]
    if cls == S.BUTTERFLY_VALVE:
        parts.append(cyl(rb, l, "x", (0, 0, za)))
    elif cls == S.BALL_VALVE:
        parts += flanges + [sphere(0.9 * rb, (0, 0, za)), cyl(0.5 * rb, l, "x", (0, 0, za))]
    elif cls == S.SAFETY_VALVE:
        parts += [cyl(0.6 * rb, za + 0.2 * h, "z", (0, 0, (za + 0.2 * h) / 2)),
                  cyl(0.5 * rb, l / 2, "x", (l / 4, 0, za)),
                  cyl(0.45 * rb, h - za - 0.2 * h, "z", (0, 0, za + 0.2 * h + (h - za - 0.2 * h) / 2))]
    elif cls == S.THREE_WAY_VALVE:
        parts += flanges + [sphere(0.85 * rb, (0, 0, za)), cyl(0.5 * rb, l, "x", (0, 0, za)),
                            cyl(0.5 * rb, 0.5 * w, "y", (0, -0.25 * w, za))]
    else:  # gate, globe, check, sdnr: cylindrical body + bonnet
        parts += flanges + [cyl(0.7 * rb, 0.8 * l, "x", (0, 0, za))]
        if cls in (S.GLOBE_VALVE, S.SDNR_VALVE):
            parts.append(sphere(0.8 * rb, (0, 0, za)))
        if cls == S.CHECK_VALVE:
            parts.append(frustum(0.6 * rb, 0.3 * rb, 0.5 * rb, "z", (0, 0, za + 0.75 * rb)))
    out = [Part("body", _merge(parts), col, 0.45, 0.45)]
    if cls not in (S.CHECK_VALVE, S.SAFETY_VALVE):
        top = za + rb
        act_h = max(0.0, h - top)
        stem_top = top + 0.55 * act_h
        out.append(Part("stem", cyl(0.06 * w + 0.006, max(0.02, stem_top - top), "z", (0, 0, (top + stem_top) / 2)),
                        (0.55, 0.55, 0.58), 0.7, 0.3))
        if act == ActuatorType.NONE and cls == S.BUTTERFLY_VALVE:  # hand lever
            a: Optional[Part] = Part("actuator", box(0.5 * w, 0.08 * w, 0.06 * h, 0.25 * w, 0, stem_top),
                                     ACTUATOR_COLORS[ActuatorType.NONE])
        else:  # manual valves without a drawn actuator get a handwheel
            a = _actuator(act if act != ActuatorType.NONE else ActuatorType.HANDWHEEL, w, stem_top, h)
        if a is not None:
            out.append(a)
    return out


def _instrument(cls: SymbolClass, l: float, w: float, h: float) -> list[Part]:
    r = 0.42 * min(l, h)
    zc = h - r
    if cls == S.INSTRUMENT_PANEL:  # transmitter
        return [Part("stem", cyl(0.012, zc, "z", (0, 0, zc / 2)), (0.55, 0.55, 0.58), 0.7, 0.3),
                Part("housing", _merge([cyl(r, 0.8 * w, "y", (0, 0, zc)),
                                        box(0.8 * r, 0.8 * w, 0.6 * r, 0, 0, zc - r)]),
                     (0.20, 0.40, 0.70), 0.3, 0.5)]
    parts = [Part("stem", cyl(0.012, zc, "z", (0, 0, zc / 2)), (0.55, 0.55, 0.58), 0.7, 0.3),
             Part("case", cyl(r, 0.5 * w, "y", (0, 0, zc)), COLORS["rim"], 0.5, 0.4),
             Part("dial", cyl(0.85 * r, 0.52 * w, "y", (0, 0.01, zc)), COLORS["dial"], 0.0, 0.3)]
    if cls == S.INSTRUMENT_SHARED:
        parts.append(Part("display", box(0.9 * l, 0.3 * w, 0.3 * h, 0, -0.3 * w, 0), (0.25, 0.25, 0.30)))
    return parts


def _fitting(cls: SymbolClass, l: float, w: float, h: float) -> list[Part]:
    za = axis_height(cls, (l, w, h))
    col = COLORS["fitting"]
    if cls == S.REDUCER:
        return [Part("body", frustum(0.5 * w, 0.32 * w, l, "x", (0, 0, za)), col, 0.5, 0.4)]
    if cls == S.SPECTACLE_FLANGE:
        ring = trimesh.creation.annulus(r_min=0.25 * w, r_max=0.5 * w, height=l, sections=20)
        ring.apply_transform(_ROT["x"])
        ring.apply_translation((0, 0, za))
        blind = cyl(0.42 * w, l, "x", (0, 0, za + 0.5 * w + 0.42 * w))
        return [Part("ring", ring, col, 0.5, 0.4), Part("blind", blind, (0.85, 0.15, 0.15), 0.3, 0.5)]
    if cls == S.FLEXIBLE_HOSE:
        rings = [torus(0.35 * w, 0.1 * w, (0, 0, 0)) for _ in range(5)]
        for i, rg in enumerate(rings):
            rg.apply_transform(_ROT["x"])
            rg.apply_translation((-l / 2 + l * (i + 0.5) / 5, 0, za))
        return [Part("bellows", _merge(rings + [cyl(0.3 * w, l, "x", (0, 0, za))]), (0.25, 0.25, 0.27), 0.2, 0.7)]
    return [Part("body", cyl(0.4 * w, l, "x", (0, 0, za)), col)]


# --------------------------------------------------------------------------- dispatch

_EQUIPMENT_BUILDERS = {
    S.TANK: _tank, S.PUMP_CENTRIFUGAL: _pump_centrifugal, S.PUMP_SCREW: _pump_screw,
    S.HEAT_EXCHANGER: _shell_hx, S.PLATE_HEAT_EXCHANGER: _plate_hx,
    S.HEATER: lambda l, w, h: _shell_hx(l, w, h, "casing"),
    S.FILTER: _filter, S.DUPLEX_FILTER: lambda l, w, h: _filter(l, w, h, duplex=True), S.STRAINER: _strainer,
    S.SEA_CHEST: _sea_chest, S.OVERBOARD: _overboard, S.PURIFIER: _purifier, S.COMPRESSOR: _compressor,
    S.AIR_RECEIVER: _air_receiver, S.OILY_WATER_SEPARATOR: _ows, S.BWTS_UNIT: _bwts, S.ENGINE: _engine,
    S.FLOW_METER: _flow_meter, S.OFF_PAGE_CONNECTOR: _connector, S.UNKNOWN: _unknown,
}


def category_of(cls: SymbolClass) -> Category:
    return get_symbol(cls).category if cls != S.UNKNOWN else Category.EQUIPMENT


class _MeshCache:
    """Bounded LRU of built meshes keyed by their parameters (the GLB and the plotly viewer build the
    same parts; parallel valves of one size share theirs). Entries are shared: callers copy a mesh
    before changing it (the scene does)."""

    def __init__(self, max_vertices: int):
        self.max_vertices = max_vertices
        self.vertices = 0
        self.data: OrderedDict = OrderedDict()

    def get(self, key):
        v = self.data.get(key)
        if v is not None:
            self.data.move_to_end(key)
        return v

    def put(self, key, value, n_vertices: int) -> None:
        self.data[key] = (value, n_vertices)
        self.vertices += n_vertices
        while self.vertices > self.max_vertices and len(self.data) > 1:
            _, (_, n) = self.data.popitem(last=False)
            self.vertices -= n


_PART_CACHE = _MeshCache(1_000_000)  # vertices kept (about 50 MB with faces at most)
_PIPE_CACHE = _MeshCache(2_000_000)


def object_parts(cls: SymbolClass, size: Sequence[float], actuator: ActuatorType = ActuatorType.NONE
                 ) -> list[Part]:
    """Parts of one object in its local frame (see module docstring). Cached per (class, size,
    actuator): the meshes are shared, copy one before modifying it."""
    l, w, h = (max(float(x), 0.02) for x in size)
    key = (cls, l, w, h, actuator)
    hit = _PART_CACHE.get(key)
    if hit is not None:
        return [replace(p) for p in hit[0]]
    cat = category_of(cls)
    if cat == Category.VALVE:
        parts = _valve(cls, actuator, l, w, h)
    elif cat == Category.INSTRUMENT:
        parts = _instrument(cls, l, w, h)
    elif cat == Category.FITTING:
        parts = _fitting(cls, l, w, h)
    else:
        parts = _EQUIPMENT_BUILDERS.get(cls, _unknown)(l, w, h)
    _PART_CACHE.put(key, parts, sum(len(p.mesh.vertices) for p in parts))
    return [replace(p) for p in parts]


_UNIT_CYL: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_UNIT_SPHERE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_ALIGN: dict[tuple[float, float, float], np.ndarray] = {}


def _unit_cylinder(sections: int) -> tuple[np.ndarray, np.ndarray]:
    t = _UNIT_CYL.get(sections)
    if t is None:
        m = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=sections)
        t = _UNIT_CYL[sections] = (np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64))
    return t


def _unit_sphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    t = _UNIT_SPHERE.get(subdivisions)
    if t is None:
        m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
        t = _UNIT_SPHERE[subdivisions] = (np.asarray(m.vertices, dtype=np.float64),
                                          np.asarray(m.faces, dtype=np.int64))
    return t


def _align_z(u: np.ndarray) -> np.ndarray:
    """Rotation taking +z to the unit vector ``u`` (trimesh's ``align_vectors``, cached per direction:
    the routes are orthogonal, so a handful of directions cover every segment)."""
    key = (round(float(u[0]), 9), round(float(u[1]), 9), round(float(u[2]), 9))
    rot = _ALIGN.get(key)
    if rot is None:
        rot = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], u)[:3, :3]
        if len(_ALIGN) > 4096:
            _ALIGN.clear()
        _ALIGN[key] = rot
    return rot


def pipe_mesh(points: Sequence[Sequence[float]], diameter: float, sections: int = 14) -> Optional[trimesh.Trimesh]:
    """Pipe as cylinders per straight segment plus spherical elbows at the bends: one unit cylinder
    and one unit icosphere transformed per segment / bend with numpy (the same geometry as
    ``trimesh.creation.cylinder(segment=...)`` + ``icosphere``, without building a mesh per piece).
    Cached per (points, diameter, sections); the mesh is shared, copy it before modifying it."""
    r = max(diameter / 2, 0.005)
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    key = (pts.round(6).tobytes(), round(float(r), 6), int(sections))
    hit = _PIPE_CACHE.get(key)
    if hit is not None:
        return hit[0]
    cv, cf = _unit_cylinder(sections)
    sv, sf = _unit_sphere(1)
    verts, faces = [], []
    off = 0
    for a, b in zip(pts, pts[1:]):
        v = b - a
        length = float(np.linalg.norm(v))
        if length < 1e-6:
            continue
        rot = _align_z(v / length)
        verts.append((cv * (r, r, length)) @ rot.T + (a + v * 0.5))
        faces.append(cf + off)
        off += len(cv)
    for p in pts[1:-1]:
        verts.append(sv * r + p)
        faces.append(sf + off)
        off += len(sv)
    if not verts:
        return None
    mesh = trimesh.Trimesh(vertices=np.vstack(verts), faces=np.vstack(faces), process=False)
    _PIPE_CACHE.put(key, mesh, len(mesh.vertices))
    return mesh


def pipe_color(medium: Medium) -> RGB:
    return medium_rgb(medium)


def deck_parts(length: float, breadth: float, z: float, grid: float = 2.5) -> list[Part]:
    """Translucent deck plate with grid lines (x from 0 to length, centred on y = 0, top at z)."""
    t = 0.03
    plate = box(length, breadth, t, length / 2, 0.0, z - t)
    lines = []
    nx, ny = int(length // grid), int(breadth // grid)
    for i in range(nx + 1):
        lines.append(box(0.04, breadth, 0.012, i * grid, 0.0, z))
    for j in range(ny + 1):
        lines.append(box(length, 0.04, 0.012, length / 2, -breadth / 2 + j * grid, z))
    return [Part("plate", plate, COLORS["deck"], 0.1, 0.9, 0.08),
            Part("grid", _merge(lines), COLORS["grid"], 0.1, 0.9, 0.25)]


def outline_part(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, t: float = 0.06) -> Part:
    """Engine-room envelope as 12 thin edge bars."""
    edges = []
    L, B, H = x1 - x0, y1 - y0, z1 - z0
    for y in (y0, y1):
        for z in (z0, z1):
            edges.append(box(L, t, t, (x0 + x1) / 2, y, z - t / 2))
    for x in (x0, x1):
        for z in (z0, z1):
            edges.append(box(t, B, t, x, (y0 + y1) / 2, z - t / 2))
    for x in (x0, x1):
        for y in (y0, y1):
            edges.append(box(t, t, H, x, y, z0))
    return Part("outline", _merge(edges), COLORS["outline"], 0.1, 0.9, 0.5)
