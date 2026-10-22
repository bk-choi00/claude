"""Text roles, ownership and drawing information.

Roles (first rule that applies): title block / frame margin, notes, legend,
inside an instrument bubble, inside or at the tip of an off-page connector, line
number, tag (valve / equipment / instrument patterns), label (free text such as
equipment names). Tags are assigned to symbols with a global one-to-one
assignment (Hungarian) over distance + compatibility + side-preference costs, so
a crowded tag cannot be "stolen" by a closer but less suitable symbol; line
numbers go to the nearest parallel pipe segment.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..catalog import SYSTEM_BY_CODE, SYSTEMS, classify_tag, detect_system, parse_line_number
from ..model import Category, DrawingInfo, Medium, SymbolClass, SystemType, TextRole
from .regions import Regions
from .textlines import TextLine, line_height, text_distance

BBox4 = tuple[float, float, float, float]

GENERIC_TAG_RE = re.compile(r"^[A-Z]{1,4}(?:-[A-Z]{1,3})?-?\d{1,4}[A-Z]?$")
LOOSE_LINE_RE = re.compile(
    r"^(?P<size>DN\s?\d{2,4}|\d{1,4}(?:A|B|\"|MM)?)[-\s](?P<sys>[A-Z]{2,3})[-\s](?P<n>\d{2,5})(?:[-\s](?P<spec>[0-9A-Z]{1,6}))*$"
)
INSTR_LETTERS_RE = re.compile(r"^[A-Z]{1,5}$")
INSTR_NUMBER_RE = re.compile(r"^\d{1,5}[A-Z]?$")
SHIP_TYPE_RE = re.compile(
    r"(VLCC|ULCC|SUEZMAX|AFRAMAX|PANAMAX|KAMSARMAX|CAPESIZE|VLGC|VLEC|PCTC|PCC|RO-?RO|LNG\s*CARRIER|LPG\s*CARRIER|"
    r"CONTAINER\s*(?:SHIP|CARRIER|VESSEL)|BULK\s*CARRIER|PRODUCT(?:/CHEMICAL)?\s*TANKER|CHEMICAL\s*TANKER|"
    r"CRUDE\s*OIL\s*TANKER|TANKER|FPSO|FSRU|CAR\s*CARRIER)",
    re.I,
)


@dataclass
class ObjRef:
    index: int
    cls: SymbolClass
    category: Category
    bbox: BBox4
    body: BBox4
    rotation: int = 0
    holds_text: bool = False
    circle: Optional[tuple[float, float, float]] = None  # bubble centre and radius


@dataclass
class PipeRef:
    index: int
    segments: list[tuple[float, float, float, float]]
    dashed: bool = False


@dataclass
class TextResult:
    role: TextRole
    owner: Optional[tuple[str, int]] = None  # ('obj', i) | ('pipe', i)


@dataclass
class TextAssignment:
    results: list[TextResult]
    object_tags: dict[int, str] = field(default_factory=dict)
    object_names: dict[int, str] = field(default_factory=dict)
    pipe_line_numbers: dict[int, str] = field(default_factory=dict)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().upper())


def line_number_info(text: str) -> Optional[dict[str, str]]:
    """Size / system code / number / spec of a line number (catalog patterns first)."""
    t = normalize(text)
    parsed = parse_line_number(t)
    if parsed:
        return parsed
    m = LOOSE_LINE_RE.match(t)
    if not m:
        return None
    size = m.group("size").replace(" ", "")
    if size.isdigit():
        size += "A"
    return {"size": size, "system": m.group("sys"), "number": m.group("n"), "spec": m.group("spec") or ""}


def medium_of_line(text: str) -> Medium:
    info = line_number_info(text)
    if not info:
        return Medium.UNKNOWN
    sys_t = SYSTEM_BY_CODE.get(info["system"])
    return SYSTEMS[sys_t].medium if sys_t else Medium.UNKNOWN


def tag_kind(text: str) -> Optional[str]:
    """'line_number' | 'valve' | 'equipment' | 'instrument' | 'generic' | None."""
    t = normalize(text)
    k = classify_tag(t)
    if k:
        return k
    if line_number_info(t):
        return "line_number"
    if GENERIC_TAG_RE.match(t) and any(c.isdigit() for c in t):
        return "generic"
    return None


def instrument_tag(lines: Sequence[TextLine]) -> Optional[str]:
    """Letters + loop number written inside a bubble -> 'PI-101'."""
    tokens: list[str] = []
    for t in sorted(lines, key=lambda t: (t.bbox[1], t.bbox[0])):
        tokens.extend(normalize(t.text).replace("-", " ").split())
    letters = next((tok for tok in tokens if INSTR_LETTERS_RE.match(tok)), None)
    number = next((tok for tok in tokens if INSTR_NUMBER_RE.match(tok)), None)
    if letters is None and len(tokens) == 1:
        m = re.match(r"^([A-Z]{1,5})[-\s]?(\d{1,5}[A-Z]?)$", tokens[0])
        if m:
            letters, number = m.group(1), m.group(2)
    if letters and number:
        return f"{letters}-{number}"
    if tokens:
        return " ".join(tokens)
    return None


# --------------------------------------------------------------------------- roles


def _inside_obj(t: TextLine, o: ObjRef, slack: float = 0.0) -> bool:
    cx, cy = t.cx, t.cy
    if o.circle is not None:
        x, y, r = o.circle
        return math.hypot(cx - x, cy - y) <= r * 0.98 + slack
    b = o.body
    return b[0] - slack <= cx <= b[2] + slack and b[1] - slack <= cy <= b[3] + slack


def assign_texts(lines: Sequence[TextLine], regions: Regions, objects: Sequence[ObjRef], pipes: Sequence[PipeRef],
                 unit: float) -> TextAssignment:
    results: list[TextResult] = [TextResult(TextRole.UNKNOWN) for _ in lines]
    out = TextAssignment(results)
    tag_texts: list[int] = []
    label_texts: list[int] = []
    inside_instr: dict[int, list[int]] = {}
    inside_conn: dict[int, list[int]] = {}

    for k, t in enumerate(lines):
        kind_region = regions.kind_at(t.bbox, tol=0.5)
        if kind_region in ("title_block", "margin"):
            results[k].role = TextRole.TITLE_BLOCK
            continue
        if kind_region in ("notes", "table"):
            results[k].role = TextRole.NOTE
            continue
        if kind_region == "legend":
            results[k].role = TextRole.LEGEND
            continue
        host = None
        for o in objects:
            if o.category == Category.INSTRUMENT and _inside_obj(t, o):
                host = o
                break
        if host is not None:
            results[k] = TextResult(TextRole.INSTRUMENT, ("obj", host.index))
            inside_instr.setdefault(host.index, []).append(k)
            continue
        for o in objects:
            if o.category == Category.CONNECTOR and (_inside_obj(t, o, 0.1 * unit) or _beside_connector(t, o, unit)):
                host = o
                break
        if host is not None:
            results[k] = TextResult(TextRole.CONNECTOR, ("obj", host.index))
            inside_conn.setdefault(host.index, []).append(k)
            continue
        kind = tag_kind(t.text)
        if kind == "line_number":
            results[k].role = TextRole.LINE_NUMBER
        elif kind is not None:
            results[k].role = TextRole.TAG
            tag_texts.append(k)
        elif re.search(r"[A-Za-z]", t.text):
            results[k].role = TextRole.LABEL
            label_texts.append(k)

    # ---- instrument bubbles and connectors take their tag from the text inside
    for oi, ks in inside_instr.items():
        tag = instrument_tag([lines[k] for k in ks])
        if tag:
            out.object_tags[oi] = tag
    for oi, ks in inside_conn.items():
        txt = " ".join(normalize(lines[k].text) for k in sorted(ks, key=lambda k: (lines[k].bbox[1], lines[k].bbox[0])))
        if txt:
            out.object_tags[oi] = txt

    # ---- tags -> symbols (one-to-one)
    eligible = [o for o in objects if o.index not in out.object_tags and o.category != Category.CONNECTOR
                and not (o.category == Category.INSTRUMENT and o.index in inside_instr) and o.cls != SymbolClass.UNKNOWN]
    if tag_texts and eligible:
        big = 1e6
        C = np.full((len(tag_texts), len(eligible)), big)
        for a, k in enumerate(tag_texts):
            t = lines[k]
            kind = tag_kind(t.text)
            for b, o in enumerate(eligible):
                d = text_distance(t, o.bbox) / unit
                if d > 2.6:
                    continue
                comp = _compat(kind, o.category)
                if comp is None:
                    continue
                C[a, b] = d + comp + _side_penalty(t, o)
        rows, cols = linear_sum_assignment(C)
        for a, b in zip(rows.tolist(), cols.tolist()):
            if C[a, b] >= big:
                continue
            k = tag_texts[a]
            o = eligible[b]
            results[k].owner = ("obj", o.index)
            out.object_tags[o.index] = normalize(lines[k].text)

    # ---- labels -> equipment (names); several lines may belong to one object
    equipment = [o for o in objects if o.category in (Category.EQUIPMENT,) or o.cls == SymbolClass.UNKNOWN]
    names: dict[int, list[int]] = {}
    for k in label_texts:
        t = lines[k]
        best = None
        for o in equipment:
            if _inside_obj(t, o):
                d = 0.0
            else:
                d = text_distance(t, o.bbox)
                if d > 1.2 * unit:
                    continue
            if best is None or d < best[0]:
                best = (d, o)
        if best is not None:
            results[k].owner = ("obj", best[1].index)
            names.setdefault(best[1].index, []).append(k)
    for oi, ks in names.items():
        ks.sort(key=lambda k: (lines[k].bbox[1], lines[k].bbox[0]))
        out.object_names[oi] = " ".join(normalize(lines[k].text) for k in ks)

    # ---- line numbers -> nearest parallel pipe segment
    for k, t in enumerate(lines):
        if results[k].role != TextRole.LINE_NUMBER:
            continue
        best = None
        h = line_height(t)
        for pr in pipes:
            if pr.dashed:
                continue
            for x0, y0, x1, y1 in pr.segments:
                horiz = abs(y1 - y0) <= abs(x1 - x0)
                if t.vertical == horiz:
                    continue
                if horiz:
                    lo, hi = min(x0, x1), max(x0, x1)
                    overlap = min(hi, t.bbox[2]) - max(lo, t.bbox[0])
                    span = t.bbox[2] - t.bbox[0]
                    gap = max(0.0, max(t.bbox[1] - y0, y0 - t.bbox[3]))
                else:
                    lo, hi = min(y0, y1), max(y0, y1)
                    overlap = min(hi, t.bbox[3]) - max(lo, t.bbox[1])
                    span = t.bbox[3] - t.bbox[1]
                    gap = max(0.0, max(t.bbox[0] - x0, x0 - t.bbox[2]))
                if overlap < 0.3 * span or gap > max(2.5 * h, 0.9 * unit):
                    continue
                cost = gap + 0.5 * max(0.0, span - overlap)
                if best is None or cost < best[0]:
                    best = (cost, pr.index)
        if best is not None:
            results[k].owner = ("pipe", best[1])
            out.pipe_line_numbers.setdefault(best[1], normalize(t.text))
    return out


def _beside_connector(t: TextLine, o: ObjRef, unit: float) -> bool:
    """Text written right next to a connector (outside its outline), on its axis."""
    b = o.bbox
    if t.vertical:
        return b[0] - 0.2 * unit <= t.cx <= b[2] + 0.2 * unit and min(abs(t.bbox[1] - b[3]), abs(b[1] - t.bbox[3])) <= 0.35 * unit
    return b[1] - 0.2 * unit <= t.cy <= b[3] + 0.2 * unit and min(abs(t.bbox[0] - b[2]), abs(b[0] - t.bbox[2])) <= 0.35 * unit


def _compat(kind: Optional[str], cat: Category) -> Optional[float]:
    table = {
        "valve": {Category.VALVE: 0.0, Category.FITTING: 1.0, Category.EQUIPMENT: 2.5},
        "equipment": {Category.EQUIPMENT: 0.0, Category.VALVE: 2.5, Category.FITTING: 2.5},
        "instrument": {Category.VALVE: 0.3, Category.EQUIPMENT: 0.8, Category.INSTRUMENT: 1.0, Category.FITTING: 1.5},
        "generic": {Category.VALVE: 0.3, Category.EQUIPMENT: 0.3, Category.FITTING: 0.6, Category.INSTRUMENT: 1.0},
    }
    return table.get(kind or "generic", table["generic"]).get(cat)


def _side_penalty(t: TextLine, o: ObjRef) -> float:
    """Tags sit beside the pipe, not on it: penalise text placed along the flow axis
    of an inline symbol (it is more likely the neighbour's tag)."""
    if o.category not in (Category.VALVE, Category.FITTING):
        return 0.0
    b = o.body
    if o.rotation % 180 == 0:  # horizontal flow axis
        on_axis = t.bbox[1] < (b[1] + b[3]) / 2 < t.bbox[3]
    else:
        on_axis = t.bbox[0] < (b[0] + b[2]) / 2 < t.bbox[2]
    return 0.6 if on_axis else 0.0


# --------------------------------------------------------------------------- drawing information

_LABELS = {
    "drawing_no": re.compile(r"\b(DWG\.?\s*NO|DRAWING\s*(NO|NUMBER)|DOC(UMENT)?\.?\s*NO)\b\.?\s*[:.]?\s*", re.I),
    "title": re.compile(r"\b(DRAWING\s+TITLE|TITLE)\b\s*[:.]?\s*", re.I),
    "hull_no": re.compile(r"\b(HULL\s*(NO|NUMBER)|SHIP\s*NO|YARD\s*NO|S\.\s*NO)\b\.?\s*[:.]?\s*", re.I),
    "ship_type": re.compile(r"\b(SHIP\s*TYPE|VESSEL\s*TYPE|TYPE\s*OF\s*SHIP)\b\s*[:.]?\s*", re.I),
    "revision": re.compile(r"^\s*(REV(ISION)?\.?(\s*NO)?)\b\.?\s*[:.]?\s*", re.I),
    "sheet": re.compile(r"\b(SHEET|SHT)\b\.?\s*(NO)?\.?\s*[:.]?\s*", re.I),
    "scale": re.compile(r"\bSCALE\b\s*[:.]?\s*", re.I),
}
_ANY_LABEL = re.compile(
    r"^\s*(DWG\.?\s*NO|DRAWING\s*(NO|NUMBER|TITLE)|TITLE|HULL\s*(NO|NUMBER)|SHIP\s*(NO|TYPE)|YARD\s*NO|VESSEL\s*TYPE|"
    r"REV(ISION)?|SHEET|SHT|SCALE|DATE|DRAWN|CHECKED|APPROVED|DESIGNED|PROJECT|OWNER|BUILDER|CLASS|DESCRIPTION|BY)\b",
    re.I,
)


def drawing_info(title_lines: Sequence[TextLine], all_lines: Sequence[TextLine], line_numbers: Sequence[str]) -> DrawingInfo:
    info = DrawingInfo()
    tl = list(title_lines)
    values: dict[str, str] = {}
    for field_name, rx in _LABELS.items():
        for t in tl:
            m = rx.search(t.text)
            if not m:
                continue
            if field_name == "title" and "DRAWING NO" in t.text.upper():
                continue
            rest = t.text[m.end():].strip(" :.-")
            if rest and not _ANY_LABEL.match(rest):
                values.setdefault(field_name, rest)
                break
            v = _value_near(t, tl)
            if v:
                values.setdefault(field_name, v)
                break
    if "title" not in values:
        big = [t for t in tl if not _ANY_LABEL.match(t.text) and len(t.text) >= 6]
        sys_like = [t for t in big if re.search(r"SYSTEM|DIAGRAM|P\s*&\s*I\s*D|PIPING", t.text, re.I)]
        pool = sys_like or big
        if pool:
            top = max(t.size for t in pool)
            parts = sorted((t for t in pool if t.size >= 0.95 * top), key=lambda t: (t.bbox[1], t.bbox[0]))
            values["title"] = " ".join(normalize(t.text) for t in parts[:3])
    info.drawing_no = values.get("drawing_no")
    info.title = normalize(values["title"]) if values.get("title") else None
    info.hull_no = values.get("hull_no")
    info.revision = values.get("revision")
    info.sheet = values.get("sheet")
    info.scale = values.get("scale")
    if values.get("ship_type"):
        info.ship_type = values["ship_type"]
    else:
        for t in tl:
            m = SHIP_TYPE_RE.search(t.text)
            if m:
                info.ship_type = normalize(t.text) if len(t.text) <= 48 else m.group(0).upper()
                break
    # systems: title first, then the rest of the title block, then line-number codes
    sys_votes: Counter = Counter()
    main = detect_system(info.title or "") if info.title else None
    if main is None:
        main = detect_system(" ".join(t.text for t in tl))
    for t in tl:
        s = detect_system(t.text)
        if s:
            sys_votes[s] += 2
    for ln in line_numbers:
        li = line_number_info(ln)
        if li and li["system"] in SYSTEM_BY_CODE:
            sys_votes[SYSTEM_BY_CODE[li["system"]]] += 1
    if main is None and sys_votes:
        main = sys_votes.most_common(1)[0][0]
    info.system = main or SystemType.UNKNOWN
    systems = [s for s, _ in sys_votes.most_common()]
    if main and main not in systems:
        systems.insert(0, main)
    elif main:
        systems.remove(main)
        systems.insert(0, main)
    info.systems = systems
    extra = {k: v for k, v in values.items() if k not in ("drawing_no", "title", "hull_no", "ship_type", "revision", "sheet", "scale")}
    info.extra = extra
    return info


def _value_near(label: TextLine, lines: Sequence[TextLine]) -> Optional[str]:
    """Value written to the right of a label (same row) or below it (same cell)."""
    h = max(line_height(label), 1.0)
    best = None
    for t in lines:
        if t is label or _ANY_LABEL.match(t.text):
            continue
        right = t.bbox[0] >= label.bbox[2] - 0.5 * h and abs(t.cy - label.cy) <= 0.7 * h
        below = t.bbox[1] >= label.bbox[3] - 0.3 * h and t.bbox[1] - label.bbox[3] <= 3.5 * h and \
            t.bbox[0] <= label.bbox[2] + 2 * h and t.bbox[2] >= label.bbox[0] - 2 * h
        if not (right or below):
            continue
        d = (t.bbox[0] - label.bbox[2]) if right else (t.bbox[1] - label.bbox[3]) * 1.5
        if right and d > 12 * h:
            continue
        if best is None or d < best[0]:
            best = (d, t)
    return normalize(best[1].text) if best else None
