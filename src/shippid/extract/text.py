"""Text roles, ownership and drawing information.

Roles (first rule that applies): title block / frame margin, notes, legend,
inside an instrument bubble, inside or right beside an off-page connector, line
number, tag (valve / equipment / instrument patterns), label (free text such as
equipment names and ratings). Tags are assigned to symbols with a global
one-to-one assignment (Hungarian) over distance + centre offset + compatibility +
side-preference costs, so a crowded tag cannot be "stolen" by a closer but less
suitable symbol; line numbers go to the nearest parallel pipe segment; stacked
label lines become an equipment's ``name`` (first line) and ``rating``.
Title-block fields are read from labelled table cells.
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
# a reference to a sheet note written next to a symbol: "NOTE 2", "(NOTE 3)", "SEE NOTE 4", "NOTES 1,2"
CALLOUT_RE = re.compile(r"^\(?\s*(?:SEE\s+)?NOTES?\s*(?:NO\.?\s*)?\d{1,2}[A-Z]?(?:\s*[,&]\s*\d{1,2}[A-Z]?)*\s*\)?\.?$",
                        re.I)
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
    object_attrs: dict[int, dict[str, str]] = field(default_factory=dict)  # name / rating / ref
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
        inside = True
        for o in objects:
            if o.category == Category.CONNECTOR and _inside_obj(t, o, 0.1 * unit):
                host = o
                break
        if host is None and CALLOUT_RE.match(t.text.strip()):
            # note callout: a NOTE text (never owned); the nearest symbol records the reference
            results[k].role = TextRole.NOTE
            near = min(objects, key=lambda o: text_distance(t, o.bbox), default=None)
            if near is not None and text_distance(t, near.bbox) <= 1.5 * unit:
                attrs = out.object_attrs.setdefault(near.index, {})
                ref = normalize(t.text).strip("().")
                attrs["notes"] = ref if "notes" not in attrs else attrs["notes"] + "; " + ref
            continue
        kind = tag_kind(t.text)
        if host is None and kind in (None, "generic"):
            # free text right next to an off-page connector (and closer to it than to anything else)
            near = min(objects, key=lambda o: text_distance(t, o.bbox), default=None)
            if near is not None and near.category == Category.CONNECTOR and text_distance(t, near.bbox) <= 0.8 * unit:
                host, inside = near, False
        if host is not None:
            results[k] = TextResult(TextRole.CONNECTOR, ("obj", host.index))
            inside_conn.setdefault(host.index, []).append((k, inside))
            continue
        if kind == "line_number":
            results[k].role = TextRole.LINE_NUMBER
        elif kind is not None:
            results[k].role = TextRole.TAG
            tag_texts.append(k)
        elif any(c.isalpha() for c in t.text):  # any script (Hangul labels too)
            results[k].role = TextRole.LABEL
            label_texts.append(k)

    # ---- instrument bubbles and connectors take their tag from the text inside
    for oi, ks in inside_instr.items():
        tag = instrument_tag([lines[k] for k in ks])
        if tag:
            out.object_tags[oi] = tag
    for oi, ks in inside_conn.items():
        # off-page connectors: the drawing / sheet reference is written inside, the destination beside
        order = sorted(ks, key=lambda ki: (lines[ki[0]].bbox[1], lines[ki[0]].bbox[0]))
        ref = " ".join(normalize(lines[k].text) for k, inside in order if inside)
        name = " ".join(normalize(lines[k].text) for k, inside in order if not inside)
        attrs = out.object_attrs.setdefault(oi, {})
        if ref and name:
            attrs["ref"], attrs["name"] = ref, name
        elif ref or name:
            # one text only: a destination phrase is the name, a short code the reference
            txt = ref or name
            attrs["name" if re.search(r"\b(TO|FROM)\b", txt) or len(txt) > 12 else "ref"] = txt

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
                # tags are written centred on their symbol: the centre offset breaks ties between
                # neighbours that are equally close (symbols bolted flange-to-flange)
                oc = ((o.body[0] + o.body[2]) / 2, (o.body[1] + o.body[3]) / 2)
                centre = math.hypot(t.cx - oc[0], t.cy - oc[1]) / unit
                C[a, b] = d + 0.3 * centre + comp + _side_penalty(t, o)
        rows, cols = linear_sum_assignment(C)
        for a, b in zip(rows.tolist(), cols.tolist()):
            if C[a, b] >= big:
                continue
            k = tag_texts[a]
            o = eligible[b]
            results[k].owner = ("obj", o.index)
            out.object_tags[o.index] = normalize(lines[k].text)

    # ---- labels -> equipment: stacked label lines form one block (name, then rating lines)
    equipment = [o for o in objects if o.category == Category.EQUIPMENT or o.cls == SymbolClass.UNKNOWN]
    for block in _label_blocks([lines[k] for k in label_texts], label_texts):
        bb = (min(lines[k].bbox[0] for k in block), min(lines[k].bbox[1] for k in block),
              max(lines[k].bbox[2] for k in block), max(lines[k].bbox[3] for k in block))
        best = None
        for o in equipment:
            if any(_inside_obj(lines[k], o) for k in block):
                d = 0.0
            else:
                d = math.hypot(max(0.0, max(bb[0], o.bbox[0]) - min(bb[2], o.bbox[2])),
                               max(0.0, max(bb[1], o.bbox[1]) - min(bb[3], o.bbox[3])))
                if d > 1.2 * unit:
                    continue
            if best is None or d < best[0]:
                best = (d, o)
        if best is None:
            continue
        o = best[1]
        names, ratings = [], []
        for k in block:  # first line: the name; following lines: capacity / rating / type
            results[k].owner = ("obj", o.index)
            (ratings if names else names).append(normalize(lines[k].text))
        attrs = out.object_attrs.setdefault(o.index, {})
        if names:
            attrs["name"] = " ".join(names) if "name" not in attrs else attrs["name"] + " " + " ".join(names)
        if ratings:
            attrs["rating"] = " ".join(ratings)

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


def _label_blocks(block_lines: Sequence[TextLine], keys: Sequence[int]) -> list[list[int]]:
    """Group label lines written as one stacked block (same column, line pitch gaps)."""
    items = sorted(zip(keys, block_lines), key=lambda kt: (kt[1].bbox[1], kt[1].bbox[0]))
    blocks: list[list[tuple[int, TextLine]]] = []
    for k, t in items:
        placed = False
        for b in blocks:
            last = b[-1][1]
            h = max(line_height(last), line_height(t), 1.0)
            gap = t.bbox[1] - last.bbox[3]
            overlap = min(t.bbox[2], last.bbox[2]) - max(t.bbox[0], last.bbox[0])
            if t.rotation == last.rotation and -0.3 * h <= gap <= 0.8 * h and overlap > 0:
                b.append((k, t))
                placed = True
                break
        if not placed:
            blocks.append([(k, t)])
    return [[k for k, _ in b] for b in blocks]


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
    "drawing_no": re.compile(r"^\s*(DWG\.?\s*NO|DRAWING\s*(NO|NUMBER)|DOC(UMENT)?\.?\s*NO)\b\.?\s*[:.]?\s*", re.I),
    "title": re.compile(r"^\s*(DRAWING\s+TITLE|TITLE)\b\s*[:.]?\s*", re.I),
    "hull_no": re.compile(r"^\s*(HULL\s*(NO|NUMBER)|SHIP\s*NO|YARD\s*NO|S\.\s*NO)\b\.?\s*[:.]?\s*", re.I),
    "ship_type": re.compile(r"^\s*(SHIP\s*TYPE|VESSEL\s*TYPE|TYPE\s*OF\s*SHIP)\b\s*[:.]?\s*", re.I),
    "revision": re.compile(r"^\s*(REV(ISION)?\.?(\s*NO)?)\b\.?\s*[:.]?\s*", re.I),
    "sheet": re.compile(r"^\s*(SHEET|SHT)\b\.?\s*(NO)?\.?\s*[:.]?\s*", re.I),
    "scale": re.compile(r"^\s*SCALE\b\s*[:.]?\s*", re.I),
    "date": re.compile(r"^\s*DATE\b\s*[:.]?\s*", re.I),
    "owner": re.compile(r"^\s*OWNER\b\s*[:.]?\s*", re.I),
    "paper": re.compile(r"^\s*(SIZE|PAPER\s*SIZE|FORMAT)\b\s*[:.]?\s*", re.I),
    "system_code": re.compile(r"^\s*SYSTEM\s*CODE\b\s*[:.]?\s*", re.I),
}
_ANY_LABEL = re.compile(
    r"^\s*(DWG\.?\s*NO|DRAWING\s*(NO|NUMBER|TITLE)|TITLE|HULL\s*(NO|NUMBER)|SHIP\s*(NO|TYPE)|YARD\s*NO|VESSEL\s*TYPE|"
    r"REV(ISION)?|SHEET|SHT|SCALE|DATE|DRAWN|CHECKED|APPROVED|DESIGNED|PROJECT|OWNER|BUILDER|CLASS|DESCRIPTION|BY|SIZE|"
    r"SYSTEM\s*CODE|DWN|CHK|APP)\b\.?\s*[:.]?\s*$",
    re.I,
)
_YARD_RE = re.compile(r"SHIPBUILDING|HEAVY\s+INDUSTRIES|SHIPYARD|SHIP\s*YARD|MARINE|OFFSHORE|CO\.,?\s*LTD|WORKS|ENGINEERING", re.I)


def drawing_info(title_lines: Sequence[TextLine], all_lines: Sequence[TextLine], line_numbers: Sequence[str],
                 cell_at=None) -> DrawingInfo:
    """Title-block fields: a label's value is written after it, in the same table cell
    (usually below it, often centred) or to its right; the largest-font candidate wins
    (the main title-block cell over e.g. the revision table)."""
    info = DrawingInfo()
    tl = list(title_lines)
    best: dict[str, tuple[float, str]] = {}
    for t in tl:
        for field_name, rx in _LABELS.items():
            m = rx.match(t.text)
            if not m:
                continue
            rest = t.text[m.end():].strip()
            cand: Optional[tuple[float, str]] = None
            if rest and not _ANY_LABEL.match(rest):
                cand = (t.size, normalize(rest))
            else:
                vals = _cell_values(t, tl, cell_at) if cell_at is not None else []
                if vals:
                    cand = (max(v.size for v in vals), " ".join(normalize(v.text) for v in vals))
                else:
                    v = _value_near(t, tl)
                    if v is not None:
                        cand = (v.size * 0.9, normalize(v.text))
            if cand and (field_name not in best or cand[0] > best[field_name][0]):
                best[field_name] = cand
            break
    values = {k: v for k, (_, v) in best.items()}
    if "title" not in values:
        big = [t for t in tl if not _ANY_LABEL.match(t.text) and len(t.text) >= 6 and not _YARD_RE.search(t.text)]
        sys_like = [t for t in big if re.search(r"SYSTEM|DIAGRAM|P\s*&\s*I\s*D|PIPING", t.text, re.I)]
        pool = sys_like or big
        if pool:
            top = max(t.size for t in pool)
            parts = sorted((t for t in pool if t.size >= 0.95 * top), key=lambda t: (t.bbox[1], t.bbox[0]))
            values["title"] = " ".join(normalize(t.text) for t in parts[:3])
    info.drawing_no = values.get("drawing_no")
    info.title = values.get("title")
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
    code = values.get("system_code")
    if main is None and code and code.upper() in SYSTEM_BY_CODE:
        main = SYSTEM_BY_CODE[code.upper()]
    if main is None:
        main = detect_system(" ".join(t.text for t in tl))
    for t in tl:
        s_ = detect_system(t.text)
        if s_:
            sys_votes[s_] += 2
    for ln in line_numbers:
        li = line_number_info(ln)
        if li and li["system"] in SYSTEM_BY_CODE:
            sys_votes[SYSTEM_BY_CODE[li["system"]]] += 1
    if main is None and sys_votes:
        main = sys_votes.most_common(1)[0][0]
    info.system = main or SystemType.UNKNOWN
    systems = [s_ for s_, _ in sys_votes.most_common()]
    if main is not None:
        if main in systems:
            systems.remove(main)
        systems.insert(0, main)
    info.systems = systems
    extra: dict[str, str] = {}
    yards = [t for t in tl if _YARD_RE.search(t.text) and not _ANY_LABEL.match(t.text) and not _LABELS["owner"].match(t.text)]
    if yards:
        extra["yard"] = normalize(max(yards, key=lambda t: t.size).text)
    for k in ("owner", "date", "paper", "system_code"):
        if values.get(k):
            extra[k] = values[k]
    if main is not None:
        extra["system_name"] = SYSTEMS[main].name_en
    info.extra = extra
    return info


def _cell_values(label: TextLine, lines: Sequence[TextLine], cell_at) -> list[TextLine]:
    """Non-label texts sharing the label's table cell, in reading order."""
    cell = cell_at(label.cx, label.cy)
    if cell is None:
        return []
    vals = [t for t in lines if t is not label and cell[0] <= t.cx <= cell[2] and cell[1] <= t.cy <= cell[3]
            and not _ANY_LABEL.match(t.text) and not any(rx.match(t.text) and not t.text[rx.match(t.text).end():].strip()
                                                       for rx in _LABELS.values())]
    return sorted(vals, key=lambda t: (round(t.bbox[1], 0), t.bbox[0]))


def _value_near(label: TextLine, lines: Sequence[TextLine]) -> Optional[TextLine]:
    """Value written to the right of a label (same row) or below it (same column)."""
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
    return best[1] if best else None
