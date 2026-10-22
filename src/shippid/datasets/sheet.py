"""Drawing sheet furniture: frame with zone markers, title block, notes and legend.

Everything here is real vector geometry and real text; every text is returned
with its ground-truth role (:class:`TextRole.TITLE_BLOCK`, ``NOTE``, ``LEGEND``)
so the synthetic ground truth covers *all* text on the page.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional

from ..catalog import SYSTEMS
from ..model import ActuatorType, BBox, DrawingInfo, SymbolClass, TextRole
from ..symbols import ACTUATOR_DESCRIPTIONS, Placement, get_symbol, placed_bbox
from .blueprints import SystemTopology
from .pdf_writer import PathDraw, SymbolDraw, TextDraw
from .text import BOLD_FONTS, line_height, origin_for_box, text_width, wrap

PAPERS: dict[str, tuple[float, float]] = {"A1": (2384.0, 1684.0), "A2": (1684.0, 1191.0), "A3": (1191.0, 842.0)}
PAPER_ORDER = ("A3", "A2", "A1")
_TB_WIDTH = {"A1": 520.0, "A2": 470.0, "A3": 380.0}
_ZONES = {"A1": (12, 8), "A2": (8, 6), "A3": (8, 6)}
_K = {"A1": 1.0, "A2": 0.92, "A3": 0.8}  # text size factor per paper

_PEOPLE = ("J.H. KIM", "S.Y. PARK", "M.S. LEE", "D.K. CHOI", "H.S. JUNG", "Y.J. KANG", "T.H. YOON", "K.W. LIM",
           "R. SANTOS", "A. PETROV", "L. NGUYEN", "P. MÜLLER".replace("Ü", "UE"))
_REV_TEXT = ("ISSUED FOR APPROVAL", "ISSUED FOR OWNER'S COMMENT", "APPROVED BY CLASS", "ISSUED FOR CONSTRUCTION",
             "REVISED AS PER OWNER'S COMMENT", "FINAL (AS BUILT)", "ISSUED FOR WORKING")
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


@dataclass
class LegendEntry:
    cls: SymbolClass
    actuator: ActuatorType
    text: str


@dataclass
class Sheet:
    paper: str
    width: float
    height: float
    drawing_area: BBox
    paths: list[PathDraw] = field(default_factory=list)
    texts: list[tuple[TextDraw, TextRole]] = field(default_factory=list)
    symbols: list[SymbolDraw] = field(default_factory=list)
    info: DrawingInfo = field(default_factory=DrawingInfo)
    occupied: list[tuple[float, float, float, float]] = field(default_factory=list)  # furniture boxes


@dataclass
class SheetStyle:
    font: str = "helv"
    frame_width: float = 1.2
    line_width: float = 0.5
    layout: str = "right"  # "right": notes/legend/title in a right column; "bottom": strip along the bottom
    legend: bool = False
    notes_box: bool = True
    zone_font: str = "helv"
    layer: Optional[str] = None  # OCG name for frame / title block


class _SheetBuilder:
    def __init__(self, paper: str, style: SheetStyle, rng: random.Random):
        self.paper = paper
        self.W, self.H = PAPERS[paper]
        self.st = style
        self.rng = rng
        self.k = _K[paper]
        self.sheet = Sheet(paper, self.W, self.H, BBox(x0=0, y0=0, x1=1, y1=1))
        self.bold = BOLD_FONTS.get(style.font, "hebo")

    # -- primitives -----------------------------------------------------------
    def line(self, x0: float, y0: float, x1: float, y1: float, w: Optional[float] = None) -> None:
        self.sheet.paths.append(PathDraw([("l", x0, y0, x1, y1)], width=w or self.st.line_width, layer=self.st.layer))

    def rect(self, x0: float, y0: float, x1: float, y1: float, w: Optional[float] = None) -> None:
        self.sheet.paths.append(PathDraw.rect(x0, y0, x1, y1, width=w or self.st.line_width, layer=self.st.layer))

    def text(self, s: str, x: float, y: float, size: float, role: TextRole, font: Optional[str] = None,
             anchor: str = "l", rotate: int = 0) -> tuple[float, float, float, float]:
        """Place ``s`` with its bbox anchored at ``(x, y)``: anchor 'l' = left/middle, 'c' = centre/middle,
        'tl' = top-left. Returns the bbox."""
        font = font or self.st.font
        w = text_width(s, font, size)
        h = line_height(font, size)
        if anchor == "c":
            x0, y0 = x - w / 2, y - h / 2
        elif anchor == "tl":
            x0, y0 = x, y
        elif anchor == "r":
            x0, y0 = x - w, y - h / 2
        else:
            x0, y0 = x, y - h / 2
        ox, oy = origin_for_box(s, x0, y0, size, font, rotate)
        self.sheet.texts.append((TextDraw(s, ox, oy, size, font, rotate, layer=self.st.layer), role))
        return (x0, y0, x0 + w, y0 + h)

    def fit(self, s: str, size: float, max_w: float, font: Optional[str] = None) -> float:
        font = font or self.st.font
        while size > 3.0 and text_width(s, font, size) > max_w:
            size -= 0.25
        return size

    # -- parts ----------------------------------------------------------------
    def frame(self) -> tuple[float, float, float, float]:
        r = self.rng
        m = {"A1": 28.0, "A2": 22.0, "A3": 16.0}[self.paper] + r.uniform(-3, 3)
        z = {"A1": 14.0, "A2": 12.0, "A3": 10.0}[self.paper]
        ox0, oy0, ox1, oy1 = m, m, self.W - m, self.H - m
        ix0, iy0, ix1, iy1 = ox0 + z, oy0 + z, ox1 - z, oy1 - z
        self.rect(ox0, oy0, ox1, oy1, w=self.st.line_width)
        self.rect(ix0, iy0, ix1, iy1, w=self.st.frame_width)
        nx, ny = _ZONES[self.paper]
        zs = self.k * 7.0
        for i in range(nx):
            xa = ix0 + (ix1 - ix0) * i / nx
            xb = ix0 + (ix1 - ix0) * (i + 1) / nx
            if i:
                self.line(xa, oy0, xa, iy0)
                self.line(xa, iy1, xa, oy1)
            label = str(i + 1)
            self.text(label, (xa + xb) / 2, (oy0 + iy0) / 2, zs, TextRole.TITLE_BLOCK, self.st.zone_font, "c")
            self.text(label, (xa + xb) / 2, (iy1 + oy1) / 2, zs, TextRole.TITLE_BLOCK, self.st.zone_font, "c")
        letters = "ABCDEFGHJK"
        for j in range(ny):
            ya = iy0 + (iy1 - iy0) * j / ny
            yb = iy0 + (iy1 - iy0) * (j + 1) / ny
            if j:
                self.line(ox0, ya, ix0, ya)
                self.line(ix1, ya, ox1, ya)
            self.text(letters[j], (ox0 + ix0) / 2, (ya + yb) / 2, zs, TextRole.TITLE_BLOCK, self.st.zone_font, "c")
            self.text(letters[j], (ix1 + ox1) / 2, (ya + yb) / 2, zs, TextRole.TITLE_BLOCK, self.st.zone_font, "c")
        return ix0, iy0, ix1, iy1

    def title_block(self, x1: float, y1: float, topo: SystemTopology) -> tuple[BBox, DrawingInfo]:
        r, k, f, fb = self.rng, self.k, self.st.font, self.bold
        ship = topo.ship
        assert ship is not None
        tw = _TB_WIDTH[self.paper]
        x0 = x1 - tw
        lab = 4.6 * k  # field label size
        val = 6.6 * k
        rows: list[tuple[str, float]] = []
        hull_digits = re.sub(r"\D", "", ship.hull_no) or "0000"
        dwg = r.choice(["{h}-M-{n:03d}", "M{n:03d}-{c}-{h}", "{h}-PD-{c}-{m:02d}", "5{h}-{c}{n:03d}"]).format(
            h=hull_digits, n=r.randint(100, 899), c=topo.code, m=r.randint(1, 30))
        year = r.randint(2023, 2026)
        n_rev = r.randint(1, 4)
        revs = ["A", "B", "C", "D"] if r.random() < 0.6 else ["0", "1", "2", "3"]
        dates = sorted((year, r.randint(1, 12), r.randint(1, 28)) for _ in range(n_rev))
        date_style = r.randint(0, 2)

        def fmt(d: tuple[int, int, int]) -> str:
            y, mo, da = d
            return (f"{y}.{mo:02d}.{da:02d}", f"{da:02d} {_MONTHS[mo - 1]} {y}", f"{y}-{mo:02d}-{da:02d}")[date_style]

        people = r.sample(_PEOPLE, 3)
        title_lines = list(topo.title)
        if r.random() < 0.5:
            title_lines.append(r.choice(["PIPING DIAGRAM", "P & I DIAGRAM", "DIAGRAMMATIC PIPING"]))
        rev_h = 10.5 * k
        h_dwg, h_scale, h_title = 24 * k, 17 * k, (14 + 11 * len(title_lines)) * k
        h_proj, h_yard, h_appr = 28 * k, 26 * k, 17 * k
        total = h_dwg + h_scale + h_title + h_proj + h_yard + h_appr + rev_h * (n_rev + 1)
        y0 = y1 - total
        self.rect(x0, y0, x1, y1, w=self.st.frame_width)
        y = y0
        # revision table (latest revision at the bottom, as usual)
        cols = [0.0, 0.08, 0.26, 0.70, 0.80, 0.90, 1.0]
        heads = ["REV.", "DATE", "DESCRIPTION", "DWN", "CHK", "APP"]
        for i, hd in enumerate(heads):
            cx = x0 + tw * (cols[i] + cols[i + 1]) / 2
            self.text(hd, cx, y + rev_h / 2, lab, TextRole.TITLE_BLOCK, f, "c")
        for c in cols[1:-1]:
            self.line(x0 + tw * c, y, x0 + tw * c, y + rev_h * (n_rev + 1))
        y += rev_h
        self.line(x0, y, x1, y)
        for i in range(n_rev):
            vals = [revs[i], fmt(dates[i]), _REV_TEXT[min(i, len(_REV_TEXT) - 1)] if i == 0
                    else r.choice(_REV_TEXT[1:]), people[0].split()[-1], people[1].split()[-1], people[2].split()[-1]]
            for j, v in enumerate(vals):
                cw = tw * (cols[j + 1] - cols[j]) - 4
                size = self.fit(v, val * 0.85, cw)
                self.text(v, x0 + tw * cols[j] + 2, y + rev_h / 2, size, TextRole.TITLE_BLOCK, f, "l")
            y += rev_h
            self.line(x0, y, x1, y)
        # approval row
        third = tw / 4
        for j, (labt, v) in enumerate([("DRAWN", people[0]), ("CHECKED", people[1]), ("APPROVED", people[2]),
                                        ("DATE", fmt(dates[-1]))]):
            xa = x0 + third * j
            if j:
                self.line(xa, y, xa, y + h_appr)
            self.text(labt, xa + 2, y + 2, lab, TextRole.TITLE_BLOCK, f, "tl")
            self.text(v, xa + 4, y + h_appr * 0.66, self.fit(v, val * 0.9, third - 8), TextRole.TITLE_BLOCK, f, "l")
        y += h_appr
        self.line(x0, y, x1, y)
        # yard
        yard_size = self.fit(ship.yard, 10.5 * k, tw - 10, fb)
        self.text(ship.yard, x0 + tw / 2, y + h_yard * 0.42, yard_size, TextRole.TITLE_BLOCK, fb, "c")
        owner = f"OWNER : {ship.owner}"
        self.text(owner, x0 + tw / 2, y + h_yard * 0.82, self.fit(owner, lab * 1.1, tw - 10), TextRole.TITLE_BLOCK,
                  f, "c")
        y += h_yard
        self.line(x0, y, x1, y)
        # project
        half = tw * 0.38
        self.line(x0 + half, y, x0 + half, y + h_proj)
        self.text("HULL NO.", x0 + 2, y + 2, lab, TextRole.TITLE_BLOCK, f, "tl")
        self.text(ship.hull_no, x0 + half / 2, y + h_proj * 0.6, self.fit(ship.hull_no, 9.5 * k, half - 8, fb),
                  TextRole.TITLE_BLOCK, fb, "c")
        self.text("SHIP TYPE", x0 + half + 2, y + 2, lab, TextRole.TITLE_BLOCK, f, "tl")
        self.text(ship.ship_type, x0 + half + (tw - half) / 2, y + h_proj * 0.6,
                  self.fit(ship.ship_type, 8.5 * k, tw - half - 8, fb), TextRole.TITLE_BLOCK, fb, "c")
        y += h_proj
        self.line(x0, y, x1, y)
        # title
        self.text("TITLE", x0 + 2, y + 2, lab, TextRole.TITLE_BLOCK, f, "tl")
        ts = 11.0 * k
        for i, ln in enumerate(title_lines):
            size = self.fit(ln, ts, tw - 20, fb)
            self.text(ln, x0 + tw / 2, y + (12 + 11 * i) * k + 1, size, TextRole.TITLE_BLOCK, fb, "c")
        y += h_title
        self.line(x0, y, x1, y)
        # scale / sheet / size
        fourth = tw / 4
        sheet_txt = "1 OF 1" if r.random() < 0.7 else "1/1"
        for j, (labt, v) in enumerate([("SCALE", "NONE"), ("SHEET", sheet_txt), ("SIZE", self.paper),
                                        ("SYSTEM CODE", topo.code)]):
            xa = x0 + fourth * j
            if j:
                self.line(xa, y, xa, y + h_scale)
            self.text(labt, xa + 2, y + 2, lab, TextRole.TITLE_BLOCK, f, "tl")
            self.text(v, xa + fourth / 2, y + h_scale * 0.65, val, TextRole.TITLE_BLOCK, f, "c")
        y += h_scale
        self.line(x0, y, x1, y)
        # drawing number + revision
        split = tw * 0.8
        self.line(x0 + split, y, x0 + split, y1)
        self.text("DWG. NO.", x0 + 2, y + 2, lab, TextRole.TITLE_BLOCK, f, "tl")
        self.text(dwg, x0 + split / 2, y + h_dwg * 0.6, self.fit(dwg, 11 * k, split - 10, fb), TextRole.TITLE_BLOCK,
                  fb, "c")
        self.text("REV.", x0 + split + 2, y + 2, lab, TextRole.TITLE_BLOCK, f, "tl")
        self.text(revs[n_rev - 1], x0 + split + (tw - split) / 2, y + h_dwg * 0.6, 11 * k, TextRole.TITLE_BLOCK, fb,
                  "c")
        info = DrawingInfo(drawing_no=dwg, title=" ".join(title_lines), system=topo.system, systems=[topo.system],
                           hull_no=ship.hull_no, ship_type=ship.ship_type, revision=revs[n_rev - 1], sheet=sheet_txt,
                           scale="NONE",
                           extra={"yard": ship.yard, "owner": ship.owner, "date": fmt(dates[-1]), "paper": self.paper,
                                  "system_name": SYSTEMS[topo.system].name_en})
        return BBox(x0=x0, y0=y0, x1=x1, y1=y1), info

    def notes(self, x0: float, y: float, w: float, notes: list[str], max_h: float, bottom: bool = False) -> float:
        """Notes block at ``x0`` with its top (or bottom, if ``bottom``) at ``y``; returns its height
        (0 if nothing fits)."""
        k, f = self.k, self.st.font
        size = 6.2 * k * self.rng.uniform(0.9, 1.1)
        lh = line_height(f, size) * 1.25
        pad = 5.0
        lines: list[tuple[str, float]] = []
        for i, n in enumerate(notes):
            label = f"{i + 1}."
            for j, ln in enumerate(wrap(n, f, size, w - 2 * pad - 14)):
                lines.append(((label + " " if j == 0 else "") + ln, 0.0 if j == 0 else 14.0))
        head = self.rng.choice(["NOTES", "GENERAL NOTES", "NOTES :"])
        h = pad * 2 + line_height(self.bold, size * 1.3) * 1.4 + lh * len(lines)
        while lines and h > max_h:
            lines.pop()
            h -= lh
        if not lines:
            return 0.0
        y0 = y - h if bottom else y
        if self.st.notes_box:
            self.rect(x0, y0, x0 + w, y0 + h)
        self.text(head, x0 + pad, y0 + pad, size * 1.3, TextRole.NOTE, self.bold, "tl")
        y = y0 + pad + line_height(self.bold, size * 1.3) * 1.4
        for s, indent in lines:
            if indent == 0 and s[:1].isdigit() and " " in s:
                num, rest = s.split(" ", 1)
                self.text(num, x0 + pad, y, size, TextRole.NOTE, f, "tl")
                self.text(rest, x0 + pad + 14, y, size, TextRole.NOTE, f, "tl")
            else:
                self.text(s, x0 + pad + indent, y, size, TextRole.NOTE, f, "tl")
            y += lh
        self.sheet.occupied.append((x0, y0, x0 + w, y0 + h))
        return h

    def legend(self, x0: float, y0: float, w: float, entries: list[LegendEntry], max_h: float) -> float:
        k, f = self.k, self.st.font
        size = 5.8 * k
        sc = 11.0 * k
        row = max(sc * 1.45, line_height(f, size) * 1.6)
        pad = 5.0
        ncol = 2 if w > 300 else 1
        colw = (w - 2 * pad) / ncol
        head_h = line_height(self.bold, size * 1.3) * 1.6
        per_col = max(1, int((max_h - 2 * pad - head_h) // row))
        entries = entries[: per_col * ncol]
        if not entries:
            return 0.0
        nrow = min(per_col, -(-len(entries) // ncol))
        h = 2 * pad + head_h + nrow * row
        self.rect(x0, y0, x0 + w, y0 + h)
        self.text(self.rng.choice(["LEGEND", "SYMBOL LEGEND", "SYMBOLS"]), x0 + pad, y0 + pad, size * 1.3,
                  TextRole.LEGEND, self.bold, "tl")
        for i, e in enumerate(entries):
            c, rr = divmod(i, nrow)
            cx = x0 + pad + c * colw + sc * 1.1
            cy = y0 + pad + head_h + rr * row + row / 2
            spec = get_symbol(e.cls)
            u0, v0, u1, v1 = spec.extent
            s = min(sc, sc * 2.0 / max(u1 - u0, 1e-6), row * 0.8 / max(v1 - v0 + 0.8, 1e-6))
            pl = Placement(cx, cy, scale=s)
            bx = placed_bbox(spec, pl, e.actuator)
            pl = Placement(cx, cy - ((bx[1] + bx[3]) / 2 - cy), scale=s)
            self.sheet.symbols.append(SymbolDraw(e.cls, pl, e.actuator, width=0.5, layer=self.st.layer))
            tx = x0 + pad + c * colw + sc * 2.6
            size_t = self.fit(e.text, size, colw - sc * 2.8)
            self.text(e.text, tx, cy, size_t, TextRole.LEGEND, f, "l")
        self.sheet.occupied.append((x0, y0, x0 + w, y0 + h))
        return h


def legend_entries(topo: SystemTopology) -> list[LegendEntry]:
    """Distinct (class, actuator) pairs used by the topology, in a stable order."""
    seen: dict[tuple[SymbolClass, ActuatorType], LegendEntry] = {}
    for it in topo.items():
        act = it.actuator if it.category.value == "valve" else ActuatorType.NONE
        key = (it.cls, act)
        if key in seen or it.cls == SymbolClass.UNKNOWN:
            continue
        name = get_symbol(it.cls).label_en.upper()
        if act not in (ActuatorType.NONE, ActuatorType.HANDWHEEL):
            short = {ActuatorType.HYDRAULIC: "HYD. OPERATED", ActuatorType.DIAPHRAGM: "PNEU. CONTROL",
                     ActuatorType.ELECTRIC: "MOTOR OPERATED", ActuatorType.QUICK_CLOSING: "QUICK-CLOSING"}[act]
            name = f"{short} {name}"
        seen[key] = LegendEntry(it.cls, act, name)
    order = {"equipment": 0, "valve": 1, "instrument": 2, "fitting": 3, "connector": 4}
    return sorted(seen.values(), key=lambda e: (order[get_symbol(e.cls).category.value], e.cls.value, e.actuator.value))


def build_sheet(paper: str, style: SheetStyle, rng: random.Random, topo: SystemTopology) -> Sheet:
    """Frame, title block, notes and optional legend for ``paper``; sets ``drawing_area``."""
    b = _SheetBuilder(paper, style, rng)
    ix0, iy0, ix1, iy1 = b.frame()
    tb, info = b.title_block(ix1, iy1, topo)
    b.sheet.info = info
    b.sheet.occupied.append(tb.as_tuple())
    gap = 8.0
    tw = tb.width
    entries = legend_entries(topo) if style.legend else []
    if style.layout == "right":
        col_x0 = tb.x0
        b.line(col_x0, iy0, col_x0, tb.y0, w=style.frame_width)
        y = iy0 + gap
        if entries:
            y += b.legend(col_x0 + gap, y, tw - 2 * gap, entries, (tb.y0 - iy0) * 0.55) + gap
        b.notes(col_x0 + gap, tb.y0 - gap, tw - 2 * gap, topo.notes, tb.y0 - gap - y, bottom=True)
        area = BBox(x0=ix0, y0=iy0, x1=col_x0, y1=iy1)
    else:
        strip_y0 = tb.y0
        b.line(ix0, strip_y0, tb.x0, strip_y0, w=style.frame_width)
        x = ix0 + gap
        notes_w = min(520.0 * b.k, (tb.x0 - ix0) * 0.45)
        nh = b.notes(x, strip_y0 + gap, notes_w, topo.notes, tb.height - 2 * gap)
        x += (notes_w + gap) if nh else 0.0
        if entries:
            lw = min(560.0 * b.k, tb.x0 - x - gap)
            if lw > 180:
                b.legend(x, strip_y0 + gap, lw, entries, tb.height - 2 * gap)
        area = BBox(x0=ix0, y0=iy0, x1=ix1, y1=strip_y0)
    b.sheet.drawing_area = area
    return b.sheet

