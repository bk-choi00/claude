"""Multi-sheet synthetic P&ID sets: one ship system split across 2-3 sheets with off-page connectors.

Real P&ID sets of large ships are multi-sheet PDFs: a system leaves sheet 1 through an off-page
connector ("TO NO.1 W.B.TK (P) / SH.2") and continues on sheet 2 from its partner ("FROM BALLAST
PUMP / SH.1"), or both connectors carry a shared continuation mark ("A1") or a drawing / sheet
reference ("CONT. ON DWG 4446-PD-BA-12 SH.2").

:func:`generate_multisheet` builds one system topology (:mod:`shippid.datasets.blueprints`), cuts
it between stages (balanced item counts) and lays out every part as its own sheet with the
single-sheet generator (:mod:`shippid.datasets.synthetic`): the header group of every cut stage
boundary ends on the upstream sheet in an outgoing connector and starts on the downstream sheet
from an incoming one. Return lines to a tank on another sheet (purifier -> sump, OWS three-way
valve -> bilge holding tank, condensate -> boiler) are cut the same way (the tank gets an incoming
connector beside it). All sheets share one drawing style, paper size, drawing number and title
block; the title-block sheet field reads ``k OF n`` (or ``k/n``). The returned ground truth is one
:class:`~shippid.model.PIDDocument` with every page (``PageInfo`` per page, ``page`` on every
item) and globally unique, page-ordered ids (as the extractor numbers them).

Ground truth of the pairing: ``doc.meta["sheet_links"] = [{"a", "b", "key", "kind", "sheets", "directed",
"source"}]``
with ``a`` the outgoing (upstream) connector id, ``b`` the incoming one, ``key`` the continuation
key the yard style writes (``"mark:A1"``, ``"sheet:1>2"``, ``"dwg:...:1>2"``), ``kind`` ``"header"``
or ``"return"`` and ``sheets`` the 1-based sheet numbers. ``meta["multisheet"]`` records the yard
style, the stage cuts and the per-sheet title-block values; ``meta["sheets"]`` lists per page
``{"page", "sheet", "of", "drawing_no"}``. Off-page connectors that lead to *other* systems keep
their unique random references and are never paired.

Yard styles (``YARD_STYLES``; ``i`` / ``j`` are the 1-based sheet numbers of the two partners):

* ``"sheet"`` -- inside ``SH.j``; label ``TO <destination>`` / ``FROM <origin>``;
* ``"mark"`` -- inside a shared continuation mark (``A`` / ``A1`` / ``1`` schemes); label
  ``TO <destination> / SH.j`` / ``FROM <origin> / SH.i``;
* ``"dwg"`` -- inside the partner's sheet number; label ``TO <destination>`` plus
  ``CONT. ON DWG <drawing no.> SH.j`` / ``FROM <origin>`` plus ``CONT. FROM DWG <drawing no.> SH.i``.

The generator reuses the private single-sheet entry point ``synthetic._generate`` (layout of a
given topology) so the sheets look exactly like the single-sheet dataset. Everything is
deterministic per ``(seed, system, sheets, style)``.
"""

from __future__ import annotations

import copy
import itertools
import random
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import pymupdf

from ..model import Category, PageInfo, PIDDocument, SymbolClass, SystemType, TextRole
from .blueprints import Attachment, Item, SideLink, Stage, SystemTopology, Tap, Train, build_topology
from .pdf_writer import DrawingSpec, TextDraw, render
from .synthetic import ALL_SYSTEMS, LayoutError, _generate, random_style
from .text import text_bbox, text_width

S = SymbolClass

YARD_STYLES: tuple[str, ...] = ("sheet", "mark", "dwg")
MARK_SCHEMES: tuple[str, ...] = ("letter", "letter_digit", "digit")
MAX_ATTEMPTS = 8
_MS_MARKER = "__multisheet__:"  # temporary Item.attrs["service"] used to find connector objects again
_REF_LETTERS = "ABCDEFGHJK"
_MARK_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
_NAME_MAX = 30  # longest generated destination / origin name (label placement)

_NO_RE = re.compile(r"^NO\.\s*\d+[A-Z]?\s+")
_SIDE_RE = re.compile(r"\s*\((?:P|S|C|P&S|PORT|STBD)\)\s*$")


# --------------------------------------------------------------------------- planning


@dataclass
class _Pair:
    out_key: str  # item key of the outgoing connector (upstream sheet)
    in_key: str  # item key of the incoming connector (downstream sheet)
    out_sheet: int  # 0-based sheet index
    in_sheet: int
    kind: str  # "header" | "return"
    key: str = ""  # continuation key written by the yard style


@dataclass
class _Plan:
    style: str
    mark_scheme: str
    drawing_no: str
    n: int
    bounds: list[int]  # stage boundaries cut (stage index of the first stage of each later sheet)
    ranges: list[tuple[int, int]]  # stage range [lo, hi) per sheet
    pairs: list[_Pair] = field(default_factory=list)


def _stage_of(topo: SystemTopology) -> dict[str, int]:
    """Item / branch / train key -> stage index."""
    out: dict[str, int] = {}
    for k, st in enumerate(topo.stages):
        for tr in st.trains:
            out[tr.key] = k
            for it in (tr.source, tr.sink):
                if it is not None:
                    out[it.key] = k
            for el in tr.elements:
                if isinstance(el, Tap):
                    out[el.instr.key] = k
                else:
                    out[el.key] = k
    return out


def _weights(topo: SystemTopology) -> list[int]:
    w = []
    for st in topo.stages:
        n = 0
        for tr in st.trains:
            n += len(tr.elements) + (tr.source is not None) + (tr.sink is not None)
        w.append(n)
    return w


def _choose_bounds(topo: SystemTopology, n: int, rng: random.Random) -> list[int]:
    """``n - 1`` stage boundaries with balanced item counts (random among near-optimal choices)."""
    w = _weights(topo)
    S_ = len(w)
    combos = []
    for bounds in itertools.combinations(range(1, S_), n - 1):
        edges = [0, *bounds, S_]
        loads = [sum(w[a:b]) for a, b in zip(edges, edges[1:])]
        combos.append((max(loads), list(bounds)))
    best = min(c[0] for c in combos)
    good = [b for score, b in combos if score <= best * 1.2 + 1]
    return rng.choice(good)


def _base(name: str) -> str:
    return _SIDE_RE.sub("", _NO_RE.sub("", name.strip())).strip()


def _plural(base: str) -> str:
    return base if base.endswith(("S", ")", ".")) else base + "S"


def group_name(names: Sequence[Optional[str]]) -> Optional[str]:
    """Destination / origin wording for a set of equipment names: one name as it is, identical
    bases in the plural (``NO.1 / NO.2 BALLAST PUMP`` -> ``BALLAST PUMPS``), else the two most common
    bases joined with ``&``."""
    names = [n for n in names if n]
    if not names:
        return None
    if len(set(names)) == 1:
        return names[0]
    cnt = Counter(_base(n) for n in names)
    if len(cnt) == 1:
        return _plural(next(iter(cnt)))
    top = [b for b, _ in cnt.most_common(2)]
    words = [(_plural(b) if cnt[b] > 1 else b) for b in top]
    joined = " & ".join(words)
    return joined if len(joined) <= _NAME_MAX else words[0]


def _first_named(tr: Train, reverse: bool) -> Optional[str]:
    """First (``reverse``: last) named equipment of a train, else its sink / source if named."""
    items = [el for el in tr.elements if isinstance(el, Item)]
    if reverse:
        items = items[::-1]
    for it in items:
        if it.category == Category.EQUIPMENT and it.name:
            return it.name[0]
    end = tr.source if reverse else tr.sink
    if end is not None and end.cls != S.OFF_PAGE_CONNECTOR and end.name:
        return end.name[0]
    return None


def _open_end(tr: Train) -> bool:
    return tr.sink is None and not tr.end_return


def _fallback_name(topo: SystemTopology) -> str:
    return {"BA": "BALLAST MAIN", "BG": "BILGE MAIN", "FO": "F.O. TRANSFER MAIN", "FS": "F.O. SUPPLY MAIN",
            "LO": "L.O. MAIN", "SW": "C.S.W. MAIN", "CF": "C.F.W. MAIN", "FM": "FIRE MAIN", "CA": "AIR MAIN",
            "ST": "STEAM MAIN"}.get(topo.code, "MAIN LINE")


def _label(style: str, outgoing: bool, name: str, other_sheet: int, mark: str, dwg: str) -> tuple[tuple[str, ...], str]:
    """(label lines, inside text) of one connector of a continuation pair."""
    word = "TO" if outgoing else "FROM"
    if style == "sheet":
        return (f"{word} {name}",), f"SH.{other_sheet}"
    if style == "mark":
        return (f"{word} {name} / SH.{other_sheet}",), mark
    cont = "CONT. ON" if outgoing else "CONT. FROM"
    return (f"{word} {name}", f"{cont} DWG {dwg} SH.{other_sheet}"), str(other_sheet)


def _marks(scheme: str, count: int, rng: random.Random) -> list[str]:
    if scheme == "letter":
        start = rng.randint(0, 3)
        return [_MARK_LETTERS[(start + i) % len(_MARK_LETTERS)] for i in range(count)]
    if scheme == "letter_digit":
        letter = rng.choice("ABCDEFGHJK")
        return [f"{letter}{i + 1}" for i in range(count)]
    start = rng.choice([1, 1, 11, 21])
    return [str(start + i) for i in range(count)]


def _drawing_no(topo: SystemTopology, rng: random.Random) -> str:
    assert topo.ship is not None
    hull = re.sub(r"\D", "", topo.ship.hull_no) or "0000"
    return rng.choice(["{h}-M-{n:03d}", "M{n:03d}-{c}-{h}", "{h}-PD-{c}-{m:02d}", "5{h}-{c}{n:03d}"]).format(
        h=hull, n=rng.randint(100, 899), c=topo.code, m=rng.randint(1, 30))


def _norm_ref(s: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", s.upper())


# --------------------------------------------------------------------------- splitting


def _connector(key: str, lines: tuple[str, ...], inside: str) -> Item:
    return Item(key, S.OFF_PAGE_CONNECTOR, name=lines, inside=(inside,) if inside else (),
                attrs={"service": _MS_MARKER + key})


def _split(topo: SystemTopology, n: int, rng: random.Random, style: str) -> tuple[_Plan, list[SystemTopology]]:
    """Cut ``topo`` into ``n`` sheet topologies with paired continuation connectors."""
    bounds = _choose_bounds(topo, n, rng)
    edges = [0, *bounds, len(topo.stages)]
    ranges = list(zip(edges, edges[1:]))
    sheet_of_stage = {k: s for s, (lo, hi) in enumerate(ranges) for k in range(lo, hi)}
    stage_of = _stage_of(topo)
    plan = _Plan(style=style, mark_scheme=rng.choice(MARK_SCHEMES), drawing_no=_drawing_no(topo, rng), n=n,
                 bounds=bounds, ranges=ranges)

    # -- pair specifications (names / sheets), in drawing order
    specs: list[dict] = []
    for b in bounds:
        left, right = topo.stages[b - 1], topo.stages[b]
        lg = [t.group for t in left.trains if _open_end(t)]
        rg = [t.group for t in right.trains if t.source is None]
        for g in sorted(set(lg) & set(rg)):
            dest = group_name([_first_named(t, False) for t in right.trains if t.source is None and t.group == g])
            orig = group_name([_first_named(t, True) for t in left.trains if _open_end(t) and t.group == g])
            specs.append({"kind": "header", "bound": b, "group": g, "dest": dest or _fallback_name(topo),
                          "orig": orig or _fallback_name(topo), "out_sheet": sheet_of_stage[b - 1],
                          "in_sheet": sheet_of_stage[b]})
    items = {it.key: it for it in topo.items()}
    trains = {tr.key: tr for st in topo.stages for tr in st.trains}
    for ri, rl in enumerate(topo.returns):
        origin_stage = stage_of.get(rl.train or rl.item or "")
        target_stage = stage_of.get(rl.target)
        if origin_stage is None or target_stage is None:
            continue
        if sheet_of_stage[origin_stage] == sheet_of_stage[target_stage]:
            continue
        tank = items.get(rl.target)
        dest = tank.name[0] if tank is not None and tank.name else _fallback_name(topo)
        if rl.train:
            orig = _first_named(trains[rl.train], True)
        else:
            orig = None
            for tr in trains.values():
                els = [el for el in tr.elements if isinstance(el, Item)]
                keys = [el.key for el in els]
                if rl.item in keys:
                    before = els[: keys.index(rl.item)][::-1]
                    orig = next((el.name[0] for el in before if el.category == Category.EQUIPMENT and el.name), None)
                    break
        specs.append({"kind": "return", "ret": ri, "dest": dest, "orig": orig or _fallback_name(topo),
                      "out_sheet": sheet_of_stage[origin_stage], "in_sheet": sheet_of_stage[target_stage]})

    marks = _marks(plan.mark_scheme, len(specs), rng)
    conns: dict[str, Item] = {}
    for k, sp in enumerate(specs):
        i, j = sp["out_sheet"] + 1, sp["in_sheet"] + 1
        mark = marks[k]
        out_lines, out_in = _label(style, True, sp["dest"], j, mark, plan.drawing_no)
        in_lines, in_in = _label(style, False, sp["orig"], i, mark, plan.drawing_no)
        ok, ik = f"ms{k + 1}o", f"ms{k + 1}i"
        conns[ok] = _connector(ok, out_lines, out_in)
        conns[ik] = _connector(ik, in_lines, in_in)
        key = {"sheet": f"sheet:{i}>{j}", "mark": f"mark:{mark}", "dwg": f"dwg:{plan.drawing_no}:{i}>{j}"}[style]
        plan.pairs.append(_Pair(ok, ik, sp["out_sheet"], sp["in_sheet"], sp["kind"], key))
        sp["out_key"], sp["in_key"] = ok, ik

    # -- one topology per sheet
    subs: list[SystemTopology] = []
    for s, (lo, hi) in enumerate(ranges):
        t = copy.deepcopy(topo, {id(topo.namer): topo.namer, id(topo.ship): topo.ship})
        stages = t.stages[lo:hi]
        tr_by_key = {tr.key: tr for st in stages for tr in st.trains}
        if lo > 0:  # incoming connectors feeding the first stage's header groups
            st0 = Stage([])
            for sp in specs:
                if sp["kind"] == "header" and sp["bound"] == lo:
                    st0.trains.append(Train(f"{sp['in_key']}t", [], source=conns[sp["in_key"]], group=sp["group"],
                                            template=f"msin{lo}"))
            stages.insert(0, st0)
        if hi < len(t.stages):  # outgoing connectors fed by the last stage's header groups
            st1 = Stage([])
            for sp in specs:
                if sp["kind"] == "header" and sp["bound"] == hi:
                    st1.trains.append(Train(f"{sp['out_key']}t", [], sink=conns[sp["out_key"]], group=sp["group"],
                                            template=f"msout{hi}"))
            stages.append(st1)
        kept_stage_ids = {id(st) for st in stages}
        t.stages = stages
        keys = set(_stage_of(t))
        t.bypasses = [bp for bp in t.bypasses if bp.start in keys and bp.end in keys]
        t.reliefs = [r for r in t.reliefs if r.inlet in keys and r.outlet in keys]
        t.side_links = [sl for sl in t.side_links if sl.item in keys]
        t.attachments = [a for a in t.attachments if a.item in keys]
        t.drains = [d for d in t.drains if id(d.stage) in kept_stage_ids]
        returns = []
        for ri, rl in enumerate(t.returns):
            origin, target = rl.train or rl.item, rl.target
            sp = next((x for x in specs if x["kind"] == "return" and x["ret"] == ri), None)
            if sp is None:
                if origin in keys and target in keys:
                    returns.append(rl)
                continue
            if sp["out_sheet"] == s:  # the return line leaves this sheet
                if rl.train and rl.train in tr_by_key:
                    tr = tr_by_key[rl.train]
                    tr.end_return = None
                    if rl.valve is not None:
                        tr.elements.append(rl.valve)
                    tr.sink = conns[sp["out_key"]]
                elif rl.item:
                    t.side_links.append(SideLink(rl.item, rl.port or "branch", conns[sp["out_key"]], True,
                                                 topo.medium, topo.code))
            if sp["in_sheet"] == s:  # ... and arrives at its tank here
                t.attachments.append(Attachment(target, conns[sp["in_key"]], side="outer", kind="process"))
        t.returns = returns
        subs.append(t)
    _unique_refs(subs, conns, rng)
    return plan, subs


def _unique_refs(subs: list[SystemTopology], conns: dict[str, Item], rng: random.Random) -> None:
    """Connectors to other systems get references that are unique in the set and never equal a
    continuation mark (a shared reference *means* a continuation on a real drawing set)."""
    used = {_norm_ref(it.inside[0]) for it in conns.values() if it.inside}
    for t in subs:
        for it in t.items():
            if it.cls != S.OFF_PAGE_CONNECTOR or it.key in conns or not it.inside:
                continue
            ref = it.inside[0]
            while _norm_ref(ref) in used or ref.isdigit():
                ref = rng.choice(["{a}-{n}", "{a}{n}", "S.{n}"]).format(a=rng.choice(_REF_LETTERS),
                                                                       n=rng.randint(1, 40))
            used.add(_norm_ref(ref))
            it.inside = (ref,) + tuple(it.inside[1:])


# --------------------------------------------------------------------------- assembly


def _retext(doc: PIDDocument, spec: DrawingSpec, i: int, new: str) -> None:
    """Replace title-block text ``i`` (centred in its cell) by ``new``, in the ground truth and the spec."""
    t, td = doc.texts[i], spec.texts[i]
    if t.text == new:
        return
    cx = (t.bbox.x0 + t.bbox.x1) / 2
    w = text_width(new, td.font, td.size)
    x0 = cx - w / 2
    spec.texts[i] = TextDraw(new, x0, td.y, td.size, td.font, td.rotate, td.color, td.layer)
    bb = text_bbox(new, x0, td.y, td.size, td.font, td.rotate)
    doc.texts[i] = t.model_copy(update={"text": new, "bbox": t.bbox.model_copy(
        update={"x0": round(bb[0], 4), "y0": round(bb[1], 4), "x1": round(bb[2], 4), "y1": round(bb[3], 4)})})


def _title_fields(doc: PIDDocument, spec: DrawingSpec, plan: _Plan, k: int) -> dict:
    """Unify the drawing number and write ``k OF n`` into the sheet field of sheet ``k`` (0-based)."""
    info = doc.info
    tb = [i for i, t in enumerate(doc.texts) if t.role == TextRole.TITLE_BLOCK]
    sheet_txt = info.sheet or "1 OF 1"
    new_sheet = f"{k + 1}/{plan.n}" if "/" in sheet_txt else f"{k + 1} OF {plan.n}"
    label = next((doc.texts[i] for i in tb if doc.texts[i].text == "SHEET"), None)
    cands = [i for i in tb if doc.texts[i].text == sheet_txt]
    if label is not None and cands:
        lc = label.bbox.center
        cands.sort(key=lambda i: doc.texts[i].bbox.center.dist(lc))
    for i in cands[:1]:
        _retext(doc, spec, i, new_sheet)
    dwg_old = info.drawing_no
    for i in tb:
        if dwg_old and doc.texts[i].text == dwg_old:
            _retext(doc, spec, i, plan.drawing_no)
    doc.info = info.model_copy(update={"drawing_no": plan.drawing_no, "sheet": new_sheet})
    spec.metadata["title"] = f"{plan.drawing_no} {info.title}"
    return {"page": k, "sheet": k + 1, "of": plan.n, "drawing_no": plan.drawing_no, "sheet_text": new_sheet}


def _merge(parts: list[tuple[PIDDocument, DrawingSpec]]) -> PIDDocument:
    """One document with every page; ids renumbered in page order (``o``/``j``/``p``/``t``)."""
    out = PIDDocument(source=parts[0][0].source, info=parts[0][0].info.model_copy(deep=True))
    no = nj = np_ = nt = 0
    for k, (doc, _) in enumerate(parts):
        omap = {o.id: f"o{no + i + 1}" for i, o in enumerate(doc.objects)}
        jmap = {j.id: f"j{nj + i + 1}" for i, j in enumerate(doc.junctions)}
        pmap = {p.id: f"p{np_ + i + 1}" for i, p in enumerate(doc.pipes)}
        tmap = {t.id: f"t{nt + i + 1}" for i, t in enumerate(doc.texts)}
        no, nj, np_, nt = no + len(omap), nj + len(jmap), np_ + len(pmap), nt + len(tmap)
        pg = doc.pages[0]
        out.pages.append(PageInfo(index=k, width=pg.width, height=pg.height, rotation=pg.rotation,
                                  drawing_area=pg.drawing_area))
        for o in doc.objects:
            out.objects.append(o.model_copy(
                update={"id": omap[o.id], "page": k, "text_ids": [tmap[t] for t in o.text_ids]}, deep=True))
        for j in doc.junctions:
            out.junctions.append(j.model_copy(update={"id": jmap[j.id], "page": k}))

        def ep(e):
            return e.model_copy(update={"object_id": omap.get(e.object_id) if e.object_id else None,
                                        "junction_id": jmap.get(e.junction_id) if e.junction_id else None})

        for p in doc.pipes:
            out.pipes.append(p.model_copy(update={"id": pmap[p.id], "page": k, "start": ep(p.start), "end": ep(p.end),
                                                  "text_ids": [tmap[t] for t in p.text_ids]}, deep=True))
        for t in doc.texts:
            owner = t.owner_id
            if owner:
                owner = omap.get(owner) or pmap.get(owner)
            out.texts.append(t.model_copy(update={"id": tmap[t.id], "page": k, "owner_id": owner}))
    return out


def _fix_return_inlets(doc: PIDDocument, in_ids: set[str]) -> None:
    """The tank-side connector of a cut return line feeds the tank: its pipe flows connector -> tank."""
    for p in doc.pipes:
        if p.end.object_id in in_ids and p.flow_direction == 1:
            p.start, p.end = p.end, p.start
            p.points = p.points[::-1]
    for o in doc.objects:
        if o.id in in_ids:
            o.attributes["direction"] = "in"
            o.attributes["arrow_along_flow"] = False


def pdf_bytes(specs: Sequence[DrawingSpec]) -> bytes:
    """Deterministic multi-page PDF: one page per spec (rendered by :func:`pdf_writer.render`)."""
    out = pymupdf.open()
    try:
        for spec in specs:
            d = render(spec)
            try:
                out.insert_pdf(d)
            finally:
                d.close()
        meta = {k: "" for k in ("title", "author", "subject", "keywords", "creator", "producer")}
        meta.update({k: v for k, v in specs[0].metadata.items() if k in meta})
        meta.update({"creationDate": "", "modDate": ""})
        out.set_metadata(meta)
        return out.tobytes(garbage=3, deflate=True, no_new_id=True)
    finally:
        out.close()


def generate_multisheet(seed: int, system: Optional[SystemType] = None, sheets: Optional[int] = None,
                        style: Optional[str] = None) -> tuple[PIDDocument, bytes]:
    """One ship system drawn on ``sheets`` (2..3; default random 2 or 3) sheets of one drawing,
    joined by paired off-page connectors in yard ``style`` (``YARD_STYLES``; default random).

    Returns the ground truth (all pages, globally unique ids, ``meta["sheet_links"]``) and the
    PDF bytes. A topology with fewer stages than ``sheets`` uses one stage per sheet."""
    if sheets is not None and sheets < 2:
        raise ValueError("sheets must be >= 2")
    if style is not None and style not in YARD_STYLES:
        raise ValueError(f"style must be one of {YARD_STYLES}, not {style!r}")
    rng0 = random.Random(f"multisheet:{seed}")
    if system is None or system == SystemType.UNKNOWN:
        system = rng0.choice(ALL_SYSTEMS)
    n_req = sheets if sheets is not None else rng0.choice((2, 2, 3))
    yard = style or rng0.choice(YARD_STYLES)
    errors: list[str] = []
    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random(f"multisheet:{seed}:{attempt}")
        topo = build_topology(system, rng, complexity=max(0.3, 1.0 - 0.12 * attempt))
        n = max(2, min(n_req, len(topo.stages)))
        st = random_style(rng)
        plan, subs = _split(topo, n, rng, yard)
        parts: list[tuple[PIDDocument, DrawingSpec]] = []
        try:
            for k, sub in enumerate(subs):
                stk = st if k == 0 else replace(st, paper=parts[0][1].paper)
                srng = random.Random(f"multisheet:{seed}:{attempt}:sheet{k}")
                try:
                    parts.append(_generate(sub, stk, srng, seed, attempt))
                except LayoutError:
                    if k == 0:
                        raise
                    srng = random.Random(f"multisheet:{seed}:{attempt}:sheet{k}:free")
                    parts.append(_generate(sub, st, srng, seed, attempt))
        except LayoutError as exc:
            errors.append(str(exc))
            continue
        return _assemble(seed, system, plan, parts, attempt)
    raise RuntimeError(f"multi-sheet generation failed for seed={seed} system={system.value}: {errors}")


def _assemble(seed: int, system: SystemType, plan: _Plan, parts: list[tuple[PIDDocument, DrawingSpec]],
              attempt: int) -> tuple[PIDDocument, bytes]:
    sheet_rows = [_title_fields(doc, spec, plan, k) for k, (doc, spec) in enumerate(parts)]
    doc = _merge(parts)
    doc.source = f"synthetic-multisheet:{seed}"
    doc.info = parts[0][0].info.model_copy(deep=True)
    # connector objects of the continuation pairs (found by the temporary marker, then cleaned)
    by_key: dict[str, str] = {}
    for o in doc.objects:
        svc = o.attributes.get("service")
        if isinstance(svc, str) and svc.startswith(_MS_MARKER):
            by_key[svc[len(_MS_MARKER):]] = o.id
            del o.attributes["service"]
    links = []
    for pr in plan.pairs:
        a, b = by_key.get(pr.out_key), by_key.get(pr.in_key)
        if a is None or b is None:  # a cut that the layout left out (cannot happen for headers)
            continue
        links.append({"a": a, "b": b, "key": pr.key, "kind": pr.kind, "sheets": [pr.out_sheet + 1, pr.in_sheet + 1],
                      "directed": True, "source": "gt"})
    _fix_return_inlets(doc, {by_key[p.in_key] for p in plan.pairs if p.kind == "return" and p.in_key in by_key})
    pages_meta = [p[0].meta for p in parts]
    doc.meta = {
        "generator": "shippid.datasets.multisheet", "seed": seed, "attempt": attempt, "system": system.value,
        "paper": [m.get("paper") for m in pages_meta],
        "sheet_links": links,
        "sheets": [{k: v for k, v in row.items() if k != "sheet_text"} for row in sheet_rows],
        "multisheet": {"style": plan.style, "mark_scheme": plan.mark_scheme if plan.style == "mark" else None,
                       "drawing_no": plan.drawing_no, "sheets": plan.n, "stage_cuts": plan.bounds,
                       "stage_ranges": [list(r) for r in plan.ranges]},
        "labels_dropped": sum(int(m.get("labels_dropped", 0)) for m in pages_meta),
        "pages_meta": [{k: m.get(k) for k in ("attempt", "paper", "symbol_scale", "sheet_layout", "bands", "fanout",
                                              "labels_dropped")} for m in pages_meta],
    }
    return doc, pdf_bytes([spec for _, spec in parts])
