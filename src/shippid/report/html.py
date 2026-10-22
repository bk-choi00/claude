"""Self-contained HTML report of an :class:`~shippid.model.AnalysisReport`.

One HTML file, no network resources: the drawing page is embedded as a PNG data URL (rendered
with PyMuPDF) under an SVG overlay in PDF-point coordinates -- object boxes coloured by
category, pipes coloured by medium (ISO 14726, :func:`shippid.catalog.medium_rgb`), junctions --
with hover tooltips, category filters and zoom. Equipment / valve / instrument / line / flow-path
/ finding tables highlight their objects on the drawing when a row is clicked. Every piece of
text is HTML-escaped; the inline script only reads ``data-*`` attributes and sets
``textContent``. UI labels are Korean with the English term alongside.
"""

from __future__ import annotations

import base64
import html
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from ..analysis.analyzer import pages_not_analysed
from ..analysis.graph import flow_path_sheet_links, object_labels, sheet_link_pairs
from ..analysis.rules import RULES
from ..catalog import MEDIUM_COLORS, SYSTEMS, medium_rgb
from ..model import (
    AnalysisReport,
    Category,
    Finding,
    Layout3D,
    Medium,
    PIDDocument,
    PipeKind,
    SymbolClass,
    SystemType,
    Verdict,
)

log = logging.getLogger(__name__)

CATEGORY_COLORS = {
    Category.EQUIPMENT: "#1f6feb",
    Category.VALVE: "#cf222e",
    Category.INSTRUMENT: "#1a7f37",
    Category.FITTING: "#9a6700",
    Category.CONNECTOR: "#8250df",
}
CATEGORY_KO = {
    Category.EQUIPMENT: "설비", Category.VALVE: "밸브", Category.INSTRUMENT: "계기",
    Category.FITTING: "배관 부속", Category.CONNECTOR: "도면 연결",
}
VERDICT_LABELS = {
    Verdict.FAIL: "부적합 FAIL", Verdict.INSUFFICIENT_DATA: "자료부족 INSUFFICIENT",
    Verdict.PASS: "적합 PASS", Verdict.NOT_APPLICABLE: "해당없음 N/A",
}
_VERDICT_ORDER = {Verdict.FAIL: 0, Verdict.INSUFFICIENT_DATA: 1, Verdict.PASS: 2, Verdict.NOT_APPLICABLE: 3}
_SEVERITY_KO = {"error": "중대 error", "warning": "경고 warning", "info": "참고 info"}
_MEDIUM_KO = {
    Medium.SEA_WATER: "해수", Medium.FRESH_WATER: "청수", Medium.BALLAST_WATER: "평형수", Medium.BILGE: "빌지",
    Medium.FUEL_OIL: "연료유", Medium.LUBE_OIL: "윤활유", Medium.COMPRESSED_AIR: "압축공기", Medium.STEAM: "증기",
    Medium.FIRE_WATER: "소화수", Medium.UNKNOWN: "미상",
}

COLUMNS: dict[str, list[tuple[str, str]]] = {
    "equipment": [("tag", "태그 Tag"), ("name", "명칭 Name"), ("type_ko", "종류 Type"), ("class", "Class"),
                  ("rating", "용량 Rating"), ("connections", "연결 Conn."), ("connected_to", "연결 대상 Connected to"),
                  ("isolation_valves", "차단 밸브 Isolation valves"), ("confidence", "신뢰도 Conf.")],
    "valves": [("tag", "태그 Tag"), ("type_ko", "종류 Type"), ("class", "Class"), ("actuator_ko", "구동 Actuator"),
               ("remote_operated", "원격 Remote"), ("size", "관경 Size"), ("line_numbers", "라인 번호 Line no."),
               ("medium", "유체 Medium"), ("connected_to", "연결 대상 Connected to"), ("confidence", "신뢰도 Conf.")],
    "instruments": [("tag", "태그 Tag"), ("letters", "ISA"), ("description", "기능 Function"),
                    ("location", "위치 Location"), ("measured_on", "측정 위치 Measured on"),
                    ("signal_to", "신호 대상 Signal to"), ("service", "용도 Service"), ("confidence", "신뢰도 Conf.")],
    "lines": [("line_number", "라인 번호 Line no."), ("size", "관경 Size"), ("medium", "유체 Medium"),
              ("system_code", "계통 System"), ("spec", "사양 Spec"), ("from", "출발 From"), ("to", "도착 To"),
              ("valves_fittings", "밸브·부속 Valves / fittings"), ("runs", "구간 Runs"),
              ("length_pt", "길이 Length (pt)")],
}

_CSS = """
:root{--bg:#f3f5f8;--fg:#1f2328;--muted:#59636e;--card:#fff;--line:#d1d9e0;--accent:#0d2a4a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI","Malgun Gothic",
"Apple SD Gothic Neo","Noto Sans KR",sans-serif}
header{background:var(--accent);color:#fff;padding:14px 24px}
header h1{margin:0 0 4px;font-size:20px}
header .meta{opacity:.85;font-size:13px}
nav{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid var(--line);padding:6px 24px}
nav a{margin-right:14px;color:#0969da;text-decoration:none;font-size:13px}
section{background:var(--card);margin:14px 24px;padding:14px 18px;border:1px solid var(--line);border-radius:8px}
h2{font-size:17px;margin:0 0 10px}
.muted{color:var(--muted);font-size:12px}
.warn{background:#fff8c5;border:1px solid #d4a72c;padding:8px 10px;border-radius:6px;margin:8px 0}
.cards{display:flex;flex-wrap:wrap;gap:8px}
.card{border:1px solid var(--line);border-radius:8px;padding:6px 12px;min-width:104px;background:#fafbfc}
.card b{display:block;font-size:20px}
.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:8px;font-size:13px}
.toolbar label{white-space:nowrap}
.toolbar button{border:1px solid var(--line);background:#f6f8fa;border-radius:6px;padding:2px 10px;cursor:pointer}
.swatch{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-1px;margin-right:3px}
.legend span{margin-right:12px;font-size:12px;white-space:nowrap}
.drawing-wrap{overflow:auto;max-height:82vh;border:1px solid var(--line);background:#fff}
svg.drawing{display:block;width:100%;height:auto}
svg .obj{fill-opacity:.07;stroke-width:1.4;cursor:pointer}
svg .obj:hover{fill-opacity:.28}
svg .unknown{stroke-dasharray:4 2}
svg .pipe{fill:none;stroke-width:2;stroke-opacity:.8;cursor:pointer;stroke-linejoin:round}
svg .pipe:hover{stroke-width:4;stroke-opacity:1}
svg .kind-signal{stroke-dasharray:6 4}
svg .jn{stroke:#111;stroke-width:1;cursor:pointer}
svg .lbl{font-family:Arial,Helvetica,sans-serif;pointer-events:none;paint-order:stroke;stroke:#fff;stroke-width:2px}
svg .hl{stroke:#ff9f00!important;stroke-width:5!important;stroke-opacity:1!important;fill:#ffbf00;fill-opacity:.3!important}
svg.drawing.hide-equipment .cat-equipment,svg.drawing.hide-valve .cat-valve,svg.drawing.hide-instrument .cat-instrument,
svg.drawing.hide-fitting .cat-fitting,svg.drawing.hide-connector .cat-connector,svg.drawing.hide-process .kind-process,
svg.drawing.hide-signal .kind-signal,svg.drawing.hide-signal .kind-instrument_tap,svg.drawing.hide-junction .jn,
svg.drawing.hide-label .lbl,svg.drawing.hide-image .page-image{display:none}
svg.drawing.dim .page-image{opacity:.35}
.tablewrap{max-height:520px;overflow:auto;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{position:sticky;top:0;background:#eef1f4;z-index:1}
th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
tr[data-ids]{cursor:pointer}
tr[data-ids]:hover{background:#fff8c5}
tr.sel{background:#ffe27a!important}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11.5px;font-weight:600;color:#fff;white-space:nowrap}
.v-fail{background:#cf222e}.v-insufficient_data{background:#bf8700}.v-pass{background:#1a7f37}
.v-not_applicable{background:#8c959f}.src-rule{background:#57606a}.src-gemma{background:#8250df}
.src-other{background:#0969da}.sev-error{color:#cf222e;font-weight:600}.sev-warning{color:#9a6700}
pre.summary{white-space:pre-wrap;font-family:inherit;background:#f6f8fa;padding:10px;border-radius:6px;margin:6px 0}
#tip{position:fixed;display:none;pointer-events:none;background:rgba(18,22,28,.94);color:#fff;padding:6px 9px;
border-radius:6px;font-size:12px;white-space:pre-line;max-width:440px;z-index:100}
iframe.viewer{width:100%;height:640px;border:1px solid var(--line);border-radius:6px}
dl.kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;margin:0}
dl.kv dt{color:var(--muted)}dl.kv dd{margin:0}
"""

_JS = """
(function(){
  var tip=document.getElementById('tip');
  var byId={};
  document.querySelectorAll('svg.drawing [data-id]').forEach(function(el){
    var k=el.getAttribute('data-id');(byId[k]=byId[k]||[]).push(el);
  });
  function clearSel(){
    document.querySelectorAll('svg .hl').forEach(function(e){e.classList.remove('hl');});
    document.querySelectorAll('tr.sel').forEach(function(r){r.classList.remove('sel');});
  }
  function highlight(ids,scroll){
    clearSel();var first=null;
    ids.forEach(function(id){(byId[id]||[]).forEach(function(el){el.classList.add('hl');if(!first)first=el;});});
    if(scroll&&first){first.scrollIntoView({behavior:'smooth',block:'center',inline:'center'});}
    return first;
  }
  document.querySelectorAll('svg.drawing [data-tip]').forEach(function(el){
    el.addEventListener('mousemove',function(e){
      tip.textContent=el.getAttribute('data-tip');tip.style.display='block';
      var x=e.clientX+14,y=e.clientY+14;
      if(x+tip.offsetWidth>window.innerWidth)x=e.clientX-tip.offsetWidth-10;
      if(y+tip.offsetHeight>window.innerHeight)y=e.clientY-tip.offsetHeight-10;
      tip.style.left=x+'px';tip.style.top=y+'px';
    });
    el.addEventListener('mouseleave',function(){tip.style.display='none';});
    el.addEventListener('click',function(){
      var id=el.getAttribute('data-id');highlight([id],false);
      var row=document.querySelector('tr[data-ids~="'+(window.CSS&&CSS.escape?CSS.escape(id):id)+'"]');
      if(row)row.classList.add('sel');
    });
  });
  document.querySelectorAll('tr[data-ids]').forEach(function(tr){
    tr.addEventListener('click',function(){
      var ids=tr.getAttribute('data-ids').split(/\\s+/).filter(Boolean);
      if(highlight(ids,true))tr.classList.add('sel');else tr.classList.add('sel');
    });
  });
  document.querySelectorAll('input[data-hide]').forEach(function(cb){
    cb.addEventListener('change',function(){
      document.querySelectorAll('svg.drawing').forEach(function(s){s.classList.toggle('hide-'+cb.getAttribute('data-hide'),!cb.checked);});
    });
  });
  var dim=document.getElementById('dim');
  if(dim)dim.addEventListener('change',function(){
    document.querySelectorAll('svg.drawing').forEach(function(s){s.classList.toggle('dim',dim.checked);});
  });
  var zoom=1;
  function applyZoom(){document.querySelectorAll('svg.drawing').forEach(function(s){s.style.width=(zoom*100)+'%';});}
  document.querySelectorAll('button[data-zoom]').forEach(function(b){
    b.addEventListener('click',function(){
      var z=b.getAttribute('data-zoom');zoom=z==='fit'?1:Math.max(0.5,Math.min(8,zoom*parseFloat(z)));applyZoom();
    });
  });
  var vf=document.getElementById('verdict-filter');
  if(vf)vf.addEventListener('change',function(){
    document.querySelectorAll('#findings-table tbody tr').forEach(function(r){
      r.style.display=(vf.value==='all'||r.getAttribute('data-verdict')===vf.value)?'':'none';
    });
  });
})();
"""


def _e(x: Any) -> str:
    """HTML-escape any value (attributes and text)."""
    if x is None:
        return ""
    if isinstance(x, bool):
        return "예 yes" if x else "-"
    if isinstance(x, float):
        x = f"{x:.3g}" if abs(x) < 10 else f"{x:.1f}"
    return html.escape(str(x), quote=True)


def _v(x: Any) -> str:
    return getattr(x, "value", str(x))


def _safe_href(href: Optional[str]) -> Optional[str]:
    """Relative paths and http(s)/file URLs only (no ``javascript:`` / ``data:`` / control chars)."""
    if not href or not href.strip() or any(ord(c) < 32 for c in href):
        return None
    href = href.strip()
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):", href)
    if m:
        return href if m.group(1).lower() in ("http", "https", "file") else None
    return href


def _rgb_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def _fmt_pts(points: Sequence[Any]) -> str:
    return " ".join(f"{p.x:.1f},{p.y:.1f}" for p in points)


# --------------------------------------------------------------------------- page images


def _page_sizes(doc: PIDDocument) -> dict[int, tuple[float, float]]:
    sizes = {p.index: (p.width, p.height) for p in doc.pages}
    used = {o.page for o in doc.objects} | {p.page for p in doc.pipes} | {j.page for j in doc.junctions}
    for idx in sorted(used - set(sizes)) or ([] if sizes else [0]):
        xs: list[float] = [0.0]
        ys: list[float] = [0.0]
        for o in doc.objects:
            if o.page == idx:
                xs.append(o.bbox.x1)
                ys.append(o.bbox.y1)
        for p in doc.pipes:
            if p.page == idx:
                xs += [q.x for q in p.points]
                ys += [q.y for q in p.points]
        for t in doc.texts:
            if t.page == idx:
                xs.append(t.bbox.x1)
                ys.append(t.bbox.y1)
        sizes[idx] = (max(xs) + 20.0, max(ys) + 20.0)
    return dict(sorted(sizes.items()))


def render_page_png(pdf_path: str | Path, page_index: int, max_px: int = 3000,
                    max_dpi: float = 150.0) -> Optional[tuple[bytes, float, float]]:
    """PNG bytes of one PDF page (longest side <= ``max_px``) and the page size in points."""
    import pymupdf

    try:
        with pymupdf.open(str(pdf_path)) as pdf:
            if page_index >= pdf.page_count:
                return None
            page = pdf[page_index]
            w, h = page.rect.width, page.rect.height
            dpi = int(max(36.0, min(max_dpi, 72.0 * max_px / max(w, h, 1.0))))
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            return pix.tobytes("png"), w, h
    except Exception as exc:  # unreadable / missing PDF: overlay-only report
        log.warning("cannot render %s page %d: %s", pdf_path, page_index, exc)
        return None


# --------------------------------------------------------------------------- svg overlay


def _object_tip(o: Any, label: str) -> str:
    lines = [f"{o.tag or '-'}" + (f" - {label}" if label else "")]
    lines.append(f"{CATEGORY_KO.get(o.category, '')} {o.category.value} / {o.cls.value}")
    if o.category == Category.VALVE and o.actuator.value != "none":
        lines.append(f"actuator: {o.actuator.value}")
    lines.append(f"id {o.id} | conf {o.confidence:.2f} | source {o.source}")
    return "\n".join(lines)


def _svg(doc: PIDDocument, page: int, w: float, h: float, image: Optional[bytes], labels: dict[str, str],
         node_names: dict[str, str]) -> str:
    fs = max(4.0, min(w, h) / 230.0)
    out = [f'<svg class="drawing" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.1f} {h:.1f}" '
           f'data-page="{page}" role="img" aria-label="P&amp;ID page {page + 1}">',
           f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" fill="#fff"/>']
    if image:
        b64 = base64.b64encode(image).decode("ascii")
        out.append(f'<image class="page-image" x="0" y="0" width="{w:.1f}" height="{h:.1f}" '
                   f'preserveAspectRatio="none" href="data:image/png;base64,{b64}"/>')
    out.append('<g class="pipes">')
    for p in doc.pipes:
        if p.page != page or len(p.points) < 2:
            continue
        color = _rgb_hex(medium_rgb(p.medium)) if p.kind != PipeKind.SIGNAL else "#555"
        a = node_names.get(p.start.node_id or "", "open end") if p.start.node_id else "open end"
        b = node_names.get(p.end.node_id or "", "open end") if p.end.node_id else "open end"
        tip = (f"{p.line_number or '(라인 번호 없음 no line no.)'} | {p.size or '-'} | "
               f"{_MEDIUM_KO.get(p.medium, '')} {p.medium.value} | {p.kind.value}\n{a} -> {b}\n"
               f"id {p.id} | length {p.length:.0f} pt | source {p.source}")
        out.append(f'<polyline class="pipe kind-{_e(p.kind.value)}" data-id="{_e(p.id)}" data-tip="{_e(tip)}" '
                   f'stroke="{color}" vector-effect="non-scaling-stroke" points="{_fmt_pts(p.points)}"/>')
    out.append('</g><g class="junctions">')
    r = max(1.2, fs * 0.35)
    for j in doc.junctions:
        if j.page != page:
            continue
        tip = f"분기점 junction {j.id} ({j.kind}{', dot' if j.dotted else ''})"
        out.append(f'<circle class="jn" data-id="{_e(j.id)}" data-tip="{_e(tip)}" cx="{j.position.x:.1f}" '
                   f'cy="{j.position.y:.1f}" r="{r:.1f}" fill="{"#111" if j.dotted else "#fff"}"/>')
    out.append('</g><g class="objects">')
    for o in doc.objects:
        if o.page != page:
            continue
        color = CATEGORY_COLORS.get(o.category, "#e16f24")
        unknown = " unknown" if o.cls == SymbolClass.UNKNOWN else ""
        b = o.bbox
        label = labels.get(o.id, "")
        out.append(f'<rect class="obj cat-{_e(o.category.value)}{unknown}" data-id="{_e(o.id)}" '
                   f'data-tip="{_e(_object_tip(o, label))}" x="{b.x0:.1f}" y="{b.y0:.1f}" '
                   f'width="{max(b.width, 0.5):.1f}" height="{max(b.height, 0.5):.1f}" stroke="{color}" '
                   f'fill="{color}" vector-effect="non-scaling-stroke"/>')
        text = o.tag or (label[:28] if label else o.cls.value)
        out.append(f'<text class="lbl cat-{_e(o.category.value)}" x="{b.x0:.1f}" y="{b.y0 - fs * 0.3:.1f}" '
                   f'font-size="{fs:.1f}" fill="{color}">{_e(text)}</text>')
    out.append("</g></svg>")
    return "".join(out)


# --------------------------------------------------------------------------- html pieces


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str]], ids_key: str, table_id: str = "") -> str:
    if not rows:
        return '<p class="muted">항목 없음 (none)</p>'
    head = "".join(f"<th>{_e(lbl)}</th>" for _, lbl in cols)
    body = []
    for r in rows:
        ids = " ".join(str(r.get(ids_key) or "").split())
        cells = "".join(f"<td>{_e(r.get(k, ''))}</td>" for k, _ in cols)
        body.append(f'<tr data-ids="{_e(ids)}">{cells}</tr>')
    tid = f' id="{_e(table_id)}"' if table_id else ""
    return f'<div class="tablewrap"><table{tid}><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _source_badge(source: str) -> str:
    cls = "src-rule" if source == "rule" else ("src-gemma" if source == "gemma" else "src-other")
    label = {"rule": "규칙 rule", "gemma": "Gemma"}.get(source, source)
    return f'<span class="badge {cls}">{_e(label)}</span>'


def _findings_table(findings: list[Finding], names: dict[str, str], rule_titles_ko: dict[str, str]) -> str:
    if not findings:
        return '<p class="muted">검토 결과 없음 (no findings)</p>'
    sev = {"error": 0, "warning": 1, "info": 2}
    ordered = sorted(findings, key=lambda f: (_VERDICT_ORDER.get(f.verdict, 9), sev.get(f.severity.value, 9),
                                              f.rule_id, f.id))
    rows = []
    for f in ordered:
        objs = ", ".join(names.get(i, i) for i in f.object_ids[:8]) + (" ..." if len(f.object_ids) > 8 else "")
        ko = rule_titles_ko.get(f.rule_id)
        title = f"<b>{_e(f.title)}</b>" + (f'<br><span class="muted">{_e(ko)}</span>' if ko else "")
        rows.append(
            f'<tr data-ids="{_e(" ".join(f.object_ids))}" data-verdict="{_e(f.verdict.value)}">'
            f'<td><span class="badge v-{_e(f.verdict.value)}">{_e(VERDICT_LABELS.get(f.verdict, f.verdict.value))}'
            f'</span></td><td class="sev-{_e(f.severity.value)}">{_e(_SEVERITY_KO.get(f.severity.value))}</td>'
            f'<td><code>{_e(f.rule_id)}</code></td><td>{title}<br>{_e(f.description)}</td><td>{_e(objs)}</td>'
            f'<td>{_e(f.reference or "")}</td><td>{_source_badge(f.source)}</td><td>{_e(f"{f.confidence:.2f}")}</td></tr>')
    head = "".join(f"<th>{h}</th>" for h in ("판정 Verdict", "심각도 Severity", "규칙 Rule", "내용 Finding",
                                             "대상 Objects", "근거 Reference", "출처 Source", "신뢰도 Conf."))
    return (f'<div class="tablewrap"><table id="findings-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _summary_badge(source: str) -> str:
    if source == "template":
        return '<span class="badge src-rule">결정론적 템플릿 deterministic template</span>'
    if source == "gemma":
        return '<span class="badge src-gemma">Gemma 4 31B</span>'
    return f'<span class="badge src-other">{_e(source)}</span>'


def _cards(counts: dict[str, int]) -> str:
    items = [("objects", "객체 Objects"), ("equipment", "설비 Equipment"), ("valves", "밸브 Valves"),
             ("instruments", "계기 Instruments"), ("pipes", "배관 Pipes"), ("line_numbers", "라인 Lines"),
             ("flow_paths", "유동 경로 Flow paths"), ("findings_fail", "부적합 FAIL"),
             ("findings_insufficient_data", "자료부족 Insufficient"), ("findings_pass", "적합 PASS")]
    return '<div class="cards">' + "".join(
        f'<div class="card"><b>{_e(counts.get(k, 0))}</b><span class="muted">{_e(lbl)}</span></div>'
        for k, lbl in items) + "</div>"


def _skipped_warning(skipped: dict[str, list[int]]) -> str:
    """Warning box naming the pages the extractor skipped (empty string when none)."""
    unreadable, empty = skipped.get("unreadable_pages") or [], skipped.get("empty_pages") or []
    if not unreadable and not empty:
        return ""
    items = []
    if unreadable:
        pages = ", ".join(str(p + 1) for p in unreadable)
        items.append(f"<li>스캔/래스터 페이지 (판독 불가) scanned / raster-only (unreadable): 페이지 page {_e(pages)}</li>")
    if empty:
        pages = ", ".join(str(p + 1) for p in empty)
        items.append(f"<li>내용 없는 페이지 empty: 페이지 page {_e(pages)}</li>")
    return ('<div class="warn" id="pages-not-analysed"><b>분석되지 않은 페이지 (pages not analysed)</b>'
            f'<ul>{"".join(items)}</ul>이 페이지들의 객체·배관은 추출되지 않았으며 검토 결과에 포함되지 않습니다. '
            '스캔 도면은 CAD에서 내보낸 벡터 PDF로 다시 받으세요 (no objects or pipes were extracted from these pages '
            'and the findings do not cover them; obtain the vector PDF exported from CAD).</div>')


def _kv(pairs: Iterable[tuple[str, Any]]) -> str:
    return '<dl class="kv">' + "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in pairs if v not in (None, "")) + "</dl>"


def _systems_text(doc: PIDDocument, meta: dict[str, Any]) -> str:
    names = []
    for s in meta.get("systems") or [doc.info.system.value]:
        try:
            st = SystemType(s)
        except ValueError:
            continue
        if st in SYSTEMS:
            names.append(f"{SYSTEMS[st].name_ko} {SYSTEMS[st].name_en} ({SYSTEMS[st].code})")
    return ", ".join(names) or "계통 미상 unknown"


def _layout_section(layout: Optional[Layout3D], viewer_href: Optional[str]) -> str:
    if layout is None and not viewer_href:
        return ""
    parts = ['<section id="model3d"><h2>3D 모델 (3D model)</h2>']
    href = _safe_href(viewer_href)
    if viewer_href and not href:
        parts.append('<p class="warn">3D 뷰어 링크가 안전하지 않아 생략했습니다 (unsafe viewer link omitted).</p>')
    if href:
        parts.append(f'<p><a href="{_e(href)}" target="_blank" rel="noopener">3D 뷰어 열기 (open 3D viewer)</a></p>'
                     f'<iframe class="viewer" src="{_e(href)}" loading="lazy" title="3D viewer"></iframe>')
    if layout is not None:
        planner = {"gemma": "Gemma 4 31B 배치 계획 (Gemma-planned)",
                   "heuristic": "결정론적 배치 (heuristic planner)"}.get(layout.planner, layout.planner)
        parts.append(_kv([
            ("배치 방식 Planner", planner),
            ("배치 객체 Placements", len(layout.placements)),
            ("3D 배관 Routes", len(layout.routes)),
            ("갑판 Decks", ", ".join(f"{d.name} {d.z:g} m" for d in layout.decks)),
            ("범위 Bounds (m)", f"{layout.bounds_min.as_tuple()} - {layout.bounds_max.as_tuple()}"),
            ("LLM", f"{layout.llm.backend} {layout.llm.model or ''} calls={layout.llm.calls}"
             if layout.llm.calls or layout.llm.backend != "none" else "none"),
        ]))
        if layout.notes:
            parts.append("<ul>" + "".join(f"<li>{_e(n)}</li>" for n in layout.notes[:20]) + "</ul>")
    parts.append("</section>")
    return "".join(parts)


def _llm_section(report: AnalysisReport) -> str:
    u = report.llm
    parts = ['<section id="llm"><h2>LLM 사용 내역 (LLM usage)</h2>']
    if u.backend == "none" and not u.calls:
        parts.append("<p>LLM 미사용 - 모든 결과는 결정론적 규칙/알고리즘으로 생성되었습니다 "
                     "(no LLM used; all results are deterministic).</p>")
    if u.backend == "mock":
        parts.append('<p class="warn">MOCK 백엔드 출력입니다 - Gemma 출력이 아닙니다 '
                     '(mock backend output, not produced by Gemma).</p>')
    parts.append(_kv([("백엔드 Backend", u.backend), ("모델 Model", u.model), ("호출 Calls", u.calls),
                      ("입력 토큰 Prompt tokens", u.prompt_tokens), ("출력 토큰 Completion tokens", u.completion_tokens),
                      ("시간 Seconds", f"{u.seconds:.1f}"),
                      ("검토 상태 Review", report.meta.get("review_status")), ("검토자 Reviewer", report.meta.get("reviewer")),
                      ("요약 출처 Summary source", report.meta.get("summary_source"))]))
    if u.errors:
        parts.append("<p><b>오류 Errors</b></p><ul>" + "".join(f"<li>{_e(e)}</li>" for e in u.errors[:50]) + "</ul>")
    parts.append("</section>")
    return "".join(parts)


def _legend(doc: PIDDocument) -> str:
    media = sorted({p.medium for p in doc.pipes}, key=lambda m: list(Medium).index(m))
    cats = "".join(f'<span><i class="swatch" style="background:{c}"></i>{_e(CATEGORY_KO[k])} {_e(k.value)}</span>'
                   for k, c in CATEGORY_COLORS.items())
    meds = "".join(f'<span><i class="swatch" style="background:{_rgb_hex(medium_rgb(m))}"></i>'
                   f'{_e(_MEDIUM_KO.get(m, ""))} {_e(m.value)} ({_e(MEDIUM_COLORS[m][0])})</span>' for m in media)
    return f'<div class="legend">{cats}</div><div class="legend">{meds}</div>'


def _total_text(meta: dict[str, Any]) -> str:
    total = meta.get("flow_paths_total")
    if meta.get("flow_paths_total_exact", True) is False:
        return f"최소 at least {total}; 열거 한도 도달 enumeration budget reached"
    return str(total)


# --------------------------------------------------------------------------- entry point


def write_html_report(report: AnalysisReport, out_html: str | Path, pdf_path: Optional[str | Path] = None,
                      viewer_href: Optional[str] = None, layout: Optional[Layout3D | dict] = None, *,
                      max_image_px: int = 3000) -> Path:
    """Write the self-contained HTML report and return its path."""
    if isinstance(layout, dict):
        layout = Layout3D.model_validate(layout)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    doc = report.document
    labels = object_labels(doc)
    node_names: dict[str, str] = {}
    for o in doc.objects:
        lab = labels.get(o.id)
        node_names[o.id] = (f"{lab} [{o.tag}]" if lab and o.tag and o.tag not in lab else (lab or o.tag)) or \
            f"{o.cls.value}:{o.id}"
    for j in doc.junctions:
        node_names[j.id] = j.id
    for p in doc.pipes:
        node_names[p.id] = f"{p.line_number} ({p.id})" if p.line_number else p.id

    pages_html = []
    image_note = ""
    sizes = _page_sizes(doc)
    skipped = pages_not_analysed(report)
    skipped_warning = _skipped_warning(skipped)
    page_notes = {p: "스캔/래스터 페이지 - 분석 안 됨 (scanned / raster-only page, not analysed)"
                  for p in skipped["unreadable_pages"]}
    page_notes.update({p: "내용 없는 페이지 - 분석 안 됨 (empty page, not analysed)" for p in skipped["empty_pages"]})
    for idx, (w, h) in sizes.items():
        img = None
        if pdf_path:
            rendered = render_page_png(pdf_path, idx, max_px=max_image_px)
            if rendered:
                img, pw, ph = rendered
                if not doc.pages:
                    w, h = pw, ph
            else:
                image_note = ('<p class="warn">PDF 페이지를 렌더링하지 못했습니다 - 오버레이만 표시합니다 '
                              '(page render unavailable; overlay only).</p>')
        if len(sizes) > 1 or idx in page_notes:
            note = f' <span class="badge v-insufficient_data">{_e(page_notes[idx])}</span>' if idx in page_notes else ""
            pages_html.append(f"<h3>페이지 Page {idx + 1}{note}</h3>")
        pages_html.append(f'<div class="drawing-wrap">{_svg(doc, idx, w, h, img, labels, node_names)}</div>')
    if not pdf_path:
        image_note = '<p class="muted">PDF 미지정 - 추출 결과 오버레이만 표시 (no PDF given; overlay only).</p>'

    info = doc.info
    title = f"{info.drawing_no or ''} {info.title or Path(doc.source).name or 'P&ID'}".strip()
    rule_titles_ko = {rid: r.title_ko for rid, r in RULES.items()}
    flow_rows = []
    drawn = {o.id for o in doc.objects} | {j.id for j in doc.junctions} | {p.id for p in doc.pipes}
    pairs = [(a, b) for a, b, _ in sheet_link_pairs(doc)]
    page_of = {o.id: o.page + 1 for o in doc.objects}
    for k, fp in enumerate(report.flow_paths, 1):
        hops = flow_path_sheet_links(fp, pairs)
        flow_rows.append({"n": k, "from": node_names.get(fp.source_id, fp.source_id),
                          "to": node_names.get(fp.target_id, fp.target_id), "description": fp.description or "",
                          "nodes": len(fp.node_ids),
                          "sheet_links": "; ".join(f"{node_names.get(a, a)} (p.{page_of.get(a, '?')}) -> "
                                                   f"{node_names.get(b, b)} (p.{page_of.get(b, '?')})" for a, b in hops),
                          # only ids drawn on the overlay (older reports carried "sheet_link:a:b" pseudo pipe ids)
                          "ids": " ".join(i for i in dict.fromkeys(fp.node_ids + fp.pipe_ids) if i in drawn)})
    flow_cols = [("n", "#"), ("description", "경로 Path"), ("from", "출발 From"), ("to", "도착 To"),
                 ("nodes", "노드 Nodes")]
    if any(r["sheet_links"] for r in flow_rows):
        flow_cols.append(("sheet_links", "도면 간 연결 Sheet links"))
    summary_src = str(report.meta.get("summary_source", "template"))
    det = report.meta.get("summary_deterministic") or {}
    toggles = "".join(
        f'<label><input type="checkbox" checked data-hide="{key}"> {_e(lbl)}</label>'
        for key, lbl in [("equipment", "설비 equipment"), ("valve", "밸브 valve"), ("instrument", "계기 instrument"),
                         ("fitting", "부속 fitting"), ("connector", "연결 connector"), ("process", "배관 pipes"),
                         ("signal", "신호선 signal"), ("junction", "분기점 junctions"), ("label", "라벨 labels"),
                         ("image", "도면 이미지 page image")])

    body = [
        "<header>",
        f"<h1>P&amp;ID 분석 보고서 (analysis report) - {_e(title)}</h1>",
        f'<div class="meta">{_e(_systems_text(doc, report.meta))} | 도면 번호 {_e(info.drawing_no or "-")} | '
        f'호선 {_e(info.hull_no or "-")} | 선종 {_e(info.ship_type or "-")} | 개정 {_e(info.revision or "-")} | '
        f'출처 {_e(doc.source or "-")}</div>',
        "</header>",
        '<nav><a href="#summary">요약 Summary</a><a href="#drawing">도면 Drawing</a>'
        '<a href="#findings">검토 결과 Findings</a><a href="#equipment">설비 Equipment</a>'
        '<a href="#valves">밸브 Valves</a><a href="#instruments">계기 Instruments</a><a href="#lines">라인 Lines</a>'
        '<a href="#flows">유동 경로 Flow paths</a>' + ('<a href="#model3d">3D</a>' if (layout or viewer_href) else "") +
        '<a href="#llm">LLM</a></nav>',
        '<section id="summary"><h2>요약 (Summary)</h2>', skipped_warning, _cards(report.counts),
        f"<p>{_summary_badge(summary_src)}</p>",
        f'<h3>한국어</h3><pre class="summary">{_e(report.summary_ko or "-")}</pre>',
        f'<h3>English</h3><pre class="summary">{_e(report.summary_en or "-")}</pre>',
    ]
    if det and summary_src != "template":
        body.append(f'<details><summary>결정론적 요약 (deterministic summary)</summary>'
                    f'<pre class="summary">{_e(det.get("ko", ""))}</pre><pre class="summary">{_e(det.get("en", ""))}'
                    f"</pre></details>")
    body += [
        "</section>",
        '<section id="drawing"><h2>도면 및 추출 결과 (Drawing with extracted objects)</h2>', skipped_warning, image_note,
        f'<div class="toolbar">{toggles}<label><input type="checkbox" id="dim"> 도면 흐리게 dim</label>'
        '<button data-zoom="0.8">－</button><button data-zoom="1.25">＋</button>'
        '<button data-zoom="fit">맞춤 fit</button></div>',
        _legend(doc), *pages_html,
        '<p class="muted">객체/배관에 마우스를 올리면 정보가, 표의 행을 클릭하면 도면에서 강조 표시됩니다 '
        "(hover for details, click a table row to highlight).</p></section>",
        '<section id="findings"><h2>검토 결과 (Findings)</h2>',
        '<div class="toolbar"><label>판정 필터 Verdict filter <select id="verdict-filter">'
        '<option value="all">전체 all</option>' + "".join(
            f'<option value="{_e(v.value)}">{_e(VERDICT_LABELS[v])}</option>' for v in Verdict) + "</select></label></div>",
        _findings_table(report.findings, node_names, rule_titles_ko), "</section>",
        '<section id="equipment"><h2>설비 목록 (Equipment list)</h2>',
        _table(report.equipment_list, COLUMNS["equipment"], "id"), "</section>",
        '<section id="valves"><h2>밸브 목록 (Valve list)</h2>',
        _table(report.valve_list, COLUMNS["valves"], "id"), "</section>",
        '<section id="instruments"><h2>계기 목록 (Instrument list)</h2>',
        _table(report.instrument_list, COLUMNS["instruments"], "id"), "</section>",
        '<section id="lines"><h2>배관 라인 목록 (Line list)</h2>',
        _table(report.line_list, COLUMNS["lines"], "pipe_ids"), "</section>",
        '<section id="flows"><h2>유동 경로 (Flow paths)</h2>',
        _table(flow_rows, flow_cols, "ids"),
        ('<p class="muted">상위 경로만 표시 (truncated; total '
         f'{_e(_total_text(report.meta))})</p>' if report.meta.get("flow_paths_truncated") else ""),
        "</section>",
        _layout_section(layout, viewer_href),
        _llm_section(report),
        '<div id="tip"></div>',
    ]
    page = ("<!DOCTYPE html>\n<html lang=\"ko\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{_e(title)} - P&amp;ID 분석 보고서</title><style>{_CSS}</style></head><body>"
            + "".join(body) + f"<script>{_JS}</script></body></html>\n")
    out_html.write_text(page, encoding="utf-8")
    return out_html
