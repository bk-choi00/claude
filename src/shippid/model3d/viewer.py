"""Standalone HTML viewers for a :class:`Layout3D`.

* ``mode="threejs"`` -- one HTML file: three.js r180 from jsDelivr through an importmap, the GLB
  embedded as base64 and parsed with ``GLTFLoader.parseAsync``, OrbitControls, camera fitted to
  the model bounds, click-to-inspect (raycast; hits under hidden ancestors and structure are
  skipped) with the node metadata in a side panel (Korean labels), selection highlight, per
  system / category visibility checkboxes, deck and outline toggles, ISO 14726 colour legend and
  search by tag / id / label / line number (the human label is the ``label`` extra:
  GLTFLoader puts the glTF node name into ``userData.name``). Needs network access to the CDN for the three.js modules only.
* ``mode="plotly"`` -- fully offline HTML (plotly.js embedded): one Mesh3d per object / pipe
  with hover text, legend groups per system, ``aspectmode="data"``. Vertices are sent as float32,
  faces as int32, one ``color`` per single-colour item and otherwise a per-face part index
  (``intensity`` in cell mode over a discrete colour scale) instead of a per-face colour string.
  Every string from the drawing (title, drawing number, tags, labels...) is HTML-escaped: plotly
  renders pseudo-HTML (``<a href>``, ``<span style>``) in titles and hover texts.
"""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..catalog import MEDIUM_COLORS, SYSTEMS
from ..model import Layout3D, Medium, PIDDocument, SystemType
from .scene import scene_items

THREE_VERSION = "0.180.0"
THREE_CDN = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}"

MEDIUM_KO: dict[str, str] = {
    "sea_water": "해수", "fresh_water": "청수", "ballast_water": "평형수", "bilge": "빌지", "fuel_oil": "연료유",
    "lube_oil": "윤활유", "compressed_air": "압축공기", "steam": "증기", "fire_water": "소화수", "unknown": "미상",
}
CATEGORY_KO: dict[str, str] = {
    "equipment": "장비", "valve": "밸브", "instrument": "계기", "fitting": "피팅", "connector": "도면 연결",
    "pipe": "배관",
}
FIELD_KO: dict[str, str] = {
    "id": "ID", "tag": "태그", "label": "명칭", "class": "심볼 분류", "category": "범주", "system": "계통",
    "line_number": "라인 번호", "lines": "연결 라인", "size": "관경", "medium": "유체", "iso14726": "ISO 14726 색상",
    "deck": "데크", "dimensions_m": "치수 L×W×H (m)", "position_m": "위치 x,y,z (m)", "yaw_deg": "방향 (°)",
    "actuator": "구동기", "diameter_m": "외경 (m)", "length_m": "길이 (m)", "from": "시작", "to": "끝",
    "pipe_kind": "배관 종류", "source": "배치 출처", "rationale": "배치 근거",
}


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def _legend(layout: Layout3D) -> list[dict[str, str]]:
    media = sorted({r.medium for r in layout.routes}, key=lambda m: m.value)
    out = []
    for m in media:
        name, code, rgb = MEDIUM_COLORS.get(m, MEDIUM_COLORS[Medium.UNKNOWN])
        out.append({"medium": m.value, "ko": MEDIUM_KO.get(m.value, m.value), "colour": name, "code": code,
                    "hex": _hex(rgb)})
    return out


def _system_labels() -> dict[str, str]:
    out = {s.value: f"{info.name_ko} ({info.code})" for s, info in SYSTEMS.items()}
    out[SystemType.UNKNOWN.value] = "미상 계통"
    out["structure"] = "구조 (데크)"
    return out


def placement_sources(layout: Layout3D) -> dict[str, int]:
    """Placements per origin tag (``[gemma]`` / ``[mock]`` / ``[heuristic]`` / ``[router]`` rationale prefix)."""
    out: dict[str, int] = {}
    for p in layout.placements:
        m = re.match(r"^\[(\w+)\]", p.rationale or "")
        key = m.group(1) if m else "untagged"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def viewer_data(layout: Layout3D, doc: PIDDocument) -> dict[str, Any]:
    """JSON payload embedded in the viewers (drawing info, planner, legend, labels, notes)."""
    info = doc.info
    return {
        "title": info.title or doc.source or "P&ID 3D",
        "drawing": {"drawing_no": info.drawing_no, "title": info.title, "system": info.system.value,
                    "ship_type": info.ship_type, "hull_no": info.hull_no, "source": doc.source},
        "planner": layout.planner,
        "sources": placement_sources(layout),
        "llm": {"backend": layout.llm.backend, "model": layout.llm.model, "calls": layout.llm.calls,
                "errors": layout.llm.errors[:10]},
        "counts": {"placements": len(layout.placements), "routes": len(layout.routes)},
        "legend": _legend(layout),
        "systems": _system_labels(),
        "categories": CATEGORY_KO,
        "fields": FIELD_KO,
        "media": MEDIUM_KO,
        "notes": layout.notes[:60],
    }


_SCRIPT_ESCAPES = {"<": "\\u003c", ">": "\\u003e", "&": "\\u0026", " ": "\\u2028", " ": "\\u2029"}
_SCRIPT_UNSAFE = re.compile("[<>&  ]")


def _json_for_script(data: Any) -> str:
    """JSON for a ``<script type="application/json">`` element.

    Model / PDF strings (titles, notes) may contain ``</script>`` or ``<!--<script>``, which would
    end the element early or switch the HTML parser into the double-escaped script state and
    swallow the rest of the page; ``< > &`` and U+2028/2029 are therefore written as ``\\uXXXX``
    escapes (valid JSON, identical value after ``JSON.parse``).
    """
    return _SCRIPT_UNSAFE.sub(lambda m: _SCRIPT_ESCAPES[m.group(0)], json.dumps(data, ensure_ascii=False))


def write_viewer_html(glb_path: Optional[str | Path], layout: Layout3D, doc: PIDDocument, out_html: str | Path,
                      mode: str = "threejs") -> Path:
    """Write a standalone viewer; see the module docstring for the two modes."""
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    if mode == "plotly":
        return _write_plotly(layout, doc, out)
    if mode != "threejs":
        raise ValueError(f"unknown viewer mode {mode!r} (threejs | plotly)")
    if glb_path is None:
        raise ValueError("threejs viewer needs the GLB file")
    b64 = base64.b64encode(Path(glb_path).read_bytes()).decode("ascii")
    data = viewer_data(layout, doc)
    values = {"TITLE": html.escape(f"{data['title']} – 3D"), "THREE_CDN": THREE_CDN,
              "DATA": _json_for_script(data), "GLB": b64}
    page = re.sub(r"__(TITLE|THREE_CDN|DATA|GLB)__", lambda m: values[m.group(1)], _THREE_TEMPLATE)
    out.write_text(page, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- plotly


def _write_plotly(layout: Layout3D, doc: PIDDocument, out: Path) -> Path:
    import plotly.graph_objects as go

    data = viewer_data(layout, doc)
    sys_labels = data["systems"]
    fig = go.Figure()
    shown: set[str] = set()
    for it in scene_items(layout, doc, outline=False):
        verts, faces, colors = [], [], []
        base = 0
        palette: dict[str, int] = {}
        for part in it.parts:
            v = np.asarray(part.mesh.vertices, dtype=float)
            v = (np.c_[v, np.ones(len(v))] @ it.transform.T)[:, :3]
            f = np.asarray(part.mesh.faces) + base
            base += len(v)
            verts.append(v)
            faces.append(f)
            colors.append(np.full(len(f), palette.setdefault(_hex(part.color), len(palette)), dtype=np.uint8))
        if not verts:
            continue
        # float32 vertices; int64 faces, which plotly downcasts to the smallest integer type (int8 / int16 / int32)
        V, F = np.vstack(verts).astype(np.float32), np.vstack(faces).astype(np.int64)
        hexes = list(palette)
        if len(hexes) == 1:
            paint: dict[str, Any] = {"color": hexes[0]}
        else:  # part index per face over a discrete colour scale (exact colour at each stop)
            k = len(hexes) - 1
            paint = {"intensity": np.concatenate(colors), "intensitymode": "cell", "cmin": 0, "cmax": k,
                     "colorscale": [[i / k, h] for i, h in enumerate(hexes)], "showscale": False}
        ex = it.extras
        if it.kind == "deck":
            group, label = "structure", sys_labels["structure"]
            text = html.escape(str(ex.get("label", it.node)))
            opacity = 0.15
        else:
            group, label = it.system, sys_labels.get(it.system, it.system)
            rows = [f"<b>{html.escape(str(ex.get('tag') or ex.get('id')))}</b>"]
            for k in ("label", "class", "category", "line_number", "size", "medium", "deck", "rationale"):
                if ex.get(k) not in (None, "", []):
                    rows.append(f"{FIELD_KO.get(k, k)}: {html.escape(str(ex[k])[:120])}")
            text = "<br>".join(rows)
            opacity = 1.0
        fig.add_trace(go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2], **paint,
            opacity=opacity, flatshading=True, name=html.escape(str(label)), legendgroup=group,
            showlegend=group not in shown, hovertext=text, hoverinfo="text",
            lighting={"ambient": 0.55, "diffuse": 0.8, "specular": 0.2},
        ))
        shown.add(group)
    drawing = data["drawing"]
    # drawing strings come from the PDF: plotly would render <a href> / <span style> in the title
    title = " ".join(html.escape(str(x)) for x in (drawing.get("title"), drawing.get("drawing_no")) if x)
    title = f"{title} – planner: {html.escape(str(layout.planner))}".strip(" –")
    fig.update_layout(
        title=title, legend={"groupclick": "togglegroup", "title": {"text": "계통"}},
        scene={"aspectmode": "data", "xaxis": {"title": {"text": "x 선수방향 (m)"}},
               "yaxis": {"title": {"text": "y 좌현 (m)"}}, "zaxis": {"title": {"text": "z 높이 (m)"}}},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    fig.write_html(str(out), include_plotlyjs=True, full_html=True)
    return out


# --------------------------------------------------------------------------- three.js template

_THREE_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#eef1f4; --panel:#ffffffee; --fg:#1d2329; --muted:#5d6872; --line:#d3d9df; --accent:#d9480f; --sel:#fff4e6; }
@media (prefers-color-scheme: dark) { :root { --bg:#14181c; --panel:#1f252bee; --fg:#e6e9ec; --muted:#9aa5ae; --line:#36404a; --accent:#ff922b; --sel:#3a2a18; } }
* { box-sizing:border-box; }
html, body { margin:0; height:100%; background:var(--bg); color:var(--fg); font:13px/1.4 system-ui, -apple-system, "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif; overflow:hidden; }
#view { position:fixed; inset:0; }
.panel { position:fixed; top:10px; max-height:calc(100% - 20px); overflow:auto; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; box-shadow:0 2px 10px #0002; }
#left { left:10px; width:290px; }
#right { right:10px; width:330px; display:none; }
h1 { font-size:15px; margin:0 0 4px; } h2 { font-size:12px; margin:12px 0 4px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.muted { color:var(--muted); } .badge { display:inline-block; padding:1px 6px; border-radius:9px; border:1px solid var(--line); font-size:11px; }
input[type=search] { width:100%; padding:5px 7px; border:1px solid var(--line); border-radius:6px; background:transparent; color:var(--fg); }
label { display:block; cursor:pointer; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.cats { margin-left:18px; }
.legend div { display:flex; align-items:center; gap:6px; margin:2px 0; }
.sw { width:22px; height:10px; border-radius:2px; border:1px solid #0003; flex:none; }
table { border-collapse:collapse; width:100%; } td { border-top:1px solid var(--line); padding:3px 4px; vertical-align:top; word-break:break-word; } td:first-child { color:var(--muted); width:38%; }
#matches div { cursor:pointer; padding:2px 4px; border-radius:4px; } #matches div:hover { background:var(--sel); }
details summary { cursor:pointer; color:var(--muted); } #notes { font-size:11px; padding-left:16px; }
#status { position:fixed; bottom:10px; left:50%; transform:translateX(-50%); } button { font:inherit; }
@media (max-width: 700px) { #left { width:calc(100% - 20px); max-height:40%; } #right { top:auto; bottom:10px; width:calc(100% - 20px); max-height:45%; } }
</style>
<script type="importmap">
{ "imports": { "three": "__THREE_CDN__/build/three.module.js", "three/addons/": "__THREE_CDN__/examples/jsm/" } }
</script>
</head>
<body>
<div id="view"></div>
<div id="left" class="panel">
  <h1 id="title"></h1>
  <div id="meta" class="muted"></div>
  <h2>검색 (태그 / ID / 명칭 / 라인)</h2>
  <input id="search" type="search" placeholder="예: BA-P001, o12, PUMP" autocomplete="off">
  <div id="matches"></div>
  <h2>계통 / 범주 표시</h2>
  <div id="groups"></div>
  <h2>구조</h2>
  <label><input type="checkbox" id="decks" checked> 데크 표시</label>
  <label><input type="checkbox" id="outline" checked> 기관실 외곽 표시</label>
  <h2>배관 색상 (ISO 14726)</h2>
  <div id="legend" class="legend"></div>
  <details><summary>노트 (배치 / 경로 계산)</summary><ul id="notes"></ul></details>
</div>
<div id="right" class="panel">
  <div style="display:flex;justify-content:space-between;align-items:center"><h1 id="sel-title"></h1><button id="close" title="닫기">✕</button></div>
  <table id="props"></table>
</div>
<div id="status" class="panel muted">3D 모델 로딩 중…</div>
<script id="viewer-data" type="application/json">__DATA__</script>
<script id="glb-data" type="application/octet-stream">__GLB__</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const DATA = JSON.parse(document.getElementById('viewer-data').textContent);
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function b64ToBuffer(b64) {
  const bin = atob(b64.trim());
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

// ---- header / legend / notes
$('title').textContent = DATA.title;
const d = DATA.drawing;
$('meta').innerHTML = [d.drawing_no, d.ship_type, d.hull_no].filter(Boolean).map(esc).join(' · ') +
  `<br>배치: <span class="badge">${esc(DATA.planner)}</span>` +
  (DATA.sources ? ` <span class="muted">${Object.entries(DATA.sources).map(([k, n]) => esc(k) + ' ' + n).join(', ')}</span>` : '') +
  (DATA.llm && DATA.llm.calls ? ` <span class="muted">(${esc(DATA.llm.backend)} ${esc(DATA.llm.model || '')}, ${DATA.llm.calls} calls)</span>` : '');
$('legend').innerHTML = DATA.legend.map((m) => `<div><span class="sw" style="background:${m.hex}"></span><b style="white-space:nowrap">${esc(m.ko)}</b> <span class="muted">${esc(m.colour)} – ${esc(m.code)}</span></div>`).join('') || '<span class="muted">배관 없음</span>';
$('notes').innerHTML = DATA.notes.map((n) => `<li>${esc(n)}</li>`).join('');

// ---- renderer / scene
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(view.clientWidth, view.clientHeight);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
scene.background = new THREE.Color(dark ? 0x14181c : 0xeef1f4);
const camera = new THREE.PerspectiveCamera(45, view.clientWidth / view.clientHeight, 0.05, 5000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x556677, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(1, 2, 1.5); scene.add(sun);
const fill = new THREE.DirectionalLight(0xffffff, 0.5); fill.position.set(-1.5, 1, -1); scene.add(fill);

let root = null;
const owners = [];          // object / pipe nodes
const groupNodes = [];      // system / category group nodes
const structure = { deck: [], outline: [] };

function fit(target, factor = 1.25) {
  const box = target.isBox3 ? target : new THREE.Box3().setFromObject(target);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.length() / 2, 0.3);
  const dist = factor * radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2));
  const dir = new THREE.Vector3(-0.9, 0.75, 1.0).normalize();
  camera.position.copy(center).addScaledVector(dir, dist);
  camera.near = Math.max(dist / 1000, 0.01); camera.far = dist * 20; camera.updateProjectionMatrix();
  controls.target.copy(center); controls.update();
}

function visibleChain(o) { for (; o; o = o.parent) if (!o.visible) return false; return true; }
function ownerOf(o) { for (; o; o = o.parent) { const k = o.userData && o.userData.kind; if (k === 'object' || k === 'pipe') return o; if (k === 'structure') return null; } return null; }

// ---- selection
let selected = null;
function setHighlight(node, on) {
  node.traverse((m) => {
    if (!m.isMesh) return;
    const mat = m.material;
    if (on) { if (m.userData._em === undefined) m.userData._em = mat.emissive.getHex(); mat.emissive.setHex(0xff7a00); mat.emissiveIntensity = 0.55; }
    else if (m.userData._em !== undefined) { mat.emissive.setHex(m.userData._em); mat.emissiveIntensity = 1.0; }
  });
}
function fmt(v) {
  if (Array.isArray(v)) return v.map((x) => typeof x === 'number' ? (+x.toFixed(3)) : x).join(', ');
  if (typeof v === 'number') return String(+v.toFixed(3));
  return String(v);
}
function label(key, value) {
  if (key === 'category') return DATA.categories[value] ? `${DATA.categories[value]} (${value})` : value;
  if (key === 'system') return DATA.systems[value] ? `${DATA.systems[value]}` : value;
  if (key === 'medium') return DATA.media[value] ? `${DATA.media[value]} (${value})` : value;
  return fmt(value);
}
function select(node, frame = false) {
  if (selected) setHighlight(selected, false);
  selected = node;
  if (!node) { $('right').style.display = 'none'; return; }
  setHighlight(node, true);
  const u = node.userData;
  $('sel-title').textContent = u.tag || u.label || u.id;
  const rows = [];
  for (const [k, ko] of Object.entries(DATA.fields)) {
    if (u[k] === undefined || u[k] === null || u[k] === '' || (Array.isArray(u[k]) && !u[k].length)) continue;
    rows.push(`<tr><td>${esc(ko)}</td><td>${esc(label(k, u[k]))}</td></tr>`);
  }
  $('props').innerHTML = rows.join('');
  $('right').style.display = 'block';
  if (frame) fit(node, 2.2);
}
$('close').onclick = () => select(null);

// ---- picking (click without drag)
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let down = null;
renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!down || !root || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
  const r = renderer.domElement.getBoundingClientRect();
  pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  for (const hit of raycaster.intersectObject(root, true)) {
    if (!visibleChain(hit.object)) continue;   // Raycaster ignores .visible
    const owner = ownerOf(hit.object);
    if (!owner) continue;                      // decks / outline are not pickable
    select(owner); return;
  }
  select(null);
});

// ---- search
function runSearch(q, frame) {
  q = q.trim().toLowerCase();
  if (!q) { $('matches').innerHTML = ''; return; }
  const hits = owners.filter((o) => ['tag', 'id', 'label', 'line_number'].some((k) => o.userData[k] && String(o.userData[k]).toLowerCase().includes(q)));
  $('matches').innerHTML = hits.slice(0, 25).map((o, i) => `<div data-i="${i}">${esc(o.userData.tag || o.userData.id)} <span class="muted">${esc(o.userData.label || o.userData.class || '')}</span></div>`).join('') || '<span class="muted">결과 없음</span>';
  $('matches').querySelectorAll('div[data-i]').forEach((el) => { el.onclick = () => select(hits[+el.dataset.i], true); });
  if (frame && hits.length) select(hits[0], true);
}
$('search').addEventListener('input', (e) => runSearch(e.target.value, false));
$('search').addEventListener('keydown', (e) => { if (e.key === 'Enter') runSearch(e.target.value, true); });

// ---- load
try {
  const gltf = await new GLTFLoader().parseAsync(b64ToBuffer(document.getElementById('glb-data').textContent), '');
  root = gltf.scene;
  scene.add(root);
  root.traverse((o) => {
    if (o.isMesh) { o.material = o.material.clone(); }   // per-object highlight
    const u = o.userData || {};
    if (u.kind === 'object' || u.kind === 'pipe') owners.push(o);
    if (u.kind === 'group' && (u.group_type === 'system' || u.group_type === 'category')) groupNodes.push(o);
    if (u.kind === 'structure') (structure[u.category] || (structure[u.category] = [])).push(o);
  });
  // checkbox tree: systems with their categories
  const box = $('groups');
  for (const g of groupNodes.filter((n) => n.userData.group_type === 'system')) {
    const wrap = document.createElement('div');
    const n = owners.filter((o) => { let p = o.parent; while (p && p !== g) p = p.parent; return p === g; }).length;
    wrap.innerHTML = `<label><input type="checkbox" checked> <b>${esc(DATA.systems[g.userData.system] || g.userData.label || g.userData.system)}</b> <span class="muted">(${n})</span></label><div class="cats"></div>`;
    wrap.querySelector('input').onchange = (e) => { g.visible = e.target.checked; };
    const cats = wrap.querySelector('.cats');
    for (const c of g.children.filter((x) => x.userData.group_type === 'category')) {
      const lab = document.createElement('label');
      lab.innerHTML = `<input type="checkbox" checked> ${esc(DATA.categories[c.userData.category] || c.userData.category)} <span class="muted">(${c.children.length})</span>`;
      lab.querySelector('input').onchange = (e) => { c.visible = e.target.checked; };
      cats.appendChild(lab);
    }
    box.appendChild(wrap);
  }
  $('decks').onchange = (e) => structure.deck.forEach((o) => { o.visible = e.target.checked; });
  $('outline').onchange = (e) => structure.outline.forEach((o) => { o.visible = e.target.checked; });
  const content = new THREE.Box3();
  owners.forEach((o) => content.expandByObject(o));
  fit(content.isEmpty() ? root : content, 0.9);   // frame the equipment and pipes, not the whole room
  $('status').style.display = 'none';
} catch (err) {
  $('status').textContent = '3D 모델을 불러오지 못했습니다: ' + err;
  console.error(err);
}

window.addEventListener('resize', () => {
  camera.aspect = view.clientWidth / view.clientHeight; camera.updateProjectionMatrix();
  renderer.setSize(view.clientWidth, view.clientHeight);
});
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
window.__shippid = { scene, camera, controls, select, runSearch, owners };
</script>
</body>
</html>
"""
