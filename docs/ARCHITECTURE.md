# shippid architecture (implementation contract)

Python **3.10** (`.venv/bin/python`, created with `uv venv --python 3.10`). Package root `src/shippid`.
Run tests with `.venv/bin/python -m pytest tests/<file>.py -q`.

Input documents are **readable PDFs** (vector paths + real text objects, as exported by CAD
tools such as AutoCAD/Plant 3D, AVEVA, Smart P&ID, Cadmatic). Scanned/raster-only PDFs are out
of scope (they are detected and reported as unreadable).

Gemma 4 31B (instruction-tuned, multimodal: text + image in, text out) is the analysis and
3D-planning model. It is served by the user (vLLM / SGLang / Ollama / llama.cpp / LM Studio via
an OpenAI-compatible endpoint, or the Gemini API). Every stage that uses Gemma must also have a
deterministic path so the pipeline runs (and is testable) with `--llm none`; outputs record which
path produced them (`source`, `planner`, `LLMUsage`). Mock output is never presented as Gemma output.

## Shared contracts (do not modify without coordination)

| module | content |
|---|---|
| `shippid/model.py` | pydantic data model: `PIDDocument` (objects, junctions, pipes, texts, info), `AnalysisReport`, `Finding`, `FlowPath`, `Layout3D`, `Placement3D`, `PipeRoute3D`, `LLMUsage`. Page space = PDF points, origin top-left, y down. 3D = metres, z up. |
| `shippid/symbols.py` | symbol library: `SymbolSpec` per `SymbolClass` (local primitives, ports, description), actuator decorations, `Placement` transform, `draw_symbol()` (PyMuPDF Shape), `placed_bbox/ports/prims`, `sample_outline()`, `legend_text()` |
| `shippid/catalog.py` | ship systems (`SYSTEMS`, codes BA/BG/FO/FS/LO/SW/CF/FM/CA/ST), `detect_system()`, ISO 14726 `MEDIUM_COLORS`, tag regexes, `parse_line_number()`, `classify_tag()`, `PIPE_OD_MM`, `pipe_od_m()`, `TYPICAL_DIMENSIONS`, `DEFAULT_DECKS` |
| `shippid/config.py` | `Config`, `LLMConfig`, `load_config()` (env `SHIPPID_LLM_*`, `shippid.yaml`) |
| `shippid/llm/base.py` | `Message`, `ImagePart`, `LLMBackend` (ABC), `LLMClient.generate_json(messages, PydanticModel, validator=...)` / `generate_text`, `MockBackend`, `schema_for()` |
| `shippid/llm/jsonutil.py` | `extract_json()`, `strip_thinking()` |

Conventions: object ids `o<n>`, junction ids `j<n>`, pipe ids `p<n>`, text ids `t<n>` (stable,
ordered by appearance). Endpoints reference `object_id`+`port` or `junction_id`. Rotation: see
`symbols.Placement` docstring. Every module gets a `tests/test_<module>.py`. No network access in
tests. Keep runtime dependencies to those in `pyproject.toml` (+ stdlib).

## Stages

### 1. `shippid.datasets` — collection & synthetic ship P&IDs
* `blueprints.py` — per-`SystemType` templates of a ship system: sources (tanks, sea chests,
  bilge wells), parallel pump trains (suction valve, strainer, pump, check valve, discharge
  valve, pressure gauges, relief valve for screw pumps), process units (filters, heaters,
  coolers, purifiers, BWTS, OWS, flow meters, three-way TCV, with optional bypass), destinations
  (tanks, overboard, engines, off-page connectors). Realistic names (e.g. `NO.1 W.B.TK (P)`,
  `NO.2 BALLAST PUMP`, `M/E`, `H.F.O. SETTLING TK`), valve actuators by system (hydraulic
  butterfly valves for ballast, quick-closing valves on FO tank outlets, SDNR valves on bilge
  suctions), instrument tags (ISA 5.1), line numbers (`250A-BA-012-1S` and yard-style variants).
* `synthetic.py` — `generate(seed, system=None, style=None) -> (PIDDocument, DrawingSpec)`:
  topology from a blueprint, column/row layout, orthogonal pipe routing, junction dots or plain
  tees, crossing hops, tags/line numbers placed without overlaps. Ground truth is exact: object
  bboxes = `placed_bbox`, ports = `placed_ports`, pipe polylines end exactly on ports/junctions,
  text bboxes measured from the font metrics.
* `pdf_writer.py` — writes a **readable vector PDF** (PyMuPDF): A1/A2/A3 landscape, frame +
  zone markers, title block (drawing no., title, hull no., ship type, revision table), notes,
  optional legend, symbols as flattened vector paths (`draw_symbol`), pipes as polylines with
  configurable widths, dashed signal lines, real text (base-14 fonts, rotated line numbers on
  vertical pipes), optional OCG layers. Style randomisation (symbol scale, widths, fonts, tag
  style) so the extractor cannot overfit one look. No hidden metadata that leaks ground truth.
* `build.py` — mass generation with multiprocessing: `build_dataset(out_dir, count, seed, workers,
  systems=None)` → `<id>.pdf`, `<id>.json` (ground-truth `PIDDocument`), `manifest.jsonl`
  (id, pdf, gt, system, ship_type, page size, counts, split train/val/test), optional PNG previews.
* `multisheet.py` — `generate_multisheet(seed, system=None, sheets=2|3, style=None) -> (PIDDocument, pdf bytes)`:
  one system topology cut between stages onto 2–3 sheets of one drawing (one style, **one paper
  size** — a later sheet that does not fit moves the whole set to the next larger paper), joined by
  paired off-page connectors in three yard styles (`sheet`: `SH.j` inside the arrow, `mark`: shared
  continuation mark `A1`, `dwg`: `CONT. ON DWG <no.> SH.j`); cut return lines get a tank-side
  connector. Ground truth: all pages, page-ordered global ids, `meta["sheet_links"]` (`a` outgoing,
  `b` incoming, `key`, `kind` header/return, `source="gt"`), `meta["sheets"]`, `meta["multisheet"]`.
* `variants.py` — robustness suite: `VARIANTS` (plain, rotated_page, qu_bowties, pipes_through,
  polyline_curves, styled_lines, fragmented_text, layers_off, scaled, yard_symbols + LEGEND,
  shx_text, combined), `make_variant(seed, variant)` (re-rendered PDF + exact ground truth, nothing
  leaked into the PDF), `build_variant_suite(out_dir, per_variant, seed)` → `<variant>/manifest.jsonl`
  in the benchmark format (paired: item *i* of every variant starts from the same drawing).
* `collect.py` — real-drawing collection: `scan_archive(root)` walks a directory tree of PDFs
  iteratively (never follows symlinks — recorded as `skipped`; only regular files; directories
  deeper than `max_depth` or unlistable → `error`, the scan continues), checks readability (text
  chars, vector path count, image coverage), SHA-256 dedupe, extracts title-block keywords → system
  guess, writes `index.jsonl`; `fetch_sources(sources_yaml, dest)` downloads listed URLs (httpx;
  resumable, size limit, content-type check; redirects followed by the module, ≤ 5 hops, http/https,
  every hop's host resolved and refused unless public — `allow_private` for intranet sources; total
  deadline `max_seconds` + minimum rate; the sidecar records `final_url`).

### 2. `shippid.pdf` + `shippid.extract` — deterministic vector extraction
* `pdf/reader.py` — `read_page(page) -> PageContent`: segments (from `get_drawings()` items `l`,
  `re`, `qu`, `c` with style: width, color, dashes, fill, layer), filled regions, circles
  recognised from 4-Bezier loops, text spans/words with bbox, size, font and rotation
  (`dir`), OCG layer names, page size. Robust to arbitrary scale, colours and CTM.
* `extract/` — `extract_pdf(path, config) -> PIDDocument` (`source="vector"`):
  1. frame / title block / notes / legend region detection (excluded from symbol search; title
     block text kept as `TextRole.TITLE_BLOCK`);
  2. pipe-line vs symbol-stroke separation (long axis-aligned segments, stroke width, chaining);
  3. symbol candidate grouping (connected strokes/filled shapes, proximity) and classification
     by template matching against `symbols.py` (normalised chamfer distance over rotations
     0/90/180/270 and mirror, size priors) + actuator detection; per-candidate top-k scores
     stored in `attributes["candidates"]`, `confidence` = best score; unmatched → `UNKNOWN`;
  4. ports located on the symbol outline where pipes touch;
  5. pipe network: merge collinear pieces, split at symbols, tees/crosses (dot = connection,
     hop/gap = crossing without connection), `PipeRun` with `Endpoint`s, dashed → `SIGNAL`;
  6. text assembly + role classification (`catalog.classify_tag`, inside-bubble, near-connector,
     title block) and association to the nearest owner (objects for tags, pipes for line numbers),
     `size`/`medium` on pipes from parsed line numbers; per-character / per-word fragments are merged
     into lines and hidden render-mode-3 / SHX-annotation duplicates resolved.
* `extract/legend.py` — legend learning: LEGEND / SYMBOL LEGEND / 범례 blocks of the document (on
  the sheet or a separate legend sheet) are read before matching; every row's glyph + description
  (`CHECK VALVE`, `S.D.N.R`, `체크 밸브` …, bilingual keyword map → `SymbolClass` + `ActuatorType`)
  becomes a document-local template matched alongside the built-in library (matches record
  `attributes["template"] = "legend:<description>"`; unmapped rows → `meta["legend"]["unmapped"]`).

### 3. `shippid.llm` + `shippid.analysis.gemma` — Gemma 4 31B
* `llm/openai_compat.py` — OpenAI-compatible `/chat/completions` backend (vLLM, SGLang, Ollama
  `/v1`, LM Studio, llama.cpp): text + `image_url` data-URL parts, `response_format` json_schema →
  json_object → prompt fallback on HTTP 400, reasoning content separation, Gemma thinking toggle
  via `chat_template_kwargs`, retries with backoff, timeouts.
* `llm/gemini.py` — Gemini API backend (`generateContent`, `inline_data` images, JSON schema
  response config) for Gemma served by Google.
* `llm/factory.py` — `make_client(LLMConfig) -> Optional[LLMClient]`.
* `llm/render.py` — PyMuPDF rendering helpers: page / clip → PNG bytes downscaled to
  `max_image_side`, numbered contact sheets of symbol crops, overlapping tiles.
* `analysis/prompts.py`, `analysis/gemma.py` — `GemmaAnalyzer(client)`:
  `refine_symbols(doc, pdf)` (low-confidence/unknown candidates → contact sheet + vector
  signature + legend → class/actuator/tag), `read_drawing_info(doc, pdf)` (title block → `DrawingInfo`),
  `resolve_texts(doc)` (ambiguous tag/line-number ownership), `review(report)` (engineering review
  → `Finding`s with rule references + Korean/English summaries). Outputs validated against ids that
  exist in the document; merged with `source="gemma"`/`"merged"`.

### 4. `shippid.analysis` (deterministic) + `shippid.report` + `shippid.evaluate`
* `analysis/graph.py` — networkx graph (objects + junctions as nodes, pipes as edges), junction
  collapsing, sources/sinks, flow paths, isolation valves per equipment, components.
* `analysis/rules.py` — deterministic marine checks with references (SOLAS/class), each producing
  `Finding(rule_id, severity, reference, object_ids)`.
* `analysis/lists.py` — equipment / valve / instrument / line lists.
* `analysis/sheets.py` — multi-sheet sets: `link_sheets(doc)` pairs off-page connectors across (and
  within) pages from normalised continuation keys (shared marks, `SH.n` / `SHEET n` against each
  page's title-block sheet number, drawing-number references, `TO`/`FROM` names against the
  equipment beside the partner, directions, equal line numbers); a pair is accepted one-to-one only
  when both sides support it and it beats every competitor by `margin`. Stores `doc.meta`
  `sheet_links` (`a`, `b`, `key`, `confidence`, `source` rule/gemma/manual, `directed`, `evidence`),
  `sheet_links_ambiguous` (unresolved connectors + scored candidates — the Gemma step
  `GemmaAnalyzer.resolve_sheet_links` chooses among them only, `apply_link_decisions` validates) and
  `sheet_link_stats`. `graph.build_graph` adds one zero-length edge `kind="sheet_link"` (key
  `sheet_link:<a>:<b>`) per link, so flow paths, isolation, adjacency, components and rules cross the
  sheets; `FlowPath.pipe_ids` hold document pipe ids only, the cross-sheet hops come from
  `graph.flow_path_sheet_links` (`flow_paths.csv` column `sheet_links`, report column).
* `analysis/analyzer.py` — `analyze(doc, reviewer=None) -> AnalysisReport` (links a multi-page
  document first when it has no `meta["sheet_links"]`; `report.meta["sheet_links"]` = number of
  linked pairs, `["sheet_links_ambiguous"]` = number of unresolved connectors).
* `report/html.py` — self-contained HTML report (page render + SVG overlay with hover, lists,
  findings, summaries, LLM usage, link to the 3D viewer); `report/export.py` — CSV/JSON.
* `evaluate.py` — prediction vs ground truth: per-class P/R/F1 (IoU ≥ 0.5), tag accuracy,
  connectivity (object-adjacency, including the ground-truth sheet links) P/R/F1, text-role
  accuracy; dataset aggregation → JSON + Markdown.

### 5. `shippid.model3d` — 3D generation and visualisation
* `heuristic.py` — deterministic planner: drawing plane → deck plan, deck assignment by class,
  sizes from `TYPICAL_DIMENSIONS` (scaled by pipe size), overlap resolution.
* `gemma_planner.py` — Gemma plans the arrangement (JSON → `Layout3D` placements) from the
  object/connection list, system, decks and size priors; geometric validator (overlaps, bounds,
  deck consistency, nozzle reachability) feeds errors back for repair rounds; heuristic fallback
  for anything Gemma leaves out.
* `router.py` — orthogonal 3D pipe routing between nozzles (elevation bands, obstacle avoidance).
* `meshes.py` / `scene.py` — parametric meshes per class (trimesh), ISO 14726 colours, node names =
  object ids with metadata; export GLB/OBJ.
* `viewer.py` — standalone HTML (three.js via CDN importmap, embedded GLB, click-to-inspect,
  per-system/category toggles) and an offline plotly fallback.

### 6. `shippid.cli` / `shippid.pipeline` / `shippid.benchmark`
`shippid collect synthetic|scan|fetch`, `shippid extract`, `shippid analyze`, `shippid model3d`,
`shippid run` (all stages), `shippid evaluate`, `shippid llm-check` (connectivity test to the
Gemma endpoint).

* `pipeline.run_pipeline` validates `planner` / `viewer` (and `planner="gemma"` against the LLM
  client) before any stage runs; a multi-page document is linked (`pipeline.link_multisheet`, after
  the Gemma refinement when it runs) before `extracted.json` / `document.json` are saved, so both
  carry the same `sheet_links` as the document inside `analysis.json`. The `GemmaAnalyzer` is wired
  from `LLMConfig` (`max_image_side`, `concurrency`); `RunResult.llm_backend` = `backend:model`.
* `benchmark.run_benchmark(data_dir, out_dir, cfg=..., use_llm=..., client=None)` extracts with the
  given `Config` (also inside the worker processes), links multi-page predictions like the
  pipeline, evaluates every document once (predictions parsed one at a time while scored) and
  records `benchmark.llm` as `backend:model` (`mock:...` for a mock, never "gemma"); saved
  predictions are named from the sanitised manifest id, always inside `predictions/`.
* CLI exit codes: 0 ok; 1 the command reported a failure (`llm-check` not ok, failed dataset
  items); 2 invalid arguments / input (missing, empty, damaged or encrypted PDF, page out of range,
  wrong kind of JSON, `--planner gemma` / `--with-llm` without an LLM backend) printed as one
  `shippid: error: ...` line (`-v` adds the traceback); 130 interrupted.
