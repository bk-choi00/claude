# Research notes (2026-10) that drive the design

Collected by web research during development. Items marked *(unverified)* came from search
snippets because the primary host was unreachable from the build sandbox.

## Gemma 4 31B

* Hugging Face ids: `google/gemma-4-31B` (base), **`google/gemma-4-31B-it`** (instruction tuned).
  Released 2026-04-02, Apache 2.0. Dense ~30.7B params, 256K context, input text + image (+ video
  frames), output text. Gemini API name: **`gemma-4-31b-it`**.
  ([HF blog](https://raw.githubusercontent.com/huggingface/blog/main/gemma4.md),
  [vLLM recipe](https://raw.githubusercontent.com/vllm-project/recipes/main/models/Google/gemma-4-31B-it.yaml))
* Images: aspect-ratio preserving, token budget `max_soft_tokens` ∈ {70,140,280,560,1120}
  (default 280). 1120 tokens ≈ 2.6 MP, e.g. 1680×1536 px (multiples of 48). Put image parts
  **before** the text. A full A1 sheet in one image is too coarse for tag text → tile/crop.
  ([transformers image processor](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/image_processing_gemma4.py))
* Chat template has a native system role. Thinking is off by default; enable per request with
  `chat_template_kwargs: {"enable_thinking": true}` (vLLM/SGLang/llama.cpp). Thoughts come back in
  `message.reasoning` (vLLM) or `reasoning_content` (SGLang/llama.cpp); stray `<|channel>…<channel|>`
  / `<turn|>` tokens may leak into content and are stripped defensively.
* Structured output: server-side guided decoding with
  `response_format={"type":"json_schema","json_schema":{"name":…,"schema":…}}`. **The model does not
  see the schema** — field semantics must also be described in the prompt (done by `LLMClient`).
* Recommended sampling (model card): temperature 1.0, top_p 0.95, top_k 64 *(unverified)*. shippid
  uses a lower temperature for schema-constrained extraction (configurable).
* Serving (Python ≥ 3.11 for current vLLM/SGLang → run as a separate service; shippid stays 3.10):
  ```
  vllm serve google/gemma-4-31B-it --max-model-len 32768 \
    --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice \
    --limit-mm-per-prompt '{"image": 8, "audio": 0}' --mm-processor-kwargs '{"max_soft_tokens": 1120}'
  ```
  Memory: BF16 ≈ 70–75 GB (one 80 GB GPU), FP8 ≈ 38 GB (`RedHatAI/gemma-4-31B-it-FP8-dynamic`),
  INT4 ≈ 20 GB (`google/gemma-4-31B-it-qat-w4a16-ct`). Ollama: `gemma4:31b` (Q4_K_M, data-URL images
  only, fixed ~280-token image budget *(unverified)*). llama.cpp: `ggml-org/gemma-4-31B-it-GGUF`.

## Readable-PDF extraction

* PyMuPDF 2.0 requires Python ≥ 3.11 → pin `pymupdf<2.0` (1.28.2 on 3.10). Use `import pymupdf`.
* `get_drawings()` items `l`, `c` (Bezier), `re`, `qu`. **Gotcha:** a closed self-intersecting
  4-segment polyline (bow-tie valve) is returned as one `qu` quad with reordered corners — rebuild
  the polygon (ul, ur, lr, ll) instead of using `quad.rect`. Circles arrive as 4 Bezier items.
* Layers (OCG): content on layers that are OFF is silently dropped from `get_drawings/get_text`;
  turn all OCGs on and reopen (`pymupdf.open("pdf", doc.tobytes())`) before extracting.
* Rotated text: `line["dir"]`; angle = `atan2(-dir[1], dir[0])`. AutoCAD SHX text is geometry unless
  exported with PDFSHX=1 (annotations) or 2 (hidden text, render mode 3).
* Evidence from published work: deterministic line tracing for connectivity (F1 ≈ 0.96–0.98) beats
  letting a VLM read pipes (F1 ≈ 0.2); LLMs do best on structured graph/text digests plus crops
  ([pid-parser](https://github.com/dzakybd/pid-parser), [ChatP&ID](https://arxiv.org/abs/2603.22528),
  [PID2Graph](https://arxiv.org/abs/2411.13929)). Shipbuilding prior art: Kong et al. IMDC 2024,
  Han/Roh/Kong IJNAOE 2025 (DXF/image P&ID → graph).

## Marine domain

* ISO 14726 main colours: RD fire fighting, SR steam, BK waste (bilge), **OG lube/hydraulic oil**,
  BN fuel, GN sea water (ballast GN-VT-GN), BU fresh water, **GY compressed air**, MN air/sounding.
* Deterministic checks with references (verdict pass/fail/n.a./insufficient data):
  bilge branch non-return (SOLAS II-1/35-1), ≥2 bilge pumps, bilge/ballast separation, emergency
  bilge suction from SW pump (II-1/35-1.3.7, UI SC251), quick-closing valves on FO tank outlets
  (II-2/4.2.2.3.4), ≥2 FO service tanks (II-1/26.11, UI SC123), relief valves on positive-displacement
  pumps (class), ≥2 fire pumps + discharge NRV + isolation (II-2/10.2), OWS 15 ppm 3-way valve
  (MARPOL I/14.7), no sludge→bilge link (MARPOL I/12), overboard NRV + shell valve (LL Reg. 22,
  UI LL11), ≥2 sea inlets for SW cooling (class), ≥2 starting-air compressors/receivers.
* No public tag conventions exist for Korean yards → tag/line regexes are configurable.

## LLM-planned 3D

* Do not ask the LLM for raw coordinates of everything (LayoutGPT-style boxes collide/out-of-bounds).
  Use a fixed vocabulary plan (zone, deck, side, anchor, relation, orientation, size class) that a
  deterministic solver converts to coordinates, with validator feedback rounds and an algorithmic
  fallback (Holodeck, LayoutVLM, Gumin et al. 2025).
* Never let the LLM generate pipe geometry: route deterministically (orthogonal A* on a voxel grid,
  bend penalty, obstacle inflation, largest DN first).
* trimesh 5.x GLB: unique three.js-safe node names, one PBR material per mesh
  (`srgb_to_linear` colours), metadata → glTF extras. three.js r180 via importmap
  (`https://cdn.jsdelivr.net/npm/three@0.180.0/...`); Raycaster ignores `visible` → filter hits.
  Plotly Mesh3d with `include_plotlyjs=True` gives a fully offline viewer.
