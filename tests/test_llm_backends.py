"""Gemma 4 access layer: OpenAI-compatible + Gemini backends, factory, rendering helpers.

HTTP servers are emulated with ``httpx.MockTransport`` using the request /
response shapes of vLLM / SGLang / llama.cpp / Ollama ``/v1`` and the Gemini API.
"""

import base64
import json

import httpx
import pymupdf
import pytest
from pydantic import BaseModel

from shippid.config import LLMConfig
from shippid.llm.base import ImagePart, LLMClient, LLMError, Message, MockBackend, schema_for
from shippid.llm.factory import check_connection, make_client, probe_image
from shippid.llm.gemini import GeminiBackend
from shippid.llm.openai_compat import OpenAICompatBackend, normalize_base_url
from shippid.llm.render import contact_sheet, open_pdf, png_size, render_clip, render_page, tiles
from shippid.model import ActuatorType, SymbolClass
from shippid.symbols import Placement, draw_symbol, get_symbol, placed_bbox

PNG = probe_image()


class Out(BaseModel):
    label: str
    score: float


def completion(content, reasoning=None, key="reasoning", usage=(120, 30), model="google/gemma-4-31B-it"):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg[key] = reasoning
    return {"id": "chatcmpl-1", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}}


class Server:
    """Records requests and answers them with a handler(body, n) -> httpx.Response."""

    def __init__(self, handler):
        self.handler = handler
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        self.bodies.append(body)
        return self.handler(body, len(self.bodies))

    @property
    def transport(self):
        return httpx.MockTransport(self)


def oa_backend(server, **kw):
    kw.setdefault("backoff", 0.0)
    b = OpenAICompatBackend("google/gemma-4-31B-it", "http://gpu:8000/v1", transport=server.transport, **kw)
    b._sleep = lambda s: None
    return b


# --------------------------------------------------------------------------- OpenAI-compatible


def test_openai_request_shape():
    server = Server(lambda body, n: httpx.Response(200, json=completion('{"label": "valve", "score": 0.9}')))
    b = oa_backend(server, api_key="secret", top_k=64, extra_body={"repetition_penalty": 1.05})
    schema = schema_for(Out)
    msgs = [Message("system", "You are a P&ID expert."),
            Message("user", [ImagePart(PNG, label="crop 1"), "Classify the symbol."])]
    resp = b.complete(msgs, json_schema=schema, schema_name="Out", temperature=0.1, top_p=0.9, max_tokens=512,
                      thinking=False)
    req = server.requests[0]
    assert req.method == "POST" and req.url.path == "/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer secret"
    body = server.bodies[0]
    assert body["model"] == "google/gemma-4-31B-it"
    assert body["messages"][0] == {"role": "system", "content": "You are a P&ID expert."}
    parts = body["messages"][1]["content"]
    assert parts[0]["type"] == "image_url"  # image BEFORE text (Gemma convention)
    url = parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,") and base64.b64decode(url.split(",", 1)[1]) == PNG
    assert parts[-1]["type"] == "text" and parts[-1]["text"].endswith("Classify the symbol.")
    assert "Image 1: crop 1" in parts[-1]["text"]
    assert body["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "Out", "schema": schema, "strict": True}}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["top_k"] == 64 and body["repetition_penalty"] == 1.05
    assert body["temperature"] == 0.1 and body["top_p"] == 0.9 and body["max_tokens"] == 512
    assert resp.text == '{"label": "valve", "score": 0.9}'
    assert (resp.prompt_tokens, resp.completion_tokens) == (120, 30)
    assert resp.model == "google/gemma-4-31B-it" and resp.finish_reason == "stop"


def test_openai_text_only_and_thinking_default():
    server = Server(lambda body, n: httpx.Response(200, json=completion("hello")))
    b = oa_backend(server, thinking_default=True)
    b.complete([Message("user", "hi")])
    body = server.bodies[0]
    assert body["messages"] == [{"role": "user", "content": "hi"}]  # plain string for text-only turns
    assert "response_format" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    b.complete([Message("user", "hi")], thinking=False)  # per-request override
    assert server.bodies[1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_openai_response_format_fallback_is_remembered():
    def handler(body, n):
        rf = body.get("response_format")
        if rf and rf["type"] == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format type json_schema is not supported",
                                                       "type": "BadRequestError", "code": 400}})
        if rf and rf["type"] == "json_object":  # LM Studio style
            return httpx.Response(400, json={"error": "'response_format.type' must be 'json_schema'"})
        return httpx.Response(200, json=completion('{"label": "pump", "score": 0.7}'))

    server = Server(handler)
    b = oa_backend(server)
    client = LLMClient(b, schema_in_prompt=False)
    out = client.generate_json([Message("user", "Classify.")], Out)
    assert out.label == "pump"
    assert [bd.get("response_format", {}).get("type") for bd in server.bodies] == ["json_schema", "json_object", None]
    assert b.response_format_mode == "prompt" and b.structured_output == "prompt"
    # the model never saw the schema under guided decoding -> it is added to the prompt after the downgrade
    assert '"properties"' in server.bodies[-1]["messages"][-1]["content"]
    client.generate_json([Message("user", "Again.")], Out)
    assert len(server.bodies) == 4 and "response_format" not in server.bodies[3]  # downgrade remembered


def test_openai_422_schema_feature_falls_back_to_json_object():
    def handler(body, n):
        if body.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(422, json={"detail": "The provided JSON schema contains features not supported by xgrammar."})
        return httpx.Response(200, json=completion('{"label": "x", "score": 1}'))

    server = Server(handler)
    b = oa_backend(server)
    b.complete([Message("user", "q")], json_schema=schema_for(Out))
    assert b.response_format_mode == "json_object"
    assert server.bodies[-1]["response_format"] == {"type": "json_object"}


def test_openai_unrelated_400_raises_with_server_message():
    server = Server(lambda body, n: httpx.Response(
        404 if n > 99 else 400, json={"error": {"message": "The model `gemma-x` does not exist.", "code": 400}}))
    b = oa_backend(server)
    with pytest.raises(LLMError, match="does not exist"):
        b.complete([Message("user", "q")], json_schema=schema_for(Out))
    assert len(server.bodies) == 1 and b.response_format_mode == "json_schema"


@pytest.mark.parametrize("key", ["reasoning", "reasoning_content"])
def test_openai_reasoning_fields(key):
    server = Server(lambda body, n: httpx.Response(200, json=completion('{"a": 1}', reasoning="Let me think.", key=key)))
    resp = oa_backend(server).complete([Message("user", "q")], thinking=True)
    assert resp.reasoning == "Let me think." and resp.text == '{"a": 1}'


def test_openai_inline_thought_tokens_are_split():
    raw = '<|channel>thought\nThe symbol is a bow-tie.<channel|>{"label": "gate_valve", "score": 0.8}<turn|>'
    server = Server(lambda body, n: httpx.Response(200, json=completion(raw)))
    resp = oa_backend(server).complete([Message("user", "q")])
    assert resp.reasoning == "The symbol is a bow-tie."
    assert json.loads(resp.text) == {"label": "gate_valve", "score": 0.8}


def test_openai_retries_503_with_backoff():
    def handler(body, n):
        if n <= 2:
            return httpx.Response(503, text="Service Unavailable: model is loading")
        return httpx.Response(200, json=completion("ok"))

    server = Server(handler)
    b = oa_backend(server, backoff=0.5, max_retries=2)
    sleeps = []
    b._sleep = sleeps.append
    assert b.complete([Message("user", "q")]).text == "ok"
    assert sleeps == [0.5, 1.0] and len(server.bodies) == 3


def test_openai_retries_exhausted_and_retry_after():
    server = Server(lambda body, n: httpx.Response(429, headers={"Retry-After": "3"}, json={"error": "rate limited"}))
    b = oa_backend(server, max_retries=1)
    sleeps = []
    b._sleep = sleeps.append
    with pytest.raises(LLMError, match="HTTP 429.*rate limited"):
        b.complete([Message("user", "q")])
    assert sleeps == [3.0] and len(server.bodies) == 2


def test_openai_retries_timeouts_and_reports_unreachable():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json=completion("late but fine"))

    b = OpenAICompatBackend("m", "http://gpu:8000/v1", transport=httpx.MockTransport(handler), backoff=0)
    b._sleep = lambda s: None
    assert b.complete([Message("user", "q")]).text == "late but fine"

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    b2 = OpenAICompatBackend("m", "http://gpu:8000/v1", transport=httpx.MockTransport(refuse), max_retries=1, backoff=0)
    b2._sleep = lambda s: None
    with pytest.raises(LLMError, match="cannot reach http://gpu:8000/v1/chat/completions"):
        b2.complete([Message("user", "q")])


def test_openai_bad_payloads():
    server = Server(lambda body, n: httpx.Response(200, json={"object": "error", "message": "x", "choices": []}))
    with pytest.raises(LLMError, match="without choices"):
        oa_backend(server).complete([Message("user", "q")])
    server = Server(lambda body, n: httpx.Response(200, text="<html>proxy</html>"))
    with pytest.raises(LLMError, match="non-JSON"):
        oa_backend(server).complete([Message("user", "q")])


def test_openai_through_llm_client_counts_usage():
    server = Server(lambda body, n: httpx.Response(200, json=completion('{"label": "tank", "score": 0.99}', usage=(900, 12))))
    client = LLMClient(oa_backend(server))
    assert client.generate_json([Message("user", [ImagePart(PNG), "classify"])], Out).label == "tank"
    assert client.usage.calls == 1 and client.usage.prompt_tokens == 900 and client.usage.completion_tokens == 12
    assert server.bodies[0]["response_format"]["type"] == "json_schema"


def test_normalize_base_url():
    assert normalize_base_url("http://localhost:11434") == "http://localhost:11434/v1"
    assert normalize_base_url("http://h:8000/v1/") == "http://h:8000/v1"
    assert normalize_base_url("http://h:8000/v1/chat/completions") == "http://h:8000/v1"
    assert normalize_base_url("https://gw.example.com/openai/v1") == "https://gw.example.com/openai/v1"


# --------------------------------------------------------------------------- Gemini API


def gemini_answer(text, thought=None, usage=(200, 40, 10)):
    parts = ([{"text": thought, "thought": True}] if thought else []) + [{"text": text}]
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": usage[0], "candidatesTokenCount": usage[1],
                              "thoughtsTokenCount": usage[2], "totalTokenCount": sum(usage)},
            "modelVersion": "gemma-4-31b-it"}


def gem_backend(server, **kw):
    b = GeminiBackend(api_key="g-key", transport=server.transport, backoff=0, **kw)
    b._sleep = lambda s: None
    return b


def test_gemini_request_and_response_shape():
    server = Server(lambda body, n: httpx.Response(200, json=gemini_answer('{"label": "pump", "score": 0.8}',
                                                                            thought="Circle with triangle.")))
    b = gem_backend(server)
    schema = schema_for(Out)
    msgs = [Message("system", "Be precise."), Message("user", [ImagePart(PNG), "What is it?"]),
            Message("assistant", "{}"), Message("user", "Fix it.")]
    resp = b.complete(msgs, json_schema=schema, temperature=0.3, top_p=0.9, max_tokens=300, thinking=True)
    req = server.requests[0]
    assert req.url.path == "/v1beta/models/gemma-4-31b-it:generateContent"
    assert req.headers["x-goog-api-key"] == "g-key"
    body = server.bodies[0]
    assert body["systemInstruction"] == {"parts": [{"text": "Be precise."}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    first = body["contents"][0]["parts"]
    assert first[0] == {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(PNG).decode()}}
    assert first[1] == {"text": "What is it?"}
    gen = body["generationConfig"]
    assert gen["temperature"] == 0.3 and gen["topP"] == 0.9 and gen["maxOutputTokens"] == 300
    assert gen["responseMimeType"] == "application/json" and gen["responseJsonSchema"] == schema
    assert gen["thinkingConfig"] == {"thinkingLevel": "high"}
    assert resp.text == '{"label": "pump", "score": 0.8}' and resp.reasoning == "Circle with triangle."
    assert resp.prompt_tokens == 200 and resp.completion_tokens == 50 and resp.model == "gemma-4-31b-it"
    b.complete([Message("user", "x")], thinking=False)
    assert server.bodies[1]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert "responseMimeType" not in server.bodies[1]["generationConfig"]


def test_gemini_fallbacks_are_remembered():
    def handler(body, n):
        gen = body["generationConfig"]
        if "systemInstruction" in body:
            return httpx.Response(400, json={"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                                       "message": "Developer instruction is not enabled for models/gemma-4-31b-it"}})
        if "responseJsonSchema" in gen:
            return httpx.Response(400, json={"error": {"code": 400, "message": "JSON schema is not supported for this model."}})
        if "responseMimeType" in gen:
            return httpx.Response(400, json={"error": {"code": 400, "message": "JSON mode is not enabled for models/gemma-4-31b-it"}})
        return httpx.Response(200, json=gemini_answer('```json\n{"label": "tank", "score": 0.6}\n```'))

    server = Server(handler)
    b = gem_backend(server)
    client = LLMClient(b, schema_in_prompt=False)
    msgs = [Message("system", "SYSTEM RULES"), Message("user", [ImagePart(PNG), "Classify."])]
    assert client.generate_json(msgs, Out).label == "tank"
    assert len(server.bodies) == 4
    assert not b.system_instruction_ok and b.json_level == "prompt" and b.structured_output == "prompt"
    parts = server.bodies[-1]["contents"][0]["parts"]
    assert "inline_data" in parts[0] and parts[1] == {"text": "SYSTEM RULES"}  # prepended after the image
    assert '"properties"' in parts[-1]["text"]  # schema moved into the prompt
    client.generate_json(msgs, Out)
    assert len(server.bodies) == 5  # no more failing attempts


def test_gemini_errors():
    server = Server(lambda body, n: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(LLMError, match="blockReason=SAFETY"):
        gem_backend(server).complete([Message("user", "x")])
    server = Server(lambda body, n: httpx.Response(403, json={"error": {"code": 403, "message": "API key not valid."}}))
    with pytest.raises(LLMError, match="API key not valid"):
        gem_backend(server).complete([Message("user", "x")], json_schema=schema_for(Out))
    server = Server(lambda body, n: httpx.Response(500 if n == 1 else 200, json=gemini_answer("fine")
                                                   if n > 1 else {"error": {"message": "internal"}}))
    assert gem_backend(server).complete([Message("user", "x")]).text == "fine"


def test_gemini_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMError, match="API key"):
        GeminiBackend()
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert GeminiBackend("models/gemma-4-31b-it").model == "gemma-4-31b-it"


# --------------------------------------------------------------------------- factory


@pytest.mark.parametrize("name,url,model", [
    ("openai", "http://localhost:8000/v1", "google/gemma-4-31B-it"),
    ("vllm", "http://localhost:8000/v1", "google/gemma-4-31B-it"),
    ("sglang", "http://localhost:30000/v1", "google/gemma-4-31B-it"),
    ("ollama", "http://localhost:11434/v1", "gemma4:31b"),
    ("llamacpp", "http://localhost:8080/v1", "google/gemma-4-31B-it"),
    ("lmstudio", "http://localhost:1234/v1", "google/gemma-4-31B-it"),
])
def test_factory_openai_compatible(name, url, model):
    client = make_client(LLMConfig(backend=name))
    assert isinstance(client.backend, OpenAICompatBackend)
    assert client.backend.base_url == url and client.backend.model == model and client.backend.name == name
    assert client.json_mode == "json_schema"


def test_factory_options(monkeypatch):
    cfg = LLMConfig(backend="vllm", base_url="http://10.0.0.5:9000/v1", model="RedHatAI/gemma-4-31B-it-FP8-dynamic",
                    json_mode="json_object", enable_thinking=True, extra_body={"top_k": 64, "min_p": 0.0},
                    temperature=0.1, max_tokens=4000)
    client = make_client(cfg)
    b = client.backend
    assert b.base_url == "http://10.0.0.5:9000/v1" and b.model == "RedHatAI/gemma-4-31B-it-FP8-dynamic"
    assert b.response_format_mode == "json_object" and b.thinking_default is True
    assert b.top_k == 64 and b.extra_body == {"min_p": 0.0}
    assert client.temperature == 0.1 and client.max_tokens == 4000
    assert make_client(LLMConfig(backend="none")) is None
    with pytest.raises(ValueError, match="responder"):
        make_client(LLMConfig(backend="mock"))
    mock = make_client(LLMConfig(backend="mock"), responder=lambda m, s, n: {"label": "x", "score": 0})
    assert isinstance(mock.backend, MockBackend) and mock.usage.backend == "mock"
    with pytest.raises(ValueError, match="unknown LLM backend"):
        make_client(LLMConfig(backend="foo"))
    gem = make_client(LLMConfig(backend="gemini", api_key="k"))
    assert isinstance(gem.backend, GeminiBackend) and gem.backend.model == "gemma-4-31b-it"
    assert gem.backend.url.startswith("https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it")
    assert make_client(LLMConfig(backend="ollama", base_url="http://box:11434")).backend.base_url == "http://box:11434/v1"
    assert make_client(LLMConfig(backend="vllm", json_mode="prompt")).json_mode == "prompt"


def test_check_connection_reports_capabilities():
    def handler(body, n):
        content = body["messages"][1]["content"]
        assert content[0]["type"] == "image_url" and content[-1]["type"] == "text"
        return httpx.Response(200, json=completion('{"shape": "filled circle", "color": "red", "text": "P1"}',
                                                   reasoning="", model="gemma-4-31B-it-FP8"))

    server = Server(handler)
    client = LLMClient(oa_backend(server))
    rep = check_connection(client)
    assert rep["ok"] and rep["json_valid"] and rep["images"] and rep["json_schema"]
    assert rep["server_model"] == "gemma-4-31B-it-FP8" and rep["error"] is None
    assert server.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert client.usage.calls == 0  # diagnostic call bypasses cache and accounting


def test_check_connection_degraded_and_down():
    def handler(body, n):
        if body.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, json={"error": {"message": "guided decoding not available"}})
        return httpx.Response(200, json=completion('{"shape": "square", "color": "blue", "text": ""}'))

    rep = check_connection(LLMClient(oa_backend(Server(handler))))
    assert rep["ok"] and rep["json_valid"] and not rep["json_schema"] and rep["structured_output"] == "json_object"
    assert not rep["images"]  # wrong answer about the test image

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    b = OpenAICompatBackend("m", "http://localhost:8000/v1", transport=httpx.MockTransport(refuse), max_retries=0)
    rep = check_connection(LLMClient(b))
    assert not rep["ok"] and "cannot reach" in rep["error"]


# --------------------------------------------------------------------------- rendering


@pytest.fixture()
def symbol_pdf(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=842, height=595)
    shape = page.new_shape()
    rects = []
    specs = [(SymbolClass.GATE_VALVE, ActuatorType.HANDWHEEL), (SymbolClass.PUMP_CENTRIFUGAL, ActuatorType.NONE),
             (SymbolClass.BUTTERFLY_VALVE, ActuatorType.HYDRAULIC), (SymbolClass.TANK, ActuatorType.NONE)]
    for i, (cls, act) in enumerate(specs):
        spec, pl = get_symbol(cls), Placement(100 + 150 * i, 200, scale=20)
        draw_symbol(shape, spec, pl, act)
        rects.append(pymupdf.Rect(placed_bbox(spec, pl, act)))
    shape.commit()
    hidden = doc.add_ocg("HIDDEN", on=False)
    page.draw_circle((700, 500), 20, color=(0, 0, 0), fill=(0, 0, 0), oc=hidden)
    path = tmp_path / "symbols.pdf"
    doc.save(path)
    return path, rects


def _dark_pixels(png):
    pix = pymupdf.Pixmap(png)
    samples = pix.samples  # property: copies the buffer on every access
    return sum(1 for v in samples[:: pix.n] if v < 100)


def test_render_clip_dimensions(symbol_pdf):
    path, rects = symbol_pdf
    page = open_pdf(path)[0]
    for max_side in (1536, 700, 200):
        for r in rects + [page.rect, pymupdf.Rect(0, 0, 842, 10)]:
            w, h = png_size(render_clip(page, r, max_side=max_side, margin=4))
            assert w % 48 == 0 and h % 48 == 0
            assert max(w, h) <= max_side and min(w, h) >= 48
    w, h = png_size(render_clip(page, rects[0], max_side=480))
    assert max(w, h) == 480
    assert png_size(render_page(page, 1536)) == (1536, 1104)
    assert _dark_pixels(render_clip(page, rects[1], max_side=480, margin=5)) > 200  # the pump is visible
    edge = render_clip(page, pymupdf.Rect(820, 580, 900, 640), max_side=240)  # partly off-page -> white padding
    assert png_size(edge) == (240, 192)


def test_open_pdf_turns_layers_on(symbol_pdf):
    path, _ = symbol_pdf
    region = pymupdf.Rect(670, 470, 730, 530)
    plain = pymupdf.open(path)[0]
    assert _dark_pixels(render_clip(plain, region, max_side=96)) == 0
    assert _dark_pixels(render_clip(open_pdf(path)[0], region, max_side=96)) > 500


def test_contact_sheet_layout(symbol_pdf):
    path, rects = symbol_pdf
    page = open_pdf(path)[0]
    items = [(str(i + 1), r + (-6, -6, 6, 6)) for i, r in enumerate(rects)]
    sheets = contact_sheet(page, items, cell_px=192, cols=6)
    assert len(sheets) == 1 and sheets[0][1] == ["1", "2", "3", "4"]
    w, h = png_size(sheets[0][0])
    assert (w, h) == (768, 240)  # 4 cells of 192 px, one row of 192 + 24 px label strip -> padded to 240
    many = contact_sheet(page, items * 12, cell_px=192, cols=6, max_side=1536)
    assert [len(lbls) for _, lbls in many] == [42, 6]
    for png, _ in many:
        w, h = png_size(png)
        assert w % 48 == 0 and h % 48 == 0 and max(w, h) <= 1536
    # the label strip carries real (rasterised) text: re-open the PNG and check dark pixels in the strip
    pix = pymupdf.Pixmap(sheets[0][0])
    strip = [pix.pixel(x, y)[0] for x in range(2, 40) for y in range(2, 22)]
    assert min(strip) < 80


def test_tiles_cover_page(symbol_pdf):
    path, _ = symbol_pdf
    page = open_pdf(path)[0]
    result = tiles(page, tile_px=768, overlap=0.2, dpi=144)
    rects = [r for r, _ in result]
    assert len(result) == 6
    assert min(r.x0 for r in rects) == 0 and max(r.x1 for r in rects) == pytest.approx(842)
    assert min(r.y0 for r in rects) == 0 and max(r.y1 for r in rects) == pytest.approx(595)
    for r, png in result:
        assert png_size(png) == (768, 768) and r.width == pytest.approx(384)
    xs = sorted({round(r.x0, 3) for r in rects})
    assert xs[1] - xs[0] <= 384 * 0.8 + 1e-6  # overlap at least 20 %
    small = tiles(page, tile_px=1536, dpi=72)  # page smaller than one tile
    assert len(small) == 1 and png_size(small[0][1]) == (864, 624)


def test_openai_repair_round_is_sent_as_chat_turns():
    answers = iter(['{"label": "pump"}', '{"label": "pump", "score": 0.8}'])
    server = Server(lambda body, n: httpx.Response(200, json=completion(next(answers))))
    client = LLMClient(oa_backend(server))
    assert client.generate_json([Message("system", "sys"), Message("user", "classify")], Out).score == 0.8
    roles = [m["role"] for m in server.bodies[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert server.bodies[1]["messages"][2]["content"] == '{"label": "pump"}'
    assert "score" in server.bodies[1]["messages"][3]["content"]  # pydantic error fed back to the model


def test_factory_gemini_json_object_mode():
    gem = make_client(LLMConfig(backend="gemini", api_key="k", json_mode="json_object", enable_thinking=False))
    assert gem.backend.json_level == "json_mime" and gem.backend.structured_output == "json_object"
    body = gem.backend.build_request([Message("user", "x")], json_schema=schema_for(Out))
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert "responseJsonSchema" not in body["generationConfig"]
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert '"properties"' in body["contents"][0]["parts"][-1]["text"]
