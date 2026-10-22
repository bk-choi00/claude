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
from shippid.config import load_config
from shippid.llm.factory import base_url_is_explicit, check_connection, make_client, probe_image
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
def test_factory_openai_compatible(name, url, model, monkeypatch, tmp_path):
    monkeypatch.delenv("SHIPPID_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SHIPPID_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
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


# --------------------------------------------------------------------------- regressions (review round 2)


def test_ollama_thinking_uses_reasoning_effort():
    """Ollama /v1 ignores chat_template_kwargs; Gemma 4 thinking is switched with reasoning_effort."""
    def handler(body, n):
        thinking = body.get("reasoning_effort", "none") != "none"
        msg = completion('{"label": "pump", "score": 0.7}', reasoning="Circle with a triangle." if thinking else None,
                         model="gemma4:31b")
        return httpx.Response(200, json=msg)

    server = Server(handler)
    b = OpenAICompatBackend("gemma4:31b", "http://localhost:11434/v1", transport=server.transport, name="ollama")
    assert b.thinking_param == "reasoning_effort"
    off = b.complete([Message("user", [ImagePart(PNG), "classify"])], json_schema=schema_for(Out), thinking=False)
    on = b.complete([Message("user", "classify")], json_schema=schema_for(Out), thinking=True)
    first, second = server.bodies
    assert first["reasoning_effort"] == "none" and second["reasoning_effort"] == "high"
    assert "chat_template_kwargs" not in first and "chat_template_kwargs" not in second
    assert first["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert first["temperature"] == 0.2 and first["top_p"] == 0.95  # sent explicitly (Ollama forces top_p=1 otherwise)
    assert first["response_format"]["type"] == "json_schema"
    assert (off.reasoning, on.reasoning) == ("", "Circle with a triangle.") and on.model == "gemma4:31b"
    b.complete([Message("user", "x")])  # thinking=None and no default: server default, nothing sent
    assert "reasoning_effort" not in server.bodies[2] and "chat_template_kwargs" not in server.bodies[2]


def test_thinking_parameter_per_server(monkeypatch, tmp_path):
    monkeypatch.delenv("SHIPPID_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SHIPPID_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    ollama = make_client(LLMConfig(backend="ollama")).backend
    body = ollama.build_request([Message("user", "hi")])  # LLMConfig.enable_thinking=False -> explicitly off
    assert body["model"] == "gemma4:31b" and body["reasoning_effort"] == "none" and "chat_template_kwargs" not in body
    assert ollama.build_request([Message("user", "hi")], thinking=True)["reasoning_effort"] == "high"
    # an Ollama server addressed through the generic backend is recognised by its port
    generic = make_client(LLMConfig(backend="openai", base_url="http://gpu-box:11434/v1")).backend
    assert generic.thinking_param == "reasoning_effort"
    for name in ("vllm", "sglang", "llamacpp"):
        body = make_client(LLMConfig(backend=name)).backend.build_request([Message("user", "hi")], thinking=True)
        assert body["chat_template_kwargs"] == {"enable_thinking": True} and "reasoning_effort" not in body
    forced = OpenAICompatBackend("m", "http://h:8000/v1", thinking_param="reasoning_effort", reasoning_effort="low")
    assert forced.build_request([Message("user", "hi")], thinking=True)["reasoning_effort"] == "low"
    with pytest.raises(ValueError, match="thinking_param"):
        OpenAICompatBackend("m", "http://h:8000/v1", thinking_param="think")
    with pytest.raises(ValueError, match="reasoning_effort"):
        OpenAICompatBackend("m", "http://h:8000/v1", reasoning_effort="none")


def test_factory_keeps_an_explicit_default_looking_base_url(monkeypatch, tmp_path):
    """http://localhost:8000/v1 set on purpose (e.g. llama.cpp on port 8000) must not be replaced."""
    monkeypatch.delenv("SHIPPID_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SHIPPID_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    url8000 = "http://localhost:8000/v1/chat/completions"
    cfg = LLMConfig(backend="llamacpp")
    assert not base_url_is_explicit(cfg)
    unset = make_client(cfg)
    assert unset.backend.url == "http://localhost:8080/v1/chat/completions"
    assert unset.backend.base_url_source == "backend default"
    assert make_client(cfg, base_url_explicit=True).backend.url == url8000
    flagged = LLMConfig(backend="sglang")
    flagged.base_url_explicit = True  # set by a caller that knows the value came from the user
    assert make_client(flagged).backend.url == url8000
    monkeypatch.setenv("SHIPPID_LLM_BACKEND", "llamacpp")
    monkeypatch.setenv("SHIPPID_LLM_BASE_URL", "http://localhost:8000/v1")
    from_env = make_client(load_config().llm)
    assert from_env.backend.url == url8000 and from_env.backend.base_url_source == "configured"
    monkeypatch.delenv("SHIPPID_LLM_BASE_URL")
    monkeypatch.delenv("SHIPPID_LLM_BACKEND")
    (tmp_path / "shippid.yaml").write_text("llm:\n  backend: ollama\n  base_url: http://localhost:8000/v1\n")
    from_yaml = make_client(load_config().llm)
    assert from_yaml.backend.name == "ollama" and from_yaml.backend.url == url8000
    gem = make_client(LLMConfig(backend="gemini", api_key="k"), base_url_explicit=True)
    assert gem.backend.url.startswith("http://localhost:8000/v1/models/gemma-4-31b-it")
    (tmp_path / "shippid.yaml").write_text("llm:\n  backend: llamacpp\n")
    assert make_client(load_config().llm).backend.url == "http://localhost:8080/v1/chat/completions"


def test_check_connection_reports_base_url_source(monkeypatch, tmp_path):
    monkeypatch.delenv("SHIPPID_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SHIPPID_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    client = make_client(LLMConfig(backend="ollama"))
    server = Server(lambda body, n: httpx.Response(200, json=completion('{"shape": "circle", "color": "red"}')))
    client.backend._http = httpx.Client(transport=server.transport)
    report = check_connection(client)
    assert report["ok"] and report["base_url_source"] == "backend default"
    assert report["url"] == "http://localhost:11434/v1/chat/completions"
    assert server.bodies[0]["reasoning_effort"] == "none" and "chat_template_kwargs" not in server.bodies[0]


# --------------------------------------------------------------------------- credentials never in errors

SECRET = "sk-PROD-KEY-0123456789"


def _no_secret(*texts, secrets=(SECRET,)):
    for t in texts:
        for s in secrets:
            assert s not in str(t), f"secret leaked: {str(t)[:200]}"


def test_api_key_whitespace_is_stripped_and_control_characters_rejected():
    server = Server(lambda body, n: httpx.Response(200, json=completion('{"label": "x", "score": 1}')))
    for raw in (SECRET + "\n", SECRET + "\r\n", "  " + SECRET + "\t"):  # YAML block scalar, CRLF env file
        b = oa_backend(server, api_key=raw)
        assert b.api_key == SECRET
        LLMClient(b).generate_json([Message("user", "q")], Out)
        assert server.requests[-1].headers["authorization"] == f"Bearer {SECRET}"
    with pytest.raises(LLMError) as err:
        OpenAICompatBackend("m", "http://gpu:8000/v1", api_key="sk-PROD\nKEY-0123456789")
    assert "control character" in str(err.value)
    _no_secret(err.value, secrets=("sk-PROD", "KEY-0123456789"))
    with pytest.raises(LLMError) as err:
        GeminiBackend(api_key="AIzaSy\x00FAKE012345")
    _no_secret(err.value, secrets=("AIzaSy", "FAKE012345"))
    gem = GeminiBackend(api_key="AIzaSyFAKE012345\n", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert gem.api_key == "AIzaSyFAKE012345"


def test_transport_errors_never_quote_api_key_or_url_credentials():
    def protocol_error(request):  # what h11 raises for a header value with a line break
        raise httpx.LocalProtocolError(f"Illegal header value b'Bearer {SECRET}\\n'")

    b = OpenAICompatBackend("m", "http://gpu:8000/v1", api_key=SECRET, transport=httpx.MockTransport(protocol_error),
                            max_retries=0)
    with pytest.raises(LLMError) as err:
        LLMClient(b).generate_text([Message("user", "q")])
    assert "could not be sent" in str(err.value) and "api_key" in str(err.value)
    _no_secret(err.value)

    def refuse(request):
        raise httpx.ConnectError(f"refused {request.url}", request=request)

    url = "http://svc-user:S3cr3tTok3n@gpu:8000/v1?key=QKEY9876"
    b = OpenAICompatBackend("m", url, transport=httpx.MockTransport(refuse), max_retries=0)
    assert b.url == "http://svc-user:S3cr3tTok3n@gpu:8000/v1/chat/completions?key=QKEY9876"  # query stays last
    assert b.display_url == "http://gpu:8000/v1/chat/completions"
    with pytest.raises(LLMError) as err:
        LLMClient(b).generate_text([Message("user", "q")])
    assert "cannot reach http://gpu:8000/v1/chat/completions" in str(err.value)
    _no_secret(err.value, secrets=("S3cr3tTok3n", "QKEY9876"))
    rep = check_connection(LLMClient(b))
    assert rep["url"] == "http://gpu:8000/v1/chat/completions"
    _no_secret(rep, secrets=("S3cr3tTok3n", "QKEY9876"))


def test_http_error_bodies_echoing_the_key_are_redacted(caplog):
    echo = lambda body, n: httpx.Response(401, json={"error": {"message": f"invalid api key {SECRET} for user"}})
    b = oa_backend(Server(echo), api_key=SECRET + "\n")
    with pytest.raises(LLMError) as err:
        LLMClient(b).generate_text([Message("user", "q")])
    assert "HTTP 401" in str(err.value) and "***" in str(err.value)
    _no_secret(err.value, caplog.text)
    retry = lambda body, n: httpx.Response(503, json={"error": {"message": f"upstream rejected key {SECRET}"}})
    b = oa_backend(Server(retry), api_key=SECRET, max_retries=1)
    with pytest.raises(LLMError) as err:
        LLMClient(b).generate_text([Message("user", "q")])
    _no_secret(err.value, caplog.text)  # the retry warning is logged too

    def gem_echo(request):
        return httpx.Response(400, json={"error": {"message": "API key not valid: AIzaSyFAKE012345"}})

    gem = GeminiBackend(api_key="AIzaSyFAKE012345", transport=httpx.MockTransport(gem_echo), max_retries=0)
    with pytest.raises(LLMError) as err:
        LLMClient(gem).generate_text([Message("user", "q")])
    _no_secret(err.value, secrets=("AIzaSyFAKE012345",))

    def gem_protocol(request):
        raise httpx.LocalProtocolError("Illegal header value b'AIzaSyFAKE012345\\n'")

    gem = GeminiBackend(api_key="AIzaSyFAKE012345", transport=httpx.MockTransport(gem_protocol), max_retries=0)
    with pytest.raises(LLMError) as err:
        LLMClient(gem).generate_text([Message("user", "q")])
    _no_secret(err.value, secrets=("AIzaSyFAKE012345",))


def test_normalize_base_url_keeps_query_last():
    assert normalize_base_url("http://gw:8000/v1/chat/completions?api-version=2") == "http://gw:8000/v1?api-version=2"
    assert normalize_base_url("http://gw:8000?x=1") == "http://gw:8000/v1?x=1"


# --------------------------------------------------------------------------- replay cache key


def test_cache_key_covers_thinking_default_top_p_and_backend_settings(tmp_path):
    def handler(body, n):
        tag = (f"thinking={body.get('chat_template_kwargs', {}).get('enable_thinking')} top_p={body['top_p']} "
               f"top_k={body.get('top_k')}")
        return httpx.Response(200, json=completion(json.dumps({"label": tag, "score": 1.0})))

    def run(thinking_default, top_p=0.95, top_k=None, extra=None):
        server = Server(handler)
        b = oa_backend(server, thinking_default=thinking_default, top_k=top_k, extra_body=extra)
        out = LLMClient(b, cache_dir=tmp_path, top_p=top_p).generate_json([Message("user", "q")], Out)
        return out.label, len(server.bodies)

    assert run(False) == ("thinking=False top_p=0.95 top_k=None", 1)
    assert run(False) == ("thinking=False top_p=0.95 top_k=None", 0)  # replayed
    assert run(True) == ("thinking=True top_p=0.95 top_k=None", 1)  # effective thinking switch changed
    assert run(True, top_p=0.5) == ("thinking=True top_p=0.5 top_k=None", 1)
    assert run(True, top_p=0.5, top_k=64) == ("thinking=True top_p=0.5 top_k=64", 1)
    assert run(True, top_p=0.5, top_k=64, extra={"min_p": 0.1})[1] == 1
    assert run(True, top_p=0.5, top_k=64, extra={"min_p": 0.1})[1] == 0


def test_cache_replays_after_a_response_format_downgrade(tmp_path):
    def handler(body, n):
        if body.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, json={"error": {"message": "json_schema response_format not supported"}})
        return httpx.Response(200, json=completion('{"label": "pump", "score": 0.9}'))

    for expected_requests in (3, 0):  # run 1: rejected + 2 answers; run 2: everything replayed
        server = Server(handler)
        client = LLMClient(oa_backend(server), cache_dir=tmp_path)
        for q in ("a", "b"):
            assert client.generate_json([Message("user", q)], Out).label == "pump"
        assert len(server.bodies) == expected_requests


# --------------------------------------------------------------------------- cut-off answers


def test_cut_off_answer_is_retried_without_echo_with_thinking_off_and_more_tokens():
    cut = '{"label": "a very long label that never ends'

    def handler(body, n):
        if n == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": cut}, "finish_reason": "length"}],
                                             "usage": {"prompt_tokens": 50, "completion_tokens": 1000}})
        return httpx.Response(200, json=completion('{"label": "tank", "score": 0.9}'))

    server = Server(handler)
    client = LLMClient(oa_backend(server, thinking_default=True), max_tokens=1000, context_tokens=32768)
    msgs = [Message("system", "s"), Message("user", "classify")]
    assert client.generate_json(msgs, Out).label == "tank"
    first, second = server.bodies
    assert first["max_tokens"] == 1000 and first["chat_template_kwargs"] == {"enable_thinking": True}
    assert second["max_tokens"] == 2000 and second["chat_template_kwargs"] == {"enable_thinking": False}
    assert [m["role"] for m in second["messages"]] == ["system", "user"]  # no assistant echo of the fragment
    assert cut not in json.dumps(second) and "cut off at the max_tokens limit (1000 tokens)" in second["messages"][-1]["content"]
    assert "cut off" in client.usage.errors[0]
    # unknown context: same budget (a larger one could be rejected), thinking still switched off
    server = Server(handler)
    client = LLMClient(oa_backend(server, thinking_default=True), max_tokens=1000)
    client.generate_json(msgs, Out)
    assert server.bodies[1]["max_tokens"] == 1000 and server.bodies[1]["chat_template_kwargs"]["enable_thinking"] is False
    # the raised budget stays inside the context
    server = Server(handler)
    client = LLMClient(oa_backend(server), max_tokens=1000, context_tokens=1500)
    client.generate_json(msgs, Out)
    assert 1000 <= server.bodies[1]["max_tokens"] < 1500


def test_gemini_max_tokens_finish_counts_as_cut_off():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"label": "x'}]},
                                                             "finishReason": "MAX_TOKENS"}]})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"label": "x", "score": 1}'}]},
                                                         "finishReason": "STOP"}]})

    gem = GeminiBackend(api_key="k", transport=httpx.MockTransport(handler), thinking_default=True)
    LLMClient(gem, max_tokens=800, context_tokens=32768).generate_json([Message("user", "q")], Out)
    assert calls[1]["generationConfig"]["maxOutputTokens"] == 1600
    assert calls[1]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert len(calls[1]["contents"]) == 1  # no model turn echoing the fragment


# --------------------------------------------------------------------------- context window setting


def test_context_tokens_from_config_reaches_client(monkeypatch):
    assert LLMConfig().context_tokens == 32768
    assert make_client(LLMConfig(backend="vllm")).context_tokens == 32768
    assert make_client(LLMConfig(backend="vllm", context_tokens=16384)).context_tokens == 16384
    monkeypatch.setenv("SHIPPID_LLM_CONTEXT_TOKENS", "65536")
    assert load_config().llm.context_tokens == 65536
    assert LLMClient(MockBackend(lambda m, s, n: {})).context_tokens is None
