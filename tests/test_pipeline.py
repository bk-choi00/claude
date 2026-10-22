"""End-to-end tests of the full pipeline (PDF -> extraction -> analysis -> 3D -> reports).

The Gemma path is exercised with a scripted MockBackend: it returns schema-valid but
deliberately uninformative answers, so the test checks the plumbing (calls made, usage
recorded, honest fallbacks) -- not model quality, which needs a real Gemma 4 31B server.
"""

import json
import re
import struct

import pytest

from shippid.cli import main as cli_main
from shippid.config import Config
from shippid.datasets.pdf_writer import write_pdf
from shippid.datasets.synthetic import generate
from shippid.llm.base import LLMClient, MockBackend
from shippid.model import SystemType
from shippid.pipeline import run_pipeline, summarize


@pytest.fixture(scope="module")
def ballast_pdf(tmp_path_factory):
    d = tmp_path_factory.mktemp("pdf")
    gt, spec = generate(11, SystemType.BALLAST)
    path = write_pdf(spec, d / "ballast.pdf")
    gt.save(d / "ballast.json")
    return path, gt


def _glb_json(path):
    data = path.read_bytes()
    magic, _version, _length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF"
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    assert chunk_type == 0x4E4F534A  # JSON
    return json.loads(data[20:20 + chunk_len])


def test_pipeline_deterministic(ballast_pdf, tmp_path):
    pdf, gt = ballast_pdf
    cfg = Config()
    cfg.llm.backend = "none"
    res = run_pipeline(pdf, tmp_path / "out", cfg, viewer="both")
    s = summarize(res)
    assert s["llm"] == "none"
    assert res.document.info.system == SystemType.BALLAST
    assert len(res.document.objects) >= 0.9 * len(gt.objects)
    for key in ("extracted", "document", "analysis", "layout", "glb", "viewer:threejs", "viewer:plotly", "report"):
        assert res.paths[key].is_file(), key
    assert res.layout.planner == "heuristic"
    assert {p.object_id for p in res.layout.placements} == {o.id for o in res.document.objects}
    gltf = _glb_json(res.paths["glb"])
    names = [n.get("name", "") for n in gltf["nodes"]]
    assert len(names) == len(set(names))
    assert not any(re.search(r"[\s\[\].:/]", n) for n in names)
    html = res.paths["report"].read_text(encoding="utf-8")
    assert "viewer.html" in html and "R_" in html
    assert res.report.llm.calls == 0


def _scripted_gemma(messages, schema, name):
    """Schema-valid, uninformative answers (never adopted over the vector result)."""
    text = "\n".join(m.text() for m in messages if m.role == "user")
    if name == "SymbolBatch":
        ids = list(dict.fromkeys(re.findall(r'"id":\s*"(o\d+)"', text)))
        return {"items": [{"id": i, "reason": "scripted", "cls": "unknown", "actuator": "none", "confidence": 0.0}
                          for i in ids]}
    if name == "DrawingInfoOut":
        return {"system": "ballast", "systems": ["ballast"]}
    if name == "TextAssignments":
        return {"assignments": []}
    if name == "ReviewOut":
        return {"findings": [], "summary_en": "scripted review", "summary_ko": "스크립트 검토 결과"}
    return {"items": []}  # layout plan: empty -> the planner must fall back honestly


def test_pipeline_with_llm_client(ballast_pdf, tmp_path):
    pdf, _ = ballast_pdf
    backend = MockBackend(_scripted_gemma, model="scripted-gemma")
    client = LLMClient(backend, max_attempts=2)
    res = run_pipeline(pdf, tmp_path / "out", Config(), client=client)
    assert backend.calls, "the Gemma steps must call the LLM"
    called = {name for _, _, name in backend.calls}
    assert {"ReviewOut"} <= called
    assert res.report.llm.calls > 0
    assert res.report.summary_ko == "스크립트 검토 결과"
    # an empty layout plan is not a Gemma plan: the arrangement must be labelled heuristic
    assert res.layout.planner == "heuristic"
    assert {p.object_id for p in res.layout.placements} == {o.id for o in res.document.objects}
    # uninformative symbol answers never override the vector classification
    vec = json.loads((tmp_path / "out" / "extracted.json").read_text())
    assert [o["cls"] for o in vec["objects"]] == [o.cls.value for o in res.document.objects]


def test_cli_extract_and_symbols(ballast_pdf, tmp_path, capsys):
    pdf, _ = ballast_pdf
    out = tmp_path / "x.json"
    assert cli_main(["extract", str(pdf), "-o", str(out)]) == 0
    assert json.loads(out.read_text())["objects"]
    legend = tmp_path / "legend.pdf"
    assert cli_main(["symbols", "-o", str(legend)]) == 0
    assert legend.stat().st_size > 1000


def test_cli_llm_check_without_backend(capsys, monkeypatch):
    monkeypatch.delenv("SHIPPID_LLM_BACKEND", raising=False)
    assert cli_main(["llm-check", "--llm", "none"]) == 2
