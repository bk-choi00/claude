# shippid — 대형 선박 P&ID 객체 추출 · Gemma 4 31B 분석 · 3D 모델 생성

데이터가 살아 있는 **리더블 PDF**(CAD에서 내보낸 벡터 선분 + 실제 텍스트) P&ID 도면에서
설비·밸브·계기·배관·태그를 추출하고, **Gemma 4 31B**(`google/gemma-4-31B-it`)로 도면을
분석한 뒤, Gemma가 배치를 계획한 **3D 기관실 모델**(GLB + 웹 뷰어)을 생성하는 Python 3.10 패키지입니다.

```
readable PDF ──► ① 벡터 추출(결정론) ──► ② Gemma 4 31B 정제 ──► ③ 분석(규칙 + Gemma 검토) ──► 보고서(HTML/CSV/JSON)
                    PyMuPDF                심볼 재분류 / 표제란 /       그래프·유동경로·격리밸브
                    심볼 템플릿 매칭         태그 소유 판정               SOLAS·MARPOL·LL·선급 규칙
                    배관 토폴로지                                         Gemma 엔지니어링 리뷰(한/영 요약)
                                                         └──► ④ 3D: Gemma 배치 계획 → 솔버 → A* 배관 라우팅 → GLB + three.js/plotly 뷰어
```

설계 원칙 (근거: [docs/RESEARCH.md](docs/RESEARCH.md))
* **연결관계는 결정론적으로 추적**합니다. 공개 연구에서 VLM이 배관을 직접 읽으면 연결 F1이 ≈0.2,
  벡터/CV 추적은 ≈0.96–0.98이었습니다. Gemma는 구조화된 추출 결과(JSON 다이제스트 + 심볼 크롭 이미지)를
  받아 **의미 판단**(모호한 심볼 분류, 표제란 해석, 태그 소유, 규정 검토, 배치 계획)을 담당합니다.
* **Gemma 출력은 항상 검증**합니다. JSON 스키마(guided decoding) → 파싱/pydantic 검증 → 문서에 존재하는
  id·문자열인지 확인(grounding) → 실패 시 오류를 되먹여 재시도 → 그래도 실패하면 결정론 결과 유지.
  모든 결과에 출처(`source`: vector / gemma / merged, `planner`: gemma / heuristic)가 기록됩니다.
* **3D 좌표를 LLM이 직접 쓰지 않습니다.** Gemma는 구역·데크·현측·기준객체·관계로 된 배치 계획을 내고,
  결정론 솔버가 충돌/여유공간/범위 제약을 지켜 좌표로 바꾸며, 배관은 직교 A* 라우터가 경로를 만듭니다.
* `--llm none`이면 Gemma 없이 전 과정이 결정론 경로로 동작합니다(테스트·오프라인 용도).

## 설치 (Python 3.10)

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate     # 또는 python3.10 -m venv .venv
uv pip install -e ".[dev]"                                     # pip install -e ".[dev]"
pytest                                                         # 362개 테스트
```

의존성: PyMuPDF(<2.0, 2.x는 Python 3.11+), numpy, scipy, shapely, networkx, pydantic v2, httpx,
pyyaml, jsonschema, trimesh, plotly. **PyMuPDF는 AGPL-3.0**(또는 Artifex 상용 라이선스)입니다.

## Gemma 4 31B 서버 연결

shippid는 HTTP로 Gemma를 호출합니다(서버는 별도 GPU 머신/컨테이너, 도면이 외부로 나가지 않는 사내 배포 가능).

| 서버 | 실행 예 | shippid 옵션 |
|---|---|---|
| vLLM (권장) | `vllm serve google/gemma-4-31B-it --max-model-len 32768 --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice --limit-mm-per-prompt '{"image": 8, "video": 0}' --mm-processor-kwargs '{"max_soft_tokens": 1120}'` | `--llm vllm --base-url http://GPU:8000/v1` |
| SGLang | `sglang serve --model-path google/gemma-4-31B-it --tp 2 --reasoning-parser gemma4` | `--llm sglang --base-url http://GPU:30000/v1` |
| Ollama | `ollama run gemma4:31b` | `--llm ollama` (모델 `gemma4:31b`) |
| llama.cpp | `llama-server -hf ggml-org/gemma-4-31B-it-GGUF` | `--llm llamacpp --base-url http://GPU:8080/v1` |
| Gemini API | (Google 호스팅) | `--llm gemini` + `SHIPPID_LLM_API_KEY` (모델 `gemma-4-31b-it`) |

GPU 메모리: BF16 ≈ 75 GB(80 GB 1장), FP8 ≈ 38 GB(`RedHatAI/gemma-4-31B-it-FP8-dynamic`),
INT4 ≈ 20 GB(`google/gemma-4-31B-it-qat-w4a16-ct`). 현재 vLLM/SGLang 최신판은 Python 3.11+ 이므로
서버는 별도 환경(Docker `vllm/vllm-openai` 등)에서 실행하세요.

```bash
export SHIPPID_LLM_BACKEND=vllm SHIPPID_LLM_BASE_URL=http://gpu-host:8000/v1
shippid llm-check          # 텍스트+이미지+JSON 스키마 요청으로 연결/기능 점검
```

설정은 환경변수(`SHIPPID_LLM_*`), `shippid.yaml`, CLI 옵션 순으로 덮어씁니다.
`--llm-cache DIR`을 주면 요청/응답이 캐시되어 재실행이 재현 가능하고 비용이 들지 않습니다.

## 사용법

```bash
# 1) 도면 1장 전체 처리: 추출 → Gemma 정제 → 분석 → 3D → 보고서
shippid run drawing.pdf -o out/ --llm vllm --base-url http://gpu-host:8000/v1 --viewer both

# 단계별
shippid extract drawing.pdf -o drawing.json                 # 결정론 벡터 추출만
shippid analyze drawing.pdf -o out/                          # 추출 + Gemma 정제 + 분석 + 보고서
shippid model3d out/analysis.json -o out3d/ --planner gemma  # 3D만 (PDF/문서 JSON/분석 JSON 입력 가능)
shippid symbols -o legend.pdf                                # 내장 심볼 라이브러리 범례
```

출력(`out/`): `extracted.json`(벡터 추출) · `document.json`(Gemma 정제 후) · `analysis.json` ·
`tables/*.csv`(설비/밸브/계기/라인 리스트, 검토 결과, Excel 호환 UTF-8 BOM) · `report.html`(도면 위
객체 오버레이, 표, 규칙 판정, 한/영 요약, LLM 사용량) · `layout3d.json` · `model.glb` ·
`viewer.html`(three.js; 클릭 시 객체 정보, 계통/범주 토글, 태그 검색) · `viewer_plotly.html`(완전 오프라인).

> three.js 뷰어는 jsDelivr CDN에서 `three@0.180.0` 모듈을 불러옵니다(GLB는 HTML에 내장).
> 인터넷이 막힌 환경에서는 `--viewer plotly`(약 5–6 MB, 완전 오프라인)를 쓰세요.

## 도면 수집 (대량 데이터)

실제 조선소 P&ID는 기밀이라 공개 대량 데이터가 없습니다(조사 결과: [docs/RESEARCH.md](docs/RESEARCH.md)).
그래서 세 가지 수집 경로를 제공합니다.

```bash
# (a) 선박 계통 합성 도면 대량 생성 — 리더블 벡터 PDF + 정답(JSON), train/val/test 분할
shippid collect synthetic --out data/synthetic --count 2000 --seed 2026 --workers 4

# (b) 사내 도면 아카이브 색인 — 리더블 여부(텍스트/벡터/이미지 비율), SHA-256 중복 제거, 계통 추정
shippid collect scan /nas/drawings --index data/archive/index.jsonl

# (c) URL 목록 다운로드 (재개/용량 제한/PDF 검사)
shippid collect fetch --sources sources.yaml --dest data/downloads
```

합성 생성기는 10개 계통(평형수+BWTS, 빌지+OWS, 연료유 이송/공급, 윤활유, 해수 냉각, 중앙 청수 냉각,
소화, 압축공기, 증기/응축수)을 선박 관행대로 그립니다: 유압 버터플라이(평형수), 연료유 탱크 출구
급속차단밸브, 빌지 흡입 SDNR, 스크류 펌프 릴리프, 펌프 토출 체크밸브, 3방향 TCV, ISA 5.1 계기 태그,
`250A-BA-012-1S` 형식 라인번호(조선소별 3가지 스타일), A1–A3 도면틀·표제란·개정표·주석·범례·레이어.
4코어에서 초당 약 24장, 2,000장 생성 시 실패 0건(객체 75,879개).

## 성능 평가

```bash
shippid evaluate --data data/synthetic --split test -o out/bench --workers 3 --llm none
shippid evaluate --data data/synthetic --split test -o out/bench_gemma --with-llm --llm vllm ...
```

합성 test 분할 200장(결정론 경로, Gemma 미사용):

| 지표 | 값 |
|---|---:|
| 객체 검출 F1 (IoU ≥ 0.5) | 1.000 |
| 클래스 포함 검출 F1 | 1.000 |
| 연결관계(객체 쌍) F1 | 0.9999 |
| 태그 정확도 | 0.999 |
| 라인번호 정확도 | 0.977 |
| 텍스트 역할 정확도 | 1.000 |
| 처리 시간 | 약 0.5 s/장 |

**주의:** 위 수치는 생성기와 추출기가 같은 심볼 라이브러리를 공유하는 *분포 내* 결과라 낙관적입니다.
실제 조선소 PDF의 다양성(페이지 회전, CAD식 경로, 레이어, 조선소 고유 심볼 등)에 대한 견고성은
[스타일 변형 평가](#견고성-평가-스타일-변형) 절을 보세요. Gemma 4 31B가 포함된 정확도는 GPU 서버에서
`--with-llm`으로 측정해야 하며, 이 저장소의 테스트는 Gemma를 모의(mock) 응답으로만 검증합니다.

## 견고성 평가 (스타일 변형)

(작성 중 — 변형 생성기와 기준 성능 측정이 끝나면 갱신됩니다.)

## 패키지 구조

| 경로 | 내용 |
|---|---|
| `src/shippid/model.py` | 공통 데이터 모델(PIDDocument, AnalysisReport, Layout3D …) |
| `src/shippid/symbols.py` | 심볼 라이브러리 33종(로컬 기하, 포트, 작동기) |
| `src/shippid/catalog.py` | 선박 계통 코드, ISO 14726 배관 색상, 태그/라인번호 규칙, 배관 규격 |
| `src/shippid/datasets/` | 합성 도면 생성기, PDF 작성기, 데이터셋 빌더, 아카이브 수집기 |
| `src/shippid/pdf/`, `src/shippid/extract/` | 리더블 PDF 판독, 심볼 매칭, 배관 토폴로지, 텍스트 연관 |
| `src/shippid/llm/` | Gemma 4 31B 백엔드(OpenAI 호환/Gemini), JSON 검증·재시도·캐시, 이미지 렌더링 |
| `src/shippid/analysis/` | 그래프·유동경로·규칙 판정·리스트, Gemma 분석(`gemma.py`, `prompts.py`) |
| `src/shippid/model3d/` | 기관실 구역, Gemma/휴리스틱 배치 계획, 솔버, A* 라우터, 메시, GLB, 뷰어 |
| `src/shippid/report/` | HTML 보고서, CSV/JSON 내보내기 |
| `src/shippid/evaluate.py`, `benchmark.py` | 정답 대비 평가 지표, 데이터셋 벤치마크 |
| `docs/ARCHITECTURE.md`, `docs/RESEARCH.md` | 구현 계약, 조사 근거(출처 포함) |

## 한계

* Gemma 4 31B 실제 추론은 이 개발 환경(GPU 없음)에서 실행하지 못했습니다. HTTP 요청 형식(vLLM·Ollama·
  llama.cpp·Gemini)과 검증/재시도/대체 경로는 모의 서버로 테스트했지만, **실제 프롬프트 품질과 정확도는
  GPU 서버에서 `shippid evaluate --with-llm`으로 측정이 필요**합니다.
* P&ID에는 공간 정보가 없으므로 3D 모델은 실측 배치가 아닌 *그럴듯한 기관실 배치*입니다(기관실은 선종별
  경험치 크기의 직육면체, 0.5 m 격자 라우팅).
* 스캔(래스터) PDF는 범위 밖이며 `unreadable_pages`로 보고됩니다.
* 조선소별 태그 규칙은 공개 자료가 없어 정규식 기본값을 사용합니다(`catalog.py`에서 조정).
