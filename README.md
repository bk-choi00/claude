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
pytest                                                         # 전체 테스트 (2026-10-22 기준 602개, GPU·네트워크 불필요)
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

오류 처리: 입력 문제는 traceback 대신 한 줄(`shippid: error: ...`, stderr)로 알리고 종료 코드로 구분합니다 —
`0` 성공, `1` 명령은 실행됐지만 실패를 보고(`llm-check` 연결 실패, 합성 생성 실패 항목), `2` 인수·입력 오류(없는·빈·손상·암호화
PDF, 범위 밖 `--pages`, 잘못된 종류의 JSON — 예: `model3d`에 `layout3d.json`, LLM 백엔드 없이 `--planner gemma`·`--with-llm`),
`130` 중단(Ctrl-C). `-v`를 주면 traceback도 함께 출력합니다. `--planner gemma`는 LLM이 없으면 어떤 단계도 실행하기 전에
거부되므로 일부 산출물만 남는 일이 없습니다(`run`/`analyze`/`model3d` 동일).

### 여러 장(멀티시트) 도면

대형 선박 계통은 여러 장(SH.1 … SH.n)에 걸쳐 그려지고 페이지 간 연결 표시(off-page connector)로 이어집니다. 여러 페이지 PDF는
페이지마다 추출한 뒤 연결 표시를 짝지어 하나의 계통으로 잇습니다(`analysis/sheets.py`, `shippid run`/`analyze`/`evaluate` 공통).

* **짝짓기 근거** — 화살표 안의 연속 기호(`A`, `A1`, `B-2`), 시트 참조(`SH.2`, `SHEET 2`)와 각 페이지 표제란의 시트 번호
  (`2 OF 3`, `2/3`), 도면번호 참조(`CONT. ON DWG 4446-PD-BA-12 SH.2`), `TO <설비>` / `FROM <설비>` 명칭과 상대 시트에서 그
  연결 표시 옆 설비(`BALLAST PUMPS` ↔ `NO.1 BALLAST PUMP`), 방향(TO·CONT. ON = 나감, FROM·CONT. FROM = 들어옴), 양쪽 배관의 같은
  라인번호. *양쪽* 문구가 모두 상대를 지지하고 반박 근거가 없으며 경쟁 후보보다 점수 차가 충분할 때만 자동으로 연결합니다.
  문서에 없는 도면번호를 가리키는 연결 표시는 외부 연결이라 짝짓지 않습니다.
* **결과(`document.json`·`extracted.json`·`analysis.json`의 `meta`)** — `sheet_links`: 연결 쌍
  `{"a": 나가는 쪽, "b": 들어오는 쪽, "key": "mark:A1+sheet:1>2+name", "confidence", "source": "rule"|"gemma"|"manual", "directed", "evidence"}`;
  `sheet_links_ambiguous`: 후보가 둘 이상이거나 점수 차가 부족해 **미결**로 남은 연결 표시와 후보 목록(id·페이지·점수·근거) —
  Gemma를 쓰면 `resolve_sheet_links`가 이 후보 중에서만 고르고(일대일·상호 일관성 검증), 그 밖에는 사람이 검토할 목록입니다;
  `sheet_link_stats`: 평가한 쌍 수, 페이지 → 시트 번호 표, 추정한 시트 번호 등.
* **분석** — 연결 쌍마다 그래프에 `kind="sheet_link"` 간선(길이 0, 방향 a→b)이 추가되어 유동경로·격리밸브·인접관계·규칙 판정이
  시트를 넘어 이어집니다. 유동경로의 `pipe_ids`에는 실제 배관 id만 들어가고(예전 보고서의 `sheet_link:a:b` 가짜 배관 id는 더 이상
  쓰지 않음), 시트 간 이동은 `flow_paths.csv`의 `sheet_links` 열(`a>b`)과 HTML 보고서의 "도면 간 연결" 열에 나옵니다.
  `analysis.json`의 `report.meta["sheet_links"]`는 연결된 쌍의 **개수**, `["sheet_links_ambiguous"]`는 미결 연결 표시 개수,
  `["flow_path_sheet_links"]`는 시트를 넘는 유동경로와 그 이동 구간입니다.

### 조선소 고유 심볼 (범례 학습)

조선소마다 밸브·펌프·계기를 다르게 그리고 그 약속을 도면(또는 별도 범례 시트)의 LEGEND / SYMBOL LEGEND / 범례 블록에 적습니다.
추출기는 문서의 범례를 먼저 읽어 행마다 심볼 그림과 설명(`CHECK VALVE`, `S.D.N.R`, `NRV`, `체크 밸브` …)을 짝지어 **그 문서 전용
템플릿**을 만들고 내장 라이브러리와 함께 매칭합니다(`extract/legend.py`). 그렇게 분류된 객체에는
`attributes["template"] = "legend:<설명>"`이 남고, 클래스로 매핑되지 않은 범례 행은 `meta["legend"]["unmapped"]`에 기록되어 Gemma
판단에 쓰입니다. 범례 영역 자체는 객체로 추출하지 않습니다.

## 도면 수집 (대량 데이터)

실제 조선소 P&ID는 기밀이라 공개 대량 데이터가 없습니다(조사 결과: [docs/RESEARCH.md](docs/RESEARCH.md)).
그래서 세 가지 수집 경로를 제공합니다.

```bash
# (a) 선박 계통 합성 도면 대량 생성 — 리더블 벡터 PDF + 정답(JSON), train/val/test 분할
shippid collect synthetic --out data/synthetic --count 2000 --seed 2026 --workers 4

# (b) 사내 도면 아카이브 색인 — 리더블 여부(텍스트/벡터/이미지 비율), SHA-256 중복 제거, 계통 추정
shippid collect scan /nas/drawings --index data/archive/index.jsonl

# (c) URL 목록 다운로드 (재개/용량 제한/PDF 검사, 사설 주소 차단)
shippid collect fetch --sources sources.yaml --dest data/downloads            # 사내 서버: --allow-private
```

* `collect scan`은 스캔 루트 밖을 읽지 않습니다. 심볼릭 링크(파일·디렉터리)는 따라가지 않고 `skipped`로 기록하고, 일반 파일만
  엽니다(이름이 `*.pdf`인 FIFO·장치 파일 제외). 64단계보다 깊거나 목록을 읽을 수 없는 디렉터리는 `error`로 기록하고 나머지 색인은
  계속합니다(재귀 대신 반복 순회).
* `collect fetch`는 리다이렉트를 직접 따라가며(최대 5회, `http`/`https`만) 매 단계 호스트 주소를 확인해 루프백·사설·링크로컬
  (클라우드 메타데이터 169.254.169.254)·CGNAT·멀티캐스트 등 공인 주소가 아니면 요청하지 않고 `skipped`(`refused: ...`)로
  기록합니다 — 사내 서버에서 받을 때만 `--allow-private`. 다운로드마다 총 시간 제한(기본 15분)과 최소 평균 속도(30초 이후
  1 KB/s)가 있어 바이트를 조금씩 흘리는 서버가 연결을 붙잡지 못하며, 중단된 다운로드는 `.part`로 남아 다음 실행에서 이어받습니다.
  리다이렉트된 경우 최종 URL이 결과 기록과 출처 파일 `<file>.source.json`의 `final_url`에 남습니다.

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

`evaluate`는 `run`과 같은 설정(`--config shippid.yaml`의 추출 허용오차 등)으로 추출하고, 여러 장 도면은 시트 간 연결 표시를
짝지은 뒤 채점합니다(정답의 연결관계에는 시트 간 연결 쌍이 포함됩니다). 문서마다 한 번만 평가하고, 채점 중에는 예측·정답
문서를 한 쌍씩만 메모리에 올립니다. `evaluation.json`의 `benchmark.llm`에는 정제에 쓴 클라이언트가 `backend:model`로
기록되며(`none`, `openai:google/gemma-4-31B-it` …; 모의 백엔드는 `mock:...`로 남아 Gemma 결과로 표시되지 않음) `evaluation.md`
끝의 Benchmark 표에도 나옵니다. `--with-llm`은 LLM 백엔드가 설정돼 있어야 하고(아니면 오류), `--save-predictions`의 파일은
manifest id를 정리한 이름(`predictions/<id>.pred.json`)으로만 저장되어 출력 폴더 밖에 쓰이지 않습니다.

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

멀티시트 합성 세트(`datasets/multisheet.py`, 한 계통을 2–3장으로 나누고 3가지 조선소식 연결 표기, seed 100–109의 10세트,
2026-10-22 측정): 연결관계 F1 **1.000**(TP 662, FN 0). 같은 예측을 시트 연결 없이 채점하면 0.987(FN 17 — 놓친 17쌍이 정확히
정답의 시트 간 연결)이었습니다.

**주의:** 위 수치는 생성기와 추출기가 같은 심볼 라이브러리를 공유하는 *분포 내* 결과라 낙관적입니다.
실제 조선소 PDF의 다양성(페이지 회전, CAD식 경로, 레이어, 조선소 고유 심볼 등)에 대한 견고성은
[스타일 변형 평가](#견고성-평가-스타일-변형) 절을 보세요. Gemma 4 31B가 포함된 정확도는 GPU 서버에서
`--with-llm`으로 측정해야 하며, 이 저장소의 테스트는 Gemma를 모의(mock) 응답으로만 검증합니다.

## 견고성 평가 (스타일 변형)

같은 합성 도면을 실제 CAD 내보내기처럼 *다르게 써서* 다시 그린 분포 밖(OOD) 변형으로 측정합니다(`datasets/variants.py`).
변형마다 새 PDF와 그 PDF에 정확히 맞는 정답을 만들고, 변형 *i*번은 모두 같은 원본 도면에서 출발하므로 변형끼리 짝지은 비교가 됩니다.

```bash
python -c "from shippid.datasets.variants import build_variant_suite; build_variant_suite('data/variants', per_variant=20, seed=1000)"
shippid evaluate --data data/variants/fragmented_text --split test -o out/variants/fragmented_text --workers 3 --llm none   # 변형마다
```

변형당 20장(seed 1000, 계통 균등 배분), 결정론 경로(Gemma 미사용), micro 집계, 2026-10-22 최종 코드에서 측정(오류 0건):

| 변형 | 바뀌는 점 | 검출 F1 | 클래스 F1 | 연결 F1 | 텍스트 F1 | 태그 | 라인번호 |
|---|---|---:|---:|---:|---:|---:|---:|
| `plain` | 대조군: 변형 작성기로 그대로 다시 그림 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 |
| `rotated_page` | `/Rotate` 90·270 (세로 용지에 가로 도면을 출력한 CAD 플로터 형식) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 |
| `qu_bowties` | 나비넥타이 밸브 몸체를 자기교차 4점 폐곡선(PyMuPDF `qu`)으로 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 |
| `pipes_through` | 배관이 인라인 밸브·필터·유량계를 관통해 그려지고 심볼이 위에 얹힘 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 |
| `polyline_curves` | 모든 원·호(심볼, 접합점, 교차 홉)를 24–48개 직선으로 분할 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 |
| `styled_lines` | 선폭 0.25–2.0 pt, 심볼이 배관보다 굵음, 레이어별 색, 캡 모양, 이중 작도 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 |
| `fragmented_text` | 글자/단어 단위 텍스트, Courier·Times·Helvetica 혼용, 한글 라벨, 숨은 render-mode-3 중복(PDFSHX=2) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 |
| `layers_off` | OCG 레이어; NOTES·치수·구 개정·작도 레이어는 OFF(정답에 없음) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 |
| `scaled` | 도면 전체 0.6배·1.5배(A1 → A0), 심볼·글자 크기가 기본값과 크게 다름 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 |
| `yard_symbols` | 게이트·체크·SDNR·볼 밸브, 원심 펌프, 현장 계기를 조선소 고유 기호로 그리고 LEGEND로 설명 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.974 |
| `shx_text` | AutoCAD SHX 텍스트: 모든 글자가 단일 획 선분, 문자열은 투명 텍스트(PDFSHX=2) 또는 숨은 주석(PDFSHX=1) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 |
| `combined` | 위 변형 2–4개를 도면마다 무작위 조합 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.972 |

* 라인번호 정확도 0.971은 대조군 `plain`과 같으므로 변형 때문이 아니라 원본 도면 자체에서 생기는 오차입니다(원인은 이 측정에서
  따로 분석하지 않았습니다).
* 이전 측정의 기준선(텍스트 조각 병합·숨은 텍스트 처리, 범례 학습 등이 들어가기 전)은 `fragmented_text` 텍스트 F1 0.791(조각과
  숨은 중복이 별개 텍스트로 보고됨, 정밀도 0.657), `yard_symbols` 클래스 F1 0.841(조선소 체크밸브 → 게이트, 원심 펌프 → 현장 계기,
  볼·SDNR → 글로브), `polyline_curves` 0.995, `combined` 0.956이었습니다.
* **한계:** 변형 작성기 역시 이 저장소에서 만든 것이라, 실제 조선소 PDF에서만 나타나는 표현(다른 심볼 형상, 범례 없는 고유 기호,
  블록 참조·해칭 등)은 포함되지 않습니다. 실제 도면에서의 정확도는 사내 도면 일부에 정답을 만들어 `shippid evaluate`로 확인해야 합니다.

## 패키지 구조

| 경로 | 내용 |
|---|---|
| `src/shippid/model.py` | 공통 데이터 모델(PIDDocument, AnalysisReport, Layout3D …) |
| `src/shippid/symbols.py` | 심볼 라이브러리 33종(로컬 기하, 포트, 작동기) |
| `src/shippid/catalog.py` | 선박 계통 코드, ISO 14726 배관 색상, 태그/라인번호 규칙, 배관 규격 |
| `src/shippid/datasets/` | 합성 도면 생성기, PDF 작성기, 데이터셋 빌더, 멀티시트 세트(`multisheet.py`), 견고성 변형(`variants.py`), 아카이브 수집기 |
| `src/shippid/pdf/`, `src/shippid/extract/` | 리더블 PDF 판독, 심볼 매칭, 범례 학습(`legend.py`), 배관 토폴로지, 텍스트 연관 |
| `src/shippid/llm/` | Gemma 4 31B 백엔드(OpenAI 호환/Gemini), JSON 검증·재시도·캐시, 이미지 렌더링 |
| `src/shippid/analysis/` | 그래프·유동경로·규칙 판정·리스트, 시트 간 연결(`sheets.py`), Gemma 분석(`gemma.py`, `prompts.py`) |
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
* 시트 간 연결은 규칙 기반이라 근거가 부족한 연결 표시는 미결(`sheet_links_ambiguous`)로 남깁니다. Gemma 없이 실행하면 이 목록은
  사람이 검토해야 합니다.
* `collect fetch`의 주소 검사는 요청 직전의 DNS 해석 결과로 판단하므로, 그 사이 답이 바뀌는 DNS 리바인딩까지 막지는 못합니다.
  프록시 뒤에서 로컬 DNS로 외부 이름을 해석할 수 없으면 해당 출처는 `ConnectError`(cannot resolve host)로 기록됩니다.
