# 샘플 결과 (samples)

`shippid run <drawing.pdf> -o output --llm none` 로 만든 결과입니다(Gemma 미사용 결정론 경로 — Gemma 4 31B
서버가 있으면 `--llm vllm --base-url ...` 로 같은 명령을 실행하세요). 각 폴더:

| 파일 | 내용 |
|---|---|
| `drawing.pdf` | 합성 선박 P&ID (리더블 벡터 PDF) |
| `ground_truth.json` | 생성기 정답 (PIDDocument) |
| `output/report.html` | 분석 보고서 — 도면 위 추출 객체 오버레이, 설비/밸브/계기/라인 리스트, 규칙 판정, 요약 |
| `output/viewer.html` | 3D 뷰어 (three.js, 인터넷 필요 — CDN) — 클릭 시 객체 정보, 계통/범주 토글, 태그 검색 |
| `output/model.glb` | 3D 모델 (glTF binary; Blender·Windows 3D 뷰어 등에서 열림) |
| `output/analysis.json`, `output/tables/*.csv` | 분석 결과, Excel용 CSV |
| `output/layout3d.json` | 3D 배치·배관 경로 |
| `run_summary.json` | 실행 요약 |

1. `01_ballast_single_sheet` — 평형수 계통(BWTS, 유압 버터플라이 밸브), A1 1장
2. `02_fo_transfer_multisheet` — 연료유 이송 계통 2장, 페이지 간 연결 표시로 하나의 계통으로 연결
3. `03_sw_cooling_yard_symbols_legend` — 해수 냉각 계통, 조선소 고유 심볼 + LEGEND (범례 학습으로 분류)

세 샘플 모두 정답 대비 클래스 포함 검출 F1 1.000, 연결 F1 1.000 (합성 데이터 — 실제 도면 성능은 README의 한계 참고).
