# OCR 기반 멀티모달 시니어 복약지도 서버

인식 → 추론 → 전송 파이프라인으로, 처방전/약봉투 이미지 OCR, DUR 검증, 내부·외부 LLM 추론, MCP 연동, 기기 전송까지 구현한 서버입니다.

## 구조

- **인식:** 입력(음성/이미지) → OCR(DeepSeek) → DUR(KPIC) → DB/문서화
- **추론:** 내부 LLM + 개인정보 검열 → 외부 모델 강화 → 팩트 검증 → MCP 전달
- **전송:** MCP 데이터 통합 → 의사소통 AI 에이전트 → 서버 API → 기기 음성 출력
- **메모리:** 기존 `flash/permanent` MD 저장소 + Claude Code 스타일 `structured_memory`

## 구조화 메모리

서버는 `data/md_database/structured_memory` 아래에 전역 메모리와 화자별 메모리를 분리해 저장합니다.

- `global/MEMORY.md`: 팀 공통 인덱스
- `global/*.md`: 매뉴얼, 공통 레퍼런스
- `speakers/{speaker_id}/MEMORY.md`: 화자별 인덱스
- `speakers/{speaker_id}/*.md`: 환자 프로필, 최근 복약 맥락, DUR 관련 장기 메모

이 계층은 다음 흐름으로 동작합니다.

- frontmatter 기반 topic file 저장
- `MEMORY.md` 인덱스 자동 재생성
- 얕은 스캔으로 메모리 헤더만 빠르게 수집
- 질의와 실제로 매칭되는 메모리만 선별
- 오래된 메모리에는 freshness 경고 추가

## 실행

```bash
pip install -r requirements.txt
cp .env.example .env   # API 키 등 설정
python -m uvicorn app.main:app --reload --host 0.0.0.0
```

- API 문서: http://localhost:8000/docs
- Health: http://localhost:8000/health
- 파이프라인: POST /query/pipeline (이미지 업로드 → OCR → DUR → DB/문서화)
- 답변 생성: POST /query/ask (session_id, query_text 선택) → 내부 LLM → 검열 → 외부 LLM → 검증 → 시니어 친화 → MCP/기기 전송

## 테스트

```bash
pytest
```

`pytest.ini`에서 `tests/`만 수집하도록 고정해 두었기 때문에, 루트의 실험용 스크립트는 자동 수집되지 않습니다.

### 엔진 계약/데이터셋 게이트

`ai-server/.github/workflows/ci.yml`의 `test` job은 아래 계약 게이트를 추가로 강제한다.

```bash
python scripts/split_reasoning_dataset.py

python scripts/evaluate_engine_datasets.py \
  --dataset data/fine_tuning/qwen_router_samples.jsonl \
  --task-family router --strict

python scripts/evaluate_engine_datasets.py \
  --dataset data/fine_tuning/qwen_memory_samples.jsonl \
  --task-family memory --strict

python scripts/evaluate_engine_datasets.py \
  --dataset data/fine_tuning/qwen_delivery_samples.jsonl \
  --task-family delivery --strict
```

레거시 monolithic 셋(`qwen_reasoning_samples.jsonl`)은 아래 명령으로 런타임 불일치(`"<think>"`, tool-calling 혼합)를 리포트한다.

```bash
python scripts/evaluate_engine_datasets.py \
  --dataset data/fine_tuning/qwen_reasoning_samples.jsonl \
  --task-family reasoning
```

## 환경 변수

`.env.example` 참고. 필수: `DATABASE_URL`, OCR/DUR/LLM 관련 API URL·키(해당 단계 사용 시).
